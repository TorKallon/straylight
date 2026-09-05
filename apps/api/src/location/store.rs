use std::collections::{BTreeMap, BTreeSet, HashMap};

use chrono::{DateTime, Duration, FixedOffset, Utc};
use chrono_tz::Tz;
use serde_json::json;
use sqlx::{Postgres, QueryBuilder, Row, Transaction};
use uuid::Uuid;

use crate::{
    AppState,
    auth::AuthContext,
    error::{ApiError, ApiResult},
    simple_core::{self, WriteRequest},
};

use super::{
    places::{PlacesWarning, parse_places},
    rules::{
        self, Confidence, Coordinate, FUTURE_CLOCK_TOLERANCE, HistoryRow, LocationReport,
        OpenVisit, PoiCandidate, PresenceState, ReportDisposition, ReportKind, TransitionKind,
    },
};

const PLACES_PATH: &str = "Location/Places.md";
const VISITS_PREFIX: &str = "Location/Visits/";
const RETENTION: Duration = Duration::days(30);

#[derive(Clone, Debug)]
pub(crate) struct ReportEvent {
    pub report_type: &'static str,
    pub disposition: ReportDisposition,
    pub transitions: Vec<TransitionKind>,
}

#[derive(Debug)]
pub(crate) struct IngestResult {
    pub accepted: usize,
    pub dispositions: Vec<ReportDisposition>,
    pub events: Vec<ReportEvent>,
    pub presence: Option<PresenceState>,
    pub places_warning: Option<PlacesWarning>,
}

#[derive(Debug)]
pub(crate) struct RederiveResult {
    pub reports_replayed: usize,
    pub rows_written: usize,
    pub presence_updated: bool,
    pub places_warning: Option<PlacesWarning>,
}

#[derive(Clone, Debug)]
struct WorkspaceDocument {
    content: Option<String>,
    version: i64,
}

struct FoldResult {
    presence: Option<PresenceState>,
    rows: Vec<HistoryRow>,
    /// Existing month-file rows superseded by an R4 merge.
    replaced: Vec<HistoryRow>,
    events: Vec<ReportEvent>,
}

pub(crate) async fn ingest_with_retry(
    state: &AppState,
    auth: &AuthContext,
    reports: &[LocationReport],
    pings_enabled: bool,
    as_of: DateTime<Utc>,
) -> ApiResult<IngestResult> {
    match ingest_once(state, auth, reports, pings_enabled, as_of).await {
        Err(error) if is_entry_version_conflict(&error) => {
            ingest_once(state, auth, reports, pings_enabled, as_of).await
        }
        result => result,
    }
}

pub(crate) async fn ingest_once(
    state: &AppState,
    auth: &AuthContext,
    reports: &[LocationReport],
    pings_enabled: bool,
    as_of: DateTime<Utc>,
) -> ApiResult<IngestResult> {
    let mut tx = state.begin_write(auth).await?;
    lock_location_user(&mut tx, auth.user_id.0).await?;
    let previous = read_presence_for_update(&mut tx, auth.user_id.0).await?;
    let eligible = reports
        .iter()
        .filter(|report| raw_report_is_eligible(report, pings_enabled, as_of))
        .collect::<Vec<_>>();
    insert_raw_reports(&mut tx, auth.user_id.0, &eligible).await?;

    let places_document = read_workspace_document(&mut tx, auth.user_id.0, PLACES_PATH).await?;
    let parsed_places = parse_places(places_document.content.as_deref());
    let mut month_documents =
        read_ingest_month_documents(&mut tx, state, auth.user_id.0, previous.as_ref(), &eligible)
            .await?;
    let existing_rows = month_documents
        .values()
        .filter_map(|document| document.content.as_deref())
        .flat_map(rules::parse_history_rows)
        .collect::<Vec<_>>();
    let folded = fold_reports(
        previous.clone(),
        reports,
        &parsed_places.places,
        &existing_rows,
        pings_enabled,
        as_of,
    );
    let mut rows_by_month = group_rows_by_month(folded.rows);
    let replaced_by_month = group_rows_by_month(folded.replaced);
    for month in replaced_by_month.keys() {
        rows_by_month.entry(month.clone()).or_default();
    }
    let mut workspace_changed = false;
    for (month, rows) in &mut rows_by_month {
        let document = match month_documents.remove(month) {
            Some(document) => document,
            None => {
                let path = month_path(month);
                lock_workspace_path(&mut tx, state, auth.user_id.0, &path).await?;
                read_workspace_document(&mut tx, auth.user_id.0, &path).await?
            }
        };
        let replaced = replaced_by_month
            .get(month)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let content = rules::insert_rows(document.content.as_deref(), rows, replaced);
        workspace_changed |=
            write_month_document(&mut tx, state, auth, month, content, document.version).await?;
    }
    if folded.presence != previous
        && let Some(presence) = folded.presence.as_ref()
    {
        upsert_presence(&mut tx, auth.user_id.0, presence).await?;
    }
    tx.commit().await?;
    if workspace_changed {
        state.workspace_features.invalidate(auth.user_id.0).await;
    }

    let dispositions = folded
        .events
        .iter()
        .map(|event| event.disposition)
        .collect::<Vec<_>>();
    let accepted = dispositions
        .iter()
        .filter(|disposition| **disposition == ReportDisposition::Accepted)
        .count();
    Ok(IngestResult {
        accepted,
        dispositions,
        events: folded.events,
        presence: folded.presence,
        places_warning: parsed_places.warning,
    })
}

pub(crate) async fn read_presence(
    state: &AppState,
    auth: &AuthContext,
) -> ApiResult<Option<PresenceState>> {
    let mut tx = state.begin_read(auth).await?;
    let row = sqlx::query(
        r#"
        SELECT timezone,reported_at,last_lat,last_lon,last_accuracy_m,
               city,region,country,visit_arrived_at,visit_lat,visit_lon,
               visit_label,visit_kind,visit_confidence
        FROM brunn.location_presence
        WHERE user_id=$1
        "#,
    )
    .bind(auth.user_id.0)
    .fetch_optional(&mut *tx)
    .await?;
    tx.commit().await?;
    row.as_ref().map(presence_from_row).transpose()
}

pub(crate) async fn rederive(
    state: &AppState,
    auth: &AuthContext,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    pings_enabled: bool,
    as_of: DateTime<Utc>,
) -> ApiResult<RederiveResult> {
    validate_rederive_window(from, to, as_of)?;
    let mut tx = state.begin_write(auth).await?;
    lock_location_user(&mut tx, auth.user_id.0).await?;
    let previous = read_presence_for_update(&mut tx, auth.user_id.0).await?;
    let timezone = previous
        .as_ref()
        .map_or(chrono_tz::UTC, |presence| presence.timezone);
    let reports = read_raw_reports(&mut tx, auth.user_id.0, from, to, timezone).await?;
    let newest_report = sqlx::query_scalar::<_, Option<DateTime<Utc>>>(
        "SELECT max(at) FROM brunn.location_reports WHERE user_id=$1",
    )
    .bind(auth.user_id.0)
    .fetch_one(&mut *tx)
    .await?;
    let places_document = read_workspace_document(&mut tx, auth.user_id.0, PLACES_PATH).await?;
    let parsed_places = parse_places(places_document.content.as_deref());

    let mut months = months_covering_window(from, to);
    months.extend(candidate_months(None, reports.iter()));
    let mut month_documents =
        read_month_documents(&mut tx, state, auth.user_id.0, months.into_iter()).await?;
    let existing_outside_window = month_documents
        .values()
        .filter_map(|document| document.content.as_deref())
        .flat_map(rules::parse_history_rows)
        .filter(|row| row.arrived_at < from || row.arrived_at > to)
        .collect::<Vec<_>>();
    let folded = fold_reports(
        None,
        &reports,
        &parsed_places.places,
        &existing_outside_window,
        pings_enabled,
        as_of,
    );
    let replacement_rows = folded
        .rows
        .into_iter()
        .filter(|row| row.arrived_at >= from && row.arrived_at <= to)
        .collect::<Vec<_>>();
    let replacements_by_month = group_rows_by_month(replacement_rows);
    let mut affected_months = month_documents
        .iter()
        .filter(|(_, document)| {
            document
                .content
                .as_deref()
                .map(rules::parse_history_rows)
                .unwrap_or_default()
                .iter()
                .any(|row| row.arrived_at >= from && row.arrived_at <= to)
        })
        .map(|(month, _)| month.clone())
        .collect::<BTreeSet<_>>();
    affected_months.extend(replacements_by_month.keys().cloned());
    for month in replacements_by_month.keys() {
        if !month_documents.contains_key(month) {
            let path = month_path(month);
            lock_workspace_path(&mut tx, state, auth.user_id.0, &path).await?;
            let document = read_workspace_document(&mut tx, auth.user_id.0, &path).await?;
            month_documents.insert(month.clone(), document);
        }
    }

    let mut workspace_changed = false;
    let mut rows_written = 0;
    for month in affected_months {
        let document = month_documents
            .remove(&month)
            .expect("affected month documents are loaded");
        let replacements = replacements_by_month
            .get(&month)
            .map(Vec::as_slice)
            .unwrap_or_default();
        if document.content.is_none() && replacements.is_empty() {
            continue;
        }
        let content =
            rules::replace_rows_in_window(document.content.as_deref(), from, to, replacements);
        let changed =
            write_month_document(&mut tx, state, auth, &month, content, document.version).await?;
        workspace_changed |= changed;
        if changed {
            rows_written += replacements.len();
        }
    }

    let reaches_newest = newest_report.is_none_or(|newest| to >= newest);
    let next_presence = folded.presence;
    let presence_updated = reaches_newest && next_presence != previous;
    if presence_updated {
        match next_presence.as_ref() {
            Some(presence) => upsert_presence(&mut tx, auth.user_id.0, presence).await?,
            None => {
                sqlx::query("DELETE FROM brunn.location_presence WHERE user_id=$1")
                    .bind(auth.user_id.0)
                    .execute(&mut *tx)
                    .await?;
            }
        }
    }
    tx.commit().await?;
    if workspace_changed {
        state.workspace_features.invalidate(auth.user_id.0).await;
    }

    Ok(RederiveResult {
        reports_replayed: reports.len(),
        rows_written,
        presence_updated,
        places_warning: parsed_places.warning,
    })
}

pub(crate) async fn delete_live(state: &AppState, auth: &AuthContext) -> ApiResult<u64> {
    let mut tx = state.begin_write(auth).await?;
    lock_location_user(&mut tx, auth.user_id.0).await?;
    let _ = read_presence_for_update(&mut tx, auth.user_id.0).await?;
    // Do not add a user_id predicate here. Reading that column would make
    // PostgreSQL also apply the deliberately narrower raw SELECT policy.
    // FORCE RLS and location_reports_delete scope this command to the caller.
    let reports_deleted = sqlx::query("DELETE FROM brunn.location_reports")
        .execute(&mut *tx)
        .await?
        .rows_affected();
    sqlx::query("DELETE FROM brunn.location_presence WHERE user_id=$1")
        .bind(auth.user_id.0)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(reports_deleted)
}

pub(crate) async fn delete_expired_reports(
    state: &AppState,
    as_of: DateTime<Utc>,
) -> ApiResult<u64> {
    let pool = state
        .admin_pool
        .as_ref()
        .ok_or_else(|| ApiError::configuration("location retention requires DATABASE_URL_ADMIN"))?;
    let result = sqlx::query("DELETE FROM brunn.location_reports WHERE at < $1")
        .bind(as_of - RETENTION)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

async fn lock_location_user(tx: &mut Transaction<'_, Postgres>, user_id: Uuid) -> ApiResult<()> {
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))")
        .bind(format!("location-presence:{user_id}"))
        .execute(&mut **tx)
        .await?;
    Ok(())
}

fn fold_reports(
    initial_presence: Option<PresenceState>,
    reports: &[LocationReport],
    places: &[super::places::KnownPlace],
    existing_rows: &[HistoryRow],
    pings_enabled: bool,
    as_of: DateTime<Utc>,
) -> FoldResult {
    let mut ordered = reports.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        left.at
            .cmp(&right.at)
            .then_with(|| left.kind.as_str().cmp(right.kind.as_str()))
    });
    let mut presence = initial_presence;
    let mut duplicate_scope = existing_rows.to_vec();
    let mut rows = Vec::new();
    let mut replaced = Vec::new();
    let mut events = Vec::with_capacity(ordered.len());
    for report in ordered {
        if report.at > as_of + FUTURE_CLOCK_TOLERANCE {
            events.push(ReportEvent {
                report_type: report.kind.as_str(),
                disposition: ReportDisposition::FutureClock,
                transitions: Vec::new(),
            });
            continue;
        }
        let outcome = rules::apply(
            presence.as_ref(),
            report,
            places,
            &duplicate_scope,
            pings_enabled,
        );
        rules::supersede_rows(
            &mut duplicate_scope,
            &mut rows,
            &mut replaced,
            outcome.replaced,
        );
        duplicate_scope.extend(outcome.rows.iter().cloned());
        rows.extend(outcome.rows);
        presence = outcome.presence;
        events.push(ReportEvent {
            report_type: report.kind.as_str(),
            disposition: outcome.disposition,
            transitions: outcome.transitions,
        });
    }
    FoldResult {
        presence,
        rows,
        replaced,
        events,
    }
}

async fn read_presence_for_update(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
) -> ApiResult<Option<PresenceState>> {
    let row = sqlx::query(
        r#"
        SELECT timezone,reported_at,last_lat,last_lon,last_accuracy_m,
               city,region,country,visit_arrived_at,visit_lat,visit_lon,
               visit_label,visit_kind,visit_confidence
        FROM brunn.location_presence
        WHERE user_id=$1
        FOR UPDATE
        "#,
    )
    .bind(user_id)
    .fetch_optional(&mut **tx)
    .await?;
    row.as_ref().map(presence_from_row).transpose()
}

fn presence_from_row(row: &sqlx::postgres::PgRow) -> ApiResult<PresenceState> {
    let timezone = row
        .try_get::<String, _>("timezone")?
        .parse::<Tz>()
        .map_err(|_| ApiError::Internal("stored location timezone is invalid".to_owned()))?;
    let visit_arrived_at = row.try_get::<Option<DateTime<Utc>>, _>("visit_arrived_at")?;
    let visit = match visit_arrived_at {
        Some(arrived_at) => Some(OpenVisit {
            arrived_at,
            coordinate: Coordinate {
                lat: required_location_value(row, "visit_lat")?,
                lon: required_location_value(row, "visit_lon")?,
            },
            label: row.try_get("visit_label")?,
            kind: required_location_value(row, "visit_kind")?,
            confidence: parse_confidence(&required_location_value::<String>(
                row,
                "visit_confidence",
            )?)?,
            opened_by_ping: false,
        }),
        None => None,
    };
    Ok(PresenceState {
        timezone,
        reported_at: row.try_get("reported_at")?,
        last_coordinate: Coordinate {
            lat: row.try_get("last_lat")?,
            lon: row.try_get("last_lon")?,
        },
        last_accuracy_m: f64::from(row.try_get::<f32, _>("last_accuracy_m")?),
        city: row.try_get("city")?,
        region: row.try_get("region")?,
        country: row.try_get("country")?,
        visit,
    })
}

fn required_location_value<T>(row: &sqlx::postgres::PgRow, column: &str) -> ApiResult<T>
where
    T: for<'r> sqlx::Decode<'r, Postgres> + sqlx::Type<Postgres>,
{
    row.try_get::<Option<T>, _>(column)?.ok_or_else(|| {
        ApiError::Internal("stored location presence is internally inconsistent".to_owned())
    })
}

fn parse_confidence(value: &str) -> ApiResult<Confidence> {
    match value {
        "high" => Ok(Confidence::High),
        "medium" => Ok(Confidence::Medium),
        "low" => Ok(Confidence::Low),
        _ => Err(ApiError::Internal(
            "stored location confidence is invalid".to_owned(),
        )),
    }
}

async fn insert_raw_reports(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    reports: &[&LocationReport],
) -> ApiResult<()> {
    let mut inserted = Vec::new();
    for report in reports {
        let (arrived_at, departed_at) = report_dates(&report.kind);
        let geocode = report.geocode.as_ref();
        let result = sqlx::query(
            r#"
            INSERT INTO brunn.location_reports (
              user_id,at,type,offset_min,lat,lon,accuracy_m,arrived_at,
              departed_at,city,region,country,name
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT DO NOTHING
            "#,
        )
        .bind(user_id)
        .bind(report.at)
        .bind(report.kind.as_str())
        .bind(report.offset_min)
        .bind(report.coordinate.lat)
        .bind(report.coordinate.lon)
        .bind(report.accuracy_m as f32)
        .bind(arrived_at)
        .bind(departed_at)
        .bind(geocode.and_then(|value| value.city.as_deref()))
        .bind(geocode.and_then(|value| value.region.as_deref()))
        .bind(geocode.and_then(|value| value.country.as_deref()))
        .bind(geocode.and_then(|value| value.name.as_deref()))
        .execute(&mut **tx)
        .await?;
        if result.rows_affected() == 1 {
            inserted.push(*report);
        }
    }

    let poi_rows = inserted
        .iter()
        .flat_map(|report| {
            report
                .poi
                .iter()
                .enumerate()
                .map(move |(index, poi)| (*report, index, poi))
        })
        .collect::<Vec<_>>();
    if !poi_rows.is_empty() {
        let mut statement = QueryBuilder::<Postgres>::new(
            "INSERT INTO brunn.location_report_poi \
             (user_id,at,type,rank,name,category,distance_m) ",
        );
        statement.push_values(poi_rows, |mut row, (report, index, poi)| {
            row.push_bind(user_id)
                .push_bind(report.at)
                .push_bind(report.kind.as_str())
                .push_bind(i16::try_from(index + 1).unwrap_or(i16::MAX))
                .push_bind(&poi.name)
                .push_bind(poi.category.as_deref())
                .push_bind(poi.distance_m as f32);
        });
        statement.push(" ON CONFLICT DO NOTHING");
        statement.build().execute(&mut **tx).await?;
    }
    Ok(())
}

fn raw_report_is_eligible(
    report: &LocationReport,
    pings_enabled: bool,
    as_of: DateTime<Utc>,
) -> bool {
    report.at <= as_of + FUTURE_CLOCK_TOLERANCE
        && (pings_enabled || !matches!(report.kind, ReportKind::Ping))
}

fn report_dates(kind: &ReportKind) -> (Option<DateTime<Utc>>, Option<DateTime<Utc>>) {
    match kind {
        ReportKind::Ping => (None, None),
        ReportKind::VisitArrival { arrived_at } => (*arrived_at, None),
        ReportKind::VisitDeparture {
            arrived_at,
            departed_at,
        } => (Some(*arrived_at), Some(*departed_at)),
    }
}

async fn read_raw_reports(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    timezone: Tz,
) -> ApiResult<Vec<LocationReport>> {
    let poi_rows = sqlx::query(
        r#"
        SELECT at,type,rank,name,category,distance_m
        FROM brunn.location_report_poi
        WHERE user_id=$1 AND at BETWEEN $2 AND $3
        ORDER BY at,type,rank
        "#,
    )
    .bind(user_id)
    .bind(from)
    .bind(to)
    .fetch_all(&mut **tx)
    .await?;
    let mut poi_by_report = HashMap::<(DateTime<Utc>, String), Vec<PoiCandidate>>::new();
    for row in poi_rows {
        poi_by_report
            .entry((row.try_get("at")?, row.try_get("type")?))
            .or_default()
            .push(PoiCandidate {
                name: row.try_get("name")?,
                category: row.try_get("category")?,
                distance_m: f64::from(row.try_get::<f32, _>("distance_m")?),
            });
    }
    let rows = sqlx::query(
        r#"
        SELECT at,type,offset_min,lat,lon,accuracy_m,arrived_at,departed_at,
               city,region,country,name
        FROM brunn.location_reports
        WHERE user_id=$1 AND at BETWEEN $2 AND $3
        ORDER BY at,type
        "#,
    )
    .bind(user_id)
    .bind(from)
    .bind(to)
    .fetch_all(&mut **tx)
    .await?;
    rows.into_iter()
        .map(|row| {
            let at = row.try_get::<DateTime<Utc>, _>("at")?;
            let report_type = row.try_get::<String, _>("type")?;
            let kind = report_kind_from_row(&row, &report_type)?;
            Ok(LocationReport {
                kind,
                at,
                offset_min: row.try_get("offset_min")?,
                timezone,
                coordinate: Coordinate {
                    lat: row.try_get("lat")?,
                    lon: row.try_get("lon")?,
                },
                accuracy_m: f64::from(row.try_get::<f32, _>("accuracy_m")?),
                geocode: Some(rules::Geocode {
                    city: row.try_get("city")?,
                    region: row.try_get("region")?,
                    country: row.try_get("country")?,
                    name: row.try_get("name")?,
                })
                .filter(|geocode| {
                    geocode.city.is_some()
                        || geocode.region.is_some()
                        || geocode.country.is_some()
                        || geocode.name.is_some()
                }),
                poi: poi_by_report.remove(&(at, report_type)).unwrap_or_default(),
            })
        })
        .collect()
}

fn report_kind_from_row(row: &sqlx::postgres::PgRow, report_type: &str) -> ApiResult<ReportKind> {
    match report_type {
        "ping" => Ok(ReportKind::Ping),
        "visit_arrival" => Ok(ReportKind::VisitArrival {
            arrived_at: row.try_get("arrived_at")?,
        }),
        "visit_departure" => Ok(ReportKind::VisitDeparture {
            arrived_at: required_location_value(row, "arrived_at")?,
            departed_at: required_location_value(row, "departed_at")?,
        }),
        _ => Err(ApiError::Internal(
            "stored location report type is invalid".to_owned(),
        )),
    }
}

async fn upsert_presence(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    presence: &PresenceState,
) -> ApiResult<()> {
    let visit = presence.visit.as_ref();
    sqlx::query(
        r#"
        INSERT INTO brunn.location_presence (
          user_id,timezone,reported_at,last_lat,last_lon,last_accuracy_m,
          city,region,country,visit_arrived_at,visit_lat,visit_lon,
          visit_label,visit_kind,visit_confidence
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
        ON CONFLICT (user_id) DO UPDATE SET
          timezone=EXCLUDED.timezone,
          reported_at=EXCLUDED.reported_at,
          last_lat=EXCLUDED.last_lat,
          last_lon=EXCLUDED.last_lon,
          last_accuracy_m=EXCLUDED.last_accuracy_m,
          city=EXCLUDED.city,
          region=EXCLUDED.region,
          country=EXCLUDED.country,
          visit_arrived_at=EXCLUDED.visit_arrived_at,
          visit_lat=EXCLUDED.visit_lat,
          visit_lon=EXCLUDED.visit_lon,
          visit_label=EXCLUDED.visit_label,
          visit_kind=EXCLUDED.visit_kind,
          visit_confidence=EXCLUDED.visit_confidence
        "#,
    )
    .bind(user_id)
    .bind(presence.timezone.to_string())
    .bind(presence.reported_at)
    .bind(presence.last_coordinate.lat)
    .bind(presence.last_coordinate.lon)
    .bind(presence.last_accuracy_m as f32)
    .bind(presence.city.as_deref())
    .bind(presence.region.as_deref())
    .bind(presence.country.as_deref())
    .bind(visit.map(|value| value.arrived_at))
    .bind(visit.map(|value| value.coordinate.lat))
    .bind(visit.map(|value| value.coordinate.lon))
    .bind(visit.and_then(|value| value.label.as_deref()))
    .bind(visit.map(|value| value.kind.as_str()))
    .bind(visit.map(|value| confidence_as_str(value.confidence)))
    .execute(&mut **tx)
    .await?;
    Ok(())
}

fn confidence_as_str(confidence: Confidence) -> &'static str {
    match confidence {
        Confidence::High => "high",
        Confidence::Medium => "medium",
        Confidence::Low => "low",
    }
}

async fn read_workspace_document(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    path: &str,
) -> ApiResult<WorkspaceDocument> {
    let row = sqlx::query(
        r#"
        SELECT entry.current_version,version.content
        FROM brunn.entries AS entry
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
         AND version.version=entry.current_version
        WHERE entry.user_id=$1
          AND lower(normalize(entry.path, NFC))=$2
          AND entry.kind='markdown' AND entry.deleted_at IS NULL
        "#,
    )
    .bind(user_id)
    .bind(simple_core::portable_path_key(path))
    .fetch_optional(&mut **tx)
    .await?;
    match row {
        Some(row) => Ok(WorkspaceDocument {
            content: row.try_get("content")?,
            version: row.try_get("current_version")?,
        }),
        None => Ok(WorkspaceDocument {
            content: None,
            version: 0,
        }),
    }
}

async fn read_ingest_month_documents(
    tx: &mut Transaction<'_, Postgres>,
    state: &AppState,
    user_id: Uuid,
    previous: Option<&PresenceState>,
    reports: &[&LocationReport],
) -> ApiResult<BTreeMap<String, WorkspaceDocument>> {
    read_month_documents(
        tx,
        state,
        user_id,
        candidate_months(previous, reports.iter().copied()).into_iter(),
    )
    .await
}

async fn read_month_documents(
    tx: &mut Transaction<'_, Postgres>,
    state: &AppState,
    user_id: Uuid,
    months: impl Iterator<Item = String>,
) -> ApiResult<BTreeMap<String, WorkspaceDocument>> {
    let mut documents = BTreeMap::new();
    for month in months {
        let path = month_path(&month);
        lock_workspace_path(tx, state, user_id, &path).await?;
        let document = read_workspace_document(tx, user_id, &path).await?;
        documents.insert(month, document);
    }
    Ok(documents)
}

async fn lock_workspace_path(
    tx: &mut Transaction<'_, Postgres>,
    state: &AppState,
    user_id: Uuid,
    path: &str,
) -> ApiResult<()> {
    simple_core::require_local_publish_lock(
        tx,
        format!(
            "simple-entry:{user_id}:{}",
            simple_core::portable_path_key(path)
        ),
        state.config.read_path_roundtrip_v1,
    )
    .await
}

async fn write_month_document(
    tx: &mut Transaction<'_, Postgres>,
    state: &AppState,
    auth: &AuthContext,
    month: &str,
    content: String,
    expected_version: i64,
) -> ApiResult<bool> {
    let prepared = simple_core::prepare_markdown(
        state,
        WriteRequest {
            path: month_path(month),
            content,
            media_type: "text/markdown".to_owned(),
            expected_version: Some(expected_version),
            idempotency_key: None,
            metadata: json!({}),
        },
    )
    .await?;
    let result = simple_core::upsert_markdown_in_tx(
        tx,
        auth.user_id.0,
        Some(auth.credential_id.0),
        prepared,
    )
    .await?;
    Ok(!result.no_op)
}

fn candidate_months<'a>(
    previous: Option<&PresenceState>,
    reports: impl Iterator<Item = &'a LocationReport>,
) -> BTreeSet<String> {
    let reports = reports.collect::<Vec<_>>();
    let offsets = reports
        .iter()
        .map(|report| report.offset_min)
        .collect::<BTreeSet<_>>();
    let mut anchors = reports
        .iter()
        .flat_map(|report| {
            let (arrived_at, _) = report_dates(&report.kind);
            [Some(report.at), arrived_at].into_iter().flatten()
        })
        .collect::<Vec<_>>();
    if let Some(arrived_at) = previous
        .and_then(|presence| presence.visit.as_ref())
        .map(|visit| visit.arrived_at)
    {
        anchors.push(arrived_at);
    }
    let mut months = BTreeSet::new();
    for anchor in anchors {
        for offset in &offsets {
            months.insert(month_at_offset(anchor, *offset));
        }
    }
    months
}

fn months_covering_window(from: DateTime<Utc>, to: DateTime<Utc>) -> BTreeSet<String> {
    let mut months = BTreeSet::new();
    let mut day = (from - Duration::days(1)).date_naive();
    let end = (to + Duration::days(1)).date_naive();
    while day <= end {
        months.insert(day.format("%Y-%m").to_string());
        let Some(next) = day.succ_opt() else {
            break;
        };
        day = next;
    }
    months
}

fn month_at_offset(value: DateTime<Utc>, offset_min: i16) -> String {
    let offset = FixedOffset::east_opt(i32::from(offset_min) * 60)
        .unwrap_or_else(|| FixedOffset::east_opt(0).expect("UTC offset exists"));
    value.with_timezone(&offset).format("%Y-%m").to_string()
}

fn group_rows_by_month(rows: Vec<HistoryRow>) -> BTreeMap<String, Vec<HistoryRow>> {
    let mut grouped = BTreeMap::<String, Vec<HistoryRow>>::new();
    for row in rows {
        grouped.entry(rules::month_key(&row)).or_default().push(row);
    }
    grouped
}

fn month_path(month: &str) -> String {
    format!("{VISITS_PREFIX}{month}.md")
}

fn validate_rederive_window(
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    as_of: DateTime<Utc>,
) -> ApiResult<()> {
    if from > to || to > as_of || from < as_of - RETENTION || to - from > RETENTION {
        return Err(ApiError::invalid(
            "rederive window must be ordered and fall within the last 30 days",
        ));
    }
    Ok(())
}

fn is_entry_version_conflict(error: &ApiError) -> bool {
    matches!(
        error,
        ApiError::Public {
            code: "entry_version_conflict",
            ..
        }
    )
}

#[cfg(test)]
mod database_tests {
    use std::collections::HashSet;

    use axum::http::StatusCode;
    use sha2::{Digest, Sha256};
    use sqlx::{AssertSqlSafe, PgPool, postgres::PgPoolOptions};
    use url::Url;

    use crate::{
        Config,
        models::{CredentialId, UserId},
    };

    use super::*;

    struct TestTrigger {
        table: &'static str,
        trigger_name: String,
        function_name: String,
        sequence_name: Option<String>,
    }

    async fn execute_test_ddl(pool: &PgPool, statement: String, context: &str) {
        sqlx::query(AssertSqlSafe(statement.as_str()))
            .execute(pool)
            .await
            .unwrap_or_else(|error| panic!("{context}: {error}"));
    }

    fn database_url_as_role(database_url: &str, role: &str) -> String {
        let mut url = Url::parse(database_url).expect("parse disposable PostgreSQL URL");
        url.query_pairs_mut()
            .append_pair("options", &format!("-c role={role}"));
        url.into()
    }

    async fn connect_test_state(test_name: &str) -> Option<(PgPool, AppState)> {
        let Some(database_url) = std::env::var("BRUNN_TEST_DATABASE_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("BRUNN_TEST_DATABASE_URL is unset; skipping {test_name}");
            return None;
        };
        let seed_pool = PgPoolOptions::new()
            .max_connections(4)
            .connect(&database_url)
            .await
            .expect("connect to disposable PostgreSQL");
        sqlx::migrate!("./migrations")
            .run(&seed_pool)
            .await
            .expect("apply Brunn migrations");

        let mut config = Config::from_env().expect("load disposable API configuration");
        let rw_database_url = database_url_as_role(&database_url, "app_rw");
        config.database_url_rw = rw_database_url.clone();
        config.database_url_ro = rw_database_url;
        config.database_url_admin = Some(database_url);
        config.database_max_connections = 8;
        config.apns_delivery_enabled = false;
        config.messaging_enabled = false;
        config.semantic_lane = false;
        config.supersession_demotion = false;
        config.intention_ledger = false;
        config.read_path_roundtrip_v1 = true;
        config.observability_timings_ms = false;
        let state = AppState::connect(config)
            .await
            .expect("connect disposable API state");
        Some((seed_pool, state))
    }

    async fn insert_location_principal(pool: &PgPool, label: &str) -> (Uuid, Uuid, AuthContext) {
        let user_id = Uuid::now_v7();
        let credential_id = Uuid::now_v7();
        sqlx::query("INSERT INTO brunn.users(id,external_ref,display_name) VALUES($1,$2,$3)")
            .bind(user_id)
            .bind(format!("location-store-{label}:{user_id}"))
            .bind(format!("Location store {label}"))
            .execute(pool)
            .await
            .expect("insert location store user");
        let scope_id = sqlx::query_scalar::<_, Uuid>(
            "SELECT id FROM brunn.scopes WHERE user_id=$1 AND scope_ref='scope:root'",
        )
        .bind(user_id)
        .fetch_one(pool)
        .await
        .expect("load seeded root scope");
        let capabilities = ["read".to_owned(), "location.write".to_owned()];
        sqlx::query(
            r#"
            INSERT INTO brunn.api_credentials(id,user_id,label,token_hash,capabilities)
            VALUES($1,$2,$3,$4,$5)
            "#,
        )
        .bind(credential_id)
        .bind(user_id)
        .bind(format!("Location store {label}"))
        .bind(format!("location-store-token-{credential_id}"))
        .bind(&capabilities)
        .execute(pool)
        .await
        .expect("insert location store credential");
        sqlx::query(
            r#"
            INSERT INTO brunn.credential_scope_grants(credential_id,user_id,scope_id)
            VALUES($1,$2,$3)
            "#,
        )
        .bind(credential_id)
        .bind(user_id)
        .bind(scope_id)
        .execute(pool)
        .await
        .expect("grant location store root scope");
        let auth = AuthContext {
            credential_id: CredentialId(credential_id),
            user_id: UserId(user_id),
            capabilities: capabilities.into_iter().collect::<HashSet<_>>(),
            scope_refs: vec!["scope:root".to_owned()],
            read_only: false,
        };
        (user_id, credential_id, auth)
    }

    fn completed_report(at: DateTime<Utc>) -> LocationReport {
        LocationReport {
            kind: ReportKind::VisitDeparture {
                arrived_at: at - Duration::hours(2),
                departed_at: at - Duration::hours(1),
            },
            at,
            offset_min: 0,
            timezone: chrono_tz::UTC,
            coordinate: Coordinate {
                lat: 47.6205,
                lon: -122.3493,
            },
            accuracy_m: 12.0,
            geocode: Some(rules::Geocode {
                city: Some("Seattle".to_owned()),
                region: Some("Washington".to_owned()),
                country: Some("United States".to_owned()),
                name: Some("Test Place".to_owned()),
            }),
            poi: vec![PoiCandidate {
                name: "Test POI".to_owned(),
                category: Some("test".to_owned()),
                distance_m: 4.0,
            }],
        }
    }

    async fn seed_presence(
        pool: &PgPool,
        user_id: Uuid,
        reported_at: DateTime<Utc>,
    ) -> DateTime<Utc> {
        sqlx::query_scalar(
            r#"
            INSERT INTO brunn.location_presence(
              user_id,timezone,reported_at,last_lat,last_lon,last_accuracy_m,
              city,region,country
            ) VALUES($1,'UTC',$2,47.0,-122.0,20.0,'Seattle','Washington','United States')
            RETURNING reported_at
            "#,
        )
        .bind(user_id)
        .bind(reported_at)
        .fetch_one(pool)
        .await
        .expect("seed prior location presence")
    }

    async fn seed_raw_ping_with_poi(pool: &PgPool, user_id: Uuid, at: DateTime<Utc>) {
        sqlx::query(
            r#"
            INSERT INTO brunn.location_reports(
              user_id,at,type,offset_min,lat,lon,accuracy_m
            ) VALUES($1,$2,'ping',0,47.6205,-122.3493,12.0)
            "#,
        )
        .bind(user_id)
        .bind(at)
        .execute(pool)
        .await
        .expect("seed raw location ping");
        sqlx::query(
            r#"
            INSERT INTO brunn.location_report_poi(
              user_id,at,type,rank,name,category,distance_m
            ) VALUES($1,$2,'ping',1,'Live-delete POI','test',4.0)
            "#,
        )
        .bind(user_id)
        .bind(at)
        .execute(pool)
        .await
        .expect("seed raw location ping POI");
    }

    async fn seed_deleted_month(
        pool: &PgPool,
        user_id: Uuid,
        credential_id: Uuid,
        path: &str,
    ) -> Uuid {
        let entry_id = Uuid::now_v7();
        let version_id = Uuid::now_v7();
        let content = "# Deliberately different deleted month fixture\n";
        let content_sha256 = hex::encode(Sha256::digest(content.as_bytes()));
        let mut tx = pool.begin().await.expect("begin deleted month fixture");
        sqlx::query(
            r#"
            INSERT INTO brunn.entries(
              id,user_id,path,title,kind,media_type,current_version
            ) VALUES($1,$2,$3,'Deleted location month','markdown','text/markdown',0)
            "#,
        )
        .bind(entry_id)
        .bind(user_id)
        .bind(path)
        .execute(&mut *tx)
        .await
        .expect("insert deleted month entry shell");
        sqlx::query(
            r#"
            INSERT INTO brunn.entry_versions(
              id,user_id,entry_id,version,content_sha256,content,size_bytes,
              metadata,created_by_credential_id
            ) VALUES($1,$2,$3,1,$4,$5,$6,'{}'::jsonb,$7)
            "#,
        )
        .bind(version_id)
        .bind(user_id)
        .bind(entry_id)
        .bind(content_sha256)
        .bind(content)
        .bind(i64::try_from(content.len()).expect("fixture content fits i64"))
        .bind(credential_id)
        .execute(&mut *tx)
        .await
        .expect("insert deleted month version");
        sqlx::query(
            r#"
            UPDATE brunn.entries
            SET current_version=1,deleted_at=clock_timestamp()
            WHERE user_id=$1 AND id=$2
            "#,
        )
        .bind(user_id)
        .bind(entry_id)
        .execute(&mut *tx)
        .await
        .expect("mark month entry deleted");
        tx.commit().await.expect("commit deleted month fixture");
        entry_id
    }

    async fn install_month_write_failure(pool: &PgPool, user_id: Uuid) -> TestTrigger {
        let suffix = Uuid::now_v7().simple().to_string();
        let function_name = format!("test_location_month_failure_fn_{suffix}");
        let trigger_name = format!("test_location_month_failure_{suffix}");
        execute_test_ddl(
            pool,
            format!(
                r#"
            CREATE FUNCTION brunn.{function_name}() RETURNS trigger
            LANGUAGE plpgsql AS $body$
            BEGIN
              RAISE EXCEPTION 'forced location month write failure'
                USING ERRCODE = 'P0001';
            END
            $body$
            "#,
            ),
            "create forced month write failure function",
        )
        .await;
        execute_test_ddl(
            pool,
            format!(
                r#"
            CREATE TRIGGER {trigger_name}
            BEFORE INSERT ON brunn.entry_versions
            FOR EACH ROW WHEN (NEW.user_id = '{user_id}'::uuid)
            EXECUTE FUNCTION brunn.{function_name}()
            "#,
            ),
            "create forced month write failure trigger",
        )
        .await;
        TestTrigger {
            table: "entry_versions",
            trigger_name,
            function_name,
            sequence_name: None,
        }
    }

    async fn install_repeatable_cas_conflict(
        pool: &PgPool,
        user_id: Uuid,
        path: &str,
    ) -> TestTrigger {
        let suffix = Uuid::now_v7().simple().to_string();
        let function_name = format!("test_location_cas_conflict_fn_{suffix}");
        let trigger_name = format!("test_location_cas_conflict_{suffix}");
        let sequence_name = format!("test_location_cas_attempts_{suffix}");
        execute_test_ddl(
            pool,
            format!("CREATE SEQUENCE brunn.{sequence_name}"),
            "create CAS attempt sequence",
        )
        .await;
        execute_test_ddl(
            pool,
            format!("GRANT USAGE, SELECT, UPDATE ON SEQUENCE brunn.{sequence_name} TO app_rw"),
            "grant CAS attempt sequence to application role",
        )
        .await;
        execute_test_ddl(
            pool,
            format!(
                r#"
            CREATE FUNCTION brunn.{function_name}() RETURNS trigger
            LANGUAGE plpgsql AS $body$
            BEGIN
              PERFORM nextval('brunn.{sequence_name}'::regclass);
              UPDATE brunn.entries
              SET deleted_at=NULL
              WHERE user_id=NEW.user_id
                AND lower(normalize(path, NFC))=lower(normalize(NEW.path, NFC));
              RETURN NEW;
            END
            $body$
            "#,
            ),
            "create repeatable CAS conflict function",
        )
        .await;
        execute_test_ddl(
            pool,
            format!(
                r#"
            CREATE TRIGGER {trigger_name}
            BEFORE INSERT ON brunn.entries
            FOR EACH ROW WHEN (
              NEW.user_id = '{user_id}'::uuid AND NEW.path = '{path}'
            )
            EXECUTE FUNCTION brunn.{function_name}()
            "#,
            ),
            "create repeatable CAS conflict trigger",
        )
        .await;
        TestTrigger {
            table: "entries",
            trigger_name,
            function_name,
            sequence_name: Some(sequence_name),
        }
    }

    async fn cas_attempt_count(pool: &PgPool, trigger: &TestTrigger) -> (i64, bool) {
        let sequence_name = trigger
            .sequence_name
            .as_deref()
            .expect("CAS trigger has an attempt sequence");
        let statement = format!("SELECT last_value,is_called FROM brunn.{sequence_name}");
        sqlx::query_as::<_, (i64, bool)>(AssertSqlSafe(statement.as_str()))
            .fetch_one(pool)
            .await
            .expect("read CAS attempt count")
    }

    async fn drop_test_trigger(pool: &PgPool, trigger: TestTrigger) {
        execute_test_ddl(
            pool,
            format!(
                "DROP TRIGGER IF EXISTS {} ON brunn.{}",
                trigger.trigger_name, trigger.table
            ),
            "drop location test trigger",
        )
        .await;
        execute_test_ddl(
            pool,
            format!("DROP FUNCTION IF EXISTS brunn.{}()", trigger.function_name),
            "drop location test trigger function",
        )
        .await;
        if let Some(sequence_name) = trigger.sequence_name {
            execute_test_ddl(
                pool,
                format!("DROP SEQUENCE IF EXISTS brunn.{sequence_name}"),
                "drop location test sequence",
            )
            .await;
        }
    }

    async fn user_row_count(pool: &PgPool, table: &str, user_id: Uuid) -> i64 {
        assert!(matches!(
            table,
            "location_reports"
                | "location_report_poi"
                | "location_presence"
                | "entries"
                | "entry_versions"
                | "workspace_changes"
        ));
        let statement = match table {
            "location_reports" => "SELECT count(*) FROM brunn.location_reports WHERE user_id=$1",
            "location_report_poi" => {
                "SELECT count(*) FROM brunn.location_report_poi WHERE user_id=$1"
            }
            "location_presence" => "SELECT count(*) FROM brunn.location_presence WHERE user_id=$1",
            "entries" => "SELECT count(*) FROM brunn.entries WHERE user_id=$1",
            "entry_versions" => "SELECT count(*) FROM brunn.entry_versions WHERE user_id=$1",
            "workspace_changes" => "SELECT count(*) FROM brunn.workspace_changes WHERE user_id=$1",
            _ => unreachable!("table allowlist checked above"),
        };
        sqlx::query_scalar::<_, i64>(statement)
            .bind(user_id)
            .fetch_one(pool)
            .await
            .expect("count user-owned test rows")
    }

    #[tokio::test]
    async fn month_write_failure_rolls_back_raw_presence_and_workspace_file() {
        let Some((pool, state)) =
            connect_test_state("location month-write rollback database gate").await
        else {
            return;
        };
        let (user_id, _, auth) = insert_location_principal(&pool, "month-rollback").await;
        let as_of = "2026-09-05T18:00:00.123456789Z"
            .parse::<DateTime<Utc>>()
            .expect("valid nanosecond fixture timestamp");
        let original_presence_at = seed_presence(&pool, user_id, as_of - Duration::hours(3)).await;
        let report = completed_report(as_of - Duration::minutes(1));
        let trigger = install_month_write_failure(&pool, user_id).await;

        let result = ingest_once(&state, &auth, &[report], true, as_of).await;
        drop_test_trigger(&pool, trigger).await;

        assert!(
            matches!(result, Err(ApiError::Database(_))),
            "forced month write must surface its database failure"
        );
        assert_eq!(user_row_count(&pool, "location_reports", user_id).await, 0);
        assert_eq!(
            user_row_count(&pool, "location_report_poi", user_id).await,
            0
        );
        assert_eq!(user_row_count(&pool, "entries", user_id).await, 0);
        assert_eq!(user_row_count(&pool, "entry_versions", user_id).await, 0);
        assert_eq!(user_row_count(&pool, "workspace_changes", user_id).await, 0);
        let stored_presence_at = sqlx::query_scalar::<_, DateTime<Utc>>(
            "SELECT reported_at FROM brunn.location_presence WHERE user_id=$1",
        )
        .bind(user_id)
        .fetch_one(&pool)
        .await
        .expect("read presence after rolled-back month write");
        assert_eq!(stored_presence_at, original_presence_at);
    }

    #[tokio::test]
    async fn ingest_retries_one_whole_transaction_then_preserves_second_cas_conflict() {
        let Some((pool, state)) = connect_test_state("location CAS retry database gate").await
        else {
            return;
        };
        let (user_id, credential_id, auth) = insert_location_principal(&pool, "cas-retry").await;
        let as_of = Utc::now();
        let report = completed_report(as_of - Duration::minutes(1));
        let arrived_at = match &report.kind {
            ReportKind::VisitDeparture { arrived_at, .. } => *arrived_at,
            _ => unreachable!("completed report is a visit departure"),
        };
        let path = month_path(&month_at_offset(arrived_at, report.offset_min));
        let entry_id = seed_deleted_month(&pool, user_id, credential_id, &path).await;
        let trigger = install_repeatable_cas_conflict(&pool, user_id, &path).await;

        let result = ingest_with_retry(&state, &auth, &[report], true, as_of).await;
        let attempts = cas_attempt_count(&pool, &trigger).await;
        drop_test_trigger(&pool, trigger).await;

        assert_eq!(attempts, (2, true), "ingest must attempt exactly twice");
        assert!(matches!(
            result,
            Err(ApiError::Public {
                status: StatusCode::CONFLICT,
                code: "entry_version_conflict",
                ..
            })
        ));
        assert_eq!(user_row_count(&pool, "location_reports", user_id).await, 0);
        assert_eq!(
            user_row_count(&pool, "location_report_poi", user_id).await,
            0
        );
        assert_eq!(user_row_count(&pool, "location_presence", user_id).await, 0);
        let stored = sqlx::query_as::<_, (i64, bool)>(
            "SELECT current_version,deleted_at IS NOT NULL FROM brunn.entries WHERE user_id=$1 AND id=$2",
        )
        .bind(user_id)
        .bind(entry_id)
        .fetch_one(&pool)
        .await
        .expect("read deleted month after exhausted retry");
        assert_eq!(stored, (1, true));
        assert_eq!(user_row_count(&pool, "entry_versions", user_id).await, 1);
        assert_eq!(user_row_count(&pool, "workspace_changes", user_id).await, 0);
    }

    #[tokio::test]
    async fn retention_uses_explicit_as_of_cascades_poi_and_preserves_newer_reports() {
        let Some((pool, state)) = connect_test_state("location retention database gate").await
        else {
            return;
        };
        let (user_id, _, _) = insert_location_principal(&pool, "retention").await;
        let as_of = Utc::now();
        let old_at = as_of - RETENTION - Duration::seconds(1);
        let new_at = as_of - RETENTION + Duration::seconds(1);
        for at in [old_at, new_at] {
            sqlx::query(
                r#"
                INSERT INTO brunn.location_reports(
                  user_id,at,type,offset_min,lat,lon,accuracy_m
                ) VALUES($1,$2,'ping',0,47.6205,-122.3493,12.0)
                "#,
            )
            .bind(user_id)
            .bind(at)
            .execute(&pool)
            .await
            .expect("seed retained location report");
            sqlx::query(
                r#"
                INSERT INTO brunn.location_report_poi(
                  user_id,at,type,rank,name,category,distance_m
                ) VALUES($1,$2,'ping',1,'Retention POI','test',4.0)
                "#,
            )
            .bind(user_id)
            .bind(at)
            .execute(&pool)
            .await
            .expect("seed retained location POI");
        }

        let deleted = delete_expired_reports(&state, as_of)
            .await
            .expect("delete expired location reports");

        assert!(deleted >= 1, "retention must delete the seeded old report");
        for (at, expected) in [(old_at, 0_i64), (new_at, 1_i64)] {
            let report_count = sqlx::query_scalar::<_, i64>(
                "SELECT count(*) FROM brunn.location_reports WHERE user_id=$1 AND at=$2",
            )
            .bind(user_id)
            .bind(at)
            .fetch_one(&pool)
            .await
            .expect("count retained location report");
            let poi_count = sqlx::query_scalar::<_, i64>(
                "SELECT count(*) FROM brunn.location_report_poi WHERE user_id=$1 AND at=$2",
            )
            .bind(user_id)
            .bind(at)
            .fetch_one(&pool)
            .await
            .expect("count retained location POI");
            assert_eq!(report_count, expected);
            assert_eq!(poi_count, expected);
        }
    }

    #[tokio::test]
    async fn delete_live_reports_rows_affected_and_rls_preserves_other_users() {
        let Some((pool, state)) = connect_test_state("location live-delete database gate").await
        else {
            return;
        };
        let (user_id, _, auth) = insert_location_principal(&pool, "live-delete").await;
        let (other_user_id, _, _) = insert_location_principal(&pool, "live-delete-neighbor").await;
        let at = Utc::now() - Duration::minutes(1);
        for owner in [user_id, other_user_id] {
            seed_raw_ping_with_poi(&pool, owner, at).await;
            seed_presence(&pool, owner, at).await;
        }

        let reports_deleted = delete_live(&state, &auth)
            .await
            .expect("delete caller live location state");

        assert_eq!(reports_deleted, 1);
        for table in [
            "location_reports",
            "location_report_poi",
            "location_presence",
        ] {
            assert_eq!(user_row_count(&pool, table, user_id).await, 0);
            assert_eq!(user_row_count(&pool, table, other_user_id).await, 1);
        }
    }
}
