//! Hosted upload permissions. The eventual entry-version UUID is also the
//! replay identity: no upload table, token ledger, or duplicate result store.

use axum::{Extension, Json, extract::State, http::StatusCode};
use chrono::{DateTime, Utc};
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::{Postgres, Row, Transaction};
use uuid::Uuid;

use crate::{
    auth::AuthContext,
    db::AppState,
    error::{ApiError, ApiResult},
    models::{Capability, CredentialId, UserId},
    simple_core::{WorkspaceEnvelope, portable_path_key, validate_public_path, validate_sha256},
};

pub const CONTENT_ROUTE: &str = "/v1/workspace/binaries/content";
pub const MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const AUDIENCE: &str = "brunn:binary-upload:v1";

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UploadRequest {
    pub path: String,
    pub media_type: String,
    pub size_bytes: u64,
    pub sha256: Option<String>,
    pub expected_version: Option<i64>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct UploadGrant {
    aud: String,
    exp: i64,
    user_id: Uuid,
    credential_id: Uuid,
    scope_refs: Vec<String>,
    pub version_id: Uuid,
    pub entry_id: Uuid,
    pub request: UploadRequest,
}

impl UploadGrant {
    pub fn auth(&self) -> AuthContext {
        AuthContext {
            user_id: UserId(self.user_id),
            credential_id: CredentialId(self.credential_id),
            scope_refs: self.scope_refs.clone(),
            capabilities: ["save".to_owned()].into(),
            read_only: false,
        }
    }
}

pub fn verify(secret: &str, token: &str) -> ApiResult<UploadGrant> {
    let mut validation = Validation::new(Algorithm::HS256);
    validation.set_audience(&[AUDIENCE]);
    validation.leeway = 0;
    decode::<UploadGrant>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &validation,
    )
    .map(|data| data.claims)
    .map_err(|error| match error.kind() {
        jsonwebtoken::errors::ErrorKind::ExpiredSignature => ApiError::public(
            StatusCode::GONE,
            "upload_expired",
            "mint a new upload permission",
        ),
        _ => ApiError::unauthenticated(),
    })
}

pub async fn mint(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Json(mut request): Json<UploadRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    auth.require(Capability::Save)?;
    validate_public_path(&request.path)?;
    if request.media_type.len() > 255 || request.media_type.parse::<mime::Mime>().is_err() {
        return Err(ApiError::invalid("media_type must be a valid MIME type"));
    }
    if request.size_bytes > MAX_BYTES {
        return Err(ApiError::public(
            StatusCode::PAYLOAD_TOO_LARGE,
            "binary_too_large",
            "workspace binary uploads are limited to 4 GiB",
        ));
    }
    request.sha256 = request.sha256.as_deref().map(validate_sha256).transpose()?;
    let expected = request.expected_version.unwrap_or(0);
    if expected < 0 {
        return Err(ApiError::invalid("expected_version must be nonnegative"));
    }
    request.expected_version = Some(expected);
    let mut tx = state.begin_write(&auth).await?;
    let existing = sqlx::query("SELECT id,kind,current_version,deleted_at FROM brunn.entries WHERE user_id=$1 AND lower(normalize(path, NFC))=$2")
        .bind(auth.user_id.0).bind(portable_path_key(&request.path))
        .fetch_optional(&mut *tx).await?;
    let entry_id = existing
        .as_ref()
        .map_or_else(Uuid::now_v7, |row| row.get("id"));
    check_target(&request.path, expected, entry_id, existing.as_ref())?;
    drop(tx);
    let grant = UploadGrant {
        aud: AUDIENCE.to_owned(),
        exp: Utc::now().timestamp() + 15 * 60,
        user_id: auth.user_id.0,
        credential_id: auth.credential_id.0,
        scope_refs: auth.scope_refs,
        version_id: Uuid::now_v7(),
        entry_id,
        request,
    };
    let token = encode(
        &Header::new(Algorithm::HS256),
        &grant,
        &EncodingKey::from_secret(state.config.continuation_secret.as_bytes()),
    )
    .map_err(|error| ApiError::Internal(format!("could not issue upload permission: {error}")))?;
    Ok(Json(WorkspaceEnvelope::complete(json!({
        "upload_id": grant.version_id,
        "put_url": format!("{}/api{CONTENT_ROUTE}", state.config.public_url.trim_end_matches('/')),
        "headers": {"Authorization": format!("BrunnUpload {token}"), "Content-Type": grant.request.media_type},
        "expires_at": DateTime::from_timestamp(grant.exp, 0),
        "max_bytes": MAX_BYTES
    }))))
}

pub(crate) fn check_target(
    path: &str,
    expected: i64,
    entry_id: Uuid,
    existing: Option<&sqlx::postgres::PgRow>,
) -> ApiResult<()> {
    let actual = existing.map_or(0, |row| row.get::<i64, _>("current_version"));
    let matches = match existing {
        Some(row) => {
            row.get::<Uuid, _>("id") == entry_id
                && row.get::<String, _>("kind") == "binary"
                && row.get::<Option<DateTime<Utc>>, _>("deleted_at").is_none()
                && expected > 0
                && expected == actual
        }
        None => expected == 0,
    };
    if !matches {
        return Err(ApiError::conflict(
            "entry_version_conflict",
            "the destination exists or changed; read it before replacing",
            json!({"path": path, "expected_version": expected, "actual_version": actual}),
        ));
    }
    Ok(())
}

/// Read the actual published version, not a second receipt. Called before
/// receiving bytes and again under the writer's path lock for racing retries.
pub(crate) async fn reject_completed(
    tx: &mut Transaction<'_, Postgres>,
    grant: &UploadGrant,
) -> ApiResult<()> {
    let previous = sqlx::query("SELECT entry_id,version,content_sha256,size_bytes FROM brunn.entry_versions WHERE user_id=$1 AND id=$2")
        .bind(grant.user_id).bind(grant.version_id).fetch_optional(&mut **tx).await?;
    if let Some(row) = previous {
        return Err(ApiError::conflict(
            "upload_completed",
            "this upload already published",
            json!({
                "entry_ref": format!("entry:{}", row.get::<Uuid, _>("entry_id")),
                "path": grant.request.path, "version": row.get::<i64, _>("version"),
                "content_hash": format!("sha256:{}", row.get::<String, _>("content_sha256")),
                "size_bytes": row.get::<i64, _>("size_bytes"), "media_type": grant.request.media_type
            }),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn grant() -> UploadGrant {
        UploadGrant {
            aud: AUDIENCE.into(),
            exp: Utc::now().timestamp() + 900,
            user_id: Uuid::now_v7(),
            credential_id: Uuid::now_v7(),
            scope_refs: vec!["scope:root".into()],
            version_id: Uuid::now_v7(),
            entry_id: Uuid::now_v7(),
            request: UploadRequest {
                path: "Inbox/test.jpg".into(),
                media_type: "image/jpeg".into(),
                size_bytes: 4,
                sha256: None,
                expected_version: Some(0),
            },
        }
    }
    #[test]
    fn grants_are_scoped_signed_expiring_and_not_api_bearers() {
        let secret = "test-only-upload-secret-at-least-32-bytes";
        let sign = |grant: &UploadGrant| {
            encode(
                &Header::new(Algorithm::HS256),
                grant,
                &EncodingKey::from_secret(secret.as_bytes()),
            )
            .unwrap()
        };
        let mut grant = grant();
        let token = sign(&grant);
        let verified = verify(secret, &token).unwrap();
        assert_eq!(verified.request.path, "Inbox/test.jpg");
        assert_eq!(verified.auth().user_id.0, grant.user_id);
        assert_eq!(verified.auth().capability_guc(), "save");
        assert!(verify("wrong-secret", &token).is_err());
        assert!(verify(secret, &format!("{token}x")).is_err());
        grant.aud = "brunn:other-operation".into();
        assert!(verify(secret, &sign(&grant)).is_err());
        grant.aud = AUDIENCE.into();
        grant.exp = Utc::now().timestamp() - 1;
        assert!(matches!(
            verify(secret, &sign(&grant)),
            Err(ApiError::Public {
                status: StatusCode::GONE,
                ..
            })
        ));
    }
}
