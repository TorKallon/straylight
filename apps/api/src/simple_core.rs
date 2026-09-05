use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::{
    Extension, Json,
    body::Body,
    extract::{Multipart, Path, Query, State, multipart::Field},
    http::{HeaderValue, Response, StatusCode, header},
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use hmac::{Hmac, Mac};
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sqlx::{Postgres, QueryBuilder, Row, Transaction, postgres::PgRow};
use tokio::{fs::OpenOptions, io::AsyncWriteExt};
use tokio_util::io::ReaderStream;
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

use crate::{
    auth::{AuthContext, hash_token},
    db::{AppState, set_context},
    embeddings::SharedEmbedder,
    error::{ApiError, ApiResult},
    foreground_latency::ForegroundOperation,
    ingest::{DocumentChunk, normalize_document},
    location::{
        rules::{PresenceState as LocationPresenceState, presence_view},
        store as location_store,
    },
    models::{Capability, CheckpointRequest, CredentialId, ResponseStatus, UserId, canonical_json},
    retrieval_sql::{
        SIMPLE_ENTRY_LINK_CANDIDATES_SQL, SIMPLE_LEXICAL_CANDIDATES_SQL,
        SIMPLE_LEXICAL_CANDIDATES_WITH_GENERATION_SQL, SIMPLE_SEMANTIC_CANDIDATES_SQL,
    },
    semantic_policy::{PreparedQueryEmbedding, SemanticRuntime},
    usage::{ProductActivityOperation, UsageOperation},
    workspace_features::{
        DerivedFrontmatter, SupersessionAnnotation, WorkspaceFeatureDocument,
        WorkspaceFeatureSnapshot, parse_frontmatter, supersession_warnings,
    },
};

type HmacSha256 = Hmac<Sha256>;

#[derive(Clone, Debug, Serialize)]
pub struct WorkspaceEnvelope<T> {
    #[serde(skip_serializing)]
    pub request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corpus_revision: Option<String>,
    pub status: ResponseStatus,
    pub data: T,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub gaps: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings_ms: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_count: Option<usize>,
}

impl<T> WorkspaceEnvelope<T> {
    pub(crate) fn complete(data: T) -> Self {
        Self {
            request_id: crate::request_context::current_request_id(),
            session_id: None,
            corpus_revision: None,
            status: ResponseStatus::Complete,
            data,
            gaps: Vec::new(),
            timings_ms: None,
            query_count: crate::request_query_count::current(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct RetrievalTimings {
    exact: f64,
    lexical: f64,
    semantic_ready: f64,
    semantic: f64,
    embed: f64,
    semantic_db: f64,
    merge: f64,
    total: f64,
}

impl RetrievalTimings {
    fn as_value(&self) -> Value {
        json!({
            "exact": round_ms(self.exact),
            "lexical": round_ms(self.lexical),
            "semantic_ready": round_ms(self.semantic_ready),
            "semantic": round_ms(self.semantic),
            "embed": round_ms(self.embed),
            "semantic_db": round_ms(self.semantic_db),
            "merge": round_ms(self.merge),
            "total": round_ms(self.total),
        })
    }
}

#[derive(Clone, Debug)]
struct SemanticCandidates {
    candidates: Vec<Candidate>,
    embed_ms: f64,
    database_ms: f64,
}

#[derive(Clone)]
struct RequestSemanticEmbeddings {
    inner: Arc<Mutex<RequestSemanticEmbeddingState>>,
}

struct RequestSemanticEmbeddingState {
    queries: Vec<Option<String>>,
    tickets: Option<Vec<Option<PreparedQueryEmbedding>>>,
}

impl RequestSemanticEmbeddings {
    fn new(queries: Vec<Option<String>>) -> Option<Self> {
        queries.iter().any(Option::is_some).then(|| Self {
            inner: Arc::new(Mutex::new(RequestSemanticEmbeddingState {
                queries,
                tickets: None,
            })),
        })
    }

    /// The first semantic lane that passes readiness prepares every semantic
    /// query in the request. Provider work is detached and batched while each
    /// ticket remains governed by its own lane deadline.
    fn take(
        &self,
        runtime: &SemanticRuntime,
        embedder: SharedEmbedder,
        provider_timeout: Duration,
        index: usize,
    ) -> Option<PreparedQueryEmbedding> {
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        if inner.tickets.is_none() {
            let inputs = inner
                .queries
                .iter()
                .filter_map(Clone::clone)
                .collect::<Vec<_>>();
            let mut prepared = runtime
                .prepare_cached_query_embeddings(embedder, &inputs, provider_timeout)
                .into_iter();
            let tickets = inner
                .queries
                .iter()
                .map(|query| {
                    query.as_ref().map(|_| {
                        prepared
                            .next()
                            .expect("each semantic query must have one prepared ticket")
                    })
                })
                .collect();
            debug_assert!(prepared.next().is_none());
            inner.tickets = Some(tickets);
        }
        inner
            .tickets
            .as_mut()
            .and_then(|tickets| tickets.get_mut(index))
            .and_then(Option::take)
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn round_ms(value: f64) -> f64 {
    (value * 1_000.0).round() / 1_000.0
}

const DEFAULT_TOKEN_BUDGET: usize = 12_000;
const MAX_TOKEN_BUDGET: usize = 64_000;
pub(crate) const MAX_WRITE_BYTES: usize = 4 * 1024 * 1024;
const MAX_EXACT_READ_CHARS: usize = 4 * 1024 * 1024;
const MAX_READ_RESPONSE_CHARS: usize = MAX_EXACT_READ_CHARS;
const MAX_SEARCH_LIMIT: usize = 50;
const MAX_SEARCH_RESPONSE_CANDIDATES: usize = 128;
const MAX_SEARCH_RESPONSE_CHARS: usize = 96_000;
const MAX_VERBATIM_MATCHES_PER_CANDIDATE: usize = 3;
const MAX_VERBATIM_LINE_CHARS: usize = 2_400;
const MAX_VERBATIM_RESPONSE_CHARS: usize = 9_600;
const OPEN_CANDIDATE_LIMIT: usize = 32;
const HYDRATED_DOCUMENT_LIMIT: usize = 8;
const MAX_OPEN_COMPLETE_SOURCE_CHARS: usize = 24_000;
const RESUME_DELTA_SOURCE_LIMIT: usize = 8;
const RESUME_DELTA_TOTAL_CHARS: usize = 6_000;
const RESUME_DELTA_SOURCE_CHARS: usize = 2_000;
const RESUME_DELTA_WHOLE_PAIR_CHARS: usize = 2_400;
const MAX_STREAMED_BINARY_BYTES: u64 = crate::binary_upload::MAX_BYTES;
const WORKSPACE_IMPORT_FORMAT: &str = "brunn-workspace-import-manifest@v1";
const TIER_A_PORTABLE_COMPANION_FORMAT: &str = "brunn-tier-a-portable-companion@v1";
const TIER_A_HISTORY_STAGE_FORMAT: &str = "brunn-tier-a-history-stage@v1";
const TIER_A_ORDINARY_HISTORY_SEMANTICS: &str = "ordinary_content_transition";
const TIER_A_EXACT_HISTORY_SEMANTICS: &str = "preserve_intentional_exact_bytes_version";
const RETRIEVAL_LANE_TIMEOUT: Duration = Duration::from_millis(2_500);
// Each search item already fans out into lexical and semantic database lanes.
// Bound request-level fan-out so a four-item batch cannot start eight reads at once.
const SEARCH_QUERY_CONCURRENCY: usize = 2;
const CHUNK_INSERT_BATCH_SIZE: usize = 256;

#[derive(Clone, Debug, Default, Deserialize)]
pub struct OpenHints {
    pub authorization_scope: Option<String>,
    #[serde(default)]
    pub root_refs: Vec<String>,
    #[serde(default)]
    pub open_object_refs: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenRequest {
    pub task: String,
    #[serde(default)]
    pub hints: OpenHints,
    pub resume_checkpoint_ref: Option<String>,
    pub token_budget: Option<usize>,
    #[serde(default)]
    pub modes: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SearchQuery {
    pub id: Option<String>,
    pub query: String,
    pub goal: Option<String>,
    pub limit: Option<usize>,
    pub sort: Option<String>,
    #[serde(default)]
    pub modes: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SearchRequest {
    pub session_id: Option<String>,
    #[serde(default)]
    pub queries: Vec<SearchQuery>,
    pub query: Option<String>,
    pub limit: Option<usize>,
    pub token_budget: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReadItem {
    #[serde(rename = "ref")]
    pub reference: Option<String>,
    pub path: Option<String>,
    pub link_target: Option<String>,
    pub view: Option<String>,
    pub start: Option<usize>,
    pub end: Option<usize>,
    pub max_chars: Option<usize>,
    pub version: Option<i64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReadRequest {
    pub session_id: Option<String>,
    pub requests: Vec<ReadItem>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WriteRequest {
    pub path: String,
    pub content: String,
    #[serde(default = "markdown_media_type")]
    pub media_type: String,
    pub expected_version: Option<i64>,
    pub idempotency_key: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CaptureRequest {
    pub content: String,
    #[serde(default)]
    pub source: Value,
    pub intent: Option<String>,
    pub idempotency_key: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChangesQuery {
    #[serde(default)]
    pub since_generation: i64,
    pub limit: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BinaryListQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ManifestQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub after_path: Option<String>,
    pub after_entry_ref: Option<String>,
    pub after_version: Option<i64>,
    #[serde(default)]
    pub history: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct UsageQuery {
    pub sort: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BinaryVersionQuery {
    pub version: Option<i64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct StreamingBinaryQuery {
    #[serde(default)]
    pub path: String,
    pub media_type: Option<String>,
    #[serde(default)]
    pub expected_content_hash: String,
    pub mtime_ns: Option<i64>,
    pub mode: Option<u32>,
    pub provenance: Option<String>,
    pub expected_version: Option<i64>,
}

#[derive(Clone, Debug)]
struct PortableBinaryCompanion {
    path: String,
    content: String,
    content_sha256: String,
    modified_unix_ns: Option<i64>,
    mode: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DeleteQuery {
    pub expected_version: Option<i64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DreamRequest {
    pub since_generation: Option<i64>,
    pub focus: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct JobsQuery {
    pub status: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
struct CandidateSection {
    heading: String,
    excerpt: String,
    #[serde(skip)]
    score: f64,
}

#[derive(Clone, Debug, Serialize)]
struct VerbatimMatch {
    line_no: usize,
    byte_start: usize,
    byte_end: usize,
    text: String,
    version: i64,
    content_hash: String,
    #[serde(skip_serializing_if = "is_false")]
    truncated: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Candidate {
    entry_id: Uuid,
    path: String,
    title: String,
    version: i64,
    updated_at: DateTime<Utc>,
    content_sha256: String,
    heading: String,
    excerpt: String,
    score: f64,
    lanes: Vec<String>,
    sections: Vec<CandidateSection>,
    verbatim_matches: Vec<VerbatimMatch>,
    superseded_by: Option<SupersessionAnnotation>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchRepresentation {
    CompleteSource,
    Excerpt,
    PointerLead,
}

#[derive(Clone, Debug)]
struct SearchCandidateView {
    candidate: Candidate,
    representation: SearchRepresentation,
    complete_text: Option<String>,
    demote_additional_sections: bool,
}

#[derive(Clone, Debug)]
struct SearchHydration {
    size_bytes: usize,
    content: String,
}

#[derive(Clone, Copy, Debug)]
struct SearchBudgetOptions {
    fair_share: bool,
    top1_hydration: bool,
    char_cap: bool,
    section_demotion_top_n: Option<usize>,
    max_chars: usize,
}

impl SearchBudgetOptions {
    fn from_request(config: &crate::config::Config, token_budget: Option<usize>) -> Self {
        let char_cap = config.search_char_cap;
        let max_chars = if char_cap {
            token_budget
                .unwrap_or(DEFAULT_TOKEN_BUDGET)
                .clamp(1_000, MAX_TOKEN_BUDGET)
                .saturating_mul(4)
                .min(MAX_SEARCH_RESPONSE_CHARS)
        } else {
            MAX_SEARCH_RESPONSE_CHARS
        };
        Self {
            fair_share: config.search_fair_share,
            top1_hydration: config.search_top1_hydration,
            char_cap,
            section_demotion_top_n: char_cap
                .then_some(config.search_section_demotion_top_n)
                .flatten(),
            max_chars,
        }
    }

    fn active(self) -> bool {
        self.fair_share || self.top1_hydration || self.char_cap
    }
}

#[derive(Clone, Debug)]
struct EntryRow {
    id: Uuid,
    path: String,
    title: String,
    kind: String,
    media_type: String,
    version: i64,
    content_sha256: String,
    content: Option<String>,
    object_key: Option<String>,
    object_version_id: Option<String>,
    size_bytes: i64,
    metadata: Value,
    updated_at: DateTime<Utc>,
    workspace_generation: Option<i64>,
}

struct ChangePage {
    changes: Vec<Value>,
    truncated: bool,
    next_generation: Option<i64>,
    workspace_generation: Option<i64>,
}

#[derive(Clone, Debug)]
struct CheckpointSource {
    entry_id: Uuid,
    path: String,
    pinned_version: i64,
    pinned_sha256: String,
}

#[derive(Clone, Debug)]
struct ResumeVersionPair {
    source: CheckpointSource,
    current_version: i64,
    current_sha256: String,
    before: Option<String>,
    after: Option<String>,
}

#[derive(Debug, Default)]
struct ResumeDeltaBatch {
    deltas: Vec<Value>,
    leads: Vec<Value>,
    charged_chars: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedMarkdown {
    entry_id_hint: Option<Uuid>,
    path: String,
    title: String,
    content: String,
    pub(crate) content_sha256: String,
    media_type: String,
    pub(crate) metadata: Value,
    chunks: Vec<DocumentChunk>,
    embeddings: Vec<Option<Vector>>,
    expected_version: Option<i64>,
    tier_a_history_stage: Option<TierAHistoryStage>,
    frontmatter: DerivedFrontmatter,
    /// Write a new version even when the content hash is unchanged. Set by
    /// callers whose authoritative payload lives in version metadata (for
    /// example briefing editions), where the equal-content NoOp rule would
    /// otherwise silently drop a metadata-only revision.
    pub(crate) force_new_version: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TierAHistorySemantics {
    OrdinaryContentTransition,
    PreserveIntentionalExactBytesVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TierAHistoryStage {
    target_lineage_ordinal: i64,
    semantics: TierAHistorySemantics,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TierAExactHistoryAction {
    NotRequested,
    Append,
    Idempotent,
}

#[derive(Clone, Debug)]
pub(crate) struct MarkdownUpsertResult {
    pub(crate) entry_id: Uuid,
    pub(crate) version: i64,
    pub(crate) version_id: Option<Uuid>,
    pub(crate) generation: Option<i64>,
    pub(crate) no_op: bool,
    metadata_only: bool,
}

#[derive(Clone, Debug)]
struct CheckpointWriteResult {
    receipt: Value,
    committed_bytes: u64,
    created: bool,
}

#[derive(Clone, Debug)]
struct BulkMarkdown {
    entry_id: Uuid,
    version_id: Uuid,
    path: String,
    title: String,
    content: String,
    content_sha256: String,
    media_type: String,
    metadata: Value,
    chunks: Vec<DocumentChunk>,
    embeddings: Vec<Option<Vector>>,
    frontmatter: DerivedFrontmatter,
}

fn markdown_media_type() -> String {
    "text/markdown".to_owned()
}

// Evaluation import DTOs, shared by the batched simple import surface.
#[derive(Clone, Deserialize, Serialize)]
pub struct EvalImportRequest {
    pub schema: String,
    pub run_id: String,
    pub case_id: String,
    pub authorization_scope: String,
    pub display_scope: String,
    pub access_mode: String,
    pub documents: Vec<EvalDocument>,
    #[serde(default)]
    pub delta_documents: Vec<EvalDocument>,
    pub seed_checkpoint: Option<EvalSeedCheckpoint>,
    pub idempotency_key: String,
    #[serde(default)]
    pub batch_index: Option<usize>,
    #[serde(default)]
    pub batch_count: Option<usize>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct EvalDocument {
    pub path: String,
    pub content: String,
    pub content_sha256: String,
    pub media_type: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct EvalSeedCheckpoint {
    pub state: Value,
    #[serde(default)]
    pub source_refs: Vec<String>,
}

pub async fn open(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Json(request): Json<OpenRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    let total_started = Instant::now();
    auth.require(Capability::Open)?;
    metrics::counter!("simple.open.requests").increment(1);
    if request.task.trim().is_empty() {
        return Err(ApiError::invalid("task is required"));
    }
    let budget = request
        .token_budget
        .unwrap_or(DEFAULT_TOKEN_BUDGET)
        .clamp(1_000, MAX_TOKEN_BUDGET);
    let session_id = format!("session:{}", Uuid::now_v7());
    let generation_started = Instant::now();
    let eager_generation = if !state.config.read_path_roundtrip_v1
        || state.config.supersession_demotion
        || state.config.intention_ledger
    {
        Some(current_generation(&state, &auth).await?)
    } else {
        None
    };
    let generation_ms = elapsed_ms(generation_started);
    let features_started = Instant::now();
    let features = match eager_generation {
        Some(generation) => feature_snapshot(&state, &auth, generation).await?,
        None => None,
    };
    let features_ms = elapsed_ms(features_started);
    let checkpoint_and_changes = async {
        let checkpoint_started = Instant::now();
        let checkpoint = match request.resume_checkpoint_ref.as_deref() {
            Some(reference) => Some(read_checkpoint(&state, &auth, reference).await?),
            None => None,
        };
        let checkpoint_read_ms = elapsed_ms(checkpoint_started);
        let checkpoint_generation = checkpoint
            .as_ref()
            .and_then(|value| value.get("workspace_generation"))
            .and_then(Value::as_i64);
        let changes_started = Instant::now();
        let changes = match checkpoint_generation {
            Some(since) => changes_since(&state, &auth, since, 200).await?,
            None => ChangePage {
                changes: vec![],
                truncated: false,
                next_generation: None,
                workspace_generation: None,
            },
        };
        let changes_ms = elapsed_ms(changes_started);
        Ok::<_, ApiError>((checkpoint, changes, checkpoint_read_ms, changes_ms))
    };
    let retrieve = async {
        let retrieval_started = Instant::now();
        let (search, hinted) = tokio::join!(
            search_one(
                &state,
                &auth,
                &request.task,
                OPEN_CANDIDATE_LIMIT,
                &request.modes,
                None,
                features.as_deref(),
                None,
            ),
            open_hint_candidates(&state, &auth, &request.hints)
        );
        (search, hinted, elapsed_ms(retrieval_started))
    };
    let presence_now = Utc::now();
    let retrieve_and_presence = async {
        tokio::join!(
            retrieve,
            owner_presence_for_open(&state, &auth, presence_now)
        )
    };
    let (checkpoint_and_changes, ((search, hinted, retrieval_wall_ms), owner_presence)) =
        if state.config.read_path_roundtrip_v1 {
            tokio::join!(checkpoint_and_changes, retrieve_and_presence)
        } else {
            (checkpoint_and_changes.await, retrieve_and_presence.await)
        };
    let (mut checkpoint, change_page, checkpoint_read_ms, changes_ms) = checkpoint_and_changes?;

    let (checkpoint_text_truncated, mut evidence_budget) =
        apply_checkpoint_budget(&mut checkpoint, budget);
    let resume_delta_batch =
        if state.config.resume_deltas && request.resume_checkpoint_ref.is_some() {
            materialize_resume_deltas(
                &state,
                &auth,
                checkpoint.as_ref(),
                &change_page.changes,
                evidence_budget.saturating_mul(4),
            )
            .await?
        } else {
            ResumeDeltaBatch::default()
        };
    evidence_budget =
        evidence_budget.saturating_sub(resume_delta_batch.charged_chars.saturating_add(3) / 4);
    let (search_candidates, mut lane_failures, search_timings) = match search {
        Ok((candidates, failures, timings, _)) => (candidates, failures, timings),
        Err(error) => {
            tracing::warn!(?error, "simple workspace initial search failed");
            (vec![], vec!["search"], RetrievalTimings::default())
        }
    };
    let (hint_candidates, hint_gaps) = hinted?;
    let (continuation_paths, changed_path_keys) =
        continuation_paths(checkpoint.as_ref(), &change_page.changes);
    let continuation_candidates = if continuation_paths.is_empty() {
        vec![]
    } else {
        match bounded_retrieval_lane(
            "continuation_exact",
            exact_candidates(&state, &auth, &continuation_paths, None),
        )
        .await
        {
            Ok(mut candidates) => {
                for candidate in &mut candidates {
                    candidate.score =
                        if changed_path_keys.contains(&portable_path_key(&candidate.path)) {
                            30.0
                        } else {
                            20.0
                        };
                    candidate.lanes.push("continuation".to_owned());
                }
                candidates
            }
            Err(error) => {
                tracing::warn!(?error, "simple continuation source lookup failed");
                lane_failures.push("continuation_exact");
                vec![]
            }
        }
    };
    let mut merged = HashMap::new();
    for candidate in continuation_candidates
        .into_iter()
        .chain(hint_candidates)
        .chain(search_candidates)
    {
        merge_candidate(&mut merged, candidate);
    }
    let mut candidates = merged.into_values().collect::<Vec<_>>();
    annotate_candidates(
        &mut candidates,
        features.as_deref(),
        state.config.supersession_demotion,
    );
    sort_candidates(&mut candidates, SearchSort::BestMatch);
    candidates.truncate(OPEN_CANDIDATE_LIMIT);
    let continuation_evidence = candidates
        .iter()
        .filter(|candidate| {
            candidate
                .lanes
                .iter()
                .any(|lane| lane == "continuation" || lane == "hint")
        })
        .cloned()
        .collect::<Vec<_>>();
    let evidence_candidates = if continuation_evidence.is_empty() {
        &candidates
    } else {
        &continuation_evidence
    };
    let hydrate_started = Instant::now();
    let (evidence, hydrated_generation) =
        hydrate_candidates(&state, &auth, evidence_candidates, evidence_budget).await?;
    let hydrate_ms = elapsed_ms(hydrate_started);
    let generation = eager_generation.or(hydrated_generation).ok_or_else(|| {
        ApiError::Internal("read-path hydration did not return workspace generation".to_owned())
    })?;
    let hydrated_refs = evidence
        .iter()
        .filter_map(|item| item.get("reference").and_then(Value::as_str))
        .collect::<HashSet<_>>();
    let mut evidence_leads = candidates
        .iter()
        .filter(|candidate| {
            !hydrated_refs.contains(format!("entry:{}", candidate.entry_id).as_str())
        })
        .map(render_evidence_lead)
        .collect::<Vec<_>>();
    evidence_leads.extend(resume_delta_batch.leads);
    record_candidate_usage(&state, &auth, &candidates);

    let mut response_data = json!({
        "workspace_generation": generation,
        "authorization_scope": request.hints.authorization_scope,
        "evidence": evidence,
        "evidence_leads": evidence_leads,
        "checkpoint": checkpoint,
        "changes_since_checkpoint": change_page.changes,
        "changes_truncated": change_page.truncated,
        "next_changes_generation": change_page.next_generation,
        "checkpoint_text_truncated": checkpoint_text_truncated,
        "retrieval_sufficiency": {
            "status": if candidates.is_empty() { "no_evidence" } else { "bounded_evidence" },
            "complete_source_count": evidence.iter()
                .filter(|item| item.get("representation").and_then(Value::as_str) == Some("complete_source"))
                .count(),
            "selected_source_count": evidence.len(),
            "pointer_source_count": evidence_leads.len()
        }
    });
    if let Some(owner_presence) = owner_presence {
        response_data["owner_presence"] = owner_presence;
    }
    if state.config.intention_ledger {
        response_data["pending_intentions"] = json!(
            features
                .as_deref()
                .map(|snapshot| snapshot.pending_intentions(&request.task))
                .unwrap_or_default()
        );
    }
    if state.config.resume_deltas && request.resume_checkpoint_ref.is_some() {
        response_data["resume_deltas"] = Value::Array(resume_delta_batch.deltas);
    }
    let mut envelope = WorkspaceEnvelope::complete(response_data);
    envelope.session_id = Some(session_id);
    envelope.corpus_revision = Some(format!("generation:{generation}"));
    if checkpoint_text_truncated {
        envelope.gaps.push(json!({
            "kind": "checkpoint_text_budget",
            "message": "checkpoint text was truncated to the requested open budget"
        }));
    }
    envelope.gaps.extend(hint_gaps);
    if !lane_failures.is_empty() {
        envelope.status = if candidates.is_empty() {
            ResponseStatus::Degraded
        } else {
            ResponseStatus::Partial
        };
        envelope
            .gaps
            .extend(lane_failures.into_iter().map(retrieval_lane_gap));
    } else if !envelope.gaps.is_empty() {
        envelope.status = ResponseStatus::Partial;
    }
    let total_ms = elapsed_ms(total_started);
    if state.config.observability_timings_ms {
        let checkpoint_and_changes_ms = checkpoint_read_ms + changes_ms;
        let read_path_wall_ms = if state.config.read_path_roundtrip_v1 {
            checkpoint_and_changes_ms.max(retrieval_wall_ms)
        } else {
            checkpoint_and_changes_ms + retrieval_wall_ms
        };
        let attributed_ms = generation_ms + features_ms + read_path_wall_ms + hydrate_ms;
        envelope.timings_ms = Some(json!({
            "generation": round_ms(generation_ms),
            "features": round_ms(features_ms),
            "checkpoint_read": round_ms(checkpoint_read_ms),
            "changes": round_ms(changes_ms),
            "retrieval_wall": round_ms(retrieval_wall_ms),
            "lanes": search_timings.as_value(),
            "hydrate": round_ms(hydrate_ms),
            "unattributed": round_ms((total_ms - attributed_ms).max(0.0)),
            "total": round_ms(total_ms),
        }));
    }
    metrics::histogram!("simple.open.duration_ms").record(total_ms);
    state
        .foreground_latency
        .record(ForegroundOperation::Open, total_ms);
    metrics::histogram!("simple.open.evidence_sources").record(evidence.len() as f64);
    metrics::histogram!("simple.open.evidence_leads").record(evidence_leads.len() as f64);
    record_serialized_product_read(&state, &auth, ProductActivityOperation::Open, &envelope);
    Ok(Json(envelope))
}

async fn owner_presence_for_open(
    state: &AppState,
    auth: &AuthContext,
    now: DateTime<Utc>,
) -> Option<Value> {
    if !state.config.location_presence_in_open {
        record_location_presence_block("false:flag_off");
        return None;
    }
    render_owner_presence(location_store::read_presence(state, auth).await, now)
}

fn render_owner_presence(
    presence: ApiResult<Option<LocationPresenceState>>,
    now: DateTime<Utc>,
) -> Option<Value> {
    let presence = match presence {
        Ok(Some(presence)) => presence,
        Ok(None) => {
            record_location_presence_block("false:no_row");
            return None;
        }
        Err(_) => {
            record_location_presence_block("false:lookup_error");
            tracing::warn!("location presence lookup omitted from memory.open");
            return None;
        }
    };
    if now - presence.reported_at >= chrono::Duration::days(7) {
        record_location_presence_block("false:stale");
        return None;
    }
    match serde_json::to_value(presence_view(&presence, now)) {
        Ok(value) => {
            record_location_presence_block("true");
            Some(value)
        }
        Err(_) => {
            record_location_presence_block("false:render_error");
            tracing::warn!("location presence rendering omitted from memory.open");
            None
        }
    }
}

fn record_location_presence_block(included: &'static str) {
    metrics::counter!(
        "brunn.location.presence_block",
        "included" => included
    )
    .increment(1);
}

pub async fn search(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Json(request): Json<SearchRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    auth.require(Capability::Query)?;
    metrics::counter!("simple.search.requests").increment(1);
    let started = Instant::now();
    let budget_options = SearchBudgetOptions::from_request(&state.config, request.token_budget);
    let generation_started = Instant::now();
    let eager_generation = if !state.config.read_path_roundtrip_v1
        || state.config.supersession_demotion
        || state.config.intention_ledger
    {
        Some(current_generation(&state, &auth).await?)
    } else {
        None
    };
    let mut generation_ms = elapsed_ms(generation_started);
    let features_started = Instant::now();
    let features = match eager_generation {
        Some(generation) => feature_snapshot(&state, &auth, generation).await?,
        None => None,
    };
    let features_ms = elapsed_ms(features_started);
    let queries = if request.queries.is_empty() {
        vec![SearchQuery {
            id: Some("q0".to_owned()),
            query: request
                .query
                .ok_or_else(|| ApiError::invalid("query is required"))?,
            goal: None,
            limit: request.limit,
            sort: None,
            modes: vec![],
        }]
    } else {
        request.queries
    };
    if queries.len() > 16 {
        return Err(ApiError::invalid("at most 16 queries may be batched"));
    }
    for query in &queries {
        if query.query.trim().is_empty() {
            return Err(ApiError::invalid("search query is required"));
        }
        retrieval_lane_selection(&query.modes)?;
        SearchSort::parse(query.sort.as_deref())?;
    }
    let request_semantic_embeddings = if state.config.semantic_lane && state.config.embed_cache {
        RequestSemanticEmbeddings::new(
            queries
                .iter()
                .map(|query| {
                    retrieval_lane_selection(&query.modes)
                        .expect("search modes were validated above")
                        .semantic
                        .then(|| query.query.clone())
                })
                .collect(),
        )
    } else {
        None
    };
    let query_execution_started = Instant::now();
    let mut completed =
        futures::stream::iter(queries.into_iter().enumerate().map(|(index, query)| {
            let state = state.clone();
            let auth = auth.clone();
            let features = features.clone();
            let request_semantic_embeddings = request_semantic_embeddings.clone();
            async move {
                let limit = query.limit.unwrap_or(8).clamp(1, MAX_SEARCH_LIMIT);
                let result = search_one(
                    &state,
                    &auth,
                    &query.query,
                    limit,
                    &query.modes,
                    query.sort.as_deref(),
                    features.as_deref(),
                    request_semantic_embeddings.map(|batch| (batch, index)),
                )
                .await?;
                Ok::<_, ApiError>((index, query, result))
            }
        }))
        .buffer_unordered(SEARCH_QUERY_CONCURRENCY)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<ApiResult<Vec<_>>>()?;
    let query_execution_ms = elapsed_ms(query_execution_started);
    completed.sort_by_key(|(index, _, _)| *index);

    let budget_started = Instant::now();
    let (budgeted_views, budgeted_truncated) = if budget_options.active() {
        let candidate_sets = completed
            .iter()
            .map(|(_, _, (candidates, _, _, _))| candidates.clone())
            .collect::<Vec<_>>();
        let (selected, selection_truncated) = select_search_candidate_sets(
            &candidate_sets,
            budget_options.fair_share,
            MAX_SEARCH_RESPONSE_CANDIDATES,
        );
        let hydration = if budget_options.top1_hydration {
            fetch_search_top1_hydration(&state, &auth, &selected).await?
        } else {
            HashMap::new()
        };
        let (views, evidence_truncated) =
            assemble_search_candidate_views(selected, &hydration, budget_options);
        (Some(views), selection_truncated || evidence_truncated)
    } else {
        (None, false)
    };
    let mut result_sets = Vec::with_capacity(completed.len());
    let mut query_timings = Vec::with_capacity(completed.len());
    let mut any_candidates = false;
    let mut any_failures = false;
    let mut any_semantic_deferred = false;
    let mut all_candidates = Vec::new();
    let mut remaining_candidates = MAX_SEARCH_RESPONSE_CANDIDATES;
    let mut remaining_excerpt_chars = MAX_SEARCH_RESPONSE_CHARS;
    let mut remaining_verbatim_chars = MAX_VERBATIM_RESPONSE_CHARS;
    let mut response_truncated = budgeted_truncated;
    let mut piggyback_generation = None;
    for (position, (index, query, (mut candidates, failures, timings, generation))) in
        completed.into_iter().enumerate()
    {
        piggyback_generation = piggyback_generation.into_iter().chain(generation).max();
        query_timings.push(json!({
            "id": query.id.clone().unwrap_or_else(|| format!("q{index}")),
            "phases": timings.as_value(),
        }));
        let rendered = if let Some(views) = &budgeted_views {
            let query_views = &views[position];
            any_candidates |= !query_views.is_empty();
            all_candidates.extend(query_views.iter().map(|view| view.candidate.clone()));
            query_views
                .iter()
                .map(|view| render_budgeted_search_candidate(view, &mut remaining_verbatim_chars))
                .collect::<Vec<_>>()
        } else {
            if candidates.len() > remaining_candidates {
                candidates.truncate(remaining_candidates);
                response_truncated = true;
            }
            remaining_candidates = remaining_candidates.saturating_sub(candidates.len());
            for candidate in &mut candidates {
                if truncate_candidate_evidence(candidate, &mut remaining_excerpt_chars) {
                    response_truncated = true;
                }
            }
            any_candidates |= !candidates.is_empty();
            all_candidates.extend(candidates.iter().cloned());
            candidates
                .iter()
                .map(|candidate| render_search_candidate(candidate, &mut remaining_verbatim_chars))
                .collect::<Vec<_>>()
        };
        any_failures |= !failures.is_empty();
        any_semantic_deferred |= failures.contains(&"semantic_deferred");
        let mut result = serde_json::Map::from_iter([
            (
                "id".to_owned(),
                Value::String(query.id.unwrap_or_else(|| format!("q{index}"))),
            ),
            ("candidates".to_owned(), Value::Array(rendered)),
        ]);
        if let Some(goal) = query.goal {
            result.insert("goal".to_owned(), Value::String(goal));
        }
        if !failures.is_empty() {
            result.insert(
                "query_status".to_owned(),
                Value::String("partial".to_owned()),
            );
            result.insert("lane_failures".to_owned(), json!(failures));
        }
        result_sets.push(Value::Object(result));
    }
    let budget_ms = elapsed_ms(budget_started);
    record_candidate_usage(&state, &auth, &all_candidates);
    let generation = if let Some(generation) = eager_generation.or(piggyback_generation) {
        generation
    } else {
        let fallback_started = Instant::now();
        let generation = current_generation(&state, &auth).await?;
        generation_ms += elapsed_ms(fallback_started);
        generation
    };
    let mut response_data = serde_json::Map::from_iter([
        ("workspace_generation".to_owned(), json!(generation)),
        ("results".to_owned(), Value::Array(result_sets)),
    ]);
    if response_truncated {
        response_data.insert("response_truncated".to_owned(), Value::Bool(true));
    }
    let mut envelope = WorkspaceEnvelope::complete(Value::Object(response_data));
    envelope.session_id = request.session_id;
    envelope.corpus_revision = Some(format!("generation:{generation}"));
    if any_failures {
        envelope.status = if any_candidates {
            ResponseStatus::Partial
        } else {
            ResponseStatus::Degraded
        };
    }
    if any_semantic_deferred {
        envelope.gaps.push(json!({
            "kind": "retrieval_lane_deferred",
            "lane": "semantic",
            "reason": "deadline_deferred",
            "message": "semantic retrieval exceeded its accelerator deadline; exact and lexical evidence was retained"
        }));
    }
    let total_ms = elapsed_ms(started);
    if state.config.observability_timings_ms {
        let attributed_ms = query_execution_ms + budget_ms + generation_ms + features_ms;
        envelope.timings_ms = Some(json!({
            "queries": query_timings,
            "retrieval_wall": round_ms(query_execution_ms),
            "budget": round_ms(budget_ms),
            "generation": round_ms(generation_ms),
            "features": round_ms(features_ms),
            "unattributed": round_ms((total_ms - attributed_ms).max(0.0)),
            "total": round_ms(total_ms),
        }));
    }
    metrics::histogram!("simple.search.duration_ms").record(total_ms);
    state
        .foreground_latency
        .record(ForegroundOperation::Search, total_ms);
    metrics::histogram!("simple.search.candidates").record(all_candidates.len() as f64);
    record_serialized_product_read(&state, &auth, ProductActivityOperation::Search, &envelope);
    Ok(Json(envelope))
}

pub async fn read(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Json(request): Json<ReadRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    auth.require(Capability::Read)?;
    let started = Instant::now();
    if request.requests.is_empty() || request.requests.len() > 32 {
        return Err(ApiError::invalid(
            "read requires between 1 and 32 exact requests",
        ));
    }
    let requested_count = request.requests.len();
    let eager_generation = if !state.config.read_path_roundtrip_v1
        || state.config.supersession_demotion
        || state.config.intention_ledger
    {
        Some(current_generation(&state, &auth).await?)
    } else {
        None
    };
    let features = match eager_generation {
        Some(generation) => feature_snapshot(&state, &auth, generation).await?,
        None => None,
    };
    let mut items = Vec::with_capacity(requested_count);
    let mut used_entries = Vec::new();
    let mut remaining_chars = MAX_READ_RESPONSE_CHARS;
    let mut skipped_requests = 0_usize;
    let mut missing_requests = 0_usize;
    let mut piggyback_generation = None;
    let mut resolved = futures::stream::iter(request.requests.into_iter().enumerate().map(
        |(index, item)| {
            let state = state.clone();
            let auth = auth.clone();
            let features = features.clone();
            async move {
                let mut entry = if let Some(link_target) = item.link_target.as_deref() {
                    if item.path.is_some() || item.reference.is_some() || item.version.is_some() {
                        Err(ApiError::invalid(
                            "a link-target read cannot also specify a path, reference, or version",
                        ))
                    } else {
                        resolve_entry_link_version(&state, &auth, link_target).await
                    }
                } else {
                    resolve_entry_version(
                        &state,
                        &auth,
                        item.path.as_deref(),
                        item.reference.as_deref(),
                        item.version,
                    )
                    .await
                };
                let mut current_truth = None;
                let mut disabled_notice = false;
                if item.view.as_deref() == Some("current_truth") {
                    if state.config.supersession_demotion {
                        if let (Ok(requested), Some(snapshot)) = (&entry, features.as_deref()) {
                            let resolution = snapshot.resolve_current_truth(&requested.path);
                            if resolution.warning.is_none()
                                && resolution.head_path != requested.path
                            {
                                entry = resolve_entry_version(
                                    &state,
                                    &auth,
                                    Some(&resolution.head_path),
                                    None,
                                    None,
                                )
                                .await;
                            }
                            current_truth = Some(resolution);
                        }
                    } else {
                        disabled_notice = true;
                    }
                }
                (index, item, entry, current_truth, disabled_notice)
            }
        },
    ))
    .buffer_unordered(8)
    .collect::<Vec<_>>()
    .await;
    resolved.sort_by_key(|(index, _, _, _, _)| *index);
    for (_, item, resolved_entry, current_truth, disabled_notice) in resolved {
        if remaining_chars == 0 {
            skipped_requests += 1;
            continue;
        }
        let entry = match resolved_entry {
            Ok(entry) => entry,
            Err(ApiError::Public {
                status: StatusCode::NOT_FOUND,
                code,
                message,
                details,
            }) => {
                missing_requests += 1;
                items.push(json!({
                    "status": "not_found",
                    "path": item.path.clone().or_else(|| item.link_target.clone()),
                    "reference": item.reference,
                    "link_target": item.link_target,
                    "error": {
                        "code": code,
                        "message": message,
                        "details": details
                    }
                }));
                continue;
            }
            Err(error) => return Err(error),
        };
        piggyback_generation = piggyback_generation
            .into_iter()
            .chain(entry.workspace_generation)
            .max();
        used_entries.push(entry.id);
        let exact_read_limit = exact_read_char_limit(requested_count, &item, &entry);
        if requested_count == 1 {
            remaining_chars = exact_read_limit;
        }
        let max_chars = item
            .max_chars
            .unwrap_or(256_000)
            .clamp(1, exact_read_limit)
            .min(remaining_chars);
        let mut rendered = render_read(&entry, &item, max_chars)?;
        if let Some(resolution) = current_truth {
            rendered["supersession_chain"] = json!(resolution.chain);
            if let Some(warning) = resolution.warning {
                rendered["supersession_warning"] = json!(warning);
            }
        } else if disabled_notice {
            rendered["current_truth_notice"] = Value::String(
                "supersession_demotion is disabled; the requested document was returned unchanged"
                    .to_owned(),
            );
        }
        remaining_chars = remaining_chars.saturating_sub(
            rendered
                .get("text")
                .and_then(Value::as_str)
                .map(|text| text.chars().count())
                .unwrap_or(0),
        );
        items.push(rendered);
    }
    record_entry_usage(&state, &auth, &used_entries, UsageOperation::Read);
    let returned_entries = used_entries.len();
    let generation = if let Some(generation) = eager_generation.or(piggyback_generation) {
        generation
    } else {
        current_generation(&state, &auth).await?
    };
    let mut envelope = WorkspaceEnvelope::complete(json!({
        "workspace_generation": generation,
        "items": items,
        "response_truncated": skipped_requests > 0,
        "missing_requests": missing_requests,
        "requested_count": requested_count
    }));
    envelope.session_id = request.session_id;
    envelope.corpus_revision = Some(format!("generation:{generation}"));
    if skipped_requests > 0 || missing_requests > 0 {
        envelope.status = if returned_entries == 0 {
            ResponseStatus::Degraded
        } else {
            ResponseStatus::Partial
        };
    }
    if skipped_requests > 0 {
        envelope.gaps.push(json!({
            "kind": "read_response_budget",
            "skipped_requests": skipped_requests,
            "message": "request the remaining exact entries in a follow-up read"
        }));
    }
    if missing_requests > 0 {
        envelope.gaps.push(json!({
            "kind": "read_entries_not_found",
            "missing_requests": missing_requests,
            "message": "missing entries were reported per item; valid exact reads were retained"
        }));
    }
    metrics::histogram!("simple.read.duration_ms")
        .record(started.elapsed().as_secs_f64() * 1_000.0);
    metrics::histogram!("simple.read.entries").record(returned_entries as f64);
    record_serialized_product_read(&state, &auth, ProductActivityOperation::Read, &envelope);
    Ok(Json(envelope))
}

fn exact_read_char_limit(requested_count: usize, request: &ReadItem, entry: &EntryRow) -> usize {
    if requested_count == 1
        && request.view.as_deref().unwrap_or("full") == "full"
        && entry.kind == "markdown"
        && entry.media_type == "text/markdown"
        && entry.size_bytes >= 0
        && usize::try_from(entry.size_bytes)
            .is_ok_and(|size| size <= crate::messaging_protocol::MAX_CANONICAL_CONVERSATION_BYTES)
        && entry.content.as_deref().is_some_and(|content| {
            crate::messaging_protocol::validate_conversation_entry(
                &entry.path,
                &entry.metadata,
                content,
            )
            .is_ok_and(|value| value.is_some())
        })
    {
        crate::messaging_protocol::MAX_CANONICAL_CONVERSATION_BYTES
    } else {
        MAX_EXACT_READ_CHARS
    }
}

pub async fn write(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Json(request): Json<WriteRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    require_write_capabilities(&auth, &request.path)?;
    validate_write_path(&request)?;
    // Inlet rule: chronicle "no durable memory" no-op summaries under
    // agent-memory/** are not admitted. The path prefix IS the
    // classification; the producer receives a clean no-op receipt.
    if is_agent_memory_noop_summary(&request.path, &request.content) {
        let mut envelope = WorkspaceEnvelope::complete(json!({
            "path": request.path,
            "no_op": true,
            "reason": "agent_memory_noop_summary",
        }));
        envelope.status = ResponseStatus::NoOp;
        return Ok(Json(envelope));
    }
    let prepared = prepare_markdown(&state, request).await?;
    let committed_bytes = u64::try_from(prepared.content.len()).unwrap_or(u64::MAX);
    let receipt = commit_markdown(&state, &auth, prepared).await?;
    let mut envelope = WorkspaceEnvelope::complete(receipt.clone());
    envelope.status = if receipt.get("no_op") == Some(&Value::Bool(true)) {
        ResponseStatus::NoOp
    } else {
        ResponseStatus::Committed
    };
    envelope.corpus_revision = receipt
        .get("workspace_generation")
        .and_then(Value::as_i64)
        .map(|generation| format!("generation:{generation}"));
    if receipt.get("no_op") != Some(&Value::Bool(true)) {
        record_product_activity(
            &state,
            &auth,
            ProductActivityOperation::Write,
            committed_bytes,
        );
    }
    Ok(Json(envelope))
}

pub async fn capture(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Json(request): Json<CaptureRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    auth.require(Capability::Save)?;
    if request.content.trim().is_empty() {
        return Err(ApiError::invalid("capture content is required"));
    }
    let stable_capture = request.idempotency_key.is_some();
    let path = match request.idempotency_key.as_deref() {
        Some(key) => {
            validate_idempotency_key(key)?;
            let digest = hex::encode(Sha256::digest(
                format!("{}\0{key}", auth.user_id.0).as_bytes(),
            ));
            format!("Inbox/Captures/Stable/{digest}.md")
        }
        None => {
            let date = Utc::now().format("%Y/%m/%d");
            format!("Inbox/Captures/{date}/{}.md", Uuid::now_v7())
        }
    };
    let source = serde_json::to_string_pretty(&request.source)?;
    let content = format!(
        "# Captured context\n\n{}\n\n## Source\n\n```json\n{}\n```\n",
        request.content.trim(),
        source
    );
    let prepared = prepare_markdown(
        &state,
        WriteRequest {
            path,
            content,
            media_type: markdown_media_type(),
            expected_version: stable_capture.then_some(0),
            idempotency_key: request.idempotency_key,
            metadata: json!({
                "kind": "capture",
                "intent": request.intent
            }),
        },
    )
    .await?;
    let committed_bytes = u64::try_from(prepared.content.len()).unwrap_or(u64::MAX);
    let receipt = commit_markdown(&state, &auth, prepared).await?;
    let mut envelope = WorkspaceEnvelope::complete(receipt.clone());
    envelope.status = if receipt.get("no_op") == Some(&Value::Bool(true)) {
        ResponseStatus::NoOp
    } else {
        ResponseStatus::Committed
    };
    envelope.corpus_revision = receipt
        .get("workspace_generation")
        .and_then(Value::as_i64)
        .map(|generation| format!("generation:{generation}"));
    if receipt.get("no_op") != Some(&Value::Bool(true)) {
        record_product_activity(
            &state,
            &auth,
            ProductActivityOperation::Capture,
            committed_bytes,
        );
    }
    Ok(Json(envelope))
}

pub async fn checkpoint(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Json(request): Json<CheckpointRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    auth.require(Capability::Checkpoint)?;
    let (idempotency_key, request_hash) = validate_checkpoint_request(&request)?;
    let checkpoint_uuid = checkpoint_entry_id_for_new_write(&request);
    let checkpoint_ref = format!("checkpoint:{checkpoint_uuid}");
    let path = format!(".brunn/checkpoints/{checkpoint_uuid}.md");
    let mut tx = state.begin_write(&auth).await?;
    lock_checkpoint_idempotency(&mut tx, auth.user_id.0, &idempotency_key).await?;
    if let Some(receipt) =
        replay_checkpoint_receipt_in_tx(&mut tx, auth.user_id.0, &idempotency_key, &request_hash)
            .await?
    {
        tx.commit().await?;
        return Ok(Json(checkpoint_envelope(
            &request.session_id,
            receipt,
            true,
        )?));
    }
    if let Some(adopted) = adopt_legacy_checkpoint_receipt_in_tx(
        &mut tx,
        auth.user_id.0,
        Some(auth.credential_id.0),
        &idempotency_key,
        &request_hash,
        &request,
    )
    .await?
    {
        tx.commit().await?;
        return Ok(Json(checkpoint_envelope(
            &request.session_id,
            adopted.receipt,
            true,
        )?));
    }
    if let Some(parent_checkpoint_ref) = request.parent_checkpoint_id.as_deref() {
        validate_checkpoint_parent_in_tx(&mut tx, auth.user_id.0, parent_checkpoint_ref).await?;
    }
    let (source_entries, source_snapshot_generation) = resolve_checkpoint_sources_in_tx(
        &mut tx,
        auth.user_id.0,
        &request.state,
        &request.source_refs,
    )
    .await?;
    let pinned_generation = match source_snapshot_generation {
        Some(generation) => generation,
        None => max_generation_in_tx(&mut tx, auth.user_id.0).await?,
    };
    let content = render_checkpoint_markdown(
        checkpoint_uuid,
        pinned_generation,
        &request,
        &source_entries,
    )?;
    let source_entry_receipts = source_entries
        .iter()
        .map(entry_reference)
        .collect::<Vec<_>>();
    let mut prepared = prepare_markdown(
        &state,
        WriteRequest {
            path: path.clone(),
            content,
            media_type: markdown_media_type(),
            expected_version: Some(0),
            idempotency_key: Some(idempotency_key.clone()),
            metadata: json!({
                "kind": "checkpoint",
                "checkpoint_ref": checkpoint_ref,
                "workspace_generation": pinned_generation,
                "pinned_workspace_generation": pinned_generation,
                "resulting_workspace_generation": null,
                "session_id": request.session_id.clone(),
                "parent_checkpoint_ref": request.parent_checkpoint_id.clone(),
                "checkpoint_state": request.state.clone(),
                "project": request.state.get("project").cloned().unwrap_or(Value::Null),
                "source_refs": request.source_refs.clone(),
                "source_entries": source_entry_receipts.clone(),
                "request_hash": format!("sha256:{request_hash}"),
                "operation_kind": "checkpoint"
            }),
        },
    )
    .await?;
    prepared.entry_id_hint = Some(checkpoint_uuid);
    let write = commit_checkpoint_in_tx(
        &mut tx,
        auth.user_id.0,
        Some(auth.credential_id.0),
        &idempotency_key,
        &request_hash,
        &checkpoint_ref,
        &path,
        pinned_generation,
        source_entry_receipts,
        prepared,
    )
    .await?;
    tx.commit().await?;
    state.workspace_features.invalidate(auth.user_id.0).await;
    if write.created {
        record_product_activity(
            &state,
            &auth,
            ProductActivityOperation::Checkpoint,
            write.committed_bytes,
        );
    }
    Ok(Json(checkpoint_envelope(
        &request.session_id,
        write.receipt,
        false,
    )?))
}

pub async fn changes(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Query(query): Query<ChangesQuery>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    auth.require(Capability::Read)?;
    let limit = query.limit.unwrap_or(200).clamp(1, 2_000);
    let page = changes_since(&state, &auth, query.since_generation, limit).await?;
    let generation = match (
        state.config.read_path_roundtrip_v1,
        page.workspace_generation,
    ) {
        (true, Some(generation)) => generation,
        _ => current_generation(&state, &auth).await?,
    };
    let mut envelope = WorkspaceEnvelope::complete(json!({
        "since_generation": query.since_generation,
        "workspace_generation": generation,
        "changes": page.changes,
        "truncated": page.truncated,
        "next_generation": page.next_generation
    }));
    envelope.corpus_revision = Some(format!("generation:{generation}"));
    Ok(Json(envelope))
}

pub async fn delete_entry(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(entry_ref): Path<String>,
    Query(query): Query<DeleteQuery>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    auth.require(Capability::Delete)?;
    let entry_id = entry_ref
        .strip_prefix("entry:")
        .unwrap_or(&entry_ref)
        .parse::<Uuid>()
        .map_err(|_| ApiError::invalid("delete requires an entry:<uuid> reference"))?;
    let mut tx = state.begin_write(&auth).await?;
    let row = sqlx::query(
        r#"
        SELECT entry.path,entry.kind,entry.current_version,entry.deleted_at,
               version.content_sha256,version.metadata
        FROM brunn.entries AS entry
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
         AND version.version=entry.current_version
        WHERE entry.user_id=$1 AND entry.id=$2
        FOR UPDATE OF entry
        "#,
    )
    .bind(auth.user_id.0)
    .bind(entry_id)
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(|| ApiError::not_found("entry_not_found", &entry_ref))?;
    let path: String = row.get("path");
    validate_public_path(&path)?;
    let version: i64 = row.get("current_version");
    if let Some(expected_version) = query.expected_version
        && expected_version != version
    {
        return Err(ApiError::conflict(
            "entry_version_conflict",
            "the entry changed since it was read",
            json!({
                "entry_ref": format!("entry:{entry_id}"),
                "expected_version": expected_version,
                "actual_version": version
            }),
        ));
    }
    if row.get::<Option<DateTime<Utc>>, _>("deleted_at").is_some() {
        tx.commit().await?;
        let mut envelope = WorkspaceEnvelope::complete(json!({
            "entry_ref": format!("entry:{entry_id}"),
            "path": path,
            "version": version,
            "no_op": true
        }));
        envelope.status = ResponseStatus::NoOp;
        return Ok(Json(envelope));
    }
    sqlx::query("DELETE FROM brunn.search_chunks WHERE user_id=$1 AND entry_id=$2")
        .bind(auth.user_id.0)
        .bind(entry_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query(
        r#"
        UPDATE brunn.entries
        SET deleted_at=clock_timestamp(),updated_at=clock_timestamp()
        WHERE user_id=$1 AND id=$2
        "#,
    )
    .bind(auth.user_id.0)
    .bind(entry_id)
    .execute(&mut *tx)
    .await?;
    let mut generation = sqlx::query_scalar::<_, i64>(
        r#"
        INSERT INTO brunn.workspace_changes (
          user_id,entry_id,entry_version,operation,path,content_sha256
        ) VALUES ($1,$2,$3,'delete',$4,$5)
        RETURNING generation
        "#,
    )
    .bind(auth.user_id.0)
    .bind(entry_id)
    .bind(version)
    .bind(&path)
    .bind(row.get::<String, _>("content_sha256"))
    .fetch_one(&mut *tx)
    .await?;
    let entry_metadata: Value = row.get("metadata");
    if row.get::<String, _>("kind") == "binary"
        && let Some(companion_path) = entry_metadata.get("companion_path").and_then(Value::as_str)
    {
        let companion = sqlx::query(
            r#"
            SELECT entry.id,entry.current_version,version.content_sha256
            FROM brunn.entries AS entry
            JOIN brunn.entry_versions AS version
              ON version.user_id=entry.user_id
             AND version.entry_id=entry.id
             AND version.version=entry.current_version
            WHERE entry.user_id=$1
              AND lower(normalize(entry.path, NFC))=$2
              AND entry.deleted_at IS NULL
            FOR UPDATE OF entry
            "#,
        )
        .bind(auth.user_id.0)
        .bind(portable_path_key(companion_path))
        .fetch_optional(&mut *tx)
        .await?;
        if let Some(companion) = companion {
            let companion_id: Uuid = companion.get("id");
            let companion_version: i64 = companion.get("current_version");
            sqlx::query("DELETE FROM brunn.search_chunks WHERE user_id=$1 AND entry_id=$2")
                .bind(auth.user_id.0)
                .bind(companion_id)
                .execute(&mut *tx)
                .await?;
            sqlx::query(
                r#"
                UPDATE brunn.entries
                SET deleted_at=clock_timestamp(),updated_at=clock_timestamp()
                WHERE user_id=$1 AND id=$2
                "#,
            )
            .bind(auth.user_id.0)
            .bind(companion_id)
            .execute(&mut *tx)
            .await?;
            let companion_generation = sqlx::query_scalar::<_, i64>(
                r#"
                INSERT INTO brunn.workspace_changes (
                  user_id,entry_id,entry_version,operation,path,content_sha256
                ) VALUES ($1,$2,$3,'delete',$4,$5)
                RETURNING generation
                "#,
            )
            .bind(auth.user_id.0)
            .bind(companion_id)
            .bind(companion_version)
            .bind(companion_path)
            .bind(companion.get::<String, _>("content_sha256"))
            .fetch_one(&mut *tx)
            .await?;
            generation = generation.max(companion_generation);
        }
    }
    tx.commit().await?;
    state.workspace_features.invalidate(auth.user_id.0).await;
    let mut envelope = WorkspaceEnvelope::complete(json!({
        "entry_ref": format!("entry:{entry_id}"),
        "path": path,
        "version": version,
        "workspace_generation": generation,
        "no_op": false
    }));
    envelope.status = ResponseStatus::Committed;
    envelope.corpus_revision = Some(format!("generation:{generation}"));
    record_product_activity(&state, &auth, ProductActivityOperation::Delete, 0);
    Ok(Json(envelope))
}

pub async fn list_jobs(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Query(query): Query<JobsQuery>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    auth.require(Capability::Read)?;
    if let Some(status) = query.status.as_deref()
        && !matches!(status, "queued" | "running" | "complete" | "failed")
    {
        return Err(ApiError::invalid("unsupported job status"));
    }
    let limit = query.limit.unwrap_or(100).clamp(1, 500) as i64;
    let offset = query.offset.unwrap_or(0).min(100_000) as i64;
    let mut tx = state.begin_read(&auth).await?;
    let rows = sqlx::query(
        r#"
        SELECT id,kind,status,payload,watermark,attempts,available_at,
               started_at,finished_at,last_error,created_at
        FROM brunn.jobs
        WHERE user_id=$1
          AND ($2::text IS NULL OR status=$2)
        ORDER BY created_at DESC,id DESC
        LIMIT $3 OFFSET $4
        "#,
    )
    .bind(auth.user_id.0)
    .bind(query.status.as_deref())
    .bind(limit)
    .bind(offset)
    .fetch_all(&mut *tx)
    .await?;
    tx.commit().await?;
    let jobs = rows
        .into_iter()
        .map(|row| {
            let id: Uuid = row.get("id");
            json!({
                "job_ref": format!("job:{id}"),
                "kind": row.get::<String, _>("kind"),
                "status": row.get::<String, _>("status"),
                "payload": row.get::<Value, _>("payload"),
                "watermark": row.get::<Option<i64>, _>("watermark"),
                "attempts": row.get::<i32, _>("attempts"),
                "available_at": row.get::<DateTime<Utc>, _>("available_at"),
                "started_at": row.get::<Option<DateTime<Utc>>, _>("started_at"),
                "finished_at": row.get::<Option<DateTime<Utc>>, _>("finished_at"),
                "last_error": row.get::<Option<String>, _>("last_error"),
                "created_at": row.get::<DateTime<Utc>, _>("created_at")
            })
        })
        .collect::<Vec<_>>();
    Ok(Json(WorkspaceEnvelope::complete(json!({
        "jobs": jobs,
        "offset": offset,
        "limit": limit,
        "truncated": jobs.len() == usize::try_from(limit).unwrap_or(usize::MAX)
    }))))
}

pub async fn list_binaries(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Query(query): Query<BinaryListQuery>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    auth.require(Capability::Read)?;
    let limit = query.limit.unwrap_or(100).clamp(1, 500) as i64;
    let offset = query.offset.unwrap_or(0).min(100_000) as i64;
    let mut tx = state.begin_read(&auth).await?;
    let rows = sqlx::query(
        r#"
        SELECT entry.id,entry.path,entry.title,entry.media_type,
               entry.current_version,version.content_sha256,
               version.size_bytes,version.metadata,entry.updated_at,
               companion_version.metadata AS companion_metadata
        FROM brunn.entries AS entry
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
         AND version.version=entry.current_version
        LEFT JOIN brunn.entries AS companion
          ON companion.user_id=entry.user_id
         AND lower(normalize(companion.path, NFC))
             =lower(normalize(version.metadata->>'companion_path', NFC))
         AND companion.kind='markdown'
         AND companion.deleted_at IS NULL
        LEFT JOIN brunn.entry_versions AS companion_version
          ON companion_version.user_id=companion.user_id
         AND companion_version.entry_id=companion.id
         AND companion_version.version=companion.current_version
        WHERE entry.user_id=$1
          AND entry.kind='binary'
          AND entry.deleted_at IS NULL
        ORDER BY entry.path
        LIMIT $2 OFFSET $3
        "#,
    )
    .bind(auth.user_id.0)
    .bind(limit)
    .bind(offset)
    .fetch_all(&mut *tx)
    .await?;
    tx.commit().await?;
    let binaries = rows
        .into_iter()
        .map(|row| {
            let id: Uuid = row.get("id");
            let metadata = merge_binary_description_metadata(
                row.get("metadata"),
                row.get("companion_metadata"),
            );
            json!({
                "entry_ref": format!("entry:{id}"),
                "path": row.get::<String, _>("path"),
                "title": row.get::<String, _>("title"),
                "media_type": row.get::<String, _>("media_type"),
                "version": row.get::<i64, _>("current_version"),
                "content_hash": format!("sha256:{}", row.get::<String, _>("content_sha256")),
                "size_bytes": row.get::<i64, _>("size_bytes"),
                "metadata": metadata,
                "updated_at": row.get::<DateTime<Utc>, _>("updated_at")
            })
        })
        .collect::<Vec<_>>();
    Ok(Json(WorkspaceEnvelope::complete(
        json!({"binaries": binaries}),
    )))
}

pub async fn manifest(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Query(query): Query<ManifestQuery>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    auth.require(Capability::Read)?;
    let limit = query.limit.unwrap_or(1_000).clamp(1, 5_000) as i64;
    let offset = query.offset.unwrap_or(0).min(10_000_000) as i64;
    let cursor_supplied = query.after_path.is_some()
        || query.after_entry_ref.is_some()
        || query.after_version.is_some();
    if cursor_supplied && offset > 0 {
        return Err(ApiError::invalid(
            "manifest cursor and offset cannot be used together",
        ));
    }
    let cursor_entry_id = match (
        query.after_path.as_deref(),
        query.after_entry_ref.as_deref(),
    ) {
        (None, None) => None,
        (Some(_), Some(reference)) => Some(
            reference
                .strip_prefix("entry:")
                .unwrap_or(reference)
                .parse::<Uuid>()
                .map_err(|_| ApiError::invalid("after_entry_ref must be an entry:<uuid>"))?,
        ),
        _ => {
            return Err(ApiError::invalid(
                "after_path and after_entry_ref must be provided together",
            ));
        }
    };
    if query.history {
        if cursor_entry_id.is_some() != query.after_version.is_some() {
            return Err(ApiError::invalid(
                "history cursors require after_path, after_entry_ref, and after_version",
            ));
        }
        if query.after_version.is_some_and(|version| version <= 0) {
            return Err(ApiError::invalid("after_version must be positive"));
        }
    } else if query.after_version.is_some() {
        return Err(ApiError::invalid(
            "after_version is only valid for history manifests",
        ));
    }

    let mut tx = state.begin_read(&auth).await?;
    let mut statement = if query.history {
        QueryBuilder::<Postgres>::new(
            r#"
            SELECT entry.id,entry.path,entry.title,entry.kind,entry.media_type,
                   entry.current_version,entry.deleted_at,
                   version.id AS version_id,version.version AS version_number,
                   version.content_sha256,version.size_bytes,version.metadata,
                   version.created_at AS version_created_at
            FROM brunn.entries AS entry
            JOIN brunn.entry_versions AS version
              ON version.user_id=entry.user_id
             AND version.entry_id=entry.id
            WHERE entry.user_id=
            "#,
        )
    } else {
        QueryBuilder::<Postgres>::new(
            r#"
            SELECT entry.id,entry.path,entry.title,entry.kind,entry.media_type,
                   entry.current_version,entry.deleted_at,
                   version.id AS version_id,version.version AS version_number,
                   version.content_sha256,version.size_bytes,version.metadata,
                   version.created_at AS version_created_at
            FROM brunn.entries AS entry
            JOIN brunn.entry_versions AS version
              ON version.user_id=entry.user_id
             AND version.entry_id=entry.id
             AND version.version=entry.current_version
            WHERE entry.user_id=
            "#,
        )
    };
    statement.push_bind(auth.user_id.0);
    if !query.history {
        statement.push(" AND entry.deleted_at IS NULL");
    }
    if let (Some(after_path), Some(entry_id)) = (query.after_path.as_deref(), cursor_entry_id) {
        if query.history {
            statement
                .push(" AND (entry.path,entry.id,version.version) > (")
                .push_bind(after_path)
                .push(",")
                .push_bind(entry_id)
                .push(",")
                .push_bind(query.after_version.expect("validated above"))
                .push(")");
        } else {
            statement
                .push(" AND (entry.path,entry.id) > (")
                .push_bind(after_path)
                .push(",")
                .push_bind(entry_id)
                .push(")");
        }
    }
    statement.push(" ORDER BY entry.path,entry.id");
    if query.history {
        statement.push(",version.version");
    }
    statement.push(" LIMIT ").push_bind(limit + 1);
    if !cursor_supplied && offset > 0 {
        statement.push(" OFFSET ").push_bind(offset);
    }
    let mut rows = statement.build().fetch_all(&mut *tx).await?;
    let generation =
        sqlx::query_scalar::<_, Option<i64>>("SELECT brunn_auth.workspace_generation($1)")
            .bind(auth.user_id.0)
            .fetch_one(&mut *tx)
            .await?
            .unwrap_or(0);
    tx.commit().await?;
    let truncated = rows.len() > usize::try_from(limit).unwrap_or(usize::MAX);
    if truncated {
        rows.truncate(usize::try_from(limit).unwrap_or(usize::MAX));
    }
    let next = if truncated {
        rows.last().map(|row| {
            let id: Uuid = row.get("id");
            let mut cursor = serde_json::Map::from_iter([
                (
                    "after_path".to_owned(),
                    Value::String(row.get::<String, _>("path")),
                ),
                (
                    "after_entry_ref".to_owned(),
                    Value::String(format!("entry:{id}")),
                ),
            ]);
            if query.history {
                cursor.insert(
                    "after_version".to_owned(),
                    json!(row.get::<i64, _>("version_number")),
                );
            }
            Value::Object(cursor)
        })
    } else {
        None
    };
    let entries = rows
        .into_iter()
        .map(|row| {
            let id: Uuid = row.get("id");
            json!({
                "entry_ref": format!("entry:{id}"),
                "path": row.get::<String, _>("path"),
                "title": row.get::<String, _>("title"),
                "kind": row.get::<String, _>("kind"),
                "media_type": row.get::<String, _>("media_type"),
                "version": row.get::<i64, _>("version_number"),
                "version_ref": format!(
                    "entry-version:{}",
                    row.get::<Uuid, _>("version_id")
                ),
                "current": row.get::<i64, _>("version_number")
                    == row.get::<i64, _>("current_version"),
                "deleted": row.get::<Option<DateTime<Utc>>, _>("deleted_at").is_some(),
                "content_hash": format!("sha256:{}", row.get::<String, _>("content_sha256")),
                "size_bytes": row.get::<i64, _>("size_bytes"),
                "metadata": row.get::<Value, _>("metadata"),
                "created_at": row.get::<DateTime<Utc>, _>("version_created_at")
            })
        })
        .collect::<Vec<_>>();
    let mut envelope = WorkspaceEnvelope::complete(json!({
        "workspace_generation": generation,
        "entries": entries,
        "history": query.history,
        "offset": offset,
        "limit": limit,
        "truncated": truncated,
        "next": next
    }));
    envelope.corpus_revision = Some(format!("generation:{generation}"));
    Ok(Json(envelope))
}

pub async fn usage(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Query(query): Query<UsageQuery>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    auth.require(Capability::Read)?;
    let sort = query.sort.as_deref().unwrap_or("most_used");
    let ordering = match sort {
        "most_used" => {
            "(coalesce(usage.read_count,0)+coalesce(usage.search_count,0)) DESC, entry.path"
        }
        "least_used" => "(coalesce(usage.read_count,0)+coalesce(usage.search_count,0)), entry.path",
        "least_recently_used" => "usage.last_used_at NULLS FIRST, entry.path",
        "most_recently_used" => "usage.last_used_at DESC NULLS LAST, entry.path",
        _ => {
            return Err(ApiError::invalid(
                "usage sort must be most_used, least_used, least_recently_used, or most_recently_used",
            ));
        }
    };
    let limit = query.limit.unwrap_or(100).clamp(1, 500) as i64;
    let offset = query.offset.unwrap_or(0).min(10_000_000) as i64;
    let mut statement = QueryBuilder::<Postgres>::new(
        r#"
        SELECT entry.id,entry.path,entry.title,entry.kind,
               coalesce(usage.read_count,0) AS read_count,
               coalesce(usage.search_count,0) AS search_count,
               usage.first_used_at,usage.last_used_at,
               usage.last_read_at,usage.last_search_at
        FROM brunn.entries AS entry
        LEFT JOIN brunn.entry_usage AS usage
          ON usage.user_id=entry.user_id AND usage.entry_id=entry.id
        WHERE entry.user_id=
        "#,
    );
    statement
        .push_bind(auth.user_id.0)
        .push(" AND entry.deleted_at IS NULL ORDER BY ")
        .push(ordering)
        .push(" LIMIT ")
        .push_bind(limit)
        .push(" OFFSET ")
        .push_bind(offset);
    let mut tx = state.begin_read(&auth).await?;
    let rows = statement.build().fetch_all(&mut *tx).await?;
    tx.commit().await?;
    let entries = rows
        .into_iter()
        .map(|row| {
            let id: Uuid = row.get("id");
            let reads: i64 = row.get("read_count");
            let searches: i64 = row.get("search_count");
            json!({
                "entry_ref": format!("entry:{id}"),
                "path": row.get::<String, _>("path"),
                "title": row.get::<String, _>("title"),
                "kind": row.get::<String, _>("kind"),
                "read_count": reads,
                "search_count": searches,
                "total_uses": reads.saturating_add(searches),
                "first_used_at": row.get::<Option<DateTime<Utc>>, _>("first_used_at"),
                "last_used_at": row.get::<Option<DateTime<Utc>>, _>("last_used_at"),
                "last_read_at": row.get::<Option<DateTime<Utc>>, _>("last_read_at"),
                "last_search_at": row.get::<Option<DateTime<Utc>>, _>("last_search_at")
            })
        })
        .collect::<Vec<_>>();
    Ok(Json(WorkspaceEnvelope::complete(json!({
        "sort": sort,
        "entries": entries,
        "offset": offset,
        "limit": limit
    }))))
}

pub async fn upload_binary(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    mut multipart: Multipart,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    auth.require(Capability::Stage)?;
    let mut path = None;
    let mut media_type = None;
    let mut description = None;
    let mut provenance = None;
    let mut limitations = None;
    let mut mtime_ns = None;
    let mut mode = None;
    let mut expected_version = None;
    let mut expected_content_hash = None;
    let mut portable_companion_format = None;
    let mut portable_companion_path = None;
    let mut portable_companion_sha256 = None;
    let mut portable_companion_mtime_ns = None;
    let mut portable_companion_mode = None;
    let mut bytes: Option<Bytes> = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::invalid(format!("invalid binary upload: {error}")))?
    {
        let name = field.name().unwrap_or_default().to_owned();
        match name.as_str() {
            "file" => {
                if bytes.is_some() {
                    return Err(ApiError::invalid(
                        "workspace binary upload accepts exactly one file",
                    ));
                }
                if media_type.is_none() {
                    media_type = field.content_type().map(ToOwned::to_owned);
                }
                bytes = Some(field.bytes().await.map_err(|error| {
                    ApiError::invalid(format!("invalid binary bytes: {error}"))
                })?);
            }
            "path" => path = Some(read_small_field(field, "path", 1_024).await?),
            "media_type" => media_type = Some(read_small_field(field, "media_type", 255).await?),
            "description" => {
                description = Some(read_small_field(field, "description", 256_000).await?)
            }
            "provenance" => provenance = Some(read_small_field(field, "provenance", 32_000).await?),
            "limitations" => {
                limitations = Some(read_small_field(field, "limitations", 32_000).await?)
            }
            "mtime_ns" => {
                let value = read_small_field(field, "mtime_ns", 32).await?;
                mtime_ns = Some(
                    value
                        .parse::<i64>()
                        .map_err(|_| ApiError::invalid("mtime_ns must be an integer"))?,
                );
            }
            "mode" => {
                let value = read_small_field(field, "mode", 16).await?;
                let parsed = value
                    .parse::<u32>()
                    .map_err(|_| ApiError::invalid("mode must be an integer"))?;
                if parsed > 0o777 {
                    return Err(ApiError::invalid(
                        "mode may contain only portable permission bits",
                    ));
                }
                mode = Some(parsed);
            }
            "expected_version" => {
                let value = read_small_field(field, "expected_version", 32).await?;
                expected_version = Some(
                    value
                        .parse::<i64>()
                        .map_err(|_| ApiError::invalid("expected_version must be an integer"))?,
                );
            }
            "expected_content_hash" => {
                expected_content_hash =
                    Some(read_small_field(field, "expected_content_hash", 80).await?);
            }
            "portable_companion_format" => {
                portable_companion_format =
                    Some(read_small_field(field, "portable_companion_format", 120).await?)
            }
            "portable_companion_path" => {
                portable_companion_path =
                    Some(read_small_field(field, "portable_companion_path", 1_024).await?)
            }
            "portable_companion_sha256" => {
                portable_companion_sha256 =
                    Some(read_small_field(field, "portable_companion_sha256", 80).await?)
            }
            "portable_companion_mtime_ns" => {
                let value = read_small_field(field, "portable_companion_mtime_ns", 32).await?;
                portable_companion_mtime_ns = Some(value.parse::<i64>().map_err(|_| {
                    ApiError::invalid("portable_companion_mtime_ns must be an integer")
                })?);
            }
            "portable_companion_mode" => {
                let value = read_small_field(field, "portable_companion_mode", 16).await?;
                portable_companion_mode = Some(value.parse::<u32>().map_err(|_| {
                    ApiError::invalid("portable_companion_mode must be an integer")
                })?);
            }
            _ => {
                return Err(ApiError::invalid(format!(
                    "unsupported binary upload field {name}"
                )));
            }
        }
    }
    let path = path.ok_or_else(|| ApiError::invalid("binary path is required"))?;
    validate_public_path(&path)?;
    let expected_content_hash = validate_sha256(
        expected_content_hash
            .as_deref()
            .ok_or_else(|| ApiError::invalid("expected_content_hash is required"))?,
    )?;
    let bytes = bytes.ok_or_else(|| ApiError::invalid("binary file is required"))?;
    let media_type = media_type
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            mime_guess::from_path(&path)
                .first_or_octet_stream()
                .essence_str()
                .to_owned()
        });
    let portable_companion = prepare_portable_binary_companion(
        state.config.evaluation_api_enabled,
        &path,
        description.as_deref(),
        portable_companion_format.as_deref(),
        portable_companion_path.as_deref(),
        portable_companion_sha256.as_deref(),
        portable_companion_mtime_ns,
        portable_companion_mode,
    )?;
    let stored = state
        .object_store
        .put_user_blob(auth.user_id, Some(&media_type), bytes)
        .await?;
    if validate_sha256(&stored.sha256)? != expected_content_hash {
        return Err(ApiError::conflict(
            "content_hash_mismatch",
            "uploaded binary bytes changed after the caller calculated their hash",
            json!({"path": path}),
        ));
    }
    let portable_metadata = (mtime_ns.is_some() || mode.is_some()).then(|| {
        json!({
            "modified_unix_ns": mtime_ns,
            "mode": mode
        })
    });
    let receipt = commit_binary_with_companion(
        &state,
        &auth,
        &path,
        &media_type,
        &stored.sha256,
        u64::try_from(stored.size_bytes).unwrap_or(u64::MAX),
        &stored.object_key,
        stored.object_version_id.as_deref(),
        description.as_deref(),
        provenance.as_deref(),
        limitations.as_deref(),
        portable_metadata,
        expected_version,
        portable_companion,
        None,
    )
    .await?;
    let mut envelope = WorkspaceEnvelope::complete(receipt.clone());
    envelope.status = if receipt.get("no_op") == Some(&Value::Bool(true)) {
        ResponseStatus::NoOp
    } else {
        ResponseStatus::Committed
    };
    envelope.corpus_revision = receipt
        .get("workspace_generation")
        .and_then(Value::as_i64)
        .map(|generation| format!("generation:{generation}"));
    if receipt.get("no_op") != Some(&Value::Bool(true)) {
        record_product_activity(
            &state,
            &auth,
            ProductActivityOperation::BinaryUpload,
            u64::try_from(stored.size_bytes).unwrap_or(u64::MAX),
        );
    }
    Ok(Json(envelope))
}

pub async fn upload_binary_stream(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    grant: Option<Extension<crate::binary_upload::UploadGrant>>,
    Query(mut query): Query<StreamingBinaryQuery>,
    headers: axum::http::HeaderMap,
    body: Body,
) -> ApiResult<(StatusCode, Json<WorkspaceEnvelope<Value>>)> {
    let grant = grant.as_ref().map(|Extension(grant)| grant);
    if let Some(grant) = grant {
        // Destination and bounds come only from the signed permission, never
        // caller-supplied query parameters on the PUT.
        query = StreamingBinaryQuery {
            path: grant.request.path.clone(),
            media_type: Some(grant.request.media_type.clone()),
            expected_content_hash: grant.request.sha256.clone().unwrap_or_default(),
            expected_version: grant.request.expected_version,
            mtime_ns: None,
            mode: None,
            provenance: None,
        };
        if headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            != Some(grant.request.media_type.as_str())
        {
            return Err(ApiError::invalid(
                "Content-Type must match the upload permission",
            ));
        }
        let mut tx = state.begin_read(&auth).await?;
        crate::binary_upload::reject_completed(&mut tx, grant).await?;
    } else {
        auth.require(Capability::Stage)?;
    }
    validate_public_path(&query.path)?;
    let expected_content_hash = if grant.is_some() && query.expected_content_hash.is_empty() {
        None
    } else {
        Some(validate_sha256(&query.expected_content_hash)?)
    };
    let media_type = query
        .media_type
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            mime_guess::from_path(&query.path)
                .first_or_octet_stream()
                .essence_str()
                .to_owned()
        });
    if media_type.len() > 255 {
        return Err(ApiError::invalid("media_type is limited to 255 characters"));
    }
    if query.mode.is_some_and(|value| value > 0o777) {
        return Err(ApiError::invalid(
            "mode may contain only portable permission bits",
        ));
    }
    let temporary_path =
        std::env::temp_dir().join(format!("brunn-binary-upload-{}", Uuid::now_v7()));
    let transfer = async {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
            .await
            .map_err(|error| {
                ApiError::Internal(format!("could not create upload buffer: {error}"))
            })?;
        let mut stream = body.into_data_stream();
        let mut size_bytes = 0_u64;
        let mut digest = Sha256::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| {
                ApiError::invalid(format!("binary upload stream failed: {error}"))
            })?;
            size_bytes = size_bytes
                .checked_add(u64::try_from(chunk.len()).unwrap_or(u64::MAX))
                .ok_or_else(|| ApiError::invalid("binary upload size overflow"))?;
            if size_bytes > MAX_STREAMED_BINARY_BYTES {
                return Err(ApiError::public(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "binary_too_large",
                    "workspace binary uploads are limited to 4 GiB",
                ));
            }
            if grant.is_some_and(|grant| size_bytes > grant.request.size_bytes) {
                return Err(ApiError::invalid("uploaded size does not match size_bytes"));
            }
            if grant.is_some() && expected_content_hash.is_some() {
                digest.update(&chunk);
            }
            file.write_all(&chunk).await.map_err(|error| {
                ApiError::Internal(format!("could not buffer binary upload: {error}"))
            })?;
        }
        file.flush().await.map_err(|error| {
            ApiError::Internal(format!("could not flush binary upload: {error}"))
        })?;
        drop(file);
        if grant.is_some_and(|grant| size_bytes != grant.request.size_bytes) {
            return Err(ApiError::invalid("uploaded size does not match size_bytes"));
        }
        if grant.is_some()
            && expected_content_hash
                .as_ref()
                .is_some_and(|expected| *expected != hex::encode(digest.finalize()))
        {
            return Err(ApiError::invalid("uploaded SHA-256 does not match sha256"));
        }
        state
            .object_store
            .put_user_file_blob(auth.user_id, &media_type, &temporary_path)
            .await
    }
    .await;
    let _ = tokio::fs::remove_file(&temporary_path).await;
    let stored = transfer?;
    if expected_content_hash
        .as_ref()
        .is_some_and(|expected| stored.sha256.trim_start_matches("sha256:") != expected)
    {
        return Err(ApiError::conflict(
            "content_hash_mismatch",
            "uploaded binary bytes changed after the caller calculated their hash",
            json!({"path": query.path}),
        ));
    }
    let portable_metadata = json!({
        "modified_unix_ns": query.mtime_ns,
        "mode": query.mode
    });
    let receipt = commit_binary_with_companion(
        &state,
        &auth,
        &query.path,
        &media_type,
        &stored.sha256,
        stored.size_bytes,
        &stored.object_key,
        stored.object_version_id.as_deref(),
        None,
        query.provenance.as_deref(),
        None,
        Some(portable_metadata),
        query.expected_version,
        None,
        grant,
    )
    .await?;
    let mut envelope = WorkspaceEnvelope::complete(receipt.clone());
    envelope.status = if receipt.get("no_op") == Some(&Value::Bool(true)) {
        ResponseStatus::NoOp
    } else {
        ResponseStatus::Committed
    };
    envelope.corpus_revision = receipt
        .get("workspace_generation")
        .and_then(Value::as_i64)
        .map(|generation| format!("generation:{generation}"));
    if receipt.get("no_op") != Some(&Value::Bool(true)) {
        record_product_activity(
            &state,
            &auth,
            ProductActivityOperation::BinaryUpload,
            stored.size_bytes,
        );
    }
    Ok((
        if grant.is_some() {
            StatusCode::CREATED
        } else {
            StatusCode::OK
        },
        Json(envelope),
    ))
}

pub async fn fetch_binary(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(entry_ref): Path<String>,
    Query(query): Query<BinaryVersionQuery>,
) -> ApiResult<Response<Body>> {
    auth.require(Capability::Read)?;
    let entry = resolve_entry_version(&state, &auth, None, Some(&entry_ref), query.version).await?;
    if entry.kind != "binary" {
        return Err(ApiError::invalid("the requested entry is not binary"));
    }
    let key = entry
        .object_key
        .as_deref()
        .ok_or_else(|| ApiError::Internal("binary entry has no object key".to_owned()))?;
    let stream = state
        .object_store
        .get_stream_version(key, entry.object_version_id.as_deref())
        .await?;
    if stream.content_length != Some(entry.size_bytes) {
        return Err(ApiError::conflict(
            "content_size_mismatch",
            "stored binary size no longer matches its immutable metadata",
            json!({"entry_ref": format!("entry:{}", entry.id)}),
        ));
    }
    let mut response = Response::new(Body::from_stream(ReaderStream::new(
        stream.body.into_async_read(),
    )));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(&entry.media_type)
            .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream")),
    );
    response.headers_mut().insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&entry.size_bytes.to_string())
            .map_err(|error| ApiError::Internal(error.to_string()))?,
    );
    response.headers_mut().insert(
        "x-brunn-state-sha256",
        HeaderValue::from_str(&format!("sha256:{}", entry.content_sha256))
            .map_err(|error| ApiError::Internal(error.to_string()))?,
    );
    response.headers_mut().insert(
        "x-brunn-state-asset-ref",
        HeaderValue::from_str(&format!("entry:{}", entry.id))
            .map_err(|error| ApiError::Internal(error.to_string()))?,
    );
    response.headers_mut().insert(
        "x-brunn-state-asset-version",
        HeaderValue::from_str(&entry.version.to_string())
            .map_err(|error| ApiError::Internal(error.to_string()))?,
    );
    response.headers_mut().insert(
        "x-brunn-state-integrity",
        HeaderValue::from_static("client-verify-sha256"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    response.headers_mut().insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    record_entry_usage(&state, &auth, &[entry.id], UsageOperation::Read);
    record_product_activity(
        &state,
        &auth,
        ProductActivityOperation::BinaryFetch,
        u64::try_from(entry.size_bytes).unwrap_or(0),
    );
    Ok(response)
}

pub async fn binary_metadata(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(entry_ref): Path<String>,
    Query(query): Query<BinaryVersionQuery>,
) -> ApiResult<Json<WorkspaceEnvelope<Value>>> {
    auth.require(Capability::Read)?;
    let entry = resolve_entry_version(&state, &auth, None, Some(&entry_ref), query.version).await?;
    if entry.kind != "binary" {
        return Err(ApiError::invalid("the requested entry is not binary"));
    }
    let metadata = load_binary_description_metadata(&state, &auth, &entry).await?;
    record_entry_usage(&state, &auth, &[entry.id], UsageOperation::Read);
    Ok(Json(WorkspaceEnvelope::complete(json!({
        "entry_ref": format!("entry:{}", entry.id),
        "path": entry.path,
        "title": entry.title,
        "media_type": entry.media_type,
        "version": entry.version,
        "content_hash": format!("sha256:{}", entry.content_sha256),
        "size_bytes": entry.size_bytes,
        "metadata": metadata,
        "updated_at": entry.updated_at
    }))))
}

async fn load_binary_description_metadata(
    state: &AppState,
    auth: &AuthContext,
    entry: &EntryRow,
) -> ApiResult<Value> {
    let Some(companion_path) = entry.metadata.get("companion_path").and_then(Value::as_str) else {
        return Ok(entry.metadata.clone());
    };
    let mut tx = state.begin_read(auth).await?;
    let companion_metadata = sqlx::query_scalar::<_, Value>(
        r#"
        SELECT version.metadata
        FROM brunn.entries AS entry
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
         AND version.version=entry.current_version
        WHERE entry.user_id=$1
          AND lower(normalize(entry.path, NFC))=$2
          AND entry.kind='markdown'
          AND entry.deleted_at IS NULL
        "#,
    )
    .bind(auth.user_id.0)
    .bind(portable_path_key(companion_path))
    .fetch_optional(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(merge_binary_description_metadata(
        entry.metadata.clone(),
        companion_metadata,
    ))
}

fn merge_binary_description_metadata(mut binary: Value, companion: Option<Value>) -> Value {
    let Some(companion) = companion else {
        return binary;
    };
    if let Value::Object(values) = &mut binary {
        if let Some(status) = companion.get("description_status") {
            values.insert("description_status".to_owned(), status.clone());
        }
        values.insert("description".to_owned(), companion);
    }
    binary
}

async fn read_small_field(field: Field<'_>, name: &str, limit: usize) -> ApiResult<String> {
    let bytes = field
        .bytes()
        .await
        .map_err(|error| ApiError::invalid(format!("invalid {name}: {error}")))?;
    if bytes.len() > limit {
        return Err(ApiError::public(
            StatusCode::PAYLOAD_TOO_LARGE,
            "binary_metadata_too_large",
            format!("{name} is limited to {limit} bytes"),
        ));
    }
    String::from_utf8(bytes.to_vec())
        .map_err(|_| ApiError::invalid(format!("{name} must be UTF-8")))
}

#[allow(clippy::too_many_arguments)]
fn prepare_portable_binary_companion(
    evaluation_api_enabled: bool,
    binary_path: &str,
    description: Option<&str>,
    format: Option<&str>,
    path: Option<&str>,
    content_sha256: Option<&str>,
    modified_unix_ns: Option<i64>,
    mode: Option<u32>,
) -> ApiResult<Option<PortableBinaryCompanion>> {
    let requested = format.is_some()
        || path.is_some()
        || content_sha256.is_some()
        || modified_unix_ns.is_some()
        || mode.is_some();
    if !requested {
        return Ok(None);
    }
    if !evaluation_api_enabled {
        return Err(ApiError::invalid(
            "exact portable binary companions require an evaluation-only stack",
        ));
    }
    if format != Some(TIER_A_PORTABLE_COMPANION_FORMAT) {
        return Err(ApiError::invalid(
            "portable_companion_format is missing or unsupported",
        ));
    }
    let path = path.ok_or_else(|| ApiError::invalid("portable_companion_path is required"))?;
    validate_public_path(path)?;
    if portable_path_key(path) == portable_path_key(binary_path) {
        return Err(ApiError::invalid(
            "portable binary and companion paths must be different",
        ));
    }
    let content = description
        .ok_or_else(|| ApiError::invalid("exact portable companion content is required"))?
        .to_owned();
    let expected = validate_sha256(
        content_sha256.ok_or_else(|| ApiError::invalid("portable_companion_sha256 is required"))?,
    )?;
    let actual = hex::encode(Sha256::digest(content.as_bytes()));
    if actual != expected {
        return Err(ApiError::conflict(
            "portable_companion_hash_mismatch",
            "portable companion bytes do not match their pinned SHA-256",
            json!({"path": path}),
        ));
    }
    if mode.is_some_and(|value| value > 0o777) {
        return Err(ApiError::invalid(
            "portable_companion_mode may contain only portable permission bits",
        ));
    }
    Ok(Some(PortableBinaryCompanion {
        path: path.to_owned(),
        content,
        content_sha256: actual,
        modified_unix_ns,
        mode,
    }))
}

#[allow(clippy::too_many_arguments)]
async fn commit_binary_with_companion(
    state: &AppState,
    auth: &AuthContext,
    path: &str,
    media_type: &str,
    stored_hash: &str,
    size_bytes: u64,
    object_key: &str,
    object_version_id: Option<&str>,
    description: Option<&str>,
    provenance: Option<&str>,
    limitations: Option<&str>,
    portable_metadata: Option<Value>,
    expected_version: Option<i64>,
    portable_companion: Option<PortableBinaryCompanion>,
    upload_grant: Option<&crate::binary_upload::UploadGrant>,
) -> ApiResult<Value> {
    let object_version_id = object_version_id.ok_or_else(|| {
        ApiError::Internal("versioned object upload returned no exact version ID".to_owned())
    })?;
    let content_sha256 = stored_hash.trim_start_matches("sha256:").to_owned();
    let companion_key = hex::encode(Sha256::digest(portable_path_key(path).as_bytes()));
    let companion_path = portable_companion.as_ref().map_or_else(
        || format!(".brunn/binaries/{}.md", &companion_key[..32]),
        |companion| companion.path.clone(),
    );
    validate_path(&companion_path)?;
    let title = std::path::Path::new(path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(path);
    let supplied_description = description.map(str::trim).filter(|value| !value.is_empty());
    let description_text = supplied_description
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| {
            format!(
                "Binary file `{path}` ({media_type}, {size_bytes} bytes). \
                 Content-specific description is pending background inspection."
            )
        });
    let provenance_text = provenance
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("Uploaded through the Brunn workspace binary API.");
    let limitations_text = limitations
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(
            "The immutable binary bytes are authoritative. This Markdown companion is for retrieval and may be incomplete.",
        );
    let companion_content = portable_companion.as_ref().map_or_else(
        || {
            Ok::<_, ApiError>(format!(
                "---\nbrunn_kind: binary_description\nbinary_path: {}\n\
                 content_hash: {}\nmedia_type: {}\nsize_bytes: {}\n\
                 description_status: {}\n---\n\n# Binary: {}\n\n\
                 ## Description\n\n{}\n\n## Provenance\n\n{}\n\n\
                 ## Limitations\n\n{}\n",
                serde_json::to_string(path)?,
                serde_json::to_string(&format!("sha256:{content_sha256}"))?,
                serde_json::to_string(media_type)?,
                size_bytes,
                if supplied_description.is_some() {
                    "provided"
                } else {
                    "pending"
                },
                title,
                description_text,
                provenance_text,
                limitations_text
            ))
        },
        |companion| Ok(companion.content.clone()),
    )?;
    let description_status = if portable_companion.is_some() {
        "byte_copied"
    } else if supplied_description.is_some() {
        "provided"
    } else {
        "pending"
    };
    let companion_portable_metadata = portable_companion.as_ref().map_or_else(
        || portable_metadata.clone(),
        |companion| {
            Some(json!({
                "modified_unix_ns": companion.modified_unix_ns,
                "mode": companion.mode
            }))
        },
    );
    let companion = prepare_markdown(
        state,
        WriteRequest {
            path: companion_path.clone(),
            content: companion_content,
            media_type: markdown_media_type(),
            expected_version: None,
            idempotency_key: None,
            metadata: json!({
                "kind": "binary_description",
                "binary_path": path,
                "content_hash": format!("sha256:{content_sha256}"),
                "portable": companion_portable_metadata,
                "description_status": description_status,
                "_brunn_import": portable_companion.as_ref().map(|companion| json!({
                    "format": WORKSPACE_IMPORT_FORMAT,
                    "portable_companion_format": TIER_A_PORTABLE_COMPANION_FORMAT,
                    "content_sha256": format!("sha256:{}", companion.content_sha256)
                }))
            }),
        },
    )
    .await?;

    let mut tx = state.begin_write(auth).await?;
    let mut lock_paths = [portable_path_key(path), portable_path_key(&companion_path)];
    lock_paths.sort();
    for lock_path in lock_paths {
        require_local_publish_lock(
            &mut tx,
            format!("simple-entry:{}:{lock_path}", auth.user_id.0),
            state.config.read_path_roundtrip_v1,
        )
        .await?;
    }

    if let Some(grant) = upload_grant {
        crate::binary_upload::reject_completed(&mut tx, grant).await?;
    }

    let existing = sqlx::query(
        r#"
        SELECT entry.id,entry.kind,entry.media_type,entry.current_version,entry.deleted_at,
               version.content_sha256,version.metadata
        FROM brunn.entries AS entry
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
         AND version.version=entry.current_version
        WHERE entry.user_id=$1
          AND lower(normalize(entry.path, NFC))=$2
        FOR UPDATE OF entry
        "#,
    )
    .bind(auth.user_id.0)
    .bind(portable_path_key(path))
    .fetch_optional(&mut *tx)
    .await?;
    if existing
        .as_ref()
        .is_some_and(|row| row.get::<String, _>("kind") != "binary")
    {
        return Err(ApiError::conflict(
            "entry_kind_conflict",
            "a Markdown entry already uses this binary path",
            json!({"path": path}),
        ));
    }
    if let Some(grant) = upload_grant {
        crate::binary_upload::check_target(
            path,
            grant.request.expected_version.unwrap_or(0),
            grant.entry_id,
            existing.as_ref(),
        )?;
    }
    let binary_no_op = upload_grant.is_none()
        && existing
            .as_ref()
            .is_some_and(|row| row.get::<String, _>("content_sha256") == content_sha256);
    if !binary_no_op && let Some(expected) = expected_version {
        let actual = existing
            .as_ref()
            .map(|row| row.get::<i64, _>("current_version"))
            .unwrap_or(0);
        let restoring_deleted = existing.as_ref().is_some_and(|row| {
            row.get::<Option<DateTime<Utc>>, _>("deleted_at").is_some() && expected == 0
        });
        if actual != expected && !restoring_deleted {
            return Err(ApiError::conflict(
                "entry_version_conflict",
                "the entry changed since it was read",
                json!({
                    "path": path,
                    "expected_version": expected,
                    "actual_version": actual
                }),
            ));
        }
    }
    let mut binary_annotation_changed = false;
    let (binary_entry_id, binary_version, binary_generation) = if binary_no_op {
        let row = existing.as_ref().expect("checked above");
        let was_deleted = row.get::<Option<DateTime<Utc>>, _>("deleted_at").is_some();
        let mut metadata = row.get::<Value, _>("metadata");
        let previous_metadata = metadata.clone();
        if let Value::Object(values) = &mut metadata {
            values.insert(
                "companion_path".to_owned(),
                Value::String(companion_path.clone()),
            );
            if let Some(portable) = &portable_metadata {
                values.insert("portable".to_owned(), portable.clone());
            }
            values.insert(
                "description_status".to_owned(),
                Value::String(description_status.to_owned()),
            );
            if provenance.is_some() {
                values.insert(
                    "provenance".to_owned(),
                    Value::String(provenance_text.to_owned()),
                );
            }
            if limitations.is_some() {
                values.insert(
                    "limitations".to_owned(),
                    Value::String(limitations_text.to_owned()),
                );
            }
        }
        binary_annotation_changed = was_deleted
            || metadata != previous_metadata
            || row.get::<String, _>("media_type") != media_type;
        let generation = if binary_annotation_changed {
            sqlx::query(
                r#"
                UPDATE brunn.entry_versions
                SET metadata=$4
                WHERE user_id=$1 AND entry_id=$2 AND version=$3
                "#,
            )
            .bind(auth.user_id.0)
            .bind(row.get::<Uuid, _>("id"))
            .bind(row.get::<i64, _>("current_version"))
            .bind(&metadata)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                r#"
                UPDATE brunn.entries
                SET path=$3,media_type=$4,deleted_at=NULL,
                    updated_at=clock_timestamp()
                WHERE user_id=$1 AND id=$2
                "#,
            )
            .bind(auth.user_id.0)
            .bind(row.get::<Uuid, _>("id"))
            .bind(path)
            .bind(media_type)
            .execute(&mut *tx)
            .await?;
            Some(
                sqlx::query_scalar::<_, i64>(
                    r#"
                    INSERT INTO brunn.workspace_changes (
                      user_id,entry_id,entry_version,operation,path,content_sha256
                    ) VALUES ($1,$2,$3,'update',$4,$5)
                    RETURNING generation
                    "#,
                )
                .bind(auth.user_id.0)
                .bind(row.get::<Uuid, _>("id"))
                .bind(row.get::<i64, _>("current_version"))
                .bind(path)
                .bind(&content_sha256)
                .fetch_one(&mut *tx)
                .await?,
            )
        } else {
            None
        };
        (
            row.get::<Uuid, _>("id"),
            row.get::<i64, _>("current_version"),
            generation,
        )
    } else {
        let (entry_id, version, operation) = match existing {
            Some(row) => (
                row.get::<Uuid, _>("id"),
                row.get::<i64, _>("current_version") + 1,
                "update",
            ),
            None => (
                upload_grant.map_or_else(Uuid::now_v7, |grant| grant.entry_id),
                1_i64,
                "create",
            ),
        };
        if operation == "create" {
            sqlx::query(
                r#"
                INSERT INTO brunn.entries (
                  id,user_id,path,title,kind,media_type,current_version
                ) VALUES ($1,$2,$3,$4,'binary',$5,0)
                "#,
            )
            .bind(entry_id)
            .bind(auth.user_id.0)
            .bind(path)
            .bind(title)
            .bind(media_type)
            .execute(&mut *tx)
            .await?;
        }
        let version_id = upload_grant.map_or_else(Uuid::now_v7, |grant| grant.version_id);
        sqlx::query(
            r#"
            INSERT INTO brunn.entry_versions (
              id,user_id,entry_id,version,content_sha256,object_key,
              object_version_id,size_bytes,metadata,created_by_credential_id
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            "#,
        )
        .bind(version_id)
        .bind(auth.user_id.0)
        .bind(entry_id)
        .bind(version)
        .bind(&content_sha256)
        .bind(object_key)
        .bind(object_version_id)
        .bind(i64::try_from(size_bytes).unwrap_or(i64::MAX))
        .bind(json!({
            "companion_path": companion_path,
            "description_status": description_status,
            "portable": portable_metadata,
            "provenance": provenance_text,
            "limitations": limitations_text
        }))
        .bind(auth.credential_id.0)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            r#"
            UPDATE brunn.entries
            SET path=$3,title=$4,media_type=$5,current_version=$6,
                updated_at=clock_timestamp(),deleted_at=NULL
            WHERE user_id=$1 AND id=$2
            "#,
        )
        .bind(auth.user_id.0)
        .bind(entry_id)
        .bind(path)
        .bind(title)
        .bind(media_type)
        .bind(version)
        .execute(&mut *tx)
        .await?;
        let generation = sqlx::query_scalar::<_, i64>(
            r#"
            INSERT INTO brunn.workspace_changes (
              user_id,entry_id,entry_version,operation,path,content_sha256
            ) VALUES ($1,$2,$3,$4,$5,$6)
            RETURNING generation
            "#,
        )
        .bind(auth.user_id.0)
        .bind(entry_id)
        .bind(version)
        .bind(operation)
        .bind(path)
        .bind(&content_sha256)
        .fetch_one(&mut *tx)
        .await?;
        (entry_id, version, Some(generation))
    };

    let retained_companion =
        if binary_no_op && supplied_description.is_none() && portable_companion.is_none() {
            sqlx::query(
                r#"
            SELECT entry.id,entry.current_version,version.id AS version_id
            FROM brunn.entries AS entry
            JOIN brunn.entry_versions AS version
              ON version.user_id=entry.user_id
             AND version.entry_id=entry.id
             AND version.version=entry.current_version
            WHERE entry.user_id=$1
              AND lower(normalize(entry.path, NFC))=$2
              AND entry.kind='markdown'
              AND entry.deleted_at IS NULL
            "#,
            )
            .bind(auth.user_id.0)
            .bind(portable_path_key(&companion_path))
            .fetch_optional(&mut *tx)
            .await?
            .map(|row| MarkdownUpsertResult {
                entry_id: row.get("id"),
                version: row.get("current_version"),
                version_id: Some(row.get("version_id")),
                generation: None,
                no_op: true,
                metadata_only: false,
            })
        } else {
            None
        };
    let should_queue_description = upload_grant.is_none()
        && portable_companion.is_none()
        && supplied_description.is_none()
        && (!binary_no_op || retained_companion.is_none());
    let companion_result = match retained_companion {
        Some(result) => result,
        None => {
            upsert_markdown_in_tx(
                &mut tx,
                auth.user_id.0,
                Some(auth.credential_id.0),
                companion,
            )
            .await?
        }
    };
    if should_queue_description {
        sqlx::query(
            r#"
            INSERT INTO brunn.jobs (user_id,kind,payload)
            VALUES ($1,'describe_binary',$2)
            "#,
        )
        .bind(auth.user_id.0)
        .bind(json!({
            "entry_id": binary_entry_id,
            "version": binary_version,
            "companion_path": companion_path
        }))
        .execute(&mut *tx)
        .await?;
    }
    let generation = match (binary_generation, companion_result.generation) {
        (Some(left), Some(right)) => left.max(right),
        (Some(value), None) | (None, Some(value)) => value,
        (None, None) => max_generation_in_tx(&mut tx, auth.user_id.0).await?,
    };
    tx.commit().await?;
    Ok(json!({
        "entry_ref": format!("entry:{binary_entry_id}"),
        "path": path,
        "version": binary_version,
        "content_hash": format!("sha256:{content_sha256}"),
        "size_bytes": size_bytes,
        "media_type": media_type,
        "companion": {
            "entry_ref": format!("entry:{}", companion_result.entry_id),
            "path": companion_path,
            "version": companion_result.version
        },
        "workspace_generation": generation,
        "description_status": description_status,
        "metadata_only": binary_no_op && binary_annotation_changed,
        "no_op": binary_no_op && !binary_annotation_changed && companion_result.no_op
    }))
}

pub async fn import_evaluation(
    State(state): State<AppState>,
    Extension(caller): Extension<AuthContext>,
    Json(request): Json<EvalImportRequest>,
) -> ApiResult<Json<Value>> {
    if !state.config.evaluation_api_enabled {
        return Err(ApiError::not_found(
            "route_not_found",
            "/v1/workspace/admin/eval/import",
        ));
    }
    caller.require(Capability::Admin)?;
    validate_eval_import(&request)?;
    let access_mode = request.access_mode.as_str();
    let capabilities = if access_mode == "read_only" {
        vec!["open", "query", "read", "status"]
    } else {
        vec![
            "open",
            "query",
            "read",
            "status",
            "checkpoint",
            "save",
            "stage",
            "correct",
            "delete",
            "dream",
        ]
    };
    let evaluation_batch = evaluation_batch(&request)?;
    let evaluation_identity = format!(
        "{}\0{}\0{}\0{}\0{}",
        if evaluation_batch.is_some() {
            "simple-workspace-eval-batch-v1"
        } else {
            "simple-workspace-eval-v1"
        },
        request.run_id,
        request.case_id,
        request.authorization_scope,
        request.idempotency_key
    );
    let external_ref = format!(
        "eval-user:{}",
        hex::encode(Sha256::digest(evaluation_identity.as_bytes()))
    );
    let display_name = format!(
        "Brunn evaluation: {}",
        request.case_id.chars().take(160).collect::<String>()
    );
    let token = derive_eval_token(
        &state.config.continuation_secret,
        &external_ref,
        &request.idempotency_key,
    )?;
    let prepared = prepare_bulk_documents(&state, &request.documents).await?;
    let deltas = prepare_bulk_documents(&state, &request.delta_documents).await?;
    let mut tx = state.rw_pool.begin().await?;
    require_local_publish_lock(
        &mut tx,
        format!("simple-eval:{external_ref}"),
        state.config.read_path_roundtrip_v1,
    )
    .await?;
    set_context(&mut tx, &caller).await?;
    let provisioning_capabilities = vec![
        "open",
        "query",
        "read",
        "status",
        "checkpoint",
        "save",
        "stage",
        "correct",
        "delete",
        "dream",
        "task.read",
        "task.write",
    ];
    let row = sqlx::query(
        r#"
        SELECT *
        FROM brunn_auth.bootstrap_evaluation_user($1,$2,$3,$4,$5)
        "#,
    )
    .bind(&external_ref)
    .bind(&display_name)
    .bind("Evaluation provisioning")
    .bind(hash_token(&token))
    .bind(&provisioning_capabilities)
    .fetch_one(&mut *tx)
    .await?;
    let user_id: Uuid = row.get("user_id");
    let credential_id: Uuid = row.get("credential_id");
    let scope_id: Uuid = row.get("scope_id");
    let provisioning_auth = AuthContext {
        credential_id: CredentialId(credential_id),
        user_id: UserId(user_id),
        capabilities: provisioning_capabilities
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        scope_refs: vec!["scope:root".to_owned()],
        read_only: false,
    };
    set_context(&mut tx, &provisioning_auth).await?;
    let existing = sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS(SELECT 1 FROM brunn.entries WHERE user_id=$1)",
    )
    .bind(user_id)
    .fetch_one(&mut *tx)
    .await?;
    let batch_index = evaluation_batch.map(|(index, _)| index);
    if existing && batch_index.is_none_or(|index| index == 0) {
        return Err(ApiError::conflict(
            "eval_token_unavailable_on_replay",
            "the evaluation import already committed; use a new run ID",
            json!({"run_id": request.run_id, "case_id": request.case_id}),
        ));
    }
    if !existing && batch_index.is_some_and(|index| index > 0) {
        return Err(ApiError::conflict(
            "eval_batch_out_of_order",
            "evaluation import batches must begin with batch zero",
            json!({"batch_index": batch_index}),
        ));
    }
    sqlx::query("SET CONSTRAINTS ALL DEFERRED")
        .execute(&mut *tx)
        .await?;
    insert_bulk_documents(&mut tx, user_id, credential_id, &prepared, "create").await?;
    let base_generation = max_generation_in_tx(&mut tx, user_id).await?;
    let checkpoint_id = if let Some(seed) = &request.seed_checkpoint {
        let checkpoint_id = Uuid::now_v7();
        let checkpoint_path = format!(".brunn/checkpoints/{checkpoint_id}.md");
        let source_rows = sqlx::query(
            r#"
            WITH requested AS (
              SELECT path,position
              FROM unnest($2::text[]) WITH ORDINALITY AS source(path,position)
            )
            SELECT entry.id,entry.path,entry.current_version,version.content_sha256
            FROM requested
            JOIN brunn.entries AS entry
              ON entry.user_id=$1
             AND lower(normalize(entry.path,NFC))=lower(normalize(requested.path,NFC))
             AND entry.deleted_at IS NULL
            JOIN brunn.entry_versions AS version
              ON version.user_id=entry.user_id
             AND version.entry_id=entry.id
             AND version.version=entry.current_version
            ORDER BY requested.position
            "#,
        )
        .bind(user_id)
        .bind(&seed.source_refs)
        .fetch_all(&mut *tx)
        .await?;
        let source_entries = source_rows
            .into_iter()
            .map(|row| {
                json!({
                    "entry_ref": format!("entry:{}", row.get::<Uuid, _>("id")),
                    "path": row.get::<String, _>("path"),
                    "version": row.get::<i64, _>("current_version"),
                    "content_hash": format!("sha256:{}", row.get::<String, _>("content_sha256"))
                })
            })
            .collect::<Vec<_>>();
        if source_entries.len() != seed.source_refs.len() {
            return Err(ApiError::invalid(
                "every seed checkpoint source_ref must identify a base document",
            ));
        }
        let checkpoint_content = render_seed_checkpoint(
            checkpoint_id,
            base_generation,
            &seed.state,
            &seed.source_refs,
        );
        let mut checkpoint_document = prepare_one_bulk_document(
            &state,
            checkpoint_path,
            checkpoint_content,
            "text/markdown".to_owned(),
        )
        .await?;
        checkpoint_document.entry_id = checkpoint_id;
        checkpoint_document.metadata = json!({
            "kind": "checkpoint",
            "checkpoint_ref": format!("checkpoint:{checkpoint_id}"),
            "workspace_generation": base_generation,
            "source_refs": seed.source_refs,
            "source_entries": source_entries
        });
        insert_bulk_documents(
            &mut tx,
            user_id,
            credential_id,
            &[checkpoint_document],
            "create",
        )
        .await?;
        Some(format!("checkpoint:{checkpoint_id}"))
    } else {
        None
    };
    apply_bulk_deltas(&mut tx, user_id, credential_id, &deltas).await?;
    let generation = max_generation_in_tx(&mut tx, user_id).await?;
    let _ = sqlx::query(
        r#"
        SELECT *
        FROM brunn_auth.bootstrap_evaluation_user($1,$2,$3,$4,$5)
        "#,
    )
    .bind(&external_ref)
    .bind(&display_name)
    .bind(if access_mode == "read_only" {
        "Evaluation read-only"
    } else {
        "Evaluation read/write"
    })
    .bind(hash_token(&token))
    .bind(&capabilities)
    .fetch_one(&mut *tx)
    .await?;
    tx.commit().await?;
    state.workspace_features.invalidate(user_id).await;
    metrics::histogram!("simple.import.feature_declarations").record(
        prepared
            .iter()
            .chain(&deltas)
            .filter(|document| {
                !document.frontmatter.supersedes.is_empty()
                    || document.frontmatter.kind.as_deref() == Some("intention")
            })
            .count() as f64,
    );
    let chunk_count = prepared
        .iter()
        .chain(&deltas)
        .map(|document| document.chunks.len())
        .sum::<usize>();
    let embedded_count = prepared
        .iter()
        .chain(&deltas)
        .flat_map(|document| &document.embeddings)
        .filter(|embedding| embedding.is_some())
        .count();
    let semantic_ready = embedded_count == chunk_count;
    let import_id = format!("simple-import:{user_id}");
    let status_url = format!("/v1/workspace/admin/eval/imports/{import_id}");
    Ok(Json(json!({
        "status": if semantic_ready { "ready" } else { "indexing" },
        "ready_for_evaluation": semantic_ready,
        "import_id": import_id,
        "status_url": status_url,
        "authorization_scope": request.authorization_scope,
        "requested_authorization_scope": request.authorization_scope,
        "display_scope": request.display_scope,
        "access_mode": access_mode,
        "user_id": format!("user:{user_id}"),
        "scope_id": format!("scope:{scope_id}"),
        "credential_id": format!("credential:{credential_id}"),
        "base_corpus_revision": format!("generation:{base_generation}"),
        "corpus_revision": format!("generation:{generation}"),
        "checkpoint_id": checkpoint_id,
        "seed_session_id": null,
        "index_status": {
            "exact": "ready",
            "lexical": "ready",
            "semantic": if semantic_ready { "ready" } else { "pending" }
        },
        "index_counts": {
            "documents": request.documents.len() + request.delta_documents.len(),
            "chunks": chunk_count,
            "embeddings": embedded_count
        },
        "embedding_provider": state.embedder.provider(),
        "embedding_model": state.embedder.model(),
        "embedding_degraded": state.embedder.is_degraded(),
        "documents": request.documents.len(),
        "delta_documents": request.delta_documents.len(),
        "batch_index": evaluation_batch.map(|(index, _)| index),
        "batch_count": evaluation_batch.map(|(_, count)| count),
        "replayed": false,
        "token_status": "issued_once",
        "credential_token": token
    })))
}

pub async fn evaluation_status(
    State(state): State<AppState>,
    Extension(caller): Extension<AuthContext>,
    Path(import_id): Path<String>,
) -> ApiResult<Json<Value>> {
    caller.require(Capability::Status)?;
    let user_id = import_id
        .strip_prefix("simple-import:")
        .and_then(|value| Uuid::parse_str(value).ok())
        .ok_or_else(|| ApiError::invalid("invalid simple evaluation import ID"))?;
    if caller.user_id.0 != user_id {
        return Err(ApiError::public(
            StatusCode::FORBIDDEN,
            "evaluation_scope_mismatch",
            "evaluation status can only be read by its scoped credential",
        ));
    }
    let mut tx = state.begin_read(&caller).await?;
    let row = sqlx::query(
        r#"
        SELECT
          count(*) FILTER (WHERE status IN ('queued','running')) AS pending_jobs,
          count(*) FILTER (WHERE status='failed') AS failed_jobs,
          brunn_auth.workspace_generation($1) AS generation
        FROM brunn.jobs
        WHERE user_id=$1 AND kind='embed_entry'
        "#,
    )
    .bind(user_id)
    .fetch_one(&mut *tx)
    .await?;
    let pending_jobs: i64 = row.get("pending_jobs");
    let failed_jobs: i64 = row.get("failed_jobs");
    let ready = pending_jobs == 0 && failed_jobs == 0;
    Ok(Json(json!({
        "status": if ready {
            "ready"
        } else if failed_jobs > 0 {
            "failed"
        } else {
            "indexing"
        },
        "ready_for_evaluation": ready,
        "import_id": import_id,
        "corpus_revision": format!("generation:{}", row.get::<i64, _>("generation")),
        "index_status": {
            "exact": "ready",
            "lexical": "ready",
            "semantic": if ready { "ready" } else if failed_jobs > 0 { "failed" } else { "pending" }
        },
        "index_counts": {
            "pending_jobs": pending_jobs,
            "failed_jobs": failed_jobs
        }
    })))
}

pub async fn cleanup_evaluation(
    State(state): State<AppState>,
    Extension(caller): Extension<AuthContext>,
    Path(import_id): Path<String>,
) -> ApiResult<Json<Value>> {
    caller.require(Capability::Delete)?;
    let user_id = import_id
        .strip_prefix("simple-import:")
        .and_then(|value| Uuid::parse_str(value).ok())
        .ok_or_else(|| ApiError::invalid("invalid simple evaluation import ID"))?;
    if caller.user_id.0 != user_id {
        return Err(ApiError::public(
            StatusCode::FORBIDDEN,
            "evaluation_scope_mismatch",
            "evaluation cleanup requires its own scoped credential",
        ));
    }
    let mut tx = state.begin_write(&caller).await?;
    let row = sqlx::query(
        r#"
        SELECT *
        FROM brunn_auth.cleanup_evaluation_user($1,$2)
        "#,
    )
    .bind(user_id)
    .bind(caller.credential_id.0)
    .fetch_one(&mut *tx)
    .await?;
    tx.commit().await?;
    state.workspace_features.invalidate(user_id).await;
    Ok(Json(json!({
        "status": "cleaned",
        "import_id": import_id,
        "entries_removed": row.get::<i64, _>("entries_removed"),
        "search_chunks_removed": row.get::<i64, _>("search_chunks_removed"),
        "jobs_removed": row.get::<i64, _>("jobs_removed"),
        "credentials_revoked": row.get::<i64, _>("credentials_revoked"),
        "revoked_at": row.get::<DateTime<Utc>, _>("revoked_at")
    })))
}

async fn current_generation(state: &AppState, auth: &AuthContext) -> ApiResult<i64> {
    let mut tx = state.begin_read(auth).await?;
    let generation =
        sqlx::query_scalar::<_, Option<i64>>("SELECT brunn_auth.workspace_generation($1)")
            .bind(auth.user_id.0)
            .fetch_one(&mut *tx)
            .await?
            .unwrap_or(0);
    tx.commit().await?;
    Ok(generation)
}

async fn feature_snapshot(
    state: &AppState,
    auth: &AuthContext,
    generation: i64,
) -> ApiResult<Option<Arc<WorkspaceFeatureSnapshot>>> {
    if !state.config.supersession_demotion && !state.config.intention_ledger {
        return Ok(None);
    }
    if let Some(snapshot) = state
        .workspace_features
        .get(auth.user_id.0, generation)
        .await
    {
        return Ok(Some(snapshot));
    }
    let mut tx = state.begin_read(auth).await?;
    let rows = sqlx::query(
        r#"
        SELECT entry.id,entry.path,entry.title,coalesce(version.content,'') AS content
        FROM brunn.entries AS entry
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
         AND version.version=entry.current_version
        WHERE entry.user_id=$1
          AND entry.deleted_at IS NULL
          AND entry.kind='markdown'
        ORDER BY entry.path
        "#,
    )
    .bind(auth.user_id.0)
    .fetch_all(&mut *tx)
    .await?;
    tx.commit().await?;
    let documents = rows
        .into_iter()
        .map(|row| WorkspaceFeatureDocument {
            entry_id: row.get("id"),
            path: row.get("path"),
            title: row.get("title"),
            content: row.get("content"),
        })
        .collect();
    Ok(Some(
        state
            .workspace_features
            .put(
                auth.user_id.0,
                WorkspaceFeatureSnapshot::build(generation, documents),
            )
            .await,
    ))
}

/// The three retrieval lanes always start together. Hybrid retrieval returns
/// as soon as exact+lexical finish, taking semantic evidence only when it is
/// already ready; an explicit semantic-only request waits for its bounded
/// semantic result.
async fn join_retrieval_lanes<E, L, S>(
    exact: E,
    lexical: L,
    semantic: S,
    semantic_required: bool,
) -> (E::Output, L::Output, Option<S::Output>)
where
    E: std::future::Future,
    L: std::future::Future,
    S: std::future::Future,
{
    if semantic_required {
        let (exact, lexical, semantic) = tokio::join!(exact, lexical, semantic);
        return (exact, lexical, Some(semantic));
    }

    let core = async { tokio::join!(exact, lexical) };
    tokio::pin!(core);
    tokio::pin!(semantic);
    tokio::select! {
        // Poll semantic first so a result that becomes ready in the same
        // scheduler turn as the core lanes is retained without extending the
        // response by even one additional wait.
        biased;
        semantic = &mut semantic => {
            let (exact, lexical) = core.await;
            (exact, lexical, Some(semantic))
        }
        (exact, lexical) = &mut core => {
            let semantic = tokio::select! {
                // Give a semantic future woken while the core branch was
                // being polled one final non-blocking chance to contribute.
                biased;
                semantic = &mut semantic => Some(semantic),
                _ = std::future::ready(()) => None,
            };
            (exact, lexical, semantic)
        },
    }
}

async fn search_one(
    state: &AppState,
    auth: &AuthContext,
    query: &str,
    limit: usize,
    requested_modes: &[String],
    requested_sort: Option<&str>,
    features: Option<&WorkspaceFeatureSnapshot>,
    request_semantic_embeddings: Option<(RequestSemanticEmbeddings, usize)>,
) -> ApiResult<(
    Vec<Candidate>,
    Vec<&'static str>,
    RetrievalTimings,
    Option<i64>,
)> {
    let total_started = Instant::now();
    let mut timings = RetrievalTimings::default();
    if query.trim().is_empty() {
        return Err(ApiError::invalid("search query is required"));
    }
    let sort = SearchSort::parse(requested_sort)?;
    let lanes = retrieval_lane_selection(requested_modes)?;
    let lexical_enabled = lanes.lexical;
    let exact_enabled = lanes.exact;
    let exact_paths = if exact_enabled {
        path_hints(query)
    } else {
        vec![]
    };
    let exact_future = async {
        let started = Instant::now();
        if exact_enabled && !exact_paths.is_empty() {
            (
                bounded_retrieval_lane(
                    "exact",
                    exact_candidates(state, auth, &exact_paths, Some(query)),
                )
                .await
                .map(|candidates| (candidates, None)),
                elapsed_ms(started),
            )
        } else {
            (Ok((vec![], None)), elapsed_ms(started))
        }
    };
    let lexical_future = async {
        let started = Instant::now();
        if lexical_enabled {
            (
                bounded_lexical_retrieval_lane(lexical_candidates(
                    state, auth, query, sort, features,
                ))
                .await,
                elapsed_ms(started),
            )
        } else {
            (Ok((vec![], None)), elapsed_ms(started))
        }
    };
    let semantic_future = semantic_lane(
        state,
        auth,
        query,
        sort,
        features,
        lanes,
        request_semantic_embeddings,
    );
    let ((exact, exact_ms), (lexical, lexical_ms), semantic_report) = join_retrieval_lanes(
        exact_future,
        lexical_future,
        semantic_future,
        lanes.semantic_only,
    )
    .await;
    timings.exact = exact_ms;
    timings.lexical = lexical_ms;
    let mut failures = Vec::new();
    let mut workspace_generation = None;
    let mut merged: HashMap<Uuid, Candidate> = HashMap::new();
    let merge_started = Instant::now();
    for (lane, result) in [("exact", exact), ("lexical", lexical)] {
        match result {
            Ok((candidates, generation)) => {
                workspace_generation = workspace_generation.into_iter().chain(generation).max();
                for candidate in candidates {
                    merge_candidate(&mut merged, candidate);
                }
            }
            Err(error) => {
                tracing::warn!(lane, ?error, "simple retrieval lane failed");
                failures.push(lane);
            }
        }
    }
    if apply_semantic_outcome(
        semantic_report,
        lanes.semantic_only,
        &mut merged,
        &mut failures,
        &mut timings,
    ) {
        state.semantic_runtime.record_opportunistic_miss();
    }
    timings.merge += elapsed_ms(merge_started);
    let mut candidates = merged.into_values().collect::<Vec<_>>();
    annotate_candidates(
        &mut candidates,
        features,
        state.config.supersession_demotion,
    );
    sort_candidates(&mut candidates, sort);
    candidates.truncate(limit);
    timings.total = elapsed_ms(total_started);
    Ok((candidates, failures, timings, workspace_generation))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RetrievalLaneSelection {
    exact: bool,
    lexical: bool,
    semantic: bool,
    semantic_only: bool,
}

fn retrieval_lane_selection(requested_modes: &[String]) -> ApiResult<RetrievalLaneSelection> {
    let modes: HashSet<&str> = requested_modes.iter().map(String::as_str).collect();
    if modes
        .iter()
        .any(|mode| !matches!(*mode, "exact" | "lexical" | "semantic"))
    {
        return Err(ApiError::invalid(
            "search modes must be exact, lexical, or semantic",
        ));
    }
    Ok(RetrievalLaneSelection {
        exact: modes.is_empty() || modes.contains("exact"),
        lexical: modes.is_empty() || modes.contains("lexical"),
        semantic: modes.is_empty() || modes.contains("semantic"),
        semantic_only: modes.len() == 1 && modes.contains("semantic"),
    })
}

async fn semantic_search_allowed(state: &AppState, auth: &AuthContext) -> ApiResult<bool> {
    let mut tx = state.begin_read(auth).await?;
    let has_semantic_coverage = sqlx::query_scalar::<_, bool>(
        r#"
        SELECT EXISTS (
          SELECT 1
          FROM brunn.search_chunks
          WHERE user_id=$1 AND embedding IS NOT NULL
          LIMIT 1
        )
        "#,
    )
    .bind(auth.user_id.0)
    .fetch_one(&mut *tx)
    .await?;
    // This read runs under the semantic lane deadline. A cancelled COMMIT can
    // leave SQLx queuing a second ROLLBACK after Postgres has already committed.
    // Drop queues one rollback on pool return without a cancellable close await.
    drop(tx);
    Ok(has_semantic_coverage)
}

#[derive(Debug)]
enum SemanticOutcome {
    NotRequested,
    Disabled,
    IndexUnavailable,
    ReadinessError,
    Success(SemanticCandidates),
    Failed,
    Deferred,
}

#[derive(Debug)]
struct SemanticLaneReport {
    outcome: SemanticOutcome,
    ready_ms: f64,
    lane_ms: f64,
}

/// Runs the semantic lane concurrently with exact+lexical. The policy check,
/// readiness probe, and retrieval all fit inside the single semantic
/// deadline, so semantic latency is never additive and the lane can only
/// defer, never stall the request.
async fn semantic_lane(
    state: &AppState,
    auth: &AuthContext,
    query: &str,
    sort: SearchSort,
    features: Option<&WorkspaceFeatureSnapshot>,
    lanes: RetrievalLaneSelection,
    request_semantic_embeddings: Option<(RequestSemanticEmbeddings, usize)>,
) -> SemanticLaneReport {
    if !lanes.semantic {
        return SemanticLaneReport {
            outcome: SemanticOutcome::NotRequested,
            ready_ms: 0.0,
            lane_ms: 0.0,
        };
    }
    if !state.config.semantic_lane {
        state.semantic_runtime.record_disabled();
        return SemanticLaneReport {
            outcome: SemanticOutcome::Disabled,
            ready_ms: 0.0,
            lane_ms: 0.0,
        };
    }
    state.semantic_runtime.record_requested();
    let lane_started = Instant::now();
    let deadline = state
        .config
        .semantic_deadline
        .unwrap_or(RETRIEVAL_LANE_TIMEOUT)
        .min(RETRIEVAL_LANE_TIMEOUT);
    let ready_started = Instant::now();
    let readiness = if let Some(cached) = state.semantic_runtime.cached_readiness(auth.user_id.0) {
        Ok(Ok(cached))
    } else {
        let probed = tokio::time::timeout(deadline, semantic_search_allowed(state, auth)).await;
        if let Ok(Ok(ready)) = &probed {
            state
                .semantic_runtime
                .store_readiness(auth.user_id.0, *ready);
        }
        probed
    };
    let ready_ms = elapsed_ms(ready_started);
    metrics::histogram!("simple.semantic.ready_ms").record(ready_ms);
    let outcome = match readiness {
        Err(_) => {
            let deferred = state.config.semantic_deadline.is_some();
            metrics::histogram!(
                "simple.retrieval.lane_duration_ms",
                "lane" => "semantic",
                "result" => if deferred { "deferred" } else { "timeout" }
            )
            .record(elapsed_ms(lane_started));
            if deferred {
                metrics::counter!("simple.semantic.deferred").increment(1);
                state.semantic_runtime.record_deferral();
                SemanticOutcome::Deferred
            } else {
                metrics::counter!("simple.retrieval.lane_timeout", "lane" => "semantic")
                    .increment(1);
                state.semantic_runtime.record_failure();
                tracing::warn!(
                    lane = "semantic",
                    "semantic readiness probe exceeded the retrieval lane timeout"
                );
                SemanticOutcome::Failed
            }
        }
        Ok(Err(error)) => {
            tracing::warn!(
                ?error,
                "could not inspect semantic index status; using exact and lexical lanes"
            );
            state.semantic_runtime.record_readiness_error();
            SemanticOutcome::ReadinessError
        }
        Ok(Ok(false)) => {
            state.semantic_runtime.record_index_unavailable();
            SemanticOutcome::IndexUnavailable
        }
        Ok(Ok(true)) => {
            let remaining = deadline.saturating_sub(lane_started.elapsed());
            let prepared_embedding = request_semantic_embeddings.and_then(|(batch, index)| {
                batch.take(
                    &state.semantic_runtime,
                    state.embedder.clone(),
                    state.config.semantic_query_provider_timeout,
                    index,
                )
            });
            match bounded_semantic_lane(
                state,
                remaining,
                semantic_candidates(state, auth, query, sort, features, prepared_embedding),
            )
            .await
            {
                Ok(result) => {
                    state.semantic_runtime.record_success();
                    SemanticOutcome::Success(result)
                }
                Err(SemanticLaneFailure::Failed(error)) => {
                    state.semantic_runtime.record_failure();
                    tracing::warn!(lane = "semantic", ?error, "simple retrieval lane failed");
                    SemanticOutcome::Failed
                }
                Err(SemanticLaneFailure::Deferred) => {
                    state.semantic_runtime.record_deferral();
                    SemanticOutcome::Deferred
                }
            }
        }
    };
    SemanticLaneReport {
        outcome,
        ready_ms,
        lane_ms: elapsed_ms(lane_started),
    }
}

/// Exact+lexical results are always preserved. A semantic-only request reports
/// a semantic failure to its caller; hybrid retrieval treats semantic as an
/// opportunistic accelerator and never downgrades an otherwise complete core
/// result. The return value records an optional attempt that was still pending
/// when the core-lane barrier completed.
fn apply_semantic_outcome(
    report: Option<SemanticLaneReport>,
    semantic_required: bool,
    merged: &mut HashMap<Uuid, Candidate>,
    failures: &mut Vec<&'static str>,
    timings: &mut RetrievalTimings,
) -> bool {
    let Some(report) = report else {
        if semantic_required {
            failures.push("semantic");
            return false;
        }
        return true;
    };
    timings.semantic_ready = report.ready_ms;
    match report.outcome {
        SemanticOutcome::NotRequested => false,
        SemanticOutcome::Disabled => {
            if semantic_required {
                failures.push("semantic_disabled");
            }
            false
        }
        SemanticOutcome::IndexUnavailable => {
            if semantic_required {
                failures.push("semantic_index_unavailable");
            }
            false
        }
        SemanticOutcome::ReadinessError => {
            if semantic_required {
                failures.push("semantic_readiness_error");
            }
            false
        }
        SemanticOutcome::Success(result) => {
            timings.semantic = report.lane_ms;
            timings.embed = result.embed_ms;
            timings.semantic_db = result.database_ms;
            metrics::histogram!("simple.semantic.embed_ms").record(result.embed_ms);
            metrics::histogram!("simple.semantic.db_ms").record(result.database_ms);
            for candidate in result.candidates {
                merge_candidate(merged, candidate);
            }
            false
        }
        SemanticOutcome::Failed => {
            timings.semantic = report.lane_ms;
            if semantic_required {
                failures.push("semantic");
            }
            false
        }
        SemanticOutcome::Deferred => {
            timings.semantic = report.lane_ms;
            if semantic_required {
                failures.push("semantic_deferred");
            }
            false
        }
    }
}

/// Maps a retrieval failure token to its response gap. Semantic reasons are
/// split so callers can distinguish policy, index, dependency, and deadline
/// conditions instead of one conflated warning.
fn retrieval_lane_gap(lane: &'static str) -> Value {
    match lane {
        "semantic_disabled" => json!({
            "kind": "retrieval_lane_unavailable",
            "lane": "semantic",
            "reason": "policy_disabled",
            "message": "semantic retrieval is disabled by policy; exact and lexical evidence was retained"
        }),
        "semantic_index_unavailable" => json!({
            "kind": "retrieval_lane_unavailable",
            "lane": "semantic",
            "reason": "index_unavailable",
            "message": "semantic retrieval has no indexed evidence; exact and lexical evidence was retained"
        }),
        "semantic_readiness_error" => json!({
            "kind": "retrieval_lane_unavailable",
            "lane": "semantic",
            "reason": "dependency_error",
            "message": "semantic retrieval could not inspect its index; exact and lexical evidence was retained"
        }),
        "semantic_deferred" => json!({
            "kind": "retrieval_lane_deferred",
            "lane": "semantic",
            "reason": "deadline_deferred",
            "message": "semantic retrieval exceeded its accelerator deadline; exact and lexical evidence was retained"
        }),
        "semantic" => json!({
            "kind": "retrieval_lane_failed",
            "lane": "semantic",
            "reason": "dependency_error",
            "message": "this lane failed; evidence from other lanes was retained"
        }),
        other => json!({
            "kind": "retrieval_lane_failed",
            "lane": other,
            "message": "this lane failed; evidence from other lanes was retained"
        }),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchSort {
    BestMatch,
    LastModified,
    Title,
}

impl SearchSort {
    fn parse(value: Option<&str>) -> ApiResult<Self> {
        match value.unwrap_or("best_match") {
            "best_match" => Ok(Self::BestMatch),
            "last_modified" => Ok(Self::LastModified),
            "title" => Ok(Self::Title),
            _ => Err(ApiError::invalid(
                "search sort must be best_match, last_modified, or title",
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::BestMatch => "best_match",
            Self::LastModified => "last_modified",
            Self::Title => "title",
        }
    }
}

fn compare_score(left: &Candidate, right: &Candidate) -> std::cmp::Ordering {
    right
        .score
        .partial_cmp(&left.score)
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn compare_modified(left: &Candidate, right: &Candidate) -> std::cmp::Ordering {
    right.updated_at.cmp(&left.updated_at)
}

fn sort_candidates(candidates: &mut [Candidate], sort: SearchSort) {
    candidates.sort_by(|left, right| {
        let ordering = match sort {
            SearchSort::BestMatch => {
                compare_score(left, right).then_with(|| compare_modified(left, right))
            }
            SearchSort::LastModified => {
                compare_modified(left, right).then_with(|| compare_score(left, right))
            }
            SearchSort::Title => left
                .title
                .to_lowercase()
                .cmp(&right.title.to_lowercase())
                .then_with(|| compare_modified(left, right)),
        };
        ordering.then_with(|| left.path.cmp(&right.path))
    });
}

async fn bounded_retrieval_lane<F>(lane: &'static str, future: F) -> ApiResult<Vec<Candidate>>
where
    F: std::future::Future<Output = ApiResult<Vec<Candidate>>>,
{
    let started = Instant::now();
    match tokio::time::timeout(RETRIEVAL_LANE_TIMEOUT, future).await {
        Ok(Ok(candidates)) => {
            metrics::histogram!(
                "simple.retrieval.lane_duration_ms",
                "lane" => lane,
                "result" => "success"
            )
            .record(started.elapsed().as_secs_f64() * 1_000.0);
            metrics::histogram!(
                "simple.retrieval.lane_candidates",
                "lane" => lane
            )
            .record(candidates.len() as f64);
            Ok(candidates)
        }
        Ok(Err(error)) => {
            metrics::histogram!(
                "simple.retrieval.lane_duration_ms",
                "lane" => lane,
                "result" => "failure"
            )
            .record(started.elapsed().as_secs_f64() * 1_000.0);
            metrics::counter!(
                "simple.retrieval.lane_failure",
                "lane" => lane
            )
            .increment(1);
            Err(error)
        }
        Err(_) => {
            metrics::histogram!(
                "simple.retrieval.lane_duration_ms",
                "lane" => lane,
                "result" => "timeout"
            )
            .record(started.elapsed().as_secs_f64() * 1_000.0);
            metrics::counter!("simple.retrieval.lane_timeout", "lane" => lane).increment(1);
            Err(ApiError::public(
                StatusCode::SERVICE_UNAVAILABLE,
                "retrieval_lane_timeout",
                format!("{lane} retrieval exceeded its bounded time budget"),
            ))
        }
    }
}

enum SemanticLaneFailure {
    Failed(ApiError),
    Deferred,
}

async fn bounded_semantic_lane<F>(
    state: &AppState,
    deadline: Duration,
    future: F,
) -> Result<SemanticCandidates, SemanticLaneFailure>
where
    F: std::future::Future<Output = ApiResult<SemanticCandidates>>,
{
    let started = Instant::now();
    match tokio::time::timeout(deadline, future).await {
        Ok(Ok(result)) => {
            metrics::histogram!(
                "simple.retrieval.lane_duration_ms",
                "lane" => "semantic",
                "result" => "success"
            )
            .record(elapsed_ms(started));
            metrics::histogram!(
                "simple.retrieval.lane_candidates",
                "lane" => "semantic"
            )
            .record(result.candidates.len() as f64);
            Ok(result)
        }
        Ok(Err(error)) => {
            metrics::histogram!(
                "simple.retrieval.lane_duration_ms",
                "lane" => "semantic",
                "result" => "failure"
            )
            .record(elapsed_ms(started));
            metrics::counter!(
                "simple.retrieval.lane_failure",
                "lane" => "semantic"
            )
            .increment(1);
            Err(SemanticLaneFailure::Failed(error))
        }
        Err(_) if state.config.semantic_deadline.is_some() => {
            metrics::histogram!(
                "simple.retrieval.lane_duration_ms",
                "lane" => "semantic",
                "result" => "deferred"
            )
            .record(started.elapsed().as_secs_f64() * 1_000.0);
            metrics::counter!("simple.semantic.deferred").increment(1);
            Err(SemanticLaneFailure::Deferred)
        }
        Err(_) => {
            metrics::histogram!(
                "simple.retrieval.lane_duration_ms",
                "lane" => "semantic",
                "result" => "timeout"
            )
            .record(elapsed_ms(started));
            metrics::counter!(
                "simple.retrieval.lane_timeout",
                "lane" => "semantic"
            )
            .increment(1);
            Err(SemanticLaneFailure::Failed(ApiError::public(
                StatusCode::SERVICE_UNAVAILABLE,
                "retrieval_lane_timeout",
                "semantic retrieval exceeded its bounded time budget",
            )))
        }
    }
}

async fn bounded_lexical_retrieval_lane<F>(future: F) -> ApiResult<(Vec<Candidate>, Option<i64>)>
where
    F: std::future::Future<Output = ApiResult<(Vec<Candidate>, Option<i64>)>>,
{
    let started = Instant::now();
    match tokio::time::timeout(RETRIEVAL_LANE_TIMEOUT, future).await {
        Ok(Ok((candidates, generation))) => {
            metrics::histogram!(
                "simple.retrieval.lane_duration_ms",
                "lane" => "lexical",
                "result" => "success"
            )
            .record(started.elapsed().as_secs_f64() * 1_000.0);
            metrics::histogram!(
                "simple.retrieval.lane_candidates",
                "lane" => "lexical"
            )
            .record(candidates.len() as f64);
            Ok((candidates, generation))
        }
        Ok(Err(error)) => {
            metrics::histogram!(
                "simple.retrieval.lane_duration_ms",
                "lane" => "lexical",
                "result" => "failure"
            )
            .record(started.elapsed().as_secs_f64() * 1_000.0);
            metrics::counter!(
                "simple.retrieval.lane_failure",
                "lane" => "lexical"
            )
            .increment(1);
            Err(error)
        }
        Err(_) => {
            metrics::histogram!(
                "simple.retrieval.lane_duration_ms",
                "lane" => "lexical",
                "result" => "timeout"
            )
            .record(started.elapsed().as_secs_f64() * 1_000.0);
            metrics::counter!(
                "simple.retrieval.lane_timeout",
                "lane" => "lexical"
            )
            .increment(1);
            Err(ApiError::public(
                StatusCode::SERVICE_UNAVAILABLE,
                "retrieval_lane_timeout",
                "lexical retrieval exceeded its bounded time budget",
            ))
        }
    }
}

async fn exact_candidates(
    state: &AppState,
    auth: &AuthContext,
    paths: &[String],
    verbatim_query: Option<&str>,
) -> ApiResult<Vec<Candidate>> {
    let verbatim_terms = if state.config.verbatim_spans {
        verbatim_query.map(verbatim_match_terms).unwrap_or_default()
    } else {
        Vec::new()
    };
    let include_verbatim_source = !verbatim_terms.is_empty();
    let mut tx = state.begin_read(auth).await?;
    let rows = sqlx::query(
        r#"
        SELECT entry.id,entry.path,entry.title,entry.current_version,
               entry.updated_at,version.content_sha256,
               CASE
                 WHEN $3 THEN coalesce(version.content,'')
                 ELSE left(coalesce(version.content,''),2400)
               END AS content
        FROM brunn.entries AS entry
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
         AND version.version=entry.current_version
        WHERE entry.user_id=$1
          AND entry.deleted_at IS NULL
          AND entry.path=ANY($2)
        ORDER BY entry.path
        LIMIT 32
        "#,
    )
    .bind(auth.user_id.0)
    .bind(paths)
    .bind(include_verbatim_source)
    .fetch_all(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(rows
        .into_iter()
        .map(|row| {
            let content = row.get::<String, _>("content");
            let excerpt = truncate_chars(&content, 2_400);
            let version = row.get("current_version");
            let content_sha256 = row.get::<String, _>("content_sha256");
            let verbatim_matches =
                extract_verbatim_matches(&content, &verbatim_terms, version, &content_sha256);
            Candidate {
                entry_id: row.get("id"),
                path: row.get("path"),
                title: row.get("title"),
                version,
                updated_at: row.get("updated_at"),
                content_sha256,
                heading: String::new(),
                excerpt: excerpt.clone(),
                score: 10.0,
                lanes: vec!["exact".to_owned()],
                sections: vec![CandidateSection {
                    heading: String::new(),
                    excerpt,
                    score: 10.0,
                }],
                verbatim_matches,
                superseded_by: None,
            }
        })
        .collect())
}

async fn lexical_candidates(
    state: &AppState,
    auth: &AuthContext,
    query: &str,
    sort: SearchSort,
    features: Option<&WorkspaceFeatureSnapshot>,
) -> ApiResult<(Vec<Candidate>, Option<i64>)> {
    let mut tx = state.begin_read(auth).await?;
    let mut candidates = Vec::new();
    let mut workspace_generation = None;
    let anchors = search_anchors(query);
    let mut anchor_hit = false;
    if state.config.lexical_single_scan && !anchors.is_empty() {
        let consolidated = anchors
            .iter()
            .map(|anchor| format!("\"{}\"", anchor.replace('"', " ")))
            .collect::<Vec<_>>()
            .join(" OR ");
        let (found, generation) = fetch_lexical_candidates(
            &mut tx,
            &consolidated,
            query,
            sort,
            features,
            state.config.supersession_demotion,
            state.config.supersession_demotion_weight,
            state.config.read_path_roundtrip_v1,
            auth.user_id.0,
        )
        .await?;
        anchor_hit = !found.is_empty();
        candidates.extend(found);
        workspace_generation = generation;
    } else {
        for anchor in &anchors {
            let (anchor_candidates, generation) = fetch_lexical_candidates(
                &mut tx,
                anchor,
                query,
                sort,
                features,
                state.config.supersession_demotion,
                state.config.supersession_demotion_weight,
                state.config.read_path_roundtrip_v1,
                auth.user_id.0,
            )
            .await?;
            anchor_hit |= !anchor_candidates.is_empty();
            candidates.extend(anchor_candidates);
            workspace_generation = workspace_generation.into_iter().chain(generation).max();
        }
    }
    if !anchor_hit {
        // Preserve a full bounded result set for every selective pair. Folding
        // these into one globally sampled candidate set changes large-corpus
        // recall before the full-query bonus and final merge can run.
        for focused in bounded_lexical_fallback_queries(query) {
            let (focused_candidates, generation) = fetch_lexical_candidates(
                &mut tx,
                &focused,
                query,
                sort,
                features,
                state.config.supersession_demotion,
                state.config.supersession_demotion_weight,
                state.config.read_path_roundtrip_v1,
                auth.user_id.0,
            )
            .await?;
            candidates.extend(focused_candidates);
            workspace_generation = workspace_generation.into_iter().chain(generation).max();
        }
    }
    tx.commit().await?;
    let mut merged = HashMap::new();
    for candidate in candidates {
        merge_candidate(&mut merged, candidate);
    }
    Ok((merged.into_values().collect(), workspace_generation))
}

async fn fetch_lexical_candidates(
    tx: &mut Transaction<'_, Postgres>,
    retrieval_query: &str,
    scoring_query: &str,
    sort: SearchSort,
    features: Option<&WorkspaceFeatureSnapshot>,
    supersession_enabled: bool,
    supersession_weight: f64,
    include_generation: bool,
    user_id: Uuid,
) -> ApiResult<(Vec<Candidate>, Option<i64>)> {
    let rows = if include_generation {
        sqlx::query(SIMPLE_LEXICAL_CANDIDATES_WITH_GENERATION_SQL)
            .bind(user_id)
            .bind(retrieval_query)
            .bind(sort.as_str())
            .fetch_all(&mut **tx)
            .await?
    } else {
        sqlx::query(SIMPLE_LEXICAL_CANDIDATES_SQL)
            .bind(retrieval_query)
            .bind(sort.as_str())
            .fetch_all(&mut **tx)
            .await?
    };
    let workspace_generation = include_generation
        .then(|| {
            rows.first()
                .and_then(|row| row.get::<Option<i64>, _>("workspace_generation"))
        })
        .flatten();
    let candidates = rows
        .into_iter()
        .filter(|row| !include_generation || row.get::<Option<Uuid>, _>("entry_id").is_some())
        .map(|row| {
            let path: String = row.get("path");
            let title: String = row.get("title");
            let heading: String = row.get("heading");
            let excerpt = truncate_chars(&row.get::<String, _>("content"), 2_400);
            let score = 3.0
                + row.get::<f64, _>("score")
                + lexical_candidate_bonus(scoring_query, &path, &title, &heading, &excerpt)
                - derived_penalty(&path)
                - supersession_penalty(&path, features, supersession_enabled, supersession_weight);
            Candidate {
                entry_id: row.get("entry_id"),
                score,
                path,
                title,
                version: row.get("current_version"),
                updated_at: row.get("updated_at"),
                content_sha256: row.get("content_sha256"),
                heading: heading.clone(),
                excerpt: excerpt.clone(),
                lanes: vec!["lexical".to_owned()],
                sections: vec![CandidateSection {
                    heading,
                    excerpt,
                    score,
                }],
                verbatim_matches: vec![],
                superseded_by: None,
            }
        })
        .collect();
    Ok((candidates, workspace_generation))
}

async fn semantic_candidates(
    state: &AppState,
    auth: &AuthContext,
    query: &str,
    sort: SearchSort,
    features: Option<&WorkspaceFeatureSnapshot>,
    prepared_embedding: Option<PreparedQueryEmbedding>,
) -> ApiResult<SemanticCandidates> {
    let embed_started = Instant::now();
    let vector = match prepared_embedding {
        Some(prepared) => prepared.resolve().await?,
        None => {
            state
                .semantic_runtime
                .query_embedding(
                    state.embedder.clone(),
                    query,
                    state.config.embed_cache,
                    state.config.semantic_query_provider_timeout,
                )
                .await?
        }
    };
    let embed_ms = elapsed_ms(embed_started);
    let database_started = Instant::now();
    let mut tx = state.begin_read(auth).await?;
    let rows = sqlx::query(SIMPLE_SEMANTIC_CANDIDATES_SQL)
        .bind(Vector::from(vector))
        .bind(sort.as_str())
        .fetch_all(&mut *tx)
        .await?;
    // The semantic deadline also covers this read-only transaction.
    drop(tx);
    let database_ms = elapsed_ms(database_started);
    let candidates = rows
        .into_iter()
        .map(|row| {
            let distance = row.get::<f64, _>("distance");
            let path: String = row.get("path");
            let title: String = row.get("title");
            let heading: String = row.get("heading");
            let excerpt = truncate_chars(&row.get::<String, _>("content"), 2_400);
            let score = 2.0
                + (1.0 - distance).max(0.0)
                + lexical_candidate_bonus(query, &path, &title, &heading, &excerpt)
                - derived_penalty(&path)
                - supersession_penalty(
                    &path,
                    features,
                    state.config.supersession_demotion,
                    state.config.supersession_demotion_weight,
                );
            Candidate {
                entry_id: row.get("entry_id"),
                score,
                path,
                title,
                version: row.get("current_version"),
                updated_at: row.get("updated_at"),
                content_sha256: row.get("content_sha256"),
                heading: heading.clone(),
                excerpt: excerpt.clone(),
                lanes: vec!["semantic".to_owned()],
                sections: vec![CandidateSection {
                    heading,
                    excerpt,
                    score,
                }],
                verbatim_matches: vec![],
                superseded_by: None,
            }
        })
        .collect();
    Ok(SemanticCandidates {
        candidates,
        embed_ms,
        database_ms,
    })
}

fn derived_penalty(path: &str) -> f64 {
    if path.starts_with(".brunn/proposals/") {
        2.0
    } else if path.starts_with(".brunn/derived/") || path.starts_with(".brunn/dreams/") {
        1.0
    } else {
        0.0
    }
}

fn supersession_penalty(
    path: &str,
    features: Option<&WorkspaceFeatureSnapshot>,
    enabled: bool,
    weight: f64,
) -> f64 {
    if enabled && features.is_some_and(|snapshot| snapshot.superseded_by(path).is_some()) {
        weight
    } else {
        0.0
    }
}

fn annotate_candidates(
    candidates: &mut [Candidate],
    features: Option<&WorkspaceFeatureSnapshot>,
    enabled: bool,
) {
    if !enabled {
        return;
    }
    let Some(features) = features else {
        return;
    };
    for candidate in candidates {
        candidate.superseded_by = features.superseded_by(&candidate.path);
    }
}

fn merge_candidate(merged: &mut HashMap<Uuid, Candidate>, mut candidate: Candidate) {
    normalize_candidate_sections(&mut candidate);
    if let Some(existing) = merged.get_mut(&candidate.entry_id) {
        let candidate_score = candidate.score;
        let adds_lane = candidate
            .lanes
            .iter()
            .any(|lane| !existing.lanes.contains(lane));
        existing.sections.extend(candidate.sections);
        normalize_candidate_sections(existing);
        existing.score = existing.score.max(candidate_score);
        if adds_lane {
            existing.score += 0.15;
        }
        for lane in candidate.lanes {
            if !existing.lanes.contains(&lane) {
                existing.lanes.push(lane);
            }
        }
        existing.verbatim_matches.extend(candidate.verbatim_matches);
        existing
            .verbatim_matches
            .sort_by_key(|source_match| source_match.line_no);
        existing.verbatim_matches.dedup_by(|left, right| {
            left.line_no == right.line_no
                && left.byte_start == right.byte_start
                && left.byte_end == right.byte_end
        });
        existing
            .verbatim_matches
            .truncate(MAX_VERBATIM_MATCHES_PER_CANDIDATE);
    } else {
        merged.insert(candidate.entry_id, candidate);
    }
}

fn normalize_candidate_sections(candidate: &mut Candidate) {
    candidate.sections.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.heading.cmp(&right.heading))
            .then_with(|| left.excerpt.cmp(&right.excerpt))
    });
    let mut seen = HashSet::new();
    candidate
        .sections
        .retain(|section| seen.insert((section.heading.clone(), section.excerpt.clone())));
    candidate.sections.truncate(3);
    if let Some(section) = candidate.sections.first() {
        candidate.heading.clone_from(&section.heading);
        candidate.excerpt.clone_from(&section.excerpt);
    }
}

fn select_search_candidate_sets(
    candidate_sets: &[Vec<Candidate>],
    fair_share: bool,
    max_candidates: usize,
) -> (Vec<Vec<Candidate>>, bool) {
    let available = candidate_sets.iter().map(Vec::len).sum::<usize>();
    let mut selected = vec![Vec::new(); candidate_sets.len()];
    if fair_share {
        let max_rank = candidate_sets.iter().map(Vec::len).max().unwrap_or(0);
        let mut selected_count = 0_usize;
        'ranks: for rank in 0..max_rank {
            for (query_index, candidates) in candidate_sets.iter().enumerate() {
                if selected_count == max_candidates {
                    break 'ranks;
                }
                if let Some(candidate) = candidates.get(rank) {
                    selected[query_index].push(candidate.clone());
                    selected_count += 1;
                }
            }
        }
    } else {
        let mut remaining = max_candidates;
        for (query_index, candidates) in candidate_sets.iter().enumerate() {
            let retain = candidates.len().min(remaining);
            selected[query_index].extend(candidates.iter().take(retain).cloned());
            remaining = remaining.saturating_sub(retain);
        }
    }
    let retained = selected.iter().map(Vec::len).sum::<usize>();
    (selected, retained < available)
}

async fn fetch_search_top1_hydration(
    state: &AppState,
    auth: &AuthContext,
    candidate_sets: &[Vec<Candidate>],
) -> ApiResult<HashMap<Uuid, SearchHydration>> {
    let mut seen = HashSet::new();
    let ids = candidate_sets
        .iter()
        .filter_map(|candidates| candidates.first())
        .map(|candidate| candidate.entry_id)
        .filter(|entry_id| seen.insert(*entry_id))
        .collect::<Vec<_>>();
    if ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut tx = state.begin_read(auth).await?;
    let rows = sqlx::query(
        r#"
        SELECT entry.id,version.size_bytes,
               CASE WHEN version.size_bytes <= $3 THEN version.content ELSE NULL END AS content
        FROM brunn.entries AS entry
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
         AND version.version=entry.current_version
        WHERE entry.user_id=$1
          AND entry.id=ANY($2)
          AND entry.deleted_at IS NULL
        "#,
    )
    .bind(auth.user_id.0)
    .bind(&ids)
    .bind(i64::try_from(MAX_OPEN_COMPLETE_SOURCE_CHARS).unwrap_or(i64::MAX))
    .fetch_all(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(rows
        .into_iter()
        .filter_map(|row| {
            let size_bytes = usize::try_from(row.get::<i64, _>("size_bytes")).ok()?;
            let content = row.get::<Option<String>, _>("content")?;
            Some((
                row.get::<Uuid, _>("id"),
                SearchHydration {
                    size_bytes,
                    content,
                },
            ))
        })
        .collect())
}

fn assemble_search_candidate_views(
    candidate_sets: Vec<Vec<Candidate>>,
    hydration: &HashMap<Uuid, SearchHydration>,
    options: SearchBudgetOptions,
) -> (Vec<Vec<SearchCandidateView>>, bool) {
    let mut views = candidate_sets
        .into_iter()
        .map(|candidates| {
            candidates
                .into_iter()
                .enumerate()
                .map(|(rank, candidate)| SearchCandidateView {
                    candidate,
                    representation: SearchRepresentation::Excerpt,
                    complete_text: None,
                    demote_additional_sections: options
                        .section_demotion_top_n
                        .is_some_and(|top_n| rank >= top_n),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut remaining_chars = options.max_chars;
    let mut truncated = false;

    if options.top1_hydration {
        for query_views in &mut views {
            let Some(top) = query_views.first_mut() else {
                continue;
            };
            let Some(loaded) = hydration.get(&top.candidate.entry_id) else {
                continue;
            };
            let content_chars = loaded.content.chars().count();
            if loaded.size_bytes > MAX_OPEN_COMPLETE_SOURCE_CHARS
                || loaded.size_bytes > remaining_chars
                || content_chars > remaining_chars
            {
                continue;
            }
            top.representation = SearchRepresentation::CompleteSource;
            top.complete_text = Some(loaded.content.clone());
            remaining_chars = remaining_chars.saturating_sub(content_chars);
            for view in query_views.iter_mut().skip(1) {
                if desired_search_evidence_chars(view) > 0 {
                    truncated = true;
                }
                view.representation = SearchRepresentation::PointerLead;
            }
        }
    }

    if options.fair_share {
        let quotas = fair_search_char_quotas(&views, remaining_chars);
        for (query_views, quota) in views.iter_mut().zip(quotas) {
            let mut query_remaining = quota;
            for view in query_views {
                let (used, view_truncated) = apply_search_view_budget(view, query_remaining);
                query_remaining = query_remaining.saturating_sub(used);
                truncated |= view_truncated;
            }
        }
    } else {
        for query_views in &mut views {
            for view in query_views {
                let (used, view_truncated) = apply_search_view_budget(view, remaining_chars);
                remaining_chars = remaining_chars.saturating_sub(used);
                truncated |= view_truncated;
            }
        }
    }

    (views, truncated)
}

fn fair_search_char_quotas(views: &[Vec<SearchCandidateView>], max_chars: usize) -> Vec<usize> {
    if views.is_empty() {
        return Vec::new();
    }
    let desired = views
        .iter()
        .map(|query_views| {
            query_views
                .iter()
                .map(desired_search_evidence_chars)
                .sum::<usize>()
        })
        .collect::<Vec<_>>();
    let floor = max_chars / views.len();
    let mut quotas = desired
        .iter()
        .map(|desired_chars| (*desired_chars).min(floor))
        .collect::<Vec<_>>();
    let mut remaining = max_chars.saturating_sub(quotas.iter().sum());
    while remaining > 0 {
        let mut progressed = false;
        for (query_index, desired_chars) in desired.iter().enumerate() {
            let unmet = desired_chars.saturating_sub(quotas[query_index]);
            let granted = unmet.min(2_400).min(remaining);
            if granted > 0 {
                quotas[query_index] += granted;
                remaining -= granted;
                progressed = true;
            }
            if remaining == 0 {
                break;
            }
        }
        if !progressed {
            break;
        }
    }
    quotas
}

fn desired_search_evidence_chars(view: &SearchCandidateView) -> usize {
    if view.representation != SearchRepresentation::Excerpt {
        return 0;
    }
    let primary = view.candidate.excerpt.chars().count();
    if view.demote_additional_sections {
        primary
    } else {
        primary.saturating_add(
            view.candidate
                .sections
                .iter()
                .skip(1)
                .map(|section| section.excerpt.chars().count())
                .sum::<usize>(),
        )
    }
}

fn apply_search_view_budget(
    view: &mut SearchCandidateView,
    available_chars: usize,
) -> (usize, bool) {
    if view.representation != SearchRepresentation::Excerpt {
        return (0, false);
    }
    let original_chars = desired_search_evidence_chars(view);
    let mut remaining = available_chars.min(original_chars);
    let starting = remaining;
    let truncated = if view.demote_additional_sections {
        let original_primary = view.candidate.excerpt.chars().count();
        let retained = truncate_chars(&view.candidate.excerpt, remaining.min(2_400));
        remaining = remaining.saturating_sub(retained.chars().count());
        let truncated = retained.chars().count() < original_primary
            || view
                .candidate
                .sections
                .iter()
                .skip(1)
                .any(|section| !section.excerpt.is_empty());
        view.candidate.excerpt = retained;
        if let Some(primary) = view.candidate.sections.first_mut() {
            primary.excerpt.clone_from(&view.candidate.excerpt);
        }
        truncated
    } else {
        truncate_candidate_evidence(&mut view.candidate, &mut remaining)
    };
    if view.candidate.excerpt.is_empty() {
        view.representation = SearchRepresentation::PointerLead;
    }
    (starting.saturating_sub(remaining), truncated)
}

fn render_budgeted_search_candidate(
    view: &SearchCandidateView,
    remaining_verbatim_chars: &mut usize,
) -> Value {
    let candidate = &view.candidate;
    let representation = match view.representation {
        SearchRepresentation::CompleteSource => "complete_source",
        SearchRepresentation::Excerpt => "excerpt",
        SearchRepresentation::PointerLead => "pointer_lead",
    };
    let mut rendered = serde_json::Map::from_iter([
        (
            "reference".to_owned(),
            Value::String(format!("entry:{}", candidate.entry_id)),
        ),
        ("path".to_owned(), Value::String(candidate.path.clone())),
        ("title".to_owned(), Value::String(candidate.title.clone())),
        ("version".to_owned(), json!(candidate.version)),
        ("updated_at".to_owned(), json!(candidate.updated_at)),
        (
            "representation".to_owned(),
            Value::String(representation.to_owned()),
        ),
    ]);
    if !candidate.heading.is_empty() {
        rendered.insert(
            "heading".to_owned(),
            Value::String(candidate.heading.clone()),
        );
    }
    match view.representation {
        SearchRepresentation::CompleteSource => {
            rendered.insert(
                "content_hash".to_owned(),
                Value::String(format!("sha256:{}", candidate.content_sha256)),
            );
            rendered.insert(
                "text".to_owned(),
                Value::String(view.complete_text.clone().unwrap_or_default()),
            );
        }
        SearchRepresentation::PointerLead => {
            rendered.insert(
                "content_hash".to_owned(),
                Value::String(format!("sha256:{}", candidate.content_sha256)),
            );
            rendered.insert("score".to_owned(), json!(candidate.score));
        }
        SearchRepresentation::Excerpt => {
            rendered.insert(
                "excerpt".to_owned(),
                Value::String(candidate.excerpt.clone()),
            );
            let additional_sections = candidate
                .sections
                .iter()
                .skip(1)
                .filter_map(|section| {
                    if view.demote_additional_sections {
                        if section.heading.is_empty() {
                            return None;
                        }
                        Some(json!({
                            "representation": "heading_lead",
                            "heading": section.heading,
                            "path": candidate.path,
                            "version": candidate.version
                        }))
                    } else if section.excerpt.is_empty() {
                        None
                    } else {
                        let mut rendered = serde_json::Map::from_iter([(
                            "excerpt".to_owned(),
                            Value::String(section.excerpt.clone()),
                        )]);
                        if !section.heading.is_empty() {
                            rendered.insert(
                                "heading".to_owned(),
                                Value::String(section.heading.clone()),
                            );
                        }
                        Some(Value::Object(rendered))
                    }
                })
                .collect::<Vec<_>>();
            if !additional_sections.is_empty() {
                rendered.insert(
                    "additional_sections".to_owned(),
                    Value::Array(additional_sections),
                );
            }
        }
    }
    let verbatim_matches = render_verbatim_matches(candidate, remaining_verbatim_chars);
    if !verbatim_matches.is_empty() {
        rendered.insert(
            "verbatim_matches".to_owned(),
            Value::Array(verbatim_matches),
        );
    }
    if let Some(superseded_by) = &candidate.superseded_by {
        rendered.insert("superseded_by".to_owned(), json!(superseded_by));
    }
    Value::Object(rendered)
}

fn verbatim_match_terms(query: &str) -> Vec<String> {
    search_anchors(query)
        .into_iter()
        .filter(|term| !looks_like_path(term))
        .take(MAX_VERBATIM_MATCHES_PER_CANDIDATE)
        .collect()
}

fn extract_verbatim_matches(
    content: &str,
    terms: &[String],
    version: i64,
    content_sha256: &str,
) -> Vec<VerbatimMatch> {
    if terms.is_empty() {
        return vec![];
    }
    let mut byte_start = 0_usize;
    let mut matches = Vec::new();
    for (index, source_line) in content.split_inclusive('\n').enumerate() {
        let line = source_line.strip_suffix('\n').unwrap_or(source_line);
        if terms.iter().any(|term| line.contains(term)) {
            let text = truncate_chars(line, MAX_VERBATIM_LINE_CHARS);
            let truncated = text.len() < line.len();
            matches.push(VerbatimMatch {
                line_no: index + 1,
                byte_start,
                byte_end: byte_start + text.len(),
                text,
                version,
                content_hash: format!("sha256:{content_sha256}"),
                truncated,
            });
            if matches.len() >= MAX_VERBATIM_MATCHES_PER_CANDIDATE {
                break;
            }
        }
        byte_start += source_line.len();
    }
    matches
}

fn render_verbatim_matches(candidate: &Candidate, remaining_chars: &mut usize) -> Vec<Value> {
    let mut rendered = Vec::new();
    for source_match in &candidate.verbatim_matches {
        if *remaining_chars == 0 {
            break;
        }
        let mut retained = source_match.clone();
        retained.text = truncate_chars(
            &source_match.text,
            (*remaining_chars).min(MAX_VERBATIM_LINE_CHARS),
        );
        let retained_chars = retained.text.chars().count();
        if retained_chars == 0 {
            break;
        }
        *remaining_chars = remaining_chars.saturating_sub(retained_chars);
        retained.byte_end = retained.byte_start + retained.text.len();
        retained.truncated |= retained.text.len() < source_match.text.len();
        rendered.push(json!(retained));
    }
    rendered
}

fn render_search_candidate(candidate: &Candidate, remaining_verbatim_chars: &mut usize) -> Value {
    let mut rendered = serde_json::Map::from_iter([
        (
            "reference".to_owned(),
            Value::String(format!("entry:{}", candidate.entry_id)),
        ),
        ("path".to_owned(), Value::String(candidate.path.clone())),
        ("title".to_owned(), Value::String(candidate.title.clone())),
        ("version".to_owned(), json!(candidate.version)),
        ("updated_at".to_owned(), json!(candidate.updated_at)),
        (
            "excerpt".to_owned(),
            Value::String(candidate.excerpt.clone()),
        ),
    ]);
    if !candidate.heading.is_empty() {
        rendered.insert(
            "heading".to_owned(),
            Value::String(candidate.heading.clone()),
        );
    }
    let additional_sections = candidate
        .sections
        .iter()
        .skip(1)
        .map(|section| {
            let mut rendered = serde_json::Map::from_iter([(
                "excerpt".to_owned(),
                Value::String(section.excerpt.clone()),
            )]);
            if !section.heading.is_empty() {
                rendered.insert("heading".to_owned(), Value::String(section.heading.clone()));
            }
            Value::Object(rendered)
        })
        .collect::<Vec<_>>();
    if !additional_sections.is_empty() {
        rendered.insert(
            "additional_sections".to_owned(),
            Value::Array(additional_sections),
        );
    }
    let verbatim_matches = render_verbatim_matches(candidate, remaining_verbatim_chars);
    if !verbatim_matches.is_empty() {
        rendered.insert(
            "verbatim_matches".to_owned(),
            Value::Array(verbatim_matches),
        );
    }
    if let Some(superseded_by) = &candidate.superseded_by {
        rendered.insert("superseded_by".to_owned(), json!(superseded_by));
    }
    Value::Object(rendered)
}

fn render_evidence_lead(candidate: &Candidate) -> Value {
    let mut rendered = serde_json::Map::from_iter([
        (
            "reference".to_owned(),
            Value::String(format!("entry:{}", candidate.entry_id)),
        ),
        ("path".to_owned(), Value::String(candidate.path.clone())),
        ("title".to_owned(), Value::String(candidate.title.clone())),
        ("version".to_owned(), json!(candidate.version)),
        ("updated_at".to_owned(), json!(candidate.updated_at)),
    ]);
    if !candidate.heading.is_empty() {
        rendered.insert(
            "heading".to_owned(),
            Value::String(candidate.heading.clone()),
        );
    }
    if let Some(superseded_by) = &candidate.superseded_by {
        rendered.insert("superseded_by".to_owned(), json!(superseded_by));
    }
    Value::Object(rendered)
}

fn apply_checkpoint_budget(checkpoint: &mut Option<Value>, token_budget: usize) -> (bool, usize) {
    let total_chars = token_budget.saturating_mul(4);
    let Some(checkpoint) = checkpoint.as_mut() else {
        return (false, token_budget);
    };
    let Some(text) = checkpoint.get("text").and_then(Value::as_str) else {
        return (false, token_budget);
    };
    let text_chars = text.chars().count();
    let retained_chars = text_chars.min(total_chars);
    let truncated = text_chars > retained_chars;
    if truncated {
        checkpoint["text"] = Value::String(truncate_chars(text, retained_chars));
        checkpoint["text_truncated"] = Value::Bool(true);
    }
    (truncated, total_chars.saturating_sub(retained_chars) / 4)
}

async fn open_hint_candidates(
    state: &AppState,
    auth: &AuthContext,
    hints: &OpenHints,
) -> ApiResult<(Vec<Candidate>, Vec<Value>)> {
    let mut requested = Vec::new();
    let mut seen = HashSet::new();
    for hint in hints.root_refs.iter().chain(&hints.open_object_refs) {
        let normalized = hint.trim();
        if !normalized.is_empty() && seen.insert(normalized.to_lowercase()) {
            requested.push(normalized.to_owned());
        }
    }
    if requested.len() > 32 {
        return Err(ApiError::invalid(
            "open hints are limited to 32 exact paths or entry references",
        ));
    }
    let mut results =
        futures::stream::iter(requested.into_iter().enumerate().map(|(index, hint)| {
            let state = state.clone();
            let auth = auth.clone();
            async move {
                let result = if hint.starts_with("entry:")
                    || hint.starts_with("checkpoint:")
                    || Uuid::parse_str(&hint).is_ok()
                {
                    resolve_entry_summary(&state, &auth, None, Some(&hint)).await
                } else if validate_path(&hint).is_ok() {
                    resolve_entry_summary(&state, &auth, Some(&hint), None).await
                } else {
                    Err(ApiError::invalid(
                        "open hints must be exact paths, entry refs, or checkpoint refs",
                    ))
                };
                (index, hint, result)
            }
        }))
        .buffer_unordered(8)
        .collect::<Vec<_>>()
        .await;
    results.sort_by_key(|(index, _, _)| *index);
    let mut candidates = Vec::new();
    let mut gaps = Vec::new();
    for (_, hint, result) in results {
        match result {
            Ok(entry) => candidates.push(Candidate {
                entry_id: entry.id,
                path: entry.path,
                title: entry.title,
                version: entry.version,
                updated_at: entry.updated_at,
                content_sha256: entry.content_sha256,
                heading: String::new(),
                excerpt: String::new(),
                score: 40.0,
                lanes: vec!["hint".to_owned()],
                sections: vec![],
                verbatim_matches: vec![],
                superseded_by: None,
            }),
            Err(ApiError::Public {
                status: StatusCode::NOT_FOUND,
                ..
            }) => gaps.push(json!({
                "kind": "open_hint_not_found",
                "hint": hint,
                "message": "the exact hinted source was not found; other evidence was retained"
            })),
            Err(ApiError::Public { code, message, .. }) => gaps.push(json!({
                "kind": "open_hint_invalid",
                "hint": hint,
                "code": code,
                "message": message
            })),
            Err(error) => return Err(error),
        }
    }
    Ok((candidates, gaps))
}

async fn hydrate_candidates(
    state: &AppState,
    auth: &AuthContext,
    candidates: &[Candidate],
    token_budget: usize,
) -> ApiResult<(Vec<Value>, Option<i64>)> {
    let ids = candidates
        .iter()
        .take(HYDRATED_DOCUMENT_LIMIT)
        .map(|candidate| candidate.entry_id)
        .collect::<Vec<_>>();
    if ids.is_empty() && !state.config.read_path_roundtrip_v1 {
        return Ok((vec![], None));
    }
    let mut tx = state.begin_read(auth).await?;
    let rows = if state.config.read_path_roundtrip_v1 {
        sqlx::query(
            r#"
            WITH generation AS (
              SELECT brunn_auth.workspace_generation($1) AS workspace_generation
            ), documents AS MATERIALIZED (
              SELECT entry.id,version.size_bytes,version.content
              FROM brunn.entries AS entry
              JOIN brunn.entry_versions AS version
                ON version.user_id=entry.user_id
               AND version.entry_id=entry.id
               AND version.version=entry.current_version
              WHERE entry.user_id=$1
                AND entry.id=ANY($2)
                AND entry.deleted_at IS NULL
            )
            SELECT generation.workspace_generation,
                   documents.id,documents.size_bytes,documents.content
            FROM generation
            LEFT JOIN documents ON true
            "#,
        )
        .bind(auth.user_id.0)
        .bind(&ids)
        .fetch_all(&mut *tx)
        .await?
    } else {
        sqlx::query(
            r#"
            SELECT NULL::bigint AS workspace_generation,
                   entry.id,version.size_bytes,NULL::text AS content
            FROM brunn.entries AS entry
            JOIN brunn.entry_versions AS version
              ON version.user_id=entry.user_id
             AND version.entry_id=entry.id
             AND version.version=entry.current_version
            WHERE entry.user_id=$1
              AND entry.id=ANY($2)
              AND entry.deleted_at IS NULL
            "#,
        )
        .bind(auth.user_id.0)
        .bind(&ids)
        .fetch_all(&mut *tx)
        .await?
    };
    let generation = rows
        .first()
        .and_then(|row| row.get::<Option<i64>, _>("workspace_generation"));
    let mut sizes = HashMap::new();
    let mut complete_content = HashMap::new();
    for row in rows {
        let Some(id) = row.get::<Option<Uuid>, _>("id") else {
            continue;
        };
        if let Ok(size) = usize::try_from(row.get::<Option<i64>, _>("size_bytes").unwrap_or(0)) {
            sizes.insert(id, size);
        }
        if state.config.read_path_roundtrip_v1 {
            complete_content.insert(
                id,
                row.get::<Option<String>, _>("content").unwrap_or_default(),
            );
        }
    }
    let mut remaining_chars = token_budget.saturating_mul(4);
    let mut selections = Vec::new();
    let mut complete_ids = Vec::new();
    for (index, candidate) in candidates.iter().take(HYDRATED_DOCUMENT_LIMIT).enumerate() {
        if let Some(annotation) = &candidate.superseded_by {
            let annotation_chars = serde_json::to_string(annotation)
                .map(|value| value.chars().count())
                .unwrap_or(0);
            remaining_chars = remaining_chars.saturating_sub(annotation_chars);
        }
        if remaining_chars == 0 {
            break;
        }
        let Some(size_bytes) = sizes.get(&candidate.entry_id) else {
            continue;
        };
        if *size_bytes <= remaining_chars && *size_bytes <= MAX_OPEN_COMPLETE_SOURCE_CHARS {
            remaining_chars -= *size_bytes;
            complete_ids.push(candidate.entry_id);
            selections.push((index, true, 0_usize));
        } else {
            let excerpt_chars = remaining_chars.min(2_400);
            remaining_chars -= excerpt_chars;
            selections.push((index, false, excerpt_chars));
        }
    }
    if !state.config.read_path_roundtrip_v1 && !complete_ids.is_empty() {
        let complete_rows = sqlx::query(
            r#"
            SELECT entry.id,version.content
            FROM brunn.entries AS entry
            JOIN brunn.entry_versions AS version
              ON version.user_id=entry.user_id
             AND version.entry_id=entry.id
             AND version.version=entry.current_version
            WHERE entry.user_id=$1
              AND entry.id=ANY($2)
              AND entry.deleted_at IS NULL
            "#,
        )
        .bind(auth.user_id.0)
        .bind(&complete_ids)
        .fetch_all(&mut *tx)
        .await?;
        complete_content.extend(complete_rows.into_iter().map(|row| {
            (
                row.get::<Uuid, _>("id"),
                row.get::<Option<String>, _>("content").unwrap_or_default(),
            )
        }));
    }
    tx.commit().await?;
    let mut evidence = Vec::new();
    for (index, complete, excerpt_chars) in selections {
        let candidate = &candidates[index];
        let text = if complete {
            complete_content
                .get(&candidate.entry_id)
                .cloned()
                .unwrap_or_default()
        } else {
            candidate_excerpt(candidate, excerpt_chars)
        };
        if text.is_empty() {
            continue;
        }
        let mut item = serde_json::Map::from_iter([
            (
                "reference".to_owned(),
                Value::String(format!("entry:{}", candidate.entry_id)),
            ),
            ("path".to_owned(), Value::String(candidate.path.clone())),
            ("title".to_owned(), Value::String(candidate.title.clone())),
            ("version".to_owned(), json!(candidate.version)),
            (
                "representation".to_owned(),
                Value::String(
                    if complete {
                        "complete_source"
                    } else {
                        "source_excerpt"
                    }
                    .to_owned(),
                ),
            ),
            ("text".to_owned(), Value::String(text)),
        ]);
        if !candidate.heading.is_empty() {
            item.insert(
                "heading".to_owned(),
                Value::String(candidate.heading.clone()),
            );
        }
        if let Some(superseded_by) = &candidate.superseded_by {
            item.insert("superseded_by".to_owned(), json!(superseded_by));
        }
        evidence.push(Value::Object(item));
    }
    Ok((evidence, generation))
}

fn truncate_candidate_evidence(candidate: &mut Candidate, remaining_chars: &mut usize) -> bool {
    if let Some(annotation) = &candidate.superseded_by {
        let annotation_chars = serde_json::to_string(annotation)
            .map(|value| value.chars().count())
            .unwrap_or(0);
        *remaining_chars = remaining_chars.saturating_sub(annotation_chars);
    }
    let original_excerpt_chars = candidate.excerpt.chars().count();
    let retained_excerpt = truncate_chars(&candidate.excerpt, (*remaining_chars).min(2_400));
    *remaining_chars = remaining_chars.saturating_sub(retained_excerpt.chars().count());
    let mut truncated = retained_excerpt.chars().count() < original_excerpt_chars;
    candidate.excerpt = retained_excerpt;
    if let Some(primary) = candidate.sections.first_mut() {
        primary.excerpt = candidate.excerpt.clone();
    }
    for section in candidate.sections.iter_mut().skip(1) {
        let original_chars = section.excerpt.chars().count();
        let retained = truncate_chars(&section.excerpt, (*remaining_chars).min(2_400));
        *remaining_chars = remaining_chars.saturating_sub(retained.chars().count());
        truncated |= retained.chars().count() < original_chars;
        section.excerpt = retained;
    }
    let mut index = 0_usize;
    candidate.sections.retain(|section| {
        let retain = index == 0 || !section.excerpt.is_empty();
        index += 1;
        retain
    });
    truncated
}

fn candidate_excerpt(candidate: &Candidate, max_chars: usize) -> String {
    let combined = candidate
        .sections
        .iter()
        .map(|section| section.excerpt.as_str())
        .filter(|excerpt| !excerpt.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    truncate_chars(
        if combined.is_empty() {
            &candidate.excerpt
        } else {
            &combined
        },
        max_chars,
    )
}

async fn resolve_entry(
    state: &AppState,
    auth: &AuthContext,
    path: Option<&str>,
    reference: Option<&str>,
) -> ApiResult<EntryRow> {
    resolve_entry_version(state, auth, path, reference, None).await
}

async fn fetch_entry_lookup(
    transaction: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    requested_version: Option<i64>,
    path: Option<&str>,
    entry_id: Option<Uuid>,
    normalized_path: bool,
    include_content: bool,
    include_generation: bool,
) -> Result<Option<PgRow>, sqlx::Error> {
    let mut statement = QueryBuilder::<Postgres>::new(
        r#"
        SELECT entry.id,entry.path,entry.title,entry.kind,entry.media_type,
               entry.current_version,entry.updated_at,
               version.id AS version_id,version.content_sha256,
        "#,
    );
    if include_content {
        statement.push("version.content,");
    } else {
        statement.push("NULL::text AS content,");
    }
    statement.push(
        r#"
               version.object_key,version.object_version_id,version.size_bytes,
               version.metadata,
        "#,
    );
    if include_generation {
        statement.push(
            r#"
               brunn_auth.workspace_generation(entry.user_id) AS workspace_generation
            "#,
        );
    } else {
        statement.push("NULL::bigint AS workspace_generation");
    }
    statement.push(
        r#"
        FROM brunn.entries AS entry
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
        WHERE entry.user_id=
        "#,
    );
    statement.push_bind(user_id);
    if let Some(version) = requested_version {
        statement.push(" AND version.version=").push_bind(version);
    } else {
        statement.push(" AND version.version=entry.current_version AND entry.deleted_at IS NULL");
    }
    if let Some(path) = path {
        if normalized_path {
            statement
                .push(" AND lower(normalize(entry.path, NFC))=")
                .push_bind(portable_path_key(path));
        } else {
            statement
                .push(" AND entry.path=")
                .push_bind(path.to_owned());
        }
    } else {
        statement
            .push(" AND entry.id=")
            .push_bind(entry_id.expect("validated exact entry reference"));
    }
    statement.push(" LIMIT 1");
    statement.build().fetch_optional(&mut **transaction).await
}

async fn resolve_entry_version(
    state: &AppState,
    auth: &AuthContext,
    path: Option<&str>,
    reference: Option<&str>,
    requested_version: Option<i64>,
) -> ApiResult<EntryRow> {
    if requested_version.is_some_and(|version| version <= 0) {
        return Err(ApiError::invalid("entry version must be positive"));
    }
    let checkpoint_path = reference
        .and_then(|value| value.strip_prefix("checkpoint:"))
        .and_then(|value| Uuid::parse_str(value).ok())
        .map(|checkpoint_id| format!(".brunn/checkpoints/{checkpoint_id}.md"));
    let effective_path = path.or(checkpoint_path.as_deref());
    let entry_id = reference
        .and_then(|value| value.strip_prefix("entry:").or(Some(value)))
        .and_then(|value| Uuid::parse_str(value).ok());
    if effective_path.is_none() && entry_id.is_none() {
        return Err(ApiError::invalid(
            "read requires an exact path or entry ref",
        ));
    }
    let mut tx = state.begin_read(auth).await?;
    let row = fetch_entry_lookup(
        &mut tx,
        auth.user_id.0,
        requested_version,
        effective_path,
        entry_id,
        false,
        true,
        state.config.read_path_roundtrip_v1,
    )
    .await?;
    let row = if row.is_none() && effective_path.is_some_and(|path| !path.starts_with(".brunn/")) {
        fetch_entry_lookup(
            &mut tx,
            auth.user_id.0,
            requested_version,
            effective_path,
            entry_id,
            true,
            true,
            state.config.read_path_roundtrip_v1,
        )
        .await?
    } else {
        row
    };
    let row = row.ok_or_else(|| {
        ApiError::not_found("entry_not_found", path.or(reference).unwrap_or("entry"))
    })?;
    tx.commit().await?;
    Ok(entry_row_from_lookup(&row, requested_version))
}

fn entry_row_from_lookup(row: &PgRow, requested_version: Option<i64>) -> EntryRow {
    EntryRow {
        id: row.get("id"),
        path: row.get("path"),
        title: row.get("title"),
        kind: row.get("kind"),
        media_type: row.get("media_type"),
        version: requested_version.unwrap_or_else(|| row.get("current_version")),
        content_sha256: row.get("content_sha256"),
        content: row.get("content"),
        object_key: row.get("object_key"),
        object_version_id: row.get("object_version_id"),
        size_bytes: row.get("size_bytes"),
        metadata: row.get("metadata"),
        updated_at: row.get("updated_at"),
        workspace_generation: row.get("workspace_generation"),
    }
}

fn entry_link_lookup_keys(raw_target: &str) -> ApiResult<Vec<String>> {
    let target = raw_target.trim();
    if target.is_empty() || target.len() > 1_024 || target.chars().any(char::is_control) {
        return Err(ApiError::invalid(
            "link_target must contain 1 to 1024 printable characters",
        ));
    }
    let target = target
        .split(['#', '?', '^', '|'])
        .next()
        .unwrap_or_default()
        .trim();
    let name = target.rsplit('/').next().unwrap_or_default().trim();
    let lowered = name.to_lowercase();
    let stem = if lowered.ends_with(".markdown") {
        &name[..name.len() - ".markdown".len()]
    } else if lowered.ends_with(".md") {
        &name[..name.len() - ".md".len()]
    } else {
        name
    }
    .trim();
    if stem.is_empty() || matches!(stem, "." | "..") {
        return Err(ApiError::invalid("link_target must name an entry"));
    }
    let key = portable_path_key(stem);
    Ok(vec![
        key.clone(),
        format!("{key}.md"),
        format!("{key}.markdown"),
    ])
}

async fn resolve_entry_link_version(
    state: &AppState,
    auth: &AuthContext,
    target: &str,
) -> ApiResult<EntryRow> {
    let filename_keys = entry_link_lookup_keys(target)?;
    let mut tx = state.begin_read(auth).await?;
    let rows = sqlx::query(SIMPLE_ENTRY_LINK_CANDIDATES_SQL)
        .bind(auth.user_id.0)
        .bind(filename_keys)
        .fetch_all(&mut *tx)
        .await?;
    tx.commit().await?;
    match rows.as_slice() {
        [] => Err(ApiError::not_found("entry_link_not_found", target)),
        [row] => Ok(entry_row_from_lookup(row, None)),
        _ => Err(ApiError::public(
            StatusCode::CONFLICT,
            "entry_link_ambiguous",
            "More than one entry matches this link. Use its full path instead.",
        )),
    }
}

async fn resolve_entry_summary(
    state: &AppState,
    auth: &AuthContext,
    path: Option<&str>,
    reference: Option<&str>,
) -> ApiResult<EntryRow> {
    let checkpoint_path = reference
        .and_then(|value| value.strip_prefix("checkpoint:"))
        .and_then(|value| Uuid::parse_str(value).ok())
        .map(|checkpoint_id| format!(".brunn/checkpoints/{checkpoint_id}.md"));
    let effective_path = path.or(checkpoint_path.as_deref());
    let entry_id = reference
        .and_then(|value| value.strip_prefix("entry:").or(Some(value)))
        .and_then(|value| Uuid::parse_str(value).ok());
    if effective_path.is_none() && entry_id.is_none() {
        return Err(ApiError::invalid(
            "checkpoint sources require an exact path or entry ref",
        ));
    }
    let mut tx = state.begin_read(auth).await?;
    let row = fetch_entry_lookup(
        &mut tx,
        auth.user_id.0,
        None,
        effective_path,
        entry_id,
        false,
        false,
        false,
    )
    .await?;
    let row = if row.is_none() && effective_path.is_some_and(|path| !path.starts_with(".brunn/")) {
        fetch_entry_lookup(
            &mut tx,
            auth.user_id.0,
            None,
            effective_path,
            entry_id,
            true,
            false,
            false,
        )
        .await?
    } else {
        row
    };
    let row = row.ok_or_else(|| {
        ApiError::not_found(
            "entry_not_found",
            effective_path.or(reference).unwrap_or("checkpoint source"),
        )
    })?;
    tx.commit().await?;
    Ok(EntryRow {
        id: row.get("id"),
        path: row.get("path"),
        title: row.get("title"),
        kind: row.get("kind"),
        media_type: row.get("media_type"),
        version: row.get("current_version"),
        content_sha256: row.get("content_sha256"),
        content: None,
        object_key: row.get("object_key"),
        object_version_id: row.get("object_version_id"),
        size_bytes: row.get("size_bytes"),
        metadata: row.get("metadata"),
        updated_at: row.get("updated_at"),
        workspace_generation: None,
    })
}

fn render_read(entry: &EntryRow, request: &ReadItem, max_chars: usize) -> ApiResult<Value> {
    let content = entry.content.as_deref().unwrap_or("");
    let view = request.view.as_deref().unwrap_or("full");
    let selected = match view {
        "full" | "current_state" | "current_truth" => content.to_owned(),
        "range" => {
            let start = request.start.unwrap_or(1).max(1);
            let end = request.end.unwrap_or(start + 199).max(start);
            content
                .lines()
                .skip(start - 1)
                .take(end - start + 1)
                .collect::<Vec<_>>()
                .join("\n")
        }
        "outline" => content
            .lines()
            .filter(|line| line.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n"),
        other => {
            return Err(ApiError::invalid(format!(
                "simple read does not support view {other}"
            )));
        }
    };
    let selected_chars = selected.chars().count();
    let mut rendered = serde_json::Map::from_iter([
        (
            "reference".to_owned(),
            Value::String(format!("entry:{}", entry.id)),
        ),
        ("path".to_owned(), Value::String(entry.path.clone())),
        ("title".to_owned(), Value::String(entry.title.clone())),
        ("version".to_owned(), json!(entry.version)),
        (
            "content_hash".to_owned(),
            Value::String(format!("sha256:{}", entry.content_sha256)),
        ),
        (
            "media_type".to_owned(),
            Value::String(entry.media_type.clone()),
        ),
        ("view".to_owned(), Value::String(view.to_owned())),
        (
            "text".to_owned(),
            Value::String(truncate_chars(&selected, max_chars)),
        ),
        ("updated_at".to_owned(), json!(entry.updated_at)),
    ]);
    if selected_chars > max_chars {
        rendered.insert("truncated".to_owned(), Value::Bool(true));
    }
    if entry.metadata != json!({}) && !entry.metadata.is_null() {
        rendered.insert("metadata".to_owned(), entry.metadata.clone());
    }
    Ok(Value::Object(rendered))
}

fn parse_tier_a_history_stage(metadata: &Value) -> ApiResult<Option<TierAHistoryStage>> {
    let Some(value) = metadata.get("_brunn_tier_a_history") else {
        return Ok(None);
    };
    if value.get("format").and_then(Value::as_str) != Some(TIER_A_HISTORY_STAGE_FORMAT) {
        return Err(ApiError::invalid(
            "Tier-A history stage metadata has a missing or unsupported format",
        ));
    }
    let target_lineage_ordinal = value
        .get("target_lineage_ordinal")
        .and_then(Value::as_i64)
        .filter(|value| *value > 0)
        .ok_or_else(|| ApiError::invalid("Tier-A history stage target ordinal must be positive"))?;
    let semantics = match value.get("semantics").and_then(Value::as_str) {
        Some(TIER_A_ORDINARY_HISTORY_SEMANTICS) => TierAHistorySemantics::OrdinaryContentTransition,
        Some(TIER_A_EXACT_HISTORY_SEMANTICS) => {
            TierAHistorySemantics::PreserveIntentionalExactBytesVersion
        }
        _ => {
            return Err(ApiError::invalid(
                "Tier-A history stage semantics are missing or unsupported",
            ));
        }
    };
    if semantics == TierAHistorySemantics::PreserveIntentionalExactBytesVersion
        && target_lineage_ordinal < 2
    {
        return Err(ApiError::invalid(
            "an intentional exact-byte history version must target ordinal 2 or later",
        ));
    }
    Ok(Some(TierAHistoryStage {
        target_lineage_ordinal,
        semantics,
    }))
}

fn validate_tier_a_history_request(
    path: &str,
    metadata: &Value,
    expected_version: Option<i64>,
    evaluation_api_enabled: bool,
) -> ApiResult<Option<TierAHistoryStage>> {
    let Some(stage) = parse_tier_a_history_stage(metadata)? else {
        return Ok(None);
    };
    if metadata
        .get("_brunn_import")
        .and_then(|value| value.get("format"))
        .and_then(Value::as_str)
        != Some(WORKSPACE_IMPORT_FORMAT)
    {
        return Err(ApiError::invalid(
            "Tier-A history replay requires workspace import metadata",
        ));
    }
    let expected = expected_version
        .ok_or_else(|| ApiError::invalid("Tier-A history replay requires expected_version"))?;
    if expected < 0 || expected.checked_add(1) != Some(stage.target_lineage_ordinal) {
        return Err(ApiError::conflict(
            "tier_a_history_ordinal_mismatch",
            "Tier-A target ordinal must be exactly expected_version + 1",
            json!({
                "path": path,
                "expected_version": expected,
                "target_lineage_ordinal": stage.target_lineage_ordinal
            }),
        ));
    }
    if stage.semantics == TierAHistorySemantics::PreserveIntentionalExactBytesVersion {
        if !evaluation_api_enabled {
            return Err(ApiError::invalid(
                "intentional exact-byte history preservation requires an evaluation-only stack",
            ));
        }
        if path.starts_with(".brunn/") {
            return Err(ApiError::invalid(
                "intentional exact-byte history preservation is not supported for managed paths",
            ));
        }
    }
    Ok(Some(stage))
}

fn tier_a_exact_history_action(
    stage: Option<TierAHistoryStage>,
    current_version: Option<i64>,
    current_content_matches: bool,
    current_deleted: bool,
    current_metadata: Option<&Value>,
    proposed_metadata: &Value,
) -> ApiResult<TierAExactHistoryAction> {
    let Some(stage) = stage.filter(|value| {
        value.semantics == TierAHistorySemantics::PreserveIntentionalExactBytesVersion
    }) else {
        return Ok(TierAExactHistoryAction::NotRequested);
    };
    let target = stage.target_lineage_ordinal;
    let Some(current) = current_version else {
        return Err(ApiError::conflict(
            "tier_a_history_gap",
            "Tier-A exact-history replay has no immediate predecessor",
            json!({"target_lineage_ordinal": target}),
        ));
    };
    if current < target - 1 {
        return Err(ApiError::conflict(
            "tier_a_history_gap",
            "Tier-A exact-history replay is missing one or more predecessor versions",
            json!({
                "current_version": current,
                "target_lineage_ordinal": target
            }),
        ));
    }
    if current > target {
        return Err(ApiError::conflict(
            "tier_a_history_ahead",
            "Tier-A exact-history replay is already ahead of the requested target",
            json!({
                "current_version": current,
                "target_lineage_ordinal": target
            }),
        ));
    }
    if current_deleted || !current_content_matches {
        return Err(ApiError::conflict(
            "tier_a_exact_history_identity_mismatch",
            "Tier-A exact-history replay requires the immediate current version to have identical bytes",
            json!({
                "current_version": current,
                "target_lineage_ordinal": target
            }),
        ));
    }
    if current == target - 1 {
        return Ok(TierAExactHistoryAction::Append);
    }
    if current == target {
        if current_metadata == Some(proposed_metadata)
            && parse_tier_a_history_stage(proposed_metadata)? == Some(stage)
        {
            return Ok(TierAExactHistoryAction::Idempotent);
        }
        return Err(ApiError::conflict(
            "tier_a_exact_history_identity_mismatch",
            "the existing Tier-A target ordinal does not match the requested identity",
            json!({
                "current_version": current,
                "target_lineage_ordinal": target
            }),
        ));
    }
    Err(ApiError::conflict(
        "tier_a_history_ordinal_mismatch",
        "Tier-A exact-history replay is neither at the predecessor nor target ordinal",
        json!({
            "current_version": current,
            "target_lineage_ordinal": target
        }),
    ))
}

pub(crate) async fn prepare_markdown(
    state: &AppState,
    request: WriteRequest,
) -> ApiResult<PreparedMarkdown> {
    validate_path(&request.path)?;
    if request.media_type != "text/markdown" && request.media_type != "text/plain" {
        return Err(ApiError::invalid(
            "workspace.write accepts Markdown or plain text; upload other files as binaries",
        ));
    }
    let mut metadata = match request.metadata {
        Value::Object(values) => values,
        Value::Null => serde_json::Map::new(),
        value => serde_json::Map::from_iter([("value".to_owned(), value)]),
    };
    if let Some(idempotency_key) = request.idempotency_key {
        validate_idempotency_key(&idempotency_key)?;
        metadata.insert(
            "_brunn_idempotency_hash".to_owned(),
            Value::String(hex::encode(Sha256::digest(idempotency_key.as_bytes()))),
        );
    }
    let metadata = Value::Object(metadata);
    let conversation_candidate =
        crate::messaging_protocol::is_conversation_candidate(&request.path, &metadata);
    if conversation_candidate {
        if request.content.len() > crate::messaging_protocol::MAX_CANONICAL_CONVERSATION_BYTES {
            return Err(ApiError::public(
                StatusCode::PAYLOAD_TOO_LARGE,
                "conversation_entry_too_large",
                "canonical conversation Markdown is limited to 12 MiB",
            ));
        }
        if !crate::messaging_protocol::is_workspace_import(&metadata) {
            return Err(ApiError::invalid(
                "canonical conversation entries may be written only by workspace import",
            ));
        }
    } else if request.content.len() > MAX_WRITE_BYTES {
        return Err(ApiError::public(
            StatusCode::PAYLOAD_TOO_LARGE,
            "entry_too_large",
            "Markdown writes are limited to 4 MiB; use a binary entry and companion Markdown",
        ));
    }
    if conversation_candidate {
        crate::messaging_protocol::validate_conversation_entry(
            &request.path,
            &metadata,
            &request.content,
        )
        .map_err(|error| ApiError::invalid(error.to_string()))?
        .ok_or_else(|| ApiError::invalid("canonical conversation metadata is required"))?;
    }
    let normalized = normalize_document(&request.path, &request.content);
    let frontmatter = if state.config.supersession_demotion || state.config.intention_ledger {
        parse_frontmatter(&request.content)
    } else {
        DerivedFrontmatter::default()
    };
    let task_entry = crate::task_service::validate_task_entry(&request.path, &metadata)?;
    let chunks = if task_entry || conversation_candidate {
        Vec::new()
    } else {
        normalized.chunks
    };
    let embeddings = vec![None; chunks.len()];
    let tier_a_history_stage = validate_tier_a_history_request(
        &request.path,
        &metadata,
        request.expected_version,
        state.config.evaluation_api_enabled,
    )?;
    Ok(PreparedMarkdown {
        entry_id_hint: None,
        path: request.path,
        title: normalized.title,
        content_sha256: normalized
            .content_hash
            .trim_start_matches("sha256:")
            .to_owned(),
        content: request.content,
        media_type: request.media_type,
        metadata,
        chunks,
        embeddings,
        expected_version: request.expected_version,
        tier_a_history_stage,
        frontmatter,
        force_new_version: task_entry || conversation_candidate,
    })
}

pub(crate) fn prepare_task_markdown_for_update(
    path: String,
    content: String,
    metadata: Value,
    expected_version: i64,
) -> ApiResult<PreparedMarkdown> {
    validate_path(&path)?;
    if content.len() > MAX_WRITE_BYTES {
        return Err(ApiError::public(
            StatusCode::PAYLOAD_TOO_LARGE,
            "entry_too_large",
            "Task Markdown is limited to 4 MiB",
        ));
    }
    if !crate::task_service::validate_task_entry(&path, &metadata)? {
        return Err(ApiError::invalid(
            "managed task updates require canonical task.v1 metadata",
        ));
    }
    let normalized = normalize_document(&path, &content);
    Ok(PreparedMarkdown {
        entry_id_hint: None,
        path,
        title: normalized.title,
        content_sha256: normalized
            .content_hash
            .trim_start_matches("sha256:")
            .to_owned(),
        content,
        media_type: markdown_media_type(),
        metadata,
        chunks: Vec::new(),
        embeddings: Vec::new(),
        expected_version: Some(expected_version),
        tier_a_history_stage: None,
        frontmatter: DerivedFrontmatter::default(),
        force_new_version: true,
    })
}

async fn commit_markdown(
    state: &AppState,
    auth: &AuthContext,
    prepared: PreparedMarkdown,
) -> ApiResult<Value> {
    let started = Instant::now();
    let path = prepared.path.clone();
    let content_sha256 = prepared.content_sha256.clone();
    let frontmatter = if state.config.supersession_demotion {
        prepared.frontmatter.clone()
    } else {
        DerivedFrontmatter::default()
    };
    let warnings = if frontmatter.supersedes.is_empty() {
        Vec::new()
    } else {
        let generation = current_generation(state, auth).await?;
        match feature_snapshot(state, auth, generation).await? {
            Some(snapshot) => supersession_warnings(&frontmatter, &snapshot),
            None => Vec::new(),
        }
    };
    let mut tx = state.begin_write(auth).await?;
    require_local_publish_lock(
        &mut tx,
        format!(
            "simple-entry:{}:{}",
            auth.user_id.0,
            portable_path_key(&prepared.path)
        ),
        state.config.read_path_roundtrip_v1,
    )
    .await?;
    let result = upsert_markdown_in_tx(
        &mut tx,
        auth.user_id.0,
        Some(auth.credential_id.0),
        prepared,
    )
    .await?;
    let generation = match result.generation {
        Some(value) => value,
        None => max_generation_in_tx(&mut tx, auth.user_id.0).await?,
    };
    tx.commit().await?;
    state.workspace_features.invalidate(auth.user_id.0).await;
    metrics::histogram!("simple.write.duration_ms")
        .record(started.elapsed().as_secs_f64() * 1_000.0);
    metrics::histogram!("simple.write.changed_entries").record(if result.no_op {
        0.0
    } else {
        1.0
    });
    let mut receipt = json!({
        "entry_ref": format!("entry:{}", result.entry_id),
        "version_ref": result.version_id.map(|id| format!("entry-version:{id}")),
        "path": path,
        "version": result.version,
        "content_hash": format!("sha256:{content_sha256}"),
        "workspace_generation": generation,
        "search_status": if result.no_op {
            "unchanged"
        } else if result.metadata_only {
            "content_unchanged"
        } else {
            "lexical_ready_semantic_queued"
        },
        "metadata_only": result.metadata_only,
        "no_op": result.no_op
    });
    if !warnings.is_empty() {
        receipt["supersession_warnings"] = json!(warnings);
    }
    Ok(receipt)
}

pub(crate) async fn write_markdown_as_worker(
    state: &AppState,
    user_id: Uuid,
    path: String,
    content: String,
    metadata: Value,
    expected_version: Option<i64>,
    guard_entry: Option<(Uuid, i64)>,
) -> ApiResult<Option<Value>> {
    let prepared = prepare_markdown(
        state,
        WriteRequest {
            path,
            content,
            media_type: markdown_media_type(),
            expected_version,
            idempotency_key: None,
            metadata,
        },
    )
    .await?;
    let result_path = prepared.path.clone();
    let content_sha256 = prepared.content_sha256.clone();
    let pool = state
        .admin_pool
        .as_ref()
        .ok_or_else(|| ApiError::configuration("the worker requires DATABASE_URL_ADMIN"))?;
    let mut tx = pool.begin().await?;
    require_local_publish_lock(
        &mut tx,
        format!(
            "simple-entry:{user_id}:{}",
            portable_path_key(&prepared.path)
        ),
        state.config.read_path_roundtrip_v1,
    )
    .await?;
    if let Some((entry_id, expected_entry_version)) = guard_entry {
        let current_entry_version = sqlx::query_scalar::<_, Option<i64>>(
            r#"
            SELECT current_version
            FROM brunn.entries
            WHERE user_id=$1 AND id=$2 AND deleted_at IS NULL
            FOR SHARE
            "#,
        )
        .bind(user_id)
        .bind(entry_id)
        .fetch_optional(&mut *tx)
        .await?
        .flatten();
        if current_entry_version != Some(expected_entry_version) {
            tx.commit().await?;
            return Ok(None);
        }
    }
    let result = upsert_markdown_in_tx(&mut tx, user_id, None, prepared).await?;
    let generation = match result.generation {
        Some(value) => value,
        None => max_generation_in_tx(&mut tx, user_id).await?,
    };
    tx.commit().await?;
    state.workspace_features.invalidate(user_id).await;
    Ok(Some(json!({
        "entry_ref": format!("entry:{}", result.entry_id),
        "version_ref": result.version_id.map(|id| format!("entry-version:{id}")),
        "path": result_path,
        "version": result.version,
        "content_hash": format!("sha256:{content_sha256}"),
        "workspace_generation": generation,
        "metadata_only": result.metadata_only,
        "no_op": result.no_op
    })))
}

pub(crate) async fn fetch_locked_markdown_entry(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    path: &str,
) -> ApiResult<Option<PgRow>> {
    Ok(sqlx::query(
        r#"
        SELECT entry.id,entry.kind,entry.current_version,entry.deleted_at,
               version.id AS version_id,version.content_sha256,version.metadata
        FROM brunn.entries AS entry
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
         AND version.version=entry.current_version
        WHERE entry.user_id=$1
          AND lower(normalize(entry.path, NFC))=$2
        FOR UPDATE OF entry
        "#,
    )
    .bind(user_id)
    .bind(portable_path_key(path))
    .fetch_optional(&mut **tx)
    .await?)
}

async fn sync_managed_entry_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    entry_id: Uuid,
    entry_version: i64,
    path: &str,
    metadata: &Value,
) -> ApiResult<()> {
    crate::task_service::sync_managed_entry_in_tx(
        tx,
        user_id,
        entry_id,
        entry_version,
        path,
        metadata,
    )
    .await?;
    crate::messaging_service::sync_managed_entry_in_tx(
        tx,
        user_id,
        entry_id,
        entry_version,
        path,
        metadata,
    )
    .await
}

pub(crate) async fn upsert_markdown_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    created_by_credential_id: Option<Uuid>,
    prepared: PreparedMarkdown,
) -> ApiResult<MarkdownUpsertResult> {
    let checkpoint_import = is_portable_checkpoint_import(&prepared.path, &prepared.metadata);
    if checkpoint_import {
        validate_imported_checkpoint_parent_in_tx(tx, user_id, &prepared.path, &prepared.metadata)
            .await?;
    }
    let proposed_entry_id = prepared.entry_id_hint.unwrap_or_else(Uuid::now_v7);
    let inserted_entry_id = sqlx::query_scalar::<_, Uuid>(
        r#"
        INSERT INTO brunn.entries (
          id,user_id,path,title,kind,media_type,current_version
        ) VALUES ($1,$2,$3,$4,'markdown',$5,0)
        ON CONFLICT (user_id,(lower(normalize(path, NFC)))) DO NOTHING
        RETURNING id
        "#,
    )
    .bind(proposed_entry_id)
    .bind(user_id)
    .bind(&prepared.path)
    .bind(&prepared.title)
    .bind(&prepared.media_type)
    .fetch_optional(&mut **tx)
    .await?;
    let existing = if inserted_entry_id.is_some() {
        None
    } else {
        fetch_locked_markdown_entry(tx, user_id, &prepared.path).await?
    };
    if existing
        .as_ref()
        .is_some_and(|row| row.get::<String, _>("kind") != "markdown")
    {
        return Err(ApiError::conflict(
            "entry_kind_conflict",
            "a binary entry already uses this Markdown path",
            json!({"path": prepared.path}),
        ));
    }
    let exact_history_action = tier_a_exact_history_action(
        prepared.tier_a_history_stage,
        existing
            .as_ref()
            .map(|row| row.get::<i64, _>("current_version")),
        existing
            .as_ref()
            .is_some_and(|row| row.get::<String, _>("content_sha256") == prepared.content_sha256),
        existing
            .as_ref()
            .is_some_and(|row| row.get::<Option<DateTime<Utc>>, _>("deleted_at").is_some()),
        existing
            .as_ref()
            .map(|row| row.get::<Value, _>("metadata"))
            .as_ref(),
        &prepared.metadata,
    )?;
    if exact_history_action == TierAExactHistoryAction::Idempotent {
        let row = existing
            .as_ref()
            .expect("an idempotent Tier-A history target has an existing entry");
        sync_managed_entry_in_tx(
            tx,
            user_id,
            row.get("id"),
            row.get("current_version"),
            &prepared.path,
            &row.get("metadata"),
        )
        .await?;
        return Ok(MarkdownUpsertResult {
            entry_id: row.get("id"),
            version: row.get("current_version"),
            version_id: Some(row.get("version_id")),
            generation: None,
            no_op: true,
            metadata_only: false,
        });
    }
    if let Some(row) = &existing
        && is_checkpoint_path(&prepared.path)
    {
        if row.get::<String, _>("content_sha256") == prepared.content_sha256
            && row.get::<Option<DateTime<Utc>>, _>("deleted_at").is_none()
        {
            sync_managed_entry_in_tx(
                tx,
                user_id,
                row.get("id"),
                row.get("current_version"),
                &prepared.path,
                &row.get("metadata"),
            )
            .await?;
            return Ok(MarkdownUpsertResult {
                entry_id: row.get("id"),
                version: row.get("current_version"),
                version_id: Some(row.get("version_id")),
                generation: None,
                no_op: true,
                metadata_only: false,
            });
        }
        return Err(ApiError::conflict(
            "checkpoint_immutable",
            "checkpoint entries are immutable; create a new checkpoint instead",
            json!({"path": prepared.path}),
        ));
    }
    if let Some(row) = &existing
        && row.get::<String, _>("content_sha256") == prepared.content_sha256
        && exact_history_action != TierAExactHistoryAction::Append
        && !prepared.force_new_version
    {
        let was_deleted = row.get::<Option<DateTime<Utc>>, _>("deleted_at").is_some();
        let current_metadata = row.get::<Value, _>("metadata");
        let metadata_only = (prepared.metadata.get("_brunn_import").is_some()
            || prepared.metadata.get("portable").is_some())
            && current_metadata != prepared.metadata;
        if metadata_only {
            sqlx::query(
                r#"
                UPDATE brunn.entry_versions
                SET metadata=$4
                WHERE user_id=$1 AND entry_id=$2 AND version=$3
                "#,
            )
            .bind(user_id)
            .bind(row.get::<Uuid, _>("id"))
            .bind(row.get::<i64, _>("current_version"))
            .bind(&prepared.metadata)
            .execute(&mut **tx)
            .await?;
        }
        if was_deleted {
            sqlx::query(
                r#"
                UPDATE brunn.entries
                SET path=$3,title=$4,media_type=$5,deleted_at=NULL,
                    updated_at=clock_timestamp()
                WHERE user_id=$1 AND id=$2
                "#,
            )
            .bind(user_id)
            .bind(row.get::<Uuid, _>("id"))
            .bind(&prepared.path)
            .bind(&prepared.title)
            .bind(&prepared.media_type)
            .execute(&mut **tx)
            .await?;
            sqlx::query("DELETE FROM brunn.search_chunks WHERE user_id=$1 AND entry_id=$2")
                .bind(user_id)
                .bind(row.get::<Uuid, _>("id"))
                .execute(&mut **tx)
                .await?;
            insert_chunks(
                tx,
                user_id,
                row.get("id"),
                row.get("version_id"),
                &prepared.path,
                &prepared.chunks,
                &prepared.embeddings,
            )
            .await?;
            if prepared.embeddings.iter().any(Option::is_none) {
                sqlx::query(
                    r#"
                    INSERT INTO brunn.jobs (user_id,kind,payload)
                    VALUES ($1,'embed_entry',$2)
                    "#,
                )
                .bind(user_id)
                .bind(json!({
                    "entry_id": row.get::<Uuid, _>("id"),
                    "version": row.get::<i64, _>("current_version")
                }))
                .execute(&mut **tx)
                .await?;
            }
        } else if metadata_only {
            sqlx::query(
                r#"
                UPDATE brunn.entries
                SET updated_at=clock_timestamp()
                WHERE user_id=$1 AND id=$2
                "#,
            )
            .bind(user_id)
            .bind(row.get::<Uuid, _>("id"))
            .execute(&mut **tx)
            .await?;
        }
        let changed = was_deleted || metadata_only;
        let generation = if changed {
            Some(
                sqlx::query_scalar::<_, i64>(
                    r#"
                    INSERT INTO brunn.workspace_changes (
                      user_id,entry_id,entry_version,operation,path,content_sha256
                    ) VALUES ($1,$2,$3,'update',$4,$5)
                    RETURNING generation
                    "#,
                )
                .bind(user_id)
                .bind(row.get::<Uuid, _>("id"))
                .bind(row.get::<i64, _>("current_version"))
                .bind(&prepared.path)
                .bind(&prepared.content_sha256)
                .fetch_one(&mut **tx)
                .await?,
            )
        } else {
            None
        };
        let projection_metadata = if metadata_only {
            &prepared.metadata
        } else {
            &current_metadata
        };
        sync_managed_entry_in_tx(
            tx,
            user_id,
            row.get("id"),
            row.get("current_version"),
            &prepared.path,
            projection_metadata,
        )
        .await?;
        return Ok(MarkdownUpsertResult {
            entry_id: row.get("id"),
            version: row.get("current_version"),
            version_id: Some(row.get("version_id")),
            generation,
            no_op: !changed,
            metadata_only,
        });
    }
    if let Some(expected) = prepared.expected_version {
        let actual = existing
            .as_ref()
            .map(|row| row.get::<i64, _>("current_version"))
            .unwrap_or(0);
        let restoring_deleted = existing.as_ref().is_some_and(|row| {
            row.get::<Option<DateTime<Utc>>, _>("deleted_at").is_some() && expected == 0
        });
        if actual != expected && !restoring_deleted {
            return Err(ApiError::conflict(
                "entry_version_conflict",
                "the entry changed since it was read",
                json!({
                    "path": prepared.path,
                    "expected_version": expected,
                    "actual_version": actual
                }),
            ));
        }
    }
    let (entry_id, version, operation) = match existing {
        Some(row) => (
            row.get::<Uuid, _>("id"),
            row.get::<i64, _>("current_version") + 1,
            "update",
        ),
        None => (proposed_entry_id, 1, "create"),
    };
    let version_id = Uuid::now_v7();
    sqlx::query(
        r#"
        INSERT INTO brunn.entry_versions (
          id,user_id,entry_id,version,content_sha256,content,size_bytes,
          metadata,created_by_credential_id
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        "#,
    )
    .bind(version_id)
    .bind(user_id)
    .bind(entry_id)
    .bind(version)
    .bind(&prepared.content_sha256)
    .bind(&prepared.content)
    .bind(i64::try_from(prepared.content.len()).unwrap_or(i64::MAX))
    .bind(&prepared.metadata)
    .bind(created_by_credential_id)
    .execute(&mut **tx)
    .await?;
    sqlx::query(
        r#"
        UPDATE brunn.entries
        SET path=$3,title=$4,media_type=$5,current_version=$6,
            updated_at=clock_timestamp(),deleted_at=NULL
        WHERE user_id=$1 AND id=$2
        "#,
    )
    .bind(user_id)
    .bind(entry_id)
    .bind(&prepared.path)
    .bind(&prepared.title)
    .bind(&prepared.media_type)
    .bind(version)
    .execute(&mut **tx)
    .await?;
    sqlx::query("DELETE FROM brunn.search_chunks WHERE user_id=$1 AND entry_id=$2")
        .bind(user_id)
        .bind(entry_id)
        .execute(&mut **tx)
        .await?;
    insert_chunks(
        tx,
        user_id,
        entry_id,
        version_id,
        &prepared.path,
        &prepared.chunks,
        &prepared.embeddings,
    )
    .await?;
    let generation = sqlx::query_scalar::<_, i64>(
        r#"
        INSERT INTO brunn.workspace_changes (
          user_id,entry_id,entry_version,operation,path,content_sha256
        ) VALUES ($1,$2,$3,$4,$5,$6)
        RETURNING generation
        "#,
    )
    .bind(user_id)
    .bind(entry_id)
    .bind(version)
    .bind(operation)
    .bind(&prepared.path)
    .bind(&prepared.content_sha256)
    .fetch_one(&mut **tx)
    .await?;
    let final_metadata = if checkpoint_import {
        let metadata = rebase_imported_checkpoint_metadata(prepared.metadata.clone(), generation);
        sqlx::query(
            r#"
            UPDATE brunn.entry_versions
            SET metadata=$4
            WHERE user_id=$1 AND entry_id=$2 AND version=$3
            "#,
        )
        .bind(user_id)
        .bind(entry_id)
        .bind(version)
        .bind(&metadata)
        .execute(&mut **tx)
        .await?;
        metadata
    } else {
        prepared.metadata.clone()
    };
    sync_managed_entry_in_tx(
        tx,
        user_id,
        entry_id,
        version,
        &prepared.path,
        &final_metadata,
    )
    .await?;
    if prepared.embeddings.iter().any(Option::is_none) {
        sqlx::query(
            r#"
            INSERT INTO brunn.jobs (user_id,kind,payload)
            VALUES ($1,'embed_entry',$2)
            "#,
        )
        .bind(user_id)
        .bind(json!({"entry_id": entry_id, "version": version}))
        .execute(&mut **tx)
        .await?;
    }
    Ok(MarkdownUpsertResult {
        entry_id,
        version,
        version_id: Some(version_id),
        generation: Some(generation),
        no_op: false,
        metadata_only: false,
    })
}

async fn insert_chunks(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    entry_id: Uuid,
    version_id: Uuid,
    path: &str,
    chunks: &[DocumentChunk],
    embeddings: &[Option<Vector>],
) -> ApiResult<()> {
    if chunks.is_empty() {
        return Ok(());
    }
    let pairs = chunks.iter().zip(embeddings).collect::<Vec<_>>();
    for batch in pairs.chunks(CHUNK_INSERT_BATCH_SIZE) {
        let mut builder = QueryBuilder::<Postgres>::new(
            "INSERT INTO brunn.search_chunks \
             (id,user_id,entry_id,entry_version_id,chunk_index,path,heading,content,token_estimate,embedding) ",
        );
        builder.push_values(batch.iter().copied(), |mut row, (chunk, embedding)| {
            row.push_bind(Uuid::now_v7())
                .push_bind(user_id)
                .push_bind(entry_id)
                .push_bind(version_id)
                .push_bind(i32::try_from(chunk.ordinal).unwrap_or(i32::MAX))
                .push_bind(path)
                .push_bind(&chunk.heading)
                .push_bind(&chunk.content)
                .push_bind(i32::try_from(chunk.estimated_tokens).unwrap_or(i32::MAX))
                .push_bind(embedding.clone());
        });
        builder.build().execute(&mut **tx).await?;
    }
    Ok(())
}

async fn changes_since(
    state: &AppState,
    auth: &AuthContext,
    since: i64,
    limit: usize,
) -> ApiResult<ChangePage> {
    let mut tx = state.begin_read(auth).await?;
    let (rows, workspace_generation) = if state.config.read_path_roundtrip_v1 {
        let rows = sqlx::query(
            r#"
            WITH generation AS (
              SELECT coalesce(max(change.generation),0) AS workspace_generation
              FROM brunn.workspace_changes AS change
              WHERE change.user_id=$1
            ), page AS MATERIALIZED (
              SELECT change.generation,change.operation,change.path,
                     change.entry_version,change.content_sha256,change.recorded_at
              FROM brunn.workspace_changes AS change
              WHERE change.user_id=$1 AND change.generation>$2
              ORDER BY change.generation
              LIMIT $3
            )
            SELECT generation.workspace_generation,
                   page.generation,page.operation,page.path,page.entry_version,
                   page.content_sha256,page.recorded_at
            FROM generation
            LEFT JOIN page ON true
            ORDER BY page.generation NULLS LAST
            "#,
        )
        .bind(auth.user_id.0)
        .bind(since)
        .bind(i64::try_from(limit.saturating_add(1)).unwrap_or(i64::MAX))
        .fetch_all(&mut *tx)
        .await?;
        let generation = rows
            .first()
            .and_then(|row| row.get::<Option<i64>, _>("workspace_generation"));
        (rows, generation)
    } else {
        let rows = sqlx::query(
            r#"
            SELECT generation,operation,path,entry_version,content_sha256,recorded_at
            FROM brunn.workspace_changes
            WHERE user_id=$1 AND generation>$2
            ORDER BY generation
            LIMIT $3
            "#,
        )
        .bind(auth.user_id.0)
        .bind(since)
        .bind(i64::try_from(limit.saturating_add(1)).unwrap_or(i64::MAX))
        .fetch_all(&mut *tx)
        .await?;
        (rows, None)
    };
    tx.commit().await?;
    let mut changes = rows
        .into_iter()
        .filter_map(|row| {
            let generation = if state.config.read_path_roundtrip_v1 {
                row.get::<Option<i64>, _>("generation")?
            } else {
                row.get::<i64, _>("generation")
            };
            Some(json!({
                "generation": generation,
                "operation": row.get::<String, _>("operation"),
                "path": row.get::<String, _>("path"),
                "version": row.get::<i64, _>("entry_version"),
                "content_hash": format!("sha256:{}", row.get::<String, _>("content_sha256")),
                "recorded_at": row.get::<DateTime<Utc>, _>("recorded_at")
            }))
        })
        .collect::<Vec<_>>();
    let truncated = changes.len() > limit;
    if truncated {
        changes.truncate(limit);
    }
    let next_generation = truncated
        .then(|| changes.last().and_then(|value| value.get("generation")))
        .flatten()
        .and_then(Value::as_i64);
    Ok(ChangePage {
        changes,
        truncated,
        next_generation,
        workspace_generation,
    })
}

async fn read_checkpoint(
    state: &AppState,
    auth: &AuthContext,
    checkpoint_ref: &str,
) -> ApiResult<Value> {
    let entry = resolve_entry(state, auth, None, Some(checkpoint_ref)).await?;
    let metadata = entry
        .metadata
        .get("client")
        .filter(|value| value.is_object())
        .unwrap_or(&entry.metadata);
    if metadata.get("kind").and_then(Value::as_str) != Some("checkpoint") {
        return Err(ApiError::invalid(
            "resume_checkpoint_ref does not identify a checkpoint entry",
        ));
    }
    Ok(json!({
        "checkpoint_id": checkpoint_ref,
        "path": entry.path,
        "workspace_generation": metadata.get("workspace_generation"),
        "pinned_workspace_generation": metadata
            .get("pinned_workspace_generation")
            .or_else(|| metadata.get("workspace_generation")),
        "resulting_workspace_generation": metadata
            .get("resulting_workspace_generation"),
        "text": entry.content,
        "source_entries": metadata.get("source_entries")
    }))
}

fn checkpoint_sources(checkpoint: Option<&Value>) -> Vec<CheckpointSource> {
    checkpoint
        .and_then(|value| value.get("source_entries"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|source| {
            let entry_id = source
                .get("entry_ref")
                .and_then(Value::as_str)?
                .strip_prefix("entry:")?
                .parse::<Uuid>()
                .ok()?;
            let pinned_hash = source
                .get("content_hash")
                .or_else(|| source.get("pinned_sha256"))
                .and_then(Value::as_str)?;
            let pinned_sha256 = pinned_hash
                .strip_prefix("sha256:")
                .unwrap_or(pinned_hash)
                .to_ascii_lowercase();
            Some(CheckpointSource {
                entry_id,
                path: source.get("path")?.as_str()?.to_owned(),
                pinned_version: source.get("version")?.as_i64()?,
                pinned_sha256,
            })
        })
        .collect()
}

fn changed_checkpoint_sources(
    checkpoint: Option<&Value>,
    changes: &[Value],
) -> (Vec<CheckpointSource>, Vec<CheckpointSource>) {
    let changed = changes
        .iter()
        .filter_map(|change| change.get("path").and_then(Value::as_str))
        .map(portable_path_key)
        .collect::<HashSet<_>>();
    let mut selected = Vec::new();
    let mut overflow = Vec::new();
    for source in checkpoint_sources(checkpoint)
        .into_iter()
        .filter(|source| changed.contains(&portable_path_key(&source.path)))
    {
        if selected.len() < RESUME_DELTA_SOURCE_LIMIT {
            selected.push(source);
        } else {
            overflow.push(source);
        }
    }
    (selected, overflow)
}

async fn materialize_resume_deltas(
    state: &AppState,
    auth: &AuthContext,
    checkpoint: Option<&Value>,
    changes: &[Value],
    evidence_chars: usize,
) -> ApiResult<ResumeDeltaBatch> {
    let (selected, overflow) = changed_checkpoint_sources(checkpoint, changes);
    if selected.is_empty() {
        return Ok(ResumeDeltaBatch {
            leads: overflow
                .into_iter()
                .map(|source| resume_delta_pointer(&source, None, "source limit"))
                .collect(),
            ..ResumeDeltaBatch::default()
        });
    }
    let pairs = load_resume_version_pairs(state, auth, &selected).await?;
    let mut remaining = RESUME_DELTA_TOTAL_CHARS.min(evidence_chars);
    let mut result = ResumeDeltaBatch::default();
    for pair in pairs {
        let Some(before) = pair.before.as_deref() else {
            result.leads.push(resume_delta_pointer(
                &pair.source,
                Some(pair.current_version),
                "non-text pinned source",
            ));
            continue;
        };
        let Some(after) = pair.after.as_deref() else {
            result.leads.push(resume_delta_pointer(
                &pair.source,
                Some(pair.current_version),
                "non-text current source",
            ));
            continue;
        };
        let before_chars = before.chars().count();
        let after_chars = after.chars().count();
        if before_chars <= RESUME_DELTA_WHOLE_PAIR_CHARS
            && after_chars <= RESUME_DELTA_WHOLE_PAIR_CHARS
        {
            let charged = before_chars.saturating_add(after_chars);
            if charged > remaining {
                result.leads.push(resume_delta_pointer(
                    &pair.source,
                    Some(pair.current_version),
                    "delta character budget",
                ));
                continue;
            }
            result.deltas.push(render_resume_delta(
                &pair,
                "whole_pair",
                Some(before.to_owned()),
                Some(after.to_owned()),
                None,
                false,
            ));
            remaining -= charged;
            result.charged_chars += charged;
            continue;
        }
        if remaining == 0 {
            result.leads.push(resume_delta_pointer(
                &pair.source,
                Some(pair.current_version),
                "delta character budget",
            ));
            continue;
        }
        let diff = unified_line_diff(&pair.source.path, before, after);
        let cap = RESUME_DELTA_SOURCE_CHARS.min(remaining);
        let diff_chars = diff.chars().count();
        let truncated = diff_chars > cap;
        let rendered_diff = truncate_chars(&diff, cap);
        let charged = rendered_diff.chars().count();
        result.deltas.push(render_resume_delta(
            &pair,
            "unified_diff",
            None,
            None,
            Some(rendered_diff),
            truncated,
        ));
        remaining -= charged;
        result.charged_chars += charged;
    }
    result.leads.extend(
        overflow
            .into_iter()
            .map(|source| resume_delta_pointer(&source, None, "source limit")),
    );
    Ok(result)
}

async fn load_resume_version_pairs(
    state: &AppState,
    auth: &AuthContext,
    sources: &[CheckpointSource],
) -> ApiResult<Vec<ResumeVersionPair>> {
    let entry_ids = sources
        .iter()
        .map(|source| source.entry_id)
        .collect::<Vec<_>>();
    let pinned_versions = sources
        .iter()
        .map(|source| source.pinned_version)
        .collect::<Vec<_>>();
    let mut tx = state.begin_read(auth).await?;
    let rows = sqlx::query(
        r#"
        WITH requested AS (
          SELECT entry_id,pinned_version,position
          FROM unnest($2::uuid[],$3::bigint[]) WITH ORDINALITY
            AS source(entry_id,pinned_version,position)
        )
        SELECT requested.position,entry.path,entry.current_version,
               pinned.content_sha256 AS pinned_sha256,pinned.content AS before,
               current.content_sha256 AS current_sha256,current.content AS after
        FROM requested
        LEFT JOIN brunn.entries AS entry
          ON entry.user_id=$1 AND entry.id=requested.entry_id
        LEFT JOIN brunn.entry_versions AS pinned
          ON pinned.user_id=$1
         AND pinned.entry_id=requested.entry_id
         AND pinned.version=requested.pinned_version
        LEFT JOIN brunn.entry_versions AS current
          ON current.user_id=entry.user_id
         AND current.entry_id=entry.id
         AND current.version=entry.current_version
        ORDER BY requested.position
        "#,
    )
    .bind(auth.user_id.0)
    .bind(&entry_ids)
    .bind(&pinned_versions)
    .fetch_all(&mut *tx)
    .await?;
    tx.commit().await?;
    if rows.len() != sources.len() {
        return Err(resume_lineage_error(
            "the checkpoint source batch did not return every pinned source",
            json!({"expected": sources.len(), "actual": rows.len()}),
        ));
    }
    rows.into_iter()
        .zip(sources)
        .map(|(row, source)| {
            let path = row.try_get::<Option<String>, _>("path")?.ok_or_else(|| {
                resume_lineage_error(
                    "a checkpoint source entry is missing",
                    json!({"path": source.path, "entry_ref": format!("entry:{}", source.entry_id)}),
                )
            })?;
            if portable_path_key(&path) != portable_path_key(&source.path) {
                return Err(resume_lineage_error(
                    "a checkpoint source now resolves to a different path",
                    json!({"pinned_path": source.path, "current_path": path}),
                ));
            }
            let pinned_sha256 = row
                .try_get::<Option<String>, _>("pinned_sha256")?
                .ok_or_else(|| {
                    resume_lineage_error(
                        "a checkpoint-pinned version is missing",
                        json!({"path": source.path, "version": source.pinned_version}),
                    )
                })?;
            if pinned_sha256 != source.pinned_sha256 {
                return Err(resume_lineage_error(
                    "checkpoint metadata does not match the pinned version hash",
                    json!({
                        "path": source.path,
                        "version": source.pinned_version,
                        "checkpoint_hash": format!("sha256:{}", source.pinned_sha256),
                        "stored_hash": format!("sha256:{pinned_sha256}")
                    }),
                ));
            }
            let before = row.try_get::<Option<String>, _>("before")?;
            if let Some(content) = before.as_deref() {
                verify_resume_content_hash(source, source.pinned_version, &pinned_sha256, content)?;
            }
            let current_version = row
                .try_get::<Option<i64>, _>("current_version")?
                .ok_or_else(|| {
                    resume_lineage_error(
                        "the current checkpoint source entry is missing",
                        json!({"path": source.path}),
                    )
                })?;
            let current_sha256 = row
                .try_get::<Option<String>, _>("current_sha256")?
                .ok_or_else(|| {
                    resume_lineage_error(
                        "the current checkpoint source version is missing",
                        json!({"path": source.path, "version": current_version}),
                    )
                })?;
            let after = row.try_get::<Option<String>, _>("after")?;
            if let Some(content) = after.as_deref() {
                verify_resume_content_hash(source, current_version, &current_sha256, content)?;
            }
            Ok(ResumeVersionPair {
                source: source.clone(),
                current_version,
                current_sha256,
                before,
                after,
            })
        })
        .collect()
}

fn verify_resume_content_hash(
    source: &CheckpointSource,
    version: i64,
    expected: &str,
    content: &str,
) -> ApiResult<()> {
    let actual = hex::encode(Sha256::digest(content.as_bytes()));
    if actual == expected {
        return Ok(());
    }
    Err(resume_lineage_error(
        "a checkpoint source version failed content-hash verification",
        json!({
            "path": source.path,
            "version": version,
            "expected_hash": format!("sha256:{expected}"),
            "actual_hash": format!("sha256:{actual}")
        }),
    ))
}

fn resume_lineage_error(message: impl Into<String>, details: Value) -> ApiError {
    ApiError::conflict("checkpoint_lineage_error", message, details)
}

fn render_resume_delta(
    pair: &ResumeVersionPair,
    mode: &str,
    before: Option<String>,
    after: Option<String>,
    diff: Option<String>,
    truncated: bool,
) -> Value {
    let mut value = json!({
        "path": pair.source.path,
        "pinned_version": pair.source.pinned_version,
        "pinned_sha256": format!("sha256:{}", pair.source.pinned_sha256),
        "current_version": pair.current_version,
        "current_sha256": format!("sha256:{}", pair.current_sha256),
        "mode": mode
    });
    if let Some(before) = before {
        value["before"] = Value::String(before);
    }
    if let Some(after) = after {
        value["after"] = Value::String(after);
    }
    if let Some(diff) = diff {
        value["diff"] = Value::String(diff);
    }
    if truncated {
        value["truncated"] = Value::Bool(true);
    }
    value
}

fn resume_delta_pointer(
    source: &CheckpointSource,
    current_version: Option<i64>,
    reason: &str,
) -> Value {
    let transition = current_version
        .map(|version| format!("version {} \u{2192} {version}", source.pinned_version))
        .unwrap_or_else(|| format!("version {} \u{2192} current", source.pinned_version));
    json!({
        "reference": format!("entry:{}", source.entry_id),
        "path": source.path,
        "version": current_version,
        "annotation": format!("changed since checkpoint: {transition}"),
        "delta_omitted_reason": reason
    })
}

fn unified_line_diff(path: &str, before: &str, after: &str) -> String {
    let left = before.lines().collect::<Vec<_>>();
    let right = after.lines().collect::<Vec<_>>();
    let prefix = left
        .iter()
        .zip(&right)
        .take_while(|(before, after)| before == after)
        .count();
    let suffix = left[prefix..]
        .iter()
        .rev()
        .zip(right[prefix..].iter().rev())
        .take_while(|(before, after)| before == after)
        .count();
    let context_start = prefix.saturating_sub(3);
    let left_changed_end = left.len().saturating_sub(suffix);
    let right_changed_end = right.len().saturating_sub(suffix);
    let left_end = (left_changed_end + 3).min(left.len());
    let right_end = (right_changed_end + 3).min(right.len());
    let left_count = left_end.saturating_sub(context_start);
    let right_count = right_end.saturating_sub(context_start);
    let mut output = format!(
        "--- a/{path}\n+++ b/{path}\n@@ -{},{} +{},{} @@\n",
        context_start + 1,
        left_count,
        context_start + 1,
        right_count
    );
    for line in &left[context_start..prefix] {
        output.push(' ');
        output.push_str(line);
        output.push('\n');
    }
    for line in &left[prefix..left_changed_end] {
        output.push('-');
        output.push_str(line);
        output.push('\n');
    }
    for line in &right[prefix..right_changed_end] {
        output.push('+');
        output.push_str(line);
        output.push('\n');
    }
    for line in &right[right_changed_end..right_end] {
        output.push(' ');
        output.push_str(line);
        output.push('\n');
    }
    output
}

async fn resolve_checkpoint_sources_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    checkpoint_state: &Value,
    source_refs: &[String],
) -> ApiResult<(Vec<EntryRow>, Option<i64>)> {
    let explicit = source_refs
        .iter()
        .filter(|candidate| {
            !candidate.starts_with("source_episode:") && !candidate.starts_with("evidence:")
        })
        .map(|candidate| {
            let entry_id = if let Some(value) = candidate.strip_prefix("entry:") {
                Some(Uuid::parse_str(value).map_err(|_| {
                    ApiError::invalid("checkpoint sources require an exact path or entry ref")
                })?)
            } else {
                None
            };
            Ok::<_, ApiError>((candidate.clone(), entry_id))
        })
        .collect::<ApiResult<Vec<_>>>()?;

    let mut inferred = Vec::new();
    collect_markdown_paths(checkpoint_state, &mut inferred);
    inferred.sort();
    inferred.dedup();
    inferred.truncate(64);

    let entry_ids = explicit
        .iter()
        .filter_map(|(_, entry_id)| *entry_id)
        .collect::<Vec<_>>();
    let mut paths = explicit
        .iter()
        .filter_map(|(candidate, entry_id)| entry_id.is_none().then_some(candidate.clone()))
        .chain(inferred.iter().cloned())
        .collect::<Vec<_>>();
    paths.sort();
    paths.dedup();
    let normalized_path_keys = paths
        .iter()
        .filter(|path| !path.starts_with(".brunn/"))
        .map(|path| portable_path_key(path))
        .collect::<Vec<_>>();

    // The path lanes cannot use indexes beneath the RLS barrier (LIKE and
    // normalize() are not leakproof), so a SECURITY DEFINER resolver maps
    // them to entry ids first and the visible query stays on leakproof
    // id equality.
    let mut candidate_ids = entry_ids.clone();
    if !paths.is_empty() || !normalized_path_keys.is_empty() {
        let resolved: Vec<Uuid> =
            sqlx::query_scalar("SELECT brunn_auth.resolve_entry_ids_by_path($1,$2,$3)")
                .bind(user_id)
                .bind(&paths)
                .bind(&normalized_path_keys)
                .fetch_one(&mut **tx)
                .await?;
        candidate_ids.extend(resolved);
    }
    candidate_ids.sort();
    candidate_ids.dedup();
    if candidate_ids.is_empty() {
        return Ok((Vec::new(), None));
    }
    let rows = sqlx::query(
        r#"
        SELECT entry.id,entry.path,entry.title,entry.kind,entry.media_type,
               entry.current_version,entry.updated_at,
               version.id AS version_id,version.content_sha256,
               NULL::text AS content,version.object_key,version.object_version_id,
               version.size_bytes,version.metadata,
               brunn_auth.workspace_generation($1) AS pinned_workspace_generation
        FROM brunn.entries AS entry
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
         AND version.version=entry.current_version
        WHERE entry.user_id=$1
          AND entry.deleted_at IS NULL
          AND entry.id=ANY($2)
        "#,
    )
    .bind(user_id)
    .bind(&candidate_ids)
    .fetch_all(&mut **tx)
    .await?;
    let pinned_workspace_generation = rows
        .first()
        .map(|row| row.get::<i64, _>("pinned_workspace_generation"));

    let entries = rows
        .into_iter()
        .map(|row| EntryRow {
            id: row.get("id"),
            path: row.get("path"),
            title: row.get("title"),
            kind: row.get("kind"),
            media_type: row.get("media_type"),
            version: row.get("current_version"),
            content_sha256: row.get("content_sha256"),
            content: None,
            object_key: row.get("object_key"),
            object_version_id: row.get("object_version_id"),
            size_bytes: row.get("size_bytes"),
            metadata: row.get("metadata"),
            updated_at: row.get("updated_at"),
            workspace_generation: None,
        })
        .collect::<Vec<_>>();
    let by_id = entries
        .iter()
        .map(|entry| (entry.id, entry.clone()))
        .collect::<HashMap<_, _>>();
    let by_exact_path = entries
        .iter()
        .map(|entry| (entry.path.clone(), entry.clone()))
        .collect::<HashMap<_, _>>();
    let mut by_normalized_path = HashMap::new();
    for entry in &entries {
        by_normalized_path
            .entry(portable_path_key(&entry.path))
            .or_insert_with(|| entry.clone());
    }

    let lookup_path = |path: &str| {
        by_exact_path.get(path).cloned().or_else(|| {
            (!path.starts_with(".brunn/"))
                .then(|| by_normalized_path.get(&portable_path_key(path)).cloned())
                .flatten()
        })
    };
    let mut resolved = Vec::new();
    let mut seen = HashSet::new();
    for (candidate, entry_id) in explicit {
        let entry = entry_id
            .and_then(|id| by_id.get(&id).cloned())
            .or_else(|| lookup_path(&candidate))
            .ok_or_else(|| ApiError::not_found("entry_not_found", &candidate))?;
        if seen.insert(entry.id) {
            resolved.push(entry);
        }
    }
    let remaining = 64_usize.saturating_sub(resolved.len());
    for candidate in inferred.into_iter().take(remaining) {
        if let Some(entry) = lookup_path(&candidate)
            && seen.insert(entry.id)
        {
            resolved.push(entry);
        }
    }
    Ok((resolved, pinned_workspace_generation))
}

fn collect_markdown_paths(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            if looks_like_path(text) {
                output.push(text.to_owned());
            }
            for path in path_hints(text) {
                output.push(path);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_markdown_paths(item, output);
            }
        }
        Value::Object(object) => {
            for item in object.values() {
                collect_markdown_paths(item, output);
            }
        }
        _ => {}
    }
}

fn render_checkpoint_markdown(
    checkpoint_id: Uuid,
    generation: i64,
    request: &CheckpointRequest,
    source_entries: &[EntryRow],
) -> ApiResult<String> {
    let objective = request
        .state
        .get("objective")
        .and_then(Value::as_str)
        .unwrap_or("Resume durable work");
    let project = request
        .state
        .get("project")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !project.is_empty() {
        crate::task_service::validate_project_slug(project)?;
    }
    let mut output = format!(
        "---\nbrunn_kind: checkpoint\ncheckpoint_id: {checkpoint_id}\n\
         workspace_generation: {generation}\nsession_id: {}\nparent_checkpoint_id: {}\nproject: {}\n---\n\n\
         # Checkpoint: {}\n\n",
        request.session_id,
        request.parent_checkpoint_id.as_deref().unwrap_or(""),
        project,
        objective
    );
    for (field, heading) in [
        ("current_state", "Current state"),
        ("decisions", "Decisions"),
        ("open_questions", "Open questions"),
        ("next_actions", "Next actions"),
        ("artifacts", "Artifacts"),
    ] {
        output.push_str(&format!("## {heading}\n\n"));
        match request.state.get(field) {
            Some(Value::Array(items)) => {
                for item in items {
                    output.push_str(&format!(
                        "- {}\n",
                        item.as_str()
                            .map(ToOwned::to_owned)
                            .unwrap_or_else(|| item.to_string())
                    ));
                }
            }
            Some(Value::String(text)) => output.push_str(&format!("- {text}\n")),
            _ => output.push_str("- None recorded.\n"),
        }
        output.push('\n');
    }
    output.push_str("## Exact source references\n\n");
    if source_entries.is_empty() {
        output.push_str("- None recorded.\n");
    } else {
        for entry in source_entries {
            output.push_str(&format!(
                "- `{}` | version {} | `sha256:{}`\n",
                entry.path, entry.version, entry.content_sha256
            ));
        }
    }
    let external_refs = request
        .source_refs
        .iter()
        .filter(|value| value.starts_with("source_episode:") || value.starts_with("evidence:"))
        .collect::<Vec<_>>();
    if !external_refs.is_empty() {
        output.push_str("\n## External source references\n\n");
        for reference in external_refs {
            output.push_str(&format!("- `{reference}`\n"));
        }
    }
    Ok(output)
}

fn deterministic_checkpoint_id(request: &CheckpointRequest) -> Uuid {
    let identity = json!({
        "session_id": request.session_id,
        "parent_checkpoint_id": request.parent_checkpoint_id,
        "state": request.state,
        "source_refs": request.source_refs,
        "idempotency_key": request.idempotency_key
    });
    let digest = Sha256::digest(
        serde_json::to_vec(&identity).unwrap_or_else(|_| Uuid::now_v7().as_bytes().to_vec()),
    );
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    Uuid::from_bytes(bytes)
}

// Preserve the path identity understood by the pre-receipt binary so a
// rolling deploy or rollback recognizes a checkpoint written by this binary.
// Durable operation identity itself lives in workspace_idempotency_receipts.
fn checkpoint_entry_id_for_new_write(request: &CheckpointRequest) -> Uuid {
    deterministic_checkpoint_id(request)
}

fn implicit_checkpoint_idempotency_key(request: &CheckpointRequest) -> String {
    format!("implicit:{}", checkpoint_entry_id_for_new_write(request))
}

fn is_reserved_implicit_checkpoint_key(key: &str) -> bool {
    key.strip_prefix("implicit:")
        .is_some_and(|suffix| Uuid::parse_str(suffix).is_ok())
}

fn validate_checkpoint_request(request: &CheckpointRequest) -> ApiResult<(String, String)> {
    if request.session_id.is_empty()
        || request.session_id.len() > 256
        || request.session_id.chars().any(char::is_control)
    {
        return Err(ApiError::invalid(
            "checkpoint session_id must contain 1 to 256 printable characters",
        ));
    }
    if let Some(idempotency_key) = request.idempotency_key.as_deref() {
        validate_idempotency_key(idempotency_key)?;
        if is_reserved_implicit_checkpoint_key(idempotency_key) {
            return Err(ApiError::invalid(
                "checkpoint idempotency keys matching implicit:<uuid> are reserved",
            ));
        }
    }
    if !request.state.is_object() {
        return Err(ApiError::invalid("checkpoint state must be an object"));
    }
    if let Some(project) = request
        .state
        .get("project")
        .filter(|value| !value.is_null())
    {
        let project = project
            .as_str()
            .ok_or_else(|| ApiError::invalid("checkpoint state.project must be a string"))?;
        crate::task_service::validate_project_slug(project)?;
    }
    if request.source_refs.len() > 64 {
        return Err(ApiError::invalid(
            "checkpoint source_refs are limited to 64 exact references",
        ));
    }
    if request.source_refs.iter().any(|reference| {
        reference.is_empty() || reference.len() > 4_096 || reference.chars().any(char::is_control)
    }) {
        return Err(ApiError::invalid(
            "checkpoint source_refs must contain 1 to 4096 printable characters",
        ));
    }
    if request
        .parent_checkpoint_id
        .as_ref()
        .is_some_and(|reference| {
            reference.is_empty() || reference.len() > 256 || reference.chars().any(char::is_control)
        })
    {
        return Err(ApiError::invalid(
            "checkpoint parent_checkpoint_id must contain 1 to 256 printable characters",
        ));
    }
    let identity = json!({
        "schema": "brunn-simple-checkpoint-request@v1",
        "parent_checkpoint_id": request.parent_checkpoint_id,
        "state": request.state,
        "source_refs": request.source_refs
    });
    let canonical = canonical_json(&identity)?;
    if canonical.len() > MAX_WRITE_BYTES {
        return Err(ApiError::public(
            StatusCode::PAYLOAD_TOO_LARGE,
            "checkpoint_too_large",
            "checkpoint requests are limited to 4 MiB",
        ));
    }
    let effective_idempotency_key = request
        .idempotency_key
        .clone()
        .unwrap_or_else(|| implicit_checkpoint_idempotency_key(request));
    Ok((
        effective_idempotency_key,
        hex::encode(Sha256::digest(canonical.as_bytes())),
    ))
}

fn checkpoint_envelope(
    session_id: &str,
    receipt: Value,
    replayed: bool,
) -> ApiResult<WorkspaceEnvelope<Value>> {
    let resulting_generation = receipt
        .get("resulting_workspace_generation")
        .or_else(|| receipt.get("workspace_generation"))
        .and_then(Value::as_i64)
        .ok_or_else(|| {
            ApiError::Internal("checkpoint receipt is missing its resulting generation".to_owned())
        })?;
    let mut envelope = WorkspaceEnvelope::complete(receipt);
    envelope.status = ResponseStatus::Committed;
    envelope.session_id = Some(session_id.to_owned());
    envelope.corpus_revision = Some(format!("generation:{resulting_generation}"));
    // A replay intentionally returns the same logical response as the first
    // successful call. Per-request SQL counts are diagnostics, not receipt
    // data, so only a fresh checkpoint reports its own count.
    if replayed {
        envelope.query_count = None;
    }
    Ok(envelope)
}

async fn lock_checkpoint_idempotency(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    idempotency_key: &str,
) -> ApiResult<()> {
    let key_hash = hex::encode(Sha256::digest(idempotency_key.as_bytes()));
    let lock_key = format!("simple-idempotency:{user_id}:checkpoint:{key_hash}");
    sqlx::query_scalar::<_, ()>("SELECT pg_advisory_xact_lock(hashtextextended($1,0))")
        .bind(lock_key)
        .fetch_one(&mut **tx)
        .await?;
    Ok(())
}

async fn replay_checkpoint_receipt_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    idempotency_key: &str,
    request_hash: &str,
) -> ApiResult<Option<Value>> {
    let row = sqlx::query(
        r#"
        SELECT request_hash,receipt
        FROM brunn.workspace_idempotency_receipts
        WHERE user_id=$1 AND operation_kind='checkpoint' AND idempotency_key=$2
        "#,
    )
    .bind(user_id)
    .bind(idempotency_key)
    .fetch_optional(&mut **tx)
    .await?;
    let Some(row) = row else {
        return Ok(None);
    };
    let existing_hash: String = row.get("request_hash");
    if existing_hash != request_hash {
        return Err(checkpoint_idempotency_conflict(idempotency_key));
    }
    Ok(Some(row.get("receipt")))
}

fn checkpoint_idempotency_conflict(idempotency_key: &str) -> ApiError {
    ApiError::conflict(
        "idempotency_conflict",
        "checkpoint idempotency key was already used for a different request",
        json!({
            "operation_kind": "checkpoint",
            "idempotency_key": idempotency_key,
            "idempotency_key_reused": true
        }),
    )
}

async fn persist_checkpoint_receipt_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    credential_id: Option<Uuid>,
    idempotency_key: &str,
    request_hash: &str,
    entry_id: Uuid,
    pinned_generation: i64,
    resulting_generation: i64,
    receipt: &Value,
) -> ApiResult<()> {
    sqlx::query(
        r#"
        INSERT INTO brunn.workspace_idempotency_receipts (
          user_id,operation_kind,idempotency_key,request_hash,
          checkpoint_entry_id,pinned_workspace_generation,
          resulting_workspace_generation,receipt,created_by_credential_id
        ) VALUES ($1,'checkpoint',$2,$3,$4,$5,$6,$7,$8)
        "#,
    )
    .bind(user_id)
    .bind(idempotency_key)
    .bind(request_hash)
    .bind(entry_id)
    .bind(pinned_generation)
    .bind(resulting_generation)
    .bind(receipt)
    .bind(credential_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

fn set_checkpoint_resulting_generation(
    mut metadata: Value,
    pinned_generation: i64,
    resulting_generation: i64,
) -> Value {
    let target = if metadata.get("client").is_some_and(Value::is_object) {
        metadata
            .get_mut("client")
            .and_then(Value::as_object_mut)
            .expect("checked checkpoint client metadata")
    } else {
        metadata
            .as_object_mut()
            .expect("prepared Markdown metadata is always an object")
    };
    target.insert("workspace_generation".to_owned(), json!(pinned_generation));
    target.insert(
        "pinned_workspace_generation".to_owned(),
        json!(pinned_generation),
    );
    target.insert(
        "resulting_workspace_generation".to_owned(),
        json!(resulting_generation),
    );
    metadata
}

#[allow(clippy::too_many_arguments)]
fn checkpoint_receipt_value(
    checkpoint_ref: &str,
    path: &str,
    entry_id: Uuid,
    version_id: Uuid,
    version: i64,
    content_sha256: &str,
    pinned_generation: i64,
    resulting_generation: i64,
    source_entries: Vec<Value>,
) -> Value {
    json!({
        "checkpoint_id": checkpoint_ref,
        "checkpoint_ref": checkpoint_ref,
        "path": path,
        // Compatibility: callers historically treated workspace_generation as
        // the generation after the checkpoint write.
        "workspace_generation": resulting_generation,
        "pinned_workspace_generation": pinned_generation,
        "resulting_workspace_generation": resulting_generation,
        "source_entries": source_entries,
        "write": {
            "entry_ref": format!("entry:{entry_id}"),
            "version_ref": format!("entry-version:{version_id}"),
            "path": path,
            "version": version,
            "content_hash": format!("sha256:{content_sha256}"),
            "workspace_generation": resulting_generation,
            "pinned_workspace_generation": pinned_generation,
            "resulting_workspace_generation": resulting_generation,
            "search_status": "lexical_ready_semantic_queued",
            "metadata_only": false,
            "no_op": false
        }
    })
}

#[allow(clippy::too_many_arguments)]
async fn commit_checkpoint_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    credential_id: Option<Uuid>,
    idempotency_key: &str,
    request_hash: &str,
    checkpoint_ref: &str,
    path: &str,
    pinned_generation: i64,
    source_entries: Vec<Value>,
    prepared: PreparedMarkdown,
) -> ApiResult<CheckpointWriteResult> {
    let committed_bytes = u64::try_from(prepared.content.len()).unwrap_or(u64::MAX);
    let content_sha256 = prepared.content_sha256.clone();
    let initial_metadata = prepared.metadata.clone();
    require_local_publish_lock(
        tx,
        format!("simple-entry:{user_id}:{}", portable_path_key(path)),
        true,
    )
    .await?;
    let result = upsert_markdown_in_tx(tx, user_id, credential_id, prepared).await?;
    let resulting_generation = match result.generation {
        Some(generation) => generation,
        None => sqlx::query_scalar::<_, Option<i64>>(
            r#"
            SELECT max(generation)
            FROM brunn.workspace_changes
            WHERE user_id=$1 AND entry_id=$2 AND entry_version=$3
            "#,
        )
        .bind(user_id)
        .bind(result.entry_id)
        .bind(result.version)
        .fetch_one(&mut **tx)
        .await?
        .ok_or_else(|| {
            ApiError::Internal(
                "checkpoint entry exists without a workspace change receipt".to_owned(),
            )
        })?,
    };
    let version_id = result.version_id.ok_or_else(|| {
        ApiError::Internal("checkpoint write did not produce a version reference".to_owned())
    })?;
    let metadata = set_checkpoint_resulting_generation(
        initial_metadata,
        pinned_generation,
        resulting_generation,
    );
    sqlx::query(
        r#"
        UPDATE brunn.entry_versions
        SET metadata=$4
        WHERE user_id=$1 AND entry_id=$2 AND version=$3
        "#,
    )
    .bind(user_id)
    .bind(result.entry_id)
    .bind(result.version)
    .bind(metadata)
    .execute(&mut **tx)
    .await?;
    let receipt = checkpoint_receipt_value(
        checkpoint_ref,
        path,
        result.entry_id,
        version_id,
        result.version,
        &content_sha256,
        pinned_generation,
        resulting_generation,
        source_entries,
    );
    persist_checkpoint_receipt_in_tx(
        tx,
        user_id,
        credential_id,
        idempotency_key,
        request_hash,
        result.entry_id,
        pinned_generation,
        resulting_generation,
        &receipt,
    )
    .await?;
    Ok(CheckpointWriteResult {
        receipt,
        committed_bytes: if result.no_op { 0 } else { committed_bytes },
        created: !result.no_op,
    })
}

async fn adopt_legacy_checkpoint_receipt_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    credential_id: Option<Uuid>,
    idempotency_key: &str,
    request_hash: &str,
    request: &CheckpointRequest,
) -> ApiResult<Option<CheckpointWriteResult>> {
    let idempotency_hash = request
        .idempotency_key
        .as_ref()
        .map(|_| hex::encode(Sha256::digest(idempotency_key.as_bytes())));
    let implicit_path = format!(
        ".brunn/checkpoints/{}.md",
        deterministic_checkpoint_id(request)
    );
    // LIKE and the metadata hash expression cannot reach their indexes
    // beneath the RLS barrier, so a SECURITY DEFINER resolver bounds the
    // candidate set by id first; every original predicate is re-checked on
    // that bounded set.
    let adoption_ids: Vec<Uuid> =
        sqlx::query_scalar("SELECT brunn_auth.resolve_checkpoint_adoption_ids($1,$2,$3)")
            .bind(user_id)
            .bind(idempotency_hash.as_deref())
            .bind(&implicit_path)
            .fetch_one(&mut **tx)
            .await?;
    if adoption_ids.is_empty() {
        return Ok(None);
    }
    let rows = sqlx::query(
        r#"
        SELECT entry.id,entry.path,entry.current_version,
               version.id AS version_id,version.content_sha256,version.metadata
        FROM brunn.entries AS entry
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
         AND version.version=entry.current_version
        WHERE entry.user_id=$1
          AND entry.id=ANY($4)
          AND entry.deleted_at IS NULL
          AND entry.path LIKE '.brunn/checkpoints/%'
          AND version.metadata->>'kind'='checkpoint'
          AND (
            ($2::text IS NOT NULL
             AND version.metadata->>'_brunn_idempotency_hash'=$2)
            OR ($2::text IS NULL AND entry.path=$3)
          )
        ORDER BY entry.created_at,entry.id
        "#,
    )
    .bind(user_id)
    .bind(idempotency_hash)
    .bind(implicit_path)
    .bind(&adoption_ids)
    .fetch_all(&mut **tx)
    .await?;
    if rows.is_empty() {
        return Ok(None);
    }
    // Explicit-key adoption scans by the legacy key hash, then rebuilds each
    // retired ID with the session stored on that immutable row and the
    // caller's canonical payload. Missing-key adoption is deliberately
    // session-scoped and starts from the exact legacy path. Both modes reject a
    // row whose original deterministic request identity does not match. If the
    // old implementation wrote one explicit-key operation from multiple
    // sessions, retain the earliest equivalent receipt.
    let mut matching_rows = Vec::with_capacity(rows.len());
    for row in &rows {
        let metadata: Value = row.get("metadata");
        let client_metadata = metadata
            .get("client")
            .filter(|value| value.is_object())
            .unwrap_or(&metadata);
        let Some(legacy_session_id) = client_metadata.get("session_id").and_then(Value::as_str)
        else {
            return Err(checkpoint_idempotency_conflict(idempotency_key));
        };
        let mut legacy_request = request.clone();
        legacy_request.session_id = legacy_session_id.to_owned();
        let expected_checkpoint_id = deterministic_checkpoint_id(&legacy_request);
        let expected_path = format!(".brunn/checkpoints/{expected_checkpoint_id}.md");
        if row.get::<String, _>("path") != expected_path {
            return Err(checkpoint_idempotency_conflict(idempotency_key));
        }
        matching_rows.push((row, expected_checkpoint_id, expected_path));
    }
    let (row, expected_checkpoint_id, expected_path) = matching_rows
        .into_iter()
        .next()
        .expect("non-empty legacy rows produced a match");
    let entry_id: Uuid = row.get("id");
    let version: i64 = row.get("current_version");
    let version_id: Uuid = row.get("version_id");
    let metadata: Value = row.get("metadata");
    let client_metadata = metadata
        .get("client")
        .filter(|value| value.is_object())
        .unwrap_or(&metadata);
    if client_metadata.get("kind").and_then(Value::as_str) != Some("checkpoint") {
        return Err(checkpoint_idempotency_conflict(idempotency_key));
    }
    let pinned_generation = client_metadata
        .get("pinned_workspace_generation")
        .or_else(|| client_metadata.get("workspace_generation"))
        .and_then(Value::as_i64)
        .ok_or_else(|| {
            ApiError::Internal("legacy checkpoint is missing its pinned generation".to_owned())
        })?;
    let resulting_generation = sqlx::query_scalar::<_, Option<i64>>(
        r#"
        SELECT max(generation)
        FROM brunn.workspace_changes
        WHERE user_id=$1 AND entry_id=$2 AND entry_version=$3
        "#,
    )
    .bind(user_id)
    .bind(entry_id)
    .bind(version)
    .fetch_one(&mut **tx)
    .await?
    .ok_or_else(|| {
        ApiError::Internal("legacy checkpoint is missing its workspace change".to_owned())
    })?;
    let checkpoint_ref = format!("checkpoint:{expected_checkpoint_id}");
    let receipt = checkpoint_receipt_value(
        &checkpoint_ref,
        &expected_path,
        entry_id,
        version_id,
        version,
        &row.get::<String, _>("content_sha256"),
        pinned_generation,
        resulting_generation,
        client_metadata
            .get("source_entries")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
    );
    persist_checkpoint_receipt_in_tx(
        tx,
        user_id,
        credential_id,
        idempotency_key,
        request_hash,
        entry_id,
        pinned_generation,
        resulting_generation,
        &receipt,
    )
    .await?;
    Ok(Some(CheckpointWriteResult {
        receipt,
        committed_bytes: 0,
        created: false,
    }))
}

async fn validate_checkpoint_parent_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    checkpoint_ref: &str,
) -> ApiResult<()> {
    let checkpoint_path = checkpoint_ref
        .strip_prefix("checkpoint:")
        .and_then(|value| Uuid::parse_str(value).ok())
        .map(|checkpoint_id| format!(".brunn/checkpoints/{checkpoint_id}.md"));
    let entry_id = checkpoint_ref
        .strip_prefix("entry:")
        .or(Some(checkpoint_ref))
        .and_then(|value| Uuid::parse_str(value).ok());
    if checkpoint_path.is_none() && entry_id.is_none() {
        return Err(ApiError::invalid(
            "parent_checkpoint_id must be a checkpoint or entry ref",
        ));
    }
    let row = fetch_entry_lookup(
        tx,
        user_id,
        None,
        checkpoint_path.as_deref(),
        entry_id,
        false,
        false,
        false,
    )
    .await?
    .ok_or_else(|| ApiError::not_found("entry_not_found", checkpoint_ref))?;
    let metadata: Value = row.get("metadata");
    let metadata = metadata
        .get("client")
        .filter(|value| value.is_object())
        .unwrap_or(&metadata);
    if metadata.get("kind").and_then(Value::as_str) != Some("checkpoint") {
        return Err(ApiError::invalid(
            "parent_checkpoint_id does not identify a checkpoint entry",
        ));
    }
    Ok(())
}

fn entry_reference(entry: &EntryRow) -> Value {
    json!({
        "entry_ref": format!("entry:{}", entry.id),
        "path": entry.path,
        "version": entry.version,
        "content_hash": format!("sha256:{}", entry.content_sha256)
    })
}

fn path_hints(query: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let mut unquoted = query.to_owned();
    for delimiter in ['`', '"'] {
        let mut remainder = query;
        while let Some(start) = remainder.find(delimiter) {
            remainder = &remainder[start + delimiter.len_utf8()..];
            let Some(end) = remainder.find(delimiter) else {
                break;
            };
            let candidate = &remainder[..end];
            if looks_like_path(candidate) {
                paths.push(candidate.to_owned());
            }
            remainder = &remainder[end + delimiter.len_utf8()..];
        }
        let mut inside = false;
        unquoted = unquoted
            .chars()
            .map(|character| {
                if character == delimiter {
                    inside = !inside;
                    ' '
                } else if inside {
                    ' '
                } else {
                    character
                }
            })
            .collect();
    }
    paths.extend(
        unquoted
            .split_whitespace()
            .map(|token| {
                token.trim_matches(|character: char| {
                    matches!(
                        character,
                        '`' | '"' | '\'' | ',' | ';' | ':' | '(' | ')' | '[' | ']'
                    )
                })
            })
            .filter(|token| looks_like_path(token))
            .map(ToOwned::to_owned),
    );
    paths.sort();
    paths.dedup();
    paths.truncate(16);
    paths
}

fn search_anchors(query: &str) -> Vec<String> {
    let mut anchors = Vec::new();
    for delimiter in ['`', '"'] {
        let mut remainder = query;
        while let Some(start) = remainder.find(delimiter) {
            remainder = &remainder[start + delimiter.len_utf8()..];
            let Some(end) = remainder.find(delimiter) else {
                break;
            };
            let candidate = remainder[..end].trim();
            if (3..=256).contains(&candidate.chars().count()) {
                anchors.push(candidate.to_owned());
            }
            remainder = &remainder[end + delimiter.len_utf8()..];
        }
    }
    anchors.extend(
        query
            .split_whitespace()
            .map(|token| {
                token.trim_matches(|character: char| {
                    !character.is_alphanumeric() && !matches!(character, '-' | '_' | '/' | '.')
                })
            })
            .filter(|token| {
                (6..=256).contains(&token.chars().count())
                    && looks_like_unquoted_search_anchor(token)
            })
            .map(ToOwned::to_owned),
    );
    anchors.retain(|anchor| !anchor.eq_ignore_ascii_case(query.trim()));
    let mut deduplicated = Vec::new();
    for anchor in anchors {
        if !deduplicated
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(&anchor))
        {
            deduplicated.push(anchor);
        }
    }
    deduplicated.truncate(2);
    deduplicated
}

fn looks_like_unquoted_search_anchor(value: &str) -> bool {
    let has_digit = value.chars().any(|character| character.is_ascii_digit());
    value.contains('_')
        || (has_digit
            && value
                .chars()
                .any(|character| matches!(character, '-' | '/')))
}

fn ranked_lexical_fallback_queries(query: &str) -> Vec<String> {
    let terms = lexical_term_candidates(query, 16);
    let mut pairs = terms
        .windows(2)
        .take(8)
        .enumerate()
        .map(|(index, terms)| {
            let pair = format!("{} {}", terms[0].text, terms[1].text);
            let length = pair
                .chars()
                .filter(|character| character.is_alphanumeric())
                .count();
            let distinctive = terms.iter().filter(|term| term.distinctive).count();
            (distinctive, length, index, pair)
        })
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| {
        right.0.cmp(&left.0).then_with(|| {
            if left.0 > 0 {
                left.2.cmp(&right.2)
            } else {
                right.1.cmp(&left.1).then_with(|| left.2.cmp(&right.2))
            }
        })
    });
    pairs.into_iter().map(|(_, _, _, pair)| pair).collect()
}

fn bounded_lexical_fallback_queries(query: &str) -> Vec<String> {
    let mut fallbacks = ranked_lexical_fallback_queries(query);
    if fallbacks.is_empty()
        && let Some(term) = lexical_terms(query, 1).into_iter().next()
    {
        fallbacks.push(term);
    }
    fallbacks.truncate(4);
    fallbacks
}

fn lexical_terms(query: &str, limit: usize) -> Vec<String> {
    lexical_term_candidates(query, limit)
        .into_iter()
        .map(|term| term.text)
        .collect()
}

struct LexicalTerm {
    text: String,
    distinctive: bool,
}

fn lexical_term_candidates(query: &str, limit: usize) -> Vec<LexicalTerm> {
    let mut distinctive = Vec::new();
    let mut ordinary = Vec::new();
    let mut seen = HashSet::new();
    for raw in query
        .split(|character: char| {
            !character.is_alphanumeric() && !matches!(character, '-' | '_' | '/' | '.')
        })
        .map(|raw| raw.trim_matches(|character: char| !character.is_alphanumeric()))
        .filter(|raw| !raw.is_empty())
    {
        let term = raw.to_lowercase();
        let has_digit = term.chars().any(|character| character.is_ascii_digit());
        let is_single_digit =
            term.chars().count() == 1 && term.chars().all(|character| character.is_ascii_digit());
        if is_single_digit
            || (!has_digit && term.chars().count() < 3)
            || is_lexical_noise(&term)
            || !seen.insert(term.clone())
        {
            continue;
        }
        let letters = raw
            .chars()
            .filter(|character| character.is_alphabetic())
            .collect::<Vec<_>>();
        let uppercase_letters = letters
            .iter()
            .filter(|character| character.is_uppercase())
            .count();
        let is_acronym =
            letters.len() >= 2 && uppercase_letters >= 2 && uppercase_letters * 2 >= letters.len();
        let has_identifier_separator = raw
            .chars()
            .any(|character| matches!(character, '-' | '_' | '/' | '.'));
        let is_distinctive = has_digit || is_acronym || has_identifier_separator;
        let candidate = LexicalTerm {
            text: term,
            distinctive: is_distinctive,
        };
        if is_distinctive {
            distinctive.push(candidate);
        } else {
            ordinary.push(candidate);
        }
    }
    distinctive
        .into_iter()
        .chain(ordinary)
        .take(limit)
        .collect()
}

fn lexical_candidate_bonus(
    query: &str,
    path: &str,
    title: &str,
    heading: &str,
    excerpt: &str,
) -> f64 {
    let path_title = format!("{path} {title}");
    let path_title_words = normalized_search_words(&path_title);
    let path_title_collapsed = collapsed_search_text(&path_title);
    let heading_words = normalized_search_words(heading);
    let heading_collapsed = collapsed_search_text(heading);
    let excerpt_words = normalized_search_words(excerpt);
    let excerpt_collapsed = collapsed_search_text(excerpt);

    lexical_terms(query, 32)
        .into_iter()
        .map(|term| {
            let weight = if term.chars().any(|character| character.is_ascii_digit())
                || term.chars().count() >= 10
            {
                1.4
            } else if term.chars().count() >= 6 {
                1.0
            } else {
                0.7
            };
            if search_text_contains(&path_title_words, &path_title_collapsed, &term) {
                0.8 * weight
            } else if search_text_contains(&heading_words, &heading_collapsed, &term) {
                0.45 * weight
            } else if search_text_contains(&excerpt_words, &excerpt_collapsed, &term) {
                0.1 * weight
            } else {
                0.0
            }
        })
        .sum::<f64>()
        .min(8.0)
}

fn normalized_search_words(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len() + 2);
    normalized.push(' ');
    let mut separated = true;
    for character in value.chars().flat_map(char::to_lowercase) {
        if character.is_alphanumeric() {
            normalized.push(character);
            separated = false;
        } else if !separated {
            normalized.push(' ');
            separated = true;
        }
    }
    if !separated {
        normalized.push(' ');
    }
    normalized
}

fn collapsed_search_text(value: &str) -> String {
    value
        .chars()
        .flat_map(char::to_lowercase)
        .filter(|character| character.is_alphanumeric())
        .collect()
}

fn search_text_contains(words: &str, collapsed: &str, term: &str) -> bool {
    let normalized_term = normalized_search_words(term);
    let collapsed_term = collapsed_search_text(term);
    words.contains(&normalized_term)
        || (collapsed_term.chars().count() >= 5 && collapsed.contains(&collapsed_term))
}

fn is_lexical_noise(term: &str) -> bool {
    matches!(
        term,
        "and"
            | "are"
            | "about"
            | "after"
            | "against"
            | "also"
            | "before"
            | "being"
            | "between"
            | "could"
            | "current"
            | "does"
            | "each"
            | "for"
            | "inspect"
            | "explain"
            | "from"
            | "give"
            | "handle"
            | "have"
            | "identify"
            | "into"
            | "leave"
            | "needed"
            | "only"
            | "please"
            | "reconcile"
            | "report"
            | "request"
            | "resume"
            | "should"
            | "state"
            | "task"
            | "that"
            | "the"
            | "their"
            | "them"
            | "then"
            | "there"
            | "these"
            | "this"
            | "through"
            | "using"
            | "was"
            | "were"
            | "what"
            | "when"
            | "where"
            | "whether"
            | "which"
            | "while"
            | "with"
            | "without"
            | "would"
            | "will"
            | "work"
    )
}

fn continuation_paths(
    checkpoint: Option<&Value>,
    changes: &[Value],
) -> (Vec<String>, HashSet<String>) {
    let mut paths = Vec::new();
    let mut seen = HashSet::new();
    let mut changed_keys = HashSet::new();
    for path in changes
        .iter()
        .rev()
        .filter_map(|change| change.get("path").and_then(Value::as_str))
    {
        if path.starts_with(".brunn/") {
            continue;
        }
        let key = portable_path_key(path);
        changed_keys.insert(key.clone());
        if seen.insert(key) {
            paths.push(path.to_owned());
        }
        if paths.len() == 16 {
            break;
        }
    }
    if let Some(entries) = checkpoint
        .and_then(|value| value.get("source_entries"))
        .and_then(Value::as_array)
    {
        for path in entries
            .iter()
            .filter_map(|entry| entry.get("path").and_then(Value::as_str))
        {
            if path.starts_with(".brunn/") {
                continue;
            }
            let key = portable_path_key(path);
            if seen.insert(key) {
                paths.push(path.to_owned());
            }
            if paths.len() == 32 {
                break;
            }
        }
    }
    (paths, changed_keys)
}

fn looks_like_path(value: &str) -> bool {
    value.contains('/')
        && !value.starts_with('/')
        && ["md", "txt", "sql", "json", "jsonl", "csv"]
            .iter()
            .any(|extension| value.ends_with(&format!(".{extension}")))
}

fn validate_path(path: &str) -> ApiResult<()> {
    if path.is_empty()
        || path.len() > 1_024
        || path.starts_with('/')
        || path.contains('\\')
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || path.chars().any(char::is_control)
    {
        return Err(ApiError::invalid(
            "path must be a safe relative workspace path",
        ));
    }
    Ok(())
}

pub(crate) fn validate_public_path(path: &str) -> ApiResult<()> {
    validate_path(path)?;
    if path.starts_with(".brunn/") {
        return Err(ApiError::invalid(
            "the .brunn namespace is reserved for workspace-managed entries",
        ));
    }
    Ok(())
}

fn require_write_capabilities(auth: &AuthContext, path: &str) -> ApiResult<()> {
    auth.require(Capability::Save)?;
    if path.starts_with(crate::messaging_protocol::CONVERSATION_ENTRY_PREFIX) {
        if crate::messaging_protocol::conversation_id_from_path(path).is_none() {
            return Err(ApiError::invalid(
                "managed conversation paths require a lowercase canonical UUIDv7 filename",
            ));
        }
        auth.require(Capability::MessageWrite)?;
        if !auth.can(Capability::CredentialManage) && !auth.can(Capability::Admin) {
            return Err(ApiError::capability(Capability::CredentialManage.as_str()));
        }
    } else if path.starts_with(crate::task_service::TASK_ENTRY_PREFIX) {
        if crate::task_service::task_id_from_path(path).is_none() {
            return Err(ApiError::invalid(
                "managed task paths require a lowercase canonical UUIDv7 filename",
            ));
        }
        auth.require(Capability::TaskWrite)?;
        if !auth.can(Capability::CredentialManage) && !auth.can(Capability::Admin) {
            return Err(ApiError::capability(Capability::CredentialManage.as_str()));
        }
    } else if is_checkpoint_path(path) {
        auth.require(Capability::Checkpoint)?;
    } else if path.starts_with(".brunn/binaries/") {
        auth.require(Capability::Stage)?;
    }
    Ok(())
}

/// A chronicle-style "no durable memory" no-op summary aimed at the
/// agent-memory tree. These carry no signal and are no longer admitted; any
/// other agent-memory content passes unchanged.
pub(crate) fn is_agent_memory_noop_summary(path: &str, content: &str) -> bool {
    if !path.starts_with("agent-memory/") {
        return false;
    }
    let lower = content.to_lowercase();
    lower.contains("no durable memory")
        || lower.contains("no durable memories")
        || lower.contains("nothing durable could be extracted")
}

fn validate_write_path(request: &WriteRequest) -> ApiResult<()> {
    validate_path(&request.path)?;
    if !request.path.starts_with(".brunn/") {
        return Ok(());
    }
    let portable_restore = request
        .metadata
        .get("_brunn_import")
        .and_then(|value| value.get("format"))
        .and_then(Value::as_str)
        == Some("brunn-workspace-import-manifest@v1");
    let client_metadata = request
        .metadata
        .get("client")
        .filter(|value| value.is_object())
        .unwrap_or(&request.metadata);
    let checkpoint_restore = request
        .path
        .strip_prefix(".brunn/checkpoints/")
        .and_then(|value| value.strip_suffix(".md"))
        .and_then(|value| Uuid::parse_str(value).ok())
        .is_some_and(|checkpoint_id| {
            client_metadata.get("kind").and_then(Value::as_str) == Some("checkpoint")
                && client_metadata
                    .get("checkpoint_ref")
                    .and_then(Value::as_str)
                    .is_none_or(|reference| reference == format!("checkpoint:{checkpoint_id}"))
        });
    let binary_companion_restore = request.path.starts_with(".brunn/binaries/")
        && request.path.ends_with(".md")
        && client_metadata.get("kind").and_then(Value::as_str) == Some("binary_description")
        && client_metadata
            .get("binary_path")
            .and_then(Value::as_str)
            .is_some();
    let task_restore = crate::task_service::validate_task_entry(&request.path, &request.metadata)?;
    let conversation_restore = if request
        .path
        .starts_with(crate::messaging_protocol::CONVERSATION_ENTRY_PREFIX)
    {
        if request.content.len() > crate::messaging_protocol::MAX_CANONICAL_CONVERSATION_BYTES {
            return Err(ApiError::public(
                StatusCode::PAYLOAD_TOO_LARGE,
                "conversation_entry_too_large",
                "canonical conversation Markdown is limited to 12 MiB",
            ));
        }
        crate::messaging_protocol::validate_conversation_entry(
            &request.path,
            &request.metadata,
            &request.content,
        )
        .map_err(|error| ApiError::invalid(error.to_string()))?
        .is_some()
    } else {
        false
    };
    if portable_restore
        && request.expected_version.is_some()
        && (checkpoint_restore || binary_companion_restore || task_restore || conversation_restore)
    {
        return Ok(());
    }
    Err(ApiError::invalid(
        "the .brunn namespace is reserved for workspace-managed entries",
    ))
}

fn is_checkpoint_path(path: &str) -> bool {
    path.strip_prefix(".brunn/checkpoints/")
        .and_then(|value| value.strip_suffix(".md"))
        .and_then(|value| Uuid::parse_str(value).ok())
        .is_some()
}

fn is_portable_checkpoint_import(path: &str, metadata: &Value) -> bool {
    if !is_checkpoint_path(path)
        || metadata
            .get("_brunn_import")
            .and_then(|value| value.get("format"))
            .and_then(Value::as_str)
            != Some("brunn-workspace-import-manifest@v1")
    {
        return false;
    }
    metadata
        .get("client")
        .filter(|value| value.is_object())
        .unwrap_or(metadata)
        .get("kind")
        .and_then(Value::as_str)
        == Some("checkpoint")
}

fn checkpoint_client_metadata(metadata: &Value) -> &Value {
    metadata
        .get("client")
        .filter(|value| value.is_object())
        .unwrap_or(metadata)
}

fn imported_checkpoint_parent(metadata: &Value) -> ApiResult<Option<Uuid>> {
    let Some(value) = checkpoint_client_metadata(metadata).get("parent_checkpoint_ref") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let reference = value.as_str().ok_or_else(|| {
        ApiError::invalid("imported checkpoint parent_checkpoint_ref must be a string or null")
    })?;
    let raw = reference.strip_prefix("checkpoint:").ok_or_else(|| {
        ApiError::invalid("imported checkpoint parent must use checkpoint:<uuid>")
    })?;
    Uuid::parse_str(raw)
        .map(Some)
        .map_err(|_| ApiError::invalid("imported checkpoint parent reference is invalid"))
}

async fn validate_imported_checkpoint_parent_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    child_path: &str,
    metadata: &Value,
) -> ApiResult<()> {
    let Some(parent_id) = imported_checkpoint_parent(metadata)? else {
        return Ok(());
    };
    let child_id = child_path
        .strip_prefix(".brunn/checkpoints/")
        .and_then(|value| value.strip_suffix(".md"))
        .and_then(|value| Uuid::parse_str(value).ok())
        .ok_or_else(|| ApiError::invalid("imported checkpoint path is invalid"))?;
    if parent_id == child_id {
        return Err(ApiError::conflict(
            "checkpoint_parent_cycle",
            "an imported checkpoint cannot parent itself",
            json!({"checkpoint_ref": format!("checkpoint:{child_id}")}),
        ));
    }
    let parent_path = format!(".brunn/checkpoints/{parent_id}.md");
    let parent_metadata = sqlx::query_scalar::<_, Value>(
        r#"
        SELECT version.metadata
        FROM brunn.entries AS entry
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
         AND version.version=entry.current_version
        WHERE entry.user_id=$1
          AND lower(normalize(entry.path, NFC))=$2
          AND entry.kind='markdown'
          AND entry.deleted_at IS NULL
        "#,
    )
    .bind(user_id)
    .bind(portable_path_key(&parent_path))
    .fetch_optional(&mut **tx)
    .await?;
    let Some(parent_metadata) = parent_metadata else {
        return Err(ApiError::conflict(
            "checkpoint_parent_unresolved",
            "imported checkpoint parent must already exist for the same user",
            json!({"parent_checkpoint_ref": format!("checkpoint:{parent_id}")}),
        ));
    };
    let parent = checkpoint_client_metadata(&parent_metadata);
    let expected_parent_ref = format!("checkpoint:{parent_id}");
    if parent.get("kind").and_then(Value::as_str) != Some("checkpoint")
        || parent.get("checkpoint_ref").and_then(Value::as_str)
            != Some(expected_parent_ref.as_str())
    {
        return Err(ApiError::conflict(
            "checkpoint_parent_unresolved",
            "imported checkpoint parent does not resolve to a checkpoint entry",
            json!({"parent_checkpoint_ref": format!("checkpoint:{parent_id}")}),
        ));
    }
    Ok(())
}

fn rebase_imported_checkpoint_metadata(mut metadata: Value, generation: i64) -> Value {
    let target = if metadata.get("client").is_some_and(Value::is_object) {
        metadata
            .get_mut("client")
            .and_then(Value::as_object_mut)
            .expect("checked checkpoint client metadata")
    } else {
        metadata
            .as_object_mut()
            .expect("prepared Markdown metadata is always an object")
    };
    if let Some(origin) = target.get("workspace_generation").cloned() {
        target
            .entry("origin_workspace_generation".to_owned())
            .or_insert(origin);
    }
    target.insert("workspace_generation".to_owned(), json!(generation));
    if let Some(source_entries) = target
        .get_mut("source_entries")
        .and_then(Value::as_array_mut)
    {
        for source in source_entries {
            if let Some(source) = source.as_object_mut() {
                source.remove("entry_ref");
            }
        }
    }
    metadata
}

pub(crate) fn portable_path_key(path: &str) -> String {
    path.nfc().collect::<String>().to_lowercase()
}

fn validate_idempotency_key(key: &str) -> ApiResult<()> {
    if key.is_empty() || key.len() > 256 || key.chars().any(char::is_control) {
        return Err(ApiError::invalid(
            "idempotency_key must contain 1 to 256 printable characters",
        ));
    }
    Ok(())
}

pub(crate) fn validate_sha256(value: &str) -> ApiResult<String> {
    let normalized = value.trim().strip_prefix("sha256:").unwrap_or(value.trim());
    if normalized.len() != 64 || !normalized.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ApiError::invalid(
            "expected_content_hash must be a SHA-256 value",
        ));
    }
    Ok(normalized.to_ascii_lowercase())
}

fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_owned();
    }
    text.chars().take(limit).collect()
}

fn record_candidate_usage(state: &AppState, auth: &AuthContext, candidates: &[Candidate]) {
    let ids = candidates
        .iter()
        .map(|candidate| candidate.entry_id)
        .collect::<Vec<_>>();
    record_entry_usage(state, auth, &ids, UsageOperation::Search);
}

fn record_entry_usage(
    state: &AppState,
    auth: &AuthContext,
    entry_ids: &[Uuid],
    operation: UsageOperation,
) {
    state
        .usage_tracker
        .record(auth, entry_ids.iter().copied(), operation);
}

fn record_serialized_product_read<T: Serialize>(
    state: &AppState,
    auth: &AuthContext,
    operation: ProductActivityOperation,
    response: &T,
) {
    match serde_json::to_vec(response) {
        Ok(bytes) => record_product_activity(
            state,
            auth,
            operation,
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        ),
        Err(error) => tracing::warn!(
            ?error,
            operation = operation.as_str(),
            "product activity response sizing failed"
        ),
    }
}

fn record_product_activity(
    state: &AppState,
    auth: &AuthContext,
    operation: ProductActivityOperation,
    bytes: u64,
) {
    state
        .usage_tracker
        .record_product_activity(auth, operation, bytes);
}

fn validate_eval_import(request: &EvalImportRequest) -> ApiResult<()> {
    if request.schema != "brunn-eval-import@v1" {
        return Err(ApiError::invalid(
            "evaluation import schema must be brunn-eval-import@v1",
        ));
    }
    if !matches!(request.access_mode.as_str(), "read_only" | "read_write") {
        return Err(ApiError::invalid(
            "access_mode must be read_only or read_write",
        ));
    }
    if request.documents.is_empty() {
        return Err(ApiError::invalid(
            "evaluation import requires at least one document",
        ));
    }
    if request.documents.len() + request.delta_documents.len() > 700_000 {
        return Err(ApiError::invalid(
            "simple evaluation imports are limited to 700,000 documents",
        ));
    }
    if let Some((index, count)) = evaluation_batch(request)? {
        if request.documents.len() + request.delta_documents.len() > 10_000 {
            return Err(ApiError::invalid(
                "batched simple evaluation imports are limited to 10,000 documents per request",
            ));
        }
        if index + 1 < count
            && (!request.delta_documents.is_empty() || request.seed_checkpoint.is_some())
        {
            return Err(ApiError::invalid(
                "evaluation deltas and seed checkpoints belong only in the final batch",
            ));
        }
    }
    let total_bytes = request
        .documents
        .iter()
        .chain(&request.delta_documents)
        .map(|document| document.content.len())
        .sum::<usize>();
    if total_bytes > 512 * 1024 * 1024 {
        return Err(ApiError::public(
            StatusCode::PAYLOAD_TOO_LARGE,
            "eval_import_too_large",
            "evaluation text is limited to 512 MiB",
        ));
    }
    for document in request.documents.iter().chain(&request.delta_documents) {
        validate_path(&document.path)?;
        let digest = hex::encode(Sha256::digest(document.content.as_bytes()));
        if digest != document.content_sha256.to_lowercase() {
            return Err(ApiError::invalid(format!(
                "content_sha256 does not match {}",
                document.path
            )));
        }
    }
    Ok(())
}

fn evaluation_batch(request: &EvalImportRequest) -> ApiResult<Option<(usize, usize)>> {
    match (request.batch_index, request.batch_count) {
        (None, None) => Ok(None),
        (Some(index), Some(count)) if count > 0 && index < count => Ok(Some((index, count))),
        (Some(_), Some(_)) => Err(ApiError::invalid(
            "evaluation batch_index must be less than a positive batch_count",
        )),
        _ => Err(ApiError::invalid(
            "evaluation batch_index and batch_count must be provided together",
        )),
    }
}

pub(crate) async fn require_local_publish_lock(
    tx: &mut Transaction<'_, Postgres>,
    key: String,
    bounded_wait: bool,
) -> ApiResult<()> {
    let acquired = if bounded_wait {
        match sqlx::query(
            r#"
            WITH prior AS MATERIALIZED (
              SELECT current_setting('lock_timeout') AS lock_timeout
            ), configured AS MATERIALIZED (
              SELECT set_config('lock_timeout','250ms',true)
              FROM prior
            ), acquired AS MATERIALIZED (
              SELECT pg_advisory_xact_lock(hashtextextended($1,0))
              FROM configured
            )
            SELECT set_config('lock_timeout',prior.lock_timeout,true)
            FROM prior
            CROSS JOIN acquired
            "#,
        )
        .bind(&key)
        .execute(&mut **tx)
        .await
        {
            Ok(_) => true,
            Err(sqlx::Error::Database(error)) if error.code().as_deref() == Some("55P03") => false,
            Err(error) => return Err(error.into()),
        }
    } else {
        sqlx::query_scalar::<_, bool>("SELECT pg_try_advisory_xact_lock(hashtextextended($1,0))")
            .bind(key)
            .fetch_one(&mut **tx)
            .await?
    };
    if !acquired {
        return Err(ApiError::conflict(
            "entry_busy",
            "another write is publishing the same entry; retry",
            json!({"retryable": true}),
        ));
    }
    Ok(())
}

fn derive_eval_token(secret: &str, external_ref: &str, idempotency_key: &str) -> ApiResult<String> {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .map_err(|error| ApiError::Internal(format!("invalid evaluation key: {error}")))?;
    mac.update(external_ref.as_bytes());
    mac.update(b"\0");
    mac.update(idempotency_key.as_bytes());
    Ok(format!(
        "brunn_eval_{}",
        URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes())
    ))
}

async fn prepare_bulk_documents(
    state: &AppState,
    documents: &[EvalDocument],
) -> ApiResult<Vec<BulkMarkdown>> {
    let normalized = documents
        .iter()
        .map(|document| {
            (
                document,
                normalize_document(&document.path, &document.content),
            )
        })
        .collect::<Vec<_>>();
    let mut prepared = Vec::with_capacity(normalized.len());
    for (source, document) in normalized {
        let embeddings = vec![None; document.chunks.len()];
        prepared.push(BulkMarkdown {
            entry_id: Uuid::now_v7(),
            version_id: Uuid::now_v7(),
            path: source.path.clone(),
            title: document.title,
            content: source.content.clone(),
            content_sha256: source.content_sha256.to_lowercase(),
            media_type: source.media_type.clone(),
            metadata: json!({"kind": "evaluation_import"}),
            chunks: document.chunks,
            embeddings,
            frontmatter: if state.config.supersession_demotion || state.config.intention_ledger {
                parse_frontmatter(&source.content)
            } else {
                DerivedFrontmatter::default()
            },
        });
    }
    Ok(prepared)
}

async fn prepare_one_bulk_document(
    state: &AppState,
    path: String,
    content: String,
    media_type: String,
) -> ApiResult<BulkMarkdown> {
    let digest = hex::encode(Sha256::digest(content.as_bytes()));
    let source = EvalDocument {
        path,
        content,
        content_sha256: digest,
        media_type,
    };
    prepare_bulk_documents(state, &[source])
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::Internal("checkpoint preparation returned no document".to_owned()))
}

async fn insert_bulk_documents(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    credential_id: Uuid,
    documents: &[BulkMarkdown],
    operation: &str,
) -> ApiResult<()> {
    for batch in documents.chunks(1_000) {
        let mut entries = QueryBuilder::<Postgres>::new(
            "INSERT INTO brunn.entries \
             (id,user_id,path,title,kind,media_type,current_version) ",
        );
        entries.push_values(batch, |mut row, document| {
            row.push_bind(document.entry_id)
                .push_bind(user_id)
                .push_bind(&document.path)
                .push_bind(&document.title)
                .push_bind("markdown")
                .push_bind(&document.media_type)
                .push_bind(1_i64);
        });
        entries.build().execute(&mut **tx).await?;
    }
    for batch in documents.chunks(700) {
        let mut versions = QueryBuilder::<Postgres>::new(
            "INSERT INTO brunn.entry_versions \
             (id,user_id,entry_id,version,content_sha256,content,size_bytes,metadata,created_by_credential_id) ",
        );
        versions.push_values(batch, |mut row, document| {
            row.push_bind(document.version_id)
                .push_bind(user_id)
                .push_bind(document.entry_id)
                .push_bind(1_i64)
                .push_bind(&document.content_sha256)
                .push_bind(&document.content)
                .push_bind(i64::try_from(document.content.len()).unwrap_or(i64::MAX))
                .push_bind(&document.metadata)
                .push_bind(credential_id);
        });
        versions.build().execute(&mut **tx).await?;
    }
    insert_bulk_chunks(tx, user_id, documents).await?;
    for batch in documents.chunks(1_000) {
        let mut changes = QueryBuilder::<Postgres>::new(
            "INSERT INTO brunn.workspace_changes \
             (user_id,entry_id,entry_version,operation,path,content_sha256) ",
        );
        changes.push_values(batch, |mut row, document| {
            row.push_bind(user_id)
                .push_bind(document.entry_id)
                .push_bind(1_i64)
                .push_bind(operation)
                .push_bind(&document.path)
                .push_bind(&document.content_sha256);
        });
        changes.build().execute(&mut **tx).await?;
    }
    for batch in documents.chunks(1_000) {
        let mut jobs =
            QueryBuilder::<Postgres>::new("INSERT INTO brunn.jobs (user_id,kind,payload) ");
        jobs.push_values(batch, |mut row, document| {
            row.push_bind(user_id)
                .push_bind("embed_entry")
                .push_bind(json!({
                    "entry_id": document.entry_id,
                    "version": 1_i64
                }));
        });
        jobs.build().execute(&mut **tx).await?;
    }
    Ok(())
}

async fn insert_bulk_chunks(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    documents: &[BulkMarkdown],
) -> ApiResult<()> {
    let rows = documents
        .iter()
        .flat_map(|document| {
            document
                .chunks
                .iter()
                .zip(&document.embeddings)
                .map(move |(chunk, embedding)| (document, chunk, embedding))
        })
        .collect::<Vec<_>>();
    for batch in rows.chunks(500) {
        let mut builder = QueryBuilder::<Postgres>::new(
            "INSERT INTO brunn.search_chunks \
             (id,user_id,entry_id,entry_version_id,chunk_index,path,heading,content,token_estimate,embedding) ",
        );
        builder.push_values(batch, |mut row, (document, chunk, embedding)| {
            row.push_bind(Uuid::now_v7())
                .push_bind(user_id)
                .push_bind(document.entry_id)
                .push_bind(document.version_id)
                .push_bind(i32::try_from(chunk.ordinal).unwrap_or(i32::MAX))
                .push_bind(&document.path)
                .push_bind(&chunk.heading)
                .push_bind(&chunk.content)
                .push_bind(i32::try_from(chunk.estimated_tokens).unwrap_or(i32::MAX))
                .push_bind((*embedding).clone());
        });
        builder.build().execute(&mut **tx).await?;
    }
    Ok(())
}

async fn apply_bulk_deltas(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    credential_id: Uuid,
    deltas: &[BulkMarkdown],
) -> ApiResult<()> {
    for delta in deltas {
        let row = sqlx::query(
            "SELECT id,current_version FROM brunn.entries \
             WHERE user_id=$1 AND lower(normalize(path, NFC))=$2 FOR UPDATE",
        )
        .bind(user_id)
        .bind(portable_path_key(&delta.path))
        .fetch_optional(&mut **tx)
        .await?;
        let Some(row) = row else {
            insert_bulk_documents(
                tx,
                user_id,
                credential_id,
                std::slice::from_ref(delta),
                "create",
            )
            .await?;
            continue;
        };
        let entry_id: Uuid = row.get("id");
        let version = row.get::<i64, _>("current_version") + 1;
        sqlx::query(
            r#"
            INSERT INTO brunn.entry_versions (
              id,user_id,entry_id,version,content_sha256,content,size_bytes,
              metadata,created_by_credential_id
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            "#,
        )
        .bind(delta.version_id)
        .bind(user_id)
        .bind(entry_id)
        .bind(version)
        .bind(&delta.content_sha256)
        .bind(&delta.content)
        .bind(i64::try_from(delta.content.len()).unwrap_or(i64::MAX))
        .bind(json!({"kind": "evaluation_delta"}))
        .bind(credential_id)
        .execute(&mut **tx)
        .await?;
        sqlx::query(
            r#"
            UPDATE brunn.entries
            SET title=$3,media_type=$4,current_version=$5,
                updated_at=clock_timestamp(),deleted_at=NULL
            WHERE user_id=$1 AND id=$2
            "#,
        )
        .bind(user_id)
        .bind(entry_id)
        .bind(&delta.title)
        .bind(&delta.media_type)
        .bind(version)
        .execute(&mut **tx)
        .await?;
        sqlx::query("DELETE FROM brunn.search_chunks WHERE user_id=$1 AND entry_id=$2")
            .bind(user_id)
            .bind(entry_id)
            .execute(&mut **tx)
            .await?;
        insert_chunks(
            tx,
            user_id,
            entry_id,
            delta.version_id,
            &delta.path,
            &delta.chunks,
            &delta.embeddings,
        )
        .await?;
        sqlx::query(
            r#"
            INSERT INTO brunn.workspace_changes (
              user_id,entry_id,entry_version,operation,path,content_sha256
            ) VALUES ($1,$2,$3,'update',$4,$5)
            "#,
        )
        .bind(user_id)
        .bind(entry_id)
        .bind(version)
        .bind(&delta.path)
        .bind(&delta.content_sha256)
        .execute(&mut **tx)
        .await?;
        sqlx::query(
            r#"
            INSERT INTO brunn.jobs (user_id,kind,payload)
            VALUES ($1,'embed_entry',$2)
            "#,
        )
        .bind(user_id)
        .bind(json!({"entry_id": entry_id, "version": version}))
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

pub(crate) async fn max_generation_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
) -> ApiResult<i64> {
    Ok(sqlx::query_scalar::<_, Option<i64>>(
        "SELECT max(generation) FROM brunn.workspace_changes WHERE user_id=$1",
    )
    .bind(user_id)
    .fetch_one(&mut **tx)
    .await?
    .unwrap_or(0))
}

fn render_seed_checkpoint(
    checkpoint_id: Uuid,
    generation: i64,
    state: &Value,
    source_refs: &[String],
) -> String {
    format!(
        "---\nbrunn_kind: checkpoint\ncheckpoint_id: {checkpoint_id}\n\
         workspace_generation: {generation}\n---\n\n\
         # Seed checkpoint\n\n## State\n\n```json\n{}\n```\n\n\
         ## Source references\n\n{}\n",
        serde_json::to_string_pretty(state).unwrap_or_else(|_| "{}".to_owned()),
        if source_refs.is_empty() {
            "- None recorded.".to_owned()
        } else {
            source_refs
                .iter()
                .map(|reference| format!("- `{reference}`"))
                .collect::<Vec<_>>()
                .join("\n")
        }
    )
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use chrono::TimeZone as _;

    use super::*;
    use crate::{
        embeddings::Embedder,
        location::rules::{
            Confidence as LocationConfidence, Coordinate as LocationCoordinate,
            OpenVisit as LocationOpenVisit,
        },
    };

    struct RequestBatchProbeEmbedder {
        calls: AtomicUsize,
        batches: Mutex<Vec<Vec<String>>>,
    }

    #[async_trait]
    impl Embedder for RequestBatchProbeEmbedder {
        async fn embed(&self, input: &[String]) -> ApiResult<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.batches.lock().unwrap().push(input.to_vec());
            Ok(input
                .iter()
                .map(|text| vec![text.len() as f32, 1.0, 0.0])
                .collect())
        }

        fn provider(&self) -> &'static str {
            "mock"
        }

        fn model(&self) -> &str {
            "request-batch-probe-v1"
        }

        fn dimensions(&self) -> usize {
            3
        }

        fn is_degraded(&self) -> bool {
            false
        }
    }

    fn test_location_presence(reported_at: DateTime<Utc>) -> LocationPresenceState {
        LocationPresenceState {
            timezone: chrono_tz::America::Los_Angeles,
            reported_at,
            last_coordinate: LocationCoordinate {
                lat: 47.6205,
                lon: -122.2070,
            },
            last_accuracy_m: 20.0,
            city: Some("Bellevue".to_owned()),
            region: Some("WA".to_owned()),
            country: Some("US".to_owned()),
            visit: Some(LocationOpenVisit {
                arrived_at: reported_at - chrono::Duration::hours(2),
                coordinate: LocationCoordinate {
                    lat: 47.6205,
                    lon: -122.2070,
                },
                label: Some("Home".to_owned()),
                kind: "home".to_owned(),
                confidence: LocationConfidence::High,
                opened_by_ping: false,
            }),
        }
    }

    #[test]
    fn owner_presence_open_block_is_formatted_and_omitted_when_unavailable_or_over_seven_days_old()
    {
        let now = Utc.with_ymd_and_hms(2026, 9, 2, 20, 0, 0).single().unwrap();
        let value = render_owner_presence(
            Ok(Some(test_location_presence(
                now - chrono::Duration::hours(6),
            ))),
            now,
        )
        .unwrap();
        assert_eq!(value["status"], "at_place");
        assert_eq!(value["place"]["label"], "Home");
        assert_eq!(value["at_home"], true);
        assert_eq!(value["city"], "Bellevue");
        assert_eq!(value["timezone"], "America/Los_Angeles");

        assert!(
            render_owner_presence(
                Ok(Some(test_location_presence(
                    now - chrono::Duration::days(7),
                ))),
                now,
            )
            .is_none()
        );
        assert!(render_owner_presence(Ok(None), now).is_none());
        assert!(
            render_owner_presence(
                Err(ApiError::Internal("synthetic lookup failure".to_owned())),
                now,
            )
            .is_none()
        );
    }

    #[tokio::test]
    async fn first_ready_semantic_slot_prepares_the_entire_request_batch() {
        let runtime = SemanticRuntime::new(8);
        let embedder = Arc::new(RequestBatchProbeEmbedder {
            calls: AtomicUsize::new(0),
            batches: Mutex::new(Vec::new()),
        });
        let batch = RequestSemanticEmbeddings::new(vec![
            Some("alpha".to_owned()),
            None,
            Some("bravo".to_owned()),
        ])
        .unwrap();

        let alpha = batch
            .take(&runtime, embedder.clone(), Duration::from_secs(5), 0)
            .unwrap();
        assert!(
            batch
                .take(&runtime, embedder.clone(), Duration::from_secs(5), 1,)
                .is_none()
        );
        let bravo = batch
            .take(&runtime, embedder.clone(), Duration::from_secs(5), 2)
            .unwrap();
        let (alpha, bravo) = tokio::join!(alpha.resolve(), bravo.resolve());

        assert_eq!(alpha.unwrap(), vec![5.0, 1.0, 0.0]);
        assert_eq!(bravo.unwrap(), vec![5.0, 1.0, 0.0]);
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            embedder.batches.lock().unwrap().as_slice(),
            &[vec!["alpha".to_owned(), "bravo".to_owned()]]
        );
        let snapshot = runtime.snapshot();
        assert_eq!(snapshot.provider_batches, 1);
        assert_eq!(snapshot.provider_items, 2);
    }

    #[test]
    fn agent_memory_noop_summaries_are_not_admitted() {
        let noop =
            "# Chronicle summary\n\nNo durable memory could be extracted from this session.\n";
        assert!(is_agent_memory_noop_summary(
            "agent-memory/chronicle/2026-08-30.md",
            noop
        ));
        // The path prefix IS the classification: the same content elsewhere
        // is admitted, and signal-bearing chronicle content is kept.
        assert!(!is_agent_memory_noop_summary("sources/Notes.md", noop));
        assert!(!is_agent_memory_noop_summary(
            "agent-memory/chronicle/2026-08-30.md",
            "# Chronicle summary\n\nThe owner decided to migrate the vault on Tuesday.\n",
        ));
    }

    fn test_conversation_header() -> crate::messaging_protocol::ConversationHeader {
        use crate::messaging_protocol::{
            ConversationHeader, ConversationKind, ConversationParticipant, ConversationStatus,
        };

        let conversation_id = Uuid::parse_str("550e8400-e29b-71d4-a716-446655440000").unwrap();
        ConversationHeader {
            schema: "conversation.v1".to_owned(),
            conversation_id,
            conversation_kind: ConversationKind::Direct,
            direct_key: Some("alpha|owner".to_owned()),
            subject: None,
            status: ConversationStatus::Open,
            participants: vec![
                ConversationParticipant {
                    agent_id: "alpha".to_owned(),
                    role: "participant".to_owned(),
                },
                ConversationParticipant {
                    agent_id: "owner".to_owned(),
                    role: "participant".to_owned(),
                },
            ],
            created_by_agent_id: "owner".to_owned(),
            continues_from: None,
            agent_streak: 0,
            needs_human: false,
            latest_sync_cursor: 1,
            created_at: "2026-08-27T12:00:00Z".parse().unwrap(),
            closed_at: None,
        }
    }

    #[test]
    fn exact_read_expands_only_for_one_canonical_conversation() {
        use crate::messaging_protocol::{
            conversation_metadata, conversation_path, render_conversation,
        };

        let header = test_conversation_header();
        let conversation_id = header.conversation_id;
        let entry_id = Uuid::parse_str("550e8400-e29b-71d4-a716-446655440001").unwrap();
        let created_at = header.created_at;
        let content = render_conversation(&header, &[]).unwrap();
        let mut entry = EntryRow {
            id: entry_id,
            path: conversation_path(conversation_id),
            title: "Conversation".to_owned(),
            kind: "markdown".to_owned(),
            media_type: "text/markdown".to_owned(),
            version: 1,
            content_sha256: "a".repeat(64),
            size_bytes: i64::try_from(content.len()).unwrap(),
            content: Some(content),
            object_key: None,
            object_version_id: None,
            metadata: conversation_metadata(&header),
            updated_at: created_at,
            workspace_generation: Some(1),
        };
        let request = ReadItem {
            reference: Some(format!("entry:{entry_id}")),
            path: None,
            link_target: None,
            view: Some("full".to_owned()),
            start: None,
            end: None,
            max_chars: Some(crate::messaging_protocol::MAX_CANONICAL_CONVERSATION_BYTES),
            version: None,
        };

        assert_ne!(entry.id, conversation_id);
        assert_eq!(
            exact_read_char_limit(1, &request, &entry),
            crate::messaging_protocol::MAX_CANONICAL_CONVERSATION_BYTES
        );
        assert_eq!(
            exact_read_char_limit(2, &request, &entry),
            MAX_EXACT_READ_CHARS
        );
        entry.metadata = json!({});
        assert_eq!(
            exact_read_char_limit(1, &request, &entry),
            MAX_EXACT_READ_CHARS
        );
    }

    fn candidate_with_sections(id: u128, section_chars: &[usize]) -> Candidate {
        let sections = section_chars
            .iter()
            .enumerate()
            .map(|(index, chars)| CandidateSection {
                heading: format!("Section {index}"),
                excerpt: char::from(b'a' + u8::try_from(index).unwrap_or(0))
                    .to_string()
                    .repeat(*chars),
                score: 10.0 - index as f64,
            })
            .collect::<Vec<_>>();
        Candidate {
            entry_id: Uuid::from_u128(id),
            path: format!("Sources/{id}.md"),
            title: format!("Source {id}"),
            version: 1,
            updated_at: "2026-08-01T00:00:00Z".parse().expect("test timestamp"),
            content_sha256: format!("{id:064x}"),
            heading: sections
                .first()
                .map(|section| section.heading.clone())
                .unwrap_or_default(),
            excerpt: sections
                .first()
                .map(|section| section.excerpt.clone())
                .unwrap_or_default(),
            score: 10.0,
            lanes: vec!["lexical".to_owned()],
            sections,
            verbatim_matches: vec![],
            superseded_by: None,
        }
    }

    #[test]
    fn search_sort_defaults_to_best_match_and_validates_public_values() {
        assert_eq!(SearchSort::parse(None).unwrap(), SearchSort::BestMatch);
        assert_eq!(
            SearchSort::parse(Some("last_modified")).unwrap(),
            SearchSort::LastModified
        );
        assert_eq!(SearchSort::parse(Some("title")).unwrap(), SearchSort::Title);
        assert!(SearchSort::parse(Some("newest-ish")).is_err());

        let request: SearchRequest = serde_json::from_value(json!({
            "queries": [{"query": "signal", "sort": "last_modified"}]
        }))
        .expect("search request with sort");
        assert_eq!(request.queries[0].sort.as_deref(), Some("last_modified"));
    }

    #[test]
    fn search_sorts_by_relevance_recency_or_title_before_path() {
        let mut older_best = candidate_with_sections(1, &[8]);
        older_best.title = "Zulu".to_owned();
        older_best.score = 10.0;
        older_best.updated_at = "2026-08-01T00:00:00Z".parse().unwrap();

        let mut newer_best = candidate_with_sections(2, &[8]);
        newer_best.title = "alpha".to_owned();
        newer_best.score = 10.0;
        newer_best.updated_at = "2026-08-02T00:00:00Z".parse().unwrap();

        let mut newest_lower_score = candidate_with_sections(3, &[8]);
        newest_lower_score.title = "Middle".to_owned();
        newest_lower_score.score = 9.0;
        newest_lower_score.updated_at = "2026-08-03T00:00:00Z".parse().unwrap();

        let candidates = vec![
            older_best.clone(),
            newest_lower_score.clone(),
            newer_best.clone(),
        ];

        let mut best_match = candidates.clone();
        sort_candidates(&mut best_match, SearchSort::BestMatch);
        assert_eq!(
            best_match
                .iter()
                .map(|item| item.entry_id)
                .collect::<Vec<_>>(),
            vec![
                newer_best.entry_id,
                older_best.entry_id,
                newest_lower_score.entry_id
            ]
        );

        let mut last_modified = candidates.clone();
        sort_candidates(&mut last_modified, SearchSort::LastModified);
        assert_eq!(
            last_modified
                .iter()
                .map(|item| item.entry_id)
                .collect::<Vec<_>>(),
            vec![
                newest_lower_score.entry_id,
                newer_best.entry_id,
                older_best.entry_id
            ]
        );

        let mut title = candidates;
        sort_candidates(&mut title, SearchSort::Title);
        assert_eq!(
            title.iter().map(|item| item.entry_id).collect::<Vec<_>>(),
            vec![
                newer_best.entry_id,
                newest_lower_score.entry_id,
                older_best.entry_id
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn hybrid_retrieval_never_waits_for_optional_semantic() {
        let started = tokio::time::Instant::now();
        let (_, _, semantic) = join_retrieval_lanes(
            tokio::time::sleep(Duration::from_millis(120)),
            tokio::time::sleep(Duration::from_millis(180)),
            tokio::time::sleep(Duration::from_millis(300)),
            false,
        )
        .await;
        let elapsed = started.elapsed();
        assert!(elapsed >= Duration::from_millis(180));
        assert!(
            elapsed < Duration::from_millis(200),
            "hybrid retrieval must return with exact+lexical instead of waiting for semantic; observed {elapsed:?}"
        );
        assert!(semantic.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn hybrid_retrieval_keeps_semantic_when_it_finishes_before_core() {
        let started = tokio::time::Instant::now();
        let (_, _, semantic) = join_retrieval_lanes(
            tokio::time::sleep(Duration::from_millis(180)),
            tokio::time::sleep(Duration::from_millis(150)),
            tokio::time::sleep(Duration::from_millis(120)),
            false,
        )
        .await;
        let elapsed = started.elapsed();
        assert!(elapsed >= Duration::from_millis(180));
        assert!(elapsed < Duration::from_millis(200));
        assert!(semantic.is_some());
    }

    #[tokio::test(start_paused = true)]
    async fn semantic_only_retrieval_waits_for_its_bounded_lane() {
        let started = tokio::time::Instant::now();
        let (_, _, semantic) = join_retrieval_lanes(
            tokio::time::sleep(Duration::from_millis(120)),
            tokio::time::sleep(Duration::from_millis(180)),
            tokio::time::sleep(Duration::from_millis(300)),
            true,
        )
        .await;
        let elapsed = started.elapsed();
        assert!(elapsed >= Duration::from_millis(300));
        assert!(elapsed < Duration::from_millis(320));
        assert!(semantic.is_some());
    }

    fn semantic_report(outcome: SemanticOutcome) -> SemanticLaneReport {
        SemanticLaneReport {
            outcome,
            ready_ms: 1.0,
            lane_ms: 2.0,
        }
    }

    #[test]
    fn optional_semantic_failures_never_downgrade_core_results() {
        let mut merged = HashMap::new();
        merge_candidate(&mut merged, candidate_with_sections(1, &[64]));
        merge_candidate(&mut merged, candidate_with_sections(2, &[64]));
        let mut failures: Vec<&'static str> = vec![];
        let mut timings = RetrievalTimings::default();
        assert!(!apply_semantic_outcome(
            Some(semantic_report(SemanticOutcome::Disabled)),
            false,
            &mut merged,
            &mut failures,
            &mut timings,
        ));
        for outcome in [
            SemanticOutcome::IndexUnavailable,
            SemanticOutcome::ReadinessError,
            SemanticOutcome::Failed,
            SemanticOutcome::Deferred,
        ] {
            assert!(!apply_semantic_outcome(
                Some(semantic_report(outcome)),
                false,
                &mut merged,
                &mut failures,
                &mut timings,
            ));
            assert_eq!(merged.len(), 2);
        }
        assert!(apply_semantic_outcome(
            None,
            false,
            &mut merged,
            &mut failures,
            &mut timings,
        ));
        assert!(failures.is_empty());
    }

    #[test]
    fn semantic_only_failures_are_reported_without_removing_core_results() {
        let mut merged = HashMap::new();
        merge_candidate(&mut merged, candidate_with_sections(1, &[64]));
        merge_candidate(&mut merged, candidate_with_sections(2, &[64]));
        let mut failures: Vec<&'static str> = vec![];
        let mut timings = RetrievalTimings::default();
        for outcome in [
            SemanticOutcome::Disabled,
            SemanticOutcome::IndexUnavailable,
            SemanticOutcome::ReadinessError,
            SemanticOutcome::Failed,
            SemanticOutcome::Deferred,
        ] {
            assert!(!apply_semantic_outcome(
                Some(semantic_report(outcome)),
                true,
                &mut merged,
                &mut failures,
                &mut timings,
            ));
            assert_eq!(merged.len(), 2);
        }
        assert_eq!(
            failures,
            vec![
                "semantic_disabled",
                "semantic_index_unavailable",
                "semantic_readiness_error",
                "semantic",
                "semantic_deferred"
            ]
        );
    }

    #[test]
    fn semantic_success_merges_candidates_and_records_lane_timings() {
        let mut merged = HashMap::new();
        merge_candidate(&mut merged, candidate_with_sections(1, &[64]));
        let mut failures: Vec<&'static str> = vec![];
        let mut timings = RetrievalTimings::default();
        assert!(!apply_semantic_outcome(
            Some(SemanticLaneReport {
                outcome: SemanticOutcome::Success(SemanticCandidates {
                    candidates: vec![candidate_with_sections(9, &[64])],
                    embed_ms: 10.0,
                    database_ms: 20.0,
                }),
                ready_ms: 3.0,
                lane_ms: 40.0,
            }),
            false,
            &mut merged,
            &mut failures,
            &mut timings,
        ));
        assert_eq!(merged.len(), 2);
        assert!(failures.is_empty());
        assert_eq!(timings.semantic_ready, 3.0);
        assert_eq!(timings.semantic, 40.0);
        assert_eq!(timings.embed, 10.0);
        assert_eq!(timings.semantic_db, 20.0);
    }

    #[test]
    fn semantic_gap_reasons_are_distinct_per_condition() {
        for (token, kind, reason) in [
            (
                "semantic_disabled",
                "retrieval_lane_unavailable",
                "policy_disabled",
            ),
            (
                "semantic_index_unavailable",
                "retrieval_lane_unavailable",
                "index_unavailable",
            ),
            (
                "semantic_readiness_error",
                "retrieval_lane_unavailable",
                "dependency_error",
            ),
            (
                "semantic_deferred",
                "retrieval_lane_deferred",
                "deadline_deferred",
            ),
            ("semantic", "retrieval_lane_failed", "dependency_error"),
        ] {
            let gap = retrieval_lane_gap(token);
            assert_eq!(gap["kind"], kind, "kind for {token}");
            assert_eq!(gap["lane"], "semantic", "lane for {token}");
            assert_eq!(gap["reason"], reason, "reason for {token}");
            assert!(
                gap["message"].as_str().unwrap().contains("retained"),
                "message for {token} must state other evidence is retained"
            );
        }
        let generic = retrieval_lane_gap("lexical");
        assert_eq!(generic["kind"], "retrieval_lane_failed");
        assert_eq!(generic["lane"], "lexical");
        assert!(generic.get("reason").is_none());
    }

    #[test]
    fn no_semantic_policy_applies_to_default_open_and_search_selection() {
        let default_lanes = retrieval_lane_selection(&[]).unwrap();
        assert_eq!(
            default_lanes,
            RetrievalLaneSelection {
                exact: true,
                lexical: true,
                semantic: true,
                semantic_only: false,
            }
        );
        let semantic_policy_enabled = false;
        assert!(!(default_lanes.semantic && semantic_policy_enabled));

        let exact_lexical =
            retrieval_lane_selection(&["exact".to_owned(), "lexical".to_owned()]).unwrap();
        assert!(!exact_lexical.semantic);
        assert!(retrieval_lane_selection(&["unknown".to_owned()]).is_err());
    }

    #[test]
    fn path_hints_only_return_explicit_relative_files() {
        assert_eq!(
            path_hints("Read `Trips/Europe 2026/Plan.md` and explain it"),
            ["Trips/Europe 2026/Plan.md"]
        );
        assert!(path_hints("Find the current trip plan").is_empty());
    }

    #[test]
    fn entry_link_lookup_is_exact_normalized_and_bounded() {
        assert_eq!(
            entry_link_lookup_keys("Projects/Other/Roádmap.MD#Next").unwrap(),
            vec![
                "roádmap".to_owned(),
                "roádmap.md".to_owned(),
                "roádmap.markdown".to_owned(),
            ]
        );
        assert!(entry_link_lookup_keys("#Heading").is_err());
        assert!(entry_link_lookup_keys(&"x".repeat(1_025)).is_err());
    }

    #[test]
    fn search_anchors_keep_explicit_clues_without_replaying_the_whole_task() {
        assert_eq!(
            search_anchors(
                "What is recorded for `terminal-corpus-64000-current-answer` without a path?"
            ),
            ["terminal-corpus-64000-current-answer"]
        );
        assert_eq!(
            search_anchors("Find terminal-corpus-64000-current-answer without a path"),
            ["terminal-corpus-64000-current-answer"]
        );
        assert!(
            search_anchors("Reconcile non-negotiable recovery rules and success/failure criteria")
                .is_empty()
        );
        assert!(search_anchors("Find the current trip plan").is_empty());
    }

    #[test]
    fn lexical_fallback_turns_natural_tasks_into_focused_term_pairs() {
        let fallbacks = bounded_lexical_fallback_queries(
            "Handle the request to add every confirmed Europe flight to the family calendar. \
             State what the bounded search found, whether any write is needed, and report it.",
        );
        assert!(fallbacks.contains(&"confirmed europe".to_owned()));
        assert!(fallbacks.contains(&"family calendar".to_owned()));
        assert!(
            fallbacks
                .iter()
                .all(|fallback| !fallback.contains("handle"))
        );
        assert!(
            fallbacks
                .iter()
                .all(|fallback| !fallback.contains("request"))
        );
        let focused = bounded_lexical_fallback_queries(
            "Handle the request to add every confirmed Europe flight to the family calendar. \
             State what the bounded search found, whether any write is needed, and report it.",
        )
        .into_iter()
        .next()
        .expect("natural task should produce a focused query");
        assert_eq!(focused, "confirmed europe");
    }

    #[test]
    fn lexical_fallback_keeps_distinctive_terms_from_long_tasks() {
        let fallbacks = bounded_lexical_fallback_queries(
            "Resume the D1 parser performance and autonomy work. Reconcile the older 10M Nyx \
             tuning checkpoint with the later durable autonomy summary. Explain what the PVE34 \
             result established and leave a checkpoint.",
        );
        assert_eq!(fallbacks.first().map(String::as_str), Some("d1 10m"));
        assert!(
            fallbacks.contains(&"pve34 parser".to_owned()),
            "{fallbacks:?}"
        );
        assert!(fallbacks.contains(&"performance autonomy".to_owned()));
        assert!(
            fallbacks
                .iter()
                .all(|fallback| !fallback.contains("resume"))
        );
        assert!(
            fallbacks
                .iter()
                .all(|fallback| !fallback.contains("reconcile"))
        );
        let focused = bounded_lexical_fallback_queries(
            "Resume the D1 parser performance and autonomy work. Reconcile the older 10M Nyx \
             tuning checkpoint with the later durable autonomy summary. Explain what the PVE34 \
             result established and leave a checkpoint.",
        )
        .into_iter()
        .next()
        .expect("identifier task should produce a focused query");
        assert_eq!(focused, "d1 10m");
        let ranked = ranked_lexical_fallback_queries(
            "Resume the D1 parser performance and autonomy work. Reconcile the older 10M Nyx \
             tuning checkpoint with the later durable autonomy summary. Explain what the PVE34 \
             result established and leave a checkpoint.",
        );
        assert!(ranked.contains(&"pve34 parser".to_owned()), "{ranked:?}");
        assert!(
            ranked.contains(&"performance autonomy".to_owned()),
            "{ranked:?}"
        );
    }

    #[test]
    fn lexical_focus_preserves_release_identifiers_without_broad_or_terms() {
        let task = "Location v1.1 SQLx slow statement warnings";
        let terms = lexical_terms(task, 16);
        assert_eq!(
            &terms[..2],
            ["v1.1".to_owned(), "sqlx".to_owned()],
            "{terms:?}"
        );
        assert!(!terms.iter().any(|term| term == "1"), "{terms:?}");

        let focused = bounded_lexical_fallback_queries(task)
            .into_iter()
            .next()
            .expect("release task should be searchable");
        assert_eq!(focused, "v1.1 sqlx");
        assert!(!focused.contains(" OR "));
        assert!(
            bounded_lexical_fallback_queries(task)
                .iter()
                .all(|fallback| !fallback.contains(" OR "))
        );
    }

    #[test]
    fn lexical_terms_keep_semver_acronyms_and_structured_identifiers_intact() {
        let terms = lexical_terms(
            "Compare Brunn v1.1 with v1.0 using SQLx RLS workspace_lexical_candidates_v2 \
             and apps/api/src/simple_core.rs",
            16,
        );
        for expected in [
            "v1.1",
            "v1.0",
            "sqlx",
            "rls",
            "workspace_lexical_candidates_v2",
            "apps/api/src/simple_core.rs",
        ] {
            assert!(
                terms.iter().any(|term| term == expected),
                "missing {expected} in {terms:?}"
            );
        }
        assert!(
            !terms.iter().any(|term| matches!(term.as_str(), "0" | "1")),
            "{terms:?}"
        );
    }

    #[test]
    fn lexical_focus_keeps_ordinary_natural_language_recall() {
        let task = "Handle the request to add every confirmed Europe flight to the family calendar. \
                    State what the bounded search found, whether any write is needed, and report it.";
        let focused = bounded_lexical_fallback_queries(task)
            .into_iter()
            .next()
            .expect("natural task should produce a focused query");
        assert_eq!(focused, "confirmed europe");

        let fallbacks = bounded_lexical_fallback_queries(task);
        assert_eq!(fallbacks.len(), 4, "{fallbacks:?}");
        assert_eq!(
            fallbacks.first().map(String::as_str),
            Some("confirmed europe")
        );
        assert!(
            fallbacks.iter().any(|query| query == "family calendar"),
            "an incidental first-pair hit must not exclude the authoritative pair: {fallbacks:?}"
        );
    }

    #[test]
    fn bounded_lexical_union_keeps_an_authoritative_fourth_pair_after_a_first_pair_hit() {
        let fallbacks =
            bounded_lexical_fallback_queries("confirmed Europe itinerary family calendar");
        assert_eq!(
            fallbacks,
            [
                "confirmed europe",
                "europe itinerary",
                "itinerary family",
                "family calendar",
            ]
        );

        let mut merged = HashMap::new();
        for fallback in &fallbacks {
            let hits = match fallback.as_str() {
                "confirmed europe" => {
                    let mut incidental = candidate_with_sections(1, &[16]);
                    incidental.score = 3.0;
                    let mut shared = candidate_with_sections(2, &[16]);
                    shared.score = 4.0;
                    vec![incidental, shared]
                }
                "family calendar" => {
                    let mut authoritative = candidate_with_sections(3, &[16]);
                    authoritative.score = 20.0;
                    let mut shared = candidate_with_sections(2, &[16]);
                    shared.score = 7.0;
                    vec![authoritative, shared]
                }
                _ => vec![],
            };
            for hit in hits {
                merge_candidate(&mut merged, hit);
            }
        }
        assert_eq!(merged.len(), 3, "all four queries must run and dedupe");
        let mut ranked = merged.into_values().collect::<Vec<_>>();
        sort_candidates(&mut ranked, SearchSort::BestMatch);
        assert_eq!(ranked[0].entry_id, Uuid::from_u128(3));
        assert_eq!(ranked[1].entry_id, Uuid::from_u128(2));
        assert_eq!(ranked[1].score, 7.0);
    }

    #[test]
    fn lexical_bonus_matches_preserved_structured_identifiers() {
        let relevant = lexical_candidate_bonus(
            "Location v1.1 SQLx workspace_lexical_candidates_v2",
            "sources/Projects/Brunn/Location-v1.1-SQLx.md",
            "workspace_lexical_candidates_v2",
            "SQLx slow statements",
            "Location v1.1 release evidence.",
        );
        let generic = lexical_candidate_bonus(
            "Location v1.1 SQLx workspace_lexical_candidates_v2",
            "sources/Projects/Brunn/Location notes.md",
            "Location notes",
            "Database work",
            "General release evidence.",
        );
        assert!(relevant > generic + 2.0, "{relevant} should beat {generic}");
    }

    #[test]
    fn lexical_bonus_prefers_domain_paths_over_generic_matching_prose() {
        let task = "Resume the D1 parser performance and autonomy work. Reconcile the older 10M \
                    Nyx tuning checkpoint with the later durable autonomy summary and PVE34.";
        let warmind = lexical_candidate_bonus(
            task,
            "Projects/Warmind/D1 parser autonomy and autotuning context summary - 2026-05-25.md",
            "D1 parser autonomy and autotuning context summary",
            "Current operating model",
            "PVE34 was historical evidence rather than a fixed identity.",
        );
        let generic = lexical_candidate_bonus(
            task,
            "Projects/Brunn/Portable Personal Context Layer.md",
            "Portable Personal Context Layer",
            "Durable agent work",
            "Resume durable work from a checkpoint and preserve an operating model.",
        );
        assert!(warmind > generic + 2.0, "{warmind} should beat {generic}");

        let rupture = lexical_candidate_bonus(
            "Extend the StarRupture factory plan from current source authority.",
            "Topics/Star Rupture/Star Rupture production.md",
            "Star Rupture production",
            "Source authority",
            "Current factory planning evidence.",
        );
        assert!(rupture >= 2.0);
    }

    #[test]
    fn continuation_paths_put_recent_changes_before_checkpoint_sources() {
        let checkpoint = json!({
            "source_entries": [
                {"path": "Projects/Warmind/Rules.md"},
                {"path": "Projects/Warmind/Plan.md"}
            ]
        });
        let changes = vec![
            json!({"path": "Projects/Warmind/Rules.md"}),
            json!({"path": ".brunn/checkpoints/ignored.md"}),
            json!({"path": "Projects/Warmind/New evidence.md"}),
        ];
        let (paths, changed) = continuation_paths(Some(&checkpoint), &changes);
        assert_eq!(
            paths,
            [
                "Projects/Warmind/New evidence.md",
                "Projects/Warmind/Rules.md",
                "Projects/Warmind/Plan.md"
            ]
        );
        assert!(changed.contains(&portable_path_key("Projects/Warmind/New evidence.md")));
        assert!(changed.contains(&portable_path_key("Projects/Warmind/Rules.md")));
    }

    #[test]
    fn safe_paths_reject_traversal_and_absolute_paths() {
        assert!(validate_path("Trips/Plan.md").is_ok());
        assert!(validate_path("../Plan.md").is_err());
        assert!(validate_path("/Trips/Plan.md").is_err());
        assert!(validate_path("Trips//Plan.md").is_err());
        assert!(validate_path(r"Trips\Plan.md").is_err());
    }

    #[test]
    fn portable_path_keys_fold_unicode_and_case_without_rewriting_paths() {
        assert_eq!(
            portable_path_key("Trips/Caf\u{e9}.md"),
            portable_path_key("trips/Cafe\u{301}.md")
        );
    }

    fn tier_a_history_metadata(target: i64, semantics: &str) -> Value {
        json!({
            "_brunn_import": {
                "format": WORKSPACE_IMPORT_FORMAT
            },
            "_brunn_tier_a_history": {
                "format": TIER_A_HISTORY_STAGE_FORMAT,
                "target_lineage_ordinal": target,
                "semantics": semantics
            }
        })
    }

    #[test]
    fn exact_history_preservation_requires_an_evaluation_stack() {
        let metadata = tier_a_history_metadata(2, TIER_A_EXACT_HISTORY_SEMANTICS);
        assert!(
            validate_tier_a_history_request("Notes/repeated.md", &metadata, Some(1), false)
                .is_err()
        );
        assert_eq!(
            validate_tier_a_history_request("Notes/repeated.md", &metadata, Some(1), true).unwrap(),
            Some(TierAHistoryStage {
                target_lineage_ordinal: 2,
                semantics: TierAHistorySemantics::PreserveIntentionalExactBytesVersion
            })
        );
        assert!(
            validate_tier_a_history_request("Notes/repeated.md", &metadata, Some(0), true).is_err()
        );
    }

    #[test]
    fn exact_history_append_retry_gap_and_ahead_are_fail_closed() {
        let stage = Some(TierAHistoryStage {
            target_lineage_ordinal: 2,
            semantics: TierAHistorySemantics::PreserveIntentionalExactBytesVersion,
        });
        let metadata = tier_a_history_metadata(2, TIER_A_EXACT_HISTORY_SEMANTICS);
        assert_eq!(
            tier_a_exact_history_action(stage, Some(1), true, false, Some(&json!({})), &metadata)
                .unwrap(),
            TierAExactHistoryAction::Append
        );
        assert_eq!(
            tier_a_exact_history_action(stage, Some(2), true, false, Some(&metadata), &metadata)
                .unwrap(),
            TierAExactHistoryAction::Idempotent
        );
        assert!(tier_a_exact_history_action(stage, None, false, false, None, &metadata).is_err());
        assert!(
            tier_a_exact_history_action(stage, Some(3), true, false, Some(&metadata), &metadata)
                .is_err()
        );
        assert!(
            tier_a_exact_history_action(stage, Some(1), false, false, Some(&json!({})), &metadata)
                .is_err()
        );
        assert_eq!(
            tier_a_exact_history_action(
                Some(TierAHistoryStage {
                    target_lineage_ordinal: 2,
                    semantics: TierAHistorySemantics::OrdinaryContentTransition,
                }),
                Some(1),
                true,
                false,
                Some(&json!({})),
                &metadata,
            )
            .unwrap(),
            TierAExactHistoryAction::NotRequested
        );
    }

    #[test]
    fn ordinary_writes_cannot_mutate_workspace_managed_paths() {
        let ordinary = WriteRequest {
            path: ".brunn/checkpoints/not-a-checkpoint.md".to_owned(),
            content: "no".to_owned(),
            media_type: markdown_media_type(),
            expected_version: None,
            idempotency_key: None,
            metadata: json!({}),
        };
        assert!(validate_write_path(&ordinary).is_err());

        let checkpoint_id = Uuid::now_v7();
        let portable_restore = WriteRequest {
            path: format!(".brunn/checkpoints/{checkpoint_id}.md"),
            content: "checkpoint".to_owned(),
            media_type: markdown_media_type(),
            expected_version: Some(0),
            idempotency_key: None,
            metadata: json!({
                "kind": "checkpoint",
                "checkpoint_ref": format!("checkpoint:{checkpoint_id}"),
                "_brunn_import": {
                    "format": "brunn-workspace-import-manifest@v1"
                }
            }),
        };
        assert!(validate_write_path(&portable_restore).is_ok());

        let conversation = test_conversation_header();
        let conversation_restore = WriteRequest {
            path: crate::messaging_protocol::conversation_path(conversation.conversation_id),
            content: crate::messaging_protocol::render_conversation(&conversation, &[]).unwrap(),
            media_type: markdown_media_type(),
            expected_version: Some(0),
            idempotency_key: None,
            metadata: json!({
                "client": crate::messaging_protocol::conversation_metadata(&conversation),
                "_brunn_import": {
                    "format": crate::messaging_protocol::WORKSPACE_IMPORT_FORMAT
                }
            }),
        };
        assert!(
            validate_write_path(&conversation_restore).is_ok(),
            "a canonical portable conversation restore owns its reserved path"
        );

        let oversized_conversation_restore = WriteRequest {
            content: "x".repeat(crate::messaging_protocol::MAX_CANONICAL_CONVERSATION_BYTES + 1),
            ..conversation_restore
        };
        assert!(matches!(
            validate_write_path(&oversized_conversation_restore),
            Err(ApiError::Public {
                status: StatusCode::PAYLOAD_TOO_LARGE,
                code: "conversation_entry_too_large",
                ..
            })
        ));
    }

    #[test]
    fn portable_restore_paths_require_their_own_capabilities() {
        let checkpoint_id = Uuid::now_v7();
        let auth = |capabilities: &[Capability]| AuthContext {
            credential_id: CredentialId(Uuid::now_v7()),
            user_id: UserId(Uuid::now_v7()),
            capabilities: capabilities
                .iter()
                .map(|capability| capability.as_str().to_owned())
                .collect(),
            scope_refs: vec!["scope:root".to_owned()],
            read_only: false,
        };
        let save_only = auth(&[Capability::Save]);
        assert!(require_write_capabilities(&save_only, "Notes/ordinary.md").is_ok());
        assert!(
            require_write_capabilities(
                &save_only,
                &format!(".brunn/checkpoints/{checkpoint_id}.md")
            )
            .is_err()
        );
        assert!(require_write_capabilities(&save_only, ".brunn/binaries/receipt.md").is_err());
        let conversation_path = crate::messaging_protocol::conversation_path(Uuid::now_v7());
        assert!(require_write_capabilities(&save_only, &conversation_path).is_err());
        assert!(
            require_write_capabilities(
                &auth(&[Capability::Save, Capability::MessageWrite]),
                &conversation_path
            )
            .is_err(),
            "ordinary messaging credentials cannot forge managed conversation entries"
        );
        assert!(
            require_write_capabilities(
                &auth(&[
                    Capability::Save,
                    Capability::MessageWrite,
                    Capability::CredentialManage
                ]),
                &conversation_path
            )
            .is_ok()
        );
        let task_id = Uuid::now_v7();
        let task_path = format!(".brunn/tasks/{task_id}.md");
        assert!(require_write_capabilities(&save_only, &task_path).is_err());
        assert!(
            require_write_capabilities(
                &auth(&[Capability::Save, Capability::TaskWrite]),
                &task_path
            )
            .is_err()
        );
        assert!(
            require_write_capabilities(
                &auth(&[
                    Capability::Save,
                    Capability::TaskWrite,
                    Capability::CredentialManage
                ]),
                &task_path
            )
            .is_ok()
        );
        assert!(
            require_write_capabilities(
                &auth(&[Capability::Save, Capability::TaskWrite, Capability::Admin]),
                &task_path
            )
            .is_ok()
        );
        assert!(
            require_write_capabilities(
                &auth(&[Capability::Save, Capability::TaskWrite, Capability::Admin]),
                ".brunn/tasks/550e8400-e29b-41d4-a716-446655440000.md"
            )
            .is_err()
        );
        assert!(
            require_write_capabilities(
                &auth(&[Capability::Save, Capability::Checkpoint]),
                &format!(".brunn/checkpoints/{checkpoint_id}.md")
            )
            .is_ok()
        );
        assert!(
            require_write_capabilities(
                &auth(&[Capability::Save, Capability::Stage]),
                ".brunn/binaries/receipt.md"
            )
            .is_ok()
        );
    }

    #[test]
    fn expected_binary_hashes_are_strict() {
        let hash = "a".repeat(64);
        assert_eq!(validate_sha256(&format!("sha256:{hash}")).unwrap(), hash);
        assert!(validate_sha256("not-a-hash").is_err());
    }

    #[test]
    fn exact_portable_companions_are_eval_only_and_preserve_exact_bytes() {
        let content = "\r\n# Byte copied\r\n";
        let hash = hex::encode(Sha256::digest(content.as_bytes()));
        assert!(
            prepare_portable_binary_companion(
                false,
                "sources/file.png",
                Some(content),
                Some(TIER_A_PORTABLE_COMPANION_FORMAT),
                Some("workspace/assets/file.png.md"),
                Some(&format!("sha256:{hash}")),
                Some(123),
                Some(0o600),
            )
            .is_err()
        );
        let companion = prepare_portable_binary_companion(
            true,
            "sources/file.png",
            Some(content),
            Some(TIER_A_PORTABLE_COMPANION_FORMAT),
            Some("workspace/assets/file.png.md"),
            Some(&format!("sha256:{hash}")),
            Some(123),
            Some(0o600),
        )
        .unwrap()
        .unwrap();
        assert_eq!(companion.content, content);
        assert_eq!(companion.content_sha256, hash);
        assert_eq!(companion.path, "workspace/assets/file.png.md");
        assert_eq!(companion.modified_unix_ns, Some(123));
        assert_eq!(companion.mode, Some(0o600));
        assert!(
            prepare_portable_binary_companion(
                true,
                "sources/file.png",
                Some(content),
                Some(TIER_A_PORTABLE_COMPANION_FORMAT),
                Some("workspace/assets/file.png.md"),
                Some(&format!("sha256:{}", "0".repeat(64))),
                None,
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn imported_checkpoint_parent_references_are_strict() {
        let parent = Uuid::now_v7();
        let metadata = json!({
            "_brunn_import": {"format": WORKSPACE_IMPORT_FORMAT},
            "client": {
                "kind": "checkpoint",
                "parent_checkpoint_ref": format!("checkpoint:{parent}")
            }
        });
        assert_eq!(imported_checkpoint_parent(&metadata).unwrap(), Some(parent));
        assert_eq!(
            imported_checkpoint_parent(&json!({"parent_checkpoint_ref": null})).unwrap(),
            None
        );
        assert!(
            imported_checkpoint_parent(&json!({
                "parent_checkpoint_ref": format!("entry:{parent}")
            }))
            .is_err()
        );
    }

    #[test]
    fn checkpoint_fingerprints_and_ids_are_canonical_and_request_bound() {
        let request = CheckpointRequest {
            session_id: "session:one".to_owned(),
            parent_checkpoint_id: None,
            state: serde_json::from_str(r#"{"z":1,"objective":"Continue","a":2}"#).unwrap(),
            source_refs: vec![],
            idempotency_key: Some("same".to_owned()),
        };
        let reordered = CheckpointRequest {
            state: serde_json::from_str(r#"{"a":2,"objective":"Continue","z":1}"#).unwrap(),
            ..request.clone()
        };
        let (effective_key, request_hash) = validate_checkpoint_request(&request).unwrap();
        let (reordered_key, reordered_hash) = validate_checkpoint_request(&reordered).unwrap();
        assert_eq!(effective_key, "same");
        assert_eq!(effective_key, reordered_key);
        assert_eq!(request_hash, reordered_hash);
        let correlated_from_another_session = CheckpointRequest {
            session_id: "session:two".to_owned(),
            ..request.clone()
        };
        let (correlated_key, correlated_hash) =
            validate_checkpoint_request(&correlated_from_another_session).unwrap();
        assert_eq!(effective_key, correlated_key);
        assert_eq!(
            request_hash, correlated_hash,
            "session IDs correlate calls but do not change the durable operation"
        );
        let newly_created_id = checkpoint_entry_id_for_new_write(&request);
        let legacy_binary_id = deterministic_checkpoint_id(&request);
        assert_eq!(
            newly_created_id, legacy_binary_id,
            "rolling deploys and rollbacks must agree on checkpoint entry identity"
        );
        assert_eq!(
            format!(".brunn/checkpoints/{newly_created_id}.md"),
            format!(".brunn/checkpoints/{legacy_binary_id}.md"),
            "the new writer must publish at the path recognized by the legacy binary"
        );
        assert_eq!(
            newly_created_id,
            checkpoint_entry_id_for_new_write(&reordered),
            "equivalent JSON payloads preserve the legacy-compatible path"
        );
        assert_ne!(
            newly_created_id,
            deterministic_checkpoint_id(&correlated_from_another_session),
            "the legacy binary included session ID in path identity; new-code cross-session idempotency comes from the receipt"
        );
        let receipt = json!({
            "checkpoint_ref": format!("checkpoint:{newly_created_id}"),
            "workspace_generation": 7,
            "resulting_workspace_generation": 7
        });
        let first_envelope =
            checkpoint_envelope(&request.session_id, receipt.clone(), false).unwrap();
        let replay_envelope = checkpoint_envelope(
            &correlated_from_another_session.session_id,
            receipt.clone(),
            true,
        )
        .unwrap();
        assert_eq!(first_envelope.data, replay_envelope.data);
        assert_eq!(replay_envelope.data, receipt);
        assert_eq!(first_envelope.session_id.as_deref(), Some("session:one"));
        assert_eq!(replay_envelope.session_id.as_deref(), Some("session:two"));
        let changed = CheckpointRequest {
            state: json!({"a": 2, "objective": "Different", "z": 1}),
            ..request.clone()
        };
        let (changed_key, changed_hash) = validate_checkpoint_request(&changed).unwrap();
        assert_eq!(effective_key, changed_key);
        assert_ne!(request_hash, changed_hash);
        assert_ne!(
            newly_created_id,
            checkpoint_entry_id_for_new_write(&changed)
        );
    }

    #[test]
    fn checkpoint_input_bounds_are_enforced_before_database_work() {
        let fixture = |key: Option<String>| CheckpointRequest {
            session_id: "session:one".to_owned(),
            parent_checkpoint_id: None,
            state: json!({"objective": "Continue"}),
            source_refs: vec![],
            idempotency_key: key,
        };
        assert!(validate_checkpoint_request(&fixture(Some("x".repeat(256)))).is_ok());
        assert!(validate_checkpoint_request(&fixture(Some("x".repeat(257)))).is_err());
        assert!(validate_checkpoint_request(&fixture(None)).is_ok());
        assert!(validate_checkpoint_request(&fixture(Some("bad\nkey".to_owned()))).is_err());
        assert!(
            validate_checkpoint_request(&fixture(Some(format!("implicit:{}", Uuid::nil()))))
                .is_err()
        );
        assert!(
            validate_checkpoint_request(&fixture(Some("implicit:caller-label".to_owned()))).is_ok()
        );

        let mut request = fixture(Some("bounded".to_owned()));
        request.session_id = "x".repeat(257);
        assert!(validate_checkpoint_request(&request).is_err());
        request.session_id = "session:one".to_owned();
        request.state = Value::String("not an object".to_owned());
        assert!(validate_checkpoint_request(&request).is_err());
        request.state = json!({"objective": "Continue"});
        request.source_refs = (0..65).map(|index| format!("Sources/{index}.md")).collect();
        assert!(validate_checkpoint_request(&request).is_err());
    }

    #[test]
    fn checkpoint_missing_key_uses_stable_legacy_scoped_identity() {
        let request: CheckpointRequest = serde_json::from_value(json!({
            "session_id": "session:implicit-one",
            "parent_checkpoint_id": null,
            "state": {"objective": "Preserve the optional-key contract"},
            "source_refs": ["Sources/Plan.md"]
        }))
        .expect("the HTTP request model keeps idempotency_key optional");
        let (effective_key, request_hash) = validate_checkpoint_request(&request).unwrap();
        assert_eq!(effective_key, implicit_checkpoint_idempotency_key(&request));
        assert!(effective_key.len() <= 256);
        let (replayed_key, replayed_hash) = validate_checkpoint_request(&request).unwrap();
        assert_eq!(effective_key, replayed_key);
        assert_eq!(request_hash, replayed_hash);

        let changed_payload = CheckpointRequest {
            state: json!({"objective": "A historically distinct checkpoint"}),
            ..request.clone()
        };
        let (changed_payload_key, changed_payload_hash) =
            validate_checkpoint_request(&changed_payload).unwrap();
        assert_ne!(effective_key, changed_payload_key);
        assert_ne!(request_hash, changed_payload_hash);

        let changed_session = CheckpointRequest {
            session_id: "session:implicit-two".to_owned(),
            ..request
        };
        let (changed_session_key, changed_session_hash) =
            validate_checkpoint_request(&changed_session).unwrap();
        assert_ne!(effective_key, changed_session_key);
        assert_eq!(request_hash, changed_session_hash);
        assert_ne!(
            checkpoint_entry_id_for_new_write(&changed_payload),
            checkpoint_entry_id_for_new_write(&changed_session)
        );
    }

    #[test]
    fn workspace_envelope_omits_empty_legacy_metadata() {
        let envelope = WorkspaceEnvelope::complete(json!({"evidence": []}));
        let value = serde_json::to_value(envelope).unwrap();
        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("complete")
        );
        for omitted in [
            "request_id",
            "session_id",
            "corpus_revision",
            "gaps",
            "freshness",
            "coverage",
            "conflicts",
            "ambiguities",
            "truncation",
        ] {
            assert!(value.get(omitted).is_none(), "{omitted} should be omitted");
        }
    }

    #[test]
    fn verbatim_matches_preserve_exact_lines_beyond_excerpt_window() {
        let identifier = "STRAYID-64000-07-deadbeef";
        let prefix = format!("{}\n", "x".repeat(2_500));
        let line = format!("literal identifier: {identifier}");
        let content = format!("{prefix}{line}\ntrailing material\n");
        let terms = verbatim_match_terms(&format!("Synthetic/records/0000007.md {identifier}"));

        assert_eq!(terms, vec![identifier]);
        let matches = extract_verbatim_matches(&content, &terms, 9, &"a".repeat(64));

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].line_no, 2);
        assert_eq!(matches[0].byte_start, prefix.len());
        assert_eq!(matches[0].byte_end, prefix.len() + line.len());
        assert_eq!(matches[0].text, line);
        assert_eq!(matches[0].version, 9);
        assert_eq!(
            matches[0].content_hash,
            format!("sha256:{}", "a".repeat(64))
        );
        assert!(!matches[0].truncated);
    }

    #[test]
    fn verbatim_response_budget_truncates_text_and_offsets_together() {
        let candidate = Candidate {
            entry_id: Uuid::nil(),
            path: "Synthetic/record.md".to_owned(),
            title: "Record".to_owned(),
            version: 1,
            updated_at: "2026-08-01T00:00:00Z".parse().expect("test timestamp"),
            content_sha256: "a".repeat(64),
            heading: String::new(),
            excerpt: "prefix".to_owned(),
            score: 10.0,
            lanes: vec!["exact".to_owned()],
            sections: vec![],
            verbatim_matches: vec![VerbatimMatch {
                line_no: 3,
                byte_start: 100,
                byte_end: 112,
                text: "abcdefghijkl".to_owned(),
                version: 1,
                content_hash: format!("sha256:{}", "a".repeat(64)),
                truncated: false,
            }],
            superseded_by: None,
        };
        let mut remaining = 5;

        let rendered = render_search_candidate(&candidate, &mut remaining);
        let source_match = rendered
            .get("verbatim_matches")
            .and_then(Value::as_array)
            .and_then(|matches| matches.first())
            .expect("verbatim match");

        assert_eq!(
            source_match.get("text").and_then(Value::as_str),
            Some("abcde")
        );
        assert_eq!(
            source_match.get("byte_start").and_then(Value::as_u64),
            Some(100)
        );
        assert_eq!(
            source_match.get("byte_end").and_then(Value::as_u64),
            Some(105)
        );
        assert_eq!(
            source_match.get("truncated").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(remaining, 0);
    }

    #[test]
    fn search_candidates_expose_identity_and_evidence_without_ranking_internals() {
        let candidate = Candidate {
            entry_id: Uuid::nil(),
            path: "Trips/Plan.md".to_owned(),
            title: "Trip plan".to_owned(),
            version: 4,
            updated_at: "2026-08-02T12:30:00Z".parse().expect("test timestamp"),
            content_sha256: "a".repeat(64),
            heading: "Rail".to_owned(),
            excerpt: "Take the train.".to_owned(),
            score: 12.5,
            lanes: vec!["lexical".to_owned(), "semantic".to_owned()],
            sections: vec![
                CandidateSection {
                    heading: "Rail".to_owned(),
                    excerpt: "Take the train.".to_owned(),
                    score: 12.5,
                },
                CandidateSection {
                    heading: "Backup".to_owned(),
                    excerpt: "Keep the bus as a fallback.".to_owned(),
                    score: 11.0,
                },
            ],
            verbatim_matches: vec![],
            superseded_by: None,
        };
        let mut remaining_verbatim_chars = MAX_VERBATIM_RESPONSE_CHARS;
        let rendered = render_search_candidate(&candidate, &mut remaining_verbatim_chars);
        assert_eq!(
            rendered.get("reference").and_then(Value::as_str),
            Some("entry:00000000-0000-0000-0000-000000000000")
        );
        assert_eq!(
            rendered.get("excerpt").and_then(Value::as_str),
            Some("Take the train.")
        );
        assert_eq!(
            rendered.get("updated_at").and_then(Value::as_str),
            Some("2026-08-02T12:30:00Z")
        );
        assert_eq!(
            rendered
                .get("additional_sections")
                .and_then(Value::as_array)
                .and_then(|sections| sections.first())
                .and_then(|section| section.get("excerpt"))
                .and_then(Value::as_str),
            Some("Keep the bus as a fallback.")
        );
        for omitted in ["entry_id", "content_sha256", "score", "lanes"] {
            assert!(
                rendered.get(omitted).is_none(),
                "{omitted} should be omitted"
            );
        }
        let lead = render_evidence_lead(&candidate);
        assert_eq!(
            lead.get("path").and_then(Value::as_str),
            Some("Trips/Plan.md")
        );
        for omitted in ["entry_id", "content_sha256", "score", "lanes", "excerpt"] {
            assert!(
                lead.get(omitted).is_none(),
                "{omitted} should be omitted from evidence leads"
            );
        }
    }

    #[test]
    fn search_text_budget_counts_additional_sections() {
        let mut candidate = Candidate {
            entry_id: Uuid::nil(),
            path: "Trips/Plan.md".to_owned(),
            title: "Trip plan".to_owned(),
            version: 1,
            updated_at: "2026-08-01T00:00:00Z".parse().expect("test timestamp"),
            content_sha256: "a".repeat(64),
            heading: "Primary".to_owned(),
            excerpt: "1234".to_owned(),
            score: 1.0,
            lanes: vec!["lexical".to_owned()],
            sections: vec![
                CandidateSection {
                    heading: "Primary".to_owned(),
                    excerpt: "1234".to_owned(),
                    score: 1.0,
                },
                CandidateSection {
                    heading: "Second".to_owned(),
                    excerpt: "5678".to_owned(),
                    score: 0.9,
                },
                CandidateSection {
                    heading: "Third".to_owned(),
                    excerpt: "90".to_owned(),
                    score: 0.8,
                },
            ],
            verbatim_matches: vec![],
            superseded_by: None,
        };
        let mut remaining = 5;
        assert!(truncate_candidate_evidence(&mut candidate, &mut remaining));
        assert_eq!(remaining, 0);
        assert_eq!(candidate.excerpt, "1234");
        assert_eq!(candidate.sections.len(), 2);
        assert_eq!(candidate.sections[1].excerpt, "5");
    }

    #[test]
    fn fair_share_candidate_allocation_preserves_the_last_query_floor() {
        let candidate_sets = (0_u128..16)
            .map(|query| {
                (0_u128..50)
                    .map(|rank| candidate_with_sections(query * 100 + rank + 1, &[16]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let (selected, truncated) = select_search_candidate_sets(&candidate_sets, true, 128);
        assert!(truncated);
        assert_eq!(
            selected.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![8; 16]
        );
        assert_eq!(selected[15][0].entry_id, candidate_sets[15][0].entry_id,);
    }

    #[test]
    fn fair_share_character_allocation_preserves_every_query_floor() {
        let candidate_sets = (0_u128..16)
            .map(|query| vec![candidate_with_sections(query + 1, &[2_400, 2_400, 2_400])])
            .collect::<Vec<_>>();
        let options = SearchBudgetOptions {
            fair_share: true,
            top1_hydration: false,
            char_cap: true,
            section_demotion_top_n: None,
            max_chars: 48_000,
        };
        let (views, truncated) =
            assemble_search_candidate_views(candidate_sets, &HashMap::new(), options);
        assert!(truncated);
        assert!(
            views
                .iter()
                .all(|query| { desired_search_evidence_chars(&query[0]) == 3_000 })
        );
        assert_eq!(
            views
                .iter()
                .flat_map(|query| query.iter())
                .map(desired_search_evidence_chars)
                .sum::<usize>(),
            48_000,
        );
    }

    #[test]
    fn hydration_is_batched_budgeted_and_degrades_in_request_order() {
        let candidate_sets = (0_u128..16)
            .map(|query| {
                vec![
                    candidate_with_sections(query * 10 + 1, &[2_400]),
                    candidate_with_sections(query * 10 + 2, &[2_400]),
                ]
            })
            .collect::<Vec<_>>();
        let hydration = candidate_sets
            .iter()
            .map(|query| {
                (
                    query[0].entry_id,
                    SearchHydration {
                        size_bytes: 24_000,
                        content: "x".repeat(24_000),
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        let options = SearchBudgetOptions {
            fair_share: true,
            top1_hydration: true,
            char_cap: true,
            section_demotion_top_n: Some(8),
            max_chars: 48_000,
        };
        let (views, truncated) =
            assemble_search_candidate_views(candidate_sets, &hydration, options);
        assert!(truncated);
        assert_eq!(
            views[0][0].representation,
            SearchRepresentation::CompleteSource,
        );
        assert_eq!(
            views[1][0].representation,
            SearchRepresentation::CompleteSource,
        );
        assert_eq!(
            views[2][0].representation,
            SearchRepresentation::PointerLead,
        );
        assert_eq!(
            views
                .iter()
                .flat_map(|query| query.iter())
                .map(|view| {
                    view.complete_text
                        .as_deref()
                        .map(str::chars)
                        .map(Iterator::count)
                        .unwrap_or_else(|| desired_search_evidence_chars(view))
                })
                .sum::<usize>(),
            48_000,
        );
        assert_eq!(
            views[0][1].representation,
            SearchRepresentation::PointerLead,
        );
    }

    #[test]
    fn section_demotion_renders_heading_leads_without_body_text() {
        let candidate_sets = vec![vec![
            candidate_with_sections(1, &[8, 8]),
            candidate_with_sections(2, &[8, 8]),
        ]];
        let options = SearchBudgetOptions {
            fair_share: false,
            top1_hydration: false,
            char_cap: true,
            section_demotion_top_n: Some(1),
            max_chars: 48_000,
        };
        let (views, truncated) =
            assemble_search_candidate_views(candidate_sets, &HashMap::new(), options);
        assert!(truncated);
        let mut remaining_verbatim_chars = MAX_VERBATIM_RESPONSE_CHARS;
        let rendered =
            render_budgeted_search_candidate(&views[0][1], &mut remaining_verbatim_chars);
        let lead = rendered["additional_sections"][0].as_object().unwrap();
        assert_eq!(
            lead.get("representation").and_then(Value::as_str),
            Some("heading_lead"),
        );
        assert!(lead.get("excerpt").is_none());
        assert_eq!(
            rendered.get("representation").and_then(Value::as_str),
            Some("excerpt"),
        );
    }

    #[test]
    fn checkpoint_budget_precedes_related_evidence() {
        let mut checkpoint = Some(json!({"text": "abcdefgh"}));
        let (truncated, evidence_tokens) = apply_checkpoint_budget(&mut checkpoint, 1);
        assert!(truncated);
        assert_eq!(evidence_tokens, 0);
        assert_eq!(
            checkpoint
                .as_ref()
                .and_then(|value| value.get("text"))
                .and_then(Value::as_str),
            Some("abcd")
        );
    }

    #[test]
    fn resume_delta_sources_follow_checkpoint_authoring_order() {
        let first = Uuid::now_v7();
        let second = Uuid::now_v7();
        let checkpoint = json!({
            "source_entries": [
                {
                    "entry_ref": format!("entry:{first}"),
                    "path": "Plans/First.md",
                    "version": 3,
                    "content_hash": format!("sha256:{}", "a".repeat(64))
                },
                {
                    "entry_ref": format!("entry:{second}"),
                    "path": "Plans/Second.md",
                    "version": 7,
                    "content_hash": format!("sha256:{}", "b".repeat(64))
                }
            ]
        });
        let changes = vec![
            json!({"path": "Plans/Second.md"}),
            json!({"path": "Plans/First.md"}),
        ];

        let (selected, overflow) = changed_checkpoint_sources(Some(&checkpoint), &changes);

        assert!(overflow.is_empty());
        assert_eq!(
            selected
                .iter()
                .map(|source| source.path.as_str())
                .collect::<Vec<_>>(),
            ["Plans/First.md", "Plans/Second.md"]
        );
        assert_eq!(selected[0].pinned_version, 3);
        assert_eq!(selected[1].pinned_sha256, "b".repeat(64));
    }

    #[test]
    fn resume_delta_source_limit_degrades_in_authoring_order() {
        let entries = (0..10)
            .map(|index| {
                json!({
                    "entry_ref": format!("entry:{}", Uuid::now_v7()),
                    "path": format!("Plans/{index}.md"),
                    "version": 1,
                    "content_hash": format!("sha256:{}", "c".repeat(64))
                })
            })
            .collect::<Vec<_>>();
        let changes = (0..10)
            .rev()
            .map(|index| json!({"path": format!("Plans/{index}.md")}))
            .collect::<Vec<_>>();

        let (selected, overflow) =
            changed_checkpoint_sources(Some(&json!({"source_entries": entries})), &changes);

        assert_eq!(selected.len(), RESUME_DELTA_SOURCE_LIMIT);
        assert_eq!(selected[0].path, "Plans/0.md");
        assert_eq!(selected[7].path, "Plans/7.md");
        assert_eq!(
            overflow
                .iter()
                .map(|source| source.path.as_str())
                .collect::<Vec<_>>(),
            ["Plans/8.md", "Plans/9.md"]
        );
    }

    #[test]
    fn resume_unified_diff_has_three_context_lines() {
        let before = "one\ntwo\nthree\nfour\nold\nsix\nseven\neight\nnine\n";
        let after = "one\ntwo\nthree\nfour\nnew\nsix\nseven\neight\nnine\n";

        let diff = unified_line_diff("Plans/Current.md", before, after);

        assert!(
            diff.starts_with("--- a/Plans/Current.md\n+++ b/Plans/Current.md\n@@ -2,7 +2,7 @@\n")
        );
        assert!(diff.contains(" four\n-old\n+new\n six\n seven\n eight\n"));
        assert!(!diff.contains(" one\n"));
        assert!(!diff.contains(" nine\n"));
    }

    #[test]
    fn resume_content_hash_mismatch_is_a_lineage_error() {
        let source = CheckpointSource {
            entry_id: Uuid::now_v7(),
            path: "Plans/Current.md".to_owned(),
            pinned_version: 1,
            pinned_sha256: "0".repeat(64),
        };

        let error = verify_resume_content_hash(&source, 1, &"0".repeat(64), "changed")
            .expect_err("mismatched content must fail");

        assert!(matches!(
            error,
            ApiError::Public {
                code: "checkpoint_lineage_error",
                ..
            }
        ));
    }

    #[test]
    fn imported_checkpoint_metadata_rebases_and_drops_origin_entry_ids() {
        let metadata = json!({
            "_brunn_import": {
                "format": "brunn-workspace-import-manifest@v1"
            },
            "client": {
                "kind": "checkpoint",
                "workspace_generation": 1_000_000,
                "source_entries": [{
                    "entry_ref": "entry:00000000-0000-0000-0000-000000000001",
                    "path": "Trips/Plan.md"
                }]
            }
        });
        let rebased = rebase_imported_checkpoint_metadata(metadata, 42);
        assert_eq!(rebased["client"]["workspace_generation"], 42);
        assert_eq!(rebased["client"]["origin_workspace_generation"], 1_000_000);
        assert!(
            rebased["client"]["source_entries"][0]
                .get("entry_ref")
                .is_none()
        );
    }

    #[test]
    fn evaluation_batches_are_bounded_and_ordered() {
        let content = "# Fixture";
        let document = EvalDocument {
            path: "Fixture.md".to_owned(),
            content: content.to_owned(),
            content_sha256: hex::encode(Sha256::digest(content.as_bytes())),
            media_type: markdown_media_type(),
        };
        let mut request = EvalImportRequest {
            schema: "brunn-eval-import@v1".to_owned(),
            run_id: "run".to_owned(),
            case_id: "case".to_owned(),
            authorization_scope: "eval:run/case".to_owned(),
            display_scope: "Fixture".to_owned(),
            access_mode: "read_write".to_owned(),
            documents: vec![document],
            delta_documents: vec![],
            seed_checkpoint: None,
            idempotency_key: "import".to_owned(),
            batch_index: Some(1),
            batch_count: Some(3),
        };
        assert_eq!(evaluation_batch(&request).unwrap(), Some((1, 3)));
        assert!(validate_eval_import(&request).is_ok());
        request.batch_count = Some(1);
        assert!(validate_eval_import(&request).is_err());
        request.batch_index = Some(0);
        request.batch_count = None;
        assert!(validate_eval_import(&request).is_err());
    }

    #[test]
    fn idempotency_keys_are_bounded_and_printable() {
        assert!(validate_idempotency_key("capture:stable").is_ok());
        assert!(validate_idempotency_key("").is_err());
        assert!(validate_idempotency_key(&"x".repeat(257)).is_err());
        assert!(validate_idempotency_key("bad\nkey").is_err());
    }

    fn prepared_markdown_fixture(metadata: Value, force_new_version: bool) -> PreparedMarkdown {
        let content = "# Morning briefing - 2026-08-01\n";
        PreparedMarkdown {
            entry_id_hint: None,
            path: "Briefings/2026/Morning briefing - 2026-08-01.md".to_owned(),
            title: "Morning briefing - 2026-08-01".to_owned(),
            content: content.to_owned(),
            content_sha256: hex::encode(Sha256::digest(content.as_bytes())),
            media_type: "text/markdown".to_owned(),
            metadata,
            chunks: Vec::new(),
            embeddings: Vec::new(),
            expected_version: None,
            tier_a_history_stage: None,
            frontmatter: DerivedFrontmatter::default(),
            force_new_version,
        }
    }

    fn prepared_task_fixture(
        task_id: Uuid,
        content: &str,
        metadata: Value,
        expected_version: Option<i64>,
    ) -> PreparedMarkdown {
        let path = format!(".brunn/tasks/{task_id}.md");
        let normalized = normalize_document(&path, content);
        PreparedMarkdown {
            entry_id_hint: None,
            path,
            title: normalized.title,
            content: content.to_owned(),
            content_sha256: normalized
                .content_hash
                .trim_start_matches("sha256:")
                .to_owned(),
            media_type: markdown_media_type(),
            metadata,
            chunks: Vec::new(),
            embeddings: Vec::new(),
            expected_version,
            tier_a_history_stage: None,
            frontmatter: DerivedFrontmatter::default(),
            force_new_version: true,
        }
    }

    fn prepared_checkpoint_fixture(
        request: &CheckpointRequest,
        effective_idempotency_key: &str,
        request_hash: &str,
        pinned_generation: i64,
    ) -> (String, String, PreparedMarkdown) {
        prepared_checkpoint_fixture_for_id(
            checkpoint_entry_id_for_new_write(request),
            request,
            Some(effective_idempotency_key),
            Some(request_hash),
            pinned_generation,
        )
    }

    fn prepared_legacy_checkpoint_fixture(
        request: &CheckpointRequest,
        pinned_generation: i64,
    ) -> (String, String, PreparedMarkdown) {
        prepared_checkpoint_fixture_for_id(
            deterministic_checkpoint_id(request),
            request,
            request.idempotency_key.as_deref(),
            None,
            pinned_generation,
        )
    }

    fn prepared_checkpoint_fixture_for_id(
        checkpoint_id: Uuid,
        request: &CheckpointRequest,
        metadata_idempotency_key: Option<&str>,
        request_hash: Option<&str>,
        pinned_generation: i64,
    ) -> (String, String, PreparedMarkdown) {
        let checkpoint_ref = format!("checkpoint:{checkpoint_id}");
        let path = format!(".brunn/checkpoints/{checkpoint_id}.md");
        let content =
            render_checkpoint_markdown(checkpoint_id, pinned_generation, request, &[]).unwrap();
        let normalized = normalize_document(&path, &content);
        let chunks = normalized.chunks;
        let embeddings = vec![None; chunks.len()];
        let mut metadata = json!({
            "kind": "checkpoint",
            "checkpoint_ref": checkpoint_ref,
            "workspace_generation": pinned_generation,
            "session_id": request.session_id,
            "parent_checkpoint_ref": request.parent_checkpoint_id,
            "checkpoint_state": request.state,
            "project": request.state.get("project").cloned().unwrap_or(Value::Null),
            "source_refs": request.source_refs,
            "source_entries": []
        });
        if let Some(idempotency_key) = metadata_idempotency_key {
            metadata
                .as_object_mut()
                .expect("checkpoint fixture metadata is an object")
                .insert(
                    "_brunn_idempotency_hash".to_owned(),
                    json!(hex::encode(Sha256::digest(idempotency_key.as_bytes()))),
                );
        }
        if let Some(request_hash) = request_hash {
            let metadata = metadata
                .as_object_mut()
                .expect("checkpoint fixture metadata is an object");
            metadata.insert(
                "pinned_workspace_generation".to_owned(),
                json!(pinned_generation),
            );
            metadata.insert("resulting_workspace_generation".to_owned(), Value::Null);
            metadata.insert(
                "request_hash".to_owned(),
                json!(format!("sha256:{request_hash}")),
            );
            metadata.insert("operation_kind".to_owned(), json!("checkpoint"));
        }
        (
            checkpoint_ref.clone(),
            path.clone(),
            PreparedMarkdown {
                entry_id_hint: Some(checkpoint_id),
                path,
                title: normalized.title,
                content,
                content_sha256: normalized
                    .content_hash
                    .trim_start_matches("sha256:")
                    .to_owned(),
                media_type: markdown_media_type(),
                metadata,
                chunks,
                embeddings,
                expected_version: Some(0),
                tier_a_history_stage: None,
                frontmatter: DerivedFrontmatter::default(),
                force_new_version: false,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    async fn checkpoint_attempt_with_auth(
        pool: &sqlx::PgPool,
        user_id: Uuid,
        key: &str,
        request_hash: &str,
        checkpoint_ref: &str,
        path: &str,
        pinned_generation: i64,
        prepared: PreparedMarkdown,
        auth: Option<&AuthContext>,
    ) -> ApiResult<Value> {
        let mut tx = pool.begin().await?;
        let credential_id = auth.map(|auth| auth.credential_id.0);
        if let Some(auth) = auth {
            set_context(&mut tx, auth).await?;
        }
        lock_checkpoint_idempotency(&mut tx, user_id, key).await?;
        if let Some(receipt) =
            replay_checkpoint_receipt_in_tx(&mut tx, user_id, key, request_hash).await?
        {
            tx.commit().await?;
            return Ok(receipt);
        }
        let result = commit_checkpoint_in_tx(
            &mut tx,
            user_id,
            credential_id,
            key,
            request_hash,
            checkpoint_ref,
            path,
            pinned_generation,
            vec![],
            prepared,
        )
        .await?;
        tx.commit().await?;
        Ok(result.receipt)
    }

    #[allow(clippy::too_many_arguments)]
    async fn checkpoint_attempt(
        pool: &sqlx::PgPool,
        user_id: Uuid,
        key: &str,
        request_hash: &str,
        checkpoint_ref: &str,
        path: &str,
        pinned_generation: i64,
        prepared: PreparedMarkdown,
    ) -> ApiResult<Value> {
        checkpoint_attempt_with_auth(
            pool,
            user_id,
            key,
            request_hash,
            checkpoint_ref,
            path,
            pinned_generation,
            prepared,
            None,
        )
        .await
    }

    #[tokio::test]
    async fn checkpoint_receipts_are_atomic_replay_exact_and_concurrency_safe() {
        let Some(database_url) = std::env::var("BRUNN_TEST_DATABASE_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("BRUNN_TEST_DATABASE_URL is unset; skipping checkpoint receipt test");
            return;
        };
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(6)
            .connect(&database_url)
            .await
            .expect("connect to disposable Postgres");
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .expect("apply Brunn migrations");
        let user_id = Uuid::now_v7();
        sqlx::query("INSERT INTO brunn.users (id,external_ref,display_name) VALUES ($1,$2,$3)")
            .bind(user_id)
            .bind(format!("checkpoint-receipt-test:{user_id}"))
            .bind("Checkpoint receipt test")
            .execute(&pool)
            .await
            .expect("insert checkpoint test user");
        let checkpoint_credential_id = Uuid::now_v7();
        sqlx::query(
            r#"
            INSERT INTO brunn.api_credentials (
              id,user_id,label,token_hash,capabilities
            ) VALUES ($1,$2,'Checkpoint receipt writer',$3,ARRAY['checkpoint'])
            "#,
        )
        .bind(checkpoint_credential_id)
        .bind(user_id)
        .bind(hex::encode(Sha256::digest(
            format!("checkpoint-receipt:{checkpoint_credential_id}").as_bytes(),
        )))
        .execute(&pool)
        .await
        .expect("insert checkpoint receipt credential");
        sqlx::query(
            r#"
            INSERT INTO brunn.credential_scope_grants (
              credential_id,user_id,scope_id
            )
            SELECT $1,$2,id FROM brunn.scopes
            WHERE user_id=$2 AND scope_ref='scope:root'
            "#,
        )
        .bind(checkpoint_credential_id)
        .bind(user_id)
        .execute(&pool)
        .await
        .expect("grant checkpoint receipt root scope");
        let checkpoint_auth = AuthContext {
            credential_id: CredentialId(checkpoint_credential_id),
            user_id: UserId(user_id),
            capabilities: ["checkpoint".to_owned()].into_iter().collect(),
            scope_refs: vec!["scope:root".to_owned()],
            read_only: false,
        };

        let request = CheckpointRequest {
            session_id: "session:correlation-only".to_owned(),
            parent_checkpoint_id: None,
            state: json!({"objective": "Test durable replay"}),
            source_refs: vec![],
            idempotency_key: Some("checkpoint-concurrent".to_owned()),
        };
        let (key, request_hash) = validate_checkpoint_request(&request).unwrap();
        let pinned_generation = 0;
        let (checkpoint_ref, path, prepared) =
            prepared_checkpoint_fixture(&request, &key, &request_hash, pinned_generation);
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let run = |prepared: PreparedMarkdown| {
            let pool = pool.clone();
            let key = key.clone();
            let request_hash = request_hash.clone();
            let checkpoint_ref = checkpoint_ref.clone();
            let path = path.clone();
            let barrier = barrier.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                checkpoint_attempt(
                    &pool,
                    user_id,
                    &key,
                    &request_hash,
                    &checkpoint_ref,
                    &path,
                    pinned_generation,
                    prepared,
                )
                .await
                .expect("concurrent checkpoint attempt")
            })
        };
        let first_task = run(prepared.clone());
        let second_task = run(prepared.clone());
        let first = first_task.await.expect("first task joins");
        let second = second_task.await.expect("second task joins");
        assert_eq!(first, second, "concurrent replay returns the exact receipt");
        let replay_request = CheckpointRequest {
            session_id: "session:after-client-restart".to_owned(),
            ..request.clone()
        };
        let (_, replay_hash) = validate_checkpoint_request(&replay_request).unwrap();
        assert_eq!(request_hash, replay_hash);
        let cross_session_replay = checkpoint_attempt(
            &pool,
            user_id,
            &key,
            &replay_hash,
            &checkpoint_ref,
            &path,
            pinned_generation,
            prepared.clone(),
        )
        .await
        .expect("cross-session checkpoint replay");
        assert_eq!(first, cross_session_replay);
        let first_envelope =
            checkpoint_envelope(&request.session_id, first.clone(), false).unwrap();
        let replay_envelope =
            checkpoint_envelope(&replay_request.session_id, cross_session_replay, true).unwrap();
        assert_eq!(first_envelope.data, replay_envelope.data);
        assert_eq!(
            first_envelope.session_id.as_deref(),
            Some("session:correlation-only")
        );
        assert_eq!(
            replay_envelope.session_id.as_deref(),
            Some("session:after-client-restart"),
            "a durable replay carries the current correlation ID"
        );

        for (table, query, count) in [
            (
                "entries",
                "SELECT count(*) FROM brunn.entries WHERE user_id=$1",
                1_i64,
            ),
            (
                "entry_versions",
                "SELECT count(*) FROM brunn.entry_versions WHERE user_id=$1",
                1,
            ),
            (
                "workspace_changes",
                "SELECT count(*) FROM brunn.workspace_changes WHERE user_id=$1",
                1,
            ),
            (
                "jobs",
                "SELECT count(*) FROM brunn.jobs WHERE user_id=$1",
                1,
            ),
            (
                "workspace_idempotency_receipts",
                "SELECT count(*) FROM brunn.workspace_idempotency_receipts WHERE user_id=$1",
                1,
            ),
        ] {
            let actual = sqlx::query_scalar::<_, i64>(query)
                .bind(user_id)
                .fetch_one(&pool)
                .await
                .unwrap();
            assert_eq!(actual, count, "unexpected {table} count");
        }

        // A committed response can be lost by the caller; the next attempt is
        // reconstructed from the atomic durable receipt, byte-for-byte.
        let lost_response_replay = checkpoint_attempt(
            &pool,
            user_id,
            &key,
            &request_hash,
            &checkpoint_ref,
            &path,
            pinned_generation,
            prepared,
        )
        .await
        .expect("lost-response replay");
        assert_eq!(first, lost_response_replay);

        let mut conflict_tx = pool.begin().await.unwrap();
        lock_checkpoint_idempotency(&mut conflict_tx, user_id, &key)
            .await
            .unwrap();
        let conflict =
            replay_checkpoint_receipt_in_tx(&mut conflict_tx, user_id, &key, &"f".repeat(64))
                .await
                .unwrap_err();
        assert!(matches!(
            conflict,
            ApiError::Public {
                status: StatusCode::CONFLICT,
                code: "idempotency_conflict",
                ..
            }
        ));
        conflict_tx.rollback().await.unwrap();

        // A checkpoint written before durable receipts used session_id in its
        // path identity. Adoption reconstructs that retired identity with the
        // immutable session stored on the row, while binding the receipt to
        // the new session-independent canonical request hash.
        let legacy_pinned_generation = sqlx::query_scalar::<_, Option<i64>>(
            "SELECT max(generation) FROM brunn.workspace_changes WHERE user_id=$1",
        )
        .bind(user_id)
        .fetch_one(&pool)
        .await
        .unwrap()
        .unwrap();
        let legacy_request = CheckpointRequest {
            session_id: "session:legacy-original".to_owned(),
            parent_checkpoint_id: None,
            state: json!({"objective": "Adopt an old checkpoint receipt"}),
            source_refs: vec![],
            idempotency_key: Some("checkpoint-legacy-adoption".to_owned()),
        };
        let (legacy_key, legacy_hash) = validate_checkpoint_request(&legacy_request).unwrap();
        let legacy_key = legacy_key.to_owned();
        let (legacy_ref, legacy_path, legacy_prepared) =
            prepared_legacy_checkpoint_fixture(&legacy_request, legacy_pinned_generation);
        let mut legacy_write_tx = pool.begin().await.unwrap();
        upsert_markdown_in_tx(&mut legacy_write_tx, user_id, None, legacy_prepared)
            .await
            .expect("write pre-receipt checkpoint fixture");
        legacy_write_tx.commit().await.unwrap();

        let legacy_conflicting_request = CheckpointRequest {
            session_id: "session:legacy-conflicting-retry".to_owned(),
            state: json!({"objective": "A different operation under the same key"}),
            ..legacy_request.clone()
        };
        let (_, legacy_conflicting_hash) =
            validate_checkpoint_request(&legacy_conflicting_request).unwrap();
        let mut legacy_conflict_tx = pool.begin().await.unwrap();
        set_context(&mut legacy_conflict_tx, &checkpoint_auth)
            .await
            .unwrap();
        lock_checkpoint_idempotency(&mut legacy_conflict_tx, user_id, &legacy_key)
            .await
            .unwrap();
        let legacy_conflict = adopt_legacy_checkpoint_receipt_in_tx(
            &mut legacy_conflict_tx,
            user_id,
            None,
            &legacy_key,
            &legacy_conflicting_hash,
            &legacy_conflicting_request,
        )
        .await
        .unwrap_err();
        assert!(
            matches!(
                legacy_conflict,
                ApiError::Public {
                    status: StatusCode::CONFLICT,
                    code: "idempotency_conflict",
                    ..
                }
            ),
            "unexpected legacy checkpoint conflict: {legacy_conflict:?}"
        );
        legacy_conflict_tx.rollback().await.unwrap();

        let legacy_replay_request = CheckpointRequest {
            session_id: "session:legacy-replay-after-restart".to_owned(),
            ..legacy_request.clone()
        };
        let (_, legacy_replay_hash) = validate_checkpoint_request(&legacy_replay_request).unwrap();
        assert_eq!(legacy_hash, legacy_replay_hash);
        let mut adoption_tx = pool.begin().await.unwrap();
        set_context(&mut adoption_tx, &checkpoint_auth)
            .await
            .unwrap();
        lock_checkpoint_idempotency(&mut adoption_tx, user_id, &legacy_key)
            .await
            .unwrap();
        let adopted = adopt_legacy_checkpoint_receipt_in_tx(
            &mut adoption_tx,
            user_id,
            None,
            &legacy_key,
            &legacy_replay_hash,
            &legacy_replay_request,
        )
        .await
        .expect("legacy adoption lookup")
        .expect("legacy checkpoint is adopted");
        adoption_tx.commit().await.unwrap();
        assert!(!adopted.created);
        assert_eq!(adopted.receipt["checkpoint_ref"], legacy_ref);
        assert_eq!(adopted.receipt["path"], legacy_path);
        let adopted_envelope = checkpoint_envelope(
            &legacy_replay_request.session_id,
            adopted.receipt.clone(),
            true,
        )
        .unwrap();
        assert_eq!(
            adopted_envelope.session_id.as_deref(),
            Some("session:legacy-replay-after-restart")
        );
        let mut adopted_replay_tx = pool.begin().await.unwrap();
        lock_checkpoint_idempotency(&mut adopted_replay_tx, user_id, &legacy_key)
            .await
            .unwrap();
        let adopted_replay = replay_checkpoint_receipt_in_tx(
            &mut adopted_replay_tx,
            user_id,
            &legacy_key,
            &legacy_hash,
        )
        .await
        .unwrap()
        .expect("adopted receipt replays durably");
        adopted_replay_tx.commit().await.unwrap();
        assert_eq!(adopted.receipt, adopted_replay);

        // Direct HTTP clients historically could omit idempotency_key. The
        // API derives a stable session-scoped key and still stores an atomic
        // receipt, so an exact retry succeeds without changing the public
        // optional field contract.
        let implicit_pinned_generation = sqlx::query_scalar::<_, Option<i64>>(
            "SELECT max(generation) FROM brunn.workspace_changes WHERE user_id=$1",
        )
        .bind(user_id)
        .fetch_one(&pool)
        .await
        .unwrap()
        .unwrap();
        let implicit_request = CheckpointRequest {
            session_id: "session:implicit-route".to_owned(),
            parent_checkpoint_id: None,
            state: json!({"objective": "Checkpoint without an explicit key"}),
            source_refs: vec![],
            idempotency_key: None,
        };
        let (implicit_key, implicit_hash) = validate_checkpoint_request(&implicit_request).unwrap();
        let (implicit_ref, implicit_path, implicit_prepared) = prepared_checkpoint_fixture(
            &implicit_request,
            &implicit_key,
            &implicit_hash,
            implicit_pinned_generation,
        );
        let implicit_receipt = checkpoint_attempt(
            &pool,
            user_id,
            &implicit_key,
            &implicit_hash,
            &implicit_ref,
            &implicit_path,
            implicit_pinned_generation,
            implicit_prepared.clone(),
        )
        .await
        .expect("checkpoint without an explicit idempotency key");
        let (implicit_replay_key, implicit_replay_hash) =
            validate_checkpoint_request(&implicit_request).unwrap();
        let implicit_replay = checkpoint_attempt(
            &pool,
            user_id,
            &implicit_replay_key,
            &implicit_replay_hash,
            &implicit_ref,
            &implicit_path,
            implicit_pinned_generation,
            implicit_prepared,
        )
        .await
        .expect("exact missing-key replay");
        assert_eq!(implicit_receipt, implicit_replay);
        let implicit_other_session = CheckpointRequest {
            session_id: "session:implicit-route-other".to_owned(),
            ..implicit_request.clone()
        };
        let (implicit_other_key, implicit_other_hash) =
            validate_checkpoint_request(&implicit_other_session).unwrap();
        assert_ne!(implicit_key, implicit_other_key);
        assert_eq!(implicit_hash, implicit_other_hash);

        // Pre-receipt checkpoints without caller keys have no legacy key hash.
        // Adopt them by their exact deterministic path for the same session;
        // another session remains a distinct historical operation.
        let implicit_legacy_pinned = sqlx::query_scalar::<_, Option<i64>>(
            "SELECT max(generation) FROM brunn.workspace_changes WHERE user_id=$1",
        )
        .bind(user_id)
        .fetch_one(&pool)
        .await
        .unwrap()
        .unwrap();
        let implicit_legacy_request = CheckpointRequest {
            session_id: "session:implicit-legacy".to_owned(),
            parent_checkpoint_id: None,
            state: json!({"objective": "Adopt a missing-key legacy checkpoint"}),
            source_refs: vec![],
            idempotency_key: None,
        };
        let (implicit_legacy_key, implicit_legacy_hash) =
            validate_checkpoint_request(&implicit_legacy_request).unwrap();
        let (implicit_legacy_ref, implicit_legacy_path, implicit_legacy_prepared) =
            prepared_legacy_checkpoint_fixture(&implicit_legacy_request, implicit_legacy_pinned);
        let mut implicit_legacy_write_tx = pool.begin().await.unwrap();
        upsert_markdown_in_tx(
            &mut implicit_legacy_write_tx,
            user_id,
            None,
            implicit_legacy_prepared,
        )
        .await
        .expect("write missing-key legacy checkpoint fixture");
        implicit_legacy_write_tx.commit().await.unwrap();

        let implicit_legacy_other_session = CheckpointRequest {
            session_id: "session:implicit-legacy-other".to_owned(),
            ..implicit_legacy_request.clone()
        };
        let (implicit_legacy_other_key, implicit_legacy_other_hash) =
            validate_checkpoint_request(&implicit_legacy_other_session).unwrap();
        let mut implicit_other_adoption_tx = pool.begin().await.unwrap();
        set_context(&mut implicit_other_adoption_tx, &checkpoint_auth)
            .await
            .unwrap();
        lock_checkpoint_idempotency(
            &mut implicit_other_adoption_tx,
            user_id,
            &implicit_legacy_other_key,
        )
        .await
        .unwrap();
        let implicit_other_adoption = adopt_legacy_checkpoint_receipt_in_tx(
            &mut implicit_other_adoption_tx,
            user_id,
            None,
            &implicit_legacy_other_key,
            &implicit_legacy_other_hash,
            &implicit_legacy_other_session,
        )
        .await
        .unwrap();
        implicit_other_adoption_tx.commit().await.unwrap();
        assert!(implicit_other_adoption.is_none());

        let mut implicit_adoption_tx = pool.begin().await.unwrap();
        set_context(&mut implicit_adoption_tx, &checkpoint_auth)
            .await
            .unwrap();
        lock_checkpoint_idempotency(&mut implicit_adoption_tx, user_id, &implicit_legacy_key)
            .await
            .unwrap();
        let implicit_adopted = adopt_legacy_checkpoint_receipt_in_tx(
            &mut implicit_adoption_tx,
            user_id,
            None,
            &implicit_legacy_key,
            &implicit_legacy_hash,
            &implicit_legacy_request,
        )
        .await
        .expect("missing-key legacy adoption lookup")
        .expect("missing-key legacy checkpoint is adopted");
        implicit_adoption_tx.commit().await.unwrap();
        assert_eq!(
            implicit_adopted.receipt["checkpoint_ref"],
            implicit_legacy_ref
        );
        assert_eq!(implicit_adopted.receipt["path"], implicit_legacy_path);

        // Pin a generation, allow another writer to publish, then commit a
        // second checkpoint using the same session correlation ID. The two
        // generation meanings stay explicit and replay-stable.
        let pinned_before_interleaved = sqlx::query_scalar::<_, Option<i64>>(
            "SELECT max(generation) FROM brunn.workspace_changes WHERE user_id=$1",
        )
        .bind(user_id)
        .fetch_one(&pool)
        .await
        .unwrap()
        .unwrap();
        let mut writer_tx = pool.begin().await.unwrap();
        let writer = upsert_markdown_in_tx(
            &mut writer_tx,
            user_id,
            None,
            prepared_markdown_fixture(json!({"kind": "interleaved"}), false),
        )
        .await
        .unwrap();
        writer_tx.commit().await.unwrap();
        let writer_generation = writer.generation.unwrap();

        let interleaved_request = CheckpointRequest {
            state: json!({"objective": "Checkpoint after an interleaved writer"}),
            idempotency_key: Some("checkpoint-interleaved".to_owned()),
            ..request
        };
        let (interleaved_key, interleaved_hash) =
            validate_checkpoint_request(&interleaved_request).unwrap();
        let (interleaved_ref, interleaved_path, interleaved_prepared) = prepared_checkpoint_fixture(
            &interleaved_request,
            &interleaved_key,
            &interleaved_hash,
            pinned_before_interleaved,
        );
        let interleaved = checkpoint_attempt(
            &pool,
            user_id,
            &interleaved_key,
            &interleaved_hash,
            &interleaved_ref,
            &interleaved_path,
            pinned_before_interleaved,
            interleaved_prepared.clone(),
        )
        .await
        .unwrap();
        assert_eq!(
            interleaved["pinned_workspace_generation"],
            pinned_before_interleaved
        );
        let resulting = interleaved["resulting_workspace_generation"]
            .as_i64()
            .unwrap();
        assert!(resulting > writer_generation);
        assert_eq!(interleaved["workspace_generation"], resulting);
        let interleaved_replay = checkpoint_attempt(
            &pool,
            user_id,
            &interleaved_key,
            &interleaved_hash,
            &interleaved_ref,
            &interleaved_path,
            resulting,
            interleaved_prepared,
        )
        .await
        .unwrap();
        assert_eq!(interleaved, interleaved_replay);

        let checkpoint_count = sqlx::query_scalar::<_, i64>(
            "SELECT count(*) FROM brunn.entries WHERE user_id=$1 AND path LIKE '.brunn/checkpoints/%'",
        )
        .bind(user_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(
            checkpoint_count, 5,
            "session_id is correlation and does not limit checkpoints"
        );

        // Receipts are immutable during normal operation, but canonical
        // account deletion disables user-table triggers for the purge and
        // discovers this table by its user_id column. Verify both the receipt
        // and its referenced entry rows are removed without weakening normal
        // immutability.
        let immutable_delete =
            sqlx::query("DELETE FROM brunn.workspace_idempotency_receipts WHERE user_id=$1")
                .bind(user_id)
                .execute(&pool)
                .await
                .expect_err("ordinary receipt deletion must be rejected");
        let immutable_delete_code = immutable_delete
            .as_database_error()
            .and_then(|error| error.code())
            .map(|code| code.into_owned());
        assert_eq!(immutable_delete_code.as_deref(), Some("55000"));
        let purge_credential_id = Uuid::now_v7();
        sqlx::query(
            r#"
            INSERT INTO brunn.api_credentials (
              id,user_id,label,token_hash,capabilities
            ) VALUES ($1,$2,'Checkpoint purge test',$3,ARRAY['checkpoint','status'])
            "#,
        )
        .bind(purge_credential_id)
        .bind(user_id)
        .bind(hex::encode(Sha256::digest(
            format!("checkpoint-purge:{purge_credential_id}").as_bytes(),
        )))
        .execute(&pool)
        .await
        .unwrap();
        let deletion_request_id = Uuid::now_v7();
        sqlx::query(
            r#"
            INSERT INTO brunn.account_deletion_requests (
              id,user_id,requested_by_credential_id,status,confirmation_hash,
              reason,backup_expiry_due_at
            ) VALUES (
              $1,$2,$3,'queued',$4,'checkpoint receipt purge test',
              clock_timestamp() + interval '1 day'
            )
            "#,
        )
        .bind(deletion_request_id)
        .bind(user_id)
        .bind(purge_credential_id)
        .bind("b".repeat(64))
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            r#"
            UPDATE brunn.users
            SET account_status='deleting',deletion_requested_at=clock_timestamp()
            WHERE id=$1
            "#,
        )
        .bind(user_id)
        .execute(&pool)
        .await
        .expect("activate account deletion fence");
        let purge_result =
            sqlx::query_scalar::<_, Value>("SELECT brunn.purge_account_user_rows($1)")
                .bind(user_id)
                .fetch_one(&pool)
                .await
                .expect("purge checkpoint receipt owner");
        assert_eq!(
            purge_result["workspace_idempotency_receipts"], 5,
            "all checkpoint receipts are included in schema-derived account purge"
        );
        for (table, query) in [
            (
                "workspace_idempotency_receipts",
                "SELECT count(*) FROM brunn.workspace_idempotency_receipts WHERE user_id=$1",
            ),
            (
                "entries",
                "SELECT count(*) FROM brunn.entries WHERE user_id=$1",
            ),
        ] {
            let survivors = sqlx::query_scalar::<_, i64>(query)
                .bind(user_id)
                .fetch_one(&pool)
                .await
                .unwrap();
            assert_eq!(survivors, 0, "{table} rows survived account purge");
        }
    }

    #[tokio::test]
    async fn task_entries_rebuild_projection_version_metadata_and_link_checkpoint_projects() {
        let Some(database_url) = std::env::var("BRUNN_TEST_DATABASE_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("BRUNN_TEST_DATABASE_URL is unset; skipping task storage test");
            return;
        };
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(2)
            .connect(&database_url)
            .await
            .expect("connect to disposable Postgres");
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .expect("apply Brunn migrations");
        let user_id = Uuid::now_v7();
        sqlx::query("INSERT INTO brunn.users (id,external_ref,display_name) VALUES ($1,$2,$3)")
            .bind(user_id)
            .bind(format!("task-storage-test:{user_id}"))
            .bind("Task storage test")
            .execute(&pool)
            .await
            .expect("insert task storage user");
        let credential_id = Uuid::now_v7();
        sqlx::query(
            r#"
            INSERT INTO brunn.api_credentials (
              id,user_id,label,token_hash,capabilities
            ) VALUES ($1,$2,'Task storage writer',$3,ARRAY['task.read','task.write'])
            "#,
        )
        .bind(credential_id)
        .bind(user_id)
        .bind(format!("task-storage-token-{credential_id}"))
        .execute(&pool)
        .await
        .expect("insert task storage credential");
        sqlx::query(
            r#"
            INSERT INTO brunn.credential_scope_grants (
              credential_id,user_id,scope_id
            )
            SELECT $1,$2,id
            FROM brunn.scopes
            WHERE user_id=$2 AND scope_ref='scope:root'
            "#,
        )
        .bind(credential_id)
        .bind(user_id)
        .execute(&pool)
        .await
        .expect("grant task storage credential root scope");
        let checkpoint_auth = AuthContext {
            credential_id: CredentialId(credential_id),
            user_id: UserId(user_id),
            capabilities: ["task.read", "task.write"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            scope_refs: vec!["scope:root".to_owned()],
            read_only: false,
        };
        sqlx::query(
            r#"
            INSERT INTO brunn.task_projects (
              user_id,slug,title,hub_path,repo_path,created_by
            ) VALUES (
              $1,'brunn','Brunn',
              'sources/Projects/Brunn/Brunn.md',
              '/Volumes/NyxFastData/dev/projects/brunn','owner'
            )
            "#,
        )
        .bind(user_id)
        .execute(&pool)
        .await
        .expect("register task checkpoint project");

        let task_id = Uuid::now_v7();
        let content = "# Downgrade Charlemagne\n\nPreserve these exact task bytes.\n";
        let sourced = |value: Value, source: &str| {
            json!({
                "value": value,
                "source": source,
                "set_at": "2026-08-27T07:00:00Z"
            })
        };
        let open_task = json!({
            "kind": "task",
            "schema": "task.v1",
            "task": {
                "id": task_id,
                "title": "Downgrade Charlemagne",
                "status": sourced(json!("open"), "owner"),
                "project": sourced(json!("brunn"), "agent:codex"),
                "soft_due": sourced(json!("2026-08-31"), "agent:codex"),
                "cost_of_delay": sourced(json!({
                    "amount_cents": 700,
                    "per": "week",
                    "since": "2026-08-01"
                }), "agent:codex"),
                "required_contexts": sourced(json!(["home", "online"]), "owner"),
                "today_pin": sourced(json!("2026-08-27"), "owner"),
                "provenance": {
                    "captured_by": "agent:codex",
                    "captured_from": "entry:test",
                    "created_at": "2026-08-27T07:00:00Z"
                }
            }
        });
        let mut tx = pool.begin().await.expect("begin canonical task write");
        let first = upsert_markdown_in_tx(
            &mut tx,
            user_id,
            None,
            prepared_task_fixture(task_id, content, open_task.clone(), Some(0)),
        )
        .await
        .expect("write canonical task");
        tx.commit().await.expect("commit canonical task");
        assert_eq!(first.version, 1);

        let stored_projection: (i64, String, Option<i64>, Option<String>, Vec<String>) =
            sqlx::query_as(
                r#"
                SELECT entry_version,status,cost_amount_cents,cost_period,required_contexts
                FROM brunn.task_index
                WHERE user_id=$1 AND task_id=$2
                "#,
            )
            .bind(user_id)
            .bind(task_id)
            .fetch_one(&pool)
            .await
            .expect("read canonical task projection");
        assert_eq!(stored_projection.0, 1);
        assert_eq!(stored_projection.1, "open");
        assert_eq!(stored_projection.2, Some(700));
        assert_eq!(stored_projection.3.as_deref(), Some("week"));
        assert_eq!(stored_projection.4, ["home", "online"]);

        sqlx::query("DELETE FROM brunn.task_index WHERE user_id=$1 AND task_id=$2")
            .bind(user_id)
            .bind(task_id)
            .execute(&pool)
            .await
            .expect("simulate a missing rebuildable projection");
        let mut no_op_replay = prepared_task_fixture(task_id, content, open_task, Some(1));
        no_op_replay.force_new_version = false;
        let mut tx = pool.begin().await.expect("begin projection rebuild replay");
        let rebuilt = upsert_markdown_in_tx(&mut tx, user_id, None, no_op_replay)
            .await
            .expect("generic replay rebuilds task projection");
        tx.commit().await.expect("commit projection rebuild replay");
        assert!(rebuilt.no_op);
        assert_eq!(
            sqlx::query_scalar::<_, i64>(
                "SELECT count(*) FROM brunn.task_index WHERE user_id=$1 AND task_id=$2",
            )
            .bind(user_id)
            .bind(task_id)
            .fetch_one(&pool)
            .await
            .unwrap(),
            1
        );

        let done_task = json!({
            "_brunn_import": {
                "format": "brunn-workspace-import-manifest@v1"
            },
            "client": {
                "kind": "task",
                "schema": "task.v1",
                "task": {
                    "id": task_id,
                    "title": "Downgrade Charlemagne",
                    "status": sourced(json!("done"), "owner"),
                    "project": sourced(json!("brunn"), "agent:codex"),
                    "soft_due": sourced(json!("2026-08-31"), "agent:codex"),
                    "cost_of_delay": sourced(json!({
                        "amount_cents": 700,
                        "per": "week",
                        "since": "2026-08-01"
                    }), "agent:codex"),
                    "required_contexts": sourced(json!(["home", "online"]), "owner"),
                    "done_at": "2026-08-27T08:00:00Z",
                    "provenance": {
                        "captured_by": "agent:codex",
                        "created_at": "2026-08-27T07:00:00Z"
                    }
                }
            }
        });
        let mut tx = pool
            .begin()
            .await
            .expect("begin metadata-only task mutation");
        let second = upsert_markdown_in_tx(
            &mut tx,
            user_id,
            None,
            prepared_task_fixture(task_id, content, done_task, Some(1)),
        )
        .await
        .expect("unchanged Markdown with changed task state versions");
        tx.commit()
            .await
            .expect("commit metadata-only task mutation");
        assert_eq!(second.version, 2);
        assert!(!second.no_op);
        let projected: (i64, String, Option<DateTime<Utc>>) = sqlx::query_as(
            "SELECT entry_version,status,done_at FROM brunn.task_index WHERE user_id=$1 AND task_id=$2",
        )
        .bind(user_id)
        .bind(task_id)
        .fetch_one(&pool)
        .await
        .expect("read updated task projection");
        assert_eq!(projected.0, 2);
        assert_eq!(projected.1, "done");
        assert_eq!(
            projected.2.unwrap().to_rfc3339(),
            "2026-08-27T08:00:00+00:00"
        );
        let project_activity_after_task = sqlx::query_scalar::<_, DateTime<Utc>>(
            "SELECT last_activity_at FROM brunn.task_projects WHERE user_id=$1 AND slug='brunn'",
        )
        .bind(user_id)
        .fetch_one(&pool)
        .await
        .expect("task mutation advances project activity");
        let task_projection_updated_at = sqlx::query_scalar::<_, DateTime<Utc>>(
            "SELECT updated_at FROM brunn.task_index WHERE user_id=$1 AND task_id=$2",
        )
        .bind(user_id)
        .bind(task_id)
        .fetch_one(&pool)
        .await
        .expect("read task projection activity time");
        assert_eq!(project_activity_after_task, task_projection_updated_at);

        let task_entry_id = first.entry_id;
        let history = sqlx::query_scalar::<_, String>(
            "SELECT content FROM brunn.entry_versions WHERE user_id=$1 AND entry_id=$2 ORDER BY version",
        )
        .bind(user_id)
        .bind(task_entry_id)
        .fetch_all(&pool)
        .await
        .expect("read exact task entry history");
        assert_eq!(history, [content.to_owned(), content.to_owned()]);
        let task_path = format!(".brunn/tasks/{task_id}.md");
        let changes = sqlx::query_as::<_, (String, i64, String)>(
            "SELECT path,entry_version,operation FROM brunn.workspace_changes WHERE user_id=$1 AND entry_id=$2 ORDER BY generation",
        )
        .bind(user_id)
        .bind(task_entry_id)
        .fetch_all(&pool)
        .await
        .expect("read task workspace changes");
        assert_eq!(
            changes,
            [
                (task_path.clone(), 1, "create".to_owned()),
                (task_path, 2, "update".to_owned()),
            ]
        );
        let chunk_count = sqlx::query_scalar::<_, i64>(
            "SELECT count(*) FROM brunn.search_chunks WHERE user_id=$1 AND entry_id=$2",
        )
        .bind(user_id)
        .bind(task_entry_id)
        .fetch_one(&pool)
        .await
        .expect("count forbidden task search chunks");
        assert_eq!(chunk_count, 0, "task entry created search chunks");
        let job_count = sqlx::query_scalar::<_, i64>(
            "SELECT count(*) FROM brunn.jobs WHERE user_id=$1 AND payload->>'entry_id'=$2",
        )
        .bind(user_id)
        .bind(task_entry_id.to_string())
        .fetch_one(&pool)
        .await
        .expect("count forbidden task jobs");
        assert_eq!(job_count, 0, "task entry created jobs");

        let explicit_request = CheckpointRequest {
            session_id: format!("session:task-explicit-project:{user_id}"),
            parent_checkpoint_id: None,
            state: json!({
                "objective": "Resume Brunn",
                "project": "brunn",
                "current_state": ["Task storage is durable"]
            }),
            source_refs: vec![],
            idempotency_key: Some(format!("task-explicit-project:{user_id}")),
        };
        let (explicit_key, explicit_hash) = validate_checkpoint_request(&explicit_request).unwrap();
        let (explicit_ref, explicit_path, explicit_prepared) =
            prepared_checkpoint_fixture(&explicit_request, &explicit_key, &explicit_hash, 2);
        checkpoint_attempt_with_auth(
            &pool,
            user_id,
            &explicit_key,
            &explicit_hash,
            &explicit_ref,
            &explicit_path,
            2,
            explicit_prepared,
            Some(&checkpoint_auth),
        )
        .await
        .expect("write project-explicit checkpoint");
        let explicit_id = checkpoint_entry_id_for_new_write(&explicit_request);
        let explicit_link: (String, String) = sqlx::query_as(
            "SELECT project_slug,attribution FROM brunn.task_checkpoint_links WHERE user_id=$1 AND checkpoint_entry_id=$2",
        )
        .bind(user_id)
        .bind(explicit_id)
        .fetch_one(&pool)
        .await
        .expect("read explicit checkpoint project link");
        assert_eq!(explicit_link, ("brunn".to_owned(), "explicit".to_owned()));
        let (project_activity_after_checkpoint, checkpoint_created_at) =
            sqlx::query_as::<_, (DateTime<Utc>, DateTime<Utc>)>(
                r#"
                SELECT project.last_activity_at,version.created_at
                FROM brunn.task_projects AS project
                JOIN brunn.entries AS entry
                  ON entry.user_id=project.user_id AND entry.id=$2
                JOIN brunn.entry_versions AS version
                  ON version.user_id=entry.user_id
                 AND version.entry_id=entry.id
                 AND version.version=entry.current_version
                WHERE project.user_id=$1 AND project.slug='brunn'
                "#,
            )
            .bind(user_id)
            .bind(explicit_id)
            .fetch_one(&pool)
            .await
            .expect("checkpoint link advances project activity");
        assert_eq!(project_activity_after_checkpoint, checkpoint_created_at);
        assert!(project_activity_after_checkpoint >= project_activity_after_task);
        let checkpoint_payload: (String, Value) = sqlx::query_as(
            r#"
            SELECT version.content,version.metadata
            FROM brunn.entry_versions AS version
            WHERE version.user_id=$1 AND version.entry_id=$2 AND version.version=1
            "#,
        )
        .bind(user_id)
        .bind(explicit_id)
        .fetch_one(&pool)
        .await
        .expect("read durable checkpoint state");
        assert!(checkpoint_payload.0.contains("project: brunn"));
        assert_eq!(checkpoint_payload.1["project"], "brunn");
        assert_eq!(
            checkpoint_payload.1["checkpoint_state"]["current_state"][0],
            "Task storage is durable"
        );

        let fallback_request = CheckpointRequest {
            session_id: format!("session:task-path-project:{user_id}"),
            parent_checkpoint_id: None,
            state: json!({"objective": "Resume by source path"}),
            source_refs: vec!["sources/Projects/Brunn/Agent notes.md".to_owned()],
            idempotency_key: Some(format!("task-path-project:{user_id}")),
        };
        let (fallback_key, fallback_hash) = validate_checkpoint_request(&fallback_request).unwrap();
        let (fallback_ref, fallback_path, fallback_prepared) =
            prepared_checkpoint_fixture(&fallback_request, &fallback_key, &fallback_hash, 3);
        checkpoint_attempt_with_auth(
            &pool,
            user_id,
            &fallback_key,
            &fallback_hash,
            &fallback_ref,
            &fallback_path,
            3,
            fallback_prepared,
            Some(&checkpoint_auth),
        )
        .await
        .expect("write path-fallback checkpoint");
        let fallback_id = checkpoint_entry_id_for_new_write(&fallback_request);
        let fallback_link: (String, String, Option<String>) = sqlx::query_as(
            "SELECT project_slug,attribution,matched_path FROM brunn.task_checkpoint_links WHERE user_id=$1 AND checkpoint_entry_id=$2",
        )
        .bind(user_id)
        .bind(fallback_id)
        .fetch_one(&pool)
        .await
        .expect("read fallback checkpoint project link");
        assert_eq!(fallback_link.0, "brunn");
        assert_eq!(fallback_link.1, "path_fallback");
        assert_eq!(fallback_link.2.as_deref(), Some("sources/Projects/Brunn/"));

        sqlx::query(
            r#"
            INSERT INTO brunn.task_surface_defaults (user_id,surface,contexts)
            VALUES ($1,'test',ARRAY['home','online'])
            ON CONFLICT (user_id,surface) DO UPDATE SET contexts=EXCLUDED.contexts
            "#,
        )
        .bind(user_id)
        .execute(&pool)
        .await
        .expect("seed mergeable surface contexts");
        let mut context_tx = pool.begin().await.expect("begin context registry flow");
        let blocked = crate::task_service::create_context_in_tx(
            &mut context_tx,
            user_id,
            credential_id,
            "phne",
            None,
            None,
            "agent:codex",
            false,
        )
        .await
        .expect_err("near-match context creation must require explicit confirmation");
        assert!(matches!(
            blocked,
            ApiError::Public {
                status: StatusCode::CONFLICT,
                code: "context_confirmation_required",
                ..
            }
        ));
        let created = crate::task_service::create_context_in_tx(
            &mut context_tx,
            user_id,
            credential_id,
            "Workshop",
            None,
            Some("At the workshop"),
            "agent:codex",
            false,
        )
        .await
        .expect("distinct dynamic context is created");
        assert_eq!(created, "workshop");
        let rewritten = crate::task_service::merge_contexts_in_tx(
            &mut context_tx,
            user_id,
            credential_id,
            "home",
            "online",
            "agent:codex",
            "2026-08-27T09:00:00Z".parse().unwrap(),
        )
        .await
        .expect("explicit context merge rewrites canonical task state");
        assert_eq!(rewritten, 1);
        context_tx
            .commit()
            .await
            .expect("commit context registry flow");

        let merged_contexts = sqlx::query_scalar::<_, Vec<String>>(
            "SELECT required_contexts FROM brunn.task_index WHERE user_id=$1 AND task_id=$2",
        )
        .bind(user_id)
        .bind(task_id)
        .fetch_one(&pool)
        .await
        .expect("read context-merged projection");
        assert_eq!(merged_contexts, ["online"]);
        let canonical_contexts = sqlx::query_scalar::<_, Value>(
            r#"
            SELECT version.metadata #> '{client,task,required_contexts,value}'
            FROM brunn.entries AS entry
            JOIN brunn.entry_versions AS version
              ON version.user_id=entry.user_id
             AND version.entry_id=entry.id
             AND version.version=entry.current_version
            WHERE entry.user_id=$1 AND entry.id=$2
            "#,
        )
        .bind(user_id)
        .bind(task_entry_id)
        .fetch_one(&pool)
        .await
        .expect("read canonical merged context cell");
        assert_eq!(canonical_contexts, json!(["online"]));
        let canonical_context_source = sqlx::query_scalar::<_, String>(
            r#"
            SELECT version.metadata #>> '{client,task,required_contexts,source}'
            FROM brunn.entries AS entry
            JOIN brunn.entry_versions AS version
              ON version.user_id=entry.user_id
             AND version.entry_id=entry.id
             AND version.version=entry.current_version
            WHERE entry.user_id=$1 AND entry.id=$2
            "#,
        )
        .bind(user_id)
        .bind(task_entry_id)
        .fetch_one(&pool)
        .await
        .expect("read canonical merged context source");
        assert_eq!(canonical_context_source, "owner");
        let alias_target = sqlx::query_scalar::<_, String>(
            "SELECT context_slug FROM brunn.task_context_aliases WHERE user_id=$1 AND alias='home'",
        )
        .bind(user_id)
        .fetch_one(&pool)
        .await
        .expect("read durable context merge alias");
        assert_eq!(alias_target, "online");
        let merged_defaults = sqlx::query_scalar::<_, Vec<String>>(
            "SELECT contexts FROM brunn.task_surface_defaults WHERE user_id=$1 AND surface='test'",
        )
        .bind(user_id)
        .fetch_one(&pool)
        .await
        .expect("read merged surface contexts");
        assert_eq!(merged_defaults, ["online"]);
        assert_eq!(
            sqlx::query_scalar::<_, i64>(
                "SELECT count(*) FROM brunn.task_corrections WHERE user_id=$1 AND task_id=$2 AND field_name='required_contexts'",
            )
            .bind(user_id)
            .bind(task_id)
            .fetch_one(&pool)
            .await
            .unwrap(),
            1
        );
        assert_eq!(
            sqlx::query_as::<_, (Option<String>, String)>(
                "SELECT previous_source,corrected_source FROM brunn.task_corrections WHERE user_id=$1 AND task_id=$2 AND field_name='required_contexts'",
            )
            .bind(user_id)
            .bind(task_id)
            .fetch_one(&pool)
            .await
            .unwrap(),
            (Some("owner".to_owned()), "owner".to_owned())
        );
        assert_eq!(
            sqlx::query_scalar::<_, i64>(
                "SELECT count(*) FROM brunn.task_audit_events WHERE user_id=$1 AND action IN ('context.create','context.merge')",
            )
            .bind(user_id)
            .fetch_one(&pool)
            .await
            .unwrap(),
            2
        );
    }

    #[tokio::test]
    async fn force_new_version_bumps_identical_content_with_new_metadata() {
        let Some(database_url) = std::env::var("BRUNN_TEST_DATABASE_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("BRUNN_TEST_DATABASE_URL is unset; skipping force-new-version test");
            return;
        };
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(2)
            .connect(&database_url)
            .await
            .expect("connect to disposable Postgres");
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .expect("apply Brunn migrations");
        let user_id = Uuid::now_v7();
        sqlx::query("INSERT INTO brunn.users (id,external_ref,display_name) VALUES ($1,$2,$3)")
            .bind(user_id)
            .bind(format!("force-version-test:{user_id}"))
            .bind("Force version test")
            .execute(&pool)
            .await
            .expect("insert test user");

        let original_metadata =
            json!({"kind": "briefing_edition", "briefing": {"date": "2026-08-01"}});
        let mut tx = pool.begin().await.expect("begin first write");
        let first = upsert_markdown_in_tx(
            &mut tx,
            user_id,
            None,
            prepared_markdown_fixture(original_metadata, false),
        )
        .await
        .expect("first write");
        tx.commit().await.expect("commit first write");
        assert!(!first.no_op);
        assert_eq!(first.version, 1);

        let revised_metadata = json!({
            "kind": "briefing_edition",
            "briefing": {"date": "2026-08-01", "summary_md": ["changed"]}
        });
        let mut tx = pool.begin().await.expect("begin unforced replay");
        let replay = upsert_markdown_in_tx(
            &mut tx,
            user_id,
            None,
            prepared_markdown_fixture(revised_metadata.clone(), false),
        )
        .await
        .expect("unforced replay");
        tx.commit().await.expect("commit unforced replay");
        assert!(
            replay.no_op,
            "identical content without the flag stays a NoOp"
        );
        assert_eq!(replay.version, 1);

        let mut tx = pool.begin().await.expect("begin forced write");
        let forced = upsert_markdown_in_tx(
            &mut tx,
            user_id,
            None,
            prepared_markdown_fixture(revised_metadata.clone(), true),
        )
        .await
        .expect("forced write");
        tx.commit().await.expect("commit forced write");
        assert!(!forced.no_op);
        assert_eq!(forced.version, 2, "the forced write is a real new version");
        assert!(
            forced.generation.is_some(),
            "the forced write records a workspace_changes row",
        );

        let stored: (i64, Value) = sqlx::query_as(
            r#"
            SELECT entry.current_version,version.metadata
            FROM brunn.entries AS entry
            JOIN brunn.entry_versions AS version
              ON version.user_id=entry.user_id
             AND version.entry_id=entry.id
             AND version.version=entry.current_version
            WHERE entry.user_id=$1
            "#,
        )
        .bind(user_id)
        .fetch_one(&pool)
        .await
        .expect("stored row loads");
        assert_eq!(stored.0, 2);
        assert_eq!(stored.1, revised_metadata, "the new metadata is stored");
    }
}
