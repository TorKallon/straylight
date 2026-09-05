use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    time::Duration,
};

use axum::{
    Extension, Json, Router,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode, header::AUTHORIZATION},
    routing::{get, patch, post, put},
};
use chrono::{DateTime, SubsecRound, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sqlx::{Postgres, Row, Transaction, postgres::PgRow};
use uuid::Uuid;

use crate::{
    auth::AuthContext,
    db::AppState,
    error::{ApiError, ApiResult},
    messaging_protocol::{
        self, CanonicalMessage, ConversationHeader, ConversationKind, ConversationParticipant,
        ConversationStatus, MessageKind, MessageRef, SendMessageInput,
    },
    models::{Capability, CredentialId, ResponseStatus, UserId},
    notification_service::{self, NotificationTarget, PublishAccess, PublishRequest},
    simple_core::WorkspaceEnvelope,
    task_guard,
};

const MAX_SYNC_MESSAGES: i64 = 200;
const MAX_WAIT_SECONDS: u64 = 25;
const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(500);
const PRESENCE_LEASE_SECONDS: i64 = 60;
const SENDER_RATE_LIMIT: i64 = 60;
const CONVERSATION_RATE_LIMIT: i64 = 200;
const AGENT_STREAK_LIMIT: i32 = 20;
const MAX_CONTINUATION_HOPS: usize = 64;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/workspace/messaging/sync", get(sync))
        .route(
            "/workspace/messaging/conversations",
            post(create_conversation),
        )
        .route(
            "/workspace/messaging/conversations/{conversation_id}/messages",
            post(send_message),
        )
        .route(
            "/workspace/messaging/conversations/{conversation_id}/read",
            post(mark_read),
        )
        .route(
            "/workspace/messaging/conversations/{conversation_id}/resume",
            post(resume_conversation),
        )
        .route(
            "/workspace/messaging/conversations/{conversation_id}/close",
            post(close_conversation),
        )
        .route(
            "/workspace/messaging/agents",
            get(list_agents).post(create_agent),
        )
        .route(
            "/workspace/messaging/agents/{agent_id}",
            patch(update_agent),
        )
        .route(
            "/workspace/messaging/agents/{agent_id}/credential",
            put(bind_agent_credential),
        )
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyncQuery {
    #[serde(default)]
    pub cursor: i64,
    #[serde(default)]
    pub wait: u64,
    pub conversation_id: Option<Uuid>,
    pub after_seq: Option<i64>,
    pub limit: Option<i64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateConversationRequest {
    pub participants: Vec<String>,
    pub subject: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadRequest {
    pub last_read_seq: i64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmptyRequest {}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateAgentRequest {
    pub agent_id: String,
    pub display_name: String,
    pub principal_kind: String,
    #[serde(default = "default_delivery_mode")]
    pub delivery_mode: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateAgentRequest {
    pub display_name: Option<String>,
    pub delivery_mode: Option<String>,
    pub archived: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BindCredentialRequest {
    pub credential_id: Option<Uuid>,
}

fn default_delivery_mode() -> String {
    "pull".to_owned()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParticipantView {
    pub agent_id: String,
    pub role: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConversationView {
    pub conversation_id: Uuid,
    pub conversation_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    pub status: String,
    pub participants: Vec<ParticipantView>,
    pub last_seq: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_message_at: Option<DateTime<Utc>>,
    pub last_read_seq: i64,
    pub unread_count: i64,
    pub needs_human: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continues_from: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continuation_id: Option<Uuid>,
    pub latest_sync_cursor: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MessageView {
    pub conversation_id: Uuid,
    pub seq: i64,
    pub message_id: Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_key: Option<String>,
    #[serde(skip)]
    request_hash: Option<String>,
    pub kind: String,
    pub body_md: String,
    pub refs: Vec<MessageRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_reply_to_conversation_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_reply_to: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    pub expects_reply: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_by: Option<DateTime<Utc>>,
    pub sync_cursor: i64,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AgentView {
    pub agent_id: String,
    pub display_name: String,
    pub principal_kind: String,
    pub delivery_mode: String,
    pub online: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_seen_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lease_expires_at: Option<DateTime<Utc>>,
    pub archived: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credential_names: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SyncResponse {
    pub status: String,
    pub cursor: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume_cursor: Option<i64>,
    pub has_more: bool,
    pub messages: Vec<MessageView>,
    pub conversations: Vec<ConversationView>,
    pub presence: Vec<AgentView>,
    pub unread: BTreeMap<Uuid, i64>,
    pub as_of: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CreateConversationResponse {
    pub conversation_id: Uuid,
    pub conversation: ConversationView,
    pub duplicate: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct SendMessageResponse {
    pub conversation_id: Uuid,
    pub seq: i64,
    pub message: MessageView,
    pub duplicate: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continuation_id: Option<Uuid>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReadResponse {
    pub conversation_id: Uuid,
    pub last_read_seq: i64,
    pub cursor: i64,
    pub duplicate: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConversationMutationResponse {
    pub conversation_id: Uuid,
    pub status: String,
    pub cursor: i64,
    pub duplicate: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct AgentListResponse {
    pub agents: Vec<AgentView>,
    pub as_of: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AgentMutationResponse {
    pub agent: AgentView,
}

#[derive(Clone, Debug, Serialize)]
pub struct CredentialBindingResponse {
    pub agent_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credential_id: Option<Uuid>,
    pub bound: bool,
}

#[derive(Clone, Debug)]
struct Principal {
    agent_id: String,
    principal_kind: String,
}

#[derive(Clone, Debug)]
struct ConversationRow {
    conversation_id: Uuid,
    entry_id: Uuid,
    conversation_kind: String,
    direct_key: Option<String>,
    subject: Option<String>,
    status: String,
    created_by_agent_id: String,
    last_seq: i64,
    agent_streak: i32,
    needs_human: bool,
    continues_from: Option<Uuid>,
    latest_sync_cursor: i64,
    closed_at: Option<DateTime<Utc>>,
    created_at: DateTime<Utc>,
}

impl ConversationRow {
    fn latest_sync_cursor(&self) -> i64 {
        self.latest_sync_cursor
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RolloverPlan {
    user_seq: i64,
    pause_system_in_current: bool,
    rollover: bool,
    pause_system_in_continuation: bool,
}

fn rollover_plan(last_seq: i64, pauses: bool) -> ApiResult<RolloverPlan> {
    let user_seq = last_seq + 1;
    if !(1..=messaging_protocol::MAX_MESSAGES_PER_CONVERSATION).contains(&user_seq) {
        return Err(ApiError::conflict(
            "conversation_full",
            "the conversation has reached its message limit",
            json!({}),
        ));
    }
    let pause_system_in_current =
        pauses && user_seq < messaging_protocol::MAX_MESSAGES_PER_CONVERSATION;
    let last_written = user_seq + i64::from(pause_system_in_current);
    Ok(RolloverPlan {
        user_seq,
        pause_system_in_current,
        rollover: last_written == messaging_protocol::MAX_MESSAGES_PER_CONVERSATION,
        pause_system_in_continuation: pauses && !pause_system_in_current,
    })
}

async fn sync(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Query(query): Query<SyncQuery>,
) -> ApiResult<Json<WorkspaceEnvelope<SyncResponse>>> {
    auth.require(Capability::MessageRead)?;
    let limit = validate_sync_query(&query)?;
    let started = tokio::time::Instant::now();
    let deadline = started + Duration::from_secs(query.wait);
    let mut renew_presence = query.wait > 0;
    let allow_web_owner_fallback = is_web_session_request(&headers);
    if query.wait > 0 {
        metrics::counter!("messaging.wait", "result" => "started").increment(1);
    }

    loop {
        let page = sync_once(
            &state,
            &auth,
            &query,
            limit,
            renew_presence,
            allow_web_owner_fallback,
        )
        .await?;
        renew_presence = false;
        if page.activity || query.wait == 0 {
            let payload_bytes = serde_json::to_vec(&page.response)
                .map(|payload| payload.len())
                .unwrap_or(0);
            metrics::histogram!(
                "messaging.sync.payload_bytes",
                "wait" => if query.wait > 0 { "long_poll" } else { "immediate" }
            )
            .record(payload_bytes as f64);
            metrics::histogram!(
                "messaging.sync.duration_ms",
                "wait" => if query.wait > 0 { "long_poll" } else { "immediate" },
                "result" => "complete"
            )
            .record(started.elapsed().as_secs_f64() * 1_000.0);
            return Ok(complete_envelope(page.response));
        }
        let now = tokio::time::Instant::now();
        if now >= deadline {
            let mut response = page.response;
            response.status = "timeout".to_owned();
            response.resume_cursor = Some(if query.conversation_id.is_some() {
                query.after_seq.unwrap_or(0)
            } else {
                response.cursor
            });
            let payload_bytes = serde_json::to_vec(&response)
                .map(|payload| payload.len())
                .unwrap_or(0);
            metrics::histogram!(
                "messaging.sync.payload_bytes",
                "wait" => "long_poll"
            )
            .record(payload_bytes as f64);
            metrics::counter!("messaging.wait", "result" => "timeout").increment(1);
            metrics::histogram!(
                "messaging.sync.duration_ms",
                "wait" => "long_poll",
                "result" => "timeout"
            )
            .record(started.elapsed().as_secs_f64() * 1_000.0);
            return Ok(complete_envelope(response));
        }
        tokio::time::sleep(WAIT_POLL_INTERVAL.min(deadline - now)).await;
    }
}

async fn create_conversation(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(request): Json<CreateConversationRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<CreateConversationResponse>>> {
    auth.require(Capability::MessageWrite)?;
    let subject =
        messaging_protocol::validate_subject(request.subject.as_deref()).map_err(protocol_error)?;
    if request.participants.is_empty() || request.participants.len() > 31 {
        return Err(ApiError::invalid(
            "participants must contain between 1 and 31 agent ids",
        ));
    }
    for agent_id in &request.participants {
        messaging_protocol::validate_agent_id(agent_id).map_err(protocol_error)?;
    }

    let as_of = Utc::now();
    let mut tx = state.begin_write(&auth).await?;
    let caller = resolve_principal_in_tx(&mut tx, &auth, is_web_session_request(&headers)).await?;
    let (conversation_id, duplicate) = create_conversation_in_tx(
        &mut tx,
        &auth,
        &caller,
        request.participants,
        subject,
        as_of,
    )
    .await?;
    let conversation = load_one_conversation_view_in_tx(
        &mut tx,
        auth.user_id.0,
        &caller.agent_id,
        conversation_id,
    )
    .await?;
    tx.commit().await?;
    if !duplicate {
        state.workspace_features.invalidate(auth.user_id.0).await;
    }
    let mut envelope = WorkspaceEnvelope::complete(CreateConversationResponse {
        conversation_id,
        conversation,
        duplicate,
    });
    envelope.status = if duplicate {
        ResponseStatus::NoOp
    } else {
        ResponseStatus::Committed
    };
    Ok(Json(envelope))
}

async fn send_message(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Path(conversation_id): Path<Uuid>,
    Json(mut input): Json<SendMessageInput>,
) -> ApiResult<Json<WorkspaceEnvelope<SendMessageResponse>>> {
    auth.require(Capability::MessageWrite)?;
    // Hash the same deadline precision that PostgreSQL persists.
    input.reply_by = input.reply_by.map(|deadline| deadline.trunc_subsecs(6));

    let started = std::time::Instant::now();
    let mut tx = state.begin_write(&auth).await?;
    let sender = resolve_principal_in_tx(&mut tx, &auth, is_web_session_request(&headers)).await?;
    acquire_sender_lock(&mut tx, auth.user_id.0, &sender.agent_id).await?;

    if let Some(replay) =
        load_replay_in_tx(&mut tx, auth.user_id.0, &sender.agent_id, &input.client_key).await?
    {
        let same_target = continuation_chain_contains_in_tx(
            &mut tx,
            auth.user_id.0,
            conversation_id,
            replay.conversation_id,
        )
        .await?;
        let expected_reply_conversation_id = input.in_reply_to.map(|_| conversation_id);
        let reply_target_exists = match input.in_reply_to {
            Some(seq) => {
                sqlx::query_scalar::<_, bool>(
                    r#"
                    SELECT EXISTS(
                      SELECT 1 FROM brunn.messaging_message_index
                      WHERE user_id=$1 AND conversation_id=$2 AND seq=$3
                    )
                    "#,
                )
                .bind(auth.user_id.0)
                .bind(conversation_id)
                .bind(seq)
                .fetch_one(&mut *tx)
                .await?
            }
            None => true,
        };
        let expected_hash = messaging_protocol::request_hash_with_reply_target(
            replay.conversation_id,
            expected_reply_conversation_id,
            &input,
        );
        if !same_target
            || !reply_target_exists
            || replay.in_reply_to_conversation_id != expected_reply_conversation_id
            || replay.request_hash.as_deref() != Some(expected_hash.as_str())
        {
            return Err(ApiError::conflict(
                "idempotency_conflict",
                "the client key was already used with a different messaging request",
                json!({}),
            ));
        }
        tx.commit().await?;
        metrics::counter!(
            "messaging.send",
            "result" => "duplicate",
            "principal_kind" => sender.principal_kind.clone()
        )
        .increment(1);
        metrics::histogram!(
            "messaging.send.duration_ms",
            "result" => "duplicate",
            "principal_kind" => sender.principal_kind.clone()
        )
        .record(started.elapsed().as_secs_f64() * 1_000.0);
        let response = SendMessageResponse {
            conversation_id: replay.conversation_id,
            seq: replay.seq,
            message: replay,
            duplicate: true,
            continuation_id: None,
        };
        let mut envelope = WorkspaceEnvelope::complete(response);
        envelope.status = ResponseStatus::NoOp;
        return Ok(Json(envelope));
    }

    let (target_id, mut conversation) =
        load_writable_conversation_for_update(&mut tx, auth.user_id.0, conversation_id).await?;
    // Timestamp the serialized write after following any rollover.
    let as_of = Utc::now();
    messaging_protocol::validate_send_input(&input, as_of).map_err(protocol_error)?;
    require_conversation_sender(&mut tx, auth.user_id.0, target_id, &sender, &conversation).await?;
    let sender_is_owner = sender.principal_kind == "owner";
    if sender_is_owner {
        let promoted = promote_owner_participant_in_tx(
            &mut tx,
            auth.user_id.0,
            target_id,
            &sender.agent_id,
            as_of,
        )
        .await?;
        if promoted {
            conversation.conversation_kind = "group".to_owned();
            conversation.direct_key = None;
        }
    }
    if conversation.status == "paused_for_human" && !sender_is_owner {
        return Err(ApiError::conflict(
            "conversation_paused",
            "an owner response is required before agents can continue",
            json!({"needs_human": true}),
        ));
    }
    check_send_rates_in_tx(&mut tx, auth.user_id.0, &sender.agent_id, target_id, as_of).await?;
    let in_reply_to_conversation_id = validate_reply_target_in_tx(
        &mut tx,
        auth.user_id.0,
        conversation_id,
        target_id,
        input.in_reply_to,
    )
    .await?;

    let next_streak = if sender_is_owner {
        0
    } else {
        (conversation.agent_streak + 1).min(AGENT_STREAK_LIMIT)
    };
    let pauses = !sender_is_owner && next_streak == AGENT_STREAK_LIMIT;
    let owner_is_participant =
        owner_is_participant_in_tx(&mut tx, auth.user_id.0, target_id).await?;
    let needs_human = if sender_is_owner {
        false
    } else {
        conversation.needs_human
            || pauses
            || (input.kind == MessageKind::Question && owner_is_participant)
    };
    let plan = rollover_plan(conversation.last_seq, pauses)?;
    let request_hash = messaging_protocol::request_hash_with_reply_target(
        target_id,
        in_reply_to_conversation_id,
        &input,
    );
    let message_cursor = allocate_cursor_in_tx(&mut tx, auth.user_id.0).await?;
    let message_id = Uuid::now_v7();
    insert_client_message_in_tx(
        &mut tx,
        auth.user_id.0,
        target_id,
        plan.user_seq,
        message_id,
        &sender.agent_id,
        &request_hash,
        in_reply_to_conversation_id,
        &input,
        message_cursor,
        as_of,
    )
    .await?;
    if owner_is_participant {
        publish_conversation_notification_in_tx(
            &mut tx,
            &state,
            &auth,
            target_id,
            plan.user_seq,
            "message",
            None,
            as_of,
        )
        .await?;
    }

    let mut last_seq = plan.user_seq;
    let mut latest_cursor = message_cursor;
    if plan.pause_system_in_current {
        let pause_seq = plan.user_seq + 1;
        let pause_cursor = allocate_cursor_in_tx(&mut tx, auth.user_id.0).await?;
        insert_system_message_in_tx(
            &mut tx,
            auth.user_id.0,
            target_id,
            pause_seq,
            &format!("budget:{target_id}:{pause_seq}"),
            "Agent exchange paused after 20 consecutive messages. An owner response is required.",
            pause_cursor,
            as_of,
        )
        .await?;
        last_seq = pause_seq;
        latest_cursor = pause_cursor;
    }

    let mut continuation_id = None;
    if plan.rollover {
        close_for_rollover_in_tx(
            &mut tx,
            auth.user_id.0,
            target_id,
            last_seq,
            latest_cursor,
            next_streak,
            as_of,
        )
        .await?;
        write_canonical_conversation_in_tx(&mut tx, &auth, target_id).await?;
        let continuation = create_continuation_in_tx(
            &mut tx,
            &state,
            &auth,
            &conversation,
            &sender.agent_id,
            next_streak,
            pauses,
            needs_human,
            plan.pause_system_in_continuation,
            true,
            as_of,
        )
        .await?;
        continuation_id = Some(continuation);
    } else {
        let status = if pauses { "paused_for_human" } else { "open" };
        sqlx::query(
            r#"
            UPDATE brunn.messaging_conversations
            SET last_seq=$3,last_message_at=$4,agent_streak=$5,
                needs_human=$6,status=$7,latest_sync_cursor=$8,
                closed_at=NULL,updated_at=$4
            WHERE user_id=$1 AND conversation_id=$2
            "#,
        )
        .bind(auth.user_id.0)
        .bind(target_id)
        .bind(last_seq)
        .bind(as_of)
        .bind(next_streak)
        .bind(needs_human)
        .bind(status)
        .bind(latest_cursor)
        .execute(&mut *tx)
        .await?;
        write_canonical_conversation_in_tx(&mut tx, &auth, target_id).await?;
    }

    if pauses {
        let (notification_conversation, notification_seq) = if plan.pause_system_in_continuation {
            (
                continuation_id.ok_or_else(|| {
                    ApiError::Internal("pause rollover did not create a continuation".to_owned())
                })?,
                2,
            )
        } else {
            (target_id, last_seq)
        };
        publish_conversation_notification_in_tx(
            &mut tx,
            &state,
            &auth,
            notification_conversation,
            notification_seq,
            "needs-human",
            None,
            as_of,
        )
        .await?;
        metrics::counter!("messaging.guard", "kind" => "agent_streak").increment(1);
    }

    let message =
        load_message_by_seq_in_tx(&mut tx, auth.user_id.0, target_id, plan.user_seq).await?;
    tx.commit().await?;
    state.workspace_features.invalidate(auth.user_id.0).await;
    conversation.last_seq = last_seq;
    metrics::counter!(
        "messaging.send",
        "result" => "created",
        "principal_kind" => sender.principal_kind.clone()
    )
    .increment(1);
    metrics::histogram!(
        "messaging.send.duration_ms",
        "result" => "created",
        "principal_kind" => sender.principal_kind.clone()
    )
    .record(started.elapsed().as_secs_f64() * 1_000.0);
    let mut envelope = WorkspaceEnvelope::complete(SendMessageResponse {
        conversation_id: target_id,
        seq: plan.user_seq,
        message,
        duplicate: false,
        continuation_id,
    });
    envelope.status = ResponseStatus::Committed;
    Ok(Json(envelope))
}

async fn mark_read(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Path(conversation_id): Path<Uuid>,
    Json(request): Json<ReadRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<ReadResponse>>> {
    auth.require(Capability::MessageWrite)?;
    if request.last_read_seq < 0 {
        return Err(ApiError::invalid("last_read_seq must be nonnegative"));
    }
    let mut tx = state.begin_write(&auth).await?;
    let principal =
        resolve_principal_in_tx(&mut tx, &auth, is_web_session_request(&headers)).await?;
    let conversation =
        load_conversation_for_update(&mut tx, auth.user_id.0, conversation_id).await?;
    require_membership_in_tx(
        &mut tx,
        auth.user_id.0,
        conversation_id,
        &principal.agent_id,
    )
    .await?;
    if request.last_read_seq > conversation.last_seq {
        return Err(ApiError::invalid(
            "last_read_seq cannot exceed the conversation's last sequence",
        ));
    }
    let current = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT last_read_seq FROM brunn.messaging_participants
        WHERE user_id=$1 AND conversation_id=$2 AND agent_id=$3
        FOR UPDATE
        "#,
    )
    .bind(auth.user_id.0)
    .bind(conversation_id)
    .bind(&principal.agent_id)
    .fetch_one(&mut *tx)
    .await?;
    let duplicate = request.last_read_seq <= current;
    let (last_read_seq, cursor) = if duplicate {
        (current, conversation.latest_sync_cursor())
    } else {
        let cursor = allocate_cursor_in_tx(&mut tx, auth.user_id.0).await?;
        sqlx::query(
            r#"
            UPDATE brunn.messaging_participants
            SET last_read_seq=$4,updated_at=clock_timestamp()
            WHERE user_id=$1 AND conversation_id=$2 AND agent_id=$3
            "#,
        )
        .bind(auth.user_id.0)
        .bind(conversation_id)
        .bind(&principal.agent_id)
        .bind(request.last_read_seq)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            r#"
            UPDATE brunn.messaging_conversations
            SET latest_sync_cursor=$3,updated_at=clock_timestamp()
            WHERE user_id=$1 AND conversation_id=$2
            "#,
        )
        .bind(auth.user_id.0)
        .bind(conversation_id)
        .bind(cursor)
        .execute(&mut *tx)
        .await?;
        (request.last_read_seq, cursor)
    };
    tx.commit().await?;
    let mut envelope = WorkspaceEnvelope::complete(ReadResponse {
        conversation_id,
        last_read_seq,
        cursor,
        duplicate,
    });
    envelope.status = if duplicate {
        ResponseStatus::NoOp
    } else {
        ResponseStatus::Committed
    };
    Ok(Json(envelope))
}

async fn resume_conversation(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Path(conversation_id): Path<Uuid>,
    Json(_request): Json<EmptyRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<ConversationMutationResponse>>> {
    auth.require(Capability::MessageWrite)?;
    let mut tx = state.begin_write(&auth).await?;
    let principal =
        resolve_principal_in_tx(&mut tx, &auth, is_web_session_request(&headers)).await?;
    require_owner(&principal)?;
    let conversation =
        load_conversation_for_update(&mut tx, auth.user_id.0, conversation_id).await?;
    require_membership_in_tx(
        &mut tx,
        auth.user_id.0,
        conversation_id,
        &principal.agent_id,
    )
    .await?;
    if conversation.status == "closed" {
        return Err(ApiError::conflict(
            "conversation_closed",
            "a closed conversation cannot be resumed",
            json!({}),
        ));
    }
    let duplicate = conversation.status == "open"
        && !conversation.needs_human
        && conversation.agent_streak == 0;
    let cursor = if duplicate {
        conversation.latest_sync_cursor
    } else {
        let cursor = allocate_cursor_in_tx(&mut tx, auth.user_id.0).await?;
        sqlx::query(
            r#"
            UPDATE brunn.messaging_conversations
            SET status='open',needs_human=false,agent_streak=0,
                latest_sync_cursor=$3,updated_at=clock_timestamp()
            WHERE user_id=$1 AND conversation_id=$2
            "#,
        )
        .bind(auth.user_id.0)
        .bind(conversation_id)
        .bind(cursor)
        .execute(&mut *tx)
        .await?;
        write_canonical_conversation_in_tx(&mut tx, &auth, conversation_id).await?;
        cursor
    };
    tx.commit().await?;
    if !duplicate {
        state.workspace_features.invalidate(auth.user_id.0).await;
    }
    let mut envelope = WorkspaceEnvelope::complete(ConversationMutationResponse {
        conversation_id,
        status: "open".to_owned(),
        cursor,
        duplicate,
    });
    envelope.status = if duplicate {
        ResponseStatus::NoOp
    } else {
        ResponseStatus::Committed
    };
    Ok(Json(envelope))
}

async fn close_conversation(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Path(conversation_id): Path<Uuid>,
    Json(_request): Json<EmptyRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<ConversationMutationResponse>>> {
    auth.require(Capability::MessageWrite)?;
    let as_of = Utc::now();
    let mut tx = state.begin_write(&auth).await?;
    let principal =
        resolve_principal_in_tx(&mut tx, &auth, is_web_session_request(&headers)).await?;
    require_owner(&principal)?;
    let conversation =
        load_conversation_for_update(&mut tx, auth.user_id.0, conversation_id).await?;
    require_membership_in_tx(
        &mut tx,
        auth.user_id.0,
        conversation_id,
        &principal.agent_id,
    )
    .await?;
    let duplicate = conversation.status == "closed";
    let cursor = if duplicate {
        conversation.latest_sync_cursor
    } else {
        let canceled_deadline_conversations =
            cancel_reply_deadlines_for_chain_in_tx(&mut tx, auth.user_id.0, conversation_id, as_of)
                .await?;
        let cursor = allocate_cursor_in_tx(&mut tx, auth.user_id.0).await?;
        sqlx::query(
            r#"
            UPDATE brunn.messaging_conversations
            SET status='closed',closed_at=$3,needs_human=false,
                latest_sync_cursor=$4,updated_at=$3
            WHERE user_id=$1 AND conversation_id=$2
            "#,
        )
        .bind(auth.user_id.0)
        .bind(conversation_id)
        .bind(as_of)
        .bind(cursor)
        .execute(&mut *tx)
        .await?;
        write_canonical_conversation_in_tx(&mut tx, &auth, conversation_id).await?;
        for canceled_conversation_id in canceled_deadline_conversations {
            if canceled_conversation_id != conversation_id {
                write_canonical_conversation_in_tx(&mut tx, &auth, canceled_conversation_id)
                    .await?;
            }
        }
        cursor
    };
    tx.commit().await?;
    if !duplicate {
        state.workspace_features.invalidate(auth.user_id.0).await;
    }
    let mut envelope = WorkspaceEnvelope::complete(ConversationMutationResponse {
        conversation_id,
        status: "closed".to_owned(),
        cursor,
        duplicate,
    });
    envelope.status = if duplicate {
        ResponseStatus::NoOp
    } else {
        ResponseStatus::Committed
    };
    Ok(Json(envelope))
}

async fn list_agents(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
) -> ApiResult<Json<WorkspaceEnvelope<AgentListResponse>>> {
    auth.require(Capability::MessageRead)?;
    let as_of = Utc::now();
    let web_session = is_web_session_request(&headers);
    let mut tx = state.begin_write(&auth).await?;
    let principal = resolve_principal_in_tx(&mut tx, &auth, web_session).await?;
    let reveal_bindings =
        web_session && principal.principal_kind == "owner" && has_registry_capability(&auth);
    let agents = load_agent_views_in_tx(&mut tx, auth.user_id.0, as_of, reveal_bindings).await?;
    tx.commit().await?;
    Ok(complete_envelope(AgentListResponse { agents, as_of }))
}

async fn create_agent(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(request): Json<CreateAgentRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<AgentMutationResponse>>> {
    require_registry_web_request(&headers, &auth)?;
    validate_registry_agent(
        &request.agent_id,
        &request.display_name,
        &request.principal_kind,
        &request.delivery_mode,
    )?;
    if request.principal_kind == "owner" {
        return Err(ApiError::invalid(
            "the workspace owner principal is managed by authenticated owner identity",
        ));
    }
    let mut tx = state.begin_write(&auth).await?;
    let owner = resolve_principal_in_tx(&mut tx, &auth, true).await?;
    require_owner(&owner)?;
    sqlx::query(
        r#"
        INSERT INTO brunn.messaging_agents (
          user_id,agent_id,display_name,principal_kind,delivery_mode,
          created_by_credential_id
        ) VALUES ($1,$2,$3,$4,$5,$6)
        "#,
    )
    .bind(auth.user_id.0)
    .bind(&request.agent_id)
    .bind(request.display_name.trim())
    .bind(&request.principal_kind)
    .bind(&request.delivery_mode)
    .bind(auth.credential_id.0)
    .execute(&mut *tx)
    .await
    .map_err(map_agent_registry_database_error)?;
    let agent =
        load_one_agent_view_in_tx(&mut tx, auth.user_id.0, &request.agent_id, Utc::now(), true)
            .await?;
    tx.commit().await?;
    let mut envelope = WorkspaceEnvelope::complete(AgentMutationResponse { agent });
    envelope.status = ResponseStatus::Committed;
    Ok(Json(envelope))
}

async fn update_agent(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Path(agent_id): Path<String>,
    Json(request): Json<UpdateAgentRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<AgentMutationResponse>>> {
    require_registry_web_request(&headers, &auth)?;
    messaging_protocol::validate_agent_id(&agent_id).map_err(protocol_error)?;
    if request.display_name.is_none()
        && request.delivery_mode.is_none()
        && request.archived.is_none()
    {
        return Err(ApiError::invalid("at least one agent update is required"));
    }
    if let Some(display_name) = request.display_name.as_deref() {
        validate_display_name(display_name)?;
    }
    if let Some(delivery_mode) = request.delivery_mode.as_deref() {
        validate_delivery_mode(delivery_mode)?;
    }
    let mut tx = state.begin_write(&auth).await?;
    let owner = resolve_principal_in_tx(&mut tx, &auth, true).await?;
    require_owner(&owner)?;
    let principal_kind = sqlx::query_scalar::<_, String>(
        "SELECT principal_kind FROM brunn.messaging_agents WHERE user_id=$1 AND agent_id=$2 FOR UPDATE",
    )
    .bind(auth.user_id.0)
    .bind(&agent_id)
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(|| messaging_not_found(&agent_id))?;
    if principal_kind == "owner" && request.archived == Some(true) {
        return Err(ApiError::invalid("the owner principal cannot be archived"));
    }
    sqlx::query(
        r#"
        UPDATE brunn.messaging_agents
        SET display_name=coalesce($3,display_name),
            delivery_mode=coalesce($4,delivery_mode),
            archived_at=CASE
              WHEN $5::boolean IS NULL THEN archived_at
              WHEN $5 THEN coalesce(archived_at,clock_timestamp())
              ELSE NULL
            END,
            updated_at=clock_timestamp()
        WHERE user_id=$1 AND agent_id=$2
        "#,
    )
    .bind(auth.user_id.0)
    .bind(&agent_id)
    .bind(request.display_name.as_deref().map(str::trim))
    .bind(request.delivery_mode.as_deref())
    .bind(request.archived)
    .execute(&mut *tx)
    .await?;
    let agent =
        load_one_agent_view_in_tx(&mut tx, auth.user_id.0, &agent_id, Utc::now(), true).await?;
    tx.commit().await?;
    let mut envelope = WorkspaceEnvelope::complete(AgentMutationResponse { agent });
    envelope.status = ResponseStatus::Committed;
    Ok(Json(envelope))
}

async fn bind_agent_credential(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Path(agent_id): Path<String>,
    Json(request): Json<BindCredentialRequest>,
) -> ApiResult<Json<WorkspaceEnvelope<CredentialBindingResponse>>> {
    require_registry_web_request(&headers, &auth)?;
    messaging_protocol::validate_agent_id(&agent_id).map_err(protocol_error)?;
    let mut tx = state.begin_write(&auth).await?;
    let owner = resolve_principal_in_tx(&mut tx, &auth, true).await?;
    require_owner(&owner)?;
    let agent_exists = sqlx::query_scalar::<_, bool>(
        r#"
        SELECT EXISTS(
          SELECT 1 FROM brunn.messaging_agents
          WHERE user_id=$1 AND agent_id=$2 AND archived_at IS NULL
        )
        "#,
    )
    .bind(auth.user_id.0)
    .bind(&agent_id)
    .fetch_one(&mut *tx)
    .await?;
    if !agent_exists {
        return Err(messaging_not_found(&agent_id));
    }
    let bound = if let Some(credential_id) = request.credential_id {
        let credential_exists = sqlx::query_scalar::<_, bool>(
            r#"
            SELECT EXISTS(
              SELECT 1 FROM brunn_auth.list_credentials($1)
              WHERE id=$2 AND disabled_at IS NULL
            )
            "#,
        )
        .bind(auth.user_id.0)
        .bind(credential_id)
        .fetch_one(&mut *tx)
        .await?;
        if !credential_exists {
            return Err(messaging_not_found(&credential_id.to_string()));
        }
        sqlx::query(
            r#"
            INSERT INTO brunn.messaging_credential_bindings (
              user_id,credential_id,agent_id,bound_by_credential_id
            ) VALUES ($1,$2,$3,$4)
            ON CONFLICT (user_id,credential_id) DO UPDATE
            SET agent_id=excluded.agent_id,
                bound_by_credential_id=excluded.bound_by_credential_id,
                updated_at=clock_timestamp()
            "#,
        )
        .bind(auth.user_id.0)
        .bind(credential_id)
        .bind(&agent_id)
        .bind(auth.credential_id.0)
        .execute(&mut *tx)
        .await?;
        true
    } else {
        sqlx::query(
            "DELETE FROM brunn.messaging_credential_bindings WHERE user_id=$1 AND agent_id=$2",
        )
        .bind(auth.user_id.0)
        .bind(&agent_id)
        .execute(&mut *tx)
        .await?;
        false
    };
    tx.commit().await?;
    let mut envelope = WorkspaceEnvelope::complete(CredentialBindingResponse {
        agent_id,
        credential_id: request.credential_id,
        bound,
    });
    envelope.status = ResponseStatus::Committed;
    Ok(Json(envelope))
}

fn complete_envelope<T>(data: T) -> Json<WorkspaceEnvelope<T>> {
    Json(WorkspaceEnvelope::complete(data))
}

fn protocol_error(error: messaging_protocol::ProtocolError) -> ApiError {
    ApiError::invalid(error.to_string())
}

fn has_registry_capability(auth: &AuthContext) -> bool {
    auth.capabilities.contains("credential:manage") || auth.capabilities.contains("admin")
}

fn is_web_session_request(headers: &HeaderMap) -> bool {
    !headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("Bearer "))
}

fn require_registry_web_request(headers: &HeaderMap, auth: &AuthContext) -> ApiResult<()> {
    if !is_web_session_request(headers) {
        return Err(ApiError::public(
            StatusCode::FORBIDDEN,
            "web_session_required",
            "messaging registry changes require an authenticated owner Web session",
        ));
    }
    if !has_registry_capability(auth) {
        return Err(ApiError::capability("credential:manage"));
    }
    Ok(())
}

fn require_owner(principal: &Principal) -> ApiResult<()> {
    if principal.principal_kind == "owner" {
        Ok(())
    } else {
        Err(ApiError::public(
            StatusCode::FORBIDDEN,
            "owner_required",
            "this messaging operation requires the workspace owner",
        ))
    }
}

fn validate_sync_query(query: &SyncQuery) -> ApiResult<i64> {
    if query.cursor < 0 {
        return Err(ApiError::invalid("cursor must be nonnegative"));
    }
    if query.wait > MAX_WAIT_SECONDS {
        return Err(ApiError::invalid("wait must be between 0 and 25 seconds"));
    }
    if query.after_seq.is_some_and(|value| value < 0) {
        return Err(ApiError::invalid("after_seq must be nonnegative"));
    }
    match (query.conversation_id, query.after_seq) {
        (None, Some(_)) => {
            return Err(ApiError::invalid("after_seq requires conversation_id"));
        }
        (Some(_), _) if query.cursor != 0 => {
            return Err(ApiError::invalid(
                "conversation sequence sync cannot also advance an inbox cursor",
            ));
        }
        _ => {}
    }
    let limit = query.limit.unwrap_or(MAX_SYNC_MESSAGES);
    if !(1..=MAX_SYNC_MESSAGES).contains(&limit) {
        return Err(ApiError::invalid("limit must be between 1 and 200"));
    }
    Ok(limit)
}

fn validate_display_name(value: &str) -> ApiResult<()> {
    let value = value.trim();
    if value.is_empty() || value.chars().count() > 120 || value.chars().any(char::is_control) {
        return Err(ApiError::invalid(
            "display_name must be a printable single line of at most 120 characters",
        ));
    }
    Ok(())
}

fn validate_delivery_mode(value: &str) -> ApiResult<()> {
    if matches!(value, "pull" | "apns" | "webhook") {
        Ok(())
    } else {
        Err(ApiError::invalid(
            "delivery_mode must be pull, apns, or webhook",
        ))
    }
}

fn validate_registry_agent(
    agent_id: &str,
    display_name: &str,
    principal_kind: &str,
    delivery_mode: &str,
) -> ApiResult<()> {
    messaging_protocol::validate_agent_id(agent_id).map_err(protocol_error)?;
    validate_display_name(display_name)?;
    if !matches!(principal_kind, "resident" | "task-time" | "owner") {
        return Err(ApiError::invalid(
            "principal_kind must be resident, task-time, or owner",
        ));
    }
    validate_delivery_mode(delivery_mode)
}

fn map_agent_registry_database_error(error: sqlx::Error) -> ApiError {
    if error
        .as_database_error()
        .is_some_and(|database| database.code().as_deref() == Some("23505"))
    {
        ApiError::conflict(
            "agent_id_conflict",
            "an agent with this id already exists",
            json!({}),
        )
    } else {
        error.into()
    }
}

fn messaging_not_found(reference: &str) -> ApiError {
    ApiError::not_found("messaging_not_found", reference)
}

async fn resolve_principal_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    auth: &AuthContext,
    allow_web_owner_fallback: bool,
) -> ApiResult<Principal> {
    if let Some(row) = sqlx::query(
        r#"
        SELECT agent.agent_id,agent.principal_kind
        FROM brunn.messaging_credential_bindings AS binding
        JOIN brunn.messaging_agents AS agent
          ON agent.user_id=binding.user_id AND agent.agent_id=binding.agent_id
        WHERE binding.user_id=$1 AND binding.credential_id=$2
          AND agent.archived_at IS NULL
        "#,
    )
    .bind(auth.user_id.0)
    .bind(auth.credential_id.0)
    .fetch_optional(&mut **tx)
    .await?
    {
        return Ok(Principal {
            agent_id: row.try_get("agent_id")?,
            principal_kind: row.try_get("principal_kind")?,
        });
    }
    if allow_web_owner_fallback && has_registry_capability(auth) {
        return ensure_owner_principal_in_tx(tx, auth).await;
    }
    Err(ApiError::public(
        StatusCode::FORBIDDEN,
        "messaging_principal_unbound",
        "this credential is not bound to a messaging principal",
    ))
}

async fn ensure_owner_principal_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    auth: &AuthContext,
) -> ApiResult<Principal> {
    let rows = sqlx::query(
        r#"
        SELECT agent_id,principal_kind
        FROM brunn.messaging_agents
        WHERE user_id=$1 AND principal_kind='owner' AND archived_at IS NULL
        ORDER BY agent_id
        LIMIT 2
        "#,
    )
    .bind(auth.user_id.0)
    .fetch_all(&mut **tx)
    .await?;
    if rows.len() > 1 {
        return Err(ApiError::conflict(
            "owner_principal_ambiguous",
            "the workspace has more than one active owner messaging principal",
            json!({}),
        ));
    }
    if let Some(row) = rows.first() {
        return Ok(Principal {
            agent_id: row.try_get("agent_id")?,
            principal_kind: row.try_get("principal_kind")?,
        });
    }
    let display_name =
        sqlx::query_scalar::<_, String>("SELECT display_name FROM brunn.users WHERE id=$1")
            .bind(auth.user_id.0)
            .fetch_one(&mut **tx)
            .await?;
    sqlx::query(
        r#"
        INSERT INTO brunn.messaging_agents (
          user_id,agent_id,display_name,principal_kind,delivery_mode,
          created_by_credential_id
        ) VALUES ($1,'owner',$2,'owner','apns',$3)
        "#,
    )
    .bind(auth.user_id.0)
    .bind(display_name.trim())
    .bind(auth.credential_id.0)
    .execute(&mut **tx)
    .await
    .map_err(|error| {
        if error
            .as_database_error()
            .is_some_and(|database| database.code().as_deref() == Some("23505"))
        {
            ApiError::conflict(
                "owner_principal_id_conflict",
                "the reserved owner principal id is already in use",
                json!({}),
            )
        } else {
            error.into()
        }
    })?;
    Ok(Principal {
        agent_id: "owner".to_owned(),
        principal_kind: "owner".to_owned(),
    })
}

async fn create_conversation_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    auth: &AuthContext,
    caller: &Principal,
    requested_participants: Vec<String>,
    subject: Option<String>,
    as_of: DateTime<Utc>,
) -> ApiResult<(Uuid, bool)> {
    let mut primary = requested_participants;
    primary.push(caller.agent_id.clone());
    primary.sort();
    primary.dedup();
    if primary.len() < 2 {
        return Err(ApiError::invalid(
            "a conversation requires at least one other participant",
        ));
    }
    let rows = sqlx::query(
        r#"
        SELECT agent_id,principal_kind
        FROM brunn.messaging_agents
        WHERE user_id=$1 AND agent_id=ANY($2) AND archived_at IS NULL
        ORDER BY agent_id
        "#,
    )
    .bind(auth.user_id.0)
    .bind(&primary)
    .fetch_all(&mut **tx)
    .await?;
    if rows.len() != primary.len() {
        return Err(ApiError::invalid(
            "one or more messaging participants are unavailable",
        ));
    }
    let kinds = rows
        .into_iter()
        .map(|row| {
            Ok((
                row.try_get::<String, _>("agent_id")?,
                row.try_get::<String, _>("principal_kind")?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>, sqlx::Error>>()?;

    let conversation_kind = if primary.len() == 2 {
        "direct"
    } else {
        "group"
    };
    let direct_key =
        (conversation_kind == "direct" && subject.is_none()).then(|| direct_key(&primary));
    let lock_key = direct_key
        .as_deref()
        .map(|key| format!("messaging-direct:{}:{key}", auth.user_id.0))
        .unwrap_or_else(|| format!("messaging-create:{}:{}", auth.user_id.0, caller.agent_id));
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))")
        .bind(lock_key)
        .execute(&mut **tx)
        .await?;
    if let Some(direct_key) = direct_key.as_deref()
        && let Some(existing) = sqlx::query_scalar::<_, Uuid>(
            r#"
            SELECT conversation_id
            FROM brunn.messaging_conversations
            WHERE user_id=$1 AND direct_key=$2
              AND conversation_kind='direct'
              AND status IN ('open','paused_for_human')
            "#,
        )
        .bind(auth.user_id.0)
        .bind(direct_key)
        .fetch_optional(&mut **tx)
        .await?
    {
        return Ok((existing, true));
    }

    let mut participants = primary
        .iter()
        .cloned()
        .map(|agent_id| ConversationParticipant {
            agent_id,
            role: "participant".to_owned(),
        })
        .collect::<Vec<_>>();
    let has_owner = primary
        .iter()
        .any(|agent_id| kinds.get(agent_id).is_some_and(|kind| kind == "owner"));
    if !has_owner {
        let owner = active_owner_principal_in_tx(tx, auth.user_id.0).await?;
        if !primary.contains(&owner.agent_id) {
            participants.push(ConversationParticipant {
                agent_id: owner.agent_id,
                role: "observer".to_owned(),
            });
        }
    }
    participants.sort_by(|left, right| left.agent_id.cmp(&right.agent_id));

    let conversation_id = Uuid::now_v7();
    let entry_id = new_local_entry_id(conversation_id);
    let cursor = allocate_cursor_in_tx(tx, auth.user_id.0).await?;
    sqlx::query(
        r#"
        INSERT INTO brunn.messaging_conversations (
          user_id,conversation_id,entry_id,path,conversation_kind,direct_key,
          subject,status,created_by_agent_id,last_seq,last_message_at,
          agent_streak,needs_human,continues_from,latest_sync_cursor,
          closed_at,created_at,updated_at
        ) VALUES (
          $1,$2,$3,$4,$5,$6,$7,'open',$8,0,NULL,0,false,NULL,$9,NULL,$10,$10
        )
        "#,
    )
    .bind(auth.user_id.0)
    .bind(conversation_id)
    .bind(entry_id)
    .bind(messaging_protocol::conversation_path(conversation_id))
    .bind(conversation_kind)
    .bind(direct_key)
    .bind(subject)
    .bind(&caller.agent_id)
    .bind(cursor)
    .bind(as_of)
    .execute(&mut **tx)
    .await?;
    for participant in participants {
        sqlx::query(
            r#"
            INSERT INTO brunn.messaging_participants (
              user_id,conversation_id,agent_id,role,joined_at,updated_at
            ) VALUES ($1,$2,$3,$4,$5,$5)
            "#,
        )
        .bind(auth.user_id.0)
        .bind(conversation_id)
        .bind(participant.agent_id)
        .bind(participant.role)
        .bind(as_of)
        .execute(&mut **tx)
        .await?;
    }
    write_canonical_conversation_in_tx(tx, auth, conversation_id).await?;
    metrics::counter!(
        "messaging.conversation.create",
        "kind" => conversation_kind.to_owned()
    )
    .increment(1);
    Ok((conversation_id, false))
}

fn direct_key(participants: &[String]) -> String {
    participants.join("|")
}

fn new_local_entry_id(conversation_id: Uuid) -> Uuid {
    loop {
        let entry_id = Uuid::now_v7();
        if entry_id != conversation_id {
            return entry_id;
        }
    }
}

async fn active_owner_principal_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
) -> ApiResult<Principal> {
    let rows = sqlx::query(
        r#"
        SELECT agent_id,principal_kind
        FROM brunn.messaging_agents
        WHERE user_id=$1 AND principal_kind='owner' AND archived_at IS NULL
        ORDER BY agent_id
        LIMIT 2
        "#,
    )
    .bind(user_id)
    .fetch_all(&mut **tx)
    .await?;
    match rows.as_slice() {
        [row] => Ok(Principal {
            agent_id: row.try_get("agent_id")?,
            principal_kind: row.try_get("principal_kind")?,
        }),
        [] => Err(ApiError::conflict(
            "owner_principal_missing",
            "the workspace owner messaging principal must be configured first",
            json!({}),
        )),
        _ => Err(ApiError::conflict(
            "owner_principal_ambiguous",
            "the workspace has more than one active owner messaging principal",
            json!({}),
        )),
    }
}

async fn allocate_cursor_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
) -> ApiResult<i64> {
    Ok(sqlx::query_scalar::<_, i64>(
        r#"
        INSERT INTO brunn.messaging_sync_state (user_id,current_cursor)
        VALUES ($1,1)
        ON CONFLICT (user_id) DO UPDATE
        SET current_cursor=brunn.messaging_sync_state.current_cursor+1,
            updated_at=clock_timestamp()
        RETURNING current_cursor
        "#,
    )
    .bind(user_id)
    .fetch_one(&mut **tx)
    .await?)
}

struct SyncPage {
    response: SyncResponse,
    activity: bool,
}

async fn sync_once(
    state: &AppState,
    auth: &AuthContext,
    query: &SyncQuery,
    limit: i64,
    renew_presence: bool,
    allow_web_owner_fallback: bool,
) -> ApiResult<SyncPage> {
    let as_of = Utc::now();
    let mut tx = state.begin_write(auth).await?;
    let principal = resolve_principal_in_tx(&mut tx, auth, allow_web_owner_fallback).await?;
    if let Some(conversation_id) = query.conversation_id {
        require_membership_in_tx(
            &mut tx,
            auth.user_id.0,
            conversation_id,
            &principal.agent_id,
        )
        .await?;
    }
    if renew_presence {
        renew_presence_in_tx(&mut tx, auth.user_id.0, &principal.agent_id, as_of).await?;
        metrics::counter!("messaging.presence.renew").increment(1);
    }
    let snapshot = sqlx::query_scalar::<_, i64>(
        "SELECT current_cursor FROM brunn.messaging_sync_state WHERE user_id=$1",
    )
    .bind(auth.user_id.0)
    .fetch_optional(&mut *tx)
    .await?
    .unwrap_or(0);
    if query.cursor > snapshot {
        return Err(ApiError::invalid(
            "cursor is ahead of the workspace messaging cursor",
        ));
    }
    let (messages, cursor, has_more) = if query.conversation_id.is_none() {
        let (page_cursor, has_more) = load_inbox_page_boundary_in_tx(
            &mut tx,
            auth.user_id.0,
            &principal.agent_id,
            query.cursor,
            snapshot,
            limit,
        )
        .await?;
        let messages = load_sync_messages_in_tx(
            &mut tx,
            auth.user_id.0,
            &principal.agent_id,
            query.cursor,
            page_cursor,
            None,
            0,
            limit,
        )
        .await?;
        (messages, page_cursor, has_more)
    } else {
        let mut messages = load_sync_messages_in_tx(
            &mut tx,
            auth.user_id.0,
            &principal.agent_id,
            query.cursor,
            snapshot,
            query.conversation_id,
            query.after_seq.unwrap_or(0),
            limit + 1,
        )
        .await?;
        let has_more = messages.len() > limit as usize;
        if has_more {
            messages.truncate(limit as usize);
        }
        let cursor = if has_more {
            messages
                .last()
                .map(|message| message.sync_cursor)
                .unwrap_or(query.cursor)
        } else {
            snapshot
        };
        (messages, cursor, has_more)
    };
    if principal.principal_kind != "owner" {
        advance_pull_positions_in_tx(&mut tx, auth.user_id.0, &principal.agent_id, &messages)
            .await?;
    }
    let conversations = load_conversation_views_in_tx(
        &mut tx,
        auth.user_id.0,
        &principal.agent_id,
        query.conversation_id,
        query.cursor,
        cursor,
    )
    .await?;
    let presence = load_agent_views_in_tx(&mut tx, auth.user_id.0, as_of, false).await?;
    let unread = conversations
        .iter()
        .map(|conversation| (conversation.conversation_id, conversation.unread_count))
        .collect();
    let activity = if query.conversation_id.is_some() {
        !messages.is_empty()
    } else {
        !messages.is_empty() || !conversations.is_empty()
    };
    tx.commit().await?;
    Ok(SyncPage {
        response: SyncResponse {
            status: "complete".to_owned(),
            cursor,
            resume_cursor: None,
            has_more,
            messages,
            conversations,
            presence,
            unread,
            as_of,
        },
        activity,
    })
}

async fn load_inbox_page_boundary_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    agent_id: &str,
    after_cursor: i64,
    through_cursor: i64,
    limit: i64,
) -> ApiResult<(i64, bool)> {
    let mut cursors = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT event_cursor
        FROM (
          SELECT message.sync_cursor AS event_cursor
          FROM brunn.messaging_message_index AS message
          JOIN brunn.messaging_participants AS participant
            ON participant.user_id=message.user_id
           AND participant.conversation_id=message.conversation_id
           AND participant.agent_id=$2
          WHERE message.user_id=$1
          UNION
          SELECT conversation.latest_sync_cursor AS event_cursor
          FROM brunn.messaging_conversations AS conversation
          JOIN brunn.messaging_participants AS participant
            ON participant.user_id=conversation.user_id
           AND participant.conversation_id=conversation.conversation_id
           AND participant.agent_id=$2
          WHERE conversation.user_id=$1
        ) AS event
        WHERE event_cursor>$3 AND event_cursor<=$4
        ORDER BY event_cursor
        LIMIT $5
        "#,
    )
    .bind(user_id)
    .bind(agent_id)
    .bind(after_cursor)
    .bind(through_cursor)
    .bind(limit + 1)
    .fetch_all(&mut **tx)
    .await?;
    let has_more = cursors.len() > limit as usize;
    if has_more {
        cursors.truncate(limit as usize);
    }
    let cursor = if has_more {
        cursors.last().copied().unwrap_or(after_cursor)
    } else {
        through_cursor
    };
    Ok((cursor, has_more))
}

async fn renew_presence_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    agent_id: &str,
    as_of: DateTime<Utc>,
) -> ApiResult<()> {
    let updated = sqlx::query(
        r#"
        UPDATE brunn.messaging_agents
        SET last_seen_at=$3,
            lease_expires_at=$3 + make_interval(secs => $4),
            updated_at=$3
        WHERE user_id=$1 AND agent_id=$2 AND archived_at IS NULL
        "#,
    )
    .bind(user_id)
    .bind(agent_id)
    .bind(as_of)
    .bind(PRESENCE_LEASE_SECONDS as f64)
    .execute(&mut **tx)
    .await?
    .rows_affected();
    if updated != 1 {
        return Err(ApiError::public(
            StatusCode::FORBIDDEN,
            "messaging_principal_unavailable",
            "the bound messaging principal is unavailable",
        ));
    }
    Ok(())
}

async fn load_sync_messages_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    agent_id: &str,
    after_cursor: i64,
    through_cursor: i64,
    conversation_id: Option<Uuid>,
    after_seq: i64,
    limit: i64,
) -> ApiResult<Vec<MessageView>> {
    let rows = sqlx::query(
        r#"
        SELECT message.conversation_id,message.seq,message.message_id,
               message.from_agent_id,message.client_key,message.request_hash,
               message.kind,message.body_md,message.refs,
               message.in_reply_to_conversation_id,message.in_reply_to,
               message.correlation_id,message.expects_reply,message.reply_by,
               message.sync_cursor,message.created_at
        FROM brunn.messaging_message_index AS message
        JOIN brunn.messaging_participants AS participant
          ON participant.user_id=message.user_id
         AND participant.conversation_id=message.conversation_id
         AND participant.agent_id=$2
        WHERE message.user_id=$1
          AND message.sync_cursor>$3
          AND message.sync_cursor<=$4
          AND ($5::uuid IS NULL OR message.conversation_id=$5)
          AND ($5::uuid IS NULL OR message.seq>$6)
        ORDER BY message.sync_cursor,message.conversation_id,message.seq
        LIMIT $7
        "#,
    )
    .bind(user_id)
    .bind(agent_id)
    .bind(after_cursor)
    .bind(through_cursor)
    .bind(conversation_id)
    .bind(after_seq)
    .bind(limit)
    .fetch_all(&mut **tx)
    .await?;
    rows.into_iter().map(message_view_from_row).collect()
}

async fn advance_pull_positions_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    agent_id: &str,
    messages: &[MessageView],
) -> ApiResult<()> {
    let mut maxima = BTreeMap::<Uuid, i64>::new();
    for message in messages {
        maxima
            .entry(message.conversation_id)
            .and_modify(|seq| *seq = (*seq).max(message.seq))
            .or_insert(message.seq);
    }
    for (conversation_id, last_read_seq) in maxima {
        sqlx::query(
            r#"
            UPDATE brunn.messaging_participants
            SET last_read_seq=greatest(last_read_seq,$4),updated_at=clock_timestamp()
            WHERE user_id=$1 AND conversation_id=$2 AND agent_id=$3
            "#,
        )
        .bind(user_id)
        .bind(conversation_id)
        .bind(agent_id)
        .bind(last_read_seq)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

async fn load_conversation_views_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    agent_id: &str,
    exact_conversation_id: Option<Uuid>,
    after_cursor: i64,
    through_cursor: i64,
) -> ApiResult<Vec<ConversationView>> {
    let rows = sqlx::query(
        r#"
        SELECT conversation.conversation_id,conversation.conversation_kind,
               conversation.subject,conversation.status,conversation.last_seq,
               conversation.last_message_at,participant.last_read_seq,
               greatest(conversation.last_seq-participant.last_read_seq,0) AS unread_count,
               conversation.needs_human,conversation.continues_from,
               conversation.latest_sync_cursor,
               (
                 SELECT child.conversation_id
                 FROM brunn.messaging_conversations AS child
                 WHERE child.user_id=conversation.user_id
                   AND child.continues_from=conversation.conversation_id
                 LIMIT 1
               ) AS continuation_id,
               coalesce((
                 SELECT jsonb_agg(
                   jsonb_build_object('agent_id',member.agent_id,'role',member.role)
                   ORDER BY member.agent_id
                 )
                 FROM brunn.messaging_participants AS member
                 WHERE member.user_id=conversation.user_id
                   AND member.conversation_id=conversation.conversation_id
               ),'[]'::jsonb) AS participants
        FROM brunn.messaging_conversations AS conversation
        JOIN brunn.messaging_participants AS participant
          ON participant.user_id=conversation.user_id
         AND participant.conversation_id=conversation.conversation_id
         AND participant.agent_id=$2
        WHERE conversation.user_id=$1
          AND (
            ($3::uuid IS NOT NULL AND conversation.conversation_id=$3)
            OR
            ($3::uuid IS NULL
              AND (
                (conversation.latest_sync_cursor>$4
                  AND conversation.latest_sync_cursor<=$5)
                OR EXISTS (
                  SELECT 1
                  FROM brunn.messaging_message_index AS page_message
                  WHERE page_message.user_id=conversation.user_id
                    AND page_message.conversation_id=conversation.conversation_id
                    AND page_message.sync_cursor>$4
                    AND page_message.sync_cursor<=$5
                )
              ))
          )
        ORDER BY conversation.last_message_at DESC NULLS LAST,
                 conversation.created_at DESC,conversation.conversation_id
        "#,
    )
    .bind(user_id)
    .bind(agent_id)
    .bind(exact_conversation_id)
    .bind(after_cursor)
    .bind(through_cursor)
    .fetch_all(&mut **tx)
    .await?;
    rows.into_iter().map(conversation_view_from_row).collect()
}

async fn load_one_conversation_view_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    agent_id: &str,
    conversation_id: Uuid,
) -> ApiResult<ConversationView> {
    load_conversation_views_in_tx(tx, user_id, agent_id, Some(conversation_id), 0, 0)
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| messaging_not_found(&conversation_id.to_string()))
}

fn conversation_view_from_row(row: PgRow) -> ApiResult<ConversationView> {
    let participants =
        serde_json::from_value::<Vec<ParticipantView>>(row.try_get("participants")?)?;
    Ok(ConversationView {
        conversation_id: row.try_get("conversation_id")?,
        conversation_kind: row.try_get("conversation_kind")?,
        subject: row.try_get("subject")?,
        status: row.try_get("status")?,
        participants,
        last_seq: row.try_get("last_seq")?,
        last_message_at: row.try_get("last_message_at")?,
        last_read_seq: row.try_get("last_read_seq")?,
        unread_count: row.try_get("unread_count")?,
        needs_human: row.try_get("needs_human")?,
        continues_from: row.try_get("continues_from")?,
        continuation_id: row.try_get("continuation_id")?,
        latest_sync_cursor: row.try_get("latest_sync_cursor")?,
    })
}

fn message_view_from_row(row: PgRow) -> ApiResult<MessageView> {
    let refs = serde_json::from_value::<Vec<MessageRef>>(row.try_get("refs")?)?;
    Ok(MessageView {
        conversation_id: row.try_get("conversation_id")?,
        seq: row.try_get("seq")?,
        message_id: row.try_get("message_id")?,
        from_agent_id: row.try_get("from_agent_id")?,
        client_key: row.try_get("client_key")?,
        request_hash: row.try_get("request_hash")?,
        kind: row.try_get("kind")?,
        body_md: row.try_get("body_md")?,
        refs,
        in_reply_to_conversation_id: row.try_get("in_reply_to_conversation_id")?,
        in_reply_to: row.try_get("in_reply_to")?,
        correlation_id: row.try_get("correlation_id")?,
        expects_reply: row.try_get("expects_reply")?,
        reply_by: row.try_get("reply_by")?,
        sync_cursor: row.try_get("sync_cursor")?,
        created_at: row.try_get("created_at")?,
    })
}

async fn load_agent_views_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    as_of: DateTime<Utc>,
    reveal_bindings: bool,
) -> ApiResult<Vec<AgentView>> {
    let binding_names = if reveal_bindings {
        let rows = sqlx::query(
            r#"
            SELECT binding.agent_id,credential.label
            FROM brunn.messaging_credential_bindings AS binding
            JOIN brunn_auth.list_credentials($1) AS credential
              ON credential.id=binding.credential_id
            WHERE binding.user_id=$1
            ORDER BY binding.agent_id,credential.label
            "#,
        )
        .bind(user_id)
        .fetch_all(&mut **tx)
        .await?;
        let mut names = BTreeMap::<String, Vec<String>>::new();
        for row in rows {
            names
                .entry(row.try_get("agent_id")?)
                .or_default()
                .push(row.try_get("label")?);
        }
        Some(names)
    } else {
        None
    };
    let rows = sqlx::query(
        r#"
        SELECT agent_id,display_name,principal_kind,delivery_mode,
               lease_expires_at,last_seen_at,archived_at
        FROM brunn.messaging_agents
        WHERE user_id=$1
        ORDER BY archived_at NULLS FIRST,display_name,agent_id
        "#,
    )
    .bind(user_id)
    .fetch_all(&mut **tx)
    .await?;
    rows.into_iter()
        .map(|row| {
            let agent_id: String = row.try_get("agent_id")?;
            let lease_expires_at: Option<DateTime<Utc>> = row.try_get("lease_expires_at")?;
            let archived_at: Option<DateTime<Utc>> = row.try_get("archived_at")?;
            Ok(AgentView {
                credential_names: binding_names
                    .as_ref()
                    .map(|names| names.get(&agent_id).cloned().unwrap_or_default()),
                agent_id,
                display_name: row.try_get("display_name")?,
                principal_kind: row.try_get("principal_kind")?,
                delivery_mode: row.try_get("delivery_mode")?,
                online: archived_at.is_none()
                    && lease_expires_at.is_some_and(|lease| lease > as_of),
                last_seen_at: row.try_get("last_seen_at")?,
                lease_expires_at,
                archived: archived_at.is_some(),
            })
        })
        .collect::<Result<Vec<_>, sqlx::Error>>()
        .map_err(Into::into)
}

async fn load_one_agent_view_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    agent_id: &str,
    as_of: DateTime<Utc>,
    reveal_bindings: bool,
) -> ApiResult<AgentView> {
    load_agent_views_in_tx(tx, user_id, as_of, reveal_bindings)
        .await?
        .into_iter()
        .find(|agent| agent.agent_id == agent_id)
        .ok_or_else(|| messaging_not_found(agent_id))
}

async fn require_membership_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    conversation_id: Uuid,
    agent_id: &str,
) -> ApiResult<String> {
    sqlx::query_scalar::<_, String>(
        r#"
        SELECT role
        FROM brunn.messaging_participants
        WHERE user_id=$1 AND conversation_id=$2 AND agent_id=$3
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(agent_id)
    .fetch_optional(&mut **tx)
    .await?
    .ok_or_else(|| messaging_not_found(&conversation_id.to_string()))
}

async fn owner_is_participant_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    conversation_id: Uuid,
) -> ApiResult<bool> {
    Ok(sqlx::query_scalar::<_, bool>(
        r#"
        SELECT EXISTS(
          SELECT 1
          FROM brunn.messaging_participants AS participant
          JOIN brunn.messaging_agents AS agent
            ON agent.user_id=participant.user_id
           AND agent.agent_id=participant.agent_id
          WHERE participant.user_id=$1
            AND participant.conversation_id=$2
            AND participant.role='participant'
            AND agent.principal_kind='owner'
            AND agent.archived_at IS NULL
        )
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .fetch_one(&mut **tx)
    .await?)
}

async fn promote_owner_participant_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    conversation_id: Uuid,
    owner_agent_id: &str,
    as_of: DateTime<Utc>,
) -> ApiResult<bool> {
    let promoted = sqlx::query(
        r#"
        UPDATE brunn.messaging_participants
        SET role='participant',updated_at=$4
        WHERE user_id=$1 AND conversation_id=$2 AND agent_id=$3
          AND role='observer'
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(owner_agent_id)
    .bind(as_of)
    .execute(&mut **tx)
    .await?;
    if promoted.rows_affected() == 0 {
        return Ok(false);
    }
    sqlx::query(
        r#"
        UPDATE brunn.messaging_conversations
        SET conversation_kind='group',direct_key=NULL,updated_at=$3
        WHERE user_id=$1 AND conversation_id=$2
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(as_of)
    .execute(&mut **tx)
    .await?;
    Ok(true)
}

async fn cancel_reply_deadlines_for_chain_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    conversation_id: Uuid,
    as_of: DateTime<Utc>,
) -> ApiResult<Vec<Uuid>> {
    Ok(sqlx::query_scalar::<_, Uuid>(
        r#"
        WITH RECURSIVE ancestors AS (
          SELECT conversation_id,continues_from
          FROM brunn.messaging_conversations
          WHERE user_id=$1 AND conversation_id=$2
          UNION
          SELECT parent.conversation_id,parent.continues_from
          FROM brunn.messaging_conversations AS parent
          JOIN ancestors AS child
            ON child.continues_from=parent.conversation_id
          WHERE parent.user_id=$1
        ), canceled AS (
          UPDATE brunn.messaging_message_index AS message
          SET reply_by_handled_at=GREATEST($3,message.reply_by)
          FROM ancestors
          WHERE message.user_id=$1
            AND message.conversation_id=ancestors.conversation_id
            AND message.reply_by IS NOT NULL
            AND message.reply_by_handled_at IS NULL
          RETURNING message.conversation_id
        )
        SELECT DISTINCT conversation_id FROM canceled
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(as_of)
    .fetch_all(&mut **tx)
    .await?)
}

async fn require_conversation_sender(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    conversation_id: Uuid,
    sender: &Principal,
    conversation: &ConversationRow,
) -> ApiResult<()> {
    if conversation.status == "closed" {
        return Err(ApiError::conflict(
            "conversation_closed",
            "the conversation is closed",
            json!({}),
        ));
    }
    let role = require_membership_in_tx(tx, user_id, conversation_id, &sender.agent_id).await?;
    if role != "participant" && sender.principal_kind != "owner" {
        return Err(messaging_not_found(&conversation_id.to_string()));
    }
    Ok(())
}

async fn acquire_sender_lock(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    sender_agent_id: &str,
) -> ApiResult<()> {
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))")
        .bind(format!("messaging-sender:{user_id}:{sender_agent_id}"))
        .execute(&mut **tx)
        .await?;
    Ok(())
}

async fn load_replay_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    sender_agent_id: &str,
    client_key: &str,
) -> ApiResult<Option<MessageView>> {
    sqlx::query(
        r#"
        SELECT conversation_id,seq,message_id,from_agent_id,client_key,
               request_hash,kind,body_md,refs,in_reply_to_conversation_id,
               in_reply_to,correlation_id,
               expects_reply,reply_by,sync_cursor,created_at
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND from_agent_id=$2 AND client_key=$3
        "#,
    )
    .bind(user_id)
    .bind(sender_agent_id)
    .bind(client_key)
    .fetch_optional(&mut **tx)
    .await?
    .map(message_view_from_row)
    .transpose()
}

async fn continuation_chain_contains_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    ancestor_id: Uuid,
    descendant_id: Uuid,
) -> ApiResult<bool> {
    let mut current = Some(descendant_id);
    for _ in 0..MAX_CONTINUATION_HOPS {
        let Some(conversation_id) = current else {
            return Ok(false);
        };
        if conversation_id == ancestor_id {
            return Ok(true);
        }
        current = sqlx::query_scalar::<_, Option<Uuid>>(
            r#"
            SELECT continues_from
            FROM brunn.messaging_conversations
            WHERE user_id=$1 AND conversation_id=$2
            "#,
        )
        .bind(user_id)
        .bind(conversation_id)
        .fetch_optional(&mut **tx)
        .await?
        .flatten();
    }
    Err(ApiError::Internal(
        "messaging continuation chain exceeded its bounded depth".to_owned(),
    ))
}

async fn load_writable_conversation_for_update(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    requested_id: Uuid,
) -> ApiResult<(Uuid, ConversationRow)> {
    let mut current = requested_id;
    for _ in 0..MAX_CONTINUATION_HOPS {
        let conversation = load_conversation_for_update(tx, user_id, current).await?;
        if conversation.status != "closed" {
            return Ok((current, conversation));
        }
        let next = sqlx::query_scalar::<_, Uuid>(
            r#"
            SELECT conversation_id
            FROM brunn.messaging_conversations
            WHERE user_id=$1 AND continues_from=$2
            "#,
        )
        .bind(user_id)
        .bind(current)
        .fetch_optional(&mut **tx)
        .await?;
        let Some(next) = next else {
            return Err(ApiError::conflict(
                "conversation_closed",
                "the conversation is closed",
                json!({}),
            ));
        };
        current = next;
    }
    Err(ApiError::Internal(
        "messaging continuation chain exceeded its bounded depth".to_owned(),
    ))
}

async fn load_conversation_for_update(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    conversation_id: Uuid,
) -> ApiResult<ConversationRow> {
    let row = sqlx::query(
        r#"
        SELECT conversation_id,entry_id,conversation_kind,direct_key,subject,status,
               created_by_agent_id,last_seq,agent_streak,
               needs_human,continues_from,latest_sync_cursor,closed_at,created_at
        FROM brunn.messaging_conversations
        WHERE user_id=$1 AND conversation_id=$2
        FOR UPDATE
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .fetch_optional(&mut **tx)
    .await?
    .ok_or_else(|| messaging_not_found(&conversation_id.to_string()))?;
    Ok(ConversationRow {
        conversation_id: row.try_get("conversation_id")?,
        entry_id: row.try_get("entry_id")?,
        conversation_kind: row.try_get("conversation_kind")?,
        direct_key: row.try_get("direct_key")?,
        subject: row.try_get("subject")?,
        status: row.try_get("status")?,
        created_by_agent_id: row.try_get("created_by_agent_id")?,
        last_seq: row.try_get("last_seq")?,
        agent_streak: row.try_get("agent_streak")?,
        needs_human: row.try_get("needs_human")?,
        continues_from: row.try_get("continues_from")?,
        latest_sync_cursor: row.try_get("latest_sync_cursor")?,
        closed_at: row.try_get("closed_at")?,
        created_at: row.try_get("created_at")?,
    })
}

async fn check_send_rates_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    sender_agent_id: &str,
    conversation_id: Uuid,
    as_of: DateTime<Utc>,
) -> ApiResult<()> {
    let sender = sqlx::query(
        r#"
        SELECT count(*)::bigint AS message_count,min(created_at) AS oldest
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND from_agent_id=$2
          AND created_at>$3 - interval '1 minute'
        "#,
    )
    .bind(user_id)
    .bind(sender_agent_id)
    .bind(as_of)
    .fetch_one(&mut **tx)
    .await?;
    let sender_count: i64 = sender.try_get("message_count")?;
    if sender_count >= SENDER_RATE_LIMIT {
        let oldest: Option<DateTime<Utc>> = sender.try_get("oldest")?;
        let retry_after = oldest
            .map(|value| retry_after_seconds(value + chrono::Duration::minutes(1), as_of))
            .unwrap_or(1);
        metrics::counter!("messaging.guard", "kind" => "sender_rate").increment(1);
        return Err(ApiError::with_details(
            StatusCode::TOO_MANY_REQUESTS,
            "sender_rate_limited",
            "this principal has reached the 60 messages per minute limit",
            json!({"retry_after_seconds": retry_after}),
        ));
    }
    let conversation = sqlx::query(
        r#"
        SELECT count(*)::bigint AS message_count,min(created_at) AS oldest
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND conversation_id=$2 AND kind<>'system'
          AND created_at>$3 - interval '1 hour'
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(as_of)
    .fetch_one(&mut **tx)
    .await?;
    let conversation_count: i64 = conversation.try_get("message_count")?;
    if conversation_count >= CONVERSATION_RATE_LIMIT {
        let oldest: Option<DateTime<Utc>> = conversation.try_get("oldest")?;
        let retry_after = oldest
            .map(|value| retry_after_seconds(value + chrono::Duration::hours(1), as_of))
            .unwrap_or(1);
        metrics::counter!("messaging.guard", "kind" => "conversation_rate").increment(1);
        return Err(ApiError::with_details(
            StatusCode::TOO_MANY_REQUESTS,
            "conversation_rate_limited",
            "this conversation has reached the 200 messages per hour limit",
            json!({"retry_after_seconds": retry_after}),
        ));
    }
    Ok(())
}

fn retry_after_seconds(available_at: DateTime<Utc>, as_of: DateTime<Utc>) -> i64 {
    let milliseconds = (available_at - as_of).num_milliseconds().max(1);
    ((milliseconds + 999) / 1_000).max(1)
}

async fn validate_reply_target_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    requested_conversation_id: Uuid,
    target_conversation_id: Uuid,
    in_reply_to: Option<i64>,
) -> ApiResult<Option<Uuid>> {
    let Some(in_reply_to) = in_reply_to else {
        return Ok(None);
    };
    if !continuation_chain_contains_in_tx(
        tx,
        user_id,
        requested_conversation_id,
        target_conversation_id,
    )
    .await?
    {
        return Err(ApiError::invalid(
            "in_reply_to must name a message in this conversation or one of its ancestors",
        ));
    }
    let exists = sqlx::query_scalar::<_, bool>(
        r#"
        SELECT EXISTS(
          SELECT 1 FROM brunn.messaging_message_index
          WHERE user_id=$1 AND conversation_id=$2 AND seq=$3
        )
        "#,
    )
    .bind(user_id)
    .bind(requested_conversation_id)
    .bind(in_reply_to)
    .fetch_one(&mut **tx)
    .await?;
    if exists {
        Ok(Some(requested_conversation_id))
    } else {
        Err(ApiError::invalid(
            "in_reply_to must name an existing message in this conversation or one of its ancestors",
        ))
    }
}

async fn insert_client_message_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    conversation_id: Uuid,
    seq: i64,
    message_id: Uuid,
    sender_agent_id: &str,
    request_hash: &str,
    in_reply_to_conversation_id: Option<Uuid>,
    input: &SendMessageInput,
    sync_cursor: i64,
    created_at: DateTime<Utc>,
) -> ApiResult<()> {
    sqlx::query(
        r#"
        INSERT INTO brunn.messaging_message_index (
          user_id,conversation_id,seq,message_id,from_agent_id,client_key,
          system_key,request_hash,kind,body_md,refs,
          in_reply_to_conversation_id,in_reply_to,
          correlation_id,expects_reply,reply_by,reply_by_handled_at,
          sync_cursor,created_at
        ) VALUES (
          $1,$2,$3,$4,$5,$6,NULL,$7,$8,$9,$10,$11,$12,$13,$14,$15,NULL,$16,$17
        )
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(seq)
    .bind(message_id)
    .bind(sender_agent_id)
    .bind(&input.client_key)
    .bind(request_hash)
    .bind(message_kind_str(input.kind))
    .bind(&input.body_md)
    .bind(serde_json::to_value(&input.refs)?)
    .bind(in_reply_to_conversation_id)
    .bind(input.in_reply_to)
    .bind(input.correlation_id.as_deref())
    .bind(input.expects_reply)
    .bind(input.reply_by)
    .bind(sync_cursor)
    .bind(created_at)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn insert_system_message_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    conversation_id: Uuid,
    seq: i64,
    system_key: &str,
    body_md: &str,
    sync_cursor: i64,
    created_at: DateTime<Utc>,
) -> ApiResult<()> {
    sqlx::query(
        r#"
        INSERT INTO brunn.messaging_message_index (
          user_id,conversation_id,seq,message_id,from_agent_id,client_key,
          system_key,request_hash,kind,body_md,refs,
          in_reply_to_conversation_id,in_reply_to,
          correlation_id,expects_reply,reply_by,reply_by_handled_at,
          sync_cursor,created_at
        ) VALUES (
          $1,$2,$3,$4,NULL,NULL,$5,NULL,'system',$6,'[]'::jsonb,
          NULL,NULL,NULL,false,NULL,NULL,$7,$8
        )
        ON CONFLICT (user_id,conversation_id,system_key)
          WHERE system_key IS NOT NULL DO NOTHING
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(seq)
    .bind(Uuid::now_v7())
    .bind(system_key)
    .bind(body_md)
    .bind(sync_cursor)
    .bind(created_at)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

fn message_kind_str(kind: MessageKind) -> &'static str {
    match kind {
        MessageKind::Text => "text",
        MessageKind::Question => "question",
        MessageKind::System => "system",
    }
}

async fn load_message_by_seq_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    conversation_id: Uuid,
    seq: i64,
) -> ApiResult<MessageView> {
    let row = sqlx::query(
        r#"
        SELECT conversation_id,seq,message_id,from_agent_id,client_key,
               request_hash,kind,body_md,refs,in_reply_to_conversation_id,
               in_reply_to,correlation_id,
               expects_reply,reply_by,sync_cursor,created_at
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND conversation_id=$2 AND seq=$3
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(seq)
    .fetch_optional(&mut **tx)
    .await?
    .ok_or_else(|| messaging_not_found(&conversation_id.to_string()))?;
    message_view_from_row(row)
}

async fn close_for_rollover_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    conversation_id: Uuid,
    last_seq: i64,
    latest_sync_cursor: i64,
    agent_streak: i32,
    as_of: DateTime<Utc>,
) -> ApiResult<()> {
    sqlx::query(
        r#"
        UPDATE brunn.messaging_conversations
        SET status='closed',closed_at=$4,last_seq=$3,last_message_at=$4,
            agent_streak=$5,needs_human=false,latest_sync_cursor=$6,
            updated_at=$4
        WHERE user_id=$1 AND conversation_id=$2
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(last_seq)
    .bind(as_of)
    .bind(agent_streak)
    .bind(latest_sync_cursor)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn create_continuation_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    state: &AppState,
    auth: &AuthContext,
    previous: &ConversationRow,
    created_by_agent_id: &str,
    agent_streak: i32,
    paused: bool,
    needs_human: bool,
    include_pause_system: bool,
    notify_system: bool,
    as_of: DateTime<Utc>,
) -> ApiResult<Uuid> {
    let continuation_id = Uuid::now_v7();
    let entry_id = new_local_entry_id(continuation_id);
    let continuation_cursor = allocate_cursor_in_tx(tx, auth.user_id.0).await?;
    let pause_cursor = if include_pause_system {
        Some(allocate_cursor_in_tx(tx, auth.user_id.0).await?)
    } else {
        None
    };
    let last_seq = if include_pause_system { 2 } else { 1 };
    let latest_cursor = pause_cursor.unwrap_or(continuation_cursor);
    sqlx::query(
        r#"
        INSERT INTO brunn.messaging_conversations (
          user_id,conversation_id,entry_id,path,conversation_kind,direct_key,
          subject,status,created_by_agent_id,last_seq,last_message_at,
          agent_streak,needs_human,continues_from,latest_sync_cursor,
          closed_at,created_at,updated_at
        ) VALUES (
          $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,NULL,$11,$11
        )
        "#,
    )
    .bind(auth.user_id.0)
    .bind(continuation_id)
    .bind(entry_id)
    .bind(messaging_protocol::conversation_path(continuation_id))
    .bind(&previous.conversation_kind)
    .bind(&previous.direct_key)
    .bind(&previous.subject)
    .bind(if paused { "paused_for_human" } else { "open" })
    .bind(created_by_agent_id)
    .bind(last_seq)
    .bind(as_of)
    .bind(agent_streak)
    .bind(needs_human)
    .bind(previous.conversation_id)
    .bind(latest_cursor)
    .execute(&mut **tx)
    .await?;
    sqlx::query(
        r#"
        INSERT INTO brunn.messaging_participants (
          user_id,conversation_id,agent_id,role,last_read_seq,joined_at,updated_at
        )
        SELECT user_id,$3,agent_id,role,0,$4,$4
        FROM brunn.messaging_participants
        WHERE user_id=$1 AND conversation_id=$2
        ORDER BY agent_id
        "#,
    )
    .bind(auth.user_id.0)
    .bind(previous.conversation_id)
    .bind(continuation_id)
    .bind(as_of)
    .execute(&mut **tx)
    .await?;
    insert_system_message_in_tx(
        tx,
        auth.user_id.0,
        continuation_id,
        1,
        &format!("continuation:{}", previous.conversation_id),
        messaging_protocol::CONTINUATION_SYSTEM_BODY,
        continuation_cursor,
        as_of,
    )
    .await?;
    if let Some(pause_cursor) = pause_cursor {
        insert_system_message_in_tx(
            tx,
            auth.user_id.0,
            continuation_id,
            2,
            &format!("budget:{continuation_id}:2"),
            "Agent exchange paused after 20 consecutive messages. An owner response is required.",
            pause_cursor,
            as_of,
        )
        .await?;
    }
    write_canonical_conversation_in_tx(tx, auth, continuation_id).await?;
    if notify_system {
        publish_conversation_notification_in_tx(
            tx,
            state,
            auth,
            continuation_id,
            1,
            "system",
            None,
            as_of,
        )
        .await?;
    }
    Ok(continuation_id)
}

pub async fn process_due_reply_by(state: &AppState, as_of: DateTime<Utc>) -> ApiResult<bool> {
    let pool = state.admin_pool.as_ref().ok_or_else(|| {
        ApiError::configuration("the messaging worker requires DATABASE_URL_ADMIN")
    })?;
    let mut tx = pool.begin().await?;
    let candidate = sqlx::query(
        r#"
        SELECT question.user_id,question.conversation_id,question.seq
        FROM brunn.messaging_message_index AS question
        WHERE question.kind='question'
          AND question.expects_reply
          AND question.reply_by IS NOT NULL
          AND question.reply_by<=$1
          AND question.reply_by_handled_at IS NULL
        ORDER BY question.reply_by,question.user_id,
                 question.conversation_id,question.seq
        LIMIT 1
        "#,
    )
    .bind(as_of)
    .fetch_optional(&mut *tx)
    .await?;
    let Some(candidate) = candidate else {
        tx.commit().await?;
        return Ok(false);
    };
    let user_id: Uuid = candidate.try_get("user_id")?;
    let question_conversation_id: Uuid = candidate.try_get("conversation_id")?;
    let question_seq: i64 = candidate.try_get("seq")?;

    // Match the send path's conversation-before-message lock order. This
    // serializes a reply racing its deadline without a question/conversation
    // deadlock or a false expiry.
    let (target_id, conversation) =
        load_writable_conversation_for_update(&mut tx, user_id, question_conversation_id).await?;
    let due = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT seq
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND conversation_id=$2 AND seq=$3
          AND kind='question' AND expects_reply
          AND reply_by IS NOT NULL AND reply_by<=$4
          AND reply_by_handled_at IS NULL
        FOR UPDATE SKIP LOCKED
        "#,
    )
    .bind(user_id)
    .bind(question_conversation_id)
    .bind(question_seq)
    .bind(as_of)
    .fetch_optional(&mut *tx)
    .await?;
    if due.is_none() {
        tx.commit().await?;
        return Ok(false);
    }
    let producer_credential_id = sqlx::query_scalar::<_, Uuid>(
        r#"
        SELECT version.created_by_credential_id
        FROM brunn.messaging_conversations AS conversation
        JOIN brunn.entries AS entry
          ON entry.user_id=conversation.user_id
         AND entry.id=conversation.entry_id
        JOIN brunn.entry_versions AS version
          ON version.user_id=entry.user_id
         AND version.entry_id=entry.id
         AND version.version=entry.current_version
        WHERE conversation.user_id=$1 AND conversation.conversation_id=$2
        "#,
    )
    .bind(user_id)
    .bind(question_conversation_id)
    .fetch_one(&mut *tx)
    .await?;
    let auth = worker_auth(user_id, producer_credential_id);

    let answered = sqlx::query_scalar::<_, bool>(
        r#"
        SELECT EXISTS(
          SELECT 1
          FROM brunn.messaging_message_index AS reply
          WHERE reply.user_id=$1
            AND reply.in_reply_to_conversation_id=$2
            AND reply.in_reply_to=$3
        )
        "#,
    )
    .bind(user_id)
    .bind(question_conversation_id)
    .bind(question_seq)
    .fetch_one(&mut *tx)
    .await?;

    sqlx::query(
        r#"
        UPDATE brunn.messaging_message_index
        SET reply_by_handled_at=$4
        WHERE user_id=$1 AND conversation_id=$2 AND seq=$3
          AND reply_by_handled_at IS NULL
        "#,
    )
    .bind(user_id)
    .bind(question_conversation_id)
    .bind(question_seq)
    .bind(as_of)
    .execute(&mut *tx)
    .await?;
    if answered {
        write_canonical_conversation_in_tx(&mut tx, &auth, question_conversation_id).await?;
        tx.commit().await?;
        state.workspace_features.invalidate(user_id).await;
        metrics::counter!("messaging.reply_by", "result" => "answered").increment(1);
        return Ok(true);
    }

    let system_seq = conversation.last_seq + 1;
    if system_seq > messaging_protocol::MAX_MESSAGES_PER_CONVERSATION {
        return Err(ApiError::Internal(
            "an open messaging conversation exceeded its entry budget".to_owned(),
        ));
    }
    let event_key = format!("reply-by:{question_conversation_id}:{question_seq}");
    let system_cursor = allocate_cursor_in_tx(&mut tx, user_id).await?;
    insert_system_message_in_tx(
        &mut tx,
        user_id,
        target_id,
        system_seq,
        &event_key,
        "The requested reply window expired without a response.",
        system_cursor,
        as_of,
    )
    .await?;

    if system_seq == messaging_protocol::MAX_MESSAGES_PER_CONVERSATION {
        close_for_rollover_in_tx(
            &mut tx,
            user_id,
            target_id,
            system_seq,
            system_cursor,
            conversation.agent_streak,
            as_of,
        )
        .await?;
        write_canonical_conversation_in_tx(&mut tx, &auth, target_id).await?;
        create_continuation_in_tx(
            &mut tx,
            state,
            &auth,
            &conversation,
            &conversation.created_by_agent_id,
            conversation.agent_streak,
            false,
            true,
            false,
            false,
            as_of,
        )
        .await?;
    } else {
        sqlx::query(
            r#"
            UPDATE brunn.messaging_conversations
            SET last_seq=$3,last_message_at=$4,needs_human=true,
                latest_sync_cursor=$5,updated_at=$4
            WHERE user_id=$1 AND conversation_id=$2
            "#,
        )
        .bind(user_id)
        .bind(target_id)
        .bind(system_seq)
        .bind(as_of)
        .bind(system_cursor)
        .execute(&mut *tx)
        .await?;
        write_canonical_conversation_in_tx(&mut tx, &auth, target_id).await?;
    }
    if target_id != question_conversation_id {
        write_canonical_conversation_in_tx(&mut tx, &auth, question_conversation_id).await?;
    }
    publish_conversation_notification_in_tx(
        &mut tx,
        state,
        &auth,
        target_id,
        system_seq,
        "reply-by",
        Some(&event_key),
        as_of,
    )
    .await?;
    tx.commit().await?;
    state.workspace_features.invalidate(user_id).await;
    metrics::counter!("messaging.reply_by", "result" => "expired").increment(1);
    Ok(true)
}

fn worker_auth(user_id: Uuid, credential_id: Uuid) -> AuthContext {
    AuthContext {
        credential_id: CredentialId(credential_id),
        user_id: UserId(user_id),
        capabilities: HashSet::from(["admin".to_owned(), "message.write".to_owned()]),
        scope_refs: Vec::new(),
        read_only: false,
    }
}

pub(crate) async fn sync_managed_entry_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    entry_id: Uuid,
    entry_version: i64,
    path: &str,
    metadata: &Value,
) -> ApiResult<()> {
    if !messaging_protocol::is_conversation_candidate(path, metadata) {
        return Ok(());
    }
    if !messaging_protocol::is_workspace_import(metadata) {
        return Err(ApiError::invalid(
            "canonical conversation projection rebuild requires workspace import metadata",
        ));
    }
    let content = sqlx::query_scalar::<_, Option<String>>(
        r#"
        SELECT content
        FROM brunn.entry_versions
        WHERE user_id=$1 AND entry_id=$2 AND version=$3
        "#,
    )
    .bind(user_id)
    .bind(entry_id)
    .bind(entry_version)
    .fetch_optional(&mut **tx)
    .await?
    .flatten()
    .ok_or_else(|| ApiError::invalid("canonical conversation entry content is unavailable"))?;
    let (header, messages) =
        messaging_protocol::validate_conversation_entry(path, metadata, &content)
            .map_err(protocol_error)?
            .ok_or_else(|| ApiError::invalid("canonical conversation metadata is required"))?;
    let mut required_principals = BTreeSet::new();
    required_principals.insert(header.created_by_agent_id.clone());
    required_principals.extend(
        header
            .participants
            .iter()
            .map(|participant| participant.agent_id.clone()),
    );
    required_principals.extend(
        messages
            .iter()
            .filter_map(|message| message.from_agent_id.clone()),
    );
    let required_principals = required_principals.into_iter().collect::<Vec<_>>();
    let present_principals = sqlx::query_scalar::<_, String>(
        r#"
        SELECT agent_id
        FROM brunn.messaging_agents
        WHERE user_id=$1 AND agent_id=ANY($2)
        ORDER BY agent_id
        "#,
    )
    .bind(user_id)
    .bind(&required_principals)
    .fetch_all(&mut **tx)
    .await?;
    if present_principals != required_principals {
        return Err(ApiError::invalid(
            "all canonical conversation principals must already exist for this user",
        ));
    }

    if let Some(parent_id) = header.continues_from {
        let parent = sqlx::query(
            r#"
            SELECT conversation_kind,direct_key,subject,status,last_seq
            FROM brunn.messaging_conversations
            WHERE user_id=$1 AND conversation_id=$2
            "#,
        )
        .bind(user_id)
        .bind(parent_id)
        .fetch_optional(&mut **tx)
        .await?
        .ok_or_else(|| {
            ApiError::invalid("conversation continuation parent must be imported before its child")
        })?;
        let expected_kind = match header.conversation_kind {
            ConversationKind::Direct => "direct",
            ConversationKind::Group => "group",
        };
        let parent_participants = sqlx::query(
            r#"
            SELECT agent_id,role
            FROM brunn.messaging_participants
            WHERE user_id=$1 AND conversation_id=$2
            ORDER BY agent_id
            "#,
        )
        .bind(user_id)
        .bind(parent_id)
        .fetch_all(&mut **tx)
        .await?
        .into_iter()
        .map(|row| ConversationParticipant {
            agent_id: row.get("agent_id"),
            role: row.get("role"),
        })
        .collect::<Vec<_>>();
        let first = messages.first();
        let expected_system_key = format!("continuation:{parent_id}");
        let identity_matches = parent.try_get::<String, _>("conversation_kind")? == expected_kind
            && parent.try_get::<Option<String>, _>("direct_key")? == header.direct_key
            && parent.try_get::<Option<String>, _>("subject")? == header.subject
            && parent_participants == header.participants;
        let marker_matches = first.is_some_and(|message| {
            message.seq == 1
                && message.kind == MessageKind::System
                && message.system_key.as_deref() == Some(expected_system_key.as_str())
                && message.body_md == messaging_protocol::CONTINUATION_SYSTEM_BODY
        });
        if parent.try_get::<String, _>("status")? != "closed"
            || parent.try_get::<i64, _>("last_seq")?
                != messaging_protocol::MAX_MESSAGES_PER_CONVERSATION
            || !identity_matches
            || !marker_matches
            || continuation_chain_contains_in_tx(tx, user_id, header.conversation_id, parent_id)
                .await?
        {
            return Err(ApiError::invalid(
                "conversation continuation must preserve its closed 500-message parent identity and canonical opening marker",
            ));
        }
    }

    let last_seq = messages.len() as i64;
    let last_message_at = messages.last().map(|message| message.created_at);
    let conversation_kind = match header.conversation_kind {
        ConversationKind::Direct => "direct",
        ConversationKind::Group => "group",
    };
    let status = match header.status {
        ConversationStatus::Open => "open",
        ConversationStatus::PausedForHuman => "paused_for_human",
        ConversationStatus::Closed => "closed",
    };
    sqlx::query(
        r#"
        INSERT INTO brunn.messaging_conversations (
          user_id,conversation_id,entry_id,path,conversation_kind,direct_key,
          subject,status,created_by_agent_id,last_seq,last_message_at,
          agent_streak,needs_human,continues_from,latest_sync_cursor,
          closed_at,created_at,updated_at
        ) VALUES (
          $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,
          clock_timestamp()
        )
        ON CONFLICT (user_id,conversation_id) DO UPDATE SET
          entry_id=EXCLUDED.entry_id,path=EXCLUDED.path,
          conversation_kind=EXCLUDED.conversation_kind,
          direct_key=EXCLUDED.direct_key,subject=EXCLUDED.subject,
          status=EXCLUDED.status,created_by_agent_id=EXCLUDED.created_by_agent_id,
          last_seq=EXCLUDED.last_seq,last_message_at=EXCLUDED.last_message_at,
          agent_streak=EXCLUDED.agent_streak,needs_human=EXCLUDED.needs_human,
          continues_from=EXCLUDED.continues_from,
          latest_sync_cursor=EXCLUDED.latest_sync_cursor,
          closed_at=EXCLUDED.closed_at,created_at=EXCLUDED.created_at,
          updated_at=clock_timestamp()
        "#,
    )
    .bind(user_id)
    .bind(header.conversation_id)
    .bind(entry_id)
    .bind(path)
    .bind(conversation_kind)
    .bind(header.direct_key.as_deref())
    .bind(header.subject.as_deref())
    .bind(status)
    .bind(&header.created_by_agent_id)
    .bind(last_seq)
    .bind(last_message_at)
    .bind(header.agent_streak)
    .bind(header.needs_human)
    .bind(header.continues_from)
    .bind(header.latest_sync_cursor)
    .bind(header.closed_at)
    .bind(header.created_at)
    .execute(&mut **tx)
    .await?;

    sqlx::query(
        "DELETE FROM brunn.messaging_message_index WHERE user_id=$1 AND conversation_id=$2",
    )
    .bind(user_id)
    .bind(header.conversation_id)
    .execute(&mut **tx)
    .await?;
    sqlx::query("DELETE FROM brunn.messaging_participants WHERE user_id=$1 AND conversation_id=$2")
        .bind(user_id)
        .bind(header.conversation_id)
        .execute(&mut **tx)
        .await?;

    for participant in &header.participants {
        sqlx::query(
            r#"
            INSERT INTO brunn.messaging_participants (
              user_id,conversation_id,agent_id,role,last_read_seq,joined_at,updated_at
            ) VALUES ($1,$2,$3,$4,0,$5,$5)
            "#,
        )
        .bind(user_id)
        .bind(header.conversation_id)
        .bind(&participant.agent_id)
        .bind(&participant.role)
        .bind(header.created_at)
        .execute(&mut **tx)
        .await?;
    }
    for message in &messages {
        if let Some(reply_conversation_id) = message.in_reply_to_conversation_id
            && reply_conversation_id != header.conversation_id
            && !continuation_chain_contains_in_tx(
                tx,
                user_id,
                reply_conversation_id,
                header.conversation_id,
            )
            .await?
        {
            return Err(ApiError::invalid(
                "canonical reply target must belong to this continuation chain",
            ));
        }
        sqlx::query(
            r#"
            INSERT INTO brunn.messaging_message_index (
              user_id,conversation_id,seq,message_id,from_agent_id,client_key,
              system_key,request_hash,kind,body_md,refs,
              in_reply_to_conversation_id,in_reply_to,correlation_id,
              expects_reply,reply_by,reply_by_handled_at,sync_cursor,created_at
            ) VALUES (
              $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19
            )
            "#,
        )
        .bind(user_id)
        .bind(header.conversation_id)
        .bind(message.seq)
        .bind(message.message_id)
        .bind(message.from_agent_id.as_deref())
        .bind(message.client_key.as_deref())
        .bind(message.system_key.as_deref())
        .bind(message.request_hash.as_deref())
        .bind(message_kind_str(message.kind))
        .bind(&message.body_md)
        .bind(serde_json::to_value(&message.refs)?)
        .bind(message.in_reply_to_conversation_id)
        .bind(message.in_reply_to)
        .bind(message.correlation_id.as_deref())
        .bind(message.expects_reply)
        .bind(message.reply_by)
        .bind(message.reply_by_handled_at)
        .bind(message.sync_cursor)
        .bind(message.created_at)
        .execute(&mut **tx)
        .await?;
    }
    sqlx::query(
        r#"
        INSERT INTO brunn.messaging_sync_state (user_id,current_cursor,updated_at)
        VALUES ($1,$2,clock_timestamp())
        ON CONFLICT (user_id) DO UPDATE SET
          current_cursor=GREATEST(
            brunn.messaging_sync_state.current_cursor,
            EXCLUDED.current_cursor
          ),
          updated_at=clock_timestamp()
        "#,
    )
    .bind(user_id)
    .bind(header.latest_sync_cursor)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn write_canonical_conversation_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    auth: &AuthContext,
    conversation_id: Uuid,
) -> ApiResult<i64> {
    let conversation =
        load_conversation_snapshot_in_tx(tx, auth.user_id.0, conversation_id).await?;
    let participants =
        load_canonical_participants_in_tx(tx, auth.user_id.0, conversation_id).await?;
    let messages = load_canonical_messages_in_tx(tx, auth.user_id.0, conversation_id).await?;
    let header = ConversationHeader {
        schema: "conversation.v1".to_owned(),
        conversation_id,
        conversation_kind: parse_conversation_kind(&conversation.conversation_kind)?,
        direct_key: conversation.direct_key.clone(),
        subject: conversation.subject.clone(),
        status: parse_conversation_status(&conversation.status)?,
        participants,
        created_by_agent_id: conversation.created_by_agent_id.clone(),
        continues_from: conversation.continues_from,
        agent_streak: conversation.agent_streak,
        needs_human: conversation.needs_human,
        latest_sync_cursor: conversation.latest_sync_cursor,
        created_at: conversation.created_at,
        closed_at: conversation.closed_at,
    };
    let content =
        messaging_protocol::render_conversation(&header, &messages).map_err(protocol_error)?;
    if content.len() > messaging_protocol::MAX_CANONICAL_CONVERSATION_BYTES {
        return Err(ApiError::public(
            StatusCode::PAYLOAD_TOO_LARGE,
            "conversation_entry_too_large",
            "canonical conversation Markdown is limited to 12 MiB",
        ));
    }
    let metadata = messaging_protocol::conversation_metadata(&header);
    upsert_canonical_entry_in_tx(tx, auth, &conversation, content, metadata).await
}

async fn load_conversation_snapshot_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    conversation_id: Uuid,
) -> ApiResult<ConversationRow> {
    let row = sqlx::query(
        r#"
        SELECT conversation_id,entry_id,conversation_kind,direct_key,subject,status,
               created_by_agent_id,last_seq,agent_streak,
               needs_human,continues_from,latest_sync_cursor,closed_at,created_at
        FROM brunn.messaging_conversations
        WHERE user_id=$1 AND conversation_id=$2
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .fetch_optional(&mut **tx)
    .await?
    .ok_or_else(|| messaging_not_found(&conversation_id.to_string()))?;
    Ok(ConversationRow {
        conversation_id: row.try_get("conversation_id")?,
        entry_id: row.try_get("entry_id")?,
        conversation_kind: row.try_get("conversation_kind")?,
        direct_key: row.try_get("direct_key")?,
        subject: row.try_get("subject")?,
        status: row.try_get("status")?,
        created_by_agent_id: row.try_get("created_by_agent_id")?,
        last_seq: row.try_get("last_seq")?,
        agent_streak: row.try_get("agent_streak")?,
        needs_human: row.try_get("needs_human")?,
        continues_from: row.try_get("continues_from")?,
        latest_sync_cursor: row.try_get("latest_sync_cursor")?,
        closed_at: row.try_get("closed_at")?,
        created_at: row.try_get("created_at")?,
    })
}

async fn load_canonical_participants_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    conversation_id: Uuid,
) -> ApiResult<Vec<ConversationParticipant>> {
    let rows = sqlx::query(
        r#"
        SELECT agent_id,role
        FROM brunn.messaging_participants
        WHERE user_id=$1 AND conversation_id=$2
        ORDER BY agent_id
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .fetch_all(&mut **tx)
    .await?;
    rows.into_iter()
        .map(|row| {
            Ok(ConversationParticipant {
                agent_id: row.try_get("agent_id")?,
                role: row.try_get("role")?,
            })
        })
        .collect::<Result<Vec<_>, sqlx::Error>>()
        .map_err(Into::into)
}

async fn load_canonical_messages_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    user_id: Uuid,
    conversation_id: Uuid,
) -> ApiResult<Vec<CanonicalMessage>> {
    let rows = sqlx::query(
        r#"
        SELECT seq,message_id,from_agent_id,client_key,system_key,request_hash,
               kind,body_md,refs,in_reply_to_conversation_id,in_reply_to,
               correlation_id,expects_reply,
               reply_by,reply_by_handled_at,sync_cursor,created_at
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND conversation_id=$2
        ORDER BY seq
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .fetch_all(&mut **tx)
    .await?;
    rows.into_iter()
        .map(|row| {
            Ok(CanonicalMessage {
                seq: row.try_get("seq")?,
                message_id: row.try_get("message_id")?,
                from_agent_id: row.try_get("from_agent_id")?,
                client_key: row.try_get("client_key")?,
                system_key: row.try_get("system_key")?,
                request_hash: row.try_get("request_hash")?,
                kind: parse_message_kind(&row.try_get::<String, _>("kind")?)?,
                body_md: row.try_get("body_md")?,
                refs: serde_json::from_value(row.try_get("refs")?)?,
                in_reply_to_conversation_id: row.try_get("in_reply_to_conversation_id")?,
                in_reply_to: row.try_get("in_reply_to")?,
                correlation_id: row.try_get("correlation_id")?,
                expects_reply: row.try_get("expects_reply")?,
                reply_by: row.try_get("reply_by")?,
                reply_by_handled_at: row.try_get("reply_by_handled_at")?,
                sync_cursor: row.try_get("sync_cursor")?,
                created_at: row.try_get("created_at")?,
            })
        })
        .collect()
}

fn parse_conversation_kind(value: &str) -> ApiResult<ConversationKind> {
    match value {
        "direct" => Ok(ConversationKind::Direct),
        "group" => Ok(ConversationKind::Group),
        _ => Err(ApiError::Internal(
            "stored messaging conversation kind is invalid".to_owned(),
        )),
    }
}

fn parse_conversation_status(value: &str) -> ApiResult<ConversationStatus> {
    match value {
        "open" => Ok(ConversationStatus::Open),
        "paused_for_human" => Ok(ConversationStatus::PausedForHuman),
        "closed" => Ok(ConversationStatus::Closed),
        _ => Err(ApiError::Internal(
            "stored messaging conversation status is invalid".to_owned(),
        )),
    }
}

fn parse_message_kind(value: &str) -> ApiResult<MessageKind> {
    match value {
        "text" => Ok(MessageKind::Text),
        "question" => Ok(MessageKind::Question),
        "system" => Ok(MessageKind::System),
        _ => Err(ApiError::Internal(
            "stored messaging message kind is invalid".to_owned(),
        )),
    }
}

async fn upsert_canonical_entry_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    auth: &AuthContext,
    conversation: &ConversationRow,
    content: String,
    metadata: Value,
) -> ApiResult<i64> {
    let path = messaging_protocol::conversation_path(conversation.conversation_id);
    let title = conversation
        .subject
        .clone()
        .unwrap_or_else(|| format!("Conversation {}", conversation.conversation_id));
    let existing = sqlx::query(
        r#"
        SELECT id,kind,current_version,deleted_at
        FROM brunn.entries
        WHERE user_id=$1 AND lower(normalize(path,NFC))=lower(normalize($2,NFC))
        FOR UPDATE
        "#,
    )
    .bind(auth.user_id.0)
    .bind(&path)
    .fetch_optional(&mut **tx)
    .await?;
    let (version, operation) = if let Some(row) = existing {
        let entry_id: Uuid = row.try_get("id")?;
        let kind: String = row.try_get("kind")?;
        let deleted_at: Option<DateTime<Utc>> = row.try_get("deleted_at")?;
        if entry_id != conversation.entry_id || kind != "markdown" || deleted_at.is_some() {
            return Err(ApiError::conflict(
                "conversation_entry_conflict",
                "the canonical conversation path is occupied by a different entry",
                json!({}),
            ));
        }
        (row.try_get::<i64, _>("current_version")? + 1, "update")
    } else {
        sqlx::query(
            r#"
            INSERT INTO brunn.entries (
              id,user_id,path,title,kind,media_type,current_version
            ) VALUES ($1,$2,$3,$4,'markdown','text/markdown',0)
            "#,
        )
        .bind(conversation.entry_id)
        .bind(auth.user_id.0)
        .bind(&path)
        .bind(&title)
        .execute(&mut **tx)
        .await?;
        (1, "create")
    };
    let content_sha256 = hex::encode(Sha256::digest(content.as_bytes()));
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
    .bind(auth.user_id.0)
    .bind(conversation.entry_id)
    .bind(version)
    .bind(&content_sha256)
    .bind(&content)
    .bind(i64::try_from(content.len()).unwrap_or(i64::MAX))
    .bind(metadata)
    .bind(auth.credential_id.0)
    .execute(&mut **tx)
    .await?;
    sqlx::query(
        r#"
        UPDATE brunn.entries
        SET title=$3,media_type='text/markdown',current_version=$4,
            deleted_at=NULL,updated_at=clock_timestamp()
        WHERE user_id=$1 AND id=$2
        "#,
    )
    .bind(auth.user_id.0)
    .bind(conversation.entry_id)
    .bind(title)
    .bind(version)
    .execute(&mut **tx)
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
    .bind(conversation.entry_id)
    .bind(version)
    .bind(operation)
    .bind(path)
    .bind(content_sha256)
    .fetch_one(&mut **tx)
    .await?;
    Ok(generation)
}

async fn publish_conversation_notification_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    state: &AppState,
    auth: &AuthContext,
    conversation_id: Uuid,
    seq: i64,
    event_type: &str,
    event_key_override: Option<&str>,
    occurred_at: DateTime<Utc>,
) -> ApiResult<()> {
    let derived_event_key = match event_type {
        "message" => format!("message:{conversation_id}:{seq}"),
        "needs-human" => format!("needs-human:{conversation_id}:{seq}"),
        "reply-by" => format!("reply-by:{conversation_id}:{seq}"),
        "system" => format!("message-system:{conversation_id}:{seq}"),
        _ => {
            return Err(ApiError::Internal(
                "unknown internal messaging notification event".to_owned(),
            ));
        }
    };
    let event_key = event_key_override.unwrap_or(&derived_event_key);
    let (title, body, importance) = if matches!(event_type, "needs-human" | "reply-by") {
        (
            "Agent reply needed",
            "Open Brunn to continue an agent conversation.",
            "important",
        )
    } else {
        (
            "New agent message",
            "Open Brunn to view the conversation.",
            "normal",
        )
    };
    let settings = sqlx::query(
        r#"
        SELECT timezone,quiet_hours_start,quiet_hours_end
        FROM brunn.task_settings
        WHERE user_id=$1
        "#,
    )
    .bind(auth.user_id.0)
    .fetch_optional(&mut **tx)
    .await?;
    let Some(settings) = settings else {
        return Err(ApiError::Internal(
            "messaging notification settings are missing".to_owned(),
        ));
    };
    let timezone = settings
        .try_get::<String, _>("timezone")?
        .parse::<Tz>()
        .map_err(|_| ApiError::Internal("messaging notification timezone is invalid".to_owned()))?;
    let available_at = task_guard::delivery_available_at_without_override(
        occurred_at,
        timezone,
        settings.try_get("quiet_hours_start")?,
        settings.try_get("quiet_hours_end")?,
    )?;
    let request = PublishRequest {
        event_key: event_key.to_owned(),
        correlation_id: event_key.to_owned(),
        kind: "operational".to_owned(),
        importance: importance.to_owned(),
        title: title.to_owned(),
        body: body.to_owned(),
        source: None,
        target: NotificationTarget::Conversation {
            conversation_id: conversation_id.to_string(),
            seq,
        },
        occurred_at: Some(occurred_at),
        expires_at: None,
    };
    let result = notification_service::publish_in_tx(
        tx,
        state,
        auth,
        &request,
        PublishAccess::InternalMessaging,
        Some(available_at),
    )
    .await?;
    metrics::counter!(
        "messaging.notification.publish",
        "event" => event_type.to_owned(),
        "result" => if result.inserted { "created" } else { "duplicate" }
    )
    .increment(1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn direct_keys_are_ordered_and_separator_safe() {
        let mut participants = vec!["nyx.echo".to_owned(), "owner".to_owned()];
        participants.sort();
        assert_eq!(direct_key(&participants), "nyx.echo|owner");
        assert!(!participants.iter().any(|value| value.contains('|')));
    }

    #[test]
    fn conversation_and_workspace_entry_ids_are_distinct() {
        let conversation_id = Uuid::now_v7();
        assert_ne!(new_local_entry_id(conversation_id), conversation_id);
    }

    #[test]
    fn rollover_shape_keeps_every_entry_at_five_hundred_messages() {
        assert_eq!(
            rollover_plan(498, true).expect("pause at boundary"),
            RolloverPlan {
                user_seq: 499,
                pause_system_in_current: true,
                rollover: true,
                pause_system_in_continuation: false,
            }
        );
        assert_eq!(
            rollover_plan(499, true).expect("user message at boundary"),
            RolloverPlan {
                user_seq: 500,
                pause_system_in_current: false,
                rollover: true,
                pause_system_in_continuation: true,
            }
        );
        assert_eq!(
            rollover_plan(499, false).expect("ordinary boundary"),
            RolloverPlan {
                user_seq: 500,
                pause_system_in_current: false,
                rollover: true,
                pause_system_in_continuation: false,
            }
        );
        assert!(rollover_plan(500, false).is_err());
    }

    #[test]
    fn claimed_sender_is_rejected_by_strict_send_deserialization() {
        let parsed = serde_json::from_value::<SendMessageInput>(json!({
            "client_key": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "body_md": "hello",
            "from": "owner"
        }));
        assert!(parsed.is_err());
    }

    #[test]
    fn sync_query_rejects_ambiguous_or_unbounded_positions() {
        let valid = SyncQuery {
            cursor: 0,
            wait: 25,
            conversation_id: Some(Uuid::nil()),
            after_seq: Some(0),
            limit: Some(200),
        };
        assert_eq!(validate_sync_query(&valid).expect("valid query"), 200);
        assert!(
            validate_sync_query(&SyncQuery {
                cursor: 1,
                ..valid.clone()
            })
            .is_err()
        );
        assert!(
            validate_sync_query(&SyncQuery {
                wait: 26,
                ..valid.clone()
            })
            .is_err()
        );
        assert!(
            validate_sync_query(&SyncQuery {
                conversation_id: None,
                after_seq: Some(1),
                ..valid
            })
            .is_err()
        );
    }

    #[test]
    fn retry_after_rounds_up_and_never_returns_zero() {
        let as_of = DateTime::parse_from_rfc3339("2026-08-27T08:00:00Z")
            .expect("timestamp")
            .with_timezone(&Utc);
        assert_eq!(
            retry_after_seconds(as_of + chrono::Duration::milliseconds(1), as_of),
            1
        );
        assert_eq!(
            retry_after_seconds(as_of + chrono::Duration::milliseconds(1_001), as_of),
            2
        );
        assert_eq!(retry_after_seconds(as_of, as_of), 1);
    }

    #[test]
    fn registry_mutations_distinguish_bearer_from_authenticated_session() {
        let session_headers = HeaderMap::new();
        assert!(is_web_session_request(&session_headers));
        let mut bearer_headers = HeaderMap::new();
        bearer_headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer redacted"));
        assert!(!is_web_session_request(&bearer_headers));
    }
}
