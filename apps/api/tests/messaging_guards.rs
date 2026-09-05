use axum::{
    Router,
    body::Body,
    http::{Method, Request, StatusCode, header},
};
use chrono::{DateTime, Duration as ChronoDuration, SubsecRound, Utc};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use sqlx::{PgPool, Row, postgres::PgPoolOptions};
use tower::ServiceExt;
use uuid::Uuid;

use brunn::{
    AppState, Config,
    auth::hash_token,
    messaging_protocol::{self, MessageKind, SendMessageInput},
    messaging_service, router,
};

const MESSAGING_ROOT: &str = "/v1/workspace/messaging";

#[derive(Debug)]
struct HttpResponse {
    status: StatusCode,
    body: Value,
}

struct CredentialFixture {
    id: Uuid,
    token: String,
}

struct WorkspaceFixture {
    user_id: Uuid,
    owner: CredentialFixture,
    agent: CredentialFixture,
    agent_b: CredentialFixture,
}

async fn connect_test_state() -> Option<(PgPool, AppState)> {
    let Some(database_url) = std::env::var("BRUNN_TEST_DATABASE_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        eprintln!("BRUNN_TEST_DATABASE_URL is unset; skipping messaging guard contract");
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
    config.database_url_rw = database_url.clone();
    config.database_url_ro = database_url.clone();
    config.database_url_admin = Some(database_url);
    config.database_max_connections = 4;
    config.apns_delivery_enabled = false;
    config.messaging_enabled = true;
    let state = AppState::connect(config)
        .await
        .expect("connect disposable API state");
    Some((seed_pool, state))
}

async fn insert_credential(
    pool: &PgPool,
    user_id: Uuid,
    scope_id: Uuid,
    label: &str,
) -> CredentialFixture {
    let id = Uuid::now_v7();
    let token = format!("messaging-guard-test-{}", Uuid::now_v7());
    sqlx::query(
        r#"
        INSERT INTO brunn.api_credentials (
          id,user_id,label,token_hash,capabilities
        ) VALUES ($1,$2,$3,$4,$5)
        "#,
    )
    .bind(id)
    .bind(user_id)
    .bind(label)
    .bind(hash_token(&token))
    .bind(vec!["message.read", "message.write"])
    .execute(pool)
    .await
    .expect("insert narrow messaging credential");
    sqlx::query(
        r#"
        INSERT INTO brunn.credential_scope_grants (
          credential_id,user_id,scope_id
        ) VALUES ($1,$2,$3)
        "#,
    )
    .bind(id)
    .bind(user_id)
    .bind(scope_id)
    .execute(pool)
    .await
    .expect("grant messaging test scope");
    CredentialFixture { id, token }
}

async fn insert_agent(
    pool: &PgPool,
    user_id: Uuid,
    creator_id: Uuid,
    agent_id: &str,
    principal_kind: &str,
) {
    sqlx::query(
        r#"
        INSERT INTO brunn.messaging_agents (
          user_id,agent_id,display_name,principal_kind,delivery_mode,
          created_by_credential_id
        ) VALUES ($1,$2,$3,$4,'pull',$5)
        "#,
    )
    .bind(user_id)
    .bind(agent_id)
    .bind(format!("Guard {agent_id}"))
    .bind(principal_kind)
    .bind(creator_id)
    .execute(pool)
    .await
    .expect("insert messaging guard principal");
}

async fn bind_credential(
    pool: &PgPool,
    user_id: Uuid,
    credential_id: Uuid,
    agent_id: &str,
    owner_id: Uuid,
) {
    sqlx::query(
        r#"
        INSERT INTO brunn.messaging_credential_bindings (
          user_id,credential_id,agent_id,bound_by_credential_id
        ) VALUES ($1,$2,$3,$4)
        "#,
    )
    .bind(user_id)
    .bind(credential_id)
    .bind(agent_id)
    .bind(owner_id)
    .execute(pool)
    .await
    .expect("bind messaging guard credential");
}

async fn seed_workspace(pool: &PgPool, label: &str) -> WorkspaceFixture {
    let user_id = Uuid::now_v7();
    let scope_id = Uuid::now_v7();
    sqlx::query("INSERT INTO brunn.users (id,external_ref,display_name) VALUES ($1,$2,$3)")
        .bind(user_id)
        .bind(format!("messaging-guard-test:{label}:{user_id}"))
        .bind(format!("Messaging guard {label}"))
        .execute(pool)
        .await
        .expect("insert messaging guard user");
    sqlx::query("INSERT INTO brunn.scopes (id,user_id,scope_ref,name) VALUES ($1,$2,$3,$4)")
        .bind(scope_id)
        .bind(user_id)
        .bind(format!("scope:messaging-guard-{scope_id}"))
        .bind(format!("Messaging guard {label}"))
        .execute(pool)
        .await
        .expect("insert messaging guard scope");

    let owner = insert_credential(pool, user_id, scope_id, &format!("{label} owner")).await;
    let agent = insert_credential(pool, user_id, scope_id, &format!("{label} agent-a")).await;
    let agent_b = insert_credential(pool, user_id, scope_id, &format!("{label} agent-b")).await;
    insert_agent(pool, user_id, owner.id, "owner", "owner").await;
    for agent_id in ["agent-a", "agent-b", "agent-c", "agent-d"] {
        insert_agent(pool, user_id, owner.id, agent_id, "resident").await;
    }
    bind_credential(pool, user_id, owner.id, "owner", owner.id).await;
    bind_credential(pool, user_id, agent.id, "agent-a", owner.id).await;
    bind_credential(pool, user_id, agent_b.id, "agent-b", owner.id).await;
    WorkspaceFixture {
        user_id,
        owner,
        agent,
        agent_b,
    }
}

async fn request_json(
    app: &Router,
    method: Method,
    uri: &str,
    token: &str,
    body: Value,
) -> HttpResponse {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::to_vec(&body).expect("serialize guard request"),
        ))
        .expect("build guard request");
    let response = app
        .clone()
        .oneshot(request)
        .await
        .expect("serve messaging guard request");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect messaging guard response")
        .to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    HttpResponse { status, body }
}

fn data(response: &HttpResponse) -> &Value {
    response
        .body
        .get("data")
        .expect("successful messaging response has data")
}

fn assert_error(response: &HttpResponse, status: StatusCode, code: &str) {
    assert_eq!(response.status, status, "unexpected guard response status");
    assert_eq!(
        response.body.pointer("/error/code").and_then(Value::as_str),
        Some(code),
        "unexpected guard error code"
    );
}

fn client_key(number: i64) -> String {
    format!("{number:026}")
}

fn text_send(number: i64, body_md: &str) -> Value {
    json!({
        "client_key": client_key(number),
        "kind": "text",
        "body_md": body_md
    })
}

async fn create_conversation(
    app: &Router,
    token: &str,
    participants: &[&str],
    subject: &str,
) -> Uuid {
    let response = request_json(
        app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations"),
        token,
        json!({"participants": participants, "subject": subject}),
    )
    .await;
    assert_eq!(response.status, StatusCode::OK, "create guard fixture");
    data(&response)
        .get("conversation_id")
        .and_then(Value::as_str)
        .and_then(|value| Uuid::parse_str(value).ok())
        .expect("conversation response contains a UUID")
}

#[allow(clippy::too_many_arguments)]
async fn seed_messages(
    pool: &PgPool,
    user_id: Uuid,
    conversation_id: Uuid,
    start_seq: i64,
    count: i64,
    senders: &[&str],
    created_at: DateTime<Utc>,
    agent_streak: i32,
) {
    assert!(count > 0);
    assert!(!senders.is_empty());
    let mut tx = pool.begin().await.expect("begin message seed");
    let base_cursor = sqlx::query_scalar::<_, i64>(
        "SELECT current_cursor FROM brunn.messaging_sync_state WHERE user_id=$1 FOR UPDATE",
    )
    .bind(user_id)
    .fetch_one(&mut *tx)
    .await
    .expect("lock messaging cursor");
    let senders = senders
        .iter()
        .map(|sender| (*sender).to_owned())
        .collect::<Vec<_>>();
    let request_hashes = (0..count)
        .map(|offset| {
            let input = SendMessageInput {
                client_key: client_key(start_seq + offset),
                kind: MessageKind::Text,
                body_md: "seed message".to_owned(),
                refs: Vec::new(),
                in_reply_to: None,
                correlation_id: None,
                expects_reply: false,
                reply_by: None,
            };
            messaging_protocol::request_hash(conversation_id, &input)
        })
        .collect::<Vec<_>>();
    sqlx::query(
        r#"
        INSERT INTO brunn.messaging_message_index (
          user_id,conversation_id,seq,message_id,from_agent_id,client_key,
          system_key,request_hash,kind,body_md,refs,in_reply_to,
          correlation_id,expects_reply,reply_by,reply_by_handled_at,
          sync_cursor,created_at
        )
        SELECT
          $1,$2,$3 + seed.position_index,gen_random_uuid(),
          $5[((seed.position_index % cardinality($5)) + 1)::integer],
          lpad(($3 + seed.position_index)::text,26,'0'),
          NULL,$8[(seed.position_index + 1)::integer],'text','seed message','[]'::jsonb,
          NULL,NULL,false,NULL,NULL,$6 + seed.position_index + 1,$7
        FROM generate_series(0::bigint,$4::bigint - 1) AS seed(position_index)
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(start_seq)
    .bind(count)
    .bind(senders)
    .bind(base_cursor)
    .bind(created_at)
    .bind(request_hashes)
    .execute(&mut *tx)
    .await
    .expect("seed indexed conversation messages");
    let final_seq = start_seq + count - 1;
    let final_cursor = base_cursor + count;
    sqlx::query(
        r#"
        UPDATE brunn.messaging_sync_state
        SET current_cursor=$2,updated_at=clock_timestamp()
        WHERE user_id=$1
        "#,
    )
    .bind(user_id)
    .bind(final_cursor)
    .execute(&mut *tx)
    .await
    .expect("advance seeded messaging cursor");
    sqlx::query(
        r#"
        UPDATE brunn.messaging_conversations
        SET last_seq=$3,last_message_at=$4,agent_streak=$5,
            latest_sync_cursor=$6,created_at=LEAST(created_at,$4),
            updated_at=clock_timestamp()
        WHERE user_id=$1 AND conversation_id=$2
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(final_seq)
    .bind(created_at)
    .bind(agent_streak)
    .bind(final_cursor)
    .execute(&mut *tx)
    .await
    .expect("advance seeded conversation projection");
    tx.commit().await.expect("commit message seed");
}

fn assert_typed_rate(response: &HttpResponse, code: &str, maximum_retry: i64) {
    assert_error(response, StatusCode::TOO_MANY_REQUESTS, code);
    let retry_after = response
        .body
        .pointer("/error/details/retry_after_seconds")
        .and_then(Value::as_i64)
        .expect("rate error contains retry_after_seconds");
    assert!(
        (1..=maximum_retry).contains(&retry_after),
        "retry metadata must be positive and bounded"
    );
}

#[tokio::test]
async fn messaging_guards_preserve_replay_budgets_rollover_and_reply_deadlines() {
    let Some((pool, state)) = connect_test_state().await else {
        return;
    };
    let app = router(state.clone());

    let notification_conflict = seed_workspace(&pool, "notification-conflict").await;
    let notification_conflict_conversation = create_conversation(
        &app,
        &notification_conflict.agent.token,
        &["agent-b", "owner"],
        "Notification event-key conflict",
    )
    .await;
    let conflict_event_key = format!("message:{notification_conflict_conversation}:1");
    sqlx::query(
        r#"
        INSERT INTO brunn.notifications (
          id,user_id,producer_credential_id,event_key,request_hash,
          correlation_id,kind,importance,title,body,source,target,
          occurred_at,expires_at
        ) VALUES (
          $1,$2,$3,$4,$5,$4,'operational','normal','New agent message',
          'Open Brunn to view the conversation.',NULL,$6,
          clock_timestamp(),clock_timestamp()+interval '24 hours'
        )
        "#,
    )
    .bind(Uuid::now_v7())
    .bind(notification_conflict.user_id)
    .bind(notification_conflict.agent.id)
    .bind(&conflict_event_key)
    .bind("f".repeat(64))
    .bind(json!({
        "type": "conversation",
        "conversation_id": notification_conflict_conversation,
        "seq": 1
    }))
    .execute(&pool)
    .await
    .expect("seed conflicting notification event key");
    let conflicted_send = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{notification_conflict_conversation}/messages"),
        &notification_conflict.agent.token,
        text_send(900, "this send must fail atomically"),
    )
    .await;
    assert_error(
        &conflicted_send,
        StatusCode::CONFLICT,
        "notification_event_key_conflict",
    );
    let conflicted_message_count = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)::bigint
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND conversation_id=$2
        "#,
    )
    .bind(notification_conflict.user_id)
    .bind(notification_conflict_conversation)
    .fetch_one(&pool)
    .await
    .expect("count rolled-back conflicted send");
    assert_eq!(conflicted_message_count, 0);

    let sender_rate = seed_workspace(&pool, "sender-rate").await;
    let sender_rate_conversation = create_conversation(
        &app,
        &sender_rate.agent.token,
        &["agent-b"],
        "Replay precedes sender rate",
    )
    .await;
    let sender_rate_path =
        format!("{MESSAGING_ROOT}/conversations/{sender_rate_conversation}/messages");
    let original_body = text_send(900, "original idempotent send");
    let original = request_json(
        &app,
        Method::POST,
        &sender_rate_path,
        &sender_rate.agent.token,
        original_body.clone(),
    )
    .await;
    assert_eq!(original.status, StatusCode::OK);
    let original_message_id = data(&original)
        .pointer("/message/message_id")
        .and_then(Value::as_str)
        .expect("original send has message id")
        .to_owned();
    seed_messages(
        &pool,
        sender_rate.user_id,
        sender_rate_conversation,
        2,
        59,
        &["agent-a"],
        Utc::now(),
        0,
    )
    .await;
    let replay = request_json(
        &app,
        Method::POST,
        &sender_rate_path,
        &sender_rate.agent.token,
        original_body,
    )
    .await;
    assert_eq!(replay.status, StatusCode::OK);
    assert_eq!(
        data(&replay).get("duplicate").and_then(Value::as_bool),
        Some(true),
        "an exact replay bypasses the saturated sender guard"
    );
    assert_eq!(
        data(&replay)
            .pointer("/message/message_id")
            .and_then(Value::as_str),
        Some(original_message_id.as_str())
    );
    let sender_limited = request_json(
        &app,
        Method::POST,
        &sender_rate_path,
        &sender_rate.agent.token,
        text_send(901, "new logical send must be limited"),
    )
    .await;
    assert_typed_rate(&sender_limited, "sender_rate_limited", 60);

    let conversation_rate = seed_workspace(&pool, "conversation-rate").await;
    let conversation_rate_id = create_conversation(
        &app,
        &conversation_rate.agent.token,
        &["agent-b", "agent-c", "agent-d"],
        "Conversation hourly rate",
    )
    .await;
    seed_messages(
        &pool,
        conversation_rate.user_id,
        conversation_rate_id,
        1,
        200,
        &["agent-a", "agent-b", "agent-c", "owner"],
        Utc::now(),
        0,
    )
    .await;
    let conversation_limited = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{conversation_rate_id}/messages"),
        &conversation_rate.agent.token,
        text_send(901, "new conversation send must be limited"),
    )
    .await;
    assert_typed_rate(&conversation_limited, "conversation_rate_limited", 3_600);

    let streak = seed_workspace(&pool, "agent-streak").await;
    let streak_conversation = create_conversation(
        &app,
        &streak.agent.token,
        &["agent-b"],
        "Exact twentieth message",
    )
    .await;
    seed_messages(
        &pool,
        streak.user_id,
        streak_conversation,
        1,
        19,
        &["agent-a"],
        Utc::now(),
        19,
    )
    .await;
    let twentieth = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{streak_conversation}/messages"),
        &streak.agent.token,
        text_send(900, "twentieth consecutive agent message"),
    )
    .await;
    assert_eq!(
        twentieth.status,
        StatusCode::OK,
        "the twentieth message commits before pausing"
    );
    assert_eq!(
        data(&twentieth).get("seq").and_then(Value::as_i64),
        Some(20)
    );
    let streak_row = sqlx::query(
        r#"
        SELECT status,agent_streak,needs_human,last_seq
        FROM brunn.messaging_conversations
        WHERE user_id=$1 AND conversation_id=$2
        "#,
    )
    .bind(streak.user_id)
    .bind(streak_conversation)
    .fetch_one(&pool)
    .await
    .expect("read exact streak state");
    assert_eq!(streak_row.get::<String, _>("status"), "paused_for_human");
    assert_eq!(streak_row.get::<i32, _>("agent_streak"), 20);
    assert!(streak_row.get::<bool, _>("needs_human"));
    assert_eq!(streak_row.get::<i64, _>("last_seq"), 21);
    let new_message_shape = sqlx::query(
        r#"
        SELECT count(*)::bigint AS total,
               count(*) FILTER (WHERE kind='system')::bigint AS systems
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND conversation_id=$2 AND seq>=20
        "#,
    )
    .bind(streak.user_id)
    .bind(streak_conversation)
    .fetch_one(&pool)
    .await
    .expect("read twentieth-message shape");
    assert_eq!(new_message_shape.get::<i64, _>("total"), 2);
    assert_eq!(new_message_shape.get::<i64, _>("systems"), 1);
    let needs_human_notifications = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)::bigint
        FROM brunn.notifications
        WHERE user_id=$1 AND event_key LIKE $2
        "#,
    )
    .bind(streak.user_id)
    .bind(format!("needs-human:{streak_conversation}:%"))
    .fetch_one(&pool)
    .await
    .expect("count needs-human notifications");
    let all_streak_notifications = sqlx::query_scalar::<_, i64>(
        "SELECT count(*)::bigint FROM brunn.notifications WHERE user_id=$1",
    )
    .bind(streak.user_id)
    .fetch_one(&pool)
    .await
    .expect("count streak notifications");
    assert_eq!(needs_human_notifications, 1);
    assert_eq!(
        all_streak_notifications, 1,
        "observer-only owner gets the one needs-human event, not the ordinary message"
    );

    let paused_agent = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{streak_conversation}/messages"),
        &streak.agent.token,
        text_send(902, "must wait for an owner"),
    )
    .await;
    assert_error(&paused_agent, StatusCode::CONFLICT, "conversation_paused");

    let resumed = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{streak_conversation}/resume"),
        &streak.owner.token,
        json!({}),
    )
    .await;
    assert_eq!(resumed.status, StatusCode::OK);
    assert_eq!(
        data(&resumed).get("status").and_then(Value::as_str),
        Some("open")
    );
    let resumed_state = sqlx::query_as::<_, (String, i32, bool)>(
        r#"
        SELECT status,agent_streak,needs_human
        FROM brunn.messaging_conversations
        WHERE user_id=$1 AND conversation_id=$2
        "#,
    )
    .bind(streak.user_id)
    .bind(streak_conversation)
    .fetch_one(&pool)
    .await
    .expect("read resumed state");
    assert_eq!(resumed_state, ("open".to_owned(), 0, false));

    sqlx::query(
        r#"
        UPDATE brunn.messaging_conversations
        SET status='paused_for_human',agent_streak=20,needs_human=true
        WHERE user_id=$1 AND conversation_id=$2
        "#,
    )
    .bind(streak.user_id)
    .bind(streak_conversation)
    .execute(&pool)
    .await
    .expect("restore paused state for owner-post contract");
    let owner_role_before_post = sqlx::query_scalar::<_, String>(
        r#"
        SELECT role
        FROM brunn.messaging_participants
        WHERE user_id=$1 AND conversation_id=$2 AND agent_id='owner'
        "#,
    )
    .bind(streak.user_id)
    .bind(streak_conversation)
    .fetch_one(&pool)
    .await
    .expect("read owner observer role");
    assert_eq!(owner_role_before_post, "observer");
    let owner_post = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{streak_conversation}/messages"),
        &streak.owner.token,
        text_send(901, "owner clears the pause"),
    )
    .await;
    assert_eq!(owner_post.status, StatusCode::OK);
    let owner_cleared = sqlx::query_as::<_, (String, i32, bool)>(
        r#"
        SELECT status,agent_streak,needs_human
        FROM brunn.messaging_conversations
        WHERE user_id=$1 AND conversation_id=$2
        "#,
    )
    .bind(streak.user_id)
    .bind(streak_conversation)
    .fetch_one(&pool)
    .await
    .expect("read owner-cleared state");
    assert_eq!(owner_cleared, ("open".to_owned(), 0, false));
    let owner_role_after_post = sqlx::query_scalar::<_, String>(
        r#"
        SELECT role
        FROM brunn.messaging_participants
        WHERE user_id=$1 AND conversation_id=$2 AND agent_id='owner'
        "#,
    )
    .bind(streak.user_id)
    .bind(streak_conversation)
    .fetch_one(&pool)
    .await
    .expect("read promoted owner role");
    assert_eq!(
        owner_role_after_post, "participant",
        "an owner post promotes an observer for subsequent delivery"
    );
    let owner_post_shape = sqlx::query_as::<_, (String, Option<String>)>(
        r#"
        SELECT conversation_kind,direct_key
        FROM brunn.messaging_conversations
        WHERE user_id=$1 AND conversation_id=$2
        "#,
    )
    .bind(streak.user_id)
    .bind(streak_conversation)
    .fetch_one(&pool)
    .await
    .expect("read owner-promoted conversation shape");
    assert_eq!(
        owner_post_shape,
        ("group".to_owned(), None),
        "promoting the owner makes a two-agent direct a canonical group"
    );

    let pause_rollover = seed_workspace(&pool, "pause-rollover").await;
    let pause_rollover_conversation = create_conversation(
        &app,
        &pause_rollover.agent.token,
        &["agent-b"],
        "Twentieth message at rollover",
    )
    .await;
    seed_messages(
        &pool,
        pause_rollover.user_id,
        pause_rollover_conversation,
        1,
        498,
        &["agent-a"],
        Utc::now() - ChronoDuration::hours(2),
        19,
    )
    .await;
    let pause_at_rollover = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{pause_rollover_conversation}/messages"),
        &pause_rollover.agent.token,
        text_send(900, "twentieth message fills the source entry"),
    )
    .await;
    assert_eq!(pause_at_rollover.status, StatusCode::OK);
    assert_eq!(
        data(&pause_at_rollover).get("seq").and_then(Value::as_i64),
        Some(499)
    );
    let pause_continuation = data(&pause_at_rollover)
        .get("continuation_id")
        .and_then(Value::as_str)
        .and_then(|value| Uuid::parse_str(value).ok())
        .expect("the source entry rolls after its pause system message");
    let pause_events = sqlx::query_scalar::<_, Vec<String>>(
        r#"
        SELECT array_agg(event_key ORDER BY event_key)
        FROM brunn.notifications
        WHERE user_id=$1
        "#,
    )
    .bind(pause_rollover.user_id)
    .fetch_one(&pool)
    .await
    .expect("read rollover pause notifications");
    assert_eq!(
        pause_events,
        vec![
            format!("message-system:{pause_continuation}:1"),
            format!("needs-human:{pause_rollover_conversation}:500"),
        ],
        "the attention alert targets the actual pause record, not the continuation marker"
    );

    let rollover = seed_workspace(&pool, "rollover").await;
    let rollover_conversation = create_conversation(
        &app,
        &rollover.agent.token,
        &["agent-b"],
        "Five hundred message rollover",
    )
    .await;
    seed_messages(
        &pool,
        rollover.user_id,
        rollover_conversation,
        1,
        499,
        &["agent-a", "owner"],
        Utc::now() - ChronoDuration::hours(2),
        1,
    )
    .await;
    let rollover_path = format!("{MESSAGING_ROOT}/conversations/{rollover_conversation}/messages");
    let mut sender_lock = pool.begin().await.expect("hold first rollover sender");
    let blocker_pid = sqlx::query_scalar::<_, i32>("SELECT pg_backend_pid()")
        .fetch_one(&mut *sender_lock)
        .await
        .expect("read sender lock backend");
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))")
        .bind(format!("messaging-sender:{}:agent-a", rollover.user_id))
        .execute(&mut *sender_lock)
        .await
        .expect("delay the earlier rollover request");
    let first_rollover = request_json(
        &app,
        Method::POST,
        &rollover_path,
        &rollover.agent.token,
        text_send(900, "one concurrent rollover send"),
    );
    let second_rollover = request_json(
        &app,
        Method::POST,
        &rollover_path,
        &rollover.agent_b.token,
        text_send(901, "the other concurrent rollover send"),
    );
    let (first_rollover, second_rollover) = tokio::join!(first_rollover, async {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                let waiting = sqlx::query_scalar::<_, bool>(
                    "SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE $1=ANY(pg_blocking_pids(pid)))",
                )
                .bind(blocker_pid)
                .fetch_one(&pool)
                .await
                .expect("observe the first request waiting on its sender lock");
                if waiting {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("first rollover request reaches its sender lock");
        let response = second_rollover.await;
        sender_lock.rollback().await.expect("release first sender");
        response
    });
    assert_eq!(first_rollover.status, StatusCode::OK, "{first_rollover:?}");
    assert_eq!(
        second_rollover.status,
        StatusCode::OK,
        "{second_rollover:?}"
    );
    let responses = [&first_rollover, &second_rollover];
    let rollover_response = responses
        .iter()
        .find(|response| data(response).get("continuation_id").is_some())
        .expect("exactly one concurrent send rolls the source entry");
    assert_eq!(
        data(rollover_response).get("seq").and_then(Value::as_i64),
        Some(500)
    );
    let continuation_id = data(rollover_response)
        .get("continuation_id")
        .and_then(Value::as_str)
        .and_then(|value| Uuid::parse_str(value).ok())
        .expect("five-hundredth send returns a continuation");
    let followed_response = responses
        .iter()
        .find(|response| data(response).get("continuation_id").is_none())
        .expect("the other send follows the continuation");
    assert_eq!(
        data(followed_response)
            .get("conversation_id")
            .and_then(Value::as_str),
        Some(continuation_id.to_string().as_str())
    );
    assert_eq!(
        data(followed_response).get("seq").and_then(Value::as_i64),
        Some(2),
        "the send waiting on a rolled source follows the locked continuation"
    );
    let old_rollover = sqlx::query_as::<_, (String, i64, Option<Uuid>)>(
        r#"
        SELECT status,last_seq,continues_from
        FROM brunn.messaging_conversations
        WHERE user_id=$1 AND conversation_id=$2
        "#,
    )
    .bind(rollover.user_id)
    .bind(rollover_conversation)
    .fetch_one(&pool)
    .await
    .expect("read closed rollover source");
    assert_eq!(old_rollover, ("closed".to_owned(), 500, None));
    let continuation = sqlx::query_as::<_, (String, i64, Option<Uuid>)>(
        r#"
        SELECT status,last_seq,continues_from
        FROM brunn.messaging_conversations
        WHERE user_id=$1 AND conversation_id=$2
        "#,
    )
    .bind(rollover.user_id)
    .bind(continuation_id)
    .fetch_one(&pool)
    .await
    .expect("read rollover continuation");
    assert_eq!(
        continuation,
        ("open".to_owned(), 2, Some(rollover_conversation))
    );
    let oversized_entries = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)::bigint
        FROM (
          SELECT conversation_id,count(*) AS message_count
          FROM brunn.messaging_message_index
          WHERE user_id=$1
            AND conversation_id IN ($2,$3)
          GROUP BY conversation_id
          HAVING count(*) > 500
        ) AS oversized
        "#,
    )
    .bind(rollover.user_id)
    .bind(rollover_conversation)
    .bind(continuation_id)
    .fetch_one(&pool)
    .await
    .expect("check rollover entry message caps");
    assert_eq!(oversized_entries, 0);
    let continuation_systems = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)::bigint
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND conversation_id=$2 AND seq=1 AND kind='system'
        "#,
    )
    .bind(rollover.user_id)
    .bind(continuation_id)
    .fetch_one(&pool)
    .await
    .expect("count continuation system record");
    assert_eq!(continuation_systems, 1);
    let cross_continuation_reply = request_json(
        &app,
        Method::POST,
        &rollover_path,
        &rollover.agent.token,
        json!({
            "client_key": client_key(902),
            "kind": "text",
            "body_md": "reply to the predecessor boundary",
            "in_reply_to": 500
        }),
    )
    .await;
    assert_eq!(cross_continuation_reply.status, StatusCode::OK);
    assert_eq!(
        data(&cross_continuation_reply)
            .get("conversation_id")
            .and_then(Value::as_str),
        Some(continuation_id.to_string().as_str())
    );
    let stored_reply = sqlx::query_as::<_, (Option<Uuid>, Option<i64>)>(
        r#"
        SELECT in_reply_to_conversation_id,in_reply_to
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND conversation_id=$2 AND seq=3
        "#,
    )
    .bind(rollover.user_id)
    .bind(continuation_id)
    .fetch_one(&pool)
    .await
    .expect("read cross-continuation reply reference");
    assert_eq!(
        stored_reply,
        (Some(rollover_conversation), Some(500)),
        "the server derives and stores the ancestor conversation identity"
    );
    let reply_collision_body = json!({
        "client_key": client_key(903),
        "kind": "text",
        "body_md": "reply target identity is part of idempotency",
        "in_reply_to": 2
    });
    let reply_collision_original = request_json(
        &app,
        Method::POST,
        &rollover_path,
        &rollover.agent.token,
        reply_collision_body.clone(),
    )
    .await;
    assert_eq!(reply_collision_original.status, StatusCode::OK);
    let reply_collision_replay = request_json(
        &app,
        Method::POST,
        &rollover_path,
        &rollover.agent.token,
        reply_collision_body.clone(),
    )
    .await;
    assert_eq!(reply_collision_replay.status, StatusCode::OK);
    assert_eq!(
        data(&reply_collision_replay)
            .get("duplicate")
            .and_then(Value::as_bool),
        Some(true)
    );
    let changed_reply_target = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{continuation_id}/messages"),
        &rollover.agent.token,
        reply_collision_body,
    )
    .await;
    assert_error(
        &changed_reply_target,
        StatusCode::CONFLICT,
        "idempotency_conflict",
    );

    let deadline = seed_workspace(&pool, "reply-deadline").await;
    let deadline_conversation = create_conversation(
        &app,
        &deadline.agent.token,
        &["agent-b"],
        "Injected reply deadline",
    )
    .await;
    let reply_by =
        Utc::now().trunc_subsecs(6) + ChronoDuration::minutes(1) + ChronoDuration::nanoseconds(789);
    let mut question_body = json!({
        "client_key": client_key(900),
        "kind": "question",
        "body_md": "Reply before the injected deadline",
        "expects_reply": true,
        "reply_by": reply_by
    });
    let question = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{deadline_conversation}/messages"),
        &deadline.agent.token,
        question_body.clone(),
    )
    .await;
    assert_eq!(
        question.status,
        StatusCode::OK,
        "deadline question: {:?}",
        question.body
    );
    let question_seq = data(&question)
        .get("seq")
        .and_then(Value::as_i64)
        .expect("deadline question returns a sequence");
    let question_replay = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{deadline_conversation}/messages"),
        &deadline.agent.token,
        question_body.clone(),
    )
    .await;
    assert_eq!(
        question_replay.status,
        StatusCode::OK,
        "nanosecond deadline replay: {:?}",
        question_replay.body
    );
    assert_eq!(data(&question_replay)["duplicate"], true);
    assert_eq!(data(&question_replay)["seq"], question_seq);
    question_body["reply_by"] = json!(reply_by + ChronoDuration::microseconds(1));
    let changed_deadline = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{deadline_conversation}/messages"),
        &deadline.agent.token,
        question_body,
    )
    .await;
    assert_error(
        &changed_deadline,
        StatusCode::CONFLICT,
        "idempotency_conflict",
    );
    let as_of = reply_by + ChronoDuration::seconds(1);
    assert!(
        messaging_service::process_due_reply_by(&state, as_of)
            .await
            .expect("process one due reply deadline")
    );
    assert!(
        !messaging_service::process_due_reply_by(&state, as_of)
            .await
            .expect("due reply deadline is idempotent")
    );
    let handled_at = sqlx::query_scalar::<_, Option<DateTime<Utc>>>(
        r#"
        SELECT reply_by_handled_at
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND conversation_id=$2 AND seq=$3
        "#,
    )
    .bind(deadline.user_id)
    .bind(deadline_conversation)
    .bind(question_seq)
    .fetch_one(&pool)
    .await
    .expect("read handled reply deadline");
    assert!(handled_at.is_some());
    let deadline_key = format!("reply-by:{deadline_conversation}:{question_seq}");
    let deadline_systems = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)::bigint
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND conversation_id=$2 AND system_key=$3
        "#,
    )
    .bind(deadline.user_id)
    .bind(deadline_conversation)
    .bind(&deadline_key)
    .fetch_one(&pool)
    .await
    .expect("count deadline system messages");
    let deadline_notifications = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT count(*)::bigint
        FROM brunn.notifications
        WHERE user_id=$1 AND event_key=$2
        "#,
    )
    .bind(deadline.user_id)
    .bind(deadline_key)
    .fetch_one(&pool)
    .await
    .expect("count deadline notifications");
    assert_eq!(deadline_systems, 1);
    assert_eq!(deadline_notifications, 1);

    let canceled = seed_workspace(&pool, "reply-deadline-close").await;
    let canceled_conversation = create_conversation(
        &app,
        &canceled.agent.token,
        &["agent-b"],
        "Close cancels a reply deadline",
    )
    .await;
    let canceled_reply_by = Utc::now() + ChronoDuration::minutes(1);
    let canceled_question = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{canceled_conversation}/messages"),
        &canceled.agent.token,
        json!({
            "client_key": client_key(901),
            "kind": "question",
            "body_md": "This deadline is canceled by close",
            "expects_reply": true,
            "reply_by": canceled_reply_by
        }),
    )
    .await;
    assert_eq!(canceled_question.status, StatusCode::OK);
    let canceled_question_seq = data(&canceled_question)
        .get("seq")
        .and_then(Value::as_i64)
        .expect("canceled question returns a sequence");
    let closed = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{canceled_conversation}/close"),
        &canceled.owner.token,
        json!({}),
    )
    .await;
    assert_eq!(closed.status, StatusCode::OK);
    let canceled_handled = sqlx::query_scalar::<_, Option<DateTime<Utc>>>(
        r#"
        SELECT reply_by_handled_at
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND conversation_id=$2 AND seq=$3
        "#,
    )
    .bind(canceled.user_id)
    .bind(canceled_conversation)
    .bind(canceled_question_seq)
    .fetch_one(&pool)
    .await
    .expect("read canceled reply deadline");
    assert!(
        canceled_handled.is_some(),
        "closing retires a future deadline so it cannot block the worker"
    );
    assert_eq!(
        sqlx::query_scalar::<_, i64>(
            "SELECT count(*)::bigint FROM brunn.messaging_message_index WHERE user_id=$1 AND system_key=$2",
        )
        .bind(canceled.user_id)
        .bind(format!(
            "reply-by:{canceled_conversation}:{canceled_question_seq}"
        ))
        .fetch_one(&pool)
        .await
        .expect("count canceled deadline systems"),
        0,
        "canceling a deadline does not synthesize an expiry message"
    );

    let deadline_race = seed_workspace(&pool, "reply-deadline-race").await;
    let deadline_race_conversation = create_conversation(
        &app,
        &deadline_race.agent.token,
        &["agent-b"],
        "Reply racing its deadline",
    )
    .await;
    let race_reply_by = Utc::now() + ChronoDuration::milliseconds(200);
    let race_question = request_json(
        &app,
        Method::POST,
        &format!("{MESSAGING_ROOT}/conversations/{deadline_race_conversation}/messages"),
        &deadline_race.agent.token,
        json!({
            "client_key": client_key(900),
            "kind": "question",
            "body_md": "Race the injected deadline",
            "expects_reply": true,
            "reply_by": race_reply_by
        }),
    )
    .await;
    assert_eq!(race_question.status, StatusCode::OK);
    let race_question_seq = data(&race_question)
        .get("seq")
        .and_then(Value::as_i64)
        .expect("race question returns a sequence");
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let race_path = format!("{MESSAGING_ROOT}/conversations/{deadline_race_conversation}/messages");
    let race_reply = request_json(
        &app,
        Method::POST,
        &race_path,
        &deadline_race.agent_b.token,
        json!({
            "client_key": client_key(901),
            "kind": "text",
            "body_md": "Concurrent answer",
            "in_reply_to": race_question_seq
        }),
    );
    let race_expiry = messaging_service::process_due_reply_by(&state, Utc::now());
    let (race_reply, race_expiry) =
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::join!(race_reply, race_expiry)
        })
        .await
        .expect("reply/deadline race completes without deadlock");
    assert_eq!(race_reply.status, StatusCode::OK);
    assert!(race_expiry.expect("deadline race is processed"));
    let race_effects = sqlx::query_as::<_, (i64, i64)>(
        r#"
        SELECT
          (SELECT count(*)::bigint
           FROM brunn.messaging_message_index
           WHERE user_id=$1 AND system_key=$2),
          (SELECT count(*)::bigint
           FROM brunn.notifications
           WHERE user_id=$1 AND event_key=$2)
        "#,
    )
    .bind(deadline_race.user_id)
    .bind(format!(
        "reply-by:{deadline_race_conversation}:{race_question_seq}"
    ))
    .fetch_one(&pool)
    .await
    .expect("count serialized race effects");
    assert!(
        race_effects == (0, 0) || race_effects == (1, 1),
        "the serial winner either answers before expiry or emits one complete expiry effect"
    );
}
