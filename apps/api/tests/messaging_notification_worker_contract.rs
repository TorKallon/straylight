use std::{collections::HashSet, collections::VecDeque, sync::Arc, time::Duration};

use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit, Payload},
};
use async_trait::async_trait;
use axum::{
    Extension, Json, Router,
    body::Body,
    http::{Method, Request, StatusCode, header},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use chrono::{SubsecRound, TimeZone, Utc};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sqlx::{PgPool, postgres::PgPoolOptions};
use tokio::sync::Mutex;
use tower::ServiceExt;
use uuid::Uuid;

use brunn::{
    ApiError, AppState, Config,
    auth::{AuthContext, hash_token},
    messaging_service,
    models::{CredentialId, UserId},
    notification_service::{
        self, ApnsAccepted, ApnsFailure, ApnsProvider, ApnsRequest, NotificationTarget,
        PublishRequest, process_next_on_pool,
    },
    worker,
};

const APP_ID: &str = "com.rourkem.brunn";

struct PrincipalFixture {
    user_id: Uuid,
    auth: AuthContext,
}

struct FakeProvider {
    outcomes: Mutex<VecDeque<Result<ApnsAccepted, ApnsFailure>>>,
    requests: Mutex<Vec<ApnsRequest>>,
}

impl FakeProvider {
    fn accepting() -> Self {
        Self {
            outcomes: Mutex::new(
                [Ok(ApnsAccepted {
                    provider_request_id: Some("messaging-contract-apns-id".to_owned()),
                    status: 200,
                })]
                .into(),
            ),
            requests: Mutex::new(Vec::new()),
        }
    }

    async fn one_request(&self) -> ApnsRequest {
        let requests = self.requests.lock().await;
        assert_eq!(
            requests.len(),
            1,
            "the fixture emits exactly one APNs request"
        );
        requests[0].clone()
    }
}

#[async_trait]
impl ApnsProvider for FakeProvider {
    async fn send(&self, request: ApnsRequest) -> Result<ApnsAccepted, ApnsFailure> {
        self.requests.lock().await.push(request);
        self.outcomes
            .lock()
            .await
            .pop_front()
            .expect("fake APNs outcome")
    }
}

async fn connect_pool() -> Option<(String, PgPool)> {
    let Some(database_url) = std::env::var("BRUNN_TEST_DATABASE_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        eprintln!(
            "BRUNN_TEST_DATABASE_URL is unset; skipping messaging notification/worker contract"
        );
        return None;
    };
    let pool = PgPoolOptions::new()
        .max_connections(8)
        .connect(&database_url)
        .await
        .expect("connect to disposable PostgreSQL");
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .expect("apply Brunn migrations");
    Some((database_url, pool))
}

async fn connect_state(database_url: &str, messaging_enabled: bool) -> AppState {
    let mut config = Config::from_env().expect("load disposable API configuration");
    config.database_url_rw = database_url.to_owned();
    config.database_url_ro = database_url.to_owned();
    config.database_url_admin = Some(database_url.to_owned());
    config.database_max_connections = 4;
    config.apns_delivery_enabled = false;
    config.messaging_enabled = messaging_enabled;
    AppState::connect(config)
        .await
        .expect("connect disposable API state")
}

async fn insert_principal(pool: &PgPool, label: &str) -> PrincipalFixture {
    let user_id = Uuid::now_v7();
    let owner_credential_id = Uuid::now_v7();
    let credential_id = Uuid::now_v7();
    let scope_id = Uuid::now_v7();
    let scope_ref = format!("scope:messaging-notification-{scope_id}");
    let capabilities = vec![
        "read".to_owned(),
        "notification:publish".to_owned(),
        "notification:manage".to_owned(),
        "message.read".to_owned(),
        "message.write".to_owned(),
        "admin".to_owned(),
    ];

    sqlx::query("INSERT INTO brunn.users (id,external_ref,display_name) VALUES ($1,$2,$3)")
        .bind(user_id)
        .bind(format!("messaging-notification-contract:{label}:{user_id}"))
        .bind(format!("Messaging notification contract {label}"))
        .execute(pool)
        .await
        .expect("insert notification contract user");
    sqlx::query("INSERT INTO brunn.scopes (id,user_id,scope_ref,name) VALUES ($1,$2,$3,$4)")
        .bind(scope_id)
        .bind(user_id)
        .bind(&scope_ref)
        .bind(format!("Messaging notification contract {label}"))
        .execute(pool)
        .await
        .expect("insert notification contract scope");
    for (id, credential_label) in [
        (owner_credential_id, format!("{label} owner")),
        (credential_id, format!("{label} agent-a")),
    ] {
        sqlx::query(
            r#"
            INSERT INTO brunn.api_credentials (
              id,user_id,label,token_hash,capabilities
            ) VALUES ($1,$2,$3,$4,$5)
            "#,
        )
        .bind(id)
        .bind(user_id)
        .bind(credential_label)
        .bind(hash_token(&format!("messaging-notification-contract-{id}")))
        .bind(&capabilities)
        .execute(pool)
        .await
        .expect("insert notification contract credential");
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
        .expect("grant notification contract scope");
    }

    for (agent_id, display_name, principal_kind) in [
        ("owner", "Owner", "owner"),
        ("agent-a", "Agent A", "resident"),
        ("agent-b", "Agent B", "resident"),
    ] {
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
        .bind(display_name)
        .bind(principal_kind)
        .bind(owner_credential_id)
        .execute(pool)
        .await
        .expect("insert notification contract messaging principal");
    }
    sqlx::query(
        r#"
        INSERT INTO brunn.messaging_credential_bindings (
          user_id,credential_id,agent_id,bound_by_credential_id
        ) VALUES ($1,$2,'agent-a',$3)
        "#,
    )
    .bind(user_id)
    .bind(credential_id)
    .bind(owner_credential_id)
    .execute(pool)
    .await
    .expect("bind notification contract credential");

    PrincipalFixture {
        user_id,
        auth: AuthContext {
            credential_id: CredentialId(credential_id),
            user_id: UserId(user_id),
            capabilities: capabilities.into_iter().collect::<HashSet<_>>(),
            scope_refs: vec![scope_ref],
            read_only: false,
        },
    }
}

fn publish_request(event_key: String, target: NotificationTarget) -> PublishRequest {
    PublishRequest {
        event_key,
        correlation_id: format!("messaging-notification-contract:{}", Uuid::now_v7()),
        kind: "operational".to_owned(),
        importance: "normal".to_owned(),
        title: "Generic conversation notification".to_owned(),
        body: "Open Brunn to view the conversation.".to_owned(),
        source: None,
        target,
        occurred_at: Some(Utc::now()),
        expires_at: None,
    }
}

fn assert_invalid_request(error: ApiError) {
    assert!(
        matches!(
            error,
            ApiError::Public {
                status: StatusCode::BAD_REQUEST,
                code: "invalid_request",
                ..
            }
        ),
        "malformed conversation targets fail closed: {error:?}"
    );
}

fn conversation_target(conversation_id: Value, seq: i64) -> NotificationTarget {
    serde_json::from_value(json!({
        "type": "conversation",
        "conversation_id": conversation_id,
        "seq": seq
    }))
    .expect("conversation is a typed notification target")
}

fn encryption_fixture(key: &[u8; 32], user_id: Uuid, installation_id: Uuid) -> (Vec<u8>, Vec<u8>) {
    let token = hex::encode([7_u8; 32]);
    let aad = format!("brunn.apns-token.v1|{user_id}|{installation_id}|development|{APP_ID}");
    let nonce = [11_u8; 12];
    let cipher = Aes256Gcm::new_from_slice(key).expect("AES key");
    let ciphertext = cipher
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: token.as_bytes(),
                aad: aad.as_bytes(),
            },
        )
        .expect("encrypt notification contract device token");
    (ciphertext, nonce.to_vec())
}

async fn insert_delivery(
    pool: &PgPool,
    target: Value,
    kind: &str,
    private_marker: &str,
) -> (Uuid, Uuid, Uuid) {
    let user_id = Uuid::now_v7();
    let credential_id = Uuid::now_v7();
    let installation_id = Uuid::now_v7();
    let client_installation_id = Uuid::now_v7();
    let notification_id = Uuid::now_v7();
    let delivery_id = Uuid::now_v7();
    let key = [23_u8; 32];
    let (ciphertext, nonce) = encryption_fixture(&key, user_id, client_installation_id);

    sqlx::query("INSERT INTO brunn.users (id,external_ref,display_name) VALUES ($1,$2,$3)")
        .bind(user_id)
        .bind(format!("messaging-apns-contract:{user_id}"))
        .bind("Messaging APNs contract")
        .execute(pool)
        .await
        .expect("insert APNs contract user");
    sqlx::query(
        r#"
        INSERT INTO brunn.api_credentials (
          id,user_id,label,token_hash,capabilities
        ) VALUES ($1,$2,'Messaging APNs contract',$3,$4)
        "#,
    )
    .bind(credential_id)
    .bind(user_id)
    .bind(hash_token(&format!(
        "messaging-apns-contract-{credential_id}"
    )))
    .bind(vec!["read", "notification:publish", "notification:manage"])
    .execute(pool)
    .await
    .expect("insert APNs contract credential");
    sqlx::query(
        r#"
        INSERT INTO brunn.notification_installations (
          id,user_id,client_installation_id,registered_by_credential_id,
          platform,environment,app_id,token_ciphertext,token_nonce,
          token_hash,preview,enabled
        ) VALUES (
          $1,$2,$3,$4,'ios','development',$5,$6,$7,$8,'generic',true
        )
        "#,
    )
    .bind(installation_id)
    .bind(user_id)
    .bind(client_installation_id)
    .bind(credential_id)
    .bind(APP_ID)
    .bind(ciphertext)
    .bind(nonce)
    .bind(hex::encode(Sha256::digest(
        client_installation_id.as_bytes(),
    )))
    .execute(pool)
    .await
    .expect("insert APNs contract installation");
    sqlx::query(
        r#"
        INSERT INTO brunn.notifications (
          id,user_id,producer_credential_id,event_key,request_hash,
          correlation_id,kind,importance,title,body,target,occurred_at,expires_at
        ) VALUES (
          $1,$2,$3,$4,$5,$6,$7,'normal',$8,$9,$10,
          clock_timestamp(),clock_timestamp()+interval '1 day'
        )
        "#,
    )
    .bind(notification_id)
    .bind(user_id)
    .bind(credential_id)
    .bind(format!("messaging-apns-contract:{notification_id}"))
    .bind(hex::encode(Sha256::digest(notification_id.as_bytes())))
    .bind(format!("messaging-apns-contract:{notification_id}"))
    .bind(kind)
    .bind(format!("private-{private_marker}-title"))
    .bind(format!("private-{private_marker}-body"))
    .bind(target)
    .execute(pool)
    .await
    .expect("insert APNs contract notification");
    sqlx::query(
        r#"
        INSERT INTO brunn.notification_deliveries (
          id,user_id,notification_id,installation_id,available_at
        ) VALUES ($1,$2,$3,$4,clock_timestamp()-interval '1 day')
        "#,
    )
    .bind(delivery_id)
    .bind(user_id)
    .bind(notification_id)
    .bind(installation_id)
    .execute(pool)
    .await
    .expect("insert APNs contract outbox row");
    (notification_id, delivery_id, client_installation_id)
}

async fn insert_installation_for_principal(
    pool: &PgPool,
    principal: &PrincipalFixture,
    label: &str,
) {
    let installation_id = Uuid::now_v7();
    let client_installation_id = Uuid::now_v7();
    let key = [23_u8; 32];
    let (ciphertext, nonce) = encryption_fixture(&key, principal.user_id, client_installation_id);

    sqlx::query(
        r#"
        INSERT INTO brunn.notification_installations (
          id,user_id,client_installation_id,registered_by_credential_id,
          platform,environment,app_id,token_ciphertext,token_nonce,
          token_hash,preview,enabled
        ) VALUES (
          $1,$2,$3,$4,'ios','development',$5,$6,$7,$8,'generic',true
        )
        "#,
    )
    .bind(installation_id)
    .bind(principal.user_id)
    .bind(client_installation_id)
    .bind(principal.auth.credential_id.0)
    .bind(APP_ID)
    .bind(ciphertext)
    .bind(nonce)
    .bind(hex::encode(Sha256::digest(
        format!("messaging-notification-installation:{label}:{client_installation_id}").as_bytes(),
    )))
    .execute(pool)
    .await
    .expect("insert messaging notification installation");
}

async fn process_one(pool: &PgPool) -> ApnsRequest {
    let provider = Arc::new(FakeProvider::accepting());
    let encoded_key = STANDARD.encode([23_u8; 32]);
    assert!(
        process_next_on_pool(pool, &encoded_key, provider.clone())
            .await
            .expect("process APNs contract delivery")
    );
    provider.one_request().await
}

#[tokio::test]
async fn conversation_notification_target_is_typed_and_fails_closed() {
    let Some((database_url, pool)) = connect_pool().await else {
        return;
    };
    let state = connect_state(&database_url, true).await;
    let principal = insert_principal(&pool, "target-validation").await;
    let conversation_id = Uuid::now_v7();

    assert!(
        serde_json::from_value::<NotificationTarget>(json!({
            "type": "conversation",
            "conversation_id": conversation_id,
            "seq": 1,
            "private_body": "must be rejected"
        }))
        .is_err(),
        "typed conversation targets reject unknown fields"
    );

    let invalid_version = conversation_target(json!(Uuid::new_v4()), 1);
    let error = notification_service::publish(
        axum::extract::State(state.clone()),
        Extension(principal.auth.clone()),
        Json(publish_request(
            format!("conversation-target-v4:{}", Uuid::now_v7()),
            invalid_version,
        )),
    )
    .await
    .expect_err("conversation ids must be canonical UUIDv7 values");
    assert_invalid_request(error);

    let uppercase_id =
        conversation_target(json!(conversation_id.to_string().to_ascii_uppercase()), 1);
    let error = notification_service::publish(
        axum::extract::State(state.clone()),
        Extension(principal.auth.clone()),
        Json(publish_request(
            format!("conversation-target-uppercase:{}", Uuid::now_v7()),
            uppercase_id,
        )),
    )
    .await
    .expect_err("uppercase UUIDv7 values are not canonical conversation ids");
    assert_invalid_request(error);

    let unhyphenated_id = conversation_target(json!(conversation_id.simple().to_string()), 1);
    let error = notification_service::publish(
        axum::extract::State(state.clone()),
        Extension(principal.auth.clone()),
        Json(publish_request(
            format!("conversation-target-unhyphenated:{}", Uuid::now_v7()),
            unhyphenated_id,
        )),
    )
    .await
    .expect_err("unhyphenated UUIDv7 values are not canonical conversation ids");
    assert_invalid_request(error);

    let invalid_seq = conversation_target(json!(conversation_id), 0);
    let error = notification_service::publish(
        axum::extract::State(state.clone()),
        Extension(principal.auth.clone()),
        Json(publish_request(
            format!("conversation-target-seq:{}", Uuid::now_v7()),
            invalid_seq,
        )),
    )
    .await
    .expect_err("conversation notification sequences are positive");
    assert_invalid_request(error);

    let valid_target = conversation_target(json!(conversation_id), 1);
    let gate_off_state = connect_state(&database_url, false).await;
    let error = notification_service::publish(
        axum::extract::State(gate_off_state),
        Extension(principal.auth.clone()),
        Json(publish_request(
            format!("conversation-target-gate-off:{}", Uuid::now_v7()),
            valid_target.clone(),
        )),
    )
    .await
    .expect_err("conversation targets stay unavailable through public publish with messaging off");
    assert_invalid_request(error);

    let response = notification_service::publish(
        axum::extract::State(state),
        Extension(principal.auth),
        Json(publish_request(
            format!("conversation-target-valid:{}", Uuid::now_v7()),
            valid_target,
        )),
    )
    .await
    .expect("valid typed conversation target is accepted");
    assert_eq!(
        response.0.notification.target,
        json!({
            "type": "conversation",
            "conversation_id": conversation_id,
            "seq": 1
        })
    );
}

#[tokio::test]
async fn conversation_apns_is_generic_prefetchable_and_conversation_collapsed() {
    let Some((_database_url, pool)) = connect_pool().await else {
        return;
    };
    let conversation_id = Uuid::now_v7();
    let (notification_id, delivery_id, _) = insert_delivery(
        &pool,
        json!({
            "type": "conversation",
            "conversation_id": conversation_id,
            "seq": 17
        }),
        "operational",
        "conversation-secret",
    )
    .await;
    let request = process_one(&pool).await;

    assert_eq!(
        request.payload["brunn_route"],
        format!("brunn://conversation/{conversation_id}?seq=17")
    );
    assert_eq!(request.collapse_id, conversation_id.to_string());
    assert_eq!(
        request.payload,
        json!({
            "aps": {
                "alert": {
                    "title": "Brunn",
                    "body": "A new agent message is available."
                },
                "content-available": 1
            },
            "schema": "brunn-push@v1",
            "notification_ref": format!("notification:{}", notification_id.simple()),
            "delivery_ref": format!("delivery:{}", delivery_id.simple()),
            "brunn_route": format!(
                "brunn://conversation/{conversation_id}?seq=17"
            )
        })
    );
    let serialized = serde_json::to_string(&request.payload).expect("serialize APNs payload");
    assert!(!serialized.contains("conversation-secret"));
}

#[tokio::test]
async fn invalid_stored_conversation_target_uses_private_notification_fallback() {
    let Some((_database_url, pool)) = connect_pool().await else {
        return;
    };
    let conversation_id = Uuid::now_v7();
    let (notification_id, delivery_id, _) = insert_delivery(
        &pool,
        json!({
            "type": "conversation",
            "conversation_id": conversation_id.to_string().to_ascii_uppercase(),
            "seq": 17
        }),
        "operational",
        "invalid-conversation-secret",
    )
    .await;
    let request = process_one(&pool).await;
    let fallback_route = format!(
        "brunn://notification/{}?delivery={}",
        notification_id.simple(),
        delivery_id.simple()
    );

    assert_eq!(
        request.collapse_id,
        format!("notification-{}", notification_id.simple())
    );
    assert_eq!(
        request.payload,
        json!({
            "aps": {
                "alert": {
                    "title": "Brunn",
                    "body": "Brunn has an operational alert."
                }
            },
            "schema": "brunn-push@v1",
            "notification_ref": format!("notification:{}", notification_id.simple()),
            "delivery_ref": format!("delivery:{}", delivery_id.simple()),
            "brunn_route": fallback_route
        })
    );
    let serialized = serde_json::to_string(&request.payload).expect("serialize fallback payload");
    assert!(!serialized.contains("invalid-conversation-secret"));
}

#[tokio::test]
async fn existing_notification_and_task_apns_contracts_are_unchanged() {
    let Some((_database_url, pool)) = connect_pool().await else {
        return;
    };
    let task_id = Uuid::now_v7();
    let (task_notification_id, task_delivery_id, _) = insert_delivery(
        &pool,
        json!({"type": "task", "task_ref": task_id.to_string()}),
        "task_guard",
        "task-secret",
    )
    .await;
    let task_request = process_one(&pool).await;
    assert_eq!(
        task_request.collapse_id,
        format!("notification-{}", task_notification_id.simple())
    );
    assert_eq!(
        task_request.payload,
        json!({
            "aps": {
                "alert": {
                    "title": "Brunn",
                    "body": "A new Brunn alert is available."
                }
            },
            "schema": "brunn-push@v1",
            "notification_ref": format!("notification:{}", task_notification_id.simple()),
            "delivery_ref": format!("delivery:{}", task_delivery_id.simple()),
            "brunn_route": format!("brunn://task/{task_id}")
        })
    );

    let (notification_id, delivery_id, _) = insert_delivery(
        &pool,
        json!({"type": "notification"}),
        "news_alert",
        "notification-secret",
    )
    .await;
    let notification_request = process_one(&pool).await;
    assert_eq!(
        notification_request.collapse_id,
        format!("notification-{}", notification_id.simple())
    );
    assert_eq!(
        notification_request.payload,
        json!({
            "aps": {
                "alert": {
                    "title": "Brunn",
                    "body": "A new Brunn alert is available."
                }
            },
            "schema": "brunn-push@v1",
            "notification_ref": format!("notification:{}", notification_id.simple()),
            "delivery_ref": format!("delivery:{}", delivery_id.simple()),
            "brunn_route": format!(
                "brunn://notification/{}?delivery={}",
                notification_id.simple(),
                delivery_id.simple()
            )
        })
    );
}

async fn messaging_app(state: AppState, auth: AuthContext) -> Router {
    messaging_service::router()
        .with_state(state)
        .layer(Extension(auth))
}

async fn request_json(app: &Router, method: Method, uri: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&body).expect("serialize messaging worker request"),
                ))
                .expect("build messaging worker request"),
        )
        .await
        .expect("serve messaging worker request");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect messaging worker response")
        .to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, body)
}

async fn send_owner_participant_message(
    state: AppState,
    principal: &PrincipalFixture,
    key_suffix: &str,
) -> (Uuid, i64) {
    let app = messaging_app(state, principal.auth.clone()).await;
    let (status, body) = request_json(
        &app,
        Method::POST,
        "/workspace/messaging/conversations",
        json!({
            "participants": ["owner"],
            "subject": format!("Quiet hours {key_suffix}")
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create owner conversation");
    let conversation_id = body
        .pointer("/data/conversation_id")
        .and_then(Value::as_str)
        .and_then(|value| Uuid::parse_str(value).ok())
        .expect("owner conversation id");

    let (status, body) = request_json(
        &app,
        Method::POST,
        &format!("/workspace/messaging/conversations/{conversation_id}/messages"),
        json!({
            "client_key": format!("0000000000000000000000000{key_suffix}"),
            "kind": "text",
            "body_md": "A private message body that must not affect delivery timing."
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "send owner conversation message");
    let seq = body
        .pointer("/data/seq")
        .and_then(Value::as_i64)
        .expect("owner conversation message sequence");
    (conversation_id, seq)
}

async fn delivery_times(
    pool: &PgPool,
    user_id: Uuid,
    conversation_id: Uuid,
    seq: i64,
) -> (chrono::DateTime<Utc>, chrono::DateTime<Utc>) {
    sqlx::query_as(
        r#"
        SELECT notification.occurred_at,delivery.available_at
        FROM brunn.notifications AS notification
        JOIN brunn.notification_deliveries AS delivery
          ON delivery.user_id=notification.user_id
         AND delivery.notification_id=notification.id
        WHERE notification.user_id=$1 AND notification.event_key=$2
        "#,
    )
    .bind(user_id)
    .bind(format!("message:{conversation_id}:{seq}"))
    .fetch_one(pool)
    .await
    .expect("read messaging notification delivery timing")
}

#[tokio::test]
async fn messaging_delivery_defers_to_quiet_end_without_override() {
    let Some((database_url, pool)) = connect_pool().await else {
        return;
    };
    let state = connect_state(&database_url, true).await;
    let principal = insert_principal(&pool, "quiet-hours-delayed").await;
    insert_installation_for_principal(&pool, &principal, "quiet-hours-delayed").await;
    let database_now = sqlx::query_scalar::<_, chrono::DateTime<Utc>>("SELECT clock_timestamp()")
        .fetch_one(&pool)
        .await
        .expect("read database time for quiet-hours fixture");
    let quiet_start = (database_now - chrono::Duration::hours(1)).time();
    let quiet_end = (database_now + chrono::Duration::hours(1)).time();
    sqlx::query(
        r#"
        UPDATE brunn.task_settings
        SET timezone='UTC',quiet_hours_start=$2,quiet_hours_end=$3,
            quiet_override_enabled=true,quiet_override_within_hours=168
        WHERE user_id=$1
        "#,
    )
    .bind(principal.user_id)
    .bind(quiet_start)
    .bind(quiet_end)
    .execute(&pool)
    .await
    .expect("configure in-window messaging quiet hours");

    let (conversation_id, seq) = send_owner_participant_message(state, &principal, "3").await;
    let (occurred_at, available_at) =
        delivery_times(&pool, principal.user_id, conversation_id, seq).await;
    let end_date = if quiet_start > quiet_end && occurred_at.time() >= quiet_start {
        occurred_at.date_naive() + chrono::Duration::days(1)
    } else {
        occurred_at.date_naive()
    };
    let expected_quiet_end = Utc.from_utc_datetime(&end_date.and_time(quiet_end));
    assert_eq!(
        available_at, expected_quiet_end,
        "messaging never overrides quiet hours, even when task overrides are enabled"
    );
    assert!(available_at > occurred_at);
}

#[tokio::test]
async fn messaging_delivery_is_immediate_when_quiet_hours_are_disabled() {
    let Some((database_url, pool)) = connect_pool().await else {
        return;
    };
    let state = connect_state(&database_url, true).await;
    let principal = insert_principal(&pool, "quiet-hours-disabled").await;
    insert_installation_for_principal(&pool, &principal, "quiet-hours-disabled").await;
    sqlx::query(
        r#"
        UPDATE brunn.task_settings
        SET timezone='UTC',quiet_hours_start='07:00',quiet_hours_end='07:00',
            quiet_override_enabled=true,quiet_override_within_hours=168
        WHERE user_id=$1
        "#,
    )
    .bind(principal.user_id)
    .execute(&pool)
    .await
    .expect("disable messaging quiet hours with an equal start and end");

    let (conversation_id, seq) = send_owner_participant_message(state, &principal, "4").await;
    let (occurred_at, available_at) =
        delivery_times(&pool, principal.user_id, conversation_id, seq).await;
    assert_eq!(
        available_at, occurred_at,
        "start == end disables quiet hours and preserves the message as_of"
    );
}

async fn seed_due_question(
    _pool: &PgPool,
    state: AppState,
    principal: &PrincipalFixture,
    key_suffix: &str,
) -> (Uuid, i64) {
    let app = messaging_app(state, principal.auth.clone()).await;
    let (status, body) = request_json(
        &app,
        Method::POST,
        "/workspace/messaging/conversations",
        json!({"participants": ["agent-b"], "subject": format!("Worker {key_suffix}")}),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "create worker contract conversation"
    );
    let conversation_id = body
        .pointer("/data/conversation_id")
        .and_then(Value::as_str)
        .and_then(|value| Uuid::parse_str(value).ok())
        .expect("worker contract conversation id");

    let (status, body) = request_json(
        &app,
        Method::POST,
        &format!("/workspace/messaging/conversations/{conversation_id}/messages"),
        json!({
            "client_key": format!("0000000000000000000000000{key_suffix}"),
            "kind": "question",
            "body_md": "Does the existing worker expire this question?",
            "expects_reply": true,
            "reply_by": Utc::now().trunc_subsecs(6) + chrono::Duration::seconds(1) + chrono::Duration::nanoseconds(789)
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "send worker contract question: {body}"
    );
    let question_seq = body
        .pointer("/data/seq")
        .and_then(Value::as_i64)
        .expect("worker question sequence");
    (conversation_id, question_seq)
}

async fn handled_at(pool: &PgPool, user_id: Uuid, conversation_id: Uuid, seq: i64) -> bool {
    sqlx::query_scalar::<_, Option<chrono::DateTime<Utc>>>(
        r#"
        SELECT reply_by_handled_at
        FROM brunn.messaging_message_index
        WHERE user_id=$1 AND conversation_id=$2 AND seq=$3
        "#,
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(seq)
    .fetch_one(pool)
    .await
    .expect("read worker contract deadline state")
    .is_some()
}

#[tokio::test]
async fn existing_worker_schedules_reply_by_only_when_messaging_is_enabled() {
    let Some((database_url, pool)) = connect_pool().await else {
        return;
    };

    let state_off = connect_state(&database_url, false).await;
    let off_principal = insert_principal(&pool, "worker-gate-off").await;
    let (off_conversation, off_seq) =
        seed_due_question(&pool, state_off.clone(), &off_principal, "1").await;
    let off_worker = tokio::spawn(worker::run(state_off));
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    assert!(
        !handled_at(&pool, off_principal.user_id, off_conversation, off_seq).await,
        "the default-off gate keeps reply_by work unreachable"
    );
    off_worker.abort();
    let _ = off_worker.await;
    sqlx::query(
        r#"
        UPDATE brunn.messaging_message_index
        SET reply_by_handled_at=clock_timestamp()
        WHERE user_id=$1 AND conversation_id=$2 AND seq=$3
        "#,
    )
    .bind(off_principal.user_id)
    .bind(off_conversation)
    .bind(off_seq)
    .execute(&pool)
    .await
    .expect("retire the completed gate-off fixture before the gate-on cycle");

    let state_on = connect_state(&database_url, true).await;
    let on_principal = insert_principal(&pool, "worker-gate-on").await;
    let (on_conversation, on_seq) =
        seed_due_question(&pool, state_on.clone(), &on_principal, "2").await;
    let on_worker = tokio::spawn(worker::run(state_on.clone()));
    let processed = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if handled_at(&pool, on_principal.user_id, on_conversation, on_seq).await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;
    on_worker.abort();
    let _ = on_worker.await;
    processed.expect("the existing worker processes a due reply_by with the gate on");

    let event_key = format!("reply-by:{on_conversation}:{on_seq}");
    let effects = sqlx::query_as::<_, (i64, i64)>(
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
    .bind(on_principal.user_id)
    .bind(&event_key)
    .fetch_one(&pool)
    .await
    .expect("count scheduled reply_by effects");
    assert_eq!(effects, (1, 1));

    let replay_worker = tokio::spawn(worker::run(state_on));
    tokio::time::sleep(Duration::from_millis(900)).await;
    replay_worker.abort();
    let _ = replay_worker.await;
    let replay_effects = sqlx::query_as::<_, (i64, i64)>(
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
    .bind(on_principal.user_id)
    .bind(event_key)
    .fetch_one(&pool)
    .await
    .expect("count idempotent scheduled reply_by effects");
    assert_eq!(replay_effects, (1, 1));
}
