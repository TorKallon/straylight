//! Exercise the real HTTP boundary, RLS, object store, and publication path.
//! BRUNN_TEST_DATABASE_URL must name a disposable database (same as other gates).
use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
};
use brunn::{AppState, Config, auth::hash_token, router};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sqlx::{PgPool, postgres::PgPoolOptions};
use tower::ServiceExt;
use uuid::Uuid;

async fn principal(pool: &PgPool, capabilities: &[&str]) -> (Uuid, String) {
    let user = Uuid::now_v7();
    let credential = Uuid::now_v7();
    let token = format!("binary-test-{credential}");
    sqlx::query(
        "INSERT INTO brunn.users(id,external_ref,display_name) VALUES($1,$2,'Binary upload test')",
    )
    .bind(user)
    .bind(format!("binary-test-{user}"))
    .execute(pool)
    .await
    .unwrap();
    sqlx::query("INSERT INTO brunn.api_credentials(id,user_id,label,token_hash,capabilities) VALUES($1,$2,'Binary upload test',$3,$4)")
        .bind(credential).bind(user).bind(hash_token(&token)).bind(capabilities).execute(pool).await.unwrap();
    (user, token)
}

async fn call(
    app: &Router,
    method: &str,
    path: &str,
    auth: &str,
    content_type: &str,
    body: Vec<u8>,
) -> (StatusCode, Vec<u8>) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .header("Authorization", auth)
                .header("Content-Type", content_type)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    (
        status,
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
}

async fn mint(app: &Router, token: &str, request: Value) -> (StatusCode, Value) {
    let (status, bytes) = call(
        app,
        "POST",
        "/v1/uploads",
        &format!("Bearer {token}"),
        "application/json",
        serde_json::to_vec(&request).unwrap(),
    )
    .await;
    (status, serde_json::from_slice(&bytes).unwrap())
}

async fn put(app: &Router, grant: &Value, bytes: &[u8]) -> (StatusCode, Value) {
    let (status, response) = call(
        app,
        "PUT",
        "/v1/workspace/binaries/content",
        grant["data"]["headers"]["Authorization"].as_str().unwrap(),
        "image/jpeg",
        bytes.to_vec(),
    )
    .await;
    (status, serde_json::from_slice(&response).unwrap())
}

#[tokio::test]
async fn hosted_upload_is_version_bound_replay_safe_and_tenant_isolated() {
    let Ok(database_url) = std::env::var("BRUNN_TEST_DATABASE_URL") else {
        eprintln!("BRUNN_TEST_DATABASE_URL unset; skipping binary upload database gate");
        return;
    };
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(&database_url)
        .await
        .unwrap();
    sqlx::migrate!("./migrations").run(&pool).await.unwrap();
    let mut config = Config::from_env().unwrap();
    let role_url = |role: &str| {
        let mut url = url::Url::parse(&database_url).unwrap();
        url.query_pairs_mut()
            .append_pair("options", &format!("-c role={role}"));
        url.to_string()
    };
    config.database_url_rw = role_url("app_rw");
    config.database_url_ro = role_url("app_ro");
    config.database_url_admin = None;
    config.semantic_lane = false;
    config.apns_delivery_enabled = false;
    let app = router(AppState::connect(config).await.unwrap());
    let (user, token) = principal(&pool, &["save", "read"]).await;
    let (_, reader) = principal(&pool, &["read"]).await;
    let (_, neighbor) = principal(&pool, &["save", "read"]).await;
    let bytes = b"\xff\xd8\xff\xe0binary-fixture\xff\xd9";
    let hash = hex::encode(Sha256::digest(bytes));
    let request = json!({"path":"Inbox/original.jpg","media_type":"image/jpeg","size_bytes":bytes.len(),"sha256":hash});
    assert_eq!(
        mint(&app, &reader, request.clone()).await.0,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        mint(
            &app,
            &token,
            json!({"path":"../escape.jpg","media_type":"image/jpeg","size_bytes":4})
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        mint(
            &app,
            &token,
            json!({"path":"Inbox/huge.jpg","media_type":"image/jpeg","size_bytes":4294967297u64})
        )
        .await
        .0,
        StatusCode::PAYLOAD_TOO_LARGE
    );
    let (status, grant) = mint(&app, &token, request.clone()).await;
    assert_eq!(status, StatusCode::OK, "{grant}");
    let upload_auth = grant["data"]["headers"]["Authorization"].as_str().unwrap();
    assert_eq!(
        call(
            &app,
            "POST",
            "/v1/workspace/write",
            upload_auth,
            "application/json",
            b"{}".to_vec()
        )
        .await
        .0,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(put(&app, &grant, b"wrong").await.0, StatusCode::BAD_REQUEST);
    assert_eq!(
        put(&app, &grant, &vec![0; bytes.len()]).await.0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        put(&app, &grant, &vec![0; bytes.len() + 1]).await.0,
        StatusCode::BAD_REQUEST
    );
    // Two concurrent retries produce one immutable version, not two uploads.
    let (a, b) = tokio::join!(put(&app, &grant, bytes), put(&app, &grant, bytes));
    assert!(
        matches!(
            (a.0, b.0),
            (StatusCode::CREATED, StatusCode::CONFLICT)
                | (StatusCode::CONFLICT, StatusCode::CREATED)
        ),
        "{a:?} {b:?}"
    );
    let published = if a.0 == StatusCode::CREATED { a.1 } else { b.1 };
    assert_eq!(published["data"]["version"], 1);
    assert_eq!(published["data"]["content_hash"], format!("sha256:{hash}"));
    let entry_ref = published["data"]["entry_ref"].as_str().unwrap();
    let (status, fetched) = call(
        &app,
        "GET",
        &format!("/v1/workspace/binaries/{entry_ref}/content?version=1"),
        &format!("Bearer {token}"),
        "application/json",
        vec![],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(fetched, bytes);
    assert_eq!(
        call(
            &app,
            "GET",
            &format!("/v1/workspace/binaries/{entry_ref}/content"),
            &format!("Bearer {neighbor}"),
            "application/json",
            vec![]
        )
        .await
        .0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        mint(&app, &token, request.clone()).await.0,
        StatusCode::CONFLICT
    );
    let mut replace = request.clone();
    replace["expected_version"] = json!(1);
    let (_, next) = mint(&app, &token, replace.clone()).await;
    let (_, stale) = mint(&app, &token, replace).await;
    assert_eq!(put(&app, &next, bytes).await.0, StatusCode::CREATED);
    assert_eq!(put(&app, &stale, bytes).await.0, StatusCode::CONFLICT);
    assert_eq!(
        put(&app, &grant, bytes).await.1["error"]["code"],
        "upload_completed"
    );
    // Moving the entry away must not turn the original grant into create-again.
    sqlx::query("UPDATE brunn.entries SET path='Inbox/moved.jpg' WHERE user_id=$1 AND path='Inbox/original.jpg'").bind(user).execute(&pool).await.unwrap();
    assert_eq!(
        put(&app, &grant, bytes).await.1["error"]["code"],
        "upload_completed"
    );
    assert_eq!(put(&app, &stale, bytes).await.0, StatusCode::CONFLICT);
    let jobs: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM brunn.jobs WHERE user_id=$1 AND kind='describe_binary'",
    )
    .bind(user)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        jobs, 0,
        "hosted uploads must not queue paid image descriptions"
    );
    let versions: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM brunn.entry_versions WHERE user_id=$1 AND object_key IS NOT NULL",
    )
    .bind(user)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(versions, 2);
}
