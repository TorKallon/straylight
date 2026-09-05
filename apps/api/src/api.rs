use axum::{
    Router,
    extract::DefaultBodyLimit,
    http::{HeaderName, HeaderValue, Method, StatusCode},
    middleware,
    routing::{delete, get, post, put},
};
use tower::{ServiceBuilder, limit::ConcurrencyLimitLayer};
use tower_http::{
    catch_panic::CatchPanicLayer, compression::CompressionLayer, cors::CorsLayer,
    timeout::TimeoutLayer, trace::TraceLayer,
};

use crate::{
    auth, briefing_service, dashboard_service, db::AppState, document_service, dreaming_service,
    location, messaging_service, notification_service, request_context, secret_service, service,
    simple_core, task_service, telemetry, web_auth,
};

pub fn router(state: AppState) -> Router {
    let request_id = HeaderName::from_static("x-request-id");
    let allowed_origins: Vec<HeaderValue> = state
        .config
        .allowed_origins
        .iter()
        .filter_map(|origin| HeaderValue::from_str(origin).ok())
        .collect();
    let workspace_write_body_limit = if state.config.messaging_enabled {
        16 * 1024 * 1024
    } else {
        5 * 1024 * 1024
    };
    let workspace_ordinary = Router::new()
        .route("/me", get(service::me))
        .route("/status", get(service::status))
        .route("/workspace/open", post(simple_core::open))
        .route("/uploads", post(crate::binary_upload::mint))
        .route("/workspace/search", post(simple_core::search))
        .route("/workspace/read", post(simple_core::read))
        .route(
            "/workspace/write",
            post(simple_core::write).layer(DefaultBodyLimit::max(workspace_write_body_limit)),
        )
        .route(
            "/workspace/capture",
            post(simple_core::capture).layer(DefaultBodyLimit::max(5 * 1024 * 1024)),
        )
        .route(
            "/workspace/tasks/capture",
            post(task_service::capture_tasks),
        )
        .route(
            "/workspace/tasks/candidates",
            get(task_service::task_candidates),
        )
        .route(
            "/workspace/tasks/corrections",
            get(task_service::task_corrections),
        )
        .route(
            "/workspace/tasks/done-summary",
            get(task_service::task_done_summary),
        )
        .route(
            "/workspace/tasks/settings",
            get(task_service::get_task_settings).put(task_service::update_task_settings),
        )
        .route(
            "/workspace/tasks/guard/status",
            get(task_service::task_guard_status),
        )
        .route(
            "/workspace/tasks/{task_ref}",
            get(task_service::get_task).patch(task_service::update_task),
        )
        .route(
            "/workspace/contexts",
            get(task_service::list_contexts).post(task_service::create_context),
        )
        .route(
            "/workspace/contexts/merge",
            post(task_service::merge_contexts),
        )
        .route(
            "/workspace/contexts/available/{surface}",
            put(task_service::set_available_contexts),
        )
        .route(
            "/workspace/contexts/{slug}",
            axum::routing::patch(task_service::archive_context),
        )
        .route("/workspace/projects", get(task_service::list_projects))
        .route(
            "/workspace/projects/{slug}/state",
            get(task_service::project_state),
        )
        .route(
            "/workspace/projects/{slug}/interest",
            put(task_service::set_project_interest),
        )
        .route(
            "/workspace/projects/{slug}",
            put(task_service::register_project),
        )
        .route(
            "/workspace/integrations/todoist/status",
            get(task_service::todoist_status),
        )
        .route(
            "/workspace/integrations/todoist/config",
            put(task_service::configure_todoist),
        )
        .route(
            "/workspace/integrations/todoist/pull",
            post(task_service::pull_todoist),
        )
        .route(
            "/workspace/briefings/publish",
            post(briefing_service::publish).layer(DefaultBodyLimit::max(5 * 1024 * 1024)),
        )
        .route(
            "/workspace/briefings/dedupe-check",
            post(briefing_service::dedupe_check),
        )
        .route("/workspace/briefings", get(briefing_service::list_editions))
        .route(
            "/workspace/briefings/topics",
            get(briefing_service::topics_snapshot),
        )
        .route(
            "/workspace/briefings/items/action",
            post(briefing_service::item_action),
        )
        .route(
            "/workspace/briefings/{date}/{edition}",
            get(briefing_service::get_edition),
        )
        .route(
            "/workspace/documents/publish",
            post(document_service::publish).layer(DefaultBodyLimit::max(5 * 1024 * 1024)),
        )
        .route(
            "/workspace/documents/{slug}",
            get(document_service::get_document),
        )
        .route("/workspace/changes", get(simple_core::changes))
        .route("/workspace/dashboard", get(dashboard_service::dashboard))
        .route(
            "/workspace/notifications/publish",
            post(notification_service::publish),
        )
        .route("/workspace/notifications", get(notification_service::list))
        .route(
            "/workspace/notifications/{notification_ref}",
            get(notification_service::detail),
        )
        .route(
            "/workspace/notifications/{notification_ref}/receipts",
            post(notification_service::receipt),
        )
        .route(
            "/workspace/notification-installations/{installation_id}",
            put(notification_service::upsert_installation)
                .delete(notification_service::revoke_installation),
        )
        .route("/workspace/secrets", get(secret_service::list))
        .route("/workspace/secrets/put", post(secret_service::put))
        .route("/workspace/secrets/get", post(secret_service::get))
        .route(
            "/workspace/secrets/delete",
            post(secret_service::delete_secret),
        )
        .route(
            "/workspace/checkpoint",
            post(simple_core::checkpoint).layer(DefaultBodyLimit::max(5 * 1024 * 1024)),
        )
        .route("/workspace/binaries", get(simple_core::list_binaries))
        .route("/workspace/manifest", get(simple_core::manifest))
        .route("/workspace/usage", get(simple_core::usage))
        .route("/workspace/jobs", get(simple_core::list_jobs))
        .route("/location/reports", post(location::routes::reports))
        .route("/location/presence", get(location::routes::presence))
        .route("/location/rederive", post(location::routes::rederive))
        .route("/location/live", delete(location::routes::delete_live))
        .route(
            "/workspace/entries/{entry_ref}",
            delete(simple_core::delete_entry),
        )
        .route(
            "/workspace/binaries/{entry_ref}",
            get(simple_core::binary_metadata),
        )
        .route("/admin/users", post(service::admin_provision_user))
        .route(
            "/admin/users/{user_ref}/recover",
            post(service::admin_recover_credential),
        )
        .route(
            "/credentials",
            get(service::list_credentials).post(service::create_credential),
        )
        .route(
            "/credentials/{credential_ref}",
            delete(service::revoke_credential),
        );
    let account_ordinary = Router::new()
        .route(
            "/account/exports",
            get(service::list_account_exports).post(service::request_account_export),
        )
        .route(
            "/account/exports/{export_ref}",
            get(service::get_account_export).delete(service::delete_account_export),
        )
        .route(
            "/account/deletion",
            get(service::get_latest_account_deletion).post(service::request_account_deletion),
        )
        .route(
            "/account/deletions/{request_ref}",
            get(service::get_account_deletion),
        );
    let evaluation_ordinary = Router::new().route(
        "/workspace/admin/eval/imports/{import_id}",
        get(simple_core::evaluation_status).delete(simple_core::cleanup_evaluation),
    );
    let mut ordinary = workspace_ordinary
        .merge(dreaming_service::router())
        .merge(account_ordinary);
    if state.config.messaging_enabled {
        ordinary = ordinary.merge(messaging_service::router());
    }
    if state.config.evaluation_api_enabled {
        ordinary = ordinary.merge(evaluation_ordinary);
    }
    let ordinary = ordinary.layer(TimeoutLayer::with_status_code(
        StatusCode::REQUEST_TIMEOUT,
        state.config.request_timeout,
    ));
    let workspace_transfers = Router::new()
        .route(
            "/workspace/binaries",
            post(simple_core::upload_binary).layer(DefaultBodyLimit::max(72 * 1024 * 1024)),
        )
        .route(
            "/workspace/binaries/content",
            put(simple_core::upload_binary_stream).layer(DefaultBodyLimit::disable()),
        )
        .route(
            "/workspace/binaries/{entry_ref}/content",
            get(simple_core::fetch_binary),
        );
    let account_transfers = Router::new().route(
        "/account/exports/{export_ref}/content",
        get(service::download_account_export),
    );
    let evaluation_transfers = Router::new().route(
        "/workspace/admin/eval/import",
        post(simple_core::import_evaluation).layer(DefaultBodyLimit::max(512 * 1024 * 1024)),
    );
    let mut transfers = workspace_transfers.merge(account_transfers);
    if state.config.evaluation_api_enabled {
        transfers = transfers.merge(evaluation_transfers);
    }
    let transfers = transfers
        .layer(ConcurrencyLimitLayer::new(
            state.config.max_concurrent_transfers,
        ))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            state.config.transfer_timeout,
        ));
    let protected = ordinary
        .merge(transfers)
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth::middleware,
        ));
    let web_auth_routes = Router::new()
        .route("/auth/session", get(web_auth::session))
        .route("/auth/login", post(web_auth::login))
        .route("/auth/logout", post(web_auth::logout))
        .route("/auth/forgot-password", post(web_auth::forgot_password))
        .route("/auth/reset-password", post(web_auth::reset_password))
        .layer(DefaultBodyLimit::max(16 * 1024))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            state.config.request_timeout,
        ));

    Router::new()
        .route("/health", get(service::health))
        .route(
            "/health/foreground-latency",
            get(service::foreground_latency),
        )
        .route("/ready", get(service::ready))
        .route("/openapi.json", get(service::openapi))
        .nest("/v1", web_auth_routes.merge(protected))
        .with_state(state.clone())
        .layer(
            ServiceBuilder::new()
                .layer(middleware::from_fn(request_context::middleware))
                .layer(middleware::from_fn(telemetry::http_middleware))
                .layer(TraceLayer::new_for_http())
                .layer(CompressionLayer::new())
                .layer(CatchPanicLayer::new())
                .layer(
                    CorsLayer::new()
                        .allow_origin(allowed_origins)
                        .allow_headers([
                            http::header::AUTHORIZATION,
                            http::header::CONTENT_TYPE,
                            http::header::RANGE,
                            request_id.clone(),
                            HeaderName::from_static("x-brunn-state-part-sha256"),
                            HeaderName::from_static("x-csrf-token"),
                        ])
                        .expose_headers([
                            request_id,
                            HeaderName::from_static("x-brunn-state-sha256"),
                            HeaderName::from_static("x-brunn-state-integrity"),
                            HeaderName::from_static("x-brunn-state-asset-ref"),
                            HeaderName::from_static("x-brunn-state-asset-version"),
                            http::header::ACCEPT_RANGES,
                            http::header::CONTENT_DISPOSITION,
                            http::header::CONTENT_RANGE,
                        ])
                        .allow_methods([
                            Method::GET,
                            Method::POST,
                            Method::PUT,
                            Method::PATCH,
                            Method::DELETE,
                        ])
                        .allow_credentials(true),
                ),
        )
}
