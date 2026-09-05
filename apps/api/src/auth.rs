use std::{collections::HashSet, str::FromStr, time::Instant};

use axum::{
    extract::{Request, State},
    http::{Method, header::AUTHORIZATION},
    middleware::Next,
    response::Response,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use sqlx::Row;
use uuid::Uuid;

use crate::{
    db::AppState,
    error::{ApiError, ApiResult},
    models::{Capability, CredentialId, UserId},
    web_auth,
};

#[derive(Clone, Debug, Serialize)]
pub struct AuthContext {
    pub credential_id: CredentialId,
    pub user_id: UserId,
    pub capabilities: HashSet<String>,
    pub scope_refs: Vec<String>,
    pub read_only: bool,
}

impl AuthContext {
    pub fn can(&self, capability: Capability) -> bool {
        self.capabilities.contains(capability.as_str())
    }

    pub fn require(&self, capability: Capability) -> ApiResult<()> {
        if self.can(capability) {
            Ok(())
        } else {
            metrics::counter!(
                "auth.capability_denied",
                "capability" => capability.as_str()
            )
            .increment(1);
            Err(ApiError::capability(capability.as_str()))
        }
    }

    pub fn capability_guc(&self) -> String {
        let mut values: Vec<_> = self.capabilities.iter().cloned().collect();
        values.sort();
        values.join(",")
    }

    pub fn scope_guc(&self) -> String {
        self.scope_refs.join(",")
    }
}

pub async fn middleware(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let started = Instant::now();
    // A scoped upload permission is accepted only at the byte boundary. It is
    // not an API bearer and never authorizes reads or general workspace writes.
    if let Some(token) = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("BrunnUpload "))
    {
        let path = request
            .extensions()
            .get::<axum::extract::OriginalUri>()
            .map_or(request.uri().path(), |uri| uri.0.path());
        if request.method() != Method::PUT || path != crate::binary_upload::CONTENT_ROUTE {
            return Err(ApiError::unauthenticated());
        }
        let grant = crate::binary_upload::verify(&state.config.continuation_secret, token)?;
        let auth = grant.auth();
        state.request_rate_limiter.check(auth.credential_id.0)?;
        request.extensions_mut().insert(auth);
        request.extensions_mut().insert(grant);
        return Ok(next.run(request).await);
    }
    let bearer = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .filter(|value| !value.trim().is_empty());
    let session_token = if bearer.is_none() {
        web_auth::session_token(request.headers(), &state)
    } else {
        None
    };
    if bearer.is_none() && session_token.is_none() {
        metrics::counter!("auth.attempts", "result" => "missing").increment(1);
        metrics::histogram!("auth.duration_ms", "result" => "missing")
            .record(started.elapsed().as_secs_f64() * 1_000.0);
        return Err(ApiError::unauthenticated());
    }
    state.preauth_rate_limiter.check()?;
    let (authentication, method) = if let Some(token) = bearer {
        (authenticate(&state, token).await, "bearer")
    } else if let Some(token) = session_token {
        let authentication = match web_auth::authenticate_session(&state, &token).await {
            Ok(Some(session)) => {
                if is_unsafe_method(request.method()) {
                    web_auth::require_csrf(request.headers(), &state, &token).map(|()| session.auth)
                } else {
                    Ok(session.auth)
                }
            }
            Ok(None) => Err(ApiError::unauthenticated()),
            Err(error) => Err(error),
        };
        (authentication, "session")
    } else {
        unreachable!("a bearer or session token was established above")
    };
    let auth = match authentication {
        Ok(auth) => auth,
        Err(error) => {
            metrics::counter!(
                "auth.attempts",
                "result" => "failed",
                "method" => method
            )
            .increment(1);
            metrics::histogram!("auth.duration_ms", "result" => "failed")
                .record(started.elapsed().as_secs_f64() * 1_000.0);
            return Err(error);
        }
    };
    let access = if auth.read_only {
        "read_only"
    } else {
        "read_write"
    };
    state.request_rate_limiter.check(auth.credential_id.0)?;
    metrics::counter!(
        "auth.attempts",
        "result" => "success",
        "access" => access,
        "method" => method
    )
    .increment(1);
    metrics::histogram!("auth.duration_ms", "result" => "success")
        .record(started.elapsed().as_secs_f64() * 1_000.0);
    metrics::histogram!("auth.scope_count", "access" => access)
        .record(auth.scope_refs.len() as f64);
    let activity_auth = auth.clone();
    request.extensions_mut().insert(auth);
    let response = next.run(request).await;
    if response.status().is_success() {
        state
            .usage_tracker
            .record_credential_activity(&activity_auth, "control");
    }
    Ok(response)
}

fn is_unsafe_method(method: &Method) -> bool {
    !matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS)
}

pub async fn authenticate(state: &AppState, token: &str) -> ApiResult<AuthContext> {
    let token_hash = hash_token(token);
    let row = sqlx::query(
        r#"
        SELECT credential_id, user_id, capabilities, scope_refs
        FROM brunn_auth.authenticate_credential($1)
        "#,
    )
    .bind(token_hash)
    .fetch_optional(&state.auth_pool)
    .await?
    .ok_or_else(ApiError::unauthenticated)?;

    let capabilities: Vec<String> = row.try_get("capabilities")?;
    let scope_refs: Vec<String> = row.try_get("scope_refs")?;
    let capabilities: HashSet<_> = capabilities.into_iter().collect();
    let read_only = !capabilities.contains(Capability::Save.as_str())
        && !capabilities.contains(Capability::Checkpoint.as_str());
    Ok(AuthContext {
        credential_id: CredentialId(row.try_get::<Uuid, _>("credential_id")?),
        user_id: UserId(row.try_get::<Uuid, _>("user_id")?),
        capabilities,
        scope_refs,
        read_only,
    })
}

pub fn hash_token(token: &str) -> String {
    hex::encode(Sha256::digest(token.as_bytes()))
}

pub fn parse_capability(value: &str) -> Option<Capability> {
    Capability::from_str(value).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_hash_does_not_store_bearer_secret() {
        let hash = hash_token("brunn-test-secret");
        assert_eq!(hash.len(), 64);
        assert!(!hash.contains("secret"));
    }

    #[test]
    fn csrf_is_only_needed_for_unsafe_methods() {
        assert!(!is_unsafe_method(&Method::GET));
        assert!(!is_unsafe_method(&Method::HEAD));
        assert!(is_unsafe_method(&Method::POST));
        assert!(is_unsafe_method(&Method::DELETE));
    }
}
