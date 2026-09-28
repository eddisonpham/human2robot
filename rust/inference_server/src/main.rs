use axum::{
    extract::Json,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Serialize, Deserialize, Clone)]
struct Manifest {
    input_name: String,
    output_name: String,
    state_dim: usize,
    action_dim: usize,
}

#[derive(Serialize, Deserialize)]
struct ActRequest {
    state: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct ActResponse {
    action: Vec<f32>,
    latency_us: u64,
}

#[derive(Serialize, Deserialize)]
struct HealthResponse {
    model_version: String,
    state_dim: usize,
    action_dim: usize,
}

struct AppState {
    /// `Session::run` takes `&mut self`, so inference is serialized behind a
    /// mutex. That is the right trade for a single policy on a control loop: a
    /// pool of sessions would multiply memory by the pool size to buy
    /// concurrency the caller does not need, since control steps are sequential.
    session: std::sync::Mutex<Session>,
    manifest: Manifest,
    model_version: String,
}

fn load_manifest(path: &PathBuf) -> Result<Manifest, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let manifest: Manifest = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

/// Reject a manifest whose dimensions cannot describe a real policy.
///
/// Checked at load rather than at the first request, so a bad manifest fails
/// at startup instead of looking like a client error later. A zero dimension
/// would otherwise produce a tensor of empty shape and a confusing runtime
/// failure rather than a clear one.
fn validate_manifest(manifest: &Manifest) -> Result<(), String> {
    if manifest.state_dim == 0 {
        return Err("manifest state_dim must be greater than zero".to_string());
    }
    if manifest.action_dim == 0 {
        return Err("manifest action_dim must be greater than zero".to_string());
    }
    if manifest.input_name.is_empty() {
        return Err("manifest input_name must not be empty".to_string());
    }
    Ok(())
}

async fn health(
    state: axum::extract::State<Arc<AppState>>,
) -> impl IntoResponse {
    let resp = HealthResponse {
        model_version: state.model_version.clone(),
        state_dim: state.manifest.state_dim,
        action_dim: state.manifest.action_dim,
    };
    (StatusCode::OK, Json(resp))
}

/// Whether a request's state vector has the length the manifest declares.
///
/// Split out from the handler so it can be tested: this is the one piece of
/// input validation the service has, and it is what stops a client of the wrong
/// dimension from producing a tensor the model was never exported for.
fn state_length_ok(received: usize, expected: usize) -> bool {
    received == expected
}

async fn act(
    state: axum::extract::State<Arc<AppState>>,
    Json(req): Json<ActRequest>,
) -> Result<(StatusCode, Json<ActResponse>), (StatusCode, String)> {
    if !state_length_ok(req.state.len(), state.manifest.state_dim) {
        return Ok((
            StatusCode::BAD_REQUEST,
            Json(ActResponse {
                action: vec![],
                latency_us: 0,
            }),
        ));
    }

    let start = Instant::now();
    let input = Tensor::from_array(([req.state.len()], req.state.clone()))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let action = {
        let mut session = state.session.lock().map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference session poisoned".to_string(),
            )
        })?;
        let outputs = session
            .run(vec![(state.manifest.input_name.as_str(), input)])
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let (_, value) = outputs
            .into_iter()
            .next()
            .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "model produced no output".to_string()))?;
        value
            .try_extract_array::<f32>()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            .iter()
            .copied()
            .collect::<Vec<f32>>()
    };

    let latency_us = start.elapsed().as_micros() as u64;
    if action.len() != state.manifest.action_dim {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "model returned {} actions, manifest declares {}",
                action.len(),
                state.manifest.action_dim
            ),
        ));
    }

    Ok((StatusCode::OK, Json(ActResponse { action, latency_us })))
}

#[tokio::main]
async fn main() {
    let manifest_path =
        std::env::var("MANIFEST_PATH").unwrap_or_else(|_| "policy.json".to_string());
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| "policy.onnx".to_string());
    let model_version =
        std::env::var("MODEL_VERSION").unwrap_or_else(|_| "unknown".to_string());

    let manifest = load_manifest(&PathBuf::from(&manifest_path)).expect("failed to load manifest");
    let session = Session::builder()
        .expect("failed to create session builder")
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| e.to_string())
        .expect("failed to set optimization level")
        .with_intra_threads(1)
        .map_err(|e| e.to_string())
        .expect("failed to set intra-op threads")
        .commit_from_file(&model_path)
        .expect("failed to load ONNX model");

    let state = Arc::new(AppState {
        session: std::sync::Mutex::new(session),
        manifest,
        model_version,
    });

    let app = Router::new()
        .route("/v1/health", get(health))
        .route("/v1/act", post(act))
        .with_state(state);

    let addr = std::env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind");
    println!("inference_server listening on {}", addr);
    axum::serve(listener, app).await.expect("server exited");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(state_dim: usize, action_dim: usize) -> Manifest {
        Manifest {
            input_name: "observations".to_string(),
            output_name: "actions".to_string(),
            state_dim,
            action_dim,
        }
    }

    #[test]
    fn a_manifest_with_the_declared_dimensions_is_accepted() {
        assert!(validate_manifest(&manifest(42, 22)).is_ok());
    }

    #[test]
    fn a_zero_dimension_manifest_is_rejected() {
        let err = validate_manifest(&manifest(0, 22)).unwrap_err();
        assert!(err.contains("state_dim"));
        let err = validate_manifest(&manifest(42, 0)).unwrap_err();
        assert!(err.contains("action_dim"));
    }

    #[test]
    fn a_manifest_without_an_input_name_is_rejected() {
        let mut m = manifest(42, 22);
        m.input_name = String::new();
        assert!(validate_manifest(&m).unwrap_err().contains("input_name"));
    }

    #[test]
    fn state_length_must_match_exactly() {
        assert!(state_length_ok(42, 42));
        assert!(!state_length_ok(41, 42));
        assert!(!state_length_ok(43, 42));
        assert!(!state_length_ok(0, 42));
    }

    #[test]
    fn a_manifest_round_trips_through_json() {
        let dir = std::env::temp_dir().join("inference_server_manifest_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("policy.json");
        std::fs::write(
            &path,
            r#"{"input_name":"observations","output_name":"actions","state_dim":42,"action_dim":22}"#,
        )
        .unwrap();
        let loaded = load_manifest(&path).unwrap();
        assert_eq!(loaded.state_dim, 42);
        assert_eq!(loaded.action_dim, 22);
        assert_eq!(loaded.input_name, "observations");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_malformed_manifest_file_is_reported_not_panicked() {
        let dir = std::env::temp_dir().join("inference_server_manifest_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.json");
        std::fs::write(&path, "{not json").unwrap();
        assert!(load_manifest(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
