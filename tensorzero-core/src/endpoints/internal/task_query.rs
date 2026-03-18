//! Task status query endpoint for async media generation tasks.
//!
//! `GET /internal/task/{task_id}?function_name=xxx&variant_name=yyy`
//!
//! RouterBase calls this endpoint to poll the status of a media task that was
//! previously submitted via `POST /inference`. TensorZero looks up the variant
//! to determine the provider (currently KIE), then delegates to the provider's
//! `query_task()` method and returns a normalized status response.
//!
//! Adding support for a new async provider in the future means:
//!   1. Implement `query_task()` on that provider.
//!   2. Add a branch in the `VariantConfig` match below.

use axum::Json;
use axum::extract::{Path, Query, State};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::providers::kie::{KIECredentials, KIEProvider};
use crate::utils::gateway::{AppState, AppStateData};
use crate::variant::VariantConfig;

#[derive(Debug, Deserialize)]
pub struct TaskQueryParams {
    pub function_name: String,
    pub variant_name: String,
}

#[derive(Debug, Serialize)]
pub struct TaskQueryResponse {
    pub task_id: String,
    /// `"processing"` | `"succeed"` | `"failed"`
    pub status: String,
    pub result_urls: Vec<String>,
    pub error_message: Option<String>,
}

/// Handler for `GET /internal/task/{task_id}`
///
/// Query params:
/// - `function_name` – TensorZero function name (e.g. `video_generate`)
/// - `variant_name`  – TensorZero variant name  (e.g. `kling_2_6`)
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "task.query", skip_all, fields(task_id = %task_id))]
pub async fn query_task_handler(
    State(app_state): AppState,
    Path(task_id): Path<String>,
    Query(params): Query<TaskQueryParams>,
) -> Result<Json<TaskQueryResponse>, Error> {
    // Resolve the variant config from the loaded TensorZero config.
    let function = app_state
        .config
        .functions
        .get(&params.function_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::UnknownFunction {
                name: params.function_name.clone(),
            })
        })?;

    let variants = match function.as_ref() {
        FunctionConfig::Chat(c) => &c.variants,
        FunctionConfig::Json(j) => &j.variants,
    };

    let variant_info = variants.get(&params.variant_name).ok_or_else(|| {
        Error::new(ErrorDetails::UnknownVariant {
            name: params.variant_name.clone(),
        })
    })?;

    match &variant_info.inner {
        VariantConfig::KieMedia(_kie_config) => {
            // Resolve the KIE API key from the environment variable.
            let credentials = std::env::var("KIE_API_KEY")
                .map(|k| KIECredentials::Static(secrecy::SecretString::from(k)))
                .unwrap_or(KIECredentials::None);

            // KIEProvider model field is unused for task queries.
            let provider = KIEProvider::new(String::new(), credentials);
            let no_dynamic_keys = InferenceCredentials::default();
            let status = provider
                .query_task(&task_id, &app_state.http_client, &no_dynamic_keys)
                .await?;

            Ok(Json(TaskQueryResponse {
                task_id,
                status: status.status,
                result_urls: status.result_urls,
                error_message: status.error_message,
            }))
        }
        _ => Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Variant '{}.{}' is not an async media variant; task status polling is not supported.",
                params.function_name, params.variant_name
            ),
        })),
    }
}
