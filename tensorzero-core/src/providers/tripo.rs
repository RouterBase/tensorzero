//! Tripo 3D generation (`openapi.tripo3d.ai` / `openapi.tripo3d.com`).
//!
//! Submit: `POST {base}/v3/generation/{text-to-model|image-to-model}` with the
//! model version in the body (`model`, carried in `proxy.path`). Tripo answers
//! `{code: 0, data: {task_id}}`; a non-zero `code` is a rejection.
//!
//! Poll: `GET {base}/v3/tasks/{task_id}` until `data.status` is terminal. On
//! success `data.output` carries `model_url` (`.glb`, or `.fbx` for quad
//! output) and `rendered_image_url` (a PNG preview); `data.credits_consumed`
//! is what Tripo actually charged, forwarded to RouterBase as callback
//! `usage.credits` so the generation settles at the real price.
//!
//! The host differs by region (`.ai` global, `.com` mainland China) and keys
//! are bound to it, so it comes from `TRIPO_API_BASE` per deployment.

use lazy_static::lazy_static;
use schemars::JsonSchema;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::utils::spawn_ignoring_shutdown;

const PROVIDER_NAME: &str = "Tripo";
const PROVIDER_TYPE: &str = "tripo";
const API_KEY_ENV: &str = "TRIPO3D_API_KEY";
/// Per-HTTP-request timeout for one submit or poll call.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// Total wall-clock budget for a task to reach a terminal state. Same 1h
/// ceiling as the other media providers; RouterBase never resubmits.
const ASYNC_TASK_TIMEOUT: Duration = Duration::from_secs(3600);
const POLL_INTERVAL: Duration = Duration::from_secs(5);
/// Consecutive failed poll calls (network error, 5xx, unparseable body)
/// tolerated before giving up. A poll is a read: a transient blip must not
/// fail a task that Tripo goes on to finish and bill.
const MAX_POLL_ERRORS: u32 = 6;

lazy_static! {
    // Empty counts as unset: docker-compose passes `${TRIPO_API_BASE:-}`.
    static ref TRIPO_API_BASE: String = std::env::var("TRIPO_API_BASE")
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://openapi.tripo3d.ai".to_string());
}

/// Options both generation endpoints accept, forwarded verbatim when present.
const COMMON_FIELDS: &[&str] = &[
    "face_limit",
    "texture",
    "pbr",
    "texture_seed",
    "texture_quality",
    "texture_version",
    "delight",
    "geometry_quality",
    "auto_size",
    "quad",
    "smart_low_poly",
    "generate_parts",
    "compress",
    "export_orientation",
    "model_seed",
    "export_uv",
];
const TEXT_FIELDS: &[&str] = &["negative_prompt", "image_seed"];
const IMAGE_FIELDS: &[&str] = &["enable_image_autofix", "texture_alignment", "orientation"];
/// Fields Tripo types as integers; RouterBase may carry them as strings.
const INTEGER_FIELDS: &[&str] = &["face_limit", "texture_seed", "model_seed", "image_seed"];

pub struct TripoProvider;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct TripoMediaProxyConfig {
    /// The Tripo model version sent as the body `model`
    /// (e.g. `v3.1-20260211`, `P1-20260311`).
    pub path: Arc<str>,
    /// Return the `task_id` as soon as Tripo accepts the task and poll in the
    /// background. Without it the inference call blocks until the model is
    /// done (a minute or more).
    #[serde(default)]
    pub async_submission: bool,
    pub request_shape: TripoRequestShape,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum TripoRequestShape {
    /// `POST /v3/generation/text-to-model`: `prompt` plus the shared options.
    #[serde(rename = "tripo_text_to_model")]
    TripoTextToModel,
    /// `POST /v3/generation/image-to-model`: `input` (an image URL, remapped
    /// from `image_urls[0]`) plus the shared options. No prompt.
    #[serde(rename = "tripo_image_to_model")]
    TripoImageToModel,
}

impl TripoRequestShape {
    fn endpoint(&self) -> &'static str {
        match self {
            Self::TripoTextToModel => "v3/generation/text-to-model",
            Self::TripoImageToModel => "v3/generation/image-to-model",
        }
    }
}

impl TripoProvider {
    pub async fn infer_media_proxy(
        proxy: &TripoMediaProxyConfig,
        callback_url: Option<&str>,
        input: &Value,
        http_client: &TensorzeroHttpClient,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<String, Error> {
        let callback_url = callback_url.ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: "media proxy requires a callback_url".to_string(),
            })
        })?;
        let api_key = get_api_key(dynamic_api_keys)?;
        let body = build_body(&proxy.request_shape, &proxy.path, input)?;
        let raw_request = serde_json::to_string(&body).unwrap_or_default();
        let url = format!("{}/{}", *TRIPO_API_BASE, proxy.request_shape.endpoint())
            .parse::<Url>()
            .map_err(|e| {
                Error::new(ErrorDetails::InvalidBaseUrl {
                    message: format!("Failed to construct Tripo URL: {e}"),
                })
            })?;

        let response = http_client
            .post(url)
            .bearer_auth(api_key.expose_secret())
            .json(&body)
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Tripo request failed: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;

        let status = response.status();
        let raw = response.text().await.unwrap_or_default();
        let raw_json: Option<Value> = serde_json::from_str(&raw).ok();

        // Tripo reports errors in the envelope (`code != 0`), usually with a
        // matching 4xx/5xx; either signal is a rejection.
        let code = raw_json
            .as_ref()
            .and_then(|v| v.get("code"))
            .and_then(Value::as_i64);
        if !status.is_success() || code.is_some_and(|c| c != 0) {
            let reason = raw_json
                .as_ref()
                .map(envelope_message)
                .unwrap_or_else(|| format!("HTTP {status}"));
            let code_label = code.map_or_else(|| status.as_u16().to_string(), |c| c.to_string());
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!("Tripo rejected the request ({code_label}): {reason}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(raw_request),
                raw_response: Some(raw),
            }));
        }

        let task_id = raw_json
            .as_ref()
            .and_then(|v| v.pointer("/data/task_id"))
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "Tripo submit response missing task_id".to_string(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw.clone()),
                })
            })?
            .to_string();

        if proxy.async_submission {
            // Return the task id now and finish out-of-band: holding the
            // inference call open for the whole render would hold the
            // RouterBase submit worker (and, behind it, a Cloudflare-fronted
            // connection) for minutes.
            let http_client_bg = http_client.clone();
            let api_key_bg = api_key.expose_secret().to_string();
            let callback_url_bg = callback_url.to_string();
            let task_id_bg = task_id.clone();
            spawn_ignoring_shutdown(async move {
                finish_task(&http_client_bg, &api_key_bg, &callback_url_bg, &task_id_bg).await;
            });
            return Ok(task_id);
        }

        let result = poll_async_result(http_client, api_key.expose_secret(), &task_id).await?;
        let outcome = TaskOutput::from_task(&result);
        if outcome.urls.is_empty() {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: "Tripo completed but returned no model URL".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(raw_request),
                raw_response: Some(result.to_string()),
            }));
        }
        post_media_callback(http_client, callback_url, &task_id, &outcome).await?;
        Ok(task_id)
    }
}

/// Poll a submitted task to a terminal state and report it to RouterBase.
async fn finish_task(
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    callback_url: &str,
    task_id: &str,
) {
    match poll_async_result(http_client, api_key, task_id).await {
        Ok(result) => {
            let outcome = TaskOutput::from_task(&result);
            if outcome.urls.is_empty() {
                tracing::error!("Tripo task {task_id} completed but returned no model URL");
                post_media_callback_failure(
                    http_client,
                    callback_url,
                    task_id,
                    "Generation completed but returned no model",
                )
                .await;
            } else if let Err(e) =
                post_media_callback(http_client, callback_url, task_id, &outcome).await
            {
                tracing::error!("Failed to deliver success callback for Tripo task {task_id}: {e}");
            }
        }
        Err(e) => {
            tracing::error!("Tripo task {task_id} failed: {e}");
            post_media_callback_failure(http_client, callback_url, task_id, &e.to_string()).await;
        }
    }
}

fn get_api_key(dynamic_api_keys: &InferenceCredentials) -> Result<SecretString, Error> {
    if let Some(key) = dynamic_api_keys.get(API_KEY_ENV) {
        return Ok(SecretString::from(key.expose_secret().to_string()));
    }
    std::env::var(API_KEY_ENV)
        .ok()
        .filter(|k| !k.trim().is_empty())
        .map(SecretString::from)
        .ok_or_else(|| {
            Error::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: format!("{API_KEY_ENV} is not configured"),
            })
        })
}

/// Build the generation body: `model` from the proxy path, the prompt (text)
/// or the input image (image), then every allow-listed option the caller set.
/// Anything else in `input` (RouterBase bookkeeping, fields of other
/// providers) is dropped rather than tripping Tripo's validation.
fn build_body(shape: &TripoRequestShape, model: &str, input: &Value) -> Result<Value, Error> {
    let mut body = Map::new();
    body.insert("model".into(), Value::from(model));

    let task_fields: &[&str] = match shape {
        TripoRequestShape::TripoTextToModel => {
            let prompt = input
                .get("prompt")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .ok_or_else(|| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: "Tripo text-to-3D requires a prompt".to_string(),
                    })
                })?;
            body.insert("prompt".into(), Value::from(prompt));
            TEXT_FIELDS
        }
        TripoRequestShape::TripoImageToModel => {
            let image = input
                .get("image")
                .and_then(Value::as_str)
                .or_else(|| {
                    input
                        .get("image_urls")
                        .and_then(Value::as_array)
                        .and_then(|arr| arr.first())
                        .and_then(Value::as_str)
                })
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: "Tripo image-to-3D requires an image_urls[0] image URL"
                            .to_string(),
                    })
                })?;
            body.insert("input".into(), Value::from(image));
            IMAGE_FIELDS
        }
    };

    for key in COMMON_FIELDS.iter().chain(task_fields) {
        let Some(value) = input.get(*key).filter(|v| !v.is_null()) else {
            continue;
        };
        body.insert((*key).into(), coerce(key, value));
    }
    Ok(Value::Object(body))
}

/// Integer fields arriving as numeric strings become integers, and boolean
/// options arriving as `"true"`/`"false"` become booleans; everything else is
/// forwarded unchanged.
fn coerce(key: &str, value: &Value) -> Value {
    let Some(s) = value.as_str().map(str::trim) else {
        return value.clone();
    };
    if INTEGER_FIELDS.contains(&key)
        && let Ok(n) = s.parse::<i64>()
    {
        return Value::from(n);
    }
    match s {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        _ => value.clone(),
    }
}

/// The human-readable reason in a Tripo error envelope
/// (`{code, message, suggestion}`), with the suggestion appended when given.
fn envelope_message(v: &Value) -> String {
    let non_empty = |key: &str| {
        v.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
    };
    match (non_empty("message"), non_empty("suggestion")) {
        (Some(m), Some(s)) => format!("{m} ({s})"),
        (Some(m), None) => m.to_string(),
        (None, Some(s)) => s.to_string(),
        (None, None) => "(no reason given)".to_string(),
    }
}

/// What a successful task delivered.
#[derive(Debug, Default, PartialEq)]
struct TaskOutput {
    /// The model file first, then the rendered preview image when present.
    urls: Vec<String>,
    /// `credits_consumed`: what Tripo charged for the task.
    credits: Option<f64>,
}

impl TaskOutput {
    /// Read `data.output` (or a bare task object). Current v3 keys are
    /// `model_url` / `rendered_image_url`; the v2-era `pbr_model` / `model` /
    /// `base_model` / `rendered_image` (string or `{url}`) are accepted too.
    fn from_task(body: &Value) -> Self {
        let task = body.get("data").unwrap_or(body);
        let output = task.get("output").unwrap_or(&Value::Null);
        let url_at = |key: &str| {
            let v = output.get(key)?;
            v.as_str()
                .or_else(|| v.get("url").and_then(Value::as_str))
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        let model = [
            "model_url",
            "pbr_model_url",
            "pbr_model",
            "model",
            "base_model_url",
            "base_model",
        ]
        .iter()
        .find_map(|k| url_at(k));
        let preview = ["rendered_image_url", "rendered_image"]
            .iter()
            .find_map(|k| url_at(k));
        let urls = match model {
            Some(model) => std::iter::once(model).chain(preview).collect(),
            None => Vec::new(),
        };
        let credits = task.get("credits_consumed").and_then(|c| {
            c.as_f64()
                .or_else(|| c.as_str().and_then(|s| s.trim().parse().ok()))
        });
        Self { urls, credits }
    }
}

/// Terminal classification of a single poll response.
#[derive(Debug, PartialEq)]
enum PollOutcome {
    Done,
    Failed(String),
    Pending,
}

/// `queued` / `running` (and `unknown`, which Tripo says is not terminal) keep
/// polling; `success` is done; `failed` / `cancelled` and the legacy `banned` /
/// `expired` are terminal failures.
fn classify_poll(body: &Value) -> PollOutcome {
    let task = body.get("data").unwrap_or(body);
    match task.get("status").and_then(Value::as_str).unwrap_or("") {
        "success" => PollOutcome::Done,
        "failed" | "cancelled" | "canceled" | "banned" | "expired" => {
            PollOutcome::Failed(failure_reason(task))
        }
        _ => PollOutcome::Pending,
    }
}

fn failure_reason(task: &Value) -> String {
    let status = task.get("status").and_then(Value::as_str).unwrap_or("failed");
    let message = ["error_message", "message", "error"]
        .iter()
        .find_map(|k| {
            task.get(*k)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
        });
    let code = task.get("error_code").and_then(Value::as_i64);
    match (message, code) {
        (Some(m), Some(c)) => format!("{m} (error {c})"),
        (Some(m), None) => m.to_string(),
        (None, Some(c)) => format!("task {status} (error {c})"),
        (None, None) => format!("task {status}"),
    }
}

async fn poll_async_result(
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    task_id: &str,
) -> Result<Value, Error> {
    let url = format!("{}/v3/tasks/{task_id}", *TRIPO_API_BASE);
    let deadline = Instant::now() + ASYNC_TASK_TIMEOUT;
    let mut consecutive_errors = 0u32;

    loop {
        if Instant::now() >= deadline {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Tripo generation failed: task {task_id} did not complete within {}s",
                    ASYNC_TASK_TIMEOUT.as_secs()
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            }));
        }

        match poll_once(http_client, api_key, &url).await {
            Ok(body) => {
                consecutive_errors = 0;
                match classify_poll(&body) {
                    PollOutcome::Done => return Ok(body),
                    PollOutcome::Failed(reason) => {
                        return Err(Error::new(ErrorDetails::InferenceServer {
                            message: format!("Tripo generation failed: {reason}"),
                            provider_type: PROVIDER_TYPE.to_string(),
                            raw_request: None,
                            raw_response: Some(body.to_string()),
                        }));
                    }
                    PollOutcome::Pending => {}
                }
            }
            Err(PollError::Rejected(reason)) => {
                // A 4xx with an error envelope (unknown task, bad key) will
                // not fix itself.
                return Err(Error::new(ErrorDetails::InferenceServer {
                    message: format!("Tripo generation failed: {reason}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: None,
                }));
            }
            Err(PollError::Transient(reason)) => {
                consecutive_errors += 1;
                tracing::warn!(
                    "Tripo poll for task {task_id} failed ({consecutive_errors}/{MAX_POLL_ERRORS}): {reason}"
                );
                if consecutive_errors >= MAX_POLL_ERRORS {
                    return Err(Error::new(ErrorDetails::InferenceServer {
                        message: format!("Tripo generation failed: status unavailable ({reason})"),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: None,
                        raw_response: None,
                    }));
                }
            }
        }

        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

enum PollError {
    Transient(String),
    Rejected(String),
}

async fn poll_once(
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    url: &str,
) -> Result<Value, PollError> {
    let response = http_client
        .get(url)
        .bearer_auth(api_key)
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| PollError::Transient(e.to_string()))?;
    let status = response.status();
    let raw = response.text().await.unwrap_or_default();
    let body: Option<Value> = serde_json::from_str(&raw).ok();
    let code = body
        .as_ref()
        .and_then(|v| v.get("code"))
        .and_then(Value::as_i64);
    if status.is_client_error() && status.as_u16() != 429 {
        let reason = body
            .as_ref()
            .map(envelope_message)
            .unwrap_or_else(|| format!("HTTP {status}"));
        return Err(PollError::Rejected(reason));
    }
    if !status.is_success() || code.is_some_and(|c| c != 0) {
        let reason = body
            .as_ref()
            .map(envelope_message)
            .unwrap_or_else(|| format!("HTTP {status}"));
        return Err(PollError::Transient(reason));
    }
    body.ok_or_else(|| PollError::Transient("unparseable poll response".to_string()))
}

/// Success payload for RouterBase's media callback. The credits Tripo charged
/// ride as `usage.credits` (top level and under `data`, like the other
/// fields), which credit-billed models settle from.
fn media_callback_success_body(task_id: &str, result_json: &str, outcome: &TaskOutput) -> Value {
    let mut body = json!({
        "taskId": task_id,
        "task_id": task_id,
        "state": "success",
        "resultJson": result_json,
        "resultUrls": outcome.urls,
        "data": {
            "taskId": task_id,
            "task_id": task_id,
            "resultJson": result_json,
            "resultUrls": outcome.urls,
        }
    });
    if let Some(credits) = outcome.credits {
        let usage = json!({ "credits": credits });
        body["usage"] = usage.clone();
        body["data"]["usage"] = usage;
    }
    body
}

async fn post_media_callback(
    http_client: &TensorzeroHttpClient,
    callback_url: &str,
    task_id: &str,
    outcome: &TaskOutput,
) -> Result<(), Error> {
    let result_json =
        serde_json::to_string(&json!({ "resultUrls": outcome.urls })).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize callback payload: {e}"),
            })
        })?;
    let body = media_callback_success_body(task_id, &result_json, outcome);
    let response = http_client
        .post(callback_url)
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                message: format!("RouterBase media callback failed: {e}"),
                status_code: e.status(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(body.to_string()),
                raw_response: None,
            })
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let raw = response.text().await.unwrap_or_default();
        return Err(Error::new(ErrorDetails::InferenceServer {
            message: format!("RouterBase media callback returned {status}: {raw}"),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: Some(body.to_string()),
            raw_response: Some(raw),
        }));
    }
    Ok(())
}

/// Report a task that failed after its `task_id` was already returned.
/// Best-effort: runs detached, so a delivery failure can only be logged.
async fn post_media_callback_failure(
    http_client: &TensorzeroHttpClient,
    callback_url: &str,
    task_id: &str,
    error_message: &str,
) {
    let body = json!({
        "taskId": task_id,
        "task_id": task_id,
        "state": "fail",
        "errorMessage": error_message,
        "data": {
            "taskId": task_id,
            "task_id": task_id,
            "failMsg": error_message,
        }
    });
    match http_client.post(callback_url).json(&body).send().await {
        Ok(response) if response.status().is_success() => {}
        Ok(response) => {
            let status = response.status();
            let raw = response.text().await.unwrap_or_default();
            tracing::error!(
                "RouterBase media failure-callback returned {status} for task {task_id}: {raw}"
            );
        }
        Err(e) => {
            tracing::error!(
                "RouterBase media failure-callback request failed for task {task_id}: {e}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_to_model_body_carries_model_prompt_and_allowed_options() {
        let input = json!({
            "prompt": "  a cat wearing a spacesuit ",
            "negative_prompt": "blurry",
            "texture": true,
            "pbr": false,
            "texture_quality": "detailed",
            "face_limit": "8000",
            "model_seed": 42,
            "sound": false,
            "aspect_ratio": "16:9",
            "geometry_quality": null,
        });
        let body = build_body(
            &TripoRequestShape::TripoTextToModel,
            "v3.1-20260211",
            &input,
        )
        .unwrap();
        assert_eq!(body["model"], "v3.1-20260211", "model comes from the proxy path");
        assert_eq!(body["prompt"], "a cat wearing a spacesuit", "prompt is trimmed");
        assert_eq!(body["negative_prompt"], "blurry");
        assert_eq!(body["texture_quality"], "detailed");
        assert_eq!(body["pbr"], false);
        assert_eq!(body["face_limit"], 8000, "numeric strings become integers");
        assert_eq!(body["model_seed"], 42);
        for dropped in ["sound", "aspect_ratio", "geometry_quality", "input"] {
            assert!(
                body.get(dropped).is_none(),
                "`{dropped}` is not a Tripo text-to-model field (or is null) and must not be sent"
            );
        }
    }

    #[test]
    fn text_to_model_requires_a_prompt() {
        let err = build_body(
            &TripoRequestShape::TripoTextToModel,
            "P1-20260311",
            &json!({ "prompt": "   " }),
        );
        assert!(err.is_err(), "a blank prompt must be rejected before submit");
    }

    #[test]
    fn image_to_model_maps_image_urls_to_input_and_drops_prompt() {
        let input = json!({
            "prompt": "",
            "image_urls": ["https://cdn.example/chair.png"],
            "enable_image_autofix": "true",
            "orientation": "align_image",
            "negative_prompt": "ignored for image tasks",
            "quad": true,
        });
        let body = build_body(
            &TripoRequestShape::TripoImageToModel,
            "P2-20260801",
            &input,
        )
        .unwrap();
        assert_eq!(
            body["input"], "https://cdn.example/chair.png",
            "image_urls[0] must be sent as Tripo's string `input`"
        );
        assert_eq!(body["model"], "P2-20260801");
        assert_eq!(body["enable_image_autofix"], true, "boolean strings coerce");
        assert_eq!(body["orientation"], "align_image");
        assert_eq!(body["quad"], true);
        assert!(body.get("prompt").is_none(), "image-to-model takes no prompt");
        assert!(
            body.get("negative_prompt").is_none(),
            "negative_prompt belongs to text-to-model only"
        );
    }

    #[test]
    fn image_to_model_requires_an_image() {
        let err = build_body(
            &TripoRequestShape::TripoImageToModel,
            "v3.1-20260211",
            &json!({ "image_urls": [] }),
        );
        assert!(err.is_err(), "missing image must be rejected before submit");
    }

    #[test]
    fn classify_poll_maps_every_documented_status() {
        for st in ["queued", "running", "unknown", ""] {
            assert_eq!(
                classify_poll(&json!({ "code": 0, "data": { "status": st } })),
                PollOutcome::Pending,
                "status `{st}` must keep polling"
            );
        }
        assert_eq!(
            classify_poll(&json!({ "code": 0, "data": { "status": "success" } })),
            PollOutcome::Done,
            "`success` is terminal success"
        );
        for st in ["failed", "cancelled", "banned", "expired"] {
            assert!(
                matches!(
                    classify_poll(&json!({ "data": { "status": st } })),
                    PollOutcome::Failed(_)
                ),
                "status `{st}` must be a terminal failure"
            );
        }
    }

    #[test]
    fn failure_reason_prefers_error_message_with_code() {
        let task = json!({ "status": "failed", "error_code": 2010, "error_message": "image unclear" });
        assert_eq!(failure_reason(&task), "image unclear (error 2010)");
        let bare = json!({ "status": "banned" });
        assert_eq!(failure_reason(&bare), "task banned", "status names a reasonless failure");
    }

    #[test]
    fn task_output_lists_model_then_preview_and_reads_credits() {
        let body = json!({
            "code": 0,
            "data": {
                "status": "success",
                "output": {
                    "model_url": "https://cdn.tripo3d.ai/out/model_pbr.glb",
                    "rendered_image_url": "https://cdn.tripo3d.ai/out/preview.png"
                },
                "credits_consumed": 30.0
            }
        });
        let out = TaskOutput::from_task(&body);
        assert_eq!(
            out.urls,
            vec![
                "https://cdn.tripo3d.ai/out/model_pbr.glb".to_string(),
                "https://cdn.tripo3d.ai/out/preview.png".to_string()
            ],
            "the model file must come first, the preview second"
        );
        assert_eq!(out.credits, Some(30.0));
    }

    #[test]
    fn task_output_accepts_legacy_keys_and_never_returns_a_preview_alone() {
        let legacy = json!({
            "data": { "output": { "pbr_model": { "url": "https://x/m.glb" }, "rendered_image": "https://x/p.webp" } }
        });
        assert_eq!(
            TaskOutput::from_task(&legacy).urls,
            vec!["https://x/m.glb".to_string(), "https://x/p.webp".to_string()]
        );
        let preview_only = json!({ "data": { "output": { "rendered_image_url": "https://x/p.png" } } });
        assert!(
            TaskOutput::from_task(&preview_only).urls.is_empty(),
            "a preview without a model is not a delivered model"
        );
    }

    #[test]
    fn success_callback_carries_credits_as_usage() {
        let outcome = TaskOutput {
            urls: vec!["https://x/m.glb".into()],
            credits: Some(20.0),
        };
        let body = media_callback_success_body("t1", "{}", &outcome);
        assert_eq!(body["state"], "success");
        assert_eq!(body["usage"]["credits"], 20.0, "credits ride as usage.credits");
        assert_eq!(body["data"]["usage"]["credits"], 20.0);
        let none = media_callback_success_body("t1", "{}", &TaskOutput::default());
        assert!(none.get("usage").is_none(), "no credits, no usage");
    }

    #[test]
    fn envelope_message_appends_suggestion() {
        let v = json!({ "code": 2010, "message": "Insufficient credit", "suggestion": "Top up" });
        assert_eq!(envelope_message(&v), "Insufficient credit (Top up)");
        assert_eq!(envelope_message(&json!({})), "(no reason given)");
    }
}
