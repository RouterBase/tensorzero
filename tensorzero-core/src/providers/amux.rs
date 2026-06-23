use lazy_static::lazy_static;
use schemars::JsonSchema;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;

const PROVIDER_NAME: &str = "Amux";
const PROVIDER_TYPE: &str = "amux";
/// Per-HTTP-request timeout for individual calls to Amux (submit, one poll
/// fetch). Bounds a single network round-trip, not the whole generation.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
/// Total wall-clock budget for one async video generation to reach a terminal
/// state. Mirrors the Novita provider's 1h ceiling; the RouterBase worker does
/// not retry (`MAX_ATTEMPTS = 1`), so this is the only budget.
const ASYNC_TASK_TIMEOUT: Duration = Duration::from_secs(3600);

lazy_static! {
    // `unwrap_or_else` only fires on `Err`; an env var set to an empty string
    // (e.g. docker-compose's `AMUX_API_BASE: ${AMUX_API_BASE:-}` when unset)
    // comes back as `Ok("")` and would otherwise become an empty base URL.
    // Filter empties so the public default still wins.
    static ref AMUX_API_BASE: String = std::env::var("AMUX_API_BASE")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://api.amux.ai".to_string());
}

pub struct AmuxProvider;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AmuxMediaProxyConfig {
    /// For Amux this carries the upstream model id (e.g.
    /// `doubao-seedance-2.0`), which is sent in the request body — the
    /// universal endpoints are fixed, unlike Novita's per-model URL path.
    pub path: Arc<str>,
    #[serde(default)]
    pub async_submission: bool,
    pub request_shape: AmuxRequestShape,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum AmuxRequestShape {
    /// Doubao-Seedance 2.0 text-to-video, via the universal endpoint
    /// `POST /v1/video/generations`: model, prompt, seconds (duration),
    /// metadata (ratio, resolution, seed). Used by both the standard and
    /// `-fast` model ids — they differ only by the model in `path`.
    #[serde(rename = "seedance2_text_to_video")]
    Seedance2TextToVideo,
    /// Doubao-Seedance 2.0 image-to-video: same universal endpoint plus an
    /// `image` first-frame URL (remapped from `image_urls[0]`).
    #[serde(rename = "seedance2_image_to_video")]
    Seedance2ImageToVideo,
}

impl AmuxProvider {
    pub async fn infer_media_proxy(
        proxy: &AmuxMediaProxyConfig,
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
        let url = format!("{}/v1/video/generations", *AMUX_API_BASE)
            .parse::<Url>()
            .map_err(|e| {
                Error::new(ErrorDetails::InvalidBaseUrl {
                    message: format!("Failed to construct Amux URL: {e}"),
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
                    message: format!("Amux request failed: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;

        let status = response.status();
        let raw = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!("Amux returned {status}: {raw}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: Some(raw),
            }));
        }

        let raw_json: Value = serde_json::from_str(&raw).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Failed to parse Amux response: {e}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: Some(raw.clone()),
            })
        })?;

        // Submit returns `{ id, task_id, status: "queued", ... }`. Accept
        // either `task_id` or `id`.
        let task_id = raw_json
            .get("task_id")
            .or_else(|| raw_json.get("id"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "Amux submit response missing task_id/id".to_string(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                    raw_response: Some(raw.clone()),
                })
            })?
            .to_string();

        let result_body = poll_async_result(http_client, api_key.expose_secret(), &task_id).await?;

        let urls = parse_urls(&result_body);
        if urls.is_empty() {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: "Amux completed but returned no video URL".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: Some(result_body.to_string()),
            }));
        }

        post_media_callback(http_client, callback_url, &task_id, &urls).await?;
        Ok(task_id)
    }
}

fn get_api_key(dynamic_api_keys: &InferenceCredentials) -> Result<SecretString, Error> {
    if let Some(key) = dynamic_api_keys.get("AMUX_API_KEY") {
        return Ok(SecretString::from(key.expose_secret().to_string()));
    }

    std::env::var("AMUX_API_KEY")
        .map(SecretString::from)
        .map_err(|_| {
            Error::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "AMUX_API_KEY is not configured".to_string(),
            })
        })
}

/// Build the universal `POST /v1/video/generations` body. `model` is the
/// upstream id carried in `proxy.path`. The duration/resolution/ratio/seed
/// knobs are forwarded under `metadata` (the universal endpoint's bag of
/// upstream-specific fields), while `prompt`/`image`/`seconds` are top-level.
fn build_body(shape: &AmuxRequestShape, model: &str, input: &Value) -> Result<Value, Error> {
    let prompt = input
        .get("prompt")
        .and_then(Value::as_str)
        .filter(|prompt| !prompt.is_empty())
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: "Amux-backed video variants require a prompt".to_string(),
            })
        })?;

    let mut body = serde_json::Map::new();
    body.insert("model".into(), Value::from(model));
    body.insert("prompt".into(), Value::from(prompt));

    // Duration → universal `seconds` (string, for parity with the rest of the
    // RouterBase video surface which ships duration as a string).
    if let Some(duration) = input.get("duration") {
        let seconds = match duration {
            Value::String(s) => Some(s.clone()),
            Value::Number(n) => Some(n.to_string()),
            _ => None,
        };
        if let Some(s) = seconds {
            body.insert("seconds".into(), Value::from(s));
        }
    }

    // image-to-video: forward the first-frame image URL.
    if matches!(shape, AmuxRequestShape::Seedance2ImageToVideo) {
        if let Some(value) = input.get("image").and_then(Value::as_str) {
            body.insert("image".into(), Value::from(value));
        } else if let Some(first) = input
            .get("image_urls")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(Value::as_str)
        {
            body.insert("image".into(), Value::from(first));
        }
    }

    // Upstream-specific knobs ride in `metadata`.
    let mut metadata = serde_json::Map::new();
    for key in ["ratio", "resolution", "seed", "generate_audio", "watermark"] {
        if let Some(value) = input.get(key) {
            metadata.insert(key.to_string(), value.clone());
        }
    }
    if !metadata.is_empty() {
        body.insert("metadata".into(), Value::Object(metadata));
    }

    Ok(Value::Object(body))
}

/// Extract the result video URL from a completed universal status response.
/// The completed shape is `{ status: "completed", url: "…mp4", … }`, with the
/// URL at the root `url` field; tolerate a few common nestings too.
fn parse_urls(body: &Value) -> Vec<String> {
    // Root-level `url` (the documented universal completed shape).
    if let Some(url) = body.get("url").and_then(Value::as_str) {
        return vec![url.to_string()];
    }
    // Tolerate `data.url` / arrays, mirroring the Novita parser's leniency.
    let containers = [body, body.get("data").unwrap_or(&Value::Null)];
    for container in containers {
        if let Some(url) = container.get("url").and_then(Value::as_str) {
            return vec![url.to_string()];
        }
        for key in ["video_urls", "videos", "urls"] {
            if let Some(arr) = container.get(key).and_then(Value::as_array) {
                let urls: Vec<String> = arr
                    .iter()
                    .filter_map(|item| {
                        item.as_str().map(ToString::to_string).or_else(|| {
                            item.get("url")
                                .and_then(Value::as_str)
                                .map(ToString::to_string)
                        })
                    })
                    .collect();
                if !urls.is_empty() {
                    return urls;
                }
            }
        }
    }
    Vec::new()
}

async fn poll_async_result(
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    task_id: &str,
) -> Result<Value, Error> {
    let url = format!("{}/v1/video/generations/{task_id}", *AMUX_API_BASE);
    let deadline = Instant::now() + ASYNC_TASK_TIMEOUT;
    let poll_interval = Duration::from_secs(4);

    loop {
        if Instant::now() >= deadline {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Amux async task {task_id} did not complete within {}s",
                    ASYNC_TASK_TIMEOUT.as_secs()
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            }));
        }

        let response = http_client
            .get(&url)
            .bearer_auth(api_key)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Amux poll request failed: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: None,
                })
            })?;

        let status = response.status();
        let body: Value = response.json().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Amux poll response parse failed: {e}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;

        if !status.is_success() {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!("Amux poll returned {status} for task {task_id}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(body.to_string()),
            }));
        }

        // Universal status values: queued | in_progress | completed | failed |
        // unknown.
        let status_str = body.get("status").and_then(Value::as_str).unwrap_or("");
        match status_str {
            "completed" => return Ok(body),
            "failed" => {
                let reason = body
                    .get("error")
                    .and_then(|err| err.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("(no reason given)");
                return Err(Error::new(ErrorDetails::InferenceServer {
                    message: format!("Amux generation failed: {reason}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: Some(body.to_string()),
                }));
            }
            _ => {}
        }

        tokio::time::sleep(poll_interval).await;
    }
}

async fn post_media_callback(
    http_client: &TensorzeroHttpClient,
    callback_url: &str,
    task_id: &str,
    urls: &[String],
) -> Result<(), Error> {
    let result_json = serde_json::to_string(&json!({ "resultUrls": urls })).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize callback payload: {e}"),
        })
    })?;
    let body = json!({
        "taskId": task_id,
        "task_id": task_id,
        "state": "success",
        "resultJson": result_json,
        "resultUrls": urls,
        "data": {
            "taskId": task_id,
            "task_id": task_id,
            "resultJson": result_json,
            "resultUrls": urls,
        }
    });
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
