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

const PROVIDER_NAME: &str = "MiniMax";
const PROVIDER_TYPE: &str = "minimax";
/// Per-HTTP-request timeout for a single call to MiniMax (submit, one poll
/// fetch). Bounds a single network round-trip, not the whole generation.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
/// Total wall-clock budget for one async video generation to reach a terminal
/// state. Mirrors the Novita/Amux 1h ceiling; the RouterBase worker does not
/// retry (`MAX_ATTEMPTS = 1`), so this is the only budget.
const ASYNC_TASK_TIMEOUT: Duration = Duration::from_secs(3600);

lazy_static! {
    // `unwrap_or_else` only fires on `Err`; an env var set to an empty string
    // (e.g. docker-compose's `MINIMAX_API_BASE: ${MINIMAX_API_BASE:-}` when
    // unset) comes back as `Ok("")` and would otherwise become an empty base
    // URL. Filter empties so the public default still wins.
    static ref MINIMAX_API_BASE: String = std::env::var("MINIMAX_API_BASE")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://api.minimaxi.com".to_string());
}

pub struct MinimaxProvider;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct MinimaxMediaProxyConfig {
    /// For MiniMax this carries the upstream model id (e.g. `MiniMax-H3`),
    /// which is sent in the request body — the endpoints are fixed
    /// (`/v2/video_generation`), unlike Novita's per-model URL path.
    pub path: Arc<str>,
    #[serde(default)]
    pub async_submission: bool,
    pub request_shape: MinimaxRequestShape,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum MinimaxRequestShape {
    /// MiniMax-H3 text-to-video via `POST /v2/video_generation`: `model`,
    /// `content` (a single `{type:text}` block), `duration` (integer seconds),
    /// `resolution`, `ratio`.
    #[serde(rename = "minimax_h3_text_to_video")]
    MinimaxH3TextToVideo,
    /// MiniMax-H3 image-to-video: same endpoint plus a `first_frame`
    /// `{type:image_url}` block (remapped from `image_urls[0]`).
    #[serde(rename = "minimax_h3_image_to_video")]
    MinimaxH3ImageToVideo,
    /// MiniMax-H3 reference-to-video (subject/character reference): same
    /// endpoint plus a `reference_image` `{type:image_url}` block (remapped
    /// from `image_urls[0]`), so the referenced subject is preserved across
    /// the whole clip. Differs from image-to-video only in the block `role`.
    #[serde(rename = "minimax_h3_reference_to_video")]
    MinimaxH3ReferenceToVideo,
}

impl MinimaxProvider {
    pub async fn infer_media_proxy(
        proxy: &MinimaxMediaProxyConfig,
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
        let url = format!("{}/v2/video_generation", *MINIMAX_API_BASE)
            .parse::<Url>()
            .map_err(|e| {
                Error::new(ErrorDetails::InvalidBaseUrl {
                    message: format!("Failed to construct MiniMax URL: {e}"),
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
                    message: format!("MiniMax request failed: {e}"),
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
                message: format!("MiniMax returned {status}: {raw}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: Some(raw),
            }));
        }

        let raw_json: Value = serde_json::from_str(&raw).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Failed to parse MiniMax response: {e}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: Some(raw.clone()),
            })
        })?;

        // Submit returns `{ task_id, base_resp: { status_code, status_msg } }`.
        // A non-zero `base_resp.status_code` is a synchronous rejection even
        // though the HTTP status was 200 (MiniMax's convention).
        if let Some(code) = raw_json
            .get("base_resp")
            .and_then(|b| b.get("status_code"))
            .and_then(Value::as_i64)
            && code != 0
        {
            let msg = raw_json
                .get("base_resp")
                .and_then(|b| b.get("status_msg"))
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!("MiniMax rejected the request ({code}): {msg}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: Some(raw.clone()),
            }));
        }

        let task_id = raw_json
            .get("task_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "MiniMax submit response missing task_id".to_string(),
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
                message: "MiniMax completed but returned no video URL".to_string(),
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
    if let Some(key) = dynamic_api_keys.get("MINIMAX_API_KEY") {
        return Ok(SecretString::from(key.expose_secret().to_string()));
    }

    std::env::var("MINIMAX_API_KEY")
        .map(SecretString::from)
        .map_err(|_| {
            Error::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "MINIMAX_API_KEY is not configured".to_string(),
            })
        })
}

/// Build the `POST /v2/video_generation` body. `model` is the upstream id
/// carried in `proxy.path` (e.g. `MiniMax-H3`). The prompt becomes a single
/// `text` content block; image-to-video adds a `first_frame` `image_url`
/// block. `duration` (integer seconds) and `resolution` forward verbatim;
/// RouterBase's `aspect_ratio` field is mapped to MiniMax's upstream `ratio`.
fn build_body(shape: &MinimaxRequestShape, model: &str, input: &Value) -> Result<Value, Error> {
    let prompt = input
        .get("prompt")
        .and_then(Value::as_str)
        .filter(|prompt| !prompt.is_empty())
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: "MiniMax-backed video variants require a prompt".to_string(),
            })
        })?;

    let mut content: Vec<Value> = vec![json!({ "type": "text", "text": prompt })];

    // image-to-video seeds the opening frame; reference-to-video supplies a
    // subject/character reference kept across the clip. Both forward the input
    // image as an `image_url` content block; only the `role` differs.
    let image_role = match shape {
        MinimaxRequestShape::MinimaxH3TextToVideo => None,
        MinimaxRequestShape::MinimaxH3ImageToVideo => Some("first_frame"),
        MinimaxRequestShape::MinimaxH3ReferenceToVideo => Some("reference_image"),
    };
    if let Some(role) = image_role {
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
            .ok_or_else(|| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("MiniMax {role} video requires an image_urls[0] image URL"),
                })
            })?;
        // MiniMax's `/v2/video_generation` content blocks take `image_url` as a
        // NESTED object `{ "url": ... }` (OpenAI-style), NOT a bare string — a
        // bare string is rejected by the strict upstream parameter whitelist.
        content.push(json!({
            "type": "image_url",
            "role": role,
            "image_url": { "url": image },
        }));
    }

    let mut body = serde_json::Map::new();
    body.insert("model".into(), Value::from(model));
    body.insert("content".into(), Value::Array(content));

    // Duration → integer `duration` seconds. RouterBase ships duration as a
    // string across the video surface; MiniMax wants an integer, so coerce.
    if let Some(duration) = input.get("duration") {
        let secs = match duration {
            Value::String(s) => s.trim().parse::<i64>().ok(),
            Value::Number(n) => n.as_i64(),
            _ => None,
        };
        if let Some(s) = secs {
            body.insert("duration".into(), Value::from(s));
        }
    }

    // `resolution` forwards verbatim (parameter_schema default e.g. 2K).
    if let Some(value) = input.get("resolution").filter(|v| !v.is_null()) {
        body.insert("resolution".into(), value.clone());
    }

    // Aspect ratio: RouterBase uses the platform-wide `aspect_ratio` field name
    // (the playground only renders/sends that); MiniMax's upstream field is
    // `ratio`. Map `aspect_ratio` → `ratio`, falling back to a direct `ratio`
    // for non-playground callers. t2v requires an explicit, non-`adaptive`
    // ratio — the schema default supplies one.
    if let Some(value) = input
        .get("aspect_ratio")
        .or_else(|| input.get("ratio"))
        .filter(|v| !v.is_null())
    {
        body.insert("ratio".into(), value.clone());
    }

    Ok(Value::Object(body))
}

/// Extract the result video URL from a completed MiniMax status response.
/// The completed shape is `{ task: { status, content: { url } } }`; tolerate
/// the flat/root variants defensively.
fn parse_urls(body: &Value) -> Vec<String> {
    let task = body.get("task").unwrap_or(body);
    for container in [task, body] {
        if let Some(url) = container
            .get("content")
            .and_then(|c| c.get("url"))
            .and_then(Value::as_str)
        {
            return vec![url.to_string()];
        }
        if let Some(url) = container.get("url").and_then(Value::as_str) {
            return vec![url.to_string()];
        }
    }
    Vec::new()
}

/// Terminal classification of a single poll response.
enum PollOutcome {
    Done,
    Failed(String),
    Pending,
}

/// Classify a MiniMax poll response into a terminal/pending state.
///
/// The poll endpoint returns `{ task: { status, content, ... }, base_resp }`.
/// MiniMax reports terminal success as `"succeeded"` and terminal failure as
/// `"failed"` / `"cancelled"` / `"expired"`; anything else (e.g. `queueing`,
/// `processing`, `preparing`) is still running.
fn classify_poll(body: &Value) -> PollOutcome {
    let task = body.get("task").unwrap_or(body);
    match task.get("status").and_then(Value::as_str).unwrap_or("") {
        "succeeded" | "success" | "completed" => PollOutcome::Done,
        "failed" | "cancelled" | "canceled" | "expired" | "error" => {
            PollOutcome::Failed(extract_failure_reason(body, task))
        }
        _ => PollOutcome::Pending,
    }
}

/// Pull a human-readable failure reason out of a MiniMax poll response. Probe
/// the task-level error/status_msg, then the top-level `base_resp.status_msg`,
/// falling back only when MiniMax genuinely reports nothing.
fn extract_failure_reason(body: &Value, task: &Value) -> String {
    let non_empty = |v: &Value| {
        v.as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };

    for candidate in [
        task.get("status_msg"),
        task.get("error"),
        task.get("message"),
        body.get("base_resp").and_then(|b| b.get("status_msg")),
        body.get("message"),
    ] {
        if let Some(v) = candidate {
            if let Some(s) = non_empty(v) {
                return s;
            }
            if let Some(inner) = v.get("message").and_then(&non_empty) {
                return inner;
            }
        }
    }
    "(no reason given)".to_string()
}

async fn poll_async_result(
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    task_id: &str,
) -> Result<Value, Error> {
    let url = format!("{}/v2/query/video_generation/{task_id}", *MINIMAX_API_BASE);
    let deadline = Instant::now() + ASYNC_TASK_TIMEOUT;
    // MiniMax recommends ~10s poll intervals.
    let poll_interval = Duration::from_secs(10);

    loop {
        if Instant::now() >= deadline {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "MiniMax async task {task_id} did not complete within {}s",
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
                    message: format!("MiniMax poll request failed: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: None,
                })
            })?;

        let status = response.status();
        let body: Value = response.json().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("MiniMax poll response parse failed: {e}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;

        if !status.is_success() {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!("MiniMax poll returned {status} for task {task_id}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(body.to_string()),
            }));
        }

        match classify_poll(&body) {
            PollOutcome::Done => return Ok(body),
            PollOutcome::Failed(reason) => {
                return Err(Error::new(ErrorDetails::InferenceServer {
                    message: format!("MiniMax generation failed: {reason}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: Some(body.to_string()),
                }));
            }
            PollOutcome::Pending => {}
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_body_text_to_video_shapes_content_and_knobs() {
        let input = json!({
            "prompt": "a cat surfing",
            "duration": "6",
            "resolution": "2K",
            "aspect_ratio": "16:9",
        });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3TextToVideo,
            "MiniMax-H3",
            &input,
        )
        .unwrap();
        assert_eq!(
            body["model"], "MiniMax-H3",
            "model must carry the proxy path"
        );
        assert_eq!(
            body["content"][0]["type"], "text",
            "prompt must become a text content block"
        );
        assert_eq!(body["content"][0]["text"], "a cat surfing");
        assert_eq!(
            body["duration"], 6,
            "duration must be coerced from string to integer seconds"
        );
        assert_eq!(body["resolution"], "2K");
        assert_eq!(
            body["ratio"], "16:9",
            "RouterBase aspect_ratio must be forwarded to MiniMax as ratio"
        );
    }

    #[test]
    fn build_body_falls_back_to_direct_ratio() {
        // Non-playground callers may send `ratio` directly; still honored.
        let input = json!({ "prompt": "x", "ratio": "9:16" });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3TextToVideo,
            "MiniMax-H3",
            &input,
        )
        .unwrap();
        assert_eq!(
            body["ratio"], "9:16",
            "a direct `ratio` must still map through when aspect_ratio is absent"
        );
    }

    #[test]
    fn build_body_image_to_video_adds_first_frame() {
        let input = json!({
            "prompt": "pan across the scene",
            "image_urls": ["https://cdn.example/frame.png"],
            "duration": 4,
        });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3ImageToVideo,
            "MiniMax-H3",
            &input,
        )
        .unwrap();
        let img = &body["content"][1];
        assert_eq!(img["type"], "image_url", "i2v adds an image_url block");
        assert_eq!(
            img["role"], "first_frame",
            "the reference image is the first frame"
        );
        assert_eq!(
            img["image_url"]["url"], "https://cdn.example/frame.png",
            "image_url must be a nested {{url}} object, not a bare string"
        );
    }

    #[test]
    fn build_body_reference_to_video_adds_reference_image() {
        let input = json!({
            "prompt": "the same character walking through a market",
            "image_urls": ["https://cdn.example/subject.png"],
            "duration": 6,
        });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3ReferenceToVideo,
            "MiniMax-H3",
            &input,
        )
        .unwrap();
        let img = &body["content"][1];
        assert_eq!(img["type"], "image_url", "r2v adds an image_url block");
        assert_eq!(
            img["role"], "reference_image",
            "reference-to-video tags the image as a subject reference, not first_frame"
        );
        assert_eq!(
            img["image_url"]["url"], "https://cdn.example/subject.png",
            "image_url must be a nested {{url}} object, not a bare string"
        );
    }

    #[test]
    fn classify_poll_reads_nested_task_status() {
        let succeeded = json!({ "task": { "status": "succeeded", "content": { "url": "u" } } });
        assert!(
            matches!(classify_poll(&succeeded), PollOutcome::Done),
            "task.status=succeeded must be Done"
        );
        for st in ["queueing", "processing", "preparing"] {
            let pending = json!({ "task": { "status": st } });
            assert!(
                matches!(classify_poll(&pending), PollOutcome::Pending),
                "status={st} must keep polling"
            );
        }
        for st in ["failed", "cancelled", "expired"] {
            let failed = json!({ "task": { "status": st, "status_msg": "boom" } });
            match classify_poll(&failed) {
                PollOutcome::Failed(reason) => {
                    assert_eq!(reason, "boom", "failure reason comes from status_msg")
                }
                _ => panic!("status={st} must classify as Failed"),
            }
        }
    }

    #[test]
    fn parse_urls_reads_task_content_url() {
        let body =
            json!({ "task": { "status": "succeeded", "content": { "url": "https://cdn/x.mp4" } } });
        assert_eq!(
            parse_urls(&body),
            vec!["https://cdn/x.mp4".to_string()],
            "the completed video URL must be read from task.content.url"
        );
    }
}
