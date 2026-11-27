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
        .unwrap_or_else(|| "https://gateway.amux.ai".to_string());
}

pub struct AmuxProvider;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AmuxMediaProxyConfig {
    /// For Amux this carries the upstream model id (e.g.
    /// `bytedance/seedance-2.0`), which is sent in the request body as
    /// `model` — the v3 tasks endpoint is fixed, unlike Novita's per-model
    /// URL path.
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
    /// Seedance text-to-video, via the v3 tasks endpoint
    /// `POST /api/v3/contents/generations/tasks`: model, a `content` array
    /// carrying the prompt, and top-level `duration`/`resolution`/`ratio`/
    /// `generate_audio`/`watermark`. Used by both the standard and `-fast`
    /// model ids — they differ only by the model in `path`.
    #[serde(rename = "seedance2_text_to_video")]
    Seedance2TextToVideo,
    /// Seedance image-to-video: same v3 endpoint plus a first-frame
    /// `{type:"image_url", role:"first_frame"}` content item (remapped from
    /// `image` / `image_urls[0]`) and, when `last_frame_url` is given, a
    /// `role:"last_frame"` item (first-last-frame mode).
    #[serde(rename = "seedance2_image_to_video")]
    Seedance2ImageToVideo,
    /// Seedance multimodal reference-to-video: same v3 endpoint, with every
    /// `image_urls` entry as a `reference_image` item, every `video_urls`
    /// entry as a `reference_video` item and every `audio_urls` entry as a
    /// `reference_audio` item. Reference mode cannot be mixed with a first
    /// frame upstream, so no `first_frame` item is ever sent here.
    #[serde(rename = "seedance2_reference_to_video")]
    Seedance2ReferenceToVideo,
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
        let url = format!("{}/api/v3/contents/generations/tasks", *AMUX_API_BASE)
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

        let usage = task_usage(&result_body);
        post_media_callback(http_client, callback_url, &task_id, &urls, usage.as_ref()).await?;
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

/// Build the `POST /api/v3/contents/generations/tasks` body (Amux's
/// ByteDance/Ark-native async video endpoint). `model` is the upstream id
/// carried in `proxy.path` (e.g. `bytedance/seedance-2.0`).
///
/// Shape (per the Amux OpenAPI): `prompt` and any input material live in one
/// `content` array — a `{type:"text", text}` item, plus, for image-to-video,
/// a `{type:"image_url", image_url:{url}, role:"first_frame"}` item. `resolution`,
/// `duration` (an INTEGER count of seconds, not the old string `seconds`),
/// `ratio`, `generate_audio`, `watermark` and (2.5 only) `output_format` are
/// top-level. `seed` is not part of this endpoint and is dropped.
///
/// The prompt is required for text-to-video only: with an image or reference
/// media "Text (optional)" is all Ark asks for, so an empty prompt sends no
/// text item. Reference-to-video also forwards `omni_reference_task_type`
/// (Seedance 2.5: `auto` / `reference` / `edit` / `extend`).
fn build_body(shape: &AmuxRequestShape, model: &str, input: &Value) -> Result<Value, Error> {
    let prompt = input
        .get("prompt")
        .and_then(Value::as_str)
        .filter(|prompt| !prompt.trim().is_empty());
    if prompt.is_none() && matches!(shape, AmuxRequestShape::Seedance2TextToVideo) {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Amux text-to-video requires a prompt".to_string(),
        }));
    }

    let mut body = serde_json::Map::new();
    body.insert("model".into(), Value::from(model));

    // `content`: the prompt as a text item (when there is one), plus the
    // first-frame image for image-to-video or the references.
    let mut content: Vec<Value> = prompt
        .map(|prompt| json!({ "type": "text", "text": prompt }))
        .into_iter()
        .collect();
    if matches!(shape, AmuxRequestShape::Seedance2ImageToVideo) {
        let first_frame = input
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
                    message: "Amux image-to-video requires a first-frame image URL".to_string(),
                })
            })?;
        content.push(serde_json::json!({
            "type": "image_url",
            "image_url": { "url": first_frame },
            "role": "first_frame",
        }));
        // Optional last frame (Ark first-last-frame mode).
        if let Some(last_frame) = input
            .get("last_frame_url")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|url| !url.is_empty())
        {
            content.push(json!({
                "type": "image_url",
                "image_url": { "url": last_frame },
                "role": "last_frame",
            }));
        }
    }
    if matches!(shape, AmuxRequestShape::Seedance2ReferenceToVideo) {
        let references = reference_items(input);
        if references.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Amux reference-to-video requires at least one reference image, video or audio URL"
                    .to_string(),
            }));
        }
        content.extend(references);
        if let Some(task_type) = input
            .get("omni_reference_task_type")
            .and_then(Value::as_str)
            .filter(|t| !t.trim().is_empty())
        {
            body.insert("omni_reference_task_type".into(), Value::from(task_type));
        }
    }
    body.insert("content".into(), Value::Array(content));

    // Duration → integer `duration` (seconds). Accept a numeric or a numeric
    // string from the RouterBase video surface, which ships duration as a
    // string; a non-numeric value is simply omitted (upstream default).
    if let Some(seconds) = input.get("duration").and_then(|d| match d {
        Value::Number(n) => n.as_i64(),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }) {
        body.insert("duration".into(), Value::from(seconds));
    }

    // Top-level upstream knobs (renamed off the old `metadata` bag). `seed`
    // is intentionally omitted — the v3 endpoint has no such field.
    // `output_format` (mp4/mov) exists only on bytedance/seedance-2.5; it is
    // forwarded when present, and the 2.0 models never receive it because
    // their RouterBase parameter_schema does not offer the field.
    for key in [
        "resolution",
        "ratio",
        "generate_audio",
        "watermark",
        "output_format",
    ] {
        if let Some(value) = input.get(key) {
            body.insert(key.to_string(), value.clone());
        }
    }

    Ok(Value::Object(body))
}

/// Every URL in `input[field]`, whether a single string or an array of them;
/// blank entries are dropped.
fn urls_in(input: &Value, field: &str) -> Vec<String> {
    match input.get(field) {
        Some(Value::String(s)) if !s.trim().is_empty() => vec![s.trim().to_string()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

/// The reference-to-video `content` items, in the order images, videos,
/// audio (the prompt addresses them as "image 1", "video 1", ...). Each uses
/// Ark's nested media shape `{type, <type>: {url}, role}`.
fn reference_items(input: &Value) -> Vec<Value> {
    let mut items = Vec::new();
    for (field, kind, role) in [
        ("image_urls", "image_url", "reference_image"),
        ("video_urls", "video_url", "reference_video"),
        ("audio_urls", "audio_url", "reference_audio"),
    ] {
        for url in urls_in(input, field) {
            let mut item = serde_json::Map::new();
            item.insert("type".into(), Value::from(kind));
            item.insert(kind.into(), json!({ "url": url }));
            item.insert("role".into(), Value::from(role));
            items.push(Value::Object(item));
        }
    }
    items
}

/// The upstream usage from a completed task (`usage.completion_tokens` is
/// what the platform bills), plus the delivered length as `output_seconds`
/// when the task reports `duration`. RouterBase settles token-billed
/// Seedance rows on it; `None` when the task reported neither.
fn task_usage(body: &Value) -> Option<Value> {
    let mut usage = body
        .get("usage")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(duration) = body
        .get("duration")
        .and_then(Value::as_f64)
        .filter(|d| d.is_finite() && *d > 0.0)
    {
        usage.insert("output_seconds".into(), json!(duration));
    }
    (!usage.is_empty()).then_some(Value::Object(usage))
}

/// Extract the result video URL from a completed task response.
/// The v3 completed shape is `{ status: "succeeded", content: { video_url },
/// … }`, so the URL lives at `content.video_url`; tolerate a few common
/// nestings (and the legacy root `url`) for defensiveness.
fn parse_urls(body: &Value) -> Vec<String> {
    // v3 completed shape: `content.video_url`.
    if let Some(url) = body
        .get("content")
        .and_then(|c| c.get("video_url"))
        .and_then(Value::as_str)
    {
        return vec![url.to_string()];
    }
    // Root-level `url` (legacy universal shape).
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

/// Terminal classification of a single poll response.
enum PollOutcome {
    Done,
    Failed(String),
    Pending,
}

/// Classify an Amux poll response into a terminal/pending state.
///
/// The v3 poll endpoint returns the task at the root:
/// `{ id, model, status, content: { video_url }, usage, error }`. `status` is
/// one of `queued`/`running`/`succeeded`/`failed`/`cancelled`/`expired`. We
/// still fall back to a `data`-nested status so a legacy/flat shape keeps
/// working, and accept `completed`/`error` spellings defensively.
fn classify_poll(body: &Value) -> PollOutcome {
    let task = body.get("data").unwrap_or(body);
    let status = task
        .get("status")
        .or_else(|| body.get("status"))
        .and_then(Value::as_str)
        .unwrap_or("");
    match status {
        "succeeded" | "completed" => PollOutcome::Done,
        "failed" | "error" | "cancelled" | "expired" => {
            PollOutcome::Failed(extract_failure_reason(body, task))
        }
        _ => PollOutcome::Pending,
    }
}

/// Pull a human-readable failure reason out of an Amux poll response.
///
/// The live envelope is `{ code, message, data: { error, status, url } }`, and
/// `data.error` is `null` while the task runs. Observed failures do NOT always
/// nest the reason under `data.error.message` (the shape the first cut assumed):
/// amux may hand back a bare string in `data.error`, an object with `reason`/
/// `code`, or carry the text in the envelope `message`. Reading only
/// `data.error.message` collapsed every one of those to "(no reason given)",
/// which is exactly what users saw. Probe each known location in turn so the
/// real reason survives, and only fall back when amux genuinely says nothing.
fn extract_failure_reason(body: &Value, task: &Value) -> String {
    let non_empty = |v: &Value| {
        v.as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };

    // 1) `data.error` as a bare string.
    if let Some(s) = task.get("error").and_then(&non_empty) {
        return s;
    }
    // 2) `data.error` as an object: message / reason / detail / code.
    if let Some(err) = task.get("error").filter(|e| e.is_object()) {
        for key in ["message", "reason", "detail", "code"] {
            if let Some(s) = err.get(key).and_then(&non_empty) {
                return s;
            }
        }
    }
    // 3) Envelope `message` (the `{ code, message }` wrapper) or `data.message`.
    for candidate in [body.get("message"), task.get("message")] {
        if let Some(s) = candidate.and_then(&non_empty) {
            return s;
        }
    }
    "(no reason given)".to_string()
}

async fn poll_async_result(
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    task_id: &str,
) -> Result<Value, Error> {
    let url = format!(
        "{}/api/v3/contents/generations/tasks/{task_id}",
        *AMUX_API_BASE
    );
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

        match classify_poll(&body) {
            PollOutcome::Done => return Ok(body),
            PollOutcome::Failed(reason) => {
                return Err(Error::new(ErrorDetails::InferenceServer {
                    message: format!("Amux generation failed: {reason}"),
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

/// The success body POSTed to RouterBase's media callback. `usage` rides at
/// the top level and under `data` when the task reported one.
fn media_callback_body(
    task_id: &str,
    urls: &[String],
    usage: Option<&Value>,
) -> Result<Value, Error> {
    let result_json = serde_json::to_string(&json!({ "resultUrls": urls })).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize callback payload: {e}"),
        })
    })?;
    let mut body = json!({
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
    if let Some(usage) = usage {
        body["usage"] = usage.clone();
        body["data"]["usage"] = usage.clone();
    }
    Ok(body)
}

async fn post_media_callback(
    http_client: &TensorzeroHttpClient,
    callback_url: &str,
    task_id: &str,
    urls: &[String],
    usage: Option<&Value>,
) -> Result<(), Error> {
    let body = media_callback_body(task_id, urls, usage)?;
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
    use serde_json::json;

    #[test]
    fn classify_poll_detects_nested_succeeded() {
        // The real Amux universal poll shape nests status under `data` and
        // reports success as "succeeded" — the bug this fixes: the old code
        // read root `status` and only matched "completed", so it never
        // terminated and timed out after the full async budget.
        let body = json!({
            "code": "success",
            "data": { "status": "succeeded", "url": "https://cdn.amux.ai/x.mp4" }
        });
        assert!(
            matches!(classify_poll(&body), PollOutcome::Done),
            "nested data.status=succeeded must classify as Done"
        );
    }

    #[test]
    fn classify_poll_detects_nested_failed_with_reason() {
        let body = json!({
            "code": "success",
            "data": { "status": "failed", "error": { "message": "render blew up" } }
        });
        match classify_poll(&body) {
            PollOutcome::Failed(reason) => assert_eq!(
                reason, "render blew up",
                "failure reason must come from data.error.message"
            ),
            _ => panic!("nested data.status=failed must classify as Failed"),
        }
    }

    #[test]
    fn classify_poll_reads_bare_string_error() {
        // Amux hands failure reasons back as a bare string in `data.error`
        // (not an object), which the message-only probe missed -> "(no reason
        // given)". The reason string must survive.
        let body = json!({
            "code": "success",
            "data": { "status": "failed", "error": "content policy violation" }
        });
        match classify_poll(&body) {
            PollOutcome::Failed(reason) => assert_eq!(
                reason, "content policy violation",
                "a bare-string data.error must be surfaced verbatim"
            ),
            _ => panic!("data.status=failed must classify as Failed"),
        }
    }

    #[test]
    fn classify_poll_falls_back_to_envelope_message() {
        // Failure with a null `data.error` but a populated envelope `message`
        // (the `{ code, message }` wrapper) must surface that message rather
        // than collapsing to "(no reason given)".
        let body = json!({
            "code": "task_failed",
            "message": "upstream capacity exceeded",
            "data": { "status": "failed", "error": null }
        });
        match classify_poll(&body) {
            PollOutcome::Failed(reason) => assert_eq!(
                reason, "upstream capacity exceeded",
                "envelope message must be used when data.error is null"
            ),
            _ => panic!("data.status=failed must classify as Failed"),
        }
    }

    #[test]
    fn classify_poll_failed_without_any_reason_falls_back() {
        // Only when amux genuinely provides nothing do we keep the sentinel.
        let body = json!({ "data": { "status": "failed", "error": null } });
        match classify_poll(&body) {
            PollOutcome::Failed(reason) => assert_eq!(
                reason, "(no reason given)",
                "absent reason everywhere must fall back to the sentinel"
            ),
            _ => panic!("data.status=failed must classify as Failed"),
        }
    }

    #[test]
    fn classify_poll_treats_in_progress_as_pending() {
        for status in ["queued", "in_progress", "unknown", ""] {
            let body = json!({ "data": { "status": status } });
            assert!(
                matches!(classify_poll(&body), PollOutcome::Pending),
                "status={status:?} must keep polling (Pending)"
            );
        }
    }

    #[test]
    fn classify_poll_tolerates_flat_shape() {
        // Defensive fallback: if Amux ever returns a flat (un-nested) body,
        // the root-level status is still honored.
        let body = json!({ "status": "completed", "url": "https://cdn.amux.ai/x.mp4" });
        assert!(
            matches!(classify_poll(&body), PollOutcome::Done),
            "flat root status=completed must classify as Done"
        );
    }

    #[test]
    fn parse_urls_reads_nested_data_url() {
        let body = json!({
            "code": "success",
            "data": { "status": "succeeded", "url": "https://cdn.amux.ai/x.mp4" }
        });
        assert_eq!(
            parse_urls(&body),
            vec!["https://cdn.amux.ai/x.mp4".to_string()],
            "the completed video URL must be extracted from data.url"
        );
    }

    // ── v3 tasks endpoint (`/api/v3/contents/generations/tasks`) shape ──

    #[test]
    fn classify_poll_detects_v3_root_succeeded() {
        // The v3 poll shape puts status and content at the root, with the
        // result at content.video_url.
        let body = json!({
            "id": "task_1", "model": "bytedance/seedance-2.0", "status": "succeeded",
            "content": { "video_url": "https://cdn.amux.ai/v3.mp4" }
        });
        assert!(
            matches!(classify_poll(&body), PollOutcome::Done),
            "v3 root status=succeeded must classify as Done"
        );
    }

    #[test]
    fn classify_poll_treats_cancelled_and_expired_as_failed() {
        for status in ["cancelled", "expired"] {
            let body = json!({ "status": status, "error": "task did not finish" });
            match classify_poll(&body) {
                PollOutcome::Failed(reason) => assert_eq!(
                    reason, "task did not finish",
                    "v3 terminal status={status:?} must be Failed with its reason"
                ),
                _ => panic!("v3 status={status:?} must classify as Failed"),
            }
        }
    }

    #[test]
    fn parse_urls_reads_v3_content_video_url() {
        let body = json!({
            "status": "succeeded",
            "content": { "video_url": "https://cdn.amux.ai/v3.mp4" }
        });
        assert_eq!(
            parse_urls(&body),
            vec!["https://cdn.amux.ai/v3.mp4".to_string()],
            "the v3 result URL must be read from content.video_url"
        );
    }

    #[test]
    fn build_body_t2v_uses_content_array_and_integer_duration() {
        let input = json!({
            "prompt": "a cat", "duration": "5", "resolution": "1080p",
            "ratio": "16:9", "seed": 42, "generate_audio": true
        });
        let body = build_body(
            &AmuxRequestShape::Seedance2TextToVideo,
            "bytedance/seedance-2.0",
            &input,
        )
        .expect("t2v body builds");
        assert_eq!(
            body["model"],
            json!("bytedance/seedance-2.0"),
            "model is the upstream id"
        );
        assert_eq!(
            body["content"],
            json!([{ "type": "text", "text": "a cat" }]),
            "prompt rides in a single text content item"
        );
        assert_eq!(
            body["duration"],
            json!(5),
            "duration is an integer, not the string \"5\""
        );
        assert_eq!(
            body["resolution"],
            json!("1080p"),
            "resolution is top-level"
        );
        assert_eq!(body["ratio"], json!("16:9"), "ratio is top-level");
        assert_eq!(
            body["generate_audio"],
            json!(true),
            "generate_audio is top-level"
        );
        assert!(
            body.get("seed").is_none(),
            "seed is dropped — the v3 endpoint has no such field"
        );
        assert!(body.get("metadata").is_none(), "no legacy metadata bag");
    }

    #[test]
    fn build_body_forwards_output_format_when_present() {
        // `output_format` is a Seedance 2.5-only knob (mp4/mov); it must ride
        // along at the top level like the other passthrough keys.
        let input = serde_json::json!({
            "prompt": "a cat",
            "resolution": "1080p",
            "output_format": "mov",
        });
        let body = build_body(
            &AmuxRequestShape::Seedance2TextToVideo,
            "bytedance/seedance-2.5",
            &input,
        )
        .expect("body builds");
        assert_eq!(body["output_format"], serde_json::json!("mov"));
        assert_eq!(body["resolution"], serde_json::json!("1080p"));
    }

    #[test]
    fn build_body_omits_output_format_when_absent() {
        let input = serde_json::json!({ "prompt": "a cat" });
        let body = build_body(
            &AmuxRequestShape::Seedance2TextToVideo,
            "bytedance/seedance-2.0",
            &input,
        )
        .expect("body builds");
        assert!(body.get("output_format").is_none());
    }

    #[test]
    fn build_body_i2v_appends_first_frame_image_item() {
        let input = json!({ "prompt": "walk", "image_urls": ["https://x/first.png"] });
        let body = build_body(
            &AmuxRequestShape::Seedance2ImageToVideo,
            "bytedance/seedance-2.0",
            &input,
        )
        .expect("i2v body builds");
        assert_eq!(
            body["content"],
            json!([
                { "type": "text", "text": "walk" },
                { "type": "image_url", "image_url": { "url": "https://x/first.png" }, "role": "first_frame" }
            ]),
            "i2v adds a first_frame image_url content item after the text"
        );
    }

    #[test]
    fn build_body_i2v_accepts_single_image_url() {
        let body = build_body(
            &AmuxRequestShape::Seedance2ImageToVideo,
            "bytedance/seedance-2.0",
            &json!({ "prompt": "walk", "image": "https://x/first.png" }),
        )
        .expect("i2v body builds from a single image URL");
        assert_eq!(
            body["content"][1],
            json!({
                "type": "image_url",
                "image_url": { "url": "https://x/first.png" },
                "role": "first_frame"
            }),
            "both image input forms must produce the nested upstream media shape"
        );
    }

    #[test]
    fn build_body_preserves_boolean_watermark_and_480p() {
        for model in ["bytedance/seedance-2.0", "bytedance/seedance-2.0-fast"] {
            for watermark in [false, true] {
                let body = build_body(
                    &AmuxRequestShape::Seedance2TextToVideo,
                    model,
                    &json!({
                        "prompt": "walk",
                        "watermark": watermark,
                        "generate_audio": false,
                        "resolution": "480p",
                        "duration": 4
                    }),
                )
                .expect("documented Seedance parameters build");
                assert_eq!(
                    body["watermark"],
                    json!(watermark),
                    "watermark stays boolean"
                );
                assert_eq!(
                    body["generate_audio"],
                    json!(false),
                    "false audio is retained"
                );
                assert_eq!(
                    body["resolution"],
                    json!("480p"),
                    "480p is forwarded unchanged"
                );
                assert_eq!(body["duration"], json!(4), "duration stays integer");
            }
        }
    }

    #[test]
    fn build_body_r2v_maps_every_reference_kind_to_its_role() {
        let input = json!({
            "prompt": "the man from image 1 walks like video 1",
            "image_urls": ["https://x/a.png", "https://x/b.png"],
            "video_urls": ["https://x/v.mp4"],
            "audio_urls": "https://x/s.mp3",
            "resolution": "720p",
            "ratio": "16:9",
            "duration": 4
        });
        let body = build_body(
            &AmuxRequestShape::Seedance2ReferenceToVideo,
            "bytedance/seedance-2.0",
            &input,
        )
        .expect("r2v body builds");
        assert_eq!(
            body["content"],
            json!([
                { "type": "text", "text": "the man from image 1 walks like video 1" },
                { "type": "image_url", "image_url": { "url": "https://x/a.png" }, "role": "reference_image" },
                { "type": "image_url", "image_url": { "url": "https://x/b.png" }, "role": "reference_image" },
                { "type": "video_url", "video_url": { "url": "https://x/v.mp4" }, "role": "reference_video" },
                { "type": "audio_url", "audio_url": { "url": "https://x/s.mp3" }, "role": "reference_audio" }
            ]),
            "images, videos and audio ride as nested reference items, in that order, never as a first frame"
        );
        assert_eq!(body["ratio"], json!("16:9"), "r2v keeps an explicit ratio");
        assert_eq!(body["duration"], json!(4), "duration stays an integer");
    }

    #[test]
    fn build_body_r2v_accepts_images_alone_and_skips_blank_urls() {
        let body = build_body(
            &AmuxRequestShape::Seedance2ReferenceToVideo,
            "bytedance/seedance-2.5",
            &json!({ "prompt": "p", "image_urls": ["https://x/a.png", " "], "video_urls": [] }),
        )
        .expect("an image-only reference request builds");
        assert_eq!(
            body["content"].as_array().map(Vec::len),
            Some(2),
            "the prompt plus the one non-blank image; blank and empty inputs add nothing"
        );
        assert_eq!(body["content"][1]["role"], json!("reference_image"));
    }

    #[test]
    fn build_body_r2v_without_references_is_rejected() {
        assert!(
            build_body(
                &AmuxRequestShape::Seedance2ReferenceToVideo,
                "bytedance/seedance-2.0",
                &json!({ "prompt": "p", "image_urls": [] }),
            )
            .is_err(),
            "reference-to-video without any reference must be a clean error"
        );
    }

    #[test]
    fn r2v_request_shape_deserializes_from_config_name() {
        let shape: AmuxRequestShape = serde_json::from_value(json!("seedance2_reference_to_video"))
            .expect("the TOML name parses");
        assert!(
            matches!(shape, AmuxRequestShape::Seedance2ReferenceToVideo),
            "`seedance2_reference_to_video` selects the reference shape"
        );
    }

    #[test]
    fn task_usage_carries_tokens_and_output_seconds() {
        let body = json!({
            "status": "succeeded",
            "duration": 4,
            "usage": { "completion_tokens": 302400, "total_tokens": 302400 },
            "content": { "video_url": "https://cdn/x.mp4" }
        });
        assert_eq!(
            task_usage(&body),
            Some(json!({ "completion_tokens": 302400, "total_tokens": 302400, "output_seconds": 4.0 })),
            "billed tokens pass through and the delivered length is added as output_seconds"
        );
        assert_eq!(
            task_usage(&json!({ "status": "succeeded" })),
            None,
            "no usage and no duration: nothing to report"
        );
    }

    #[test]
    fn callback_body_includes_usage_only_when_known() {
        let urls = vec!["https://cdn/x.mp4".to_string()];
        let usage = json!({ "completion_tokens": 10 });
        let with = media_callback_body("t1", &urls, Some(&usage)).expect("body builds");
        assert_eq!(with["usage"], usage, "usage at the top level");
        assert_eq!(with["data"]["usage"], usage, "and under data");
        let without = media_callback_body("t1", &urls, None).expect("body builds");
        assert!(
            without.get("usage").is_none() && without["data"].get("usage").is_none(),
            "no usage key when the task reported none"
        );
        assert_eq!(without["state"], json!("success"));
    }

    #[test]
    fn build_body_i2v_appends_last_frame_when_given() {
        let body = build_body(
            &AmuxRequestShape::Seedance2ImageToVideo,
            "bytedance/seedance-2.5",
            &json!({
                "prompt": "turn",
                "image_urls": ["https://x/first.png"],
                "last_frame_url": "https://x/last.png"
            }),
        )
        .expect("first-last-frame body builds");
        assert_eq!(
            body["content"],
            json!([
                { "type": "text", "text": "turn" },
                { "type": "image_url", "image_url": { "url": "https://x/first.png" }, "role": "first_frame" },
                { "type": "image_url", "image_url": { "url": "https://x/last.png" }, "role": "last_frame" }
            ]),
            "the last frame follows the first frame as a `last_frame` item"
        );
        let first_only = build_body(
            &AmuxRequestShape::Seedance2ImageToVideo,
            "bytedance/seedance-2.0",
            &json!({ "prompt": "turn", "image_urls": ["https://x/first.png"], "last_frame_url": " " }),
        )
        .expect("i2v body builds");
        assert_eq!(
            body_len(&first_only),
            2,
            "a blank last_frame_url leaves the request first-frame only, as before"
        );
    }

    fn body_len(body: &Value) -> usize {
        body["content"].as_array().map_or(0, Vec::len)
    }

    #[test]
    fn build_body_t2v_ignores_media_fields() {
        let body = build_body(
            &AmuxRequestShape::Seedance2TextToVideo,
            "bytedance/seedance-2.0",
            &json!({
                "prompt": "p",
                "image_urls": ["https://x/a.png"],
                "video_urls": ["https://x/v.mp4"],
                "last_frame_url": "https://x/l.png"
            }),
        )
        .expect("t2v body builds");
        assert_eq!(
            body["content"],
            json!([{ "type": "text", "text": "p" }]),
            "text-to-video stays text-only whatever else the input carries"
        );
    }

    #[test]
    fn build_body_i2v_without_image_is_rejected() {
        let input = json!({ "prompt": "walk" });
        assert!(
            build_body(
                &AmuxRequestShape::Seedance2ImageToVideo,
                "bytedance/seedance-2.0",
                &input
            )
            .is_err(),
            "image-to-video without a first-frame URL must be a clean error"
        );
    }

    #[test]
    fn build_body_prompt_is_optional_with_media_only() {
        for prompt in [json!(""), json!("  "), Value::Null] {
            let i2v = build_body(
                &AmuxRequestShape::Seedance2ImageToVideo,
                "bytedance/seedance-2.5",
                &json!({ "prompt": prompt, "image_urls": ["https://x/first.png"] }),
            )
            .expect("image-to-video without a prompt builds");
            assert_eq!(
                i2v["content"],
                json!([{
                    "type": "image_url", "image_url": { "url": "https://x/first.png" },
                    "role": "first_frame"
                }]),
                "no text item for {prompt}"
            );
            let r2v = build_body(
                &AmuxRequestShape::Seedance2ReferenceToVideo,
                "bytedance/seedance-2.5",
                &json!({ "prompt": prompt, "audio_urls": ["https://x/s.mp3"] }),
            )
            .expect("reference-to-video without a prompt builds");
            assert_eq!(r2v["content"][0]["role"], json!("reference_audio"));
            assert!(
                build_body(
                    &AmuxRequestShape::Seedance2TextToVideo,
                    "bytedance/seedance-2.5",
                    &json!({ "prompt": prompt }),
                )
                .is_err(),
                "text-to-video still needs its prompt ({prompt})"
            );
        }
    }

    #[test]
    fn build_body_r2v_forwards_the_omni_reference_task_type() {
        let input = json!({
            "prompt": "Replace the car in video 1 with the car in image 1",
            "image_urls": ["https://x/car.png"],
            "video_urls": ["https://x/v.mp4"],
            "omni_reference_task_type": "edit",
            "ratio": "adaptive",
            "duration": -1
        });
        let body = build_body(
            &AmuxRequestShape::Seedance2ReferenceToVideo,
            "bytedance/seedance-2.5",
            &input,
        )
        .unwrap();
        assert_eq!(body["omni_reference_task_type"], json!("edit"));
        assert_eq!(
            body["duration"],
            json!(-1),
            "-1 lets the model pick the length"
        );
        assert_eq!(body["ratio"], json!("adaptive"));
        let t2v = build_body(
            &AmuxRequestShape::Seedance2TextToVideo,
            "bytedance/seedance-2.5",
            &json!({ "prompt": "p", "omni_reference_task_type": "edit" }),
        )
        .unwrap();
        assert!(
            t2v.get("omni_reference_task_type").is_none(),
            "only reference-to-video has a task type"
        );
    }
}
