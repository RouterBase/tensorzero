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
    /// `{type:image_url}` block (remapped from `image_urls[0]`) and, when the
    /// caller sends one, a `last_frame` block (`last_frame_url`, or
    /// `image_urls[1]`).
    #[serde(rename = "minimax_h3_image_to_video")]
    MinimaxH3ImageToVideo,
    /// MiniMax-H3 reference-to-video (subject/character reference): same
    /// endpoint plus one `reference_image` `{type:image_url}` block per
    /// `image_urls` entry and one `reference_audio` `{type:audio_url}` block
    /// per `audio_urls` entry, so the referenced subjects are preserved across
    /// the whole clip. Reference videos are not forwarded: MiniMax bills them
    /// per input second, which the RouterBase hold does not price.
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
/// block, plus a `last_frame` block when an end frame is supplied. `duration` (integer seconds) and `resolution` forward verbatim;
/// RouterBase's `aspect_ratio` field is mapped to MiniMax's upstream `ratio`;
/// `generate_audio` forwards as a bool and `prompt_expansion_mode` (H3-Max)
/// as `extra.prompt_expansion_mode`.
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

    // image-to-video seeds the opening frame from `image_urls[0]`;
    // reference-to-video sends every `image_urls` entry as a subject/character
    // reference kept across the clip. Both forward each image as an
    // `image_url` content block; only the `role` differs.
    match shape {
        MinimaxRequestShape::MinimaxH3TextToVideo => {}
        MinimaxRequestShape::MinimaxH3ImageToVideo => {
            let image = input
                .get("image")
                .and_then(Value::as_str)
                .or_else(|| image_urls(input).next())
                .ok_or_else(|| missing_image("first_frame"))?;
            content.push(image_block("first_frame", image));
        }
        MinimaxRequestShape::MinimaxH3ReferenceToVideo => {
            let images: Vec<&str> = match input.get("image").and_then(Value::as_str) {
                Some(image) => vec![image],
                None => image_urls(input).collect(),
            };
            if images.is_empty() {
                return Err(missing_image("reference_image"));
            }
            for image in images {
                content.push(image_block("reference_image", image));
            }
            // Reference audio (free on MiniMax's price list). Reference videos
            // are deliberately NOT forwarded: they bill per input second.
            for audio in string_array(input, "audio_urls") {
                content.push(json!({
                    "type": "audio_url",
                    "role": "reference_audio",
                    "audio_url": { "url": audio },
                }));
            }
        }
    }

    // Image-to-video end frame. MiniMax pairs a `last_frame` block with the
    // `first_frame` above (every H3 tier: H3, Highspeed, Lite, Max), and the
    // clip's ratio then follows the images (`adaptive`). Accept the field
    // names the other video providers use, or a second `image_urls` entry.
    if matches!(shape, MinimaxRequestShape::MinimaxH3ImageToVideo)
        && let Some(last) = last_frame(input)
    {
        content.push(image_block("last_frame", last));
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

    // `generate_audio` (MiniMax default true = clip with audio, false =
    // silent). Accept a JSON bool or its string form.
    if let Some(generate_audio) = input.get("generate_audio").and_then(as_bool) {
        body.insert("generate_audio".into(), Value::from(generate_audio));
    }

    // H3-Max prompt expansion (`disabled` / `balanced` / `quality`). MiniMax
    // nests it under `extra`, which is `additionalProperties: false`, so
    // nothing else goes in there. The schema only offers it on Max rows.
    if let Some(mode) = input
        .get("prompt_expansion_mode")
        .and_then(Value::as_str)
        .filter(|mode| !mode.is_empty())
    {
        body.insert("extra".into(), json!({ "prompt_expansion_mode": mode }));
    }

    Ok(Value::Object(body))
}

/// An `image_url` content block. MiniMax's `/v2/video_generation` takes
/// `image_url` as a NESTED object `{ "url": ... }` (OpenAI-style), NOT a bare
/// string: a bare string is rejected by the strict upstream whitelist.
fn image_block(role: &str, url: &str) -> Value {
    json!({
        "type": "image_url",
        "role": role,
        "image_url": { "url": url },
    })
}

fn missing_image(role: &str) -> Error {
    Error::new(ErrorDetails::InvalidRequest {
        message: format!("MiniMax {role} video requires an image_urls[0] image URL"),
    })
}

/// Non-empty string entries of the `key` array (missing or non-array = none).
fn string_array<'a>(input: &'a Value, key: &str) -> impl Iterator<Item = &'a str> {
    input
        .get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|url| !url.is_empty())
}

fn image_urls(input: &Value) -> impl Iterator<Item = &str> {
    string_array(input, "image_urls")
}

fn as_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::String(s) => match s.trim() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// The image-to-video end frame: `last_frame_url` (the RouterBase schema
/// field), the `end_image` / `last_image` aliases, or `image_urls[1]`.
fn last_frame(input: &Value) -> Option<&str> {
    ["last_frame_url", "end_image", "last_image"]
        .iter()
        .find_map(|key| input.get(*key).and_then(Value::as_str))
        .or_else(|| {
            input
                .get("image_urls")
                .and_then(Value::as_array)
                .and_then(|arr| arr.get(1))
                .and_then(Value::as_str)
        })
        .filter(|url| !url.is_empty())
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
    fn build_body_image_to_video_adds_last_frame() {
        let input = json!({
            "prompt": "the girl grows up",
            "image_urls": ["https://cdn.example/first.png"],
            "last_frame_url": "https://cdn.example/last.png",
            "duration": 5,
        });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3ImageToVideo,
            "MiniMax-H3-Lite",
            &input,
        )
        .unwrap();
        let content = body["content"].as_array().unwrap();
        assert_eq!(content.len(), 3, "text + first_frame + last_frame blocks");
        assert_eq!(
            content[1]["role"], "first_frame",
            "image_urls[0] stays the first frame"
        );
        assert_eq!(
            content[2]["role"], "last_frame",
            "last_frame_url becomes the last frame"
        );
        assert_eq!(
            content[2]["image_url"]["url"], "https://cdn.example/last.png",
            "the last frame is a nested {{url}} object like the first"
        );

        let two_images = json!({
            "prompt": "the girl grows up",
            "image_urls": ["https://cdn.example/first.png", "https://cdn.example/second.png"],
        });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3ImageToVideo,
            "MiniMax-H3-Lite",
            &two_images,
        )
        .unwrap();
        assert_eq!(
            body["content"][2]["image_url"]["url"], "https://cdn.example/second.png",
            "a second image_urls entry is the last frame"
        );
    }

    #[test]
    fn build_body_without_end_frame_sends_first_frame_only() {
        let input = json!({ "prompt": "p", "image_urls": ["https://cdn.example/a.png"] });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3ImageToVideo,
            "MiniMax-H3",
            &input,
        )
        .unwrap();
        assert_eq!(
            body["content"].as_array().unwrap().len(),
            2,
            "no end frame supplied, so no last_frame block"
        );

        let r2v = json!({
            "prompt": "p",
            "image_urls": ["https://cdn.example/a.png"],
            "last_frame_url": "https://cdn.example/b.png",
        });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3ReferenceToVideo,
            "MiniMax-H3",
            &r2v,
        )
        .unwrap();
        assert_eq!(
            body["content"].as_array().unwrap().len(),
            2,
            "reference-to-video never sends a last_frame: MiniMax rejects mixing the two roles"
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
    fn build_body_reference_to_video_forwards_every_image_and_audio() {
        let input = json!({
            "prompt": "the two characters meet",
            "image_urls": ["https://cdn.example/a.png", "https://cdn.example/b.png"],
            "audio_urls": ["https://cdn.example/voice.mp3"],
            "video_urls": ["https://cdn.example/motion.mp4"],
        });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3ReferenceToVideo,
            "MiniMax-H3",
            &input,
        )
        .unwrap();
        let content = body["content"].as_array().unwrap();
        assert_eq!(
            content.len(),
            4,
            "text + two reference images + one reference audio; no video block"
        );
        assert_eq!(content[1]["role"], "reference_image");
        assert_eq!(
            content[2]["image_url"]["url"], "https://cdn.example/b.png",
            "every image_urls entry becomes a reference_image"
        );
        assert_eq!(content[3]["type"], "audio_url");
        assert_eq!(
            content[3]["role"], "reference_audio",
            "audio_urls entries become reference_audio blocks"
        );
        assert_eq!(
            content[3]["audio_url"]["url"], "https://cdn.example/voice.mp3",
            "audio_url is a nested {{url}} object like image_url"
        );
        assert!(
            content.iter().all(|c| c["type"] != "video_url"),
            "reference videos bill per input second upstream, so they are never forwarded"
        );
    }

    #[test]
    fn build_body_image_to_video_ignores_audio_urls() {
        let input = json!({
            "prompt": "p",
            "image_urls": ["https://cdn.example/a.png"],
            "audio_urls": ["https://cdn.example/voice.mp3"],
        });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3ImageToVideo,
            "MiniMax-H3",
            &input,
        )
        .unwrap();
        assert_eq!(
            body["content"].as_array().unwrap().len(),
            2,
            "image-to-video cannot carry reference roles: MiniMax rejects mixing them"
        );
    }

    #[test]
    fn build_body_forwards_generate_audio() {
        let silent = json!({ "prompt": "p", "generate_audio": false });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3TextToVideo,
            "MiniMax-H3",
            &silent,
        )
        .unwrap();
        assert_eq!(
            body["generate_audio"], false,
            "generate_audio=false must reach MiniMax as a JSON bool"
        );

        let stringly = json!({ "prompt": "p", "generate_audio": "true" });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3TextToVideo,
            "MiniMax-H3",
            &stringly,
        )
        .unwrap();
        assert_eq!(
            body["generate_audio"], true,
            "a string \"true\" is coerced to a bool"
        );

        let absent = json!({ "prompt": "p" });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3TextToVideo,
            "MiniMax-H3",
            &absent,
        )
        .unwrap();
        assert!(
            body.get("generate_audio").is_none(),
            "absent generate_audio is left to MiniMax's default"
        );
    }

    #[test]
    fn build_body_nests_prompt_expansion_mode_under_extra() {
        let input = json!({ "prompt": "p", "prompt_expansion_mode": "disabled" });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3TextToVideo,
            "MiniMax-H3-Max",
            &input,
        )
        .unwrap();
        assert_eq!(
            body["extra"],
            json!({ "prompt_expansion_mode": "disabled" }),
            "prompt_expansion_mode is MiniMax's extra.prompt_expansion_mode, alone in extra"
        );
        assert!(
            body.get("prompt_expansion_mode").is_none(),
            "it is not a top-level MiniMax field"
        );

        let absent = json!({ "prompt": "p" });
        let body = build_body(
            &MinimaxRequestShape::MinimaxH3TextToVideo,
            "MiniMax-H3-Max",
            &absent,
        )
        .unwrap();
        assert!(
            body.get("extra").is_none(),
            "no extra object without a mode"
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
                    assert_eq!(reason, "boom", "failure reason comes from status_msg");
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
