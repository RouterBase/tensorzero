//! Novita / PPIO "original-factory" video protocols: Kling (`/v3/kling/...`)
//! and Alibaba Wan 3.0 (`/v3/aliyun/...`). Same host and API key as the rest of
//! the Novita provider, but each family keeps its vendor's own request and task
//! shapes instead of Novita's flat `/v3/async/<model>` bodies:
//!
//! - Kling: `POST /v3/kling/{text-to-video|image-to-video|omni-video|motion-control}/{model}`
//!   with `prompt` / `contents[]` + `settings{}`; answers `{code: 0, data: {id}}`.
//!   Poll `GET /v3/kling/tasks?task_ids={id}` until `data[0].status` is
//!   terminal; a finished task lists `outputs[]` (`type: video`, `url`,
//!   `duration`) and, when the request carried video, `usage.input_duration`.
//! - Wan: `POST /v3/aliyun/api/v1/services/aigc/video-generation/video-synthesis`
//!   with `{model, input: {prompt, media[]}, parameters{}}`; answers
//!   `{output: {task_id}}`. Poll `GET /v3/aliyun/api/v1/tasks/{id}` until
//!   `output.task_status` is terminal; success carries `output.video_url` and
//!   `usage.{duration, input_video_duration, output_video_duration}`.
//!
//! Neither protocol accepts an upstream callback for these models, so the
//! gateway polls (detached, like the other async media) and reports the
//! result to RouterBase's media callback. Both bill by the seconds actually
//! produced (and, with video input, read), which ride on the callback as
//! `usage.{input_seconds, output_seconds, seconds}` so the generation settles
//! at the real duration instead of the submit-time hold.
//!
//! RouterBase keeps one flat parameter surface for every video model
//! (`image_urls`, `video_urls`, `duration`, `resolution`, ...); the builders
//! below map it onto each vendor shape.

use serde_json::{Map, Value, json};
use std::time::Duration;
use tokio::time::Instant;
use url::Url;

use super::{
    ASYNC_TASK_TIMEOUT, NovitaRequestShape, PROVIDER_TYPE, REQUEST_TIMEOUT, post_media_callback,
    post_media_callback_failure,
};
use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::utils::spawn_ignoring_shutdown;

/// Body `model` when the variant names none.
const WAN_DEFAULT_MODEL: &str = "wan3.0-video";
const POLL_INTERVAL: Duration = Duration::from_secs(10);
/// Consecutive failed poll calls tolerated before giving up. A poll is a read:
/// a transient blip must not fail a task the upstream goes on to finish and
/// bill.
const MAX_POLL_ERRORS: u32 = 6;
/// Kling 4.0 accepts at most 10 images per request (keyframes or references).
const KLING_V4_MAX_IMAGES: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Protocol {
    Kling,
    Wan,
}

/// The original-factory protocol a shape speaks, or `None` for the classic
/// Novita `/v3/...` shapes.
pub(super) fn protocol(shape: &NovitaRequestShape) -> Option<Protocol> {
    match shape {
        NovitaRequestShape::KlingV34kTextToVideo
        | NovitaRequestShape::KlingV34kImageToVideo
        | NovitaRequestShape::KlingV3MotionControl
        | NovitaRequestShape::KlingV4TextToVideo
        | NovitaRequestShape::KlingV4ImageToVideo
        | NovitaRequestShape::KlingV4OmniVideo => Some(Protocol::Kling),
        NovitaRequestShape::Wan3TextToVideo
        | NovitaRequestShape::Wan3ImageToVideo
        | NovitaRequestShape::Wan3ReferenceToVideo => Some(Protocol::Wan),
        _ => None,
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorDetails::InvalidRequest {
        message: message.into(),
    })
}

fn str_field<'a>(input: &'a Value, key: &str) -> Option<&'a str> {
    input
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// Every non-empty string in an array field.
fn url_list(input: &Value, key: &str) -> Vec<String> {
    input
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// The first image: `image_urls[0]`, else a bare `image`.
fn first_image(input: &Value) -> Option<String> {
    url_list(input, "image_urls")
        .into_iter()
        .next()
        .or_else(|| str_field(input, "image").map(ToString::to_string))
}

/// The closing frame, under any of the names RouterBase schemas use for it.
fn last_frame(input: &Value) -> Option<String> {
    ["end_image", "last_frame_url", "last_image"]
        .iter()
        .find_map(|k| str_field(input, k).map(ToString::to_string))
}

/// An integer option sent as a number or a numeric string (the pricing layer
/// keys some on strings, e.g. `duration: "5"`).
fn int_field(input: &Value, key: &str) -> Option<i64> {
    match input.get(key)? {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn bool_field(input: &Value, key: &str) -> Option<bool> {
    match input.get(key)? {
        Value::Bool(b) => Some(*b),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn prompt_item(prompt: Option<&str>) -> Option<Value> {
    prompt.map(|text| json!({ "type": "prompt", "text": text }))
}

/// Build the upstream body for an original-factory shape. `model` is the
/// variant's configured upstream model (Wan's body `model`).
pub(super) fn build_body(
    shape: &NovitaRequestShape,
    model: Option<&str>,
    input: &Value,
) -> Result<Value, Error> {
    let prompt = str_field(input, "prompt");
    match shape {
        NovitaRequestShape::KlingV34kTextToVideo => {
            let prompt = prompt.ok_or_else(|| invalid("Kling text-to-video requires a prompt"))?;
            let mut settings = kling_v3_settings(input);
            settings.insert("resolution".into(), json!("4k"));
            if let Some(ar) = str_field(input, "aspect_ratio") {
                settings.insert("aspect_ratio".into(), json!(ar));
            }
            Ok(json!({ "prompt": prompt, "settings": settings }))
        }
        NovitaRequestShape::KlingV34kImageToVideo => {
            let first = first_image(input)
                .ok_or_else(|| invalid("Kling image-to-video requires `image_urls`"))?;
            let mut contents: Vec<Value> = prompt_item(prompt).into_iter().collect();
            contents.push(json!({ "type": "first_frame", "url": first }));
            if let Some(last) = last_frame(input) {
                contents.push(json!({ "type": "last_frame", "url": last }));
            }
            let mut settings = kling_v3_settings(input);
            settings.insert("resolution".into(), json!("4k"));
            Ok(json!({ "contents": contents, "settings": settings }))
        }
        NovitaRequestShape::KlingV3MotionControl => {
            let image = first_image(input)
                .ok_or_else(|| invalid("Kling motion control requires `image_urls`"))?;
            let video = url_list(input, "video_urls")
                .into_iter()
                .next()
                .or_else(|| str_field(input, "video").map(ToString::to_string))
                .ok_or_else(|| invalid("Kling motion control requires `video_urls`"))?;
            let orientation = str_field(input, "character_orientation").unwrap_or("image");
            let mut contents: Vec<Value> = prompt_item(prompt).into_iter().collect();
            contents.push(json!({ "type": "image", "url": image }));
            contents.push(json!({ "type": "video", "url": video }));
            // The old protocol picked the tier by `model_name` (`-pro` = the
            // 1080p tier, `-std` = 720p); an explicit `resolution` wins.
            let resolution = str_field(input, "resolution")
                .map(str::to_ascii_lowercase)
                .unwrap_or_else(|| {
                    if str_field(input, "model_name").is_some_and(|m| m.ends_with("-std")) {
                        "720p".to_string()
                    } else {
                        "1080p".to_string()
                    }
                });
            let audio = if bool_field(input, "keep_original_sound").unwrap_or(true) {
                "original"
            } else {
                "off"
            };
            Ok(json!({
                "contents": contents,
                "settings": {
                    "resolution": resolution,
                    "audio": audio,
                    "character_orientation": orientation,
                },
            }))
        }
        NovitaRequestShape::KlingV4TextToVideo => {
            let prompt = prompt.ok_or_else(|| invalid("Kling text-to-video requires a prompt"))?;
            let mut settings = kling_v4_settings(input);
            if let Some(ar) = str_field(input, "aspect_ratio") {
                settings.insert("aspect_ratio".into(), json!(ar));
            }
            Ok(json!({ "prompt": prompt, "settings": settings }))
        }
        NovitaRequestShape::KlingV4ImageToVideo => {
            let first = first_image(input)
                .ok_or_else(|| invalid("Kling image-to-video requires `image_urls`"))?;
            let middles = url_list(input, "middle_image_urls");
            let last = last_frame(input);
            let images = 1 + middles.len() + usize::from(last.is_some());
            if images > KLING_V4_MAX_IMAGES {
                return Err(invalid(format!(
                    "Kling image-to-video takes at most {KLING_V4_MAX_IMAGES} keyframes, got {images}"
                )));
            }
            let mut contents: Vec<Value> = prompt_item(prompt).into_iter().collect();
            contents.push(json!({ "type": "first_frame", "url": first, "id": "first" }));
            for (i, url) in middles.iter().enumerate() {
                contents.push(
                    json!({ "type": "middle_frame", "url": url, "id": format!("middle{}", i + 1) }),
                );
            }
            if let Some(last) = last {
                contents.push(json!({ "type": "last_frame", "url": last, "id": "last" }));
            }
            Ok(json!({ "contents": contents, "settings": kling_v4_settings(input) }))
        }
        NovitaRequestShape::KlingV4OmniVideo => {
            let prompt = prompt.ok_or_else(|| invalid("Kling omni video requires a prompt"))?;
            let images = url_list(input, "image_urls");
            if images.len() > KLING_V4_MAX_IMAGES {
                return Err(invalid(format!(
                    "Kling omni video takes at most {KLING_V4_MAX_IMAGES} reference images, got {}",
                    images.len()
                )));
            }
            let mut contents = vec![json!({ "type": "prompt", "text": prompt })];
            for (i, url) in images.iter().enumerate() {
                contents.push(
                    json!({ "type": "refer_image", "url": url, "id": format!("image{}", i + 1) }),
                );
            }
            for (i, url) in url_list(input, "video_urls").iter().enumerate() {
                contents.push(
                    json!({ "type": "feature_video", "url": url, "id": format!("video{}", i + 1) }),
                );
            }
            if let Some(base) = str_field(input, "base_video_url") {
                contents.push(json!({ "type": "base_video", "url": base, "id": "base" }));
            }
            let mut settings = kling_v4_settings(input);
            if let Some(ar) = str_field(input, "aspect_ratio") {
                settings.insert("aspect_ratio".into(), json!(ar));
            }
            Ok(json!({ "contents": contents, "settings": settings }))
        }
        NovitaRequestShape::Wan3TextToVideo
        | NovitaRequestShape::Wan3ImageToVideo
        | NovitaRequestShape::Wan3ReferenceToVideo => build_wan_body(shape, model, prompt, input),
        _ => Err(invalid("not an original-factory video shape")),
    }
}

/// Kling 3.0 t2v / i2v settings (the caller adds `resolution`). The old
/// protocol rendered one shot and took a `sound` flag; `multi_shot` defaults
/// to `true` upstream, so it is sent explicitly to keep single-shot output
/// unless the caller asks for multi-shot.
fn kling_v3_settings(input: &Value) -> Map<String, Value> {
    let mut settings = Map::new();
    if let Some(duration) = int_field(input, "duration") {
        settings.insert("duration".into(), json!(duration));
    }
    let sound = bool_field(input, "sound")
        .or_else(|| bool_field(input, "generate_audio"))
        .unwrap_or(false);
    settings.insert("audio".into(), json!(if sound { "native" } else { "off" }));
    settings.insert(
        "multi_shot".into(),
        json!(bool_field(input, "multi_shot").unwrap_or(false)),
    );
    settings
}

/// Kling 4.0 settings shared by the three tasks. `duration` is only sent when
/// given: omitted, Kling reads the length from the prompt (or the base
/// video), and billing follows what it produced.
fn kling_v4_settings(input: &Value) -> Map<String, Value> {
    let mut settings = Map::new();
    if let Some(resolution) = str_field(input, "resolution") {
        settings.insert("resolution".into(), json!(resolution.to_ascii_lowercase()));
    }
    if let Some(duration) = int_field(input, "duration") {
        settings.insert("duration".into(), json!(duration));
    }
    settings
}

fn build_wan_body(
    shape: &NovitaRequestShape,
    model: Option<&str>,
    prompt: Option<&str>,
    input: &Value,
) -> Result<Value, Error> {
    let mut media: Vec<Value> = Vec::new();
    match shape {
        NovitaRequestShape::Wan3TextToVideo => {}
        NovitaRequestShape::Wan3ImageToVideo => {
            let first = first_image(input)
                .ok_or_else(|| invalid("Wan image-to-video requires `image_urls`"))?;
            media.push(json!({ "type": "first_frame", "url": first }));
            if let Some(last) = last_frame(input) {
                media.push(json!({ "type": "last_frame", "url": last }));
            }
        }
        _ => {
            for url in url_list(input, "image_urls") {
                media.push(json!({ "type": "reference_image", "url": url }));
            }
            for url in url_list(input, "video_urls") {
                media.push(json!({ "type": "reference_video", "url": url }));
            }
            for url in url_list(input, "audio_urls") {
                media.push(json!({ "type": "reference_audio", "url": url }));
            }
            if media.is_empty() {
                return Err(invalid(
                    "Wan reference-to-video requires at least one of `image_urls`, `video_urls` or `audio_urls`",
                ));
            }
        }
    }
    if prompt.is_none() && media.is_empty() {
        return Err(invalid("Wan text-to-video requires a prompt"));
    }

    let mut wan_input = Map::new();
    if let Some(prompt) = prompt {
        wan_input.insert("prompt".into(), json!(prompt));
    }
    if !media.is_empty() {
        wan_input.insert("media".into(), Value::Array(media));
    }

    let mut parameters = Map::new();
    if let Some(resolution) = str_field(input, "resolution") {
        parameters.insert("resolution".into(), json!(resolution.to_ascii_uppercase()));
    }
    if let Some(ratio) = str_field(input, "ratio").or_else(|| str_field(input, "aspect_ratio")) {
        parameters.insert("ratio".into(), json!(ratio));
    }
    for key in ["duration", "seed"] {
        if let Some(v) = int_field(input, key) {
            parameters.insert(key.into(), json!(v));
        }
    }
    for key in ["audio", "prompt_extend", "watermark"] {
        if let Some(v) = bool_field(input, key) {
            parameters.insert(key.into(), json!(v));
        }
    }

    Ok(json!({
        "model": model.filter(|m| !m.is_empty()).unwrap_or(WAN_DEFAULT_MODEL),
        "input": wan_input,
        "parameters": parameters,
    }))
}

/// The upstream's rejection message for a create call, or `None` when it
/// accepted. Kling answers HTTP 200 + `code != 0` for some rejections; Wan
/// uses the HTTP status with a DashScope envelope (`code` is a string there).
fn submit_rejection(protocol: Protocol, status: u16, raw: Option<&Value>) -> Option<String> {
    let kling_code = raw
        .and_then(|v| v.get("code"))
        .and_then(Value::as_i64)
        .filter(|c| *c != 0);
    let rejected =
        !(200..300).contains(&status) || (protocol == Protocol::Kling && kling_code.is_some());
    if !rejected {
        return None;
    }
    let message = raw
        .and_then(|v| v.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("(no message)");
    let code = raw.and_then(|v| v.get("code")).map(|c| match c {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    });
    Some(match code {
        Some(code) => format!("{code}: {message}"),
        None => message.to_string(),
    })
}

fn submitted_task_id(protocol: Protocol, raw: &Value) -> Option<String> {
    let pointer = match protocol {
        Protocol::Kling => "/data/id",
        Protocol::Wan => "/output/task_id",
    };
    raw.pointer(pointer)
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(ToString::to_string)
}

fn poll_url(protocol: Protocol, base: &str, task_id: &str) -> String {
    match protocol {
        Protocol::Kling => format!("{base}/v3/kling/tasks?task_ids={task_id}"),
        Protocol::Wan => format!("{base}/v3/aliyun/api/v1/tasks/{task_id}"),
    }
}

/// A finished task: the video URLs and the billable usage in seconds.
#[derive(Debug, Default, PartialEq)]
pub(super) struct TaskOutput {
    pub urls: Vec<String>,
    pub usage: Option<Value>,
}

#[derive(Debug, PartialEq)]
enum PollOutcome {
    Done(TaskOutput),
    Failed(String),
    Pending,
}

/// A seconds figure sent as a number or a decimal string (`"10.041"`).
fn seconds(v: Option<&Value>) -> Option<f64> {
    let s = match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }?;
    (s.is_finite() && s >= 0.0).then_some(s)
}

fn usage_value(input: Option<f64>, output: Option<f64>, total: Option<f64>) -> Option<Value> {
    let mut usage = Map::new();
    if let Some(v) = input {
        usage.insert("input_seconds".into(), json!(v));
    }
    if let Some(v) = output {
        usage.insert("output_seconds".into(), json!(v));
    }
    if let Some(v) = total {
        usage.insert("seconds".into(), json!(v));
    }
    (!usage.is_empty()).then_some(Value::Object(usage))
}

fn classify_poll(protocol: Protocol, body: &Value) -> PollOutcome {
    match protocol {
        Protocol::Kling => classify_kling(body),
        Protocol::Wan => classify_wan(body),
    }
}

/// `submitted` / `processing` keep polling (Kling also holds a finished task
/// at `processing`, without outputs, until the platform has settled it).
fn classify_kling(body: &Value) -> PollOutcome {
    let Some(task) = body
        .get("data")
        .and_then(Value::as_array)
        .and_then(|tasks| tasks.first())
    else {
        return PollOutcome::Pending;
    };
    let status = task
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    match status.as_str() {
        "succeeded" | "succeed" | "success" | "completed" => {
            let videos: Vec<&Value> = task
                .get("outputs")
                .and_then(Value::as_array)
                .map(|outputs| {
                    outputs
                        .iter()
                        .filter(|o| {
                            o.get("type").and_then(Value::as_str).unwrap_or("video") == "video"
                        })
                        .collect()
                })
                .unwrap_or_default();
            let urls: Vec<String> = videos
                .iter()
                .filter_map(|o| o.get("url").and_then(Value::as_str))
                .filter(|u| !u.is_empty())
                .map(ToString::to_string)
                .collect();
            let durations: Vec<f64> = videos
                .iter()
                .filter_map(|o| seconds(o.get("duration")))
                .collect();
            let output = (!durations.is_empty()).then(|| durations.iter().sum::<f64>());
            let input = seconds(task.get("usage").and_then(|u| u.get("input_duration")));
            PollOutcome::Done(TaskOutput {
                urls,
                usage: usage_value(input, output, None),
            })
        }
        "failed" | "fail" | "cancelled" | "canceled" => {
            let reason = task
                .get("message")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|m| !m.is_empty())
                .unwrap_or("task failed");
            PollOutcome::Failed(reason.to_string())
        }
        _ => PollOutcome::Pending,
    }
}

/// `PENDING` / `RUNNING` keep polling; `SUCCEEDED` is done; `FAILED`,
/// `CANCELED` and `UNKNOWN` (expired or not found) are terminal.
fn classify_wan(body: &Value) -> PollOutcome {
    let output = body.get("output").unwrap_or(&Value::Null);
    let status = output
        .get("task_status")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_uppercase();
    match status.as_str() {
        "SUCCEEDED" => {
            let urls = output
                .get("video_url")
                .and_then(Value::as_str)
                .filter(|u| !u.is_empty())
                .map(|u| vec![u.to_string()])
                .unwrap_or_default();
            let usage = body.get("usage");
            PollOutcome::Done(TaskOutput {
                urls,
                usage: usage_value(
                    seconds(usage.and_then(|u| u.get("input_video_duration"))),
                    seconds(usage.and_then(|u| u.get("output_video_duration"))),
                    seconds(usage.and_then(|u| u.get("duration"))),
                ),
            })
        }
        "FAILED" | "CANCELED" | "CANCELLED" | "UNKNOWN" => {
            let message = output
                .get("message")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|m| !m.is_empty());
            let code = output.get("code").and_then(Value::as_str);
            PollOutcome::Failed(match (code, message) {
                (Some(c), Some(m)) => format!("{c}: {m}"),
                (None, Some(m)) => m.to_string(),
                (Some(c), None) => format!("task {status} ({c})"),
                (None, None) => format!("task {status}"),
            })
        }
        _ => PollOutcome::Pending,
    }
}

/// Submit an original-factory video task, then poll and report it in the
/// background. Returns the upstream task id. `path` is the variant's create
/// route under `/v3/` (Wan's is always
/// `aliyun/api/v1/services/aigc/video-generation/video-synthesis`; the task
/// mode follows `input.media`).
#[expect(clippy::too_many_arguments)]
pub(super) async fn infer(
    shape: &NovitaRequestShape,
    model: Option<&str>,
    path: &str,
    base: &str,
    callback_url: &str,
    input: &Value,
    http_client: &TensorzeroHttpClient,
    api_key: &str,
) -> Result<String, Error> {
    let protocol = protocol(shape).ok_or_else(|| invalid("not an original-factory video shape"))?;
    let body = build_body(shape, model, input)?;
    let raw_request = serde_json::to_string(&body).unwrap_or_default();
    let url = format!("{base}/v3/{}", path.trim_start_matches('/'))
        .parse::<Url>()
        .map_err(|e| {
            Error::new(ErrorDetails::InvalidBaseUrl {
                message: format!("Failed to construct Novita URL: {e}"),
            })
        })?;

    let response = http_client
        .post(url)
        .bearer_auth(api_key)
        .json(&body)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                message: format!("Novita request failed: {e}"),
                status_code: e.status(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: None,
            })
        })?;
    let status = response.status();
    let raw = response.text().await.unwrap_or_default();
    let raw_json: Option<Value> = serde_json::from_str(&raw).ok();

    if let Some(reason) = submit_rejection(protocol, status.as_u16(), raw_json.as_ref()) {
        return Err(Error::new(ErrorDetails::InferenceServer {
            message: format!("Novita returned {status}: {reason}"),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: Some(raw_request),
            raw_response: Some(raw),
        }));
    }
    let task_id = raw_json
        .as_ref()
        .and_then(|v| submitted_task_id(protocol, v))
        .ok_or_else(|| {
            Error::new(ErrorDetails::InferenceServer {
                message: "Novita submit response missing the task id".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw.clone()),
            })
        })?;

    // Return the task id now and finish out-of-band, as the other async media
    // do: holding the submit open for the whole render trips the origin
    // timeout in front of RouterBase.
    let http_client_bg = http_client.clone();
    let api_key_bg = api_key.to_string();
    let callback_url_bg = callback_url.to_string();
    let poll_url_bg = poll_url(protocol, base, &task_id);
    let task_id_bg = task_id.clone();
    spawn_ignoring_shutdown(async move {
        finish_task(
            protocol,
            &http_client_bg,
            &api_key_bg,
            &poll_url_bg,
            &callback_url_bg,
            &task_id_bg,
        )
        .await;
    });
    Ok(task_id)
}

async fn finish_task(
    protocol: Protocol,
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    poll_url: &str,
    callback_url: &str,
    task_id: &str,
) {
    match poll_until_done(protocol, http_client, api_key, poll_url, task_id).await {
        Ok(output) if output.urls.is_empty() => {
            tracing::error!("Novita task {task_id} completed but returned no video URL");
            post_media_callback_failure(
                http_client,
                callback_url,
                task_id,
                "Generation completed but returned no media",
            )
            .await;
        }
        Ok(output) => {
            if let Err(e) = post_media_callback(
                http_client,
                callback_url,
                task_id,
                &output.urls,
                output.usage.as_ref(),
            )
            .await
            {
                tracing::error!(
                    "Failed to deliver success callback for Novita task {task_id}: {e}"
                );
            }
        }
        Err(e) => {
            tracing::error!("Novita task {task_id} failed: {e}");
            post_media_callback_failure(http_client, callback_url, task_id, &e.to_string()).await;
        }
    }
}

enum PollError {
    Transient(String),
    Rejected(String),
}

async fn poll_until_done(
    protocol: Protocol,
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    url: &str,
    task_id: &str,
) -> Result<TaskOutput, Error> {
    let deadline = Instant::now() + ASYNC_TASK_TIMEOUT;
    let mut consecutive_errors = 0u32;
    let failed = |message: String, raw: Option<String>| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!("Novita generation failed: {message}"),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: raw,
        })
    };
    loop {
        if Instant::now() >= deadline {
            return Err(failed(
                format!(
                    "task {task_id} did not complete within {}s",
                    ASYNC_TASK_TIMEOUT.as_secs()
                ),
                None,
            ));
        }
        match poll_once(protocol, http_client, api_key, url).await {
            Ok(body) => {
                consecutive_errors = 0;
                match classify_poll(protocol, &body) {
                    PollOutcome::Done(output) => return Ok(output),
                    PollOutcome::Failed(reason) => {
                        return Err(failed(reason, Some(body.to_string())));
                    }
                    PollOutcome::Pending => {}
                }
            }
            // An unknown task or a bad key will not fix itself.
            Err(PollError::Rejected(reason)) => return Err(failed(reason, None)),
            Err(PollError::Transient(reason)) => {
                consecutive_errors += 1;
                tracing::warn!(
                    "Novita poll for task {task_id} failed ({consecutive_errors}/{MAX_POLL_ERRORS}): {reason}"
                );
                if consecutive_errors >= MAX_POLL_ERRORS {
                    return Err(failed(format!("status unavailable ({reason})"), None));
                }
            }
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

async fn poll_once(
    protocol: Protocol,
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
    if let Some(reason) = submit_rejection(protocol, status.as_u16(), body.as_ref()) {
        return Err(if status.is_client_error() && status.as_u16() != 429 {
            PollError::Rejected(reason)
        } else {
            PollError::Transient(reason)
        });
    }
    body.ok_or_else(|| PollError::Transient("unparseable poll response".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(shape: NovitaRequestShape, input: Value) -> Value {
        build_body(&shape, None, &input).expect("body should build")
    }

    #[test]
    fn rendered_shape_names_select_the_original_factory_protocols() {
        // Exactly the names RouterBase's tz_config_render emits.
        for (name, expected) in [
            ("kling_v3_4k_text_to_video", Protocol::Kling),
            ("kling_v3_4k_image_to_video", Protocol::Kling),
            ("kling_v3_motion_control", Protocol::Kling),
            ("kling_v4_text_to_video", Protocol::Kling),
            ("kling_v4_image_to_video", Protocol::Kling),
            ("kling_v4_omni_video", Protocol::Kling),
            ("wan_3_text_to_video", Protocol::Wan),
            ("wan_3_image_to_video", Protocol::Wan),
            ("wan_3_reference_to_video", Protocol::Wan),
        ] {
            let proxy: super::super::NovitaMediaProxyConfig = serde_json::from_value(json!({
                "path": "kling/text-to-video/kling-4.0",
                "async_submission": true,
                "request_shape": name,
                "model": "wan3.0-video-prime",
            }))
            .unwrap_or_else(|e| panic!("`{name}` must deserialize: {e}"));
            assert_eq!(
                protocol(&proxy.request_shape),
                Some(expected),
                "`{name}` must route to its original-factory protocol"
            );
        }
        let legacy: super::super::NovitaMediaProxyConfig = serde_json::from_value(
            json!({"path": "seedream-4.5", "request_shape": "seedream_image"}),
        )
        .unwrap();
        assert_eq!(
            protocol(&legacy.request_shape),
            None,
            "classic shapes keep the /v3 flow"
        );
    }

    #[test]
    fn kling_v3_text_to_video_keeps_4k_single_shot_and_maps_sound() {
        let b = body(
            NovitaRequestShape::KlingV34kTextToVideo,
            json!({"prompt": "a fox", "duration": "8", "sound": true, "aspect_ratio": "9:16",
                   "cfg_scale": 0.5, "negative_prompt": "blur"}),
        );
        assert_eq!(
            b,
            json!({"prompt": "a fox", "settings": {"resolution": "4k", "duration": 8, "audio": "native",
                   "multi_shot": false, "aspect_ratio": "9:16"}}),
            "4K tier is fixed, sound maps to native audio, single shot by default, and the \
             fields the new protocol lacks (cfg_scale, negative_prompt) are not sent"
        );
    }

    #[test]
    fn kling_v3_image_to_video_builds_first_and_last_frames() {
        let b = body(
            NovitaRequestShape::KlingV34kImageToVideo,
            json!({"prompt": "turn", "image_urls": ["https://x/a.png"], "end_image": "https://x/b.png",
                   "duration": 5, "multi_shot": true}),
        );
        assert_eq!(
            b,
            json!({"contents": [
                {"type": "prompt", "text": "turn"},
                {"type": "first_frame", "url": "https://x/a.png"},
                {"type": "last_frame", "url": "https://x/b.png"}],
              "settings": {"duration": 5, "audio": "off", "multi_shot": true, "resolution": "4k"}}),
            "image_urls[0] is the first frame and end_image the last"
        );
        let err = build_body(
            &NovitaRequestShape::KlingV34kImageToVideo,
            None,
            &json!({"prompt": "x"}),
        );
        assert!(err.is_err(), "image-to-video without an image is rejected");
    }

    #[test]
    fn kling_v3_motion_control_maps_pro_tier_and_sound() {
        let b = body(
            NovitaRequestShape::KlingV3MotionControl,
            json!({"image_urls": ["https://x/a.png"], "video_urls": ["https://x/m.mp4"],
                   "model_name": "kling-v3-0-pro", "keep_original_sound": false,
                   "character_orientation": "video"}),
        );
        assert_eq!(
            b,
            json!({"contents": [{"type": "image", "url": "https://x/a.png"},
                                {"type": "video", "url": "https://x/m.mp4"}],
                   "settings": {"resolution": "1080p", "audio": "off", "character_orientation": "video"}}),
            "the pro tier is 1080p and keep_original_sound=false turns audio off"
        );
        let std = body(
            NovitaRequestShape::KlingV3MotionControl,
            json!({"image_urls": ["https://x/a.png"], "video_urls": ["https://x/m.mp4"],
                   "model_name": "kling-v3-0-std"}),
        );
        assert_eq!(
            std["settings"]["resolution"], "720p",
            "the std tier is 720p"
        );
        assert_eq!(
            std["settings"]["audio"], "original",
            "original sound is kept by default"
        );
        assert_eq!(std["settings"]["character_orientation"], "image");
    }

    #[test]
    fn kling_v4_bodies() {
        let t2v = body(
            NovitaRequestShape::KlingV4TextToVideo,
            json!({"prompt": "lake", "resolution": "1080P", "duration": 10, "aspect_ratio": "21:9"}),
        );
        assert_eq!(
            t2v,
            json!({"prompt": "lake", "settings": {"resolution": "1080p", "duration": 10, "aspect_ratio": "21:9"}})
        );

        let i2v = body(
            NovitaRequestShape::KlingV4ImageToVideo,
            json!({"image_urls": ["https://x/1.png"], "middle_image_urls": ["https://x/2.png", "https://x/3.png"],
                   "end_image": "https://x/4.png", "resolution": "720p"}),
        );
        assert_eq!(
            i2v["contents"],
            json!([
                {"type": "first_frame", "url": "https://x/1.png", "id": "first"},
                {"type": "middle_frame", "url": "https://x/2.png", "id": "middle1"},
                {"type": "middle_frame", "url": "https://x/3.png", "id": "middle2"},
                {"type": "last_frame", "url": "https://x/4.png", "id": "last"}]),
            "keyframes keep their order and get stable @ids; no duration means Kling decides"
        );
        assert_eq!(i2v["settings"], json!({"resolution": "720p"}));

        let omni = body(
            NovitaRequestShape::KlingV4OmniVideo,
            json!({"prompt": "restyle @base like @image1", "image_urls": ["https://x/s.png"],
                   "video_urls": ["https://x/f.mp4"], "base_video_url": "https://x/b.mp4",
                   "aspect_ratio": "adaptive"}),
        );
        assert_eq!(
            omni["contents"],
            json!([
                {"type": "prompt", "text": "restyle @base like @image1"},
                {"type": "refer_image", "url": "https://x/s.png", "id": "image1"},
                {"type": "feature_video", "url": "https://x/f.mp4", "id": "video1"},
                {"type": "base_video", "url": "https://x/b.mp4", "id": "base"}])
        );
        assert_eq!(omni["settings"], json!({"aspect_ratio": "adaptive"}));

        let too_many: Vec<String> = (0..11).map(|i| format!("https://x/{i}.png")).collect();
        assert!(
            build_body(
                &NovitaRequestShape::KlingV4OmniVideo,
                None,
                &json!({"prompt": "p", "image_urls": too_many})
            )
            .is_err(),
            "more than 10 images is rejected before submit"
        );
    }

    #[test]
    fn wan_bodies_follow_the_task() {
        let t2v = build_body(
            &NovitaRequestShape::Wan3TextToVideo,
            Some("wan3.0-video-prime"),
            &json!({"prompt": "cat", "resolution": "720p", "ratio": "16:9", "duration": "5",
                    "audio": false, "seed": 7, "prompt_extend": true, "watermark": false,
                    "image_urls": ["https://x/ignored.png"]}),
        )
        .unwrap();
        assert_eq!(
            t2v,
            json!({"model": "wan3.0-video-prime", "input": {"prompt": "cat"},
                   "parameters": {"resolution": "720P", "ratio": "16:9", "duration": 5, "seed": 7,
                                  "audio": false, "prompt_extend": true, "watermark": false}}),
            "text-to-video sends no media even when images are present; resolution is upper-cased"
        );

        let i2v = build_body(
            &NovitaRequestShape::Wan3ImageToVideo,
            None,
            &json!({"image_urls": ["https://x/f.png"], "last_frame_url": "https://x/l.png", "aspect_ratio": "9:16"}),
        )
        .unwrap();
        assert_eq!(i2v["model"], "wan3.0-video");
        assert_eq!(
            i2v["input"],
            json!({"media": [{"type": "first_frame", "url": "https://x/f.png"},
                             {"type": "last_frame", "url": "https://x/l.png"}]}),
            "a prompt is optional once a frame is given"
        );
        assert_eq!(i2v["parameters"], json!({"ratio": "9:16"}));

        let r2v = build_body(
            &NovitaRequestShape::Wan3ReferenceToVideo,
            None,
            &json!({"prompt": "dance", "image_urls": ["https://x/p.png"], "video_urls": ["https://x/v.mp4"],
                    "audio_urls": ["https://x/a.mp3"], "duration": -1}),
        )
        .unwrap();
        assert_eq!(
            r2v["input"]["media"],
            json!([{"type": "reference_image", "url": "https://x/p.png"},
                   {"type": "reference_video", "url": "https://x/v.mp4"},
                   {"type": "reference_audio", "url": "https://x/a.mp3"}])
        );
        assert_eq!(
            r2v["parameters"]["duration"], -1,
            "-1 lets Wan pick the length"
        );
        assert!(
            build_body(
                &NovitaRequestShape::Wan3ReferenceToVideo,
                None,
                &json!({"prompt": "x"})
            )
            .is_err(),
            "reference-to-video needs a reference"
        );
        assert!(
            build_body(&NovitaRequestShape::Wan3TextToVideo, None, &json!({})).is_err(),
            "text-to-video needs a prompt"
        );
    }

    #[test]
    fn submit_rejections_and_task_ids() {
        assert_eq!(
            submit_rejection(
                Protocol::Kling,
                200,
                Some(&json!({"code": 0, "data": {"id": "k1"}}))
            ),
            None
        );
        assert_eq!(
            submit_rejection(
                Protocol::Kling,
                400,
                Some(&json!({"code": 1201, "message": "prompt cannot be empty"}))
            ),
            Some("1201: prompt cannot be empty".to_string())
        );
        assert_eq!(
            submit_rejection(
                Protocol::Kling,
                200,
                Some(&json!({"code": 1200, "message": "bad"}))
            ),
            Some("1200: bad".to_string()),
            "Kling can reject with HTTP 200 and a non-zero code"
        );
        assert_eq!(
            submit_rejection(
                Protocol::Wan,
                400,
                Some(&json!({"code": "InvalidParameter", "message": "bad model"}))
            ),
            Some("InvalidParameter: bad model".to_string())
        );
        assert_eq!(
            submitted_task_id(
                Protocol::Kling,
                &json!({"code": 0, "data": {"id": "k1", "status": "submitted"}})
            ),
            Some("k1".to_string())
        );
        assert_eq!(
            submitted_task_id(
                Protocol::Wan,
                &json!({"output": {"task_id": "w1", "task_status": "PENDING"}})
            ),
            Some("w1".to_string())
        );
        assert_eq!(
            poll_url(Protocol::Kling, "https://api.novita.ai", "k1"),
            "https://api.novita.ai/v3/kling/tasks?task_ids=k1"
        );
        assert_eq!(
            poll_url(Protocol::Wan, "https://api.ppio.com", "w1"),
            "https://api.ppio.com/v3/aliyun/api/v1/tasks/w1"
        );
    }

    #[test]
    fn kling_poll_reports_urls_and_seconds() {
        assert_eq!(
            classify_poll(
                Protocol::Kling,
                &json!({"code": 0, "data": [{"id": "k1", "status": "processing"}]})
            ),
            PollOutcome::Pending,
            "processing (including awaiting settlement) keeps polling"
        );
        let done = classify_poll(
            Protocol::Kling,
            &json!({"code": 0, "data": [{"id": "k1", "status": "succeeded",
                "outputs": [{"type": "video", "url": "https://r/v.mp4", "duration": "10.041"}],
                "usage": {"input_duration": 8}}]}),
        );
        assert_eq!(
            done,
            PollOutcome::Done(TaskOutput {
                urls: vec!["https://r/v.mp4".to_string()],
                usage: Some(json!({"input_seconds": 8.0, "output_seconds": 10.041})),
            }),
            "output seconds come from outputs[].duration, input seconds from usage.input_duration"
        );
        assert_eq!(
            classify_poll(
                Protocol::Kling,
                &json!({"code": 0, "data": [{"id": "k1", "status": "failed", "message": "billing settlement failed"}]})
            ),
            PollOutcome::Failed("billing settlement failed".to_string())
        );
    }

    #[test]
    fn wan_poll_reports_urls_and_seconds() {
        assert_eq!(
            classify_poll(
                Protocol::Wan,
                &json!({"output": {"task_id": "w1", "task_status": "RUNNING"}})
            ),
            PollOutcome::Pending
        );
        assert_eq!(
            classify_poll(
                Protocol::Wan,
                &json!({"output": {"task_id": "w1", "task_status": "SUCCEEDED", "video_url": "https://r/w.mp4"},
                        "usage": {"duration": 9.0, "input_video_duration": 4.0, "output_video_duration": 5.0}})
            ),
            PollOutcome::Done(TaskOutput {
                urls: vec!["https://r/w.mp4".to_string()],
                usage: Some(json!({"input_seconds": 4.0, "output_seconds": 5.0, "seconds": 9.0})),
            })
        );
        assert_eq!(
            classify_poll(
                Protocol::Wan,
                &json!({"output": {"task_id": "w1", "task_status": "FAILED", "code": "InvalidParameter",
                        "message": "Missing required parameter 'input'!"}})
            ),
            PollOutcome::Failed(
                "InvalidParameter: Missing required parameter 'input'!".to_string()
            )
        );
        assert!(
            matches!(
                classify_poll(
                    Protocol::Wan,
                    &json!({"output": {"task_status": "UNKNOWN"}})
                ),
                PollOutcome::Failed(_)
            ),
            "UNKNOWN (expired or missing) is terminal"
        );
    }
}
