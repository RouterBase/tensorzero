//! Hyper3D Rodin 3D generation (`api.hyper3d.com/api/v2`).
//!
//! Submit: `POST {base}/rodin`, multipart form data. The Rodin `tier`
//! (`Gen-2.5-Medium`, `Gen-2`, `Regular`, ...) comes from `proxy.path`. Image
//! tasks attach one to five `images` as file parts, downloaded from the
//! caller's `image_urls` through the public-URL guard; text tasks send only a
//! `prompt`. A rejection arrives as HTTP 201 with a non-empty `error`, so the
//! body is checked as well as the status. An accepted task carries a top-level
//! `uuid`, `jobs.subscription_key`, and `consumed` (the credits charged), which
//! is forwarded to RouterBase as callback `usage.credits`.
//!
//! Poll: `POST {base}/status {subscription_key}` until every job is `Done`
//! (any `Failed` job fails the task). Results: `POST {base}/download
//! {task_uuid}` lists `{name, url}` files; the model file is reported first and
//! a preview image second.

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
use crate::providers::media_task::{
    MediaSubmission, MediaTaskCheck, failure_callback_body, server_error_text,
};
use crate::public_url;
use crate::utils::spawn_ignoring_shutdown;

const PROVIDER_NAME: &str = "Hyper3D";
const PROVIDER_TYPE: &str = "hyper3d";
const API_KEY_ENV: &str = "HYPER3D_API_KEY";
/// Per-HTTP-request timeout for one submit, status or download call.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// Total wall-clock budget for a task to reach a terminal state. Same 1h
/// ceiling as the other media providers; RouterBase never resubmits.
const ASYNC_TASK_TIMEOUT: Duration = Duration::from_secs(3600);
/// Hyper3D asks for at least 5 s before the first status check.
const POLL_INTERVAL: Duration = Duration::from_secs(5);
/// Consecutive failed status calls (network error, 5xx, 429, unparseable
/// body) tolerated before giving up. A poll is a read: a transient blip must
/// not fail a task that Hyper3D goes on to finish and bill.
const MAX_POLL_ERRORS: u32 = 6;
/// `/download` right after the last job turns `Done` can list no model yet.
const DOWNLOAD_ATTEMPTS: u32 = 4;
const MAX_IMAGES: usize = 5;
/// Largest reference image accepted for upload.
const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

lazy_static! {
    // Empty counts as unset: docker-compose passes `${HYPER3D_API_BASE:-}`.
    static ref HYPER3D_API_BASE: String = std::env::var("HYPER3D_API_BASE")
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://api.hyper3d.com/api/v2".to_string());
}

/// Rodin options forwarded verbatim when present. Which ones a tier accepts
/// is enforced by RouterBase's parameter schema; Hyper3D rejects the rest
/// with `INVALID_REQUEST`.
const FORM_FIELDS: &[&str] = &[
    "seed",
    "mesh_mode",
    "geometry_file_format",
    "material",
    "quality",
    "quality_override",
    "bbox_condition",
    "addons",
    "TAPose",
    "use_original_alpha",
    "preview_render",
    "hd_texture",
    "uhd_texture",
    "texture_mode",
    "texture_delight",
    "is_micro",
    "geometry_instruct_mode",
    "image_label",
    "is_symmetric",
    "soft",
    "detail_level",
    "quad_normal",
    "condition_mode",
    "mesh_simplify",
    "mesh_smooth",
];
/// Image-only fields: they describe the uploaded images.
const IMAGE_ONLY_FIELDS: &[&str] = &["image_label", "use_original_alpha", "condition_mode"];
const MODEL_EXTENSIONS: &[&str] = &["glb", "fbx", "obj", "usdz", "stl"];
const PREVIEW_EXTENSIONS: &[&str] = &["webp", "png", "jpg", "jpeg"];

pub struct Hyper3dProvider;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct Hyper3dMediaProxyConfig {
    /// The Rodin `tier` (e.g. `Gen-2.5-Medium`, `Gen-2`, `Regular`).
    pub path: Arc<str>,
    /// Return the task `uuid` as soon as Hyper3D accepts the task and poll in
    /// the background. Without it the inference call blocks until the model
    /// is done (a minute or more).
    #[serde(default)]
    pub async_submission: bool,
    pub request_shape: Hyper3dRequestShape,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum Hyper3dRequestShape {
    /// `prompt` only; no images.
    #[serde(rename = "hyper3d_text_to_3d")]
    Hyper3dTextTo3d,
    /// One to five `image_urls`, uploaded as `images` file parts; `prompt`
    /// optional.
    #[serde(rename = "hyper3d_image_to_3d")]
    Hyper3dImageTo3d,
}

/// What an accepted submission returned.
#[derive(Debug, PartialEq)]
struct Submitted {
    uuid: String,
    subscription_key: String,
    credits: Option<f64>,
}

impl Hyper3dProvider {
    pub async fn infer_media_proxy(
        proxy: &Hyper3dMediaProxyConfig,
        callback_url: Option<&str>,
        input: &Value,
        http_client: &TensorzeroHttpClient,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<MediaSubmission, Error> {
        let callback_url = callback_url.ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: "media proxy requires a callback_url".to_string(),
            })
        })?;
        let api_key = get_api_key(dynamic_api_keys)?;
        let fields = build_fields(proxy.request_shape, &proxy.path, input)?;
        let raw_request = serde_json::to_string(&fields_as_json(&fields)).unwrap_or_default();

        let mut form = reqwest::multipart::Form::new();
        for (name, value) in fields {
            form = form.text(name, value);
        }
        if proxy.request_shape == Hyper3dRequestShape::Hyper3dImageTo3d {
            for (i, url) in image_urls(input)?.iter().enumerate() {
                form = form.part("images", fetch_image_part(url, i).await?);
            }
        }

        let url = endpoint("rodin")?;
        let response = http_client
            .post(url)
            .bearer_auth(api_key.expose_secret())
            .multipart(form)
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Hyper3D request failed: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;
        let status = response.status();
        let raw = response.text().await.unwrap_or_default();
        let submitted = parse_submit(status, &raw).map_err(|message| {
            Error::new(ErrorDetails::InferenceServer {
                message,
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw.clone()),
            })
        })?;
        let task_id = submitted.uuid.clone();
        let format = requested_format(input);
        // The status endpoint is keyed by the subscription key, not the task
        // uuid, and the credits charged are only on the submit response: a
        // later check needs all three back.
        let submission = MediaSubmission {
            task_id: task_id.clone(),
            resume: Some(resume_state(&submitted, format.as_deref())),
        };

        if proxy.async_submission {
            // Return the task id now and finish out-of-band: holding the
            // inference call open for the whole render would hold the
            // RouterBase submit worker for minutes.
            let http_client_bg = http_client.clone();
            let api_key_bg = api_key.expose_secret().to_string();
            let callback_url_bg = callback_url.to_string();
            spawn_ignoring_shutdown(async move {
                finish_task(
                    &http_client_bg,
                    &api_key_bg,
                    &callback_url_bg,
                    &submitted,
                    format.as_deref(),
                )
                .await;
            });
            return Ok(submission);
        }

        let outcome = run_to_completion(
            http_client,
            api_key.expose_secret(),
            &submitted,
            format.as_deref(),
        )
        .await?;
        post_media_callback(http_client, callback_url, &task_id, &outcome).await?;
        Ok(submission)
    }

    /// Read a submitted task once (see [`crate::providers::media_task`]).
    /// `resume` is the state the submit returned; without it (a task
    /// submitted before the gateway returned one) there is nothing to read.
    /// A status that cannot be read is an `Err`, never a failed task.
    pub async fn check_media_task(
        task_id: &str,
        resume: Option<&Value>,
        http_client: &TensorzeroHttpClient,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<MediaTaskCheck, Error> {
        let Some((submitted, format)) = resume.and_then(|r| parse_resume_state(task_id, r)) else {
            return Ok(MediaTaskCheck::Unsupported);
        };
        let unavailable = |e: CallError| {
            let (CallError::Transient(reason) | CallError::Rejected(reason)) = e;
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Hyper3D task status unavailable: {reason}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        };
        let api_key = get_api_key(dynamic_api_keys)?;
        let key = api_key.expose_secret();
        let payload = json!({ "subscription_key": submitted.subscription_key });
        let status = post_json(http_client, key, "status", &payload)
            .await
            .map_err(unavailable)?;
        match classify_status(&status) {
            PollOutcome::Pending => Ok(MediaTaskCheck::Pending),
            PollOutcome::Failed(reason) => Ok(MediaTaskCheck::Finished(failure_callback_body(
                task_id,
                &server_error_text(
                    PROVIDER_TYPE,
                    &format!("Hyper3D generation failed: {reason}"),
                ),
            ))),
            PollOutcome::Done => {
                let listing = post_json(
                    http_client,
                    key,
                    "download",
                    &json!({ "task_uuid": submitted.uuid }),
                )
                .await
                .map_err(unavailable)?;
                let files = pick_result_files(&listing, format.as_deref());
                if files.is_empty() {
                    // The listing can trail the status by a few seconds; the
                    // next check reads it again.
                    return Ok(MediaTaskCheck::Pending);
                }
                let outcome = TaskOutput {
                    files,
                    credits: submitted.credits,
                };
                Ok(MediaTaskCheck::Finished(success_callback_body(
                    task_id, &outcome,
                )))
            }
        }
    }
}

/// The state a later status check needs, as RouterBase stores it.
fn resume_state(submitted: &Submitted, format: Option<&str>) -> Value {
    json!({
        "subscription_key": submitted.subscription_key,
        "credits": submitted.credits,
        "format": format,
    })
}

/// Rebuild the submission from [`resume_state`]. `None` when the subscription
/// key is missing: nothing can be read without it.
fn parse_resume_state(task_id: &str, resume: &Value) -> Option<(Submitted, Option<String>)> {
    let subscription_key = resume
        .get("subscription_key")
        .and_then(Value::as_str)
        .filter(|k| !k.is_empty())?
        .to_string();
    let submitted = Submitted {
        uuid: task_id.to_string(),
        subscription_key,
        credits: resume.get("credits").and_then(Value::as_f64),
    };
    let format = resume
        .get("format")
        .and_then(Value::as_str)
        .map(str::to_string);
    Some((submitted, format))
}

/// Poll a submitted task to a terminal state and report it to RouterBase.
async fn finish_task(
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    callback_url: &str,
    submitted: &Submitted,
    format: Option<&str>,
) {
    let task_id = &submitted.uuid;
    match run_to_completion(http_client, api_key, submitted, format).await {
        Ok(outcome) => {
            if let Err(e) = post_media_callback(http_client, callback_url, task_id, &outcome).await
            {
                tracing::error!(
                    "Failed to deliver success callback for Hyper3D task {task_id}: {e}"
                );
            }
        }
        Err(e) => {
            tracing::error!("Hyper3D task {task_id} failed: {e}");
            post_media_callback_failure(http_client, callback_url, task_id, &e.to_string()).await;
        }
    }
}

/// Poll until done, then list the result files.
async fn run_to_completion(
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    submitted: &Submitted,
    format: Option<&str>,
) -> Result<TaskOutput, Error> {
    poll_until_done(http_client, api_key, &submitted.subscription_key).await?;
    let mut last_listing = Value::Null;
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        match download_listing(http_client, api_key, &submitted.uuid).await {
            Ok(listing) => {
                let files = pick_result_files(&listing, format);
                if !files.is_empty() {
                    return Ok(TaskOutput {
                        files,
                        credits: submitted.credits,
                    });
                }
                last_listing = listing;
            }
            Err(reason) => {
                tracing::warn!(
                    "Hyper3D download listing for task {} failed ({attempt}/{DOWNLOAD_ATTEMPTS}): {reason}",
                    submitted.uuid
                );
            }
        }
        if attempt < DOWNLOAD_ATTEMPTS {
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }
    Err(Error::new(ErrorDetails::InferenceServer {
        message: "Hyper3D generation completed but returned no model file".to_string(),
        provider_type: PROVIDER_TYPE.to_string(),
        raw_request: None,
        raw_response: Some(last_listing.to_string()),
    }))
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

fn endpoint(name: &str) -> Result<Url, Error> {
    format!("{}/{name}", *HYPER3D_API_BASE)
        .parse::<Url>()
        .map_err(|e| {
            Error::new(ErrorDetails::InvalidBaseUrl {
                message: format!("Failed to construct Hyper3D URL: {e}"),
            })
        })
}

/// The text form fields: `tier` from the proxy path, the prompt, then every
/// allow-listed option the caller set. Arrays (`addons`, `bbox_condition`,
/// `image_label`) become one repeated field per element, as a multipart form
/// carries them. Anything else in `input` is dropped.
fn build_fields(
    shape: Hyper3dRequestShape,
    tier: &str,
    input: &Value,
) -> Result<Vec<(String, String)>, Error> {
    let mut fields = vec![("tier".to_string(), tier.to_string())];
    let prompt = input
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|p| !p.is_empty());
    match (shape, prompt) {
        (Hyper3dRequestShape::Hyper3dTextTo3d, None) => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Hyper3D text-to-3D requires a prompt".to_string(),
            }));
        }
        (_, Some(prompt)) => fields.push(("prompt".to_string(), prompt.to_string())),
        (Hyper3dRequestShape::Hyper3dImageTo3d, None) => {}
    }

    for key in FORM_FIELDS {
        if shape == Hyper3dRequestShape::Hyper3dTextTo3d && IMAGE_ONLY_FIELDS.contains(key) {
            continue;
        }
        let Some(value) = input.get(*key).filter(|v| !v.is_null()) else {
            continue;
        };
        match value {
            Value::Array(items) => {
                for item in items {
                    if let Some(text) = scalar_text(item) {
                        fields.push(((*key).to_string(), text));
                    }
                }
            }
            other => {
                if let Some(text) = scalar_text(other) {
                    fields.push(((*key).to_string(), text));
                }
            }
        }
    }
    Ok(fields)
}

fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.trim().to_string()).filter(|s| !s.is_empty()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// The form fields as JSON, for `raw_request` in errors and logs.
fn fields_as_json(fields: &[(String, String)]) -> Value {
    let mut map = Map::new();
    for (k, v) in fields {
        match map.get_mut(k) {
            Some(Value::Array(items)) => items.push(Value::from(v.as_str())),
            Some(existing) => {
                let first = existing.take();
                *existing = Value::Array(vec![first, Value::from(v.as_str())]);
            }
            None => {
                map.insert(k.clone(), Value::from(v.as_str()));
            }
        }
    }
    Value::Object(map)
}

fn image_urls(input: &Value) -> Result<Vec<String>, Error> {
    let urls: Vec<String> = input
        .get("image_urls")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    if urls.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Hyper3D image-to-3D requires at least one image in image_urls".to_string(),
        }));
    }
    if urls.len() > MAX_IMAGES {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("Hyper3D image-to-3D accepts at most {MAX_IMAGES} images"),
        }));
    }
    Ok(urls)
}

/// Download one caller image through the public-URL guard and wrap it as an
/// `images` file part.
async fn fetch_image_part(url: &str, index: usize) -> Result<reqwest::multipart::Part, Error> {
    let bad = |message: String| Error::new(ErrorDetails::InvalidRequest { message });
    let parsed = url
        .parse::<Url>()
        .map_err(|e| bad(format!("Invalid image URL: {e}")))?;
    public_url::check_url(&parsed)
        .map_err(|reason| bad(format!("Image URL is not allowed: {reason}")))?;
    let client =
        public_url::client().map_err(|e| Error::new(ErrorDetails::InternalError { message: e }))?;
    let failed = |e: &dyn std::fmt::Display| {
        tracing::warn!("Hyper3D reference image fetch failed for {url}: {e}");
        bad(format!(
            "Image {} could not be downloaded (unreachable, refused, or not a public address)",
            index + 1
        ))
    };
    let response = client.get(parsed).send().await.map_err(|e| failed(&e))?;
    if !response.status().is_success() {
        return Err(bad(format!(
            "Image {} could not be downloaded (HTTP {})",
            index + 1,
            response.status()
        )));
    }
    let header_mime = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string);
    let bytes = response.bytes().await.map_err(|e| failed(&e))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(bad(format!(
            "Image {} is larger than {} MB",
            index + 1,
            MAX_IMAGE_BYTES / (1024 * 1024)
        )));
    }
    let (mime, ext) = image_mime(&bytes, header_mime.as_deref());
    reqwest::multipart::Part::bytes(bytes.to_vec())
        .file_name(format!("image_{index}.{ext}"))
        .mime_str(mime)
        .map_err(|e| bad(format!("Invalid image content type: {e}")))
}

/// The image type from its magic bytes, falling back to the header, then PNG.
fn image_mime(bytes: &[u8], header: Option<&str>) -> (&'static str, &'static str) {
    if bytes.starts_with(b"\x89PNG") {
        return ("image/png", "png");
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return ("image/jpeg", "jpg");
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return ("image/webp", "webp");
    }
    match header.map(|h| h.split(';').next().unwrap_or("").trim()) {
        Some("image/jpeg") => ("image/jpeg", "jpg"),
        Some("image/webp") => ("image/webp", "webp"),
        _ => ("image/png", "png"),
    }
}

fn requested_format(input: &Value) -> Option<String> {
    input
        .get("geometry_file_format")
        .and_then(Value::as_str)
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
}

/// Read the submit response. Hyper3D answers a rejection with HTTP 201 and a
/// non-empty `error` (`API_INSUFFICIENT_FUNDS`, `INVALID_REQUEST`, ...), so
/// both the status and the body decide.
fn parse_submit(status: reqwest::StatusCode, raw: &str) -> Result<Submitted, String> {
    let body: Option<Value> = serde_json::from_str(raw).ok();
    let error_code = body
        .as_ref()
        .and_then(|v| v.get("error"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty());
    if !status.is_success() || error_code.is_some() {
        let label = error_code.map_or_else(|| status.as_u16().to_string(), ToString::to_string);
        let reason = body
            .as_ref()
            .and_then(|v| v.get("message").or_else(|| v.get("detail")))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map_or_else(|| format!("HTTP {status}"), ToString::to_string);
        return Err(format!("Hyper3D rejected the request ({label}): {reason}"));
    }
    let body = body.ok_or_else(|| "Hyper3D submit response is not JSON".to_string())?;
    let text_at = |pointer: &str| {
        body.pointer(pointer)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let uuid =
        text_at("/uuid").ok_or_else(|| "Hyper3D submit response missing uuid".to_string())?;
    let subscription_key = text_at("/jobs/subscription_key")
        .ok_or_else(|| "Hyper3D submit response missing jobs.subscription_key".to_string())?;
    let credits = body.get("consumed").and_then(|c| {
        c.as_f64()
            .or_else(|| c.as_str().and_then(|s| s.trim().parse().ok()))
    });
    Ok(Submitted {
        uuid,
        subscription_key,
        credits,
    })
}

/// Terminal classification of one status response.
#[derive(Debug, PartialEq)]
enum PollOutcome {
    Done,
    Failed(String),
    Pending,
}

/// Every job `Done` is done; any `Failed` job fails the task; `Waiting` /
/// `Generating` (and an empty or missing list) keep polling.
fn classify_status(body: &Value) -> PollOutcome {
    let jobs: Vec<&Value> = body
        .get("jobs")
        .and_then(Value::as_array)
        .map(|a| a.iter().collect())
        .unwrap_or_default();
    let status_of = |job: &Value| {
        job.get("status")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    if jobs
        .iter()
        .any(|j| status_of(j).eq_ignore_ascii_case("failed"))
    {
        return PollOutcome::Failed("a generation job failed".to_string());
    }
    if !jobs.is_empty()
        && jobs
            .iter()
            .all(|j| status_of(j).eq_ignore_ascii_case("done"))
    {
        return PollOutcome::Done;
    }
    PollOutcome::Pending
}

enum CallError {
    Transient(String),
    Rejected(String),
}

/// One JSON POST to `/status` or `/download`. 429 and 5xx are transient;
/// another 4xx or a body-level `error` (`NO_SUCH_TASK`) will not fix itself.
async fn post_json(
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    name: &str,
    payload: &Value,
) -> Result<Value, CallError> {
    let url = endpoint(name).map_err(|e| CallError::Rejected(e.to_string()))?;
    let response = http_client
        .post(url)
        .bearer_auth(api_key)
        .json(payload)
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| CallError::Transient(e.to_string()))?;
    let status = response.status();
    let raw = response.text().await.unwrap_or_default();
    let body: Option<Value> = serde_json::from_str(&raw).ok();
    let error_code = body
        .as_ref()
        .and_then(|v| v.get("error"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    if status.as_u16() == 429 || status.is_server_error() {
        return Err(CallError::Transient(format!("HTTP {status}")));
    }
    if !status.is_success() || error_code.is_some() {
        let reason = body
            .as_ref()
            .and_then(|v| v.get("message"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let label = error_code.unwrap_or_else(|| format!("HTTP {status}"));
        return Err(CallError::Rejected(match reason {
            Some(r) => format!("{label}: {r}"),
            None => label,
        }));
    }
    body.ok_or_else(|| CallError::Transient("unparseable response".to_string()))
}

async fn poll_until_done(
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    subscription_key: &str,
) -> Result<(), Error> {
    let fail = |message: String, raw_response: Option<String>| {
        Error::new(ErrorDetails::InferenceServer {
            message,
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response,
        })
    };
    let deadline = Instant::now() + ASYNC_TASK_TIMEOUT;
    let payload = json!({ "subscription_key": subscription_key });
    let mut consecutive_errors = 0u32;
    loop {
        tokio::time::sleep(POLL_INTERVAL).await;
        if Instant::now() >= deadline {
            return Err(fail(
                format!(
                    "Hyper3D generation failed: task did not complete within {}s",
                    ASYNC_TASK_TIMEOUT.as_secs()
                ),
                None,
            ));
        }
        match post_json(http_client, api_key, "status", &payload).await {
            Ok(body) => {
                consecutive_errors = 0;
                match classify_status(&body) {
                    PollOutcome::Done => return Ok(()),
                    PollOutcome::Failed(reason) => {
                        return Err(fail(
                            format!("Hyper3D generation failed: {reason}"),
                            Some(body.to_string()),
                        ));
                    }
                    PollOutcome::Pending => {}
                }
            }
            Err(CallError::Rejected(reason)) => {
                return Err(fail(format!("Hyper3D generation failed: {reason}"), None));
            }
            Err(CallError::Transient(reason)) => {
                consecutive_errors += 1;
                tracing::warn!(
                    "Hyper3D status poll failed ({consecutive_errors}/{MAX_POLL_ERRORS}): {reason}"
                );
                if consecutive_errors >= MAX_POLL_ERRORS {
                    return Err(fail(
                        format!("Hyper3D generation failed: status unavailable ({reason})"),
                        None,
                    ));
                }
            }
        }
    }
}

async fn download_listing(
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    task_uuid: &str,
) -> Result<Value, String> {
    post_json(
        http_client,
        api_key,
        "download",
        &json!({ "task_uuid": task_uuid }),
    )
    .await
    .map_err(|e| match e {
        CallError::Transient(r) | CallError::Rejected(r) => r,
    })
}

/// The extension of a download entry, from its `name` or else its URL path.
fn extension_of(entry: &Value) -> Option<String> {
    let from = |s: &str| {
        let s = s.split(['?', '#']).next().unwrap_or(s);
        let file = s.rsplit('/').next().unwrap_or(s);
        file.rsplit_once('.')
            .map(|(_, ext)| ext.to_ascii_lowercase())
    };
    entry
        .get("name")
        .and_then(Value::as_str)
        .and_then(from)
        .or_else(|| entry.get("url").and_then(Value::as_str).and_then(from))
}

/// One delivered file: its URL, the name Hyper3D lists it under, and what it
/// is: `model` (a 3D file), `preview` (a render image: an image whose name or
/// URL says preview / render), `texture` (any other image, e.g. the maps that
/// ship next to an `obj` / `stl`, which do not pack textures), `file` (the
/// rest, e.g. an `.mtl`).
#[derive(Debug, Clone, PartialEq)]
struct ResultFile {
    url: String,
    name: Option<String>,
    kind: &'static str,
}

fn kind_of(ext: &str, name_or_url: &str) -> &'static str {
    if MODEL_EXTENSIONS.contains(&ext) {
        "model"
    } else if PREVIEW_EXTENSIONS.contains(&ext) {
        let lower = name_or_url.to_ascii_lowercase();
        if lower.contains("preview") || lower.contains("render") {
            "preview"
        } else {
            "texture"
        }
    } else {
        "file"
    }
}

/// Every file Hyper3D lists, in a stable order: the model file first (the
/// requested `geometry_file_format`, else the first 3D file listed), then the
/// preview image, then everything else in listing order (Extreme-High's
/// extra 10M-face mesh, Gen-1.5 HighPack's high-poly model, the second set for
/// `material: All`, extra preview renders, textures). So `data[0]` stays the
/// model and `data[1]` the preview when there is one. Nothing when no model
/// file is listed: a preview alone is not a delivered model.
fn pick_result_files(listing: &Value, format: Option<&str>) -> Vec<ResultFile> {
    let entries: Vec<(String, ResultFile)> = listing
        .get("list")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let url = e.get("url").and_then(Value::as_str)?.trim();
                    if url.is_empty() {
                        return None;
                    }
                    let ext = extension_of(e).unwrap_or_default();
                    let label = e.get("name").and_then(Value::as_str).unwrap_or(url);
                    let file = ResultFile {
                        url: url.to_string(),
                        name: e
                            .get("name")
                            .and_then(Value::as_str)
                            .map(str::trim)
                            .filter(|n| !n.is_empty())
                            .map(str::to_string),
                        kind: kind_of(&ext, label),
                    };
                    Some((ext, file))
                })
                .collect()
        })
        .unwrap_or_default();
    let model = format
        .and_then(|f| entries.iter().position(|(ext, _)| ext == f))
        .or_else(|| entries.iter().position(|(_, f)| f.kind == "model"));
    let Some(model) = model else {
        return Vec::new();
    };
    // A named preview, else the first image (as before this listed every file).
    let preview = entries
        .iter()
        .position(|(_, f)| f.kind == "preview")
        .or_else(|| entries.iter().position(|(_, f)| f.kind == "texture"));
    let mut out: Vec<ResultFile> = Vec::with_capacity(entries.len());
    let lead = std::iter::once(model).chain(preview);
    let rest = (0..entries.len()).filter(|i| *i != model && Some(*i) != preview);
    for i in lead.chain(rest) {
        let file = &entries[i].1;
        if !out.iter().any(|f| f.url == file.url) {
            out.push(file.clone());
        }
    }
    out
}

/// What a successful task delivered.
#[derive(Debug, Default, PartialEq)]
struct TaskOutput {
    /// Every listed file, ordered as [`pick_result_files`] documents.
    files: Vec<ResultFile>,
    /// `consumed` at submit: the credits Hyper3D charged.
    credits: Option<f64>,
}

impl TaskOutput {
    fn urls(&self) -> Vec<&str> {
        self.files.iter().map(|f| f.url.as_str()).collect()
    }

    /// Per-file metadata for RouterBase's `resultItems` (one item per URL, in
    /// the same order), which lands on each returned `data[i]`.
    fn items(&self) -> Vec<Value> {
        self.files
            .iter()
            .map(|f| match &f.name {
                Some(name) => json!({ "kind": f.kind, "name": name }),
                None => json!({ "kind": f.kind }),
            })
            .collect()
    }
}

/// Success payload for RouterBase's media callback. The credits charged ride
/// as `usage.credits` (top level and under `data`), which credit-billed
/// models settle from.
fn media_callback_success_body(task_id: &str, result_json: &str, outcome: &TaskOutput) -> Value {
    let mut body = json!({
        "taskId": task_id,
        "task_id": task_id,
        "state": "success",
        "resultJson": result_json,
        "resultUrls": outcome.urls(),
        "resultItems": outcome.items(),
        "data": {
            "taskId": task_id,
            "task_id": task_id,
            "resultJson": result_json,
            "resultUrls": outcome.urls(),
            "resultItems": outcome.items(),
        }
    });
    if let Some(credits) = outcome.credits {
        let usage = json!({ "credits": credits });
        body["usage"] = usage.clone();
        body["data"]["usage"] = usage;
    }
    body
}

/// The success callback body for a finished task.
fn success_callback_body(task_id: &str, outcome: &TaskOutput) -> Value {
    let result_json = json!({ "resultUrls": outcome.urls() }).to_string();
    media_callback_success_body(task_id, &result_json, outcome)
}

async fn post_media_callback(
    http_client: &TensorzeroHttpClient,
    callback_url: &str,
    task_id: &str,
    outcome: &TaskOutput,
) -> Result<(), Error> {
    let body = success_callback_body(task_id, outcome);
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

/// Report a task that failed after its id was already returned. Best-effort:
/// runs detached, so a delivery failure can only be logged.
async fn post_media_callback_failure(
    http_client: &TensorzeroHttpClient,
    callback_url: &str,
    task_id: &str,
    error_message: &str,
) {
    let body = failure_callback_body(task_id, error_message);
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
    use reqwest::StatusCode;

    fn values(fields: &[(String, String)], key: &str) -> Vec<String> {
        fields
            .iter()
            .filter(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
            .collect()
    }

    #[test]
    fn text_fields_carry_tier_prompt_and_allowed_options() {
        let input = json!({
            "prompt": "  a wooden treasure chest ",
            "texture_mode": "high",
            "quality_override": 18000,
            "TAPose": true,
            "addons": ["HighPack"],
            "bbox_condition": [100, 80, 60],
            "image_label": ["F"],
            "condition_mode": "fuse",
            "aspect_ratio": "16:9",
            "seed": null,
        });
        let fields =
            build_fields(Hyper3dRequestShape::Hyper3dTextTo3d, "Gen-2.5-High", &input).unwrap();
        assert_eq!(
            values(&fields, "tier"),
            vec!["Gen-2.5-High"],
            "tier comes from the proxy path"
        );
        assert_eq!(
            values(&fields, "prompt"),
            vec!["a wooden treasure chest"],
            "prompt is trimmed"
        );
        assert_eq!(values(&fields, "texture_mode"), vec!["high"]);
        assert_eq!(values(&fields, "quality_override"), vec!["18000"]);
        assert_eq!(values(&fields, "TAPose"), vec!["true"]);
        assert_eq!(values(&fields, "addons"), vec!["HighPack"]);
        assert_eq!(
            values(&fields, "bbox_condition"),
            vec!["100", "80", "60"],
            "arrays become one repeated field per element"
        );
        for dropped in ["aspect_ratio", "seed", "image_label", "condition_mode"] {
            assert!(
                values(&fields, dropped).is_empty(),
                "`{dropped}` must not be sent for a text task (unknown, null or image-only)"
            );
        }
    }

    #[test]
    fn text_task_requires_a_prompt_but_image_task_does_not() {
        assert!(
            build_fields(
                Hyper3dRequestShape::Hyper3dTextTo3d,
                "Gen-2",
                &json!({ "prompt": "  " })
            )
            .is_err(),
            "a blank prompt must be rejected before submit"
        );
        let fields = build_fields(
            Hyper3dRequestShape::Hyper3dImageTo3d,
            "Regular",
            &json!({ "image_urls": ["https://x/a.png"], "condition_mode": "concat" }),
        )
        .unwrap();
        assert!(
            values(&fields, "prompt").is_empty(),
            "no prompt is sent when the caller gave none"
        );
        assert_eq!(values(&fields, "condition_mode"), vec!["concat"]);
        assert!(
            values(&fields, "image_urls").is_empty(),
            "image URLs are uploaded as files, never sent as a text field"
        );
    }

    #[test]
    fn image_urls_must_number_one_to_five() {
        assert!(image_urls(&json!({ "image_urls": [] })).is_err());
        let six: Vec<String> = (0..6).map(|i| format!("https://x/{i}.png")).collect();
        assert!(
            image_urls(&json!({ "image_urls": six })).is_err(),
            "more than five images must be rejected"
        );
        assert_eq!(
            image_urls(&json!({ "image_urls": [" https://x/a.png "] })).unwrap(),
            vec!["https://x/a.png"]
        );
    }

    #[test]
    fn submit_rejection_in_a_201_body_is_an_error() {
        let err = parse_submit(
            StatusCode::CREATED,
            r#"{"error":"API_INSUFFICIENT_FUNDS","message":"The fund in your wallet is insufficient to perform this operation."}"#,
        )
        .unwrap_err();
        assert_eq!(
            err,
            "Hyper3D rejected the request (API_INSUFFICIENT_FUNDS): The fund in your wallet is insufficient to perform this operation.",
            "a body-level error must not be treated as an accepted task"
        );
        let err = parse_submit(StatusCode::UNAUTHORIZED, "").unwrap_err();
        assert!(
            err.contains("(401)"),
            "transport errors carry the status: {err}"
        );
    }

    #[test]
    fn submit_success_reads_ids_and_credits() {
        let ok = parse_submit(
            StatusCode::CREATED,
            r#"{"message":"Submitted.","uuid":"u-1","jobs":{"uuids":["j-1"],"subscription_key":"sk-1"},"consumed":0.5}"#,
        )
        .unwrap();
        assert_eq!(
            ok,
            Submitted {
                uuid: "u-1".into(),
                subscription_key: "sk-1".into(),
                credits: Some(0.5),
            }
        );
        assert!(
            parse_submit(StatusCode::CREATED, r#"{"message":"Submitted.","jobs":{}}"#).is_err(),
            "an accepted task needs a uuid and a subscription key"
        );
    }

    #[test]
    fn status_waits_for_every_job_and_stops_on_any_failure() {
        let st = |a: &str, b: &str| json!({ "jobs": [ { "uuid": "1", "status": a }, { "uuid": "2", "status": b } ] });
        assert_eq!(
            classify_status(&st("Done", "Generating")),
            PollOutcome::Pending
        );
        assert_eq!(
            classify_status(&st("Waiting", "Waiting")),
            PollOutcome::Pending
        );
        assert_eq!(
            classify_status(&st("Done", "Done")),
            PollOutcome::Done,
            "all jobs done is done"
        );
        assert!(
            matches!(
                classify_status(&st("Done", "Failed")),
                PollOutcome::Failed(_)
            ),
            "any failed job fails the task"
        );
        assert_eq!(
            classify_status(&json!({ "jobs": [] })),
            PollOutcome::Pending,
            "an empty job list is not done"
        );
    }

    #[test]
    fn results_list_the_requested_model_then_a_preview_then_every_other_file() {
        let listing = json!({ "list": [
            { "name": "preview.webp", "url": "https://dl.example/preview.webp?sig=1" },
            { "name": "base_basic_pbr.fbx", "url": "https://dl.example/m.fbx" },
            { "name": "base_basic_pbr.glb", "url": "https://dl.example/m.glb?sig=2" },
            { "name": "texture_diffuse.png", "url": "https://dl.example/t.png" },
            { "name": "high_10m.glb", "url": "https://dl.example/hi.glb" },
            { "name": "base_basic_pbr.glb", "url": "https://dl.example/m.glb?sig=2" }
        ]});
        let urls = |files: Vec<ResultFile>| files.into_iter().map(|f| f.url).collect::<Vec<_>>();
        assert_eq!(
            urls(pick_result_files(&listing, Some("glb"))),
            vec![
                "https://dl.example/m.glb?sig=2",
                "https://dl.example/preview.webp?sig=1",
                "https://dl.example/m.fbx",
                "https://dl.example/t.png",
                "https://dl.example/hi.glb",
            ],
            "requested model, then the preview, then every other file once, in listing order"
        );
        let files = pick_result_files(&listing, Some("glb"));
        assert_eq!(
            files.iter().map(|f| f.kind).collect::<Vec<_>>(),
            vec!["model", "preview", "model", "texture", "model"],
            "a texture map is not a preview; extra meshes stay models"
        );
        assert_eq!(files[0].name.as_deref(), Some("base_basic_pbr.glb"));
        assert_eq!(
            pick_result_files(&listing, None)[0].url,
            "https://dl.example/m.fbx",
            "without a requested format the first 3D file wins"
        );
        let preview_only =
            json!({ "list": [ { "name": "preview.webp", "url": "https://x/p.webp" } ] });
        assert!(
            pick_result_files(&preview_only, Some("glb")).is_empty(),
            "a preview without a model is not a delivered model"
        );
        let nameless = json!({ "list": [
            { "url": "https://x/out/model.glb?x=1" },
            { "url": "https://x/out/model.mtl" }
        ] });
        let files = pick_result_files(&nameless, None);
        assert_eq!(
            files[0].url, "https://x/out/model.glb?x=1",
            "extension from the URL path"
        );
        assert_eq!(files[0].name, None);
        assert_eq!(
            files[1].kind, "file",
            "an obj material file is a plain file"
        );
        // No named preview: the first image stands in, as before.
        let unnamed = json!({ "list": [
            { "name": "m.glb", "url": "https://x/m.glb" },
            { "name": "shot.png", "url": "https://x/shot.png" }
        ] });
        let files = pick_result_files(&unnamed, Some("glb"));
        assert_eq!(files[1].url, "https://x/shot.png");
    }

    #[test]
    fn image_type_comes_from_magic_bytes() {
        assert_eq!(image_mime(b"\x89PNG\r\n", Some("text/html")).0, "image/png");
        assert_eq!(image_mime(&[0xFF, 0xD8, 0xFF, 0xE0], None).0, "image/jpeg");
        assert_eq!(image_mime(b"RIFF\0\0\0\0WEBPVP8 ", None).0, "image/webp");
        assert_eq!(
            image_mime(b"????", Some("image/jpeg; charset=binary")).0,
            "image/jpeg",
            "unknown bytes fall back to the header"
        );
    }

    #[test]
    fn success_callback_carries_credits_as_usage() {
        let outcome = TaskOutput {
            files: vec![
                ResultFile {
                    url: "https://x/m.glb".into(),
                    name: Some("base.glb".into()),
                    kind: "model",
                },
                ResultFile {
                    url: "https://x/p.webp".into(),
                    name: None,
                    kind: "preview",
                },
            ],
            credits: Some(2.5),
        };
        let body = media_callback_success_body("t1", "{}", &outcome);
        assert_eq!(body["state"], "success");
        assert_eq!(
            body["usage"]["credits"], 2.5,
            "credits ride as usage.credits"
        );
        assert_eq!(body["data"]["usage"]["credits"], 2.5);
        assert_eq!(
            body["resultUrls"],
            json!(["https://x/m.glb", "https://x/p.webp"])
        );
        assert_eq!(
            body["resultItems"],
            json!([{ "kind": "model", "name": "base.glb" }, { "kind": "preview" }]),
            "one item per URL, same order"
        );
        assert_eq!(body["data"]["resultItems"], body["resultItems"]);
        let none = media_callback_success_body("t1", "{}", &TaskOutput::default());
        assert!(none.get("usage").is_none(), "no credits, no usage");
    }
}

#[cfg(test)]
mod media_task_check_tests {
    use super::*;

    #[test]
    fn resume_state_round_trips_everything_a_check_needs() {
        let submitted = Submitted {
            uuid: "task-uuid".to_string(),
            subscription_key: "sub-key".to_string(),
            credits: Some(0.5),
        };
        let resume = resume_state(&submitted, Some("fbx"));
        let (back, format) = parse_resume_state("task-uuid", &resume).unwrap();
        assert_eq!(back, submitted);
        assert_eq!(format.as_deref(), Some("fbx"));
    }

    #[test]
    fn resume_state_without_optional_parts_still_parses() {
        let resume = json!({ "subscription_key": "sub-key", "credits": null, "format": null });
        let (back, format) = parse_resume_state("u", &resume).unwrap();
        assert_eq!(back.credits, None);
        assert_eq!(format, None);
    }

    #[test]
    fn without_a_subscription_key_there_is_nothing_to_read() {
        for resume in [json!({}), json!({ "subscription_key": "" }), json!(null)] {
            assert!(parse_resume_state("u", &resume).is_none(), "{resume}");
        }
    }
}
