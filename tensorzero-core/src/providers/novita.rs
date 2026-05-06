use lazy_static::lazy_static;
use schemars::JsonSchema;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Duration;
use std::sync::Arc;
use tokio::time::Instant;
use url::Url;
use uuid::Uuid;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;

const PROVIDER_NAME: &str = "Novita";
const PROVIDER_TYPE: &str = "novita";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

lazy_static! {
    static ref NOVITA_API_BASE: String =
        std::env::var("NOVITA_API_BASE").unwrap_or_else(|_| "https://api.novita.ai".to_string());
}

pub struct NovitaProvider;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct NovitaMediaProxyConfig {
    pub path: Arc<str>,
    #[serde(default)]
    pub async_submission: bool,
    pub request_shape: NovitaRequestShape,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum NovitaRequestShape {
    GeminiImageTextToImage,
    GeminiImageEdit,
    GptImageTextToImage,
    GptImageEdit,
    VeoTextToVideo,
    VeoImageToVideo,
    Veo31ImageToVideo,
}

impl NovitaProvider {
    pub async fn infer_media_proxy(
        proxy: &NovitaMediaProxyConfig,
        callback_url: Option<&str>,
        input: &Value,
        http_client: &TensorzeroHttpClient,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<String, Error> {
        let callback_url = callback_url.ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: "kie_media proxy requires a callback_url".to_string(),
            })
        })?;
        let api_key = get_api_key(dynamic_api_keys)?;
        let body = build_body(&proxy.request_shape, input)?;
        let url = format!("{}/v3/{}", *NOVITA_API_BASE, proxy.path)
            .parse::<Url>()
            .map_err(|e| {
                Error::new(ErrorDetails::InvalidBaseUrl {
                    message: format!("Failed to construct Novita URL: {e}"),
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
                    message: format!("Novita request failed: {e}"),
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
                message: format!("Novita returned {status}: {raw}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: Some(raw),
            }));
        }

        let raw_json: Value = serde_json::from_str(&raw).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Failed to parse Novita response: {e}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: Some(raw.clone()),
            })
        })?;

        let (task_id, result_body) = if proxy.async_submission {
            let task_id = raw_json
                .get("task_id")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: "Novita async response missing task_id".to_string(),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                        raw_response: Some(raw.clone()),
                    })
                })?
                .to_string();
            let result_body =
                poll_async_result(http_client, api_key.expose_secret(), &task_id).await?;
            (task_id, result_body)
        } else {
            (format!("novita-{}", Uuid::new_v4()), raw_json)
        };

        let urls = parse_urls(&result_body);
        if urls.is_empty() {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: "Novita completed but returned no media URLs".to_string(),
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
    if let Some(key) = dynamic_api_keys.get("NOVITA_API_KEY") {
        return Ok(SecretString::from(key.expose_secret().to_string()));
    }

    std::env::var("NOVITA_API_KEY")
        .map(SecretString::from)
        .map_err(|_| {
            Error::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "NOVITA_API_KEY is not configured".to_string(),
            })
        })
}

fn build_body(shape: &NovitaRequestShape, input: &Value) -> Result<Value, Error> {
    let prompt = input
        .get("prompt")
        .and_then(Value::as_str)
        .filter(|prompt| !prompt.is_empty())
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: "Novita-backed kie_media variants require a prompt".to_string(),
            })
        })?;

    let mut body = serde_json::Map::new();
    body.insert("prompt".into(), Value::from(prompt));

    let allowed: &[&str] = match shape {
        NovitaRequestShape::GeminiImageTextToImage => {
            &["size", "aspect_ratio", "output_format", "google"]
        }
        NovitaRequestShape::GeminiImageEdit => &[
            "size",
            "aspect_ratio",
            "output_format",
            "google",
            "image_urls",
            "image_base64s",
        ],
        NovitaRequestShape::GptImageTextToImage => &[
            "n",
            "quality",
            "background",
            "moderation",
            "output_format",
            "output_compression",
        ],
        NovitaRequestShape::GptImageEdit => &[
            "n",
            "quality",
            "background",
            "output_format",
            "image",
            "mask",
        ],
        NovitaRequestShape::VeoTextToVideo => &[
            "aspect_ratio",
            "duration_seconds",
            "enhance_prompt",
            "generate_audio",
            "negative_prompt",
            "person_generation",
            "resolution",
            "sample_count",
            "seed",
        ],
        NovitaRequestShape::VeoImageToVideo => &[
            "image_url",
            "image_base64",
            "aspect_ratio",
            "duration_seconds",
            "enhance_prompt",
            "generate_audio",
            "negative_prompt",
            "person_generation",
            "resolution",
            "sample_count",
            "seed",
        ],
        NovitaRequestShape::Veo31ImageToVideo => &[
            "image_url",
            "image_base64",
            "last_image_url",
            "last_image_base64",
            "reference_images",
            "aspect_ratio",
            "duration_seconds",
            "enhance_prompt",
            "generate_audio",
            "negative_prompt",
            "person_generation",
            "resolution",
            "sample_count",
            "seed",
        ],
    };

    if let Some(input_obj) = input.as_object() {
        for key in allowed {
            if let Some(value) = input_obj.get(*key) {
                body.insert((*key).to_string(), value.clone());
            }
        }
    }

    if matches!(
        shape,
        NovitaRequestShape::VeoImageToVideo | NovitaRequestShape::Veo31ImageToVideo
    ) {
        if !body.contains_key("image_url") {
            if let Some(value) = input.get("image").and_then(Value::as_str) {
                body.insert("image_url".into(), Value::from(value));
            } else if let Some(first) = input
                .get("image_urls")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
                .and_then(Value::as_str)
            {
                body.insert("image_url".into(), Value::from(first));
            }
        }
        if !body.contains_key("image_base64") {
            if let Some(value) = input.get("image_base64").and_then(Value::as_str) {
                body.insert("image_base64".into(), Value::from(value));
            } else if let Some(first) = input
                .get("image_base64s")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
                .and_then(Value::as_str)
            {
                body.insert("image_base64".into(), Value::from(first));
            }
        }
        if matches!(shape, NovitaRequestShape::Veo31ImageToVideo)
            && !body.contains_key("last_image_url")
        {
            if let Some(value) = input.get("last_image").and_then(Value::as_str) {
                body.insert("last_image_url".into(), Value::from(value));
            }
        }
    }

    if matches!(shape, NovitaRequestShape::GptImageEdit) && !body.contains_key("image") {
        if let Some(first) = input
            .get("image_urls")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
        {
            body.insert("image".into(), first.clone());
        }
    }

    if matches!(
        shape,
        NovitaRequestShape::GeminiImageTextToImage | NovitaRequestShape::GeminiImageEdit
    ) {
        let web = input
            .get("web_search")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let image = input
            .get("image_search")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if web || image {
            let mut google_obj = body
                .get("google")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            if web {
                google_obj.insert("web_search".into(), Value::Bool(true));
            }
            if image {
                google_obj.insert("image_search".into(), Value::Bool(true));
            }
            body.insert("google".into(), Value::Object(google_obj));
        }
    }

    if matches!(
        shape,
        NovitaRequestShape::GptImageTextToImage | NovitaRequestShape::GptImageEdit
    ) {
        let size_label = input
            .get("size")
            .and_then(Value::as_str)
            .unwrap_or("1024 × 1024");
        let normalized = size_label
            .replace('\u{00D7}', "x")
            .replace(' ', "")
            .to_lowercase();
        let mapped = match normalized.as_str() {
            "1024x1024" | "1024x1536" | "1536x1024" | "auto" => normalized,
            "1k" => match input
                .get("aspect_ratio")
                .and_then(Value::as_str)
                .unwrap_or("1:1")
            {
                "2:3" => "1024x1536".to_string(),
                "3:2" => "1536x1024".to_string(),
                _ => "1024x1024".to_string(),
            },
            _ => size_label.to_string(),
        };
        body.insert("size".into(), Value::from(mapped));
    }

    Ok(Value::Object(body))
}

fn parse_urls(body: &Value) -> Vec<String> {
    let containers = [
        body,
        body.get("data").unwrap_or(&Value::Null),
        body.get("task_result")
            .or_else(|| body.get("task"))
            .unwrap_or(&Value::Null),
    ];

    for container in containers {
        for key in ["image_urls", "images", "video_urls", "videos"] {
            if let Some(arr) = container.get(key).and_then(Value::as_array) {
                let urls: Vec<String> = arr
                    .iter()
                    .filter_map(|item| {
                        item.as_str()
                            .map(ToString::to_string)
                            .or_else(|| {
                                item.get("video_url")
                                    .or_else(|| item.get("url"))
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
    let url = format!("{}/v3/async/task-result?task_id={task_id}", *NOVITA_API_BASE);
    let deadline = Instant::now() + REQUEST_TIMEOUT;
    let poll_interval = Duration::from_secs(4);

    loop {
        if Instant::now() >= deadline {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Novita async task {task_id} did not complete within {}s",
                    REQUEST_TIMEOUT.as_secs()
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
                    message: format!("Novita poll request failed: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: None,
                })
            })?;

        let status = response.status();
        let body: Value = response.json().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Novita poll response parse failed: {e}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;

        if !status.is_success() {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!("Novita poll returned {status} for task {task_id}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(body.to_string()),
            }));
        }

        let status_str = body
            .get("task")
            .and_then(|task| task.get("status"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let normalized = status_str.to_uppercase();
        if normalized.contains("SUCCEED") || normalized == "SUCCESS" {
            return Ok(body);
        }
        if normalized.contains("FAIL") || normalized == "FAILED" {
            let reason = body
                .get("task")
                .and_then(|task| task.get("reason").or_else(|| task.get("message")))
                .and_then(Value::as_str)
                .unwrap_or("(no reason given)");
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!("Novita generation failed: {reason}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(body.to_string()),
            }));
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
