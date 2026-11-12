//! ByteDance Ark image generation (Seedream), served by BytePlus ModelArk
//! (`ark.ap-southeast.bytepluses.com`, global) and Volcengine Ark
//! (`ark.cn-beijing.volces.com`, mainland China). Both speak the same API and
//! the key is bound to its host, so the base URL comes with each variant
//! (`proxy.api_base`, rendered per region) and the key from `ARK_API_KEY`,
//! the same variable the Ark chat providers use.
//!
//! `POST {api_base}/images/generations` with the model id in the body. The call
//! is synchronous: it answers `{data: [{url, size} | {error}], usage:
//! {generated_images, output_tokens, total_tokens}}`, or `{error: {code,
//! message}}` when the request is rejected. The images are handed to
//! RouterBase's media callback before returning, like Novita's synchronous
//! image routes; RouterBase submits from a background worker, so the wait
//! never sits behind a browser connection.

use schemars::JsonSchema;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::sync::Arc;
use std::time::Duration;
use url::Url;
use uuid::Uuid;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;

const PROVIDER_NAME: &str = "Ark";
const PROVIDER_TYPE: &str = "ark";
const API_KEY_ENV: &str = "ARK_API_KEY";
/// One synchronous generation; a 4K image can take a minute or two.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

pub struct ArkProvider;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ArkMediaProxyConfig {
    /// The Ark model id sent as the body `model`
    /// (e.g. `seedream-4-5-251128`, `doubao-seedream-5-5-261028`).
    pub path: Arc<str>,
    /// The region's Ark API base, e.g. `https://ark.ap-southeast.bytepluses.com/api/v3`.
    pub api_base: Arc<str>,
    pub request_shape: ArkRequestShape,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum ArkRequestShape {
    /// Seedream text-to-image: `prompt`, optional `size` and `watermark`.
    #[serde(rename = "seedream_text_to_image")]
    SeedreamTextToImage,
    /// Seedream edit: as above plus the reference images (`image_urls`, sent
    /// as Ark's `image` array), at least one.
    #[serde(rename = "seedream_edit")]
    SeedreamEdit,
}

impl ArkProvider {
    pub async fn infer_media_proxy(
        proxy: &ArkMediaProxyConfig,
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
        let url = format!(
            "{}/images/generations",
            proxy.api_base.trim_end_matches('/')
        )
        .parse::<Url>()
        .map_err(|e| {
            Error::new(ErrorDetails::InvalidBaseUrl {
                message: format!("Failed to construct Ark URL: {e}"),
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
                    message: format!("Ark request failed: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;
        let status = response.status();
        let raw = response.text().await.unwrap_or_default();
        let parsed: Option<Value> = serde_json::from_str(&raw).ok();
        if !status.is_success() {
            let message = parsed
                .as_ref()
                .and_then(rejection)
                .unwrap_or_else(|| format!("Ark returned {status}: {raw}"));
            return Err(Error::new(ErrorDetails::InferenceServer {
                message,
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(raw_request),
                raw_response: Some(raw),
            }));
        }
        let parsed = parsed.ok_or_else(|| {
            Error::new(ErrorDetails::InferenceServer {
                message: "Failed to parse Ark response".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw.clone()),
            })
        })?;
        let urls = image_urls(&parsed);
        if urls.is_empty() {
            // Every image failed (e.g. an output moderation hit): Ark puts the
            // reason on the item instead of failing the request.
            let message = parsed["data"]
                .as_array()
                .and_then(|items| items.iter().find_map(rejection))
                .or_else(|| rejection(&parsed))
                .unwrap_or_else(|| "Ark generation failed: no image returned".to_string());
            return Err(Error::new(ErrorDetails::InferenceServer {
                message,
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(raw_request),
                raw_response: Some(raw),
            }));
        }

        let task_id = format!("ark-{}", Uuid::now_v7());
        post_media_callback(
            http_client,
            callback_url,
            &task_id,
            &urls,
            parsed.get("usage"),
        )
        .await?;
        Ok(task_id)
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

/// Build the generation body. Only what Ark documents for Seedream is sent:
/// anything else in `input` (RouterBase bookkeeping such as `n`,
/// `resolution`, `aspect_ratio`) is dropped rather than tripping validation.
///
/// - `sequential_image_generation` is pinned to `disabled`: Ark bills every
///   image of a group, and RouterBase prices one image per request.
/// - `watermark` defaults to `false`; Ark's own default stamps a visible
///   "AI generated" label. A caller can still ask for it.
/// - `response_format` is `url`; RouterBase mirrors the files to its storage.
fn build_body(shape: &ArkRequestShape, model: &str, input: &Value) -> Result<Value, Error> {
    let prompt = input
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: "`prompt` is required".to_string(),
            })
        })?;
    let mut body = Map::new();
    body.insert("model".into(), Value::from(model));
    body.insert("prompt".into(), Value::from(prompt));
    if let Some(size) = input.get("size").and_then(Value::as_str) {
        body.insert("size".into(), Value::from(size));
    }
    let watermark = input
        .get("watermark")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    body.insert("watermark".into(), Value::from(watermark));
    body.insert("sequential_image_generation".into(), json!("disabled"));
    body.insert("response_format".into(), json!("url"));

    if *shape == ArkRequestShape::SeedreamEdit {
        let images: Vec<Value> = input
            .get("image_urls")
            .and_then(Value::as_array)
            .map(|urls| {
                urls.iter()
                    .filter_map(Value::as_str)
                    .filter(|u| !u.trim().is_empty())
                    .map(Value::from)
                    .collect()
            })
            .unwrap_or_default();
        if images.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "`image_urls` needs at least one image".to_string(),
            }));
        }
        body.insert("image".into(), Value::Array(images));
    }
    Ok(Value::Object(body))
}

/// The image URLs of a successful response, skipping items that failed.
fn image_urls(response: &Value) -> Vec<String> {
    response["data"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item["url"].as_str())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Ark's reason for rejecting a request or an image, worded so RouterBase
/// shows the caller the upstream message itself:
/// `Ark rejected the request (<code>): <message>`.
fn rejection(value: &Value) -> Option<String> {
    let error = value.get("error")?;
    let message = error["message"].as_str().unwrap_or_default().trim();
    let code = error["code"].as_str().unwrap_or_default();
    if message.is_empty() && code.is_empty() {
        return None;
    }
    Some(match (code.is_empty(), message.is_empty()) {
        (false, false) => format!("Ark rejected the request ({code}): {message}"),
        (true, _) => format!("Ark rejected the request: {message}"),
        (false, true) => format!("Ark rejected the request: {code}"),
    })
}

async fn post_media_callback(
    http_client: &TensorzeroHttpClient,
    callback_url: &str,
    task_id: &str,
    urls: &[String],
    usage: Option<&Value>,
) -> Result<(), Error> {
    let body = media_callback_success_body(task_id, urls, usage);
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

/// The success callback in the shape RouterBase's `media_callback` reads (the
/// same as the other media providers), with Ark's `usage` passed through.
fn media_callback_success_body(task_id: &str, urls: &[String], usage: Option<&Value>) -> Value {
    let result_json = json!({ "resultUrls": urls }).to_string();
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
    if let Some(usage) = usage.filter(|u| u.is_object()) {
        body["usage"] = usage.clone();
        body["data"]["usage"] = usage.clone();
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_to_image_body_keeps_only_documented_fields() {
        let input = json!({
            "prompt": " a red cube ",
            "size": "2K",
            "n": 1,
            "resolution": "1K",
            "aspect_ratio": "1:1",
            "sequential_image_generation": "auto",
            "max_images": 4,
        });
        let body = build_body(
            &ArkRequestShape::SeedreamTextToImage,
            "seedream-4-5-251128",
            &input,
        )
        .unwrap();
        assert_eq!(
            body,
            json!({
                "model": "seedream-4-5-251128",
                "prompt": "a red cube",
                "size": "2K",
                "watermark": false,
                "sequential_image_generation": "disabled",
                "response_format": "url",
            }),
            "RouterBase bookkeeping is dropped, group generation stays off, watermark defaults off"
        );
    }

    #[test]
    fn caller_can_ask_for_the_watermark() {
        let input = json!({"prompt": "x", "watermark": true});
        let body = build_body(&ArkRequestShape::SeedreamTextToImage, "m", &input).unwrap();
        assert_eq!(body["watermark"], json!(true));
    }

    #[test]
    fn edit_sends_reference_images_and_requires_one() {
        let input = json!({"prompt": "make it blue", "image_urls": ["https://a/1.png", "", "https://a/2.png"]});
        let body = build_body(&ArkRequestShape::SeedreamEdit, "m", &input).unwrap();
        assert_eq!(body["image"], json!(["https://a/1.png", "https://a/2.png"]));
        assert!(
            build_body(&ArkRequestShape::SeedreamEdit, "m", &json!({"prompt": "x"})).is_err(),
            "an edit without images is rejected before calling Ark"
        );
        assert!(
            build_body(
                &ArkRequestShape::SeedreamTextToImage,
                "m",
                &json!({"prompt": "  "})
            )
            .is_err(),
            "a blank prompt is rejected"
        );
    }

    #[test]
    fn response_parsing_skips_failed_items_and_words_rejections() {
        let ok = json!({
            "data": [
                {"url": "https://out/1.jpeg", "size": "2048x2048"},
                {"error": {"code": "OutputImageSensitiveContentDetected", "message": "blocked"}}
            ],
            "usage": {"generated_images": 1}
        });
        assert_eq!(image_urls(&ok), vec!["https://out/1.jpeg".to_string()]);

        let rejected = json!({"error": {"code": "InvalidParameter", "message": "size is invalid", "type": "BadRequest"}});
        assert_eq!(
            rejection(&rejected).as_deref(),
            Some("Ark rejected the request (InvalidParameter): size is invalid"),
            "RouterBase strips the `Ark ... : ` prefix and shows the upstream message"
        );
        assert_eq!(rejection(&json!({"data": []})), None);
    }

    #[test]
    fn callback_carries_urls_and_usage() {
        let urls = vec!["https://out/1.jpeg".to_string()];
        let usage = json!({"generated_images": 1, "output_tokens": 16384});
        let body = media_callback_success_body("ark-1", &urls, Some(&usage));
        assert_eq!(body["state"], "success");
        assert_eq!(body["resultUrls"], json!(urls));
        assert_eq!(body["usage"], usage);
        assert_eq!(body["data"]["usage"], usage);
        assert!(
            media_callback_success_body("ark-1", &urls, None)
                .get("usage")
                .is_none()
        );
    }

    #[test]
    fn rendered_shape_names_deserialize() {
        let proxy: ArkMediaProxyConfig = serde_json::from_value(json!({
            "path": "doubao-seedream-5-5-261028",
            "api_base": "https://ark.cn-beijing.volces.com/api/v3",
            "request_shape": "seedream_edit",
        }))
        .unwrap();
        assert_eq!(proxy.request_shape, ArkRequestShape::SeedreamEdit);
        let shape: ArkRequestShape =
            serde_json::from_value(json!("seedream_text_to_image")).unwrap();
        assert_eq!(shape, ArkRequestShape::SeedreamTextToImage);
    }
}
