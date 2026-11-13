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
/// One synchronous generation: a 4K image takes a minute or two, a group of
/// 15 images or a split into 100 layers several.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// Attempts to deliver the success callback, `CALLBACK_RETRY_DELAY` x attempt
/// apart: Ark has already generated and billed the image by then.
/// RouterBase's callback handler is idempotent.
const CALLBACK_ATTEMPTS: u32 = 3;
const CALLBACK_RETRY_DELAY: Duration = Duration::from_secs(2);

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
    /// Seedream 5.5 layer decomposition: one input image split into a base
    /// image plus layers (`layer_decomposition`), each a PNG with its
    /// position, name and, optionally, the text it carries. The prompt is
    /// optional (it states how to split).
    #[serde(rename = "seedream_layers")]
    SeedreamLayers,
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
        let items = result_items(&parsed);
        let urls: Vec<String> = items
            .iter()
            .filter_map(|item| item["url"].as_str().map(str::to_string))
            .collect();
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
        let usage = callback_usage(&parsed, &items);
        let mut attempt = 1;
        loop {
            match post_media_callback(
                http_client,
                callback_url,
                &task_id,
                &urls,
                &items,
                usage.as_ref(),
            )
            .await
            {
                Ok(()) => return Ok(task_id),
                Err(e) if attempt < CALLBACK_ATTEMPTS => {
                    tracing::warn!(
                        "Success callback for Ark task {task_id} failed (attempt {attempt}/{CALLBACK_ATTEMPTS}): {e}"
                    );
                    tokio::time::sleep(CALLBACK_RETRY_DELAY * attempt).await;
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
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

/// Build the generation body. Only what Ark documents for Seedream is sent:
/// anything else in `input` (RouterBase bookkeeping such as `n`,
/// `resolution`, `aspect_ratio`) is dropped rather than tripping validation.
/// RouterBase resolves the size (a preset or `WIDTHxHEIGHT`) and only sends
/// the options the model's schema offers.
///
/// - `sequential_image_generation` is sent only when RouterBase asks for a
///   group (`max_images` > 1, Seedream 4.x / 5.0 lite): Ark then returns up to
///   that many images and bills each one. Seedream 5.5 rejects the field
///   outright, so it is otherwise left out (Ark's default: one image).
/// - `watermark` defaults to `false`; Ark's own default stamps a visible
///   "AI generated" label. A caller can still ask for it.
/// - `background` / `output_format` (5.5), `optimize_prompt_mode` (4.0) and
///   `safety_identifier` (RouterBase's hashed caller id) pass through.
/// - `response_format` is `url`; RouterBase mirrors the files to its storage.
fn build_body(shape: &ArkRequestShape, model: &str, input: &Value) -> Result<Value, Error> {
    let layers = *shape == ArkRequestShape::SeedreamLayers;
    let prompt = input
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|p| !p.is_empty());
    if prompt.is_none() && !layers {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "`prompt` is required".to_string(),
        }));
    }
    let mut body = Map::new();
    body.insert("model".into(), Value::from(model));
    if let Some(prompt) = prompt {
        body.insert("prompt".into(), Value::from(prompt));
    }
    for key in ["size", "output_format", "safety_identifier"] {
        if let Some(value) = input.get(key).and_then(Value::as_str) {
            body.insert(key.into(), Value::from(value));
        }
    }
    // Not allowed in layer mode (Ark rejects it there).
    if !layers && let Some(background) = input.get("background").and_then(Value::as_str) {
        body.insert("background".into(), Value::from(background));
    }
    if let Some(mode) = input.get("optimize_prompt_mode").and_then(Value::as_str) {
        body.insert("optimize_prompt_options".into(), json!({ "mode": mode }));
    }
    let watermark = input
        .get("watermark")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    body.insert("watermark".into(), Value::from(watermark));
    body.insert("response_format".into(), json!("url"));

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
    match shape {
        ArkRequestShape::SeedreamTextToImage => {}
        ArkRequestShape::SeedreamEdit => {
            if images.is_empty() {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "`image_urls` needs at least one image".to_string(),
                }));
            }
            body.insert("image".into(), Value::Array(images));
        }
        ArkRequestShape::SeedreamLayers => {
            let [image] = images.as_slice() else {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "layer decomposition takes exactly one image in `image_urls`"
                        .to_string(),
                }));
            };
            body.insert("image".into(), image.clone());
            body.insert("layer_decomposition".into(), Value::Bool(true));
            let mut options = Map::new();
            if let Some(max) = input.get("max_layers").and_then(Value::as_u64) {
                options.insert("max_images".into(), Value::from(max));
            }
            if let Some(ocr) = input.get("text_recognition").and_then(Value::as_bool) {
                options.insert("text_recognition".into(), Value::Bool(ocr));
            }
            if !options.is_empty() {
                body.insert("layer_decomposition_options".into(), Value::Object(options));
            }
        }
    }
    // A group of images (Seedream 4.x / 5.0 lite), asked for as `max_images`.
    if !layers
        && let Some(max) = input
            .get("max_images")
            .and_then(Value::as_u64)
            .filter(|m| *m > 1)
    {
        body.insert("sequential_image_generation".into(), json!("auto"));
        body.insert(
            "sequential_image_generation_options".into(),
            json!({ "max_images": max }),
        );
    }
    Ok(Value::Object(body))
}

/// Per successful image: its URL and the metadata RouterBase keeps with it
/// (size, format and, for layers, z-index, name, bounding box and text).
fn result_items(response: &Value) -> Vec<Value> {
    const KEPT: [&str; 7] = [
        "size",
        "output_format",
        "z_index",
        "name",
        "description",
        "bounding_box",
        "text_recognition",
    ];
    response["data"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter(|item| item["url"].as_str().is_some_and(|u| !u.is_empty()))
                .map(|item| {
                    let mut kept = Map::new();
                    kept.insert("url".into(), item["url"].clone());
                    for key in KEPT {
                        if let Some(v) = item.get(key).filter(|v| !v.is_null()) {
                            kept.insert(key.into(), v.clone());
                        }
                    }
                    Value::Object(kept)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The pixel count of each image Ark returned (`size` is `WIDTHxHEIGHT`), so
/// RouterBase can price every output by its resolution tier.
fn output_pixels(items: &[Value]) -> Vec<u64> {
    items
        .iter()
        .filter_map(|item| {
            let (w, h) = item["size"].as_str()?.split_once('x')?;
            Some(w.trim().parse::<u64>().ok()? * h.trim().parse::<u64>().ok()?)
        })
        .collect()
}

/// Ark's `usage` plus `output_pixels` (one entry per returned image).
fn callback_usage(response: &Value, items: &[Value]) -> Option<Value> {
    let mut usage = response
        .get("usage")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let pixels = output_pixels(items);
    if !pixels.is_empty() {
        usage.insert("output_pixels".into(), json!(pixels));
    }
    (!usage.is_empty()).then_some(Value::Object(usage))
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
    items: &[Value],
    usage: Option<&Value>,
) -> Result<(), Error> {
    let body = media_callback_success_body(task_id, urls, items, usage);
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
/// same as the other media providers), with Ark's `usage` (plus
/// `output_pixels`) passed through and each image's metadata as
/// `resultItems`.
fn media_callback_success_body(
    task_id: &str,
    urls: &[String],
    items: &[Value],
    usage: Option<&Value>,
) -> Value {
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
    if !items.is_empty() {
        body["resultItems"] = json!(items);
        body["data"]["resultItems"] = json!(items);
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
            "size": "2560x1440",
            "n": 1,
            "resolution": "1K",
            "aspect_ratio": "16:9",
            "sequential_image_generation": "auto",
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
                "size": "2560x1440",
                "watermark": false,
                "response_format": "url",
            }),
            "RouterBase bookkeeping is dropped, a caller's sequential flag isn't forwarded, watermark defaults off"
        );
    }

    #[test]
    fn model_options_pass_through() {
        let input = json!({
            "prompt": "x", "watermark": true, "background": "transparent", "output_format": "png",
            "optimize_prompt_mode": "fast", "safety_identifier": "u-abc",
        });
        let body = build_body(&ArkRequestShape::SeedreamTextToImage, "m", &input).unwrap();
        assert_eq!(
            body["watermark"],
            json!(true),
            "an explicit watermark=true is forwarded"
        );
        assert_eq!(
            body["background"],
            json!("transparent"),
            "5.5's background is forwarded"
        );
        assert_eq!(
            body["output_format"],
            json!("png"),
            "output_format is forwarded"
        );
        assert_eq!(
            body["optimize_prompt_options"],
            json!({"mode": "fast"}),
            "optimize_prompt_mode becomes Ark's optimize_prompt_options.mode"
        );
        assert_eq!(
            body["safety_identifier"],
            json!("u-abc"),
            "the hashed caller id is forwarded"
        );
    }

    #[test]
    fn a_group_is_asked_for_with_max_images() {
        let body = build_body(
            &ArkRequestShape::SeedreamTextToImage,
            "m",
            &json!({"prompt": "a 4-panel comic", "max_images": 4}),
        )
        .unwrap();
        assert_eq!(
            body["sequential_image_generation"],
            json!("auto"),
            "max_images > 1 turns the group on"
        );
        assert_eq!(
            body["sequential_image_generation_options"],
            json!({"max_images": 4}),
            "the group is capped at what RouterBase held for"
        );
        let single = build_body(
            &ArkRequestShape::SeedreamTextToImage,
            "m",
            &json!({"prompt": "x", "max_images": 1}),
        )
        .unwrap();
        assert!(
            single.get("sequential_image_generation").is_none(),
            "one image needs no group (Seedream 5.5 rejects the field)"
        );
    }

    #[test]
    fn edit_sends_reference_images_and_requires_one() {
        let input = json!({"prompt": "make it blue", "image_urls": ["https://a/1.png", "", "https://a/2.png"]});
        let body = build_body(&ArkRequestShape::SeedreamEdit, "m", &input).unwrap();
        assert_eq!(
            body["image"],
            json!(["https://a/1.png", "https://a/2.png"]),
            "reference images go out as Ark's `image` array, blanks dropped"
        );
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
    fn layers_split_exactly_one_image() {
        let body = build_body(
            &ArkRequestShape::SeedreamLayers,
            "doubao-seedream-5-5-261028",
            &json!({"image_urls": ["https://a/poster.png"], "max_layers": 12, "text_recognition": true,
                    "size": "auto", "background": "transparent", "max_images": 5}),
        )
        .unwrap();
        assert_eq!(
            body,
            json!({
                "model": "doubao-seedream-5-5-261028",
                "image": "https://a/poster.png",
                "size": "auto",
                "layer_decomposition": true,
                "layer_decomposition_options": {"max_images": 12, "text_recognition": true},
                "watermark": false,
                "response_format": "url",
            }),
            "no prompt needed; one image as a string; no background or group in layer mode"
        );
        for images in [json!([]), json!(["https://a/1.png", "https://a/2.png"])] {
            assert!(
                build_body(
                    &ArkRequestShape::SeedreamLayers,
                    "m",
                    &json!({"image_urls": images})
                )
                .is_err(),
                "layer decomposition takes exactly one image: {images}"
            );
        }
    }

    #[test]
    fn response_items_keep_metadata_and_skip_failures() {
        let response = json!({
            "data": [
                {"url": "https://out/0.png", "size": "2048x2048", "output_format": "png", "z_index": 0},
                {"error": {"code": "OutputImageSensitiveContentDetected", "message": "blocked"}},
                {"url": "https://out/1.png", "size": "1024x512", "z_index": 1, "name": "title",
                 "bounding_box": {"absolute": [10, 20, 1034, 532]}, "description": null}
            ],
            "usage": {"generated_images": 2, "output_tokens": 18432}
        });
        let items = result_items(&response);
        assert_eq!(
            items,
            vec![
                json!({"url": "https://out/0.png", "size": "2048x2048", "output_format": "png", "z_index": 0}),
                json!({"url": "https://out/1.png", "size": "1024x512", "z_index": 1, "name": "title",
                       "bounding_box": {"absolute": [10, 20, 1034, 532]}}),
            ],
            "failed items are skipped, null fields dropped, layer metadata kept"
        );
        assert_eq!(
            callback_usage(&response, &items),
            Some(json!({"generated_images": 2, "output_tokens": 18432,
                        "output_pixels": [2048 * 2048, 1024 * 512]})),
            "every returned image's pixel count rides on the usage, for tier billing"
        );

        let rejected = json!({"error": {"code": "InvalidParameter", "message": "size is invalid", "type": "BadRequest"}});
        assert_eq!(
            rejection(&rejected).as_deref(),
            Some("Ark rejected the request (InvalidParameter): size is invalid"),
            "RouterBase strips the `Ark ... : ` prefix and shows the upstream message"
        );
        assert_eq!(
            rejection(&json!({"data": []})),
            None,
            "a body without an error is not a rejection"
        );
    }

    #[test]
    fn callback_carries_urls_items_and_usage() {
        let urls = vec!["https://out/1.jpeg".to_string()];
        let items = vec![json!({"url": "https://out/1.jpeg", "size": "1024x1024"})];
        let usage = json!({"generated_images": 1, "output_pixels": [1048576]});
        let body = media_callback_success_body("ark-1", &urls, &items, Some(&usage));
        assert_eq!(body["state"], "success", "a success callback");
        assert_eq!(
            body["resultUrls"],
            json!(urls),
            "the image URLs RouterBase mirrors"
        );
        assert_eq!(body["resultItems"], json!(items), "each image's metadata");
        assert_eq!(body["usage"], usage, "the usage RouterBase settles from");
        assert_eq!(
            body["data"]["usage"], usage,
            "and mirrored under data, like the other providers"
        );
        let bare = media_callback_success_body("ark-1", &urls, &[], None);
        assert!(
            bare.get("usage").is_none() && bare.get("resultItems").is_none(),
            "no usage or items keys when there are none"
        );
    }

    #[test]
    fn rendered_shape_names_deserialize() {
        for (name, shape) in [
            (
                "seedream_text_to_image",
                ArkRequestShape::SeedreamTextToImage,
            ),
            ("seedream_edit", ArkRequestShape::SeedreamEdit),
            ("seedream_layers", ArkRequestShape::SeedreamLayers),
        ] {
            let proxy: ArkMediaProxyConfig = serde_json::from_value(json!({
                "path": "doubao-seedream-5-5-261028",
                "api_base": "https://ark.cn-beijing.volces.com/api/v3",
                "request_shape": name,
            }))
            .unwrap_or_else(|e| panic!("`{name}` must deserialize: {e}"));
            assert_eq!(
                proxy.request_shape, shape,
                "`{name}` is the shape RouterBase renders"
            );
        }
    }
}
