use lazy_static::lazy_static;
use schemars::JsonSchema;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;
use uuid::Uuid;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;

const PROVIDER_NAME: &str = "Novita";
const PROVIDER_TYPE: &str = "novita";
/// Per-HTTP-request timeout for individual calls to Novita (create-task
/// submit, single poll fetch). Bounds one network round-trip, not the whole
/// generation.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
/// Total wall-clock budget for a single async media generation to reach a
/// terminal state. Heavy models (e.g. `sora_2_pro_i2v`) routinely render well
/// past the old 300s; allow up to 1h before declaring a timeout. The
/// RouterBase worker does NOT retry after this fires (`MAX_ATTEMPTS = 1`), so
/// this is the one and only budget for a generation — no new Novita task is
/// ever spawned for the same request.
const ASYNC_TASK_TIMEOUT: Duration = Duration::from_secs(3600);

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
    /// OpenAI Sora 2, text-to-video (basic). `professional` is forced
    /// to `false` server-side; the operator picks the Pro tier by
    /// routing to the `Sora2ProTextToVideo` variant instead.
    Sora2TextToVideo,
    /// OpenAI Sora 2, text-to-video (Pro). Same Novita endpoint as
    /// `Sora2TextToVideo` but `professional` is forced to `true`.
    Sora2ProTextToVideo,
    /// OpenAI Sora 2, image-to-video (basic).
    Sora2ImageToVideo,
    /// OpenAI Sora 2, image-to-video (Pro).
    Sora2ProImageToVideo,
    /// Kling v3.0 4K, text-to-video. Per Novita's
    /// `/v3/async/kling-v3.0-4k-t2v` doc: prompt (auto),
    /// negative_prompt, aspect_ratio, duration (int 3–15),
    /// cfg_scale (0–1), sound.
    #[serde(rename = "kling_v3_4k_text_to_video")]
    KlingV34kTextToVideo,
    /// Kling v3.0 4K, image-to-video. Per
    /// `/v3/async/kling-v3.0-4k-i2v`: image (URL), prompt,
    /// negative_prompt, end_image, duration, cfg_scale, sound.
    /// `multi_prompt` is not surfaced — incompatible with end_image
    /// and would need a separate request_shape if/when we expose it.
    #[serde(rename = "kling_v3_4k_image_to_video")]
    KlingV34kImageToVideo,
    /// Kling v3.0 Motion Control. Image + reference video; the
    /// reference video's motion is transferred onto the still image.
    /// Per `/v3/async/kling-v3.0-motion-control`: image, video,
    /// prompt, negative_prompt, model_name (std/pro tier),
    /// keep_original_sound, character_orientation.
    KlingV3MotionControl,
    /// Wan 2.7 text-to-video. Per `/v3/async/wan2.7-t2v`: prompt
    /// (auto), duration (int 2–15), size (e.g. "1920*1080"), seed,
    /// audio_url (optional), negative_prompt (≤500), watermark,
    /// prompt_extend.
    #[serde(rename = "wan_2_7_text_to_video")]
    Wan27TextToVideo,
    /// Wan 2.7 image-to-video. Per `/v3/async/wan2.7-i2v`: prompt
    /// (auto, optional), image_url (remapped from image_urls[0]),
    /// duration (int 2–15), resolution (720P|1080P), seed,
    /// negative_prompt, watermark, prompt_extend, driving_audio_url,
    /// last_frame_url. `first_clip_url` (video continuation) is not
    /// surfaced — would need a separate variant if exposed.
    #[serde(rename = "wan_2_7_image_to_video")]
    Wan27ImageToVideo,
    /// Wan 2.7 reference-to-video. Per `/v3/async/wan2.7-r2v`: prompt
    /// (auto), media (array of 1–5 image/video reference URLs),
    /// duration (int 2–10), size, seed, audio (bool, default true),
    /// shot_type (single|multi), watermark, negative_prompt.
    /// Media array is built from `image_urls` + `video_urls` below.
    #[serde(rename = "wan_2_7_reference_to_video")]
    Wan27ReferenceToVideo,
    /// Wan 2.7 video editing. Per `/v3/async/wan2.7-videoedit`:
    /// video_url (required, remapped from video_urls[0]), prompt
    /// (auto, optional), duration (int 0–10, 0 preserves input),
    /// ratio (16:9|9:16|1:1|4:3|3:4), resolution (720P|1080P),
    /// audio_setting (auto|origin), seed, watermark, prompt_extend,
    /// negative_prompt, reference_image_url(_2,_3) (up to 3,
    /// remapped from image_urls[0..3]).
    #[serde(rename = "wan_2_7_video_edit")]
    Wan27VideoEdit,
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
                message: "media proxy requires a callback_url".to_string(),
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
                message: "Novita-backed media variants require a prompt".to_string(),
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
        // Sora 2 text-to-video. Per Novita's
        // `/v3/async/sora-2-text2video` doc: prompt (auto), size,
        // duration. `professional` is set explicitly below based on
        // the shape (basic vs Pro).
        NovitaRequestShape::Sora2TextToVideo | NovitaRequestShape::Sora2ProTextToVideo => {
            &["size", "duration"]
        }
        // Sora 2 image-to-video. Per `/v3/async/sora-2-img2video`:
        // prompt (auto), image (URL or Base64 string — passed
        // through as a single `image` field, no URL/Base64 split),
        // resolution, duration.
        NovitaRequestShape::Sora2ImageToVideo | NovitaRequestShape::Sora2ProImageToVideo => {
            &["image", "resolution", "duration"]
        }
        // Kling v3.0 4K, text-to-video. Per
        // `/v3/async/kling-v3.0-4k-t2v`: prompt (auto), enum
        // aspect_ratio, integer duration (3–15), float cfg_scale
        // (0–1), bool sound, optional negative_prompt.
        NovitaRequestShape::KlingV34kTextToVideo => &[
            "negative_prompt",
            "aspect_ratio",
            "duration",
            "cfg_scale",
            "sound",
        ],
        // Kling v3.0 4K, image-to-video. Per
        // `/v3/async/kling-v3.0-4k-i2v`: prompt (auto), image (URL,
        // remapped from `image_urls[0]` below), duration, cfg_scale,
        // sound, negative_prompt, end_image. Multi-shot composition
        // is exposed via Novita's `multi_prompt` array — not
        // surfaced here because it's mutually exclusive with
        // `end_image` and we'd want a separate variant if/when we
        // ship it.
        NovitaRequestShape::KlingV34kImageToVideo => &[
            "negative_prompt",
            "duration",
            "cfg_scale",
            "sound",
            "end_image",
        ],
        // Kling v3.0 Motion Control. Per
        // `/v3/async/kling-v3.0-motion-control`: image (remapped
        // from `image_urls[0]`), video (remapped from
        // `video_urls[0]`), prompt (auto, optional),
        // negative_prompt, model_name (std/pro tier),
        // keep_original_sound, character_orientation (image|video).
        NovitaRequestShape::KlingV3MotionControl => &[
            "negative_prompt",
            "model_name",
            "keep_original_sound",
            "character_orientation",
        ],
        // Wan 2.7 text-to-video. Per `/v3/async/wan2.7-t2v`: prompt
        // (auto), duration (int 2–15), size, seed, audio_url,
        // negative_prompt (≤500), watermark, prompt_extend.
        NovitaRequestShape::Wan27TextToVideo => &[
            "duration",
            "size",
            "seed",
            "audio_url",
            "negative_prompt",
            "watermark",
            "prompt_extend",
        ],
        // Wan 2.7 image-to-video. Per `/v3/async/wan2.7-i2v`: prompt
        // (auto, optional), image_url (remapped from
        // `image_urls[0]`), duration, resolution (720P|1080P), seed,
        // negative_prompt, watermark, prompt_extend,
        // driving_audio_url, last_frame_url. `first_clip_url` (video
        // continuation) is not surfaced; if exposed, give it its own
        // variant since it's mutually exclusive with image_url.
        NovitaRequestShape::Wan27ImageToVideo => &[
            "duration",
            "resolution",
            "seed",
            "negative_prompt",
            "watermark",
            "prompt_extend",
            "driving_audio_url",
            "last_frame_url",
        ],
        // Wan 2.7 reference-to-video. Per `/v3/async/wan2.7-r2v`:
        // prompt (auto), media (array of refs, built from
        // image_urls + video_urls below), duration (int 2–10), size,
        // seed, audio (bool), shot_type (single|multi), watermark,
        // negative_prompt.
        NovitaRequestShape::Wan27ReferenceToVideo => &[
            "duration",
            "size",
            "seed",
            "audio",
            "shot_type",
            "watermark",
            "negative_prompt",
            "media",
        ],
        // Wan 2.7 video editing. Per `/v3/async/wan2.7-videoedit`:
        // video_url (remapped from `video_urls[0]`), prompt (auto,
        // optional), duration (int 0–10, 0 = preserve input
        // length), ratio (5 enum), resolution (720P|1080P),
        // audio_setting (auto|origin), seed, watermark,
        // prompt_extend, negative_prompt, reference_image_url(_2,_3)
        // (up to 3, remapped from image_urls[0..3] below).
        NovitaRequestShape::Wan27VideoEdit => &[
            "duration",
            "ratio",
            "resolution",
            "audio_setting",
            "seed",
            "watermark",
            "prompt_extend",
            "negative_prompt",
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

    // Sora 2 routes the Pro/non-Pro distinction through a single Novita
    // endpoint with a `professional` body field. Force the value
    // server-side so the user can't accidentally upgrade to Pro by
    // passing `professional: true` to the basic variant.
    //
    // Also coerce `duration` to an integer. Novita's Sora 2 schema
    // strictly requires `integer` (enum: 4/8/12), but RouterBase's api
    // surface ships `duration` as a string for parity with the Veo
    // variants — a 400 from Novita is the user-visible failure mode.
    if matches!(
        shape,
        NovitaRequestShape::Sora2TextToVideo
            | NovitaRequestShape::Sora2ProTextToVideo
            | NovitaRequestShape::Sora2ImageToVideo
            | NovitaRequestShape::Sora2ProImageToVideo
    ) {
        let pro = matches!(
            shape,
            NovitaRequestShape::Sora2ProTextToVideo | NovitaRequestShape::Sora2ProImageToVideo
        );
        body.insert("professional".into(), Value::Bool(pro));

        if let Some(duration_str) = body.get("duration").and_then(Value::as_str) {
            if let Ok(duration_int) = duration_str.parse::<i64>() {
                body.insert("duration".into(), Value::from(duration_int));
            }
        }
    }

    // Sora 2 image-to-video: Novita's body field is `image` (single string,
    // URL or Base64). RouterBase's playground sends `image_urls` as an array
    // for parity with Veo's i2v form. Translate the first element through.
    if matches!(
        shape,
        NovitaRequestShape::Sora2ImageToVideo | NovitaRequestShape::Sora2ProImageToVideo
    ) && !body.contains_key("image")
    {
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

    // Sora 2 Pro text-to-video: Novita's body field is `size` (e.g.
    // `720*1280`, `1024*1792`). The playground / parameter_schema exposes
    // a simpler `resolution` enum (720p / 1080p) for symmetry with Pro
    // i2v's native `resolution` field. Translate to a Pro-supported `size`,
    // defaulting to portrait orientation (1280h / 1792h) which is the
    // Novita default and the most common Sora 2 use-case.
    if matches!(shape, NovitaRequestShape::Sora2ProTextToVideo) && !body.contains_key("size") {
        let res = input.get("resolution").and_then(Value::as_str);
        let size = match res {
            Some("1080p") => Some("1024*1792"),
            Some("720p") => Some("720*1280"),
            _ => None,
        };
        if let Some(s) = size {
            body.insert("size".into(), Value::from(s));
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

    // Kling v3.0 4K I2V + Motion Control: Novita body fields are `image`
    // (single string URL) and, for Motion Control, `video` (single URL).
    // The playground / parameter_schema exposes `image_urls` / `video_urls`
    // as arrays for parity with Veo + Sora i2v. Pluck the first element
    // through. Also forwards the user's `prompt` since these endpoints
    // accept it as a body field (not auto-injected via the messages array).
    if matches!(
        shape,
        NovitaRequestShape::KlingV34kImageToVideo | NovitaRequestShape::KlingV3MotionControl
    ) {
        if !body.contains_key("image") {
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
        if !body.contains_key("prompt") {
            if let Some(value) = input.get("prompt").and_then(Value::as_str) {
                body.insert("prompt".into(), Value::from(value));
            }
        }
    }

    if matches!(shape, NovitaRequestShape::KlingV3MotionControl) && !body.contains_key("video") {
        if let Some(value) = input.get("video").and_then(Value::as_str) {
            body.insert("video".into(), Value::from(value));
        } else if let Some(first) = input
            .get("video_urls")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(Value::as_str)
        {
            body.insert("video".into(), Value::from(first));
        }
    }

    // Kling v3.0 4K T2V: forward `prompt` from the input. Same handling
    // as Sora 2 / Veo T2V — the playground sends it as a top-level
    // field, not in a messages array.
    if matches!(shape, NovitaRequestShape::KlingV34kTextToVideo) && !body.contains_key("prompt") {
        if let Some(value) = input.get("prompt").and_then(Value::as_str) {
            body.insert("prompt".into(), Value::from(value));
        }
    }

    // Wan 2.7 I2V: body wants `image_url` (single string). Playground
    // sends `image_urls` array for parity with Veo/Sora/Kling i2v.
    if matches!(shape, NovitaRequestShape::Wan27ImageToVideo) && !body.contains_key("image_url") {
        if let Some(value) = input.get("image_url").and_then(Value::as_str) {
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

    // Wan 2.7 Video Editing: body wants `video_url` (single string,
    // required). Playground sends `video_urls` array.
    if matches!(shape, NovitaRequestShape::Wan27VideoEdit) && !body.contains_key("video_url") {
        if let Some(value) = input.get("video_url").and_then(Value::as_str) {
            body.insert("video_url".into(), Value::from(value));
        } else if let Some(first) = input
            .get("video_urls")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(Value::as_str)
        {
            body.insert("video_url".into(), Value::from(first));
        }
    }

    // Wan 2.7 Video Editing: up to 3 reference images, body wants
    // `reference_image_url`, `reference_image_url_2`,
    // `reference_image_url_3` (each a single string). Playground
    // ships `image_urls` as a flat array — split into the three
    // Wan-specific fields. Only forwarded when the caller didn't
    // already set them explicitly.
    if matches!(shape, NovitaRequestShape::Wan27VideoEdit) {
        if let Some(imgs) = input.get("image_urls").and_then(Value::as_array) {
            let slot_names = [
                "reference_image_url",
                "reference_image_url_2",
                "reference_image_url_3",
            ];
            for (idx, slot) in slot_names.iter().enumerate() {
                if body.contains_key(*slot) {
                    continue;
                }
                if let Some(url) = imgs.get(idx).and_then(Value::as_str) {
                    body.insert((*slot).to_string(), Value::from(url));
                }
            }
        }
    }

    // Wan 2.7 R2V: body wants `media` — an array of objects with a
    // `type` (`reference_image`|`reference_video`|`first_frame`) +
    // `url`, plus optional per-item `reference_voice` (voice-clone
    // audio, MP3/WAV/FLAC, 3–30s).
    //
    // Two input shapes are supported:
    //   (a) Playground/legacy flat shape: `image_urls` + `video_urls`
    //       arrays. Each entry becomes a `reference_image` /
    //       `reference_video` media item. No way to express
    //       `first_frame` or `reference_voice` in this shape.
    //   (b) New rich shape: caller passes a `media` array of objects
    //       directly. Used by the SPA's media-editor UI and any
    //       direct API caller. We pass it through verbatim, just
    //       capping at Novita's 5-item ceiling.
    //
    // Novita enforces (total ≤ 5; images 0–5; videos 0–3) on its end,
    // so we don't double-validate beyond the 5-cap.
    if matches!(shape, NovitaRequestShape::Wan27ReferenceToVideo) && !body.contains_key("media") {
        let mut media: Vec<Value> = Vec::new();
        if let Some(imgs) = input.get("image_urls").and_then(Value::as_array) {
            for u in imgs.iter().filter_map(Value::as_str) {
                media.push(json!({ "type": "reference_image", "url": u }));
            }
        }
        if let Some(vids) = input.get("video_urls").and_then(Value::as_array) {
            for u in vids.iter().filter_map(Value::as_str) {
                media.push(json!({ "type": "reference_video", "url": u }));
            }
        }
        if !media.is_empty() {
            if media.len() > 5 {
                media.truncate(5);
            }
            body.insert("media".into(), Value::Array(media));
        }
    }

    // Wan 2.7 T2V/I2V/R2V/Video Edit all accept `prompt` as a body
    // field. The shape-specific arms above explicitly *omit* "prompt"
    // from `allowed` because it's already inserted at the top of
    // build_body (line ~184). This block is a no-op for them, but
    // kept as a safety net mirroring the Kling pattern in case the
    // upstream caller paths change.
    if matches!(
        shape,
        NovitaRequestShape::Wan27TextToVideo
            | NovitaRequestShape::Wan27ImageToVideo
            | NovitaRequestShape::Wan27ReferenceToVideo
            | NovitaRequestShape::Wan27VideoEdit
    ) && !body.contains_key("prompt")
    {
        if let Some(value) = input.get("prompt").and_then(Value::as_str) {
            body.insert("prompt".into(), Value::from(value));
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
                        item.as_str().map(ToString::to_string).or_else(|| {
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
    let url = format!(
        "{}/v3/async/task-result?task_id={task_id}",
        *NOVITA_API_BASE
    );
    let deadline = Instant::now() + ASYNC_TASK_TIMEOUT;
    let poll_interval = Duration::from_secs(4);

    loop {
        if Instant::now() >= deadline {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Novita async task {task_id} did not complete within {}s",
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
