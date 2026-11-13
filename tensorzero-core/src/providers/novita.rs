use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use lazy_static::lazy_static;
use schemars::JsonSchema;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;
use uuid::Uuid;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::utils::spawn_ignoring_shutdown;

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
    // Empty counts as unset: docker-compose passes `${NOVITA_API_BASE:-}`, and
    // an empty base yields "relative URL without a base" on every media call
    // (Amux / MiniMax already filter the same way).
    static ref NOVITA_API_BASE: String = std::env::var("NOVITA_API_BASE")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://api.novita.ai".to_string());
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
    /// Upstream body `model` for shapes that serve several checkpoints from
    /// one endpoint: the OAI-native GPT Image shapes (`gpt-image-2.5-flare-oai`,
    /// …; unset keeps `gpt-image-2-oai`) and Wan 3.0 (`wan3.0-video` /
    /// `wan3.0-video-prime`; unset keeps `wan3.0-video`). Ignored by the
    /// other shapes.
    #[serde(default)]
    pub model: Option<Arc<str>>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum NovitaRequestShape {
    GeminiImageTextToImage,
    GeminiImageEdit,
    /// Gemini image models served only on Novita's Gemini-native protocol
    /// (`{base}/gemini/v1beta/models/{model}:generateContent`, `proxy.path`
    /// carries that suffix), e.g. Nano Banana 2.1. Synchronous: the image comes
    /// back inline (`inlineData`) with `usageMetadata`, which is forwarded as
    /// OpenAI-style `usage` for token billing.
    GeminiNativeTextToImage,
    /// Gemini-native edit: as above, with each `image_urls` entry fetched and
    /// sent inline next to the prompt.
    GeminiNativeEdit,
    /// GPT Image 2 via Novita's OpenAI-compatible "native protocol"
    /// (`/openai/v1/images/generations`, token-billed upstream). Synchronous:
    /// the response carries `data[].b64_json`, no hosted URL. The body `model`
    /// is injected below (`gpt-image-2`; Novita also offers `gpt-image-2-oai`).
    GptImageOaiTextToImage,
    /// GPT Image 2 edit via `/openai/v1/images/edits`. multipart/form-data with
    /// repeated `image[]` file parts (real multi-reference editing, unlike the
    /// v3 async shape which took a single `image` URL) and an optional `mask`.
    GptImageOaiEdit,
    /// ByteDance Seedream 3.0 text-to-image. Per `/v3/seedream-3-0-txt2img`:
    /// prompt (required), seed, response_format (url|b64_json), watermark. This
    /// is a SYNCHRONOUS endpoint (returns `image_urls` inline, no task_id). The
    /// upstream also requires a fixed body `model` string, injected below.
    /// `size` and `guidance_scale` are documented but the checkpoint TASK_FAILs
    /// whenever either is present, so they are NOT forwarded (see allow-list).
    #[serde(rename = "seedream_v3_text_to_image")]
    SeedreamV3TextToImage,
    /// ByteDance Seedream 4.0 text-to-image. Per `/v3/seedream-4.0`: prompt
    /// (required), size (1K/2K/4K or WxH), sequential_image_generation
    /// (auto|disabled), max_images (1–15), watermark. Synchronous (`images`).
    #[serde(rename = "seedream_v4_text_to_image")]
    SeedreamV4TextToImage,
    /// ByteDance Seedream 4.5 / 5.0-lite text-to-image — identical param
    /// surface, so one shape. Per `/v3/seedream-4.5` and `/v3/seedream-5.0-lite`:
    /// prompt (required), size (2K/4K/WxH; 5.0-lite also 3K), watermark,
    /// sequential_image_generation. Synchronous (`images`).
    #[serde(rename = "seedream_image")]
    SeedreamImage,
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
    /// Kling v3.0 4K, text-to-video, on Kling's original-factory protocol
    /// (`POST /v3/kling/text-to-video/kling-3.0`, see the Kling / Wan section below): prompt,
    /// aspect_ratio, duration (3-15), sound -> `settings.audio`, multi_shot.
    /// Resolution is fixed to `4k`. The old `/v3/async/kling-v3.0-4k-t2v`
    /// route was withdrawn (404) in 2026-11.
    #[serde(rename = "kling_v3_4k_text_to_video")]
    KlingV34kTextToVideo,
    /// Kling v3.0 4K, image-to-video (`/v3/kling/image-to-video/kling-3.0`):
    /// `image_urls[0]` is the first frame, `end_image` the last; duration,
    /// sound, multi_shot. Resolution fixed to `4k`.
    #[serde(rename = "kling_v3_4k_image_to_video")]
    KlingV34kImageToVideo,
    /// Kling v3.0 Motion Control (`/v3/kling/motion-control/kling-3.0`): the
    /// reference video's motion is transferred onto the still image.
    /// `model_name` picks the tier (`-pro` = 1080p, `-std` = 720p),
    /// keep_original_sound -> `settings.audio`, character_orientation.
    KlingV3MotionControl,
    /// Kling 4.0 text-to-video (`/v3/kling/text-to-video/kling-4.0`):
    /// prompt, resolution (720p|1080p), aspect_ratio, duration (3-30, or
    /// omitted to let Kling decide). Billed by the seconds produced.
    #[serde(rename = "kling_v4_text_to_video")]
    KlingV4TextToVideo,
    /// Kling 4.0 multi-keyframe image-to-video
    /// (`/v3/kling/image-to-video/kling-4.0`): `image_urls[0]` first frame,
    /// `middle_image_urls[]` middle frames, `end_image` last frame (10
    /// images at most, addressable as `@first`, `@middle1`.., `@last`).
    #[serde(rename = "kling_v4_image_to_video")]
    KlingV4ImageToVideo,
    /// Kling 4.0 omni reference video (`/v3/kling/omni-video/kling-4.0`):
    /// reference images (`image_urls`, `@image1`..), feature videos
    /// (`video_urls`, `@video1`..) and a video to edit (`base_video_url`,
    /// `@base`). Video input bills at the video-input tier on input + output
    /// seconds.
    #[serde(rename = "kling_v4_omni_video")]
    KlingV4OmniVideo,
    /// Alibaba Wan 3.0 text-to-video on the DashScope protocol
    /// (`/v3/aliyun/.../video-synthesis`, see the Kling / Wan section below). The body `model`
    /// (`wan3.0-video` / `wan3.0-video-prime`) comes from the variant's
    /// `model`. Billed by `ceil(input video + output)` seconds.
    #[serde(rename = "wan_3_text_to_video")]
    Wan3TextToVideo,
    /// Wan 3.0 first-frame (optionally first+last) image-to-video.
    #[serde(rename = "wan_3_image_to_video")]
    Wan3ImageToVideo,
    /// Wan 3.0 reference-to-video: reference images, videos and audio
    /// (also video extension / editing, steered by the prompt).
    #[serde(rename = "wan_3_reference_to_video")]
    Wan3ReferenceToVideo,
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
    /// Vidu Q1 text-to-video. Per `/v3/async/vidu-q1-text2video`: prompt
    /// (required), style (general|anime), duration (5 only), resolution
    /// (1080p only), aspect_ratio, movement_amplitude (auto|small|medium|
    /// large), seed, bgm. Q1 predates native audio — no `audio` field.
    #[serde(rename = "vidu_q1_text_to_video")]
    ViduQ1TextToVideo,
    /// Vidu Q1 image-to-video. Per `/v3/async/vidu-q1-img2video`: images
    /// (single-element array, remapped from `image_urls`), prompt
    /// (optional), duration (5), resolution (1080p), movement_amplitude,
    /// seed, bgm.
    #[serde(rename = "vidu_q1_image_to_video")]
    ViduQ1ImageToVideo,
    /// Vidu Q1 reference-to-video. Per `/v3/async/vidu-q1-reference2video`:
    /// images (1–7, remapped from `image_urls`), prompt (required),
    /// duration (5), resolution (1080p), aspect_ratio, movement_amplitude,
    /// seed, bgm.
    #[serde(rename = "vidu_q1_reference_to_video")]
    ViduQ1ReferenceToVideo,
    /// Vidu Q2 text-to-video. Per `/v3/async/vidu-q2-text2video`: prompt
    /// (required), duration (int 1–10), resolution (540p|720p|1080p),
    /// aspect_ratio, style, movement_amplitude (auto|small|medium|high),
    /// seed, bgm, audio (bool), watermark, subjects.
    #[serde(rename = "vidu_q2_text_to_video")]
    ViduQ2TextToVideo,
    /// Vidu Q2 image-to-video (shared by `vidu-q2-pro-img2video` and
    /// `vidu-q2-turbo-img2video`; they differ only by the slug in the URL
    /// path). images (single, remapped from `image_urls`), prompt
    /// (required), duration (1–10), resolution, movement_amplitude, seed,
    /// bgm, audio (bool), voice_id, watermark.
    #[serde(rename = "vidu_q2_image_to_video")]
    ViduQ2ImageToVideo,
    /// Vidu Q2 subject/reference-to-video (shared by `vidu-q2-reference2video`
    /// and `viduq2-pro-fast`). prompt (required), duration (1–10), subjects
    /// (array of `{name, images, voice_id}` — `name` required, prompt refers to a
    /// subject as `@{name}`; built from `image_urls` when absent), resolution,
    /// aspect_ratio, movement_amplitude, seed, bgm, audio (bool), watermark.
    #[serde(rename = "vidu_q2_subject_to_video")]
    ViduQ2SubjectToVideo,
    /// Vidu Q3 text-to-video (shared by `vidu-q3-turbo-t2v` and
    /// `vidu-q3-pro-t2v`). Native audio-video co-gen (`audio` defaults true).
    /// prompt (required), duration (int 1–16), resolution, aspect_ratio,
    /// seed, audio (bool), watermark, wm_url, wm_position.
    #[serde(rename = "vidu_q3_text_to_video")]
    ViduQ3TextToVideo,
    /// Vidu Q3 image-to-video (shared by `vidu-q3-turbo-i2v` and
    /// `vidu-q3-pro-i2v`). images (single, remapped from `image_urls`),
    /// prompt (optional), duration (1–16), resolution, aspect_ratio, style,
    /// seed, audio, audio_type, is_rec, watermark, wm_url, wm_position.
    /// NOTE: on `vidu-q3-pro-i2v` the `audio` value is an audio-track URL
    /// (lip-sync input) rather than a bool — the allow-list forwards
    /// whatever value the caller supplies, so both shapes share one arm.
    #[serde(rename = "vidu_q3_image_to_video")]
    ViduQ3ImageToVideo,
    /// Vidu Q3 start-end-to-video (shared by `vidu-q3-turbo-f2v` and
    /// `vidu-q3-pro-f2v`). images = exactly `[start_frame, end_frame]`
    /// (remapped from `image_urls`), prompt (optional), duration (1–16),
    /// resolution, seed, audio (bool), is_rec, watermark, wm_url,
    /// wm_position.
    #[serde(rename = "vidu_q3_first_last_to_video")]
    ViduQ3FirstLastToVideo,
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

        // Kling / Wan original-factory protocols: their own body, task and
        // poll shapes (the Kling / Wan section below), always submitted async with gateway polling.
        if original_factory_protocol(&proxy.request_shape).is_some() {
            return infer_original_factory(
                &proxy.request_shape,
                proxy.model.as_deref(),
                &proxy.path,
                NOVITA_API_BASE.as_str(),
                callback_url,
                input,
                http_client,
                api_key.expose_secret(),
            )
            .await;
        }

        let body = build_body(&proxy.request_shape, input)?;

        // OAI-native GPT Image 2: synchronous OpenAI-compatible endpoints under
        // /openai/v1 (proxy.path carries the full suffix), b64 response, no
        // task id. Handled apart from the /v3 flow below.
        if matches!(
            proxy.request_shape,
            NovitaRequestShape::GptImageOaiTextToImage | NovitaRequestShape::GptImageOaiEdit
        ) {
            let mut body_map = body.as_object().cloned().unwrap_or_default();
            if let Some(model) = &proxy.model {
                body_map.insert("model".into(), Value::from(model.as_ref()));
            }
            return infer_gpt_image_oai(
                proxy,
                callback_url,
                input,
                &body_map,
                http_client,
                &api_key,
            )
            .await;
        }

        // Gemini-native image models (Nano Banana 2.1): synchronous
        // `generateContent` under /gemini/v1beta (proxy.path carries the full
        // suffix), image inline, no task id.
        if matches!(
            proxy.request_shape,
            NovitaRequestShape::GeminiNativeTextToImage | NovitaRequestShape::GeminiNativeEdit
        ) {
            let body_map = body.as_object().cloned().unwrap_or_default();
            return infer_gemini_native(proxy, callback_url, &body_map, http_client, &api_key)
                .await;
        }

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

        if proxy.async_submission {
            // Async submission: Novita returns a `task_id` immediately and the
            // media renders out-of-band. We must NOT block the inference
            // response on the full render — RouterBase's submit call (and the
            // browser behind it) would hold the connection open for the entire
            // generation and trip the Cloudflare ~100s origin timeout (524),
            // even though the task succeeds server-side. Instead, return the
            // `task_id` now and poll + fire the callback in a detached task;
            // RouterBase records `pending` and settles when the callback lands.
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

            let http_client_bg = http_client.clone();
            let api_key_bg = api_key.expose_secret().to_string();
            let callback_url_bg = callback_url.to_string();
            let task_id_bg = task_id.clone();
            spawn_ignoring_shutdown(async move {
                match poll_async_result(&http_client_bg, &api_key_bg, &task_id_bg).await {
                    Ok(result_body) => {
                        let urls = parse_urls(&result_body);
                        if urls.is_empty() {
                            tracing::error!(
                                "Novita task {task_id_bg} completed but returned no media URLs"
                            );
                            post_media_callback_failure(
                                &http_client_bg,
                                &callback_url_bg,
                                &task_id_bg,
                                "Generation completed but returned no media",
                            )
                            .await;
                        } else if let Err(e) = post_media_callback(
                            &http_client_bg,
                            &callback_url_bg,
                            &task_id_bg,
                            &urls,
                            None,
                        )
                        .await
                        {
                            tracing::error!(
                                "Failed to deliver success callback for Novita task {task_id_bg}: {e}"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!("Novita task {task_id_bg} failed: {e}");
                        post_media_callback_failure(
                            &http_client_bg,
                            &callback_url_bg,
                            &task_id_bg,
                            &e.to_string(),
                        )
                        .await;
                    }
                }
            });

            return Ok(task_id);
        }

        // Synchronous submission: Novita returned the media inline. This path is
        // fast (no polling), so delivering the callback before returning does
        // not risk the origin timeout.
        let task_id = format!("novita-{}", Uuid::new_v4());
        let urls = parse_urls(&raw_json);
        if urls.is_empty() {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: "Novita completed but returned no media URLs".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: Some(raw_json.to_string()),
            }));
        }

        post_media_callback(http_client, callback_url, &task_id, &urls, None).await?;
        Ok(task_id)
    }
}

/// Synchronous GPT Image 2 call over Novita's OpenAI-compatible protocol.
///
/// Generation posts JSON to `{base}/openai/v1/images/generations`; edit posts
/// multipart/form-data to `{base}/openai/v1/images/edits` with each
/// `image_urls[]` entry downloaded and re-attached as an `image[]` file part
/// (plus an optional `mask`). Either way the response is OpenAI-shaped —
/// `data[].b64_json` — which is delivered to the media callback as `data:`
/// URIs for RouterBase to decode and mirror to R2.
async fn infer_gpt_image_oai(
    proxy: &NovitaMediaProxyConfig,
    callback_url: &str,
    input: &Value,
    body: &serde_json::Map<String, Value>,
    http_client: &TensorzeroHttpClient,
    api_key: &SecretString,
) -> Result<String, Error> {
    let url = format!("{}/{}", *NOVITA_API_BASE, proxy.path)
        .parse::<Url>()
        .map_err(|e| {
            Error::new(ErrorDetails::InvalidBaseUrl {
                message: format!("Failed to construct Novita URL: {e}"),
            })
        })?;

    let request = http_client
        .post(url)
        .bearer_auth(api_key.expose_secret())
        .timeout(REQUEST_TIMEOUT);
    let request = if matches!(proxy.request_shape, NovitaRequestShape::GptImageOaiEdit) {
        let form = build_gpt_image_oai_form(input, body, http_client).await?;
        request.multipart(form)
    } else {
        request.json(&Value::Object(body.clone()))
    };

    let response = request.send().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            message: format!("Novita request failed: {e}"),
            status_code: e.status(),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: Some(serde_json::to_string(body).unwrap_or_default()),
            raw_response: None,
        })
    })?;

    let status = response.status();
    let raw = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(Error::new(ErrorDetails::InferenceServer {
            message: format!("Novita returned {status}: {raw}"),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: Some(serde_json::to_string(body).unwrap_or_default()),
            raw_response: Some(raw),
        }));
    }

    let raw_json: Value = serde_json::from_str(&raw).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!("Failed to parse Novita response: {e}"),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: Some(serde_json::to_string(body).unwrap_or_default()),
            raw_response: Some(raw.clone()),
        })
    })?;

    // Actual token usage from the upstream (token-billed protocol). Forwarded
    // on the success callback: models that opt into usage billing are charged
    // from it; the rest keep their pricing matrix and ignore it.
    let usage = raw_json.get("usage");
    if let Some(usage) = usage {
        tracing::info!("gpt-image oai usage: {usage}");
    }

    let mime = match input.get("output_format").and_then(Value::as_str) {
        Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    };
    let urls: Vec<String> = raw_json
        .get("data")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    item.get("b64_json")
                        .and_then(Value::as_str)
                        .map(|b64| format!("data:{mime};base64,{b64}"))
                        .or_else(|| {
                            item.get("url")
                                .and_then(Value::as_str)
                                .map(ToString::to_string)
                        })
                })
                .collect()
        })
        .unwrap_or_default();
    if urls.is_empty() {
        return Err(Error::new(ErrorDetails::InferenceServer {
            message: "Novita completed but returned no image data".to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: Some(serde_json::to_string(body).unwrap_or_default()),
            raw_response: Some(raw_json.to_string()),
        }));
    }

    let task_id = format!("novita-{}", Uuid::new_v4());
    post_media_callback(http_client, callback_url, &task_id, &urls, usage).await?;
    Ok(task_id)
}

/// Synchronous Gemini-native image call (Nano Banana 2.1) over Novita's
/// `generateContent` passthrough.
///
/// Posts JSON to `{base}/{proxy.path}` (`gemini/v1beta/models/{model}:generateContent`).
/// The response carries the image as `inlineData`, delivered to the media
/// callback as a `data:` URI for RouterBase to decode and mirror to R2, plus
/// `usageMetadata`, forwarded as OpenAI-style `usage` (see `gemini_native_usage`).
async fn infer_gemini_native(
    proxy: &NovitaMediaProxyConfig,
    callback_url: &str,
    body: &serde_json::Map<String, Value>,
    http_client: &TensorzeroHttpClient,
    api_key: &SecretString,
) -> Result<String, Error> {
    let url = format!("{}/{}", *NOVITA_API_BASE, proxy.path)
        .parse::<Url>()
        .map_err(|e| {
            Error::new(ErrorDetails::InvalidBaseUrl {
                message: format!("Failed to construct Novita URL: {e}"),
            })
        })?;

    let mut images = Vec::new();
    if matches!(proxy.request_shape, NovitaRequestShape::GeminiNativeEdit) {
        let urls: Vec<&str> = body
            .get("image_urls")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if urls.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Gemini image edit requires at least one reference image".to_string(),
            }));
        }
        for url in urls {
            images.push(fetch_reference_image(http_client, url).await?);
        }
    }
    let request = build_gemini_native_request(body, &images);
    // Never log the inlined reference images.
    let request_summary = Value::Object(body.clone()).to_string();

    let response = http_client
        .post(url)
        .bearer_auth(api_key.expose_secret())
        .json(&request)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                message: format!("Novita request failed: {e}"),
                status_code: e.status(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(request_summary.clone()),
                raw_response: None,
            })
        })?;

    let status = response.status();
    let raw = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(Error::new(ErrorDetails::InferenceServer {
            message: format!("Novita returned {status}: {raw}"),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: Some(request_summary),
            raw_response: Some(raw),
        }));
    }

    let raw_json: Value = serde_json::from_str(&raw).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!("Failed to parse Novita response: {e}"),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: Some(request_summary.clone()),
            raw_response: None,
        })
    })?;

    let usage = gemini_native_usage(&raw_json);
    if let Some(usage) = &usage {
        tracing::info!("gemini-native image usage: {usage}");
    }

    let urls = gemini_native_images(&raw_json);
    if urls.is_empty() {
        return Err(Error::new(ErrorDetails::InferenceServer {
            message: gemini_native_no_image_message(&raw_json),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: Some(request_summary),
            raw_response: None,
        }));
    }

    let task_id = format!("novita-{}", Uuid::new_v4());
    post_media_callback(http_client, callback_url, &task_id, &urls, usage.as_ref()).await?;
    Ok(task_id)
}

/// Map the flat RouterBase params onto a `generateContent` request.
///
/// - `size` (or the api's default `resolution`) → `imageConfig.imageSize` when
///   it is a Gemini tier (`1K` / `2K` / `4K`); pixel sizes such as `1024x1024`
///   from OpenAI-style callers are dropped, leaving the upstream default (1K).
/// - `aspect_ratio` → `imageConfig.aspectRatio`.
/// - `thinking_level` (`minimal` / `medium` / `high`) → `thinkingConfig.thinkingLevel`.
/// - `web_search` / `image_search` → one `google_search` tool with those
///   `searchTypes`.
///
/// TEXT is always requested next to IMAGE, so a refusal comes back with the
/// model's explanation instead of a bare `NO_IMAGE`.
fn build_gemini_native_request(
    body: &serde_json::Map<String, Value>,
    images: &[(Vec<u8>, String)],
) -> Value {
    let mut parts = Vec::with_capacity(images.len() + 1);
    if let Some(prompt) = body.get("prompt").and_then(Value::as_str) {
        parts.push(json!({ "text": prompt }));
    }
    for (bytes, mime) in images {
        parts.push(json!({
            "inlineData": { "mimeType": mime, "data": BASE64_STANDARD.encode(bytes) }
        }));
    }

    let mut image_config = serde_json::Map::new();
    let tier = ["size", "resolution"].iter().find_map(|key| {
        let value = body.get(*key)?.as_str()?.trim().to_uppercase();
        matches!(value.as_str(), "1K" | "2K" | "4K").then_some(value)
    });
    if let Some(tier) = tier {
        image_config.insert("imageSize".into(), Value::from(tier));
    }
    if let Some(ratio) = body.get("aspect_ratio").and_then(Value::as_str) {
        image_config.insert("aspectRatio".into(), Value::from(ratio));
    }

    let mut generation_config = serde_json::Map::new();
    generation_config.insert("responseModalities".into(), json!(["TEXT", "IMAGE"]));
    if !image_config.is_empty() {
        generation_config.insert("imageConfig".into(), Value::Object(image_config));
    }
    if let Some(level) = body
        .get("thinking_level")
        .and_then(Value::as_str)
        .filter(|l| matches!(*l, "minimal" | "medium" | "high"))
    {
        generation_config.insert("thinkingConfig".into(), json!({ "thinkingLevel": level }));
    }

    let mut request = json!({
        "contents": [{ "role": "user", "parts": parts }],
        "generationConfig": generation_config,
    });
    let flag = |key: &str| body.get(key).and_then(Value::as_bool).unwrap_or(false);
    let mut search_types = serde_json::Map::new();
    if flag("web_search") {
        search_types.insert("webSearch".into(), json!({}));
    }
    if flag("image_search") {
        search_types.insert("imageSearch".into(), json!({}));
    }
    if !search_types.is_empty() {
        request["tools"] = json!([{ "google_search": { "searchTypes": search_types } }]);
    }
    request
}

/// The final images of a `generateContent` response as `data:` URIs. Interim
/// images the model drafts while thinking (`thought: true`) are skipped.
fn gemini_native_images(raw: &Value) -> Vec<String> {
    raw.get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|c| c.pointer("/content/parts").and_then(Value::as_array))
        .flatten()
        .filter(|part| !is_thought(part))
        .filter_map(|part| {
            let inline = part.get("inlineData").or_else(|| part.get("inline_data"))?;
            let data = inline.get("data").and_then(Value::as_str)?;
            let mime = inline
                .get("mimeType")
                .or_else(|| inline.get("mime_type"))
                .and_then(Value::as_str)
                .unwrap_or("image/png");
            Some(format!("data:{mime};base64,{data}"))
        })
        .collect()
}

/// Whether a response part is the model's thinking (`thought: true`), not output.
fn is_thought(part: &Value) -> bool {
    part.get("thought")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Error text for a response without an image: the finish reason plus whatever
/// the model said instead (a refusal is usually explained in a text part).
fn gemini_native_no_image_message(raw: &Value) -> String {
    let candidate = raw.pointer("/candidates/0");
    let reason = candidate
        .and_then(|c| c.get("finishReason"))
        .and_then(Value::as_str)
        .or_else(|| {
            raw.pointer("/promptFeedback/blockReason")
                .and_then(Value::as_str)
        })
        .unwrap_or("unknown");
    let text: Vec<&str> = candidate
        .and_then(|c| c.pointer("/content/parts"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|part| !is_thought(part))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect();
    if text.is_empty() {
        format!("Novita completed but returned no image (finish reason: {reason})")
    } else {
        format!(
            "Novita completed but returned no image (finish reason: {reason}): {}",
            text.join(" ")
        )
    }
}

/// Gemini `usageMetadata` as the OpenAI-style `usage` RouterBase bills from.
///
/// - `input_tokens` = `promptTokenCount`, split into text and everything else
///   (reference images) via `promptTokensDetails`; `cached_tokens` =
///   `cachedContentTokenCount`.
/// - `output_tokens` = `candidatesTokenCount` + `thoughtsTokenCount`, with
///   `output_tokens_details.image_tokens` the IMAGE share of the candidates
///   (all of them when the breakdown is missing, so a partial report never
///   moves image tokens to the cheaper text rate) and `text_tokens` the rest
///   (text and thinking).
/// - `search_queries` = the grounding queries the model ran (Google bills
///   grounding per query).
fn gemini_native_usage(raw: &Value) -> Option<Value> {
    let meta = raw.get("usageMetadata")?;
    let count = |v: &Value, key: &str| v.get(key).and_then(Value::as_u64).unwrap_or(0);
    let by_modality = |key: &str, modality: &str| -> Option<u64> {
        let details = meta.get(key)?.as_array()?;
        Some(
            details
                .iter()
                .filter(|d| d.get("modality").and_then(Value::as_str) == Some(modality))
                .map(|d| count(d, "tokenCount"))
                .sum(),
        )
    };

    let input = count(meta, "promptTokenCount");
    let text_in = by_modality("promptTokensDetails", "TEXT")
        .unwrap_or(input)
        .min(input);
    let candidates = count(meta, "candidatesTokenCount");
    let output = candidates + count(meta, "thoughtsTokenCount");
    let image_out = by_modality("candidatesTokensDetails", "IMAGE")
        .unwrap_or(candidates)
        .min(output);
    let search_queries: usize = raw
        .get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|c| c.get("groundingMetadata"))
        .map(|g| {
            ["webSearchQueries", "imageSearchQueries"]
                .iter()
                .filter_map(|k| g.get(*k).and_then(Value::as_array))
                .map(Vec::len)
                .sum::<usize>()
        })
        .sum();

    Some(json!({
        "input_tokens": input,
        "input_tokens_details": {
            "text_tokens": text_in,
            "image_tokens": input - text_in,
            "cached_tokens": count(meta, "cachedContentTokenCount"),
        },
        "output_tokens": output,
        "output_tokens_details": {
            "image_tokens": image_out,
            "text_tokens": output - image_out,
        },
        "search_queries": search_queries,
    }))
}

/// Build the multipart form for `/openai/v1/images/edits`: every string field
/// already vetted into `body` rides as a text part, and each reference image
/// (`image_urls`, ≤16 per the OpenAI protocol) plus the optional `mask` URL is
/// downloaded and attached as a file part.
async fn build_gpt_image_oai_form(
    input: &Value,
    body: &serde_json::Map<String, Value>,
    http_client: &TensorzeroHttpClient,
) -> Result<reqwest::multipart::Form, Error> {
    let mut form = reqwest::multipart::Form::new();
    for (key, value) in body {
        let text = match value {
            Value::String(v) => v.clone(),
            Value::Number(v) => v.to_string(),
            Value::Bool(v) => v.to_string(),
            _ => continue,
        };
        form = form.text(key.clone(), text);
    }

    let image_urls: Vec<&str> = input
        .get("image_urls")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if image_urls.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "gpt-image-2 edit requires at least one reference image".to_string(),
        }));
    }
    for (i, url) in image_urls.iter().enumerate() {
        let part = download_form_part(http_client, url, &format!("image_{i}.png")).await?;
        form = form.part("image[]", part);
    }
    if let Some(mask_url) = input.get("mask").and_then(Value::as_str) {
        let part = download_form_part(http_client, mask_url, "mask.png").await?;
        form = form.part("mask", part);
    }
    Ok(form)
}

/// Fetch one reference asset and wrap it as a multipart file part.
async fn download_form_part(
    http_client: &TensorzeroHttpClient,
    url: &str,
    filename: &str,
) -> Result<reqwest::multipart::Part, Error> {
    let (bytes, mime) = fetch_reference_image(http_client, url).await?;
    image_form_part(bytes, mime, filename)
}

fn image_form_part(
    bytes: Vec<u8>,
    mime: String,
    filename: &str,
) -> Result<reqwest::multipart::Part, Error> {
    reqwest::multipart::Part::bytes(bytes)
        .file_name(filename.to_string())
        .mime_str(&mime)
        .map_err(|e| {
            Error::new(ErrorDetails::InvalidRequest {
                message: format!("Invalid reference image content-type: {e}"),
            })
        })
}

/// Fetch one reference image as `(bytes, mime)`.
///
/// Accepts both real URLs and inline `data:<mime>;base64,<payload>` URIs —
/// the latter are decoded locally, because an HTTP GET on a `data:` scheme is
/// a reqwest builder error ("Failed to download reference image: builder
/// error for url (data:...)"). Either way the content type comes from
/// `image_part_mime`: origin servers routinely serve images as
/// `application/octet-stream`, which the upstream rejects ("Invalid file
/// 'mask': unsupported mimetype") since it only accepts `image/jpeg`,
/// `image/png`, and `image/webp`.
async fn fetch_reference_image(
    http_client: &TensorzeroHttpClient,
    url: &str,
) -> Result<(Vec<u8>, String), Error> {
    if let Some(body) = url.strip_prefix("data:") {
        return decode_data_uri(body);
    }
    let parsed = url.parse::<Url>().map_err(|e| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Invalid reference image URL: {e}"),
        })
    })?;
    let response = http_client
        .get(parsed)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                message: format!("Failed to download reference image: {e}"),
                status_code: e.status(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;
    if !response.status().is_success() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("Reference image URL returned {}: {url}", response.status()),
        }));
    }
    let header_mime = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string);
    let bytes = response.bytes().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            message: format!("Failed to read reference image: {e}"),
            status_code: None,
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        })
    })?;
    let mime = image_part_mime(&bytes, header_mime.as_deref()).to_string();
    Ok((bytes.to_vec(), mime))
}

/// Decode a `data:<mime>;base64,<payload>` URI (already stripped of the
/// `data:` prefix) into `(bytes, mime)`.
fn decode_data_uri(body: &str) -> Result<(Vec<u8>, String), Error> {
    let (meta, b64) = body.split_once(',').ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: "Malformed `data:` URI in reference image (missing comma)".to_string(),
        })
    })?;
    if !meta.contains("base64") {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Only base64-encoded `data:` URIs are supported for reference images"
                .to_string(),
        }));
    }
    let bytes = BASE64_STANDARD.decode(b64.trim()).map_err(|e| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Invalid base64 in reference image `data:` URI: {e}"),
        })
    })?;
    let declared = meta.split(';').next().filter(|m| !m.is_empty());
    let mime = image_part_mime(&bytes, declared).to_string();
    Ok((bytes, mime))
}

/// Pick the content type for an image file part: magic-byte sniff first
/// (PNG / JPEG / WebP — the formats the upstream accepts), then a declared
/// `image/*` type, then `image/png` as the last resort. Never returns a
/// non-image type: a truthful `application/octet-stream` is a guaranteed
/// upstream rejection, while a sniffable image works regardless of what the
/// origin server claimed.
fn image_part_mime<'a>(bytes: &[u8], declared: Option<&'a str>) -> &'a str {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        return "image/png";
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return "image/jpeg";
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return "image/webp";
    }
    match declared {
        Some(m) if m.starts_with("image/") => m,
        _ => "image/png",
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
    // A few Vidu image/start-end shapes accept an *optional* prompt (the
    // reference image carries the intent); every other Novita media variant
    // requires one. Insert the prompt when present, and only error when it's
    // both absent and required for this shape.
    let prompt_optional = matches!(
        shape,
        NovitaRequestShape::ViduQ1ImageToVideo
            | NovitaRequestShape::ViduQ3ImageToVideo
            | NovitaRequestShape::ViduQ3FirstLastToVideo
    );
    let mut body = serde_json::Map::new();
    match input
        .get("prompt")
        .and_then(Value::as_str)
        .filter(|prompt| !prompt.is_empty())
    {
        Some(prompt) => {
            body.insert("prompt".into(), Value::from(prompt));
        }
        None if prompt_optional => {}
        None => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Novita-backed media variants require a prompt".to_string(),
            }));
        }
    }

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
        // Gemini-native: flat RouterBase params, mapped onto `generateContent`
        // by `build_gemini_native_request` (`resolution` is the api's default
        // size key). `image_urls` are fetched and inlined there.
        NovitaRequestShape::GeminiNativeTextToImage => &[
            "size",
            "resolution",
            "aspect_ratio",
            "thinking_level",
            "web_search",
            "image_search",
        ],
        NovitaRequestShape::GeminiNativeEdit => &[
            "size",
            "resolution",
            "aspect_ratio",
            "thinking_level",
            "web_search",
            "image_search",
            "image_urls",
        ],
        // OAI-native protocol: parameters per OpenAI's Image API. `moderation`
        // is generation-only (Novita documents it as ineffective on edits).
        NovitaRequestShape::GptImageOaiTextToImage => &[
            "size",
            "n",
            "quality",
            "background",
            "moderation",
            "output_format",
            "output_compression",
        ],
        // Text fields of the edit form; `image_urls` / `mask` are downloaded
        // and attached as file parts by `build_gpt_image_oai_form`, not here.
        NovitaRequestShape::GptImageOaiEdit => {
            &["size", "n", "quality", "background", "output_format"]
        }
        // `size` and `guidance_scale` are documented Seedream 3.0 params but the
        // Novita checkpoint TASK_FAILs whenever either is present, so they are
        // deliberately NOT forwarded.
        NovitaRequestShape::SeedreamV3TextToImage => &["seed", "response_format", "watermark"],
        NovitaRequestShape::SeedreamV4TextToImage => &[
            "size",
            "sequential_image_generation",
            "max_images",
            "watermark",
        ],
        NovitaRequestShape::SeedreamImage => &["size", "sequential_image_generation", "watermark"],
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
        // Original-factory shapes never reach this builder: `infer_media_proxy`
        // hands them to `build_original_factory_body` first.
        NovitaRequestShape::KlingV34kTextToVideo
        | NovitaRequestShape::KlingV34kImageToVideo
        | NovitaRequestShape::KlingV3MotionControl
        | NovitaRequestShape::KlingV4TextToVideo
        | NovitaRequestShape::KlingV4ImageToVideo
        | NovitaRequestShape::KlingV4OmniVideo
        | NovitaRequestShape::Wan3TextToVideo
        | NovitaRequestShape::Wan3ImageToVideo
        | NovitaRequestShape::Wan3ReferenceToVideo => &[],
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
        // ── Vidu (Q1/Q2/Q3) ──
        // `off_peak` is deliberately omitted from every Vidu allow-list:
        // RouterBase always charges Vidu's peak (normal) price, so it must
        // never forward `off_peak=true` (which would make Novita bill us the
        // 50%-off rate while the user paid peak). `image_urls` is likewise
        // absent here — it's remapped to Novita's native `images` array below.
        NovitaRequestShape::ViduQ1TextToVideo => &[
            "style",
            "duration",
            "resolution",
            "aspect_ratio",
            "movement_amplitude",
            "seed",
            "bgm",
        ],
        NovitaRequestShape::ViduQ1ImageToVideo => &[
            "duration",
            "resolution",
            "movement_amplitude",
            "seed",
            "bgm",
        ],
        NovitaRequestShape::ViduQ1ReferenceToVideo => &[
            "duration",
            "resolution",
            "aspect_ratio",
            "movement_amplitude",
            "seed",
            "bgm",
        ],
        NovitaRequestShape::ViduQ2TextToVideo => &[
            "duration",
            "resolution",
            "aspect_ratio",
            "style",
            "movement_amplitude",
            "seed",
            "bgm",
            "audio",
            "watermark",
            "subjects",
        ],
        NovitaRequestShape::ViduQ2ImageToVideo => &[
            "duration",
            "resolution",
            "movement_amplitude",
            "seed",
            "bgm",
            "audio",
            "voice_id",
            "watermark",
        ],
        NovitaRequestShape::ViduQ2SubjectToVideo => &[
            "duration",
            "subjects",
            "resolution",
            "aspect_ratio",
            "movement_amplitude",
            "seed",
            "bgm",
            "audio",
            "watermark",
        ],
        NovitaRequestShape::ViduQ3TextToVideo => &[
            "duration",
            "resolution",
            "aspect_ratio",
            "seed",
            "audio",
            "watermark",
            "wm_url",
            "wm_position",
        ],
        NovitaRequestShape::ViduQ3ImageToVideo => &[
            "duration",
            "resolution",
            "aspect_ratio",
            "style",
            "seed",
            "audio",
            "audio_type",
            "is_rec",
            "watermark",
            "wm_url",
            "wm_position",
        ],
        NovitaRequestShape::ViduQ3FirstLastToVideo => &[
            "duration",
            "resolution",
            "seed",
            "audio",
            "is_rec",
            "watermark",
            "wm_url",
            "wm_position",
        ],
    };

    if let Some(input_obj) = input.as_object() {
        for key in allowed {
            if let Some(value) = input_obj.get(*key) {
                body.insert((*key).to_string(), value.clone());
            }
        }
    }

    // Vidu takes image inputs as a native `images` array (URLs or Base64).
    // RouterBase's schema/playground uses `image_urls` for parity with the
    // other i2v families; forward the whole array through under `images`.
    // For start-end (f2v) the array is already ordered `[start, end]`.
    if matches!(
        shape,
        NovitaRequestShape::ViduQ1ImageToVideo
            | NovitaRequestShape::ViduQ1ReferenceToVideo
            | NovitaRequestShape::ViduQ2ImageToVideo
            | NovitaRequestShape::ViduQ3ImageToVideo
            | NovitaRequestShape::ViduQ3FirstLastToVideo
    ) && !body.contains_key("images")
    {
        if let Some(images) = input.get("image_urls").and_then(Value::as_array) {
            body.insert("images".into(), Value::Array(images.clone()));
        }
    }

    // `vidu-q2-reference2video` / `viduq2-pro-fast` take `subjects` instead: a
    // list of reference subjects that the prompt addresses by name. Novita's
    // schema is `{name (required), images (<=3), voice_id?}`, and the prompt
    // references a subject as `@{name}` — so the bare ordinal is the name, and
    // `name: "1"` is what makes `@1` resolve (NOT `name: "@1"`, which would need
    // `@@1`).
    //
    // Every other media model — and the playground's upload control, and
    // RouterBase's base64-materialisation / URL-precheck — speaks `image_urls`,
    // so accept that here too and fan each URL out into its own subject, in
    // order, so `@N` addresses the Nth image. An explicit `subjects` always
    // wins, keeping the richer shape (several angles of one subject, `voice_id`)
    // reachable for callers that need it.
    if matches!(shape, NovitaRequestShape::ViduQ2SubjectToVideo) && !body.contains_key("subjects") {
        if let Some(urls) = input.get("image_urls").and_then(Value::as_array) {
            let subjects: Vec<Value> = urls
                .iter()
                .filter(|u| u.is_string())
                .enumerate()
                .map(|(i, url)| json!({ "name": (i + 1).to_string(), "images": [url.clone()] }))
                .collect();
            if !subjects.is_empty() {
                body.insert("subjects".into(), Value::Array(subjects));
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

    // Seedream 3.0's endpoint requires a fixed body `model` string identifying
    // the checkpoint (the URL slug alone is not enough). It's server-forced so
    // a caller can't select a different checkpoint by passing their own `model`.
    if matches!(shape, NovitaRequestShape::SeedreamV3TextToImage) {
        body.insert("model".into(), Value::from("seedream-3-0-t2i-250415"));
    }

    // The OAI-native GPT Image 2 endpoints share one path per operation and
    // select the checkpoint via a body `model`. Fixed server-side to
    // "gpt-image-2-oai": the plain "gpt-image-2" checkpoint rejects
    // `background: "transparent"` ("Transparent background is not supported
    // for this model"), while -oai renders true RGBA at identical token
    // usage (verified 2026-08-24).
    if matches!(
        shape,
        NovitaRequestShape::GptImageOaiTextToImage | NovitaRequestShape::GptImageOaiEdit
    ) {
        body.insert("model".into(), Value::from("gpt-image-2-oai"));
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

    // OAI-native protocol: parameters per OpenAI's Image API. `moderation`
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

    // Seedream 4.5 / 5.0 lite edit on PPIO: the reference images go out as
    // the route's `image` array (1 to 14 URLs); text-to-image sends none.
    if matches!(shape, NovitaRequestShape::SeedreamImage) {
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
        if !images.is_empty() {
            body.insert("image".into(), Value::Array(images));
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
        NovitaRequestShape::GptImageOaiTextToImage | NovitaRequestShape::GptImageOaiEdit
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

/// Success payload for RouterBase's media callback. `usage` is the upstream's
/// OpenAI-style token usage when it reports one (OAI-native GPT Image); it rides
/// at the top level and under `data`, like the other fields.
fn media_callback_success_body(
    task_id: &str,
    result_json: &str,
    urls: &[String],
    usage: Option<&Value>,
) -> Value {
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
    body
}

async fn post_media_callback(
    http_client: &TensorzeroHttpClient,
    callback_url: &str,
    task_id: &str,
    urls: &[String],
    usage: Option<&Value>,
) -> Result<(), Error> {
    let result_json = serde_json::to_string(&json!({ "resultUrls": urls })).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize callback payload: {e}"),
        })
    })?;
    let body = media_callback_success_body(task_id, &result_json, urls, usage);
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

/// Deliver a `fail` callback to RouterBase when an async media task errors out
/// after we've already returned the `task_id` (so the error can no longer be
/// surfaced inline on the inference response). Mirrors the shapes RouterBase's
/// `media_callback` handler probes for state + failure reason. Best-effort: a
/// delivery failure is logged, not propagated (there is no caller to receive
/// it — this runs in a detached task).
async fn post_media_callback_failure(
    http_client: &TensorzeroHttpClient,
    callback_url: &str,
    task_id: &str,
    error_message: &str,
) {
    let body = json!({
        "taskId": task_id,
        "task_id": task_id,
        "state": "fail",
        "errorMessage": error_message,
        "data": {
            "taskId": task_id,
            "task_id": task_id,
            "failMsg": error_message,
        }
    });
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

// ── Kling and Wan 3.0 on their original-factory routes ─────────────────────
//
// Novita / PPIO "original-factory" video protocols: Kling (`/v3/kling/...`)
// and Alibaba Wan 3.0 (`/v3/aliyun/...`). Same host and API key as the rest of
// the Novita provider, but each family keeps its vendor's own request and task
// shapes instead of Novita's flat `/v3/async/<model>` bodies:
//
// - Kling: `POST /v3/kling/{text-to-video|image-to-video|omni-video|motion-control}/{model}`
//   with `prompt` / `contents[]` + `settings{}`; answers `{code: 0, data: {id}}`.
//   Poll `GET /v3/kling/tasks?task_ids={id}` until `data[0].status` is
//   terminal; a finished task lists `outputs[]` (`type: video`, `url`,
//   `duration`) and, when the request carried video, `usage.input_duration`.
// - Wan: `POST /v3/aliyun/api/v1/services/aigc/video-generation/video-synthesis`
//   with `{model, input: {prompt, media[]}, parameters{}}`; answers
//   `{output: {task_id}}`. Poll `GET /v3/aliyun/api/v1/tasks/{id}` until
//   `output.task_status` is terminal; success carries `output.video_url` and
//   `usage.{duration, input_video_duration, output_video_duration}`.
//
// Neither protocol accepts an upstream callback for these models, so the
// gateway polls (detached, like the other async media) and reports the
// result to RouterBase's media callback. Both bill by the seconds actually
// produced (and, with video input, read), which ride on the callback as
// `usage.{input_seconds, output_seconds, seconds}` so the generation settles
// at the real duration instead of the submit-time hold.
//
// RouterBase keeps one flat parameter surface for every video model
// (`image_urls`, `video_urls`, `duration`, `resolution`, ...); the builders
// below map it onto each vendor shape.

/// Body `model` when the variant names none.
const WAN_DEFAULT_MODEL: &str = "wan3.0-video";
const POLL_INTERVAL: Duration = Duration::from_secs(10);
/// How long poll calls may fail in a row (network errors, 5xx, an unreadable
/// body, any 4xx) before the task is given up. A poll is a read: a status
/// outage, a key glitch or a task-store lag must not fail a task the upstream
/// goes on to finish and bill, so it gets minutes rather than a few attempts.
const POLL_ERROR_WINDOW: Duration = Duration::from_secs(15 * 60);
/// A Wan task reported `UNKNOWN` (not found) this soon after submit is still
/// polled: the status API can lag the create call.
const WAN_UNKNOWN_GRACE: Duration = Duration::from_secs(120);
/// Attempts to deliver the success callback, `CALLBACK_RETRY_DELAY` x attempt
/// apart. By then the upstream has rendered and billed the video, so a
/// RouterBase restart or 5xx at that moment must not lose it; RouterBase's
/// callback handler is idempotent.
const CALLBACK_ATTEMPTS: u32 = 4;
const CALLBACK_RETRY_DELAY: Duration = Duration::from_secs(10);
/// Kling 4.0 accepts at most 10 images per request (keyframes or references).
const KLING_V4_MAX_IMAGES: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Protocol {
    Kling,
    Wan,
}

/// The original-factory protocol a shape speaks, or `None` for the classic
/// Novita `/v3/...` shapes.
fn original_factory_protocol(shape: &NovitaRequestShape) -> Option<Protocol> {
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
fn build_original_factory_body(
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
            // The tier follows `model_name` only, as on the old protocol
            // (`-pro` = 1080p, `-std` = 720p). A caller's `resolution` is
            // ignored: RouterBase prices Motion Control by its tier, so the
            // render must not drop below the tier the request is billed at.
            let resolution = if str_field(input, "model_name").is_some_and(|m| m.ends_with("-std"))
            {
                "720p"
            } else {
                "1080p"
            };
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
struct TaskOutput {
    pub urls: Vec<String>,
    pub usage: Option<Value>,
}

#[derive(Debug, PartialEq)]
enum PollOutcome {
    Done(TaskOutput),
    Failed(String),
    /// The status API does not know the task (Wan `UNKNOWN`): terminal once
    /// the submit is older than `WAN_UNKNOWN_GRACE`, still pending before.
    Unknown(String),
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
                            o.get("type")
                                .and_then(Value::as_str)
                                .unwrap_or("video")
                                .eq_ignore_ascii_case("video")
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

/// `PENDING` / `RUNNING` keep polling; `SUCCEEDED` is done; `FAILED` and
/// `CANCELED` are terminal; `UNKNOWN` (expired or not found) is terminal only
/// once the submit is past `WAN_UNKNOWN_GRACE`.
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
            let reason = match (code, message) {
                (Some(c), Some(m)) => format!("{c}: {m}"),
                (None, Some(m)) => m.to_string(),
                (Some(c), None) => format!("task {status} ({c})"),
                (None, None) => format!("task {status}"),
            };
            if status == "UNKNOWN" {
                PollOutcome::Unknown(reason)
            } else {
                PollOutcome::Failed(reason)
            }
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
async fn infer_original_factory(
    shape: &NovitaRequestShape,
    model: Option<&str>,
    path: &str,
    base: &str,
    callback_url: &str,
    input: &Value,
    http_client: &TensorzeroHttpClient,
    api_key: &str,
) -> Result<String, Error> {
    let protocol = original_factory_protocol(shape)
        .ok_or_else(|| invalid("not an original-factory video shape"))?;
    let body = build_original_factory_body(shape, model, input)?;
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
        Ok(output) => deliver_success(http_client, callback_url, task_id, &output).await,
        Err(reason) => {
            tracing::error!("Novita task {task_id} failed: {reason}");
            post_media_callback_failure(http_client, callback_url, task_id, &reason).await;
        }
    }
}

/// Deliver the success callback, retrying on failure (see `CALLBACK_ATTEMPTS`).
async fn deliver_success(
    http_client: &TensorzeroHttpClient,
    callback_url: &str,
    task_id: &str,
    output: &TaskOutput,
) {
    for attempt in 1..=CALLBACK_ATTEMPTS {
        let delivered = post_media_callback(
            http_client,
            callback_url,
            task_id,
            &output.urls,
            output.usage.as_ref(),
        )
        .await;
        match delivered {
            Ok(()) => return,
            Err(e) if attempt < CALLBACK_ATTEMPTS => {
                tracing::warn!(
                    "Success callback for Novita task {task_id} failed (attempt {attempt}/{CALLBACK_ATTEMPTS}): {e}"
                );
                tokio::time::sleep(CALLBACK_RETRY_DELAY * attempt).await;
            }
            Err(e) => tracing::error!(
                "Failed to deliver success callback for Novita task {task_id} after {CALLBACK_ATTEMPTS} attempts: {e}"
            ),
        }
    }
}

/// Poll until the task finishes. `Err` is the reason RouterBase shows the
/// caller: the upstream's own failure message, or a neutral one when the task
/// timed out or its status could not be read. It never carries a provider
/// name or a URL; those details go to the log.
async fn poll_until_done(
    protocol: Protocol,
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    url: &str,
    task_id: &str,
) -> Result<TaskOutput, String> {
    let submitted = Instant::now();
    let deadline = submitted + ASYNC_TASK_TIMEOUT;
    let mut failing_since: Option<Instant> = None;
    loop {
        // Before every read, the first one included: the task was only just
        // created, and the status API may not know it yet.
        tokio::time::sleep(POLL_INTERVAL).await;
        if Instant::now() >= deadline {
            tracing::error!(
                "Novita task {task_id} did not complete within {}s",
                ASYNC_TASK_TIMEOUT.as_secs()
            );
            return Err(format!(
                "the task did not finish within {} minutes",
                ASYNC_TASK_TIMEOUT.as_secs() / 60
            ));
        }
        match poll_once(protocol, http_client, api_key, url).await {
            Ok(body) => {
                failing_since = None;
                match classify_poll(protocol, &body) {
                    PollOutcome::Done(output) => return Ok(output),
                    PollOutcome::Failed(reason) => return Err(reason),
                    PollOutcome::Unknown(reason) if submitted.elapsed() >= WAN_UNKNOWN_GRACE => {
                        return Err(reason);
                    }
                    PollOutcome::Unknown(_) | PollOutcome::Pending => {}
                }
            }
            Err(reason) => {
                let since = *failing_since.get_or_insert_with(Instant::now);
                tracing::warn!(
                    "Novita poll for task {task_id} failed ({}s into a run of errors): {reason}",
                    since.elapsed().as_secs()
                );
                if since.elapsed() >= POLL_ERROR_WINDOW {
                    return Err("the task status could not be read".to_string());
                }
            }
        }
    }
}

/// One status read. Every failure is an `Err` the caller tolerates for
/// `POLL_ERROR_WINDOW`, 4xx included: a read that fails does not mean the
/// upstream stopped rendering (or billing).
async fn poll_once(
    protocol: Protocol,
    http_client: &TensorzeroHttpClient,
    api_key: &str,
    url: &str,
) -> Result<Value, String> {
    let response = http_client
        .get(url)
        .bearer_auth(api_key)
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = response.status();
    let raw = response.text().await.unwrap_or_default();
    let body: Option<Value> = serde_json::from_str(&raw).ok();
    if let Some(reason) = submit_rejection(protocol, status.as_u16(), body.as_ref()) {
        return Err(format!("HTTP {status}: {reason}"));
    }
    body.ok_or_else(|| "unparseable poll response".to_string())
}

#[cfg(test)]
mod vidu_build_body_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn image_urls_remap_to_images_array() {
        let input = json!({
            "prompt": "a cat waves",
            "image_urls": ["https://x/a.jpg"],
            "duration": 5,
            "resolution": "720p"
        });
        let body = build_body(&NovitaRequestShape::ViduQ2ImageToVideo, &input).unwrap();
        assert_eq!(
            body.get("images"),
            Some(&json!(["https://x/a.jpg"])),
            "image_urls must be forwarded to Novita's native `images` array"
        );
        assert!(
            body.get("image_urls").is_none(),
            "the RouterBase-only `image_urls` key must not leak upstream"
        );
    }

    #[test]
    fn start_end_frames_preserve_order() {
        let input = json!({
            "image_urls": ["https://x/start.jpg", "https://x/end.jpg"],
            "duration": 8
        });
        let body = build_body(&NovitaRequestShape::ViduQ3FirstLastToVideo, &input).unwrap();
        assert_eq!(
            body.get("images"),
            Some(&json!(["https://x/start.jpg", "https://x/end.jpg"])),
            "start-end frames must reach `images` as an ordered [start, end] array"
        );
    }

    #[test]
    fn prompt_optional_for_image_shapes() {
        let input = json!({ "image_urls": ["https://x/a.jpg"], "duration": 5 });
        let body = build_body(&NovitaRequestShape::ViduQ3ImageToVideo, &input).unwrap();
        assert!(
            body.get("prompt").is_none(),
            "Q3 i2v must accept a missing prompt without erroring"
        );
    }

    #[test]
    fn prompt_required_for_text_shapes() {
        let input = json!({ "duration": 5, "resolution": "720p" });
        assert!(
            build_body(&NovitaRequestShape::ViduQ2TextToVideo, &input).is_err(),
            "text-to-video must reject a request with no prompt"
        );
    }

    #[test]
    fn off_peak_is_never_forwarded() {
        let input = json!({
            "prompt": "sunset",
            "duration": 5,
            "resolution": "1080p",
            "off_peak": true
        });
        let body = build_body(&NovitaRequestShape::ViduQ3TextToVideo, &input).unwrap();
        assert!(
            body.get("off_peak").is_none(),
            "off_peak must never reach Novita — RouterBase always charges the peak rate"
        );
    }

    #[test]
    fn subjects_built_from_image_urls_when_absent() {
        let input = json!({
            "prompt": "@1 and @2 shake hands",
            "duration": 5,
            "image_urls": ["https://x/a.jpg", "https://x/b.jpg"],
        });
        let body = build_body(&NovitaRequestShape::ViduQ2SubjectToVideo, &input).unwrap();
        assert_eq!(
            body.get("subjects"),
            Some(&json!([
                { "name": "1", "images": ["https://x/a.jpg"] },
                { "name": "2", "images": ["https://x/b.jpg"] },
            ])),
            "each image_url becomes its own subject, in order; Novita requires `name` and \
             resolves `@{{name}}`, so the bare ordinal is the name and `@N` addresses the Nth image"
        );
        assert!(
            body.get("image_urls").is_none(),
            "image_urls is a RouterBase-side alias and must not leak to Novita"
        );
    }

    #[test]
    fn explicit_subjects_win_over_image_urls() {
        let subjects = json!([{ "name": "1", "images": ["https://x/1.jpg", "https://x/2.jpg"] }]);
        let input = json!({
            "prompt": "@1 dances",
            "duration": 5,
            "subjects": subjects,
            "image_urls": ["https://x/ignored.jpg"],
        });
        let body = build_body(&NovitaRequestShape::ViduQ2SubjectToVideo, &input).unwrap();
        assert_eq!(
            body.get("subjects"),
            Some(&subjects),
            "an explicit subjects[] must win, so multi-image subjects stay reachable"
        );
    }

    #[test]
    fn subjects_passthrough_for_reference_shape() {
        let subjects = json!([{ "name": "1", "images": ["https://x/s.jpg"] }]);
        let input = json!({ "prompt": "@1 dances", "duration": 5, "subjects": subjects });
        let body = build_body(&NovitaRequestShape::ViduQ2SubjectToVideo, &input).unwrap();
        assert_eq!(
            body.get("subjects"),
            Some(&subjects),
            "subjects[] must be forwarded verbatim for reference/subject video"
        );
    }

    #[test]
    fn seedream_v3_forces_the_checkpoint_model_and_filters_params() {
        let input = json!({
            "prompt": "a red panda barista",
            "seed": 42,
            "size": "1024x1024",   // documented but poisons the checkpoint -> filtered
            "guidance_scale": 3.0, // documented but poisons the checkpoint -> filtered
            "model": "attacker-picked-checkpoint", // must be overridden, not honored
            "aspect_ratio": "16:9",                // not in the 3.0 allow-list
        });
        let body = build_body(&NovitaRequestShape::SeedreamV3TextToImage, &input).unwrap();
        assert_eq!(
            body.get("model").and_then(Value::as_str),
            Some("seedream-3-0-t2i-250415"),
            "Seedream 3.0 must force the checkpoint model server-side, ignoring caller input"
        );
        assert_eq!(
            body.get("prompt").and_then(Value::as_str),
            Some("a red panda barista"),
            "prompt must be forwarded"
        );
        assert_eq!(
            body.get("seed").and_then(Value::as_i64),
            Some(42),
            "seed is in the 3.0 allow-list and must be forwarded"
        );
        assert!(
            body.get("size").is_none(),
            "size TASK_FAILs the 3.0 checkpoint and must be filtered out"
        );
        assert!(
            body.get("guidance_scale").is_none(),
            "guidance_scale TASK_FAILs the 3.0 checkpoint and must be filtered out"
        );
        assert!(
            body.get("aspect_ratio").is_none(),
            "aspect_ratio is NOT a Seedream 3.0 param and must be filtered out"
        );
    }

    #[test]
    fn seedream_image_shape_forwards_only_allowed_params() {
        // 4.5 / 5.0-lite share this shape. `guidance_scale` (a 3.0-only param)
        // must not leak through; `size`/`watermark` must.
        let input = json!({
            "prompt": "a cozy coffee shop",
            "size": "2K",
            "watermark": false,
            "guidance_scale": 3.0,
        });
        let body = build_body(&NovitaRequestShape::SeedreamImage, &input).unwrap();
        assert_eq!(
            body.get("size").and_then(Value::as_str),
            Some("2K"),
            "size must forward"
        );
        assert_eq!(
            body.get("watermark").and_then(Value::as_bool),
            Some(false),
            "watermark must forward"
        );
        assert!(
            body.get("guidance_scale").is_none(),
            "guidance_scale is not a Seedream 4.5/5.0-lite param and must be filtered out"
        );
        assert!(
            body.get("model").is_none(),
            "only 3.0 injects a fixed model; 4.x/5.x select the checkpoint by URL path"
        );
        assert!(
            body.get("image").is_none(),
            "text-to-image sends no reference images"
        );
        let edit = build_body(
            &NovitaRequestShape::SeedreamImage,
            &json!({"prompt": "make it blue", "image_urls": ["https://a/1.png", " ", "https://a/2.png"]}),
        )
        .unwrap();
        assert_eq!(
            edit.get("image"),
            Some(&json!(["https://a/1.png", "https://a/2.png"])),
            "an edit sends its reference images as the route's `image` array"
        );
    }

    #[test]
    fn seedream_requires_a_prompt() {
        let input = json!({ "size": "2K" });
        assert!(
            build_body(&NovitaRequestShape::SeedreamV4TextToImage, &input).is_err(),
            "Seedream text-to-image must reject a request with no prompt"
        );
    }

    #[test]
    fn image_part_mime_sniffs_magic_bytes_over_declared_type() {
        let png = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(
            image_part_mime(&png, Some("application/octet-stream")),
            "image/png",
            "a sniffable PNG must override an octet-stream content type"
        );
        let jpeg = [0xFF, 0xD8, 0xFF, 0xE0];
        assert_eq!(
            image_part_mime(&jpeg, None),
            "image/jpeg",
            "JPEG magic bytes must be recognized without a declared type"
        );
        let webp = *b"RIFF\x00\x00\x00\x00WEBPVP8 ";
        assert_eq!(
            image_part_mime(&webp, Some("image/png")),
            "image/webp",
            "WebP magic bytes must win over a wrong declared type"
        );
    }

    #[test]
    fn image_part_mime_falls_back_to_declared_image_type_then_png() {
        assert_eq!(
            image_part_mime(b"not an image", Some("image/jpeg")),
            "image/jpeg",
            "unsniffable bytes must fall back to a declared image/* type"
        );
        assert_eq!(
            image_part_mime(b"not an image", Some("application/octet-stream")),
            "image/png",
            "a non-image declared type must never be forwarded to the upstream"
        );
        assert_eq!(
            image_part_mime(b"not an image", None),
            "image/png",
            "no declared type must default to image/png"
        );
    }

    #[test]
    fn decode_data_uri_decodes_base64_payloads() {
        let b64 = base64::engine::general_purpose::STANDARD.encode([0x89, b'P', b'N', b'G']);
        let (bytes, mime) = decode_data_uri(&format!("image/jpeg;base64,{b64}"))
            .expect("a well-formed base64 data: URI must decode");
        assert_eq!(
            bytes,
            [0x89, b'P', b'N', b'G'],
            "payload bytes must round-trip"
        );
        assert_eq!(
            mime, "image/png",
            "the sniffed PNG signature must win over the declared type"
        );
        assert!(
            decode_data_uri("image/png;base64").is_err(),
            "a data: URI without a comma must be rejected as malformed"
        );
        assert!(
            decode_data_uri("image/png,plaintext").is_err(),
            "a non-base64 data: URI must be rejected"
        );
    }
}

#[cfg(test)]
mod gemini_native_tests {
    use super::*;
    use serde_json::json;

    fn body(value: Value) -> serde_json::Map<String, Value> {
        value.as_object().cloned().unwrap()
    }

    #[test]
    fn request_maps_flat_params_onto_generate_content() {
        let request = build_gemini_native_request(
            &body(json!({
                "prompt": "a red apple",
                "size": "2k",
                "resolution": "1K",
                "aspect_ratio": "16:9",
                "thinking_level": "high",
                "web_search": true,
                "image_search": false,
            })),
            &[(vec![1, 2, 3], "image/jpeg".to_string())],
        );
        assert_eq!(
            request["contents"][0]["parts"],
            json!([
                {"text": "a red apple"},
                {"inlineData": {"mimeType": "image/jpeg", "data": "AQID"}},
            ]),
            "the prompt must lead, followed by each reference image inline"
        );
        assert_eq!(
            request["generationConfig"],
            json!({
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {"imageSize": "2K", "aspectRatio": "16:9"},
                "thinkingConfig": {"thinkingLevel": "high"},
            }),
            "`size` must win over the default `resolution` and be upper-cased"
        );
        assert_eq!(
            request["tools"],
            json!([{"google_search": {"searchTypes": {"webSearch": {}}}}]),
            "only the enabled search types must be requested"
        );
    }

    #[test]
    fn request_drops_pixel_sizes_unknown_levels_and_disabled_search() {
        let request = build_gemini_native_request(
            &body(json!({
                "prompt": "x",
                "size": "1024x1024",
                "thinking_level": "max",
            })),
            &[],
        );
        assert!(
            request["generationConfig"].get("imageConfig").is_none(),
            "a pixel size is not a Gemini tier and must leave the upstream default"
        );
        assert!(
            request["generationConfig"].get("thinkingConfig").is_none(),
            "an unsupported thinking level must not be forwarded"
        );
        assert!(
            request.get("tools").is_none(),
            "no search tool unless a search type is enabled"
        );
    }

    #[test]
    fn images_skip_thought_drafts() {
        let raw = json!({"candidates": [{"content": {"parts": [
            {"text": "planning", "thought": true},
            {"inlineData": {"mimeType": "image/png", "data": "DRAFT"}, "thought": true},
            {"text": "Here it is"},
            {"inlineData": {"mimeType": "image/jpeg", "data": "FINAL"}},
        ]}}]});
        assert_eq!(
            gemini_native_images(&raw),
            vec!["data:image/jpeg;base64,FINAL".to_string()],
            "only the final, non-thought image must be delivered"
        );
    }

    #[test]
    fn no_image_message_carries_reason_and_text() {
        let raw = json!({"candidates": [{
            "finishReason": "NO_IMAGE",
            "content": {"parts": [
                {"text": "thinking", "thought": true},
                {"text": "I can't draw that."},
            ]},
        }]});
        assert_eq!(
            gemini_native_no_image_message(&raw),
            "Novita completed but returned no image (finish reason: NO_IMAGE): I can't draw that.",
            "the error must surface the finish reason and the model's own text"
        );
        let empty =
            json!({"candidates": [{"finishReason": "NO_IMAGE", "content": {"parts": null}}]});
        assert_eq!(
            gemini_native_no_image_message(&empty),
            "Novita completed but returned no image (finish reason: NO_IMAGE)",
        );
    }

    #[test]
    fn usage_splits_image_and_text_output_and_counts_searches() {
        // Shape observed from gemini-nano-banana-2.1 with web search (2026-10-14).
        let raw = json!({
            "candidates": [{"groundingMetadata": {
                "webSearchQueries": ["tokyo weather", "tokyo forecast"],
                "imageSearchQueries": ["tokyo skyline"],
            }}],
            "usageMetadata": {
                "promptTokenCount": 1137,
                "candidatesTokenCount": 1186,
                "thoughtsTokenCount": 1106,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 17},
                    {"modality": "IMAGE", "tokenCount": 1120},
                ],
                "candidatesTokensDetails": [
                    {"modality": "IMAGE", "tokenCount": 1120},
                    {"modality": "TEXT", "tokenCount": 66},
                ],
            },
        });
        assert_eq!(
            gemini_native_usage(&raw).unwrap(),
            json!({
                "input_tokens": 1137,
                "input_tokens_details": {"text_tokens": 17, "image_tokens": 1120, "cached_tokens": 0},
                "output_tokens": 2292,
                "output_tokens_details": {"image_tokens": 1120, "text_tokens": 1172},
                "search_queries": 3,
            }),
            "thinking and text output must be split from image output"
        );
    }

    #[test]
    fn rendered_proxy_config_selects_the_gemini_native_shapes() {
        // Exactly what RouterBase's tz_config_render emits for Nano Banana 2.1.
        for (shape, edit) in [
            ("gemini_native_text_to_image", false),
            ("gemini_native_edit", true),
        ] {
            let proxy: NovitaMediaProxyConfig = serde_json::from_value(json!({
                "path": "gemini/v1beta/models/gemini-nano-banana-2.1:generateContent",
                "request_shape": shape,
            }))
            .unwrap();
            assert_eq!(
                matches!(proxy.request_shape, NovitaRequestShape::GeminiNativeEdit),
                edit,
                "`{shape}` must deserialize to its own variant"
            );
            assert!(
                !proxy.async_submission,
                "Gemini-native calls are synchronous"
            );
        }
    }

    #[test]
    fn usage_without_breakdown_bills_candidates_as_image() {
        let raw = json!({"usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 1120,
            "thoughtsTokenCount": 537,
        }});
        let usage = gemini_native_usage(&raw).unwrap();
        assert_eq!(
            usage["output_tokens_details"],
            json!({"image_tokens": 1120, "text_tokens": 537}),
            "without candidatesTokensDetails every candidate token must stay at the image rate"
        );
        assert_eq!(
            usage["input_tokens_details"]["text_tokens"], 11,
            "without promptTokensDetails all input counts as text"
        );
        assert!(
            gemini_native_usage(&json!({})).is_none(),
            "no usageMetadata means no usage"
        );
    }
}

#[cfg(test)]
mod gpt_image_usage_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn success_callback_carries_usage_when_present() {
        let usage = json!({"input_tokens": 451, "output_tokens": 1413});
        let urls = vec!["data:image/png;base64,AAAA".to_string()];
        let body = media_callback_success_body("novita-1", "{}", &urls, Some(&usage));
        assert_eq!(body["usage"], usage);
        assert_eq!(body["data"]["usage"], usage);
        assert_eq!(body["state"], "success");
        assert_eq!(body["resultUrls"], json!(urls));
    }

    #[test]
    fn success_callback_omits_usage_when_absent() {
        let body = media_callback_success_body("novita-1", "{}", &[], None);
        assert!(body.get("usage").is_none());
        assert!(body["data"].get("usage").is_none());
    }

    #[test]
    fn proxy_model_is_optional() {
        let with: NovitaMediaProxyConfig = serde_json::from_value(json!({
            "path": "openai/v1/images/generations",
            "request_shape": "gpt_image_oai_text_to_image",
            "model": "gpt-image-2.5-flare-oai"
        }))
        .unwrap();
        assert_eq!(with.model.as_deref(), Some("gpt-image-2.5-flare-oai"));
        let without: NovitaMediaProxyConfig = serde_json::from_value(json!({
            "path": "openai/v1/images/generations",
            "request_shape": "gpt_image_oai_text_to_image"
        }))
        .unwrap();
        assert!(without.model.is_none());
    }
}

#[cfg(test)]
mod kling_wan_tests {
    use super::*;

    fn body(shape: NovitaRequestShape, input: Value) -> Value {
        build_original_factory_body(&shape, None, &input).expect("body should build")
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
            let proxy: NovitaMediaProxyConfig = serde_json::from_value(json!({
                "path": "kling/text-to-video/kling-4.0",
                "async_submission": true,
                "request_shape": name,
                "model": "wan3.0-video-prime",
            }))
            .unwrap_or_else(|e| panic!("`{name}` must deserialize: {e}"));
            assert_eq!(
                original_factory_protocol(&proxy.request_shape),
                Some(expected),
                "`{name}` must route to its original-factory protocol"
            );
        }
        let legacy: NovitaMediaProxyConfig = serde_json::from_value(
            json!({"path": "seedream-4.5", "request_shape": "seedream_image"}),
        )
        .expect("a classic shape must still deserialize");
        assert_eq!(
            original_factory_protocol(&legacy.request_shape),
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
        let err = build_original_factory_body(
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
        assert_eq!(
            std["settings"]["character_orientation"], "image",
            "orientation defaults to the image"
        );
        let asked_720p = body(
            NovitaRequestShape::KlingV3MotionControl,
            json!({"image_urls": ["https://x/a.png"], "video_urls": ["https://x/m.mp4"],
                   "model_name": "kling-v3-0-pro", "resolution": "720p"}),
        );
        assert_eq!(
            asked_720p["settings"]["resolution"], "1080p",
            "a caller's resolution never moves the render off the tier it is billed at"
        );
    }

    #[test]
    fn kling_v4_bodies() {
        let t2v = body(
            NovitaRequestShape::KlingV4TextToVideo,
            json!({"prompt": "lake", "resolution": "1080P", "duration": 10, "aspect_ratio": "21:9"}),
        );
        assert_eq!(
            t2v,
            json!({"prompt": "lake", "settings": {"resolution": "1080p", "duration": 10, "aspect_ratio": "21:9"}}),
            "text-to-video lower-cases the resolution and passes duration and aspect ratio"
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
        assert_eq!(
            i2v["settings"],
            json!({"resolution": "720p"}),
            "image-to-video sends only the settings the caller set"
        );

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
                {"type": "base_video", "url": "https://x/b.mp4", "id": "base"}]),
            "omni references keep their order and get @image / @video / @base ids"
        );
        assert_eq!(
            omni["settings"],
            json!({"aspect_ratio": "adaptive"}),
            "omni passes the aspect ratio through"
        );

        let too_many: Vec<String> = (0..11).map(|i| format!("https://x/{i}.png")).collect();
        assert!(
            build_original_factory_body(
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
        let t2v = build_original_factory_body(
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

        let i2v = build_original_factory_body(
            &NovitaRequestShape::Wan3ImageToVideo,
            None,
            &json!({"image_urls": ["https://x/f.png"], "last_frame_url": "https://x/l.png", "aspect_ratio": "9:16"}),
        )
        .unwrap();
        assert_eq!(
            i2v["model"], "wan3.0-video",
            "the default checkpoint is used when the variant names none"
        );
        assert_eq!(
            i2v["input"],
            json!({"media": [{"type": "first_frame", "url": "https://x/f.png"},
                             {"type": "last_frame", "url": "https://x/l.png"}]}),
            "a prompt is optional once a frame is given"
        );
        assert_eq!(
            i2v["parameters"],
            json!({"ratio": "9:16"}),
            "aspect_ratio maps to Wan's ratio"
        );

        let r2v = build_original_factory_body(
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
                   {"type": "reference_audio", "url": "https://x/a.mp3"}]),
            "reference images, videos and audio each become a media entry"
        );
        assert_eq!(
            r2v["parameters"]["duration"], -1,
            "-1 lets Wan pick the length"
        );
        assert!(
            build_original_factory_body(
                &NovitaRequestShape::Wan3ReferenceToVideo,
                None,
                &json!({"prompt": "x"})
            )
            .is_err(),
            "reference-to-video needs a reference"
        );
        assert!(
            build_original_factory_body(&NovitaRequestShape::Wan3TextToVideo, None, &json!({}))
                .is_err(),
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
            None,
            "Kling code 0 is an accepted submit"
        );
        assert_eq!(
            submit_rejection(
                Protocol::Kling,
                400,
                Some(&json!({"code": 1201, "message": "prompt cannot be empty"}))
            ),
            Some("1201: prompt cannot be empty".to_string()),
            "a Kling rejection reads as code: message"
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
            Some("InvalidParameter: bad model".to_string()),
            "Wan rejections carry string codes"
        );
        assert_eq!(
            submitted_task_id(
                Protocol::Kling,
                &json!({"code": 0, "data": {"id": "k1", "status": "submitted"}})
            ),
            Some("k1".to_string()),
            "Kling's task id is data.id"
        );
        assert_eq!(
            submitted_task_id(
                Protocol::Wan,
                &json!({"output": {"task_id": "w1", "task_status": "PENDING"}})
            ),
            Some("w1".to_string()),
            "Wan's task id is output.task_id"
        );
        assert_eq!(
            poll_url(Protocol::Kling, "https://api.novita.ai", "k1"),
            "https://api.novita.ai/v3/kling/tasks?task_ids=k1",
            "Kling polls its task list by id"
        );
        assert_eq!(
            poll_url(Protocol::Wan, "https://api.ppio.com", "w1"),
            "https://api.ppio.com/v3/aliyun/api/v1/tasks/w1",
            "Wan polls one task by id"
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
            PollOutcome::Failed("billing settlement failed".to_string()),
            "a failed task reports Kling's own message"
        );
        assert_eq!(
            classify_poll(
                Protocol::Kling,
                &json!({"code": 0, "data": [{"id": "k1", "status": "SUCCEEDED",
                    "outputs": [{"type": "VIDEO", "url": "https://r/v.mp4"}]}]})
            ),
            PollOutcome::Done(TaskOutput {
                urls: vec!["https://r/v.mp4".to_string()],
                usage: None,
            }),
            "status and output type are matched case-insensitively"
        );
    }

    #[test]
    fn wan_poll_reports_urls_and_seconds() {
        assert_eq!(
            classify_poll(
                Protocol::Wan,
                &json!({"output": {"task_id": "w1", "task_status": "RUNNING"}})
            ),
            PollOutcome::Pending,
            "a running task keeps polling"
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
            }),
            "a finished task reports its video and Wan's input, output and total seconds"
        );
        assert_eq!(
            classify_poll(
                Protocol::Wan,
                &json!({"output": {"task_id": "w1", "task_status": "FAILED", "code": "InvalidParameter",
                        "message": "Missing required parameter 'input'!"}})
            ),
            PollOutcome::Failed(
                "InvalidParameter: Missing required parameter 'input'!".to_string()
            ),
            "a failed task reports Wan's code and message"
        );
        assert!(
            matches!(
                classify_poll(
                    Protocol::Wan,
                    &json!({"output": {"task_status": "UNKNOWN"}})
                ),
                PollOutcome::Unknown(_)
            ),
            "UNKNOWN is reported apart from a failure: the poller keeps polling it \
             within WAN_UNKNOWN_GRACE of submit, since the status API can lag"
        );
    }
}
