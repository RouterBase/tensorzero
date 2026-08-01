//! KIE media-generation variant.
//!
//! Calls KIE's async task API (`POST /api/v1/jobs/createTask`) and returns the
//! KIE task_id wrapped in a normal `ChatInferenceResult` text block, so that all
//! existing ClickHouse logging, RouterBase parsing, and observability code works
//! without changes.
//!
//! TOML config example:
//! ```toml
//! [functions.video_generate]
//! type = "chat"
//!
//! [functions.video_generate.variants.seedance]
//! type = "media"
//! model = "bytedance/seedance-1.5-pro"
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use chrono::Duration;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::config::PathWithContents;
use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::inference::{InferenceClients, InferenceModels, InferenceParams};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::batch::StartBatchModelInferenceWithMetadata;
use crate::inference::types::resolved_input::{LazyResolvedInput, LazyResolvedInputMessageContent};
use crate::inference::types::{
    ChatInferenceResult, ContentBlockChatOutput, FinishReason, InferenceResult,
    InferenceResultStream, Role, Text,
};
use crate::minijinja_util::TemplateConfig;
use crate::model::ModelTable;
use crate::providers::amux::{AmuxMediaProxyConfig, AmuxProvider};
use crate::providers::minimax::{MinimaxMediaProxyConfig, MinimaxProvider};
use crate::providers::novita::{NovitaMediaProxyConfig, NovitaProvider};
use crate::relay::TensorzeroRelay;

use super::{InferenceConfig, ModelUsedInfo, Variant};

/// Configuration for a `media` variant.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct MediaConfig {
    /// KIE model name passed verbatim to the createTask API,
    /// e.g. `"bytedance/seedance-1.5-pro"` or `"kling/kling-3.0"`.
    pub model: Arc<str>,
    /// Optional variant weight for experimentation (0.0–1.0).
    #[serde(default)]
    weight: Option<f64>,
    /// RouterBase model tags surfaced via /internal/config,
    /// e.g. `["video", "generation", "text-to-video"]`.
    /// If omitted, RouterBase infers tags from the function name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Optional non-KIE upstream proxy configuration. This keeps the
    /// RouterBase-facing contract on `media` while letting TZ forward
    /// specific variants to other media providers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<MediaProxyConfig>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum MediaProxyConfig {
    Novita(NovitaMediaProxyConfig),
    Amux(AmuxMediaProxyConfig),
    Minimax(MinimaxMediaProxyConfig),
}

impl MediaConfig {
    pub fn weight(&self) -> Option<f64> {
        self.weight
    }

    pub fn set_weight(&mut self, weight: Option<f64>) {
        self.weight = weight;
    }
}

impl Variant for MediaConfig {
    async fn infer(
        &self,
        input: Arc<LazyResolvedInput>,
        _models: InferenceModels,
        _function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error> {
        let inference_id = inference_config.ids.inference_id;

        // Extract the user prompt text from the input.
        let prompt = extract_prompt_text(&input)?;

        // Build the `input` payload for the upstream createTask call.
        // Start with any caller-supplied extra params, then inject the prompt.
        let mut media_input = inference_params.media_generation.extra.clone();
        if let Some(obj) = media_input.as_object_mut() {
            obj.entry("prompt").or_insert_with(|| prompt.clone().into());
        } else {
            media_input = serde_json::json!({ "prompt": prompt });
        }

        let task_id = match &self.proxy {
            Some(proxy) => match proxy {
                MediaProxyConfig::Novita(proxy) => {
                    NovitaProvider::infer_media_proxy(
                        proxy,
                        inference_params.media_generation.callback_url.as_deref(),
                        &media_input,
                        &clients.http_client,
                        &clients.credentials,
                    )
                    .await?
                }
                MediaProxyConfig::Amux(proxy) => {
                    AmuxProvider::infer_media_proxy(
                        proxy,
                        inference_params.media_generation.callback_url.as_deref(),
                        &media_input,
                        &clients.http_client,
                        &clients.credentials,
                    )
                    .await?
                }
                MediaProxyConfig::Minimax(proxy) => {
                    MinimaxProvider::infer_media_proxy(
                        proxy,
                        inference_params.media_generation.callback_url.as_deref(),
                        &media_input,
                        &clients.http_client,
                        &clients.credentials,
                    )
                    .await?
                }
            },
            None => {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "media variant '{}' has no proxy configured; direct KIE media generation has been removed",
                        self.model
                    ),
                }));
            }
        };

        // Serialize the task_id as a JSON string so RouterBase can parse it.
        let response_text = serde_json::to_string(&serde_json::json!({ "task_id": task_id }))
            .unwrap_or_else(|_| format!(r#"{{"task_id":"{task_id}"}}"#));

        Ok(InferenceResult::Chat(ChatInferenceResult {
            inference_id,
            created: chrono::Utc::now().timestamp() as u64,
            content: vec![ContentBlockChatOutput::Text(Text {
                text: response_text,
            })],
            model_inference_results: vec![],
            inference_params,
            original_response: None,
            finish_reason: Some(FinishReason::Stop),
        }))
    }

    async fn infer_stream(
        &self,
        _input: Arc<LazyResolvedInput>,
        _models: InferenceModels,
        _function: Arc<FunctionConfig>,
        _inference_config: Arc<InferenceConfig>,
        _clients: InferenceClients,
        _inference_params: InferenceParams,
    ) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
        // Media generation is an async task — streaming is not meaningful here.
        // Callers should always set stream=false for media variants.
        Err(ErrorDetails::InvalidRequest {
            message: "Streaming is not supported for media variants. \
                      Set `stream: false` in your inference request."
                .to_string(),
        }
        .into())
    }

    async fn validate(
        &self,
        _function: Arc<FunctionConfig>,
        _models: &ModelTable,
        _embedding_models: &EmbeddingModelTable,
        _templates: &TemplateConfig<'_>,
        _function_name: &str,
        _variant_name: &str,
        _global_outbound_http_timeout: &Duration,
        _relay: Option<&TensorzeroRelay>,
    ) -> Result<(), Error> {
        if self.model.is_empty() {
            return Err(ErrorDetails::Config {
                message: "media variant requires a non-empty `model` field".to_string(),
            }
            .into());
        }
        if let Some(proxy) = &self.proxy {
            let path = match proxy {
                MediaProxyConfig::Novita(proxy) => &proxy.path,
                MediaProxyConfig::Amux(proxy) => &proxy.path,
                MediaProxyConfig::Minimax(proxy) => &proxy.path,
            };
            if path.is_empty() {
                return Err(ErrorDetails::Config {
                    message: "media proxy requires a non-empty `path` field".to_string(),
                }
                .into());
            }
        }
        Ok(())
    }

    fn get_all_template_paths(&self) -> Vec<&PathWithContents> {
        vec![]
    }

    fn get_all_explicit_template_names(&self) -> HashSet<String> {
        HashSet::new()
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _input: &[LazyResolvedInput],
        _models: InferenceModels,
        _function: &'a FunctionConfig,
        _inference_configs: &'a [InferenceConfig],
        _clients: InferenceClients,
        _inference_params: Vec<InferenceParams>,
    ) -> Result<StartBatchModelInferenceWithMetadata<'a>, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "media".to_string(),
        }
        .into())
    }
}

/// Extract a text prompt from the last user message in the lazy resolved input.
fn extract_prompt_text(input: &LazyResolvedInput) -> Result<String, Error> {
    input
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .and_then(|m| {
            m.content.iter().find_map(|block| {
                if let LazyResolvedInputMessageContent::Text(t) = block {
                    Some(t.text.clone())
                } else {
                    None
                }
            })
        })
        .ok_or_else(|| {
            ErrorDetails::InvalidRequest {
                message: "media variant requires at least one user text message as the prompt"
                    .to_string(),
            }
            .into()
        })
}
