use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest_eventsource::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;

use super::helpers::{convert_stream_error, inject_extra_request_data_and_send, inject_extra_request_data_and_send_eventsource};
use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{DelayedError, DisplayOrDebugGateway, Error, ErrorDetails};
use crate::http::{TensorZeroEventSource, TensorzeroHttpClient};
use crate::inference::InferenceProvider;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse};
use crate::inference::types::chat_completion_inference_params::{ChatCompletionInferenceParamsV2, warn_inference_parameter_not_supported};
use crate::inference::types::usage::raw_usage_entries_from_value;
use crate::inference::types::{
    ApiType, ContentBlockChunk, ContentBlockOutput, Latency, ModelInferenceRequest, ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseArgs, ProviderInferenceResponseChunk, ProviderInferenceResponseStreamInner, TextChunk, ThoughtChunk, Thought,
};
use crate::model::{Credential, ModelProvider};
use crate::providers::chat_completions::prepare_chat_completion_tools;
use crate::providers::chat_completions::{ChatCompletionTool, ChatCompletionToolChoice};
use crate::providers::openai::{
    OpenAIFinishReason, OpenAIRequestMessage, OpenAIResponseToolCall, OpenAIUsage,
    StreamOptions, SystemOrDeveloper, handle_openai_error, prepare_system_or_developer_message, tensorzero_to_openai_messages,
};
use uuid::Uuid;

lazy_static! {
    static ref KIE_API_BASE: &'static str = "https://api.kie.ai";
}

const PROVIDER_NAME: &str = "KIE";
pub const PROVIDER_TYPE: &str = "kie";


#[derive(Clone, Debug)]
pub enum KIECredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<KIECredentials>,
        fallback: Box<KIECredentials>,
    },
}

impl TryFrom<Credential> for KIECredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(KIECredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(KIECredentials::Dynamic(key_name)),
            Credential::Missing => Ok(KIECredentials::None),
            Credential::WithFallback { default, fallback } => {
                Ok(KIECredentials::WithFallback {
                    default: Box::new((*default).try_into()?),
                    fallback: Box::new((*fallback).try_into()?),
                })
            }
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for KIE provider".to_string(),
            })),
        }
    }
}

impl KIECredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, DelayedError> {
        match self {
            KIECredentials::Static(api_key) => Ok(api_key),
            KIECredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                })
            }
            KIECredentials::WithFallback { default, fallback } => {
                match default.get_api_key(dynamic_api_keys) {
                    Ok(key) => Ok(key),
                    Err(e) => {
                        e.log_at_level(
                            "Using fallback credential, as default credential is unavailable: ",
                            tracing::Level::WARN,
                        );
                        fallback.get_api_key(dynamic_api_keys)
                    }
                }
            }
            KIECredentials::None => Err(DelayedError::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "No credentials are set".to_string(),
            })),
        }
    }
}

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct KIEProvider {
    model_name: String,
    #[serde(skip)]
    credentials: KIECredentials,
}

impl KIEProvider {
    pub fn new(model_name: String, credentials: KIECredentials) -> Self {
        KIEProvider {
            model_name,
            credentials,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl InferenceProvider for KIEProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
            model_inference_id,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = serde_json::to_value(KIERequest::new(self.model_name.as_str(), request).await?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing KIE request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;

        let request_url = format!("{}/{}/v1/chat/completions", *KIE_API_BASE, self.model_name)
            .parse::<Url>()
            .map_err(|e| {
                Error::new(ErrorDetails::InvalidBaseUrl {
                    message: format!("Failed to construct KIE chat URL: {e}"),
                })
            })?;

        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;

        let start_time = Instant::now();
        let request_builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());

        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            request_builder,
        )
        .await?;

        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            tracing::info!("raw_response: {}", raw_response);
            let response: KIEResponse = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing JSON response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };

            Ok(KIEResponseWithMetadata {
                response,
                raw_response,
                latency,
                raw_request,
                generic_request: request,
                model_inference_id,
            }
            .try_into()?)
        } else {
            let status = res.status();

            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            Err(handle_openai_error(
                &raw_request,
                status,
                &response,
                PROVIDER_TYPE,
                None,
            ))
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
            model_inference_id,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let mut request_body = serde_json::to_value(KIERequest::new(self.model_name.as_str(), request).await?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing KIE request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;

        request_body["stream"] = serde_json::json!(true);

        let request_url = format!("{}/{}/v1/chat/completions", *KIE_API_BASE, self.model_name)
            .parse::<Url>()
            .map_err(|e| {
                Error::new(ErrorDetails::InvalidBaseUrl {
                    message: format!("Failed to construct KIE chat URL: {e}"),
                })
            })?;

        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;

        let start_time = Instant::now();
        let request_builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());

        let (event_source, raw_request) = inject_extra_request_data_and_send_eventsource(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            request_builder,
        )
        .await?;

        let stream = stream_kie(event_source, start_time, &raw_request, model_inference_id).peekable();

        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}
#[derive(Debug, Default, Serialize)]
struct KIERequest<'a> {
    #[serde(skip_serializing)]
    model: &'a str,
    messages: Vec<OpenAIRequestMessage<'a>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<KIEResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ChatCompletionTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ChatCompletionToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_thoughts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum KIEResponseFormat {
    #[default]
    Text,
    JsonObject,
}

impl KIEResponseFormat {
    fn new(json_mode: ModelInferenceRequestJsonMode) -> Self {
        match json_mode {
            ModelInferenceRequestJsonMode::Off => KIEResponseFormat::Text,
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                KIEResponseFormat::JsonObject
            }
        }
    }
}

fn apply_inference_params(
    request: &mut KIERequest,
    inference_params: &ChatCompletionInferenceParamsV2,
) {
    let ChatCompletionInferenceParamsV2 {
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
    } = inference_params;

    // Apply reasoning_effort if provided
    if let Some(effort) = reasoning_effort {
        // Validate and map reasoning_effort to KIE valid values
        let normalized_effort = match effort.as_str() {
            "low" | "medium" | "high" => {
                // Map "medium" to "high" since KIE only supports "low" and "high"
                if effort == "medium" {
                    "high"
                } else {
                    effort.as_str()
                }
            }
            _ => "high", // default to high
        };
        request.reasoning_effort = Some(normalized_effort.to_string());
    }

    if service_tier.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "service_tier", None);
    }

    if thinking_budget_tokens.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "thinking_budget_tokens", None);
    }

    if verbosity.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "verbosity", None);
    }
}

impl<'a> KIERequest<'a> {
    pub async fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<KIERequest<'a>, Error> {
        let ModelInferenceRequest {
            temperature,
            max_tokens,
            seed,
            top_p,
            presence_penalty,
            frequency_penalty,
            stream,
            ..
        } = *request;

        let stream_options = if request.stream {
            Some(StreamOptions {
                include_usage: true,
            })
        } else {
            None
        };

        if request.json_mode == ModelInferenceRequestJsonMode::Strict {
            tracing::warn!(
                "KIE provider does not support strict JSON mode. Downgrading to normal JSON mode."
            );
        }

        let response_format = KIEResponseFormat::new(request.json_mode);

        let mut messages = Vec::with_capacity(request.messages.len());
        for message in &request.messages {
            messages.extend(
                tensorzero_to_openai_messages(
                    message,
                    crate::providers::openai::OpenAIMessagesConfig {
                        json_mode: Some(&request.json_mode),
                        provider_type: PROVIDER_TYPE,
                        fetch_and_encode_input_files_before_inference: request
                            .fetch_and_encode_input_files_before_inference,
                    },
                )
                .await?,
            );
        }

        if let Some(system_msg) = prepare_system_or_developer_message(
            request
                .system
                .as_deref()
                .map(|m| SystemOrDeveloper::System(Cow::Borrowed(m))),
            Some(&request.json_mode),
            &messages,
        ) {
            messages.insert(0, system_msg);
        }

        let (tools, tool_choice, _) = prepare_chat_completion_tools(request, false)?;

        let mut kie_request = KIERequest {
            model,
            messages,
            temperature,
            max_tokens,
            seed,
            top_p,
            stop: request.borrow_stop_sequences(),
            presence_penalty,
            frequency_penalty,
            stream,
            stream_options,
            response_format: Some(response_format),
            tools,
            tool_choice,
            include_thoughts: Some(true),
            reasoning_effort: Some("high".to_string()),
        };

        apply_inference_params(&mut kie_request, &request.inference_params_v2);

        Ok(kie_request)
    }
}

struct KIEResponseWithMetadata<'a> {
    response: KIEResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
    model_inference_id: Uuid,
}

impl<'a> TryFrom<KIEResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: KIEResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let KIEResponseWithMetadata {
            mut response,
            raw_response,
            latency,
            raw_request,
            generic_request,
            model_inference_id,
        } = value;

        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices {}, Expected 1",
                    response.choices.len()
                ),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
        }

        let KIEResponseChoice {
            message,
            finish_reason,
            ..
        } = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }))?;

        let mut content: Vec<ContentBlockOutput> = Vec::new();
        if let Some(reasoning) = message.reasoning_content {
            content.push(ContentBlockOutput::Thought(Thought {
                text: Some(reasoning),
                signature: None,
                summary: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
            }));
        }
        if let Some(text) = message.content {
            content.push(ContentBlockOutput::Text(crate::inference::types::Text {
                text,
            }));
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockOutput::ToolCall(tool_call.into()));
            }
        }

        let raw_usage = kie_response_to_raw_usage(&response, model_inference_id);
        let usage = response.usage.into();
        let system = generic_request.system.clone();
        let messages = generic_request.messages.clone();

        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages: messages,
                raw_request,
                raw_response,
                usage,
                raw_usage,
                provider_latency: latency,
                finish_reason: finish_reason.map(OpenAIFinishReason::into),
                id: model_inference_id,
                relay_raw_response: None,
            },
        ))
    }
}

fn stream_kie(
    mut event_source: TensorZeroEventSource,
    start_time: Instant,
    raw_request: &str,
    model_inference_id: Uuid,
) -> ProviderInferenceResponseStreamInner {
    let raw_request = raw_request.to_string();
    Box::pin(async_stream::stream! {
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    yield Err(convert_stream_error(raw_request.clone(), PROVIDER_TYPE.to_string(), *e, None).await);
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<KIEChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing chunk. Error: {e}",
                                ),
                                raw_request: Some(raw_request.clone()),
                                raw_response: Some(message.data.clone()),
                                provider_type: PROVIDER_TYPE.to_string(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            kie_to_tensorzero_chunk(
                                message.data,
                                d,
                                latency,
                                model_inference_id,
                            )
                        });
                        yield stream_message;
                    }
                },
            }
        }
    })
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct KIEChatChunkChoice {
    delta: KIEStreamDelta,
    finish_reason: Option<OpenAIFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct KIEChatChunk {
    choices: Vec<KIEChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAIUsage>,
}

/// Maps a KIE chunk to a TensorZero chunk for streaming inferences
fn kie_to_tensorzero_chunk(
    raw_message: String,
    mut chunk: KIEChatChunk,
    latency: Duration,
    model_inference_id: Uuid,
) -> Result<ProviderInferenceResponseChunk, Error> {
    if chunk.choices.len() > 1 {
        return Err(ErrorDetails::InferenceServer {
            message: "Response has invalid number of choices. Expected 1.".to_string(),
            raw_request: None,
            raw_response: Some(serde_json::to_string(&chunk).unwrap_or_default()),
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into());
    }

    let raw_usage = kie_response_to_raw_usage_from_chunk(&chunk, model_inference_id);
    let usage = chunk.usage.map(|u| u.into());
    let mut content = vec![];
    let mut finish_reason = None;

    if let Some(choice) = chunk.choices.pop() {
        if let Some(choice_finish_reason) = choice.finish_reason {
            finish_reason = Some(choice_finish_reason.into());
        }
        if let Some(text) = choice.delta.content {
            content.push(ContentBlockChunk::Text(TextChunk {
                text,
                id: "0".to_string(),
            }));
        }
        if let Some(reasoning) = choice.delta.reasoning_content {
            content.push(ContentBlockChunk::Thought(ThoughtChunk {
                text: Some(reasoning),
                signature: None,
                id: "0".to_string(),
                summary_id: None,
                summary_text: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
            }));
        }
        if let Some(tool_calls) = choice.delta.tool_calls {
            for _tool_call in tool_calls {
                // TODO: Handle streaming tool calls when available
                // For now, skip tool calls in streaming
            }
        }
    }

    Ok(ProviderInferenceResponseChunk::new_with_raw_usage(
        content,
        usage,
        raw_message,
        latency,
        finish_reason,
        raw_usage,
    ))
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct KIEStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIResponseToolCall>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct KIEResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIResponseToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct KIEResponseChoice {
    index: u8,
    message: KIEResponseMessage,
    finish_reason: Option<OpenAIFinishReason>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct KIEResponse {
    choices: Vec<KIEResponseChoice>,
    usage: OpenAIUsage,
}

#[expect(dead_code)]
fn extract_content_blocks_from_kie_response(response: &KIEResponse) -> Vec<ContentBlockOutput> {
    let mut content_blocks = Vec::new();

    for choice in &response.choices {
        let message = &choice.message;

        // Add reasoning content as thought block if present
        if let Some(ref reasoning_content) = message.reasoning_content {
            content_blocks.push(ContentBlockOutput::Thought(Thought {
                text: Some(reasoning_content.clone()),
                signature: None,
                summary: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
            }));
        }

        // Add text content
        if let Some(ref text_content) = message.content {
            content_blocks.push(ContentBlockOutput::Text(crate::inference::types::Text {
                text: text_content.clone(),
            }));
        }

        // Add tool calls
        if let Some(ref tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content_blocks.push(ContentBlockOutput::ToolCall(tool_call.clone().into()));
            }
        }
    }

    content_blocks
}

fn kie_response_to_raw_usage(
    response: &KIEResponse,
    model_inference_id: Uuid,
) -> Option<Vec<crate::inference::types::RawUsageEntry>> {
    let usage_value = serde_json::to_value(response).ok()?;
    let usage = usage_value.get("usage")?;
    if usage.is_null() {
        return None;
    }
    Some(raw_usage_entries_from_value(
        model_inference_id,
        PROVIDER_TYPE,
        ApiType::ChatCompletions,
        usage.clone(),
    ))
}

fn kie_response_to_raw_usage_from_chunk(
    chunk: &KIEChatChunk,
    model_inference_id: Uuid,
) -> Option<Vec<crate::inference::types::RawUsageEntry>> {
    let chunk_value = serde_json::to_value(chunk).ok()?;
    let usage = chunk_value.get("usage")?;
    if usage.is_null() {
        return None;
    }
    Some(raw_usage_entries_from_value(
        model_inference_id,
        PROVIDER_TYPE,
        ApiType::ChatCompletions,
        usage.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kie_request_new() {
        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            temperature: Some(0.5),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: crate::inference::types::FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let kie_request = KIERequest::new("gemini-3-pro", &request_with_tools)
            .await
            .expect("failed to create KIE Request during test");

        assert_eq!(kie_request.temperature, Some(0.5), "Expected temperature to be 0.5");
        assert_eq!(kie_request.max_tokens, Some(100), "Expected max_tokens to be 100");
        assert!(!kie_request.stream, "Expected stream to be false");
        assert_eq!(kie_request.seed, Some(69), "Expected seed to be 69");
    }

    #[test]
    fn test_kie_url_construction() {
        // Test that URLs are constructed correctly with model name
        let model_name = "gemini-3-pro";
        let expected_url = format!("{}/{}/v1/chat/completions", *KIE_API_BASE, model_name);
        
        assert_eq!(
            expected_url,
            "https://api.kie.ai/gemini-3-pro/v1/chat/completions",
            "Expected URL to include model name in correct format"
        );
    }

    #[test]
    fn test_reasoning_effort_mapping() {
        let mut request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(100),
            stream: false,
            seed: None,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: crate::inference::types::FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        // Test with "medium" reasoning_effort (should be mapped to "high")
        request.inference_params_v2.reasoning_effort = Some("medium".to_string());
        
        let mut kie_request = KIERequest {
            model: "kie-chat",
            messages: vec![],
            temperature: None,
            max_tokens: Some(100),
            seed: None,
            top_p: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            stream_options: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            include_thoughts: Some(true),
            reasoning_effort: Some("high".to_string()),
        };

        apply_inference_params(&mut kie_request, &request.inference_params_v2);
        
        assert_eq!(
            kie_request.reasoning_effort,
            Some("high".to_string()),
            "Expected 'medium' to be mapped to 'high'"
        );

        // Test with "low" reasoning_effort (should remain "low")
        request.inference_params_v2.reasoning_effort = Some("low".to_string());
        
        let mut kie_request_low = KIERequest {
            model: "kie-chat",
            messages: vec![],
            temperature: None,
            max_tokens: Some(100),
            seed: None,
            top_p: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            stream_options: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            include_thoughts: Some(true),
            reasoning_effort: Some("high".to_string()),
        };

        apply_inference_params(&mut kie_request_low, &request.inference_params_v2);
        
        assert_eq!(
            kie_request_low.reasoning_effort,
            Some("low".to_string()),
            "Expected 'low' to remain 'low'"
        );

        // Test with "high" reasoning_effort (should remain "high")
        request.inference_params_v2.reasoning_effort = Some("high".to_string());
        
        let mut kie_request_high = KIERequest {
            model: "kie-chat",
            messages: vec![],
            temperature: None,
            max_tokens: Some(100),
            seed: None,
            top_p: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            stream_options: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            include_thoughts: Some(true),
            reasoning_effort: Some("high".to_string()),
        };

        apply_inference_params(&mut kie_request_high, &request.inference_params_v2);
        
        assert_eq!(
            kie_request_high.reasoning_effort,
            Some("high".to_string()),
            "Expected 'high' to remain 'high'"
        );
    }
}

