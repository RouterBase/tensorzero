//! `POST /internal/media/task_status`: one status read of a media task a
//! media variant submitted earlier.
//!
//! RouterBase calls this for generations whose media callback is overdue,
//! which happens when the gateway process that was polling the task is gone
//! (restart, rollout) or its callback was lost. See
//! [`crate::providers::media_task`].

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::providers::media_task::MediaTaskCheck;
use crate::utils::gateway::{AppState, StructuredJson};
use crate::variant::VariantConfig;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediaTaskStatusRequest {
    pub function_name: String,
    pub variant_name: String,
    /// The task id the submit returned.
    pub task_id: String,
    /// The `resume` value the submit returned, verbatim, when it returned one.
    #[serde(default)]
    pub resume: Option<Value>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaTaskState {
    /// The upstream is still working on the task.
    Pending,
    /// Terminal. `callback` is the media callback body for the task.
    Finished,
    /// The variant keeps no upstream task that can be read.
    Unsupported,
    /// The status could not be read this time. `reason` says why. Not a
    /// verdict on the task: ask again later.
    Unknown,
}

#[derive(Debug, Serialize)]
pub struct MediaTaskStatusResponse {
    pub status: MediaTaskState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl MediaTaskStatusResponse {
    fn from_check(check: Result<MediaTaskCheck, Error>) -> Self {
        let (status, callback, reason) = match check {
            Ok(MediaTaskCheck::Pending) => (MediaTaskState::Pending, None, None),
            Ok(MediaTaskCheck::Finished(body)) => (MediaTaskState::Finished, Some(body), None),
            Ok(MediaTaskCheck::Unsupported) => (MediaTaskState::Unsupported, None, None),
            Err(e) => (MediaTaskState::Unknown, None, Some(e.to_string())),
        };
        Self {
            status,
            callback,
            reason,
        }
    }
}

pub async fn media_task_status_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<MediaTaskStatusRequest>,
) -> Result<Json<MediaTaskStatusResponse>, Error> {
    let function = app_state.config.get_function(&request.function_name)?;
    let variant = function
        .variants()
        .get(&request.variant_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::UnknownVariant {
                name: request.variant_name.clone(),
            })
        })?;
    let VariantConfig::Media(media) = &variant.inner else {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "variant `{}` of function `{}` is not a media variant",
                request.variant_name, request.function_name
            ),
        }));
    };
    let check = media
        .check_task(
            &request.task_id,
            request.resume.as_ref(),
            &app_state.http_client,
            &InferenceCredentials::default(),
        )
        .await;
    Ok(Json(MediaTaskStatusResponse::from_check(check)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_finished_check_carries_the_callback_body() {
        let body = json!({ "state": "success", "resultUrls": ["https://x/a.mp4"] });
        let response =
            MediaTaskStatusResponse::from_check(Ok(MediaTaskCheck::Finished(body.clone())));
        let wire = serde_json::to_value(&response).unwrap();
        assert_eq!(wire, json!({ "status": "finished", "callback": body }));
    }

    #[test]
    fn pending_and_unsupported_carry_nothing_else() {
        for (check, status) in [
            (MediaTaskCheck::Pending, "pending"),
            (MediaTaskCheck::Unsupported, "unsupported"),
        ] {
            let wire =
                serde_json::to_value(MediaTaskStatusResponse::from_check(Ok(check))).unwrap();
            assert_eq!(wire, json!({ "status": status }));
        }
    }

    #[test]
    fn an_unreadable_status_is_unknown_never_a_failed_task() {
        let err = Error::new(ErrorDetails::InferenceServer {
            message: "Tripo task status unavailable: HTTP 502".to_string(),
            provider_type: "tripo".to_string(),
            raw_request: None,
            raw_response: None,
        });
        let wire = serde_json::to_value(MediaTaskStatusResponse::from_check(Err(err))).unwrap();
        assert_eq!(wire["status"], "unknown");
        assert!(wire.get("callback").is_none());
        assert!(wire["reason"].as_str().unwrap().contains("HTTP 502"));
    }

    #[test]
    fn unknown_request_fields_are_rejected() {
        let parsed: Result<MediaTaskStatusRequest, _> = serde_json::from_value(json!({
            "function_name": "video_generate",
            "variant_name": "v",
            "task_id": "t",
            "callback_url": "https://example.com"
        }));
        assert!(parsed.is_err());
    }
}
