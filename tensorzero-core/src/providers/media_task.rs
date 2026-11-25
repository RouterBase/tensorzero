//! Reading an already-submitted media task once, without the poller.
//!
//! An async media submit returns the upstream task id and finishes in a
//! detached task that polls the upstream and reports to RouterBase's media
//! callback. That poller lives in this process: a gateway restart, a rollout
//! or a lost callback leaves the upstream task running (and billed) with
//! nobody listening. RouterBase stores the task id and asks for one status
//! read through `POST /internal/media/task_status`; a terminal answer carries
//! the exact body the callback would have, so both paths settle a generation
//! the same way.

use serde_json::{Value, json};

/// One read of a submitted media task.
#[derive(Debug, PartialEq)]
pub enum MediaTaskCheck {
    /// The upstream is still working on it.
    Pending,
    /// Terminal, success or failure: the media callback body for it.
    Finished(Value),
    /// The variant keeps no upstream task to read (inline endpoints that
    /// return the media on the submit response).
    Unsupported,
}

/// What an accepted media submit hands back to RouterBase.
#[derive(Debug, PartialEq)]
pub struct MediaSubmission {
    pub task_id: String,
    /// Opaque state a later [`MediaTaskCheck`] needs beyond the task id.
    /// RouterBase stores it and sends it back verbatim.
    pub resume: Option<Value>,
}

impl From<String> for MediaSubmission {
    fn from(task_id: String) -> Self {
        Self {
            task_id,
            resume: None,
        }
    }
}

/// The `fail` body RouterBase's media callback handler reads: state at the
/// top level, the reason as `errorMessage` and `data.failMsg`.
pub fn failure_callback_body(task_id: &str, error_message: &str) -> Value {
    json!({
        "taskId": task_id,
        "task_id": task_id,
        "state": "fail",
        "errorMessage": error_message,
        "data": {
            "taskId": task_id,
            "task_id": task_id,
            "failMsg": error_message,
        }
    })
}

/// The text an upstream-reported failure carries on the callback. The poller
/// reports `Error::to_string()` of an `InferenceServer` error; a check must
/// report the same string, because RouterBase derives the caller-facing
/// reason from it.
pub fn server_error_text(provider_type: &str, message: &str) -> String {
    format!("Error from {provider_type} server: {message}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{Error, ErrorDetails};

    #[test]
    fn server_error_text_matches_what_the_poller_reports() {
        let from_poller = Error::new(ErrorDetails::InferenceServer {
            message: "Tripo generation failed: nsfw".to_string(),
            provider_type: "tripo".to_string(),
            raw_request: None,
            raw_response: None,
        })
        .to_string();
        assert_eq!(
            server_error_text("tripo", "Tripo generation failed: nsfw"),
            from_poller
        );
    }

    #[test]
    fn failure_body_carries_state_and_reason_where_the_callback_reads_them() {
        let body = failure_callback_body("t1", "boom");
        assert_eq!(body["state"], "fail");
        assert_eq!(body["errorMessage"], "boom");
        assert_eq!(body["data"]["failMsg"], "boom");
        assert_eq!(body["data"]["taskId"], "t1");
    }

    #[test]
    fn a_bare_task_id_is_a_submission_without_resume_state() {
        let s = MediaSubmission::from("abc".to_string());
        assert_eq!(s.task_id, "abc");
        assert!(s.resume.is_none());
    }
}
