use serde::Deserialize;
use serde::Serialize;
use strum_macros::Display;
use strum_macros::EnumIter;
use ts_rs::TS;

/// See https://platform.openai.com/docs/guides/reasoning?api-mode=responses#get-started-with-reasoning
#[derive(
    Debug, Serialize, Deserialize, Default, Clone, Copy, PartialEq, Eq, Display, TS, EnumIter,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum ReasoningEffort {
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    /// Explicitly disables reasoning. Serialized as `none`: an explicit `none` is
    /// sent as `"effort":"none"` on the wire, never omitted and never coerced to
    /// `minimal`.
    None,
}

/// A summary of the reasoning performed by the model. This can be useful for
/// debugging and understanding the model's reasoning process.
/// See https://platform.openai.com/docs/guides/reasoning?api-mode=responses#reasoning-summaries
#[derive(Debug, Serialize, Deserialize, Default, Clone, Copy, PartialEq, Eq, Display, TS)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum ReasoningSummary {
    #[default]
    Auto,
    Concise,
    Detailed,
    /// Option to disable reasoning summaries.
    None,
}

/// Controls output length/detail on GPT-5 models via the Responses API.
/// Serialized with lowercase values to match the OpenAI API.
#[derive(Debug, Serialize, Deserialize, Default, Clone, Copy, PartialEq, Eq, Display, TS)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Verbosity {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Default, Serialize, Display, TS)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum SandboxMode {
    #[serde(rename = "read-only")]
    #[default]
    ReadOnly,

    #[serde(rename = "workspace-write")]
    WorkspaceWrite,

    #[serde(rename = "danger-full-access")]
    DangerFullAccess,
}

#[cfg(test)]
mod tests {
    use super::ReasoningEffort;
    use pretty_assertions::assert_eq;

    /// Regression: an explicit `none` must serialize as the exact string `none`,
    /// never be omitted and never collapse into `minimal`.
    #[test]
    fn reasoning_effort_none_is_explicit_and_never_minimal() {
        assert_eq!(ReasoningEffort::None.to_string(), "none");
        assert_eq!(
            serde_json::to_value(ReasoningEffort::None).expect("serialize explicit none"),
            serde_json::json!("none")
        );
        assert_eq!(
            serde_json::from_value::<ReasoningEffort>(serde_json::json!("none"))
                .expect("explicit none must deserialize"),
            ReasoningEffort::None
        );
        // Existing effort values keep their exact serialization.
        assert_eq!(ReasoningEffort::Minimal.to_string(), "minimal");
        assert_eq!(ReasoningEffort::default(), ReasoningEffort::Medium);
        assert_eq!(
            serde_json::to_value(ReasoningEffort::High).expect("serialize high"),
            serde_json::json!("high")
        );
    }
}
