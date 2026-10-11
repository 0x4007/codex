use super::*;
use codex_protocol::config_types::CollaborationMode;
use codex_protocol::config_types::Settings;
use codex_protocol::openai_models::ReasoningEffort;
use pretty_assertions::assert_eq;

#[test]
fn preset_names_use_mode_display_names() {
    let presets = builtin_collaboration_mode_presets();
    let [plan, default] = presets.as_slice() else {
        panic!("expected Plan and Default presets");
    };
    assert_eq!(plan.name, ModeKind::Plan.display_name());
    assert_eq!(default.name, ModeKind::Default.display_name());
    assert_eq!(plan.model, None);
    assert_eq!(plan.reasoning_effort, None);
    assert_eq!(default.model, None);
    assert_eq!(default.reasoning_effort, None);
}

#[test]
fn switching_modes_preserves_selected_model_and_effort() {
    let presets = builtin_collaboration_mode_presets();
    let [plan_mask, default_mask] = presets.as_slice() else {
        panic!("expected Plan and Default presets");
    };
    for effort in [
        None,
        Some(ReasoningEffort::None),
        Some(ReasoningEffort::Minimal),
        Some(ReasoningEffort::Low),
        Some(ReasoningEffort::Medium),
        Some(ReasoningEffort::High),
        Some(ReasoningEffort::XHigh),
        Some(ReasoningEffort::Max),
        Some(ReasoningEffort::Ultra),
        Some(ReasoningEffort::Persistent),
        Some(ReasoningEffort::Custom("model-defined".to_string())),
    ] {
        let initial = CollaborationMode {
            mode: ModeKind::Default,
            settings: Settings {
                model: "selected-model".to_string(),
                reasoning_effort: effort.clone(),
                developer_instructions: None,
            },
        };
        let plan = initial.apply_mask(plan_mask);
        assert_eq!(plan.mode, ModeKind::Plan);
        assert_eq!(plan.model(), initial.model());
        assert_eq!(plan.reasoning_effort(), effort);

        let restored = plan.apply_mask(default_mask);
        assert_eq!(restored.mode, ModeKind::Default);
        assert_eq!(restored.model(), initial.model());
        assert_eq!(restored.reasoning_effort(), effort);
    }
}

#[test]
fn default_mode_instructions_follow_user_input_tool_availability() {
    let presets = builtin_collaboration_mode_presets();
    let default_instructions = presets[1]
        .developer_instructions
        .as_ref()
        .expect("default preset should include instructions")
        .as_ref()
        .expect("default instructions should be set");

    assert!(default_instructions.contains(
        "Use the `request_user_input` tool only when it is listed in the available tools"
    ));
    assert!(
        default_instructions.contains("Ask the user directly with one concise plain-text question")
    );
}
