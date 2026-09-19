use std::borrow::Cow;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use crate::AuthManager;
use crate::client_common::REVIEW_PROMPT;
use crate::event_mapping::map_response_item_to_event_messages;
use async_channel::Receiver;
use async_channel::Sender;
use chrono::DateTime;
use chrono::Utc;
use codex_apply_patch::ApplyPatchAction;
use codex_apply_patch::MaybeApplyPatchVerified;
use codex_apply_patch::maybe_parse_apply_patch_verified;
use codex_protocol::mcp_protocol::ConversationId;
use codex_protocol::protocol::ConversationPathResponseEvent;
use codex_protocol::protocol::ExitedReviewModeEvent;
use codex_protocol::protocol::ReviewRequest;
use codex_protocol::protocol::RolloutItem;
use codex_protocol::protocol::TaskStartedEvent;
use codex_protocol::protocol::TurnAbortReason;
use codex_protocol::protocol::TurnAbortedEvent;
use codex_protocol::protocol::TurnContextItem;
use futures::prelude::*;
use mcp_types::CallToolResult;
use serde::Deserialize;
use serde::Serialize;
use serde_json;
use tokio::sync::oneshot;
use tokio::task::AbortHandle;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::trace;
use tracing::warn;

use crate::ModelProviderInfo;
use crate::apply_patch;
use crate::apply_patch::ApplyPatchExec;
use crate::apply_patch::CODEX_APPLY_PATCH_ARG1;
use crate::apply_patch::InternalApplyPatchInvocation;
use crate::apply_patch::convert_apply_patch_to_protocol;
use crate::client::ModelClient;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::config::Config;
use crate::config_types::ShellEnvironmentPolicy;
use crate::conversation_history::ConversationHistory;
use crate::environment_context::EnvironmentContext;
use crate::error::CodexErr;
use crate::error::Result as CodexResult;
use crate::error::SandboxErr;
use crate::error::get_error_message_ui;
use crate::exec::ExecParams;
use crate::exec::ExecToolCallOutput;
use crate::exec::SandboxType;
use crate::exec::StdoutStream;
use crate::exec::StreamOutput;
use crate::exec::process_exec_tool_call;
use crate::exec_command::EXEC_COMMAND_TOOL_NAME;
use crate::exec_command::ExecCommandParams;
use crate::exec_command::ExecSessionManager;
use crate::exec_command::WRITE_STDIN_TOOL_NAME;
use crate::exec_command::WriteStdinParams;
use crate::exec_env::create_env;
use crate::mcp_connection_manager::McpConnectionManager;
use crate::mcp_tool_call::handle_mcp_tool_call;
use crate::model_family::find_family_for_model;
use crate::openai_model_info::get_model_info;
use crate::openai_tools::ApplyPatchToolArgs;
use crate::openai_tools::ToolsConfig;
use crate::openai_tools::ToolsConfigParams;
use crate::openai_tools::get_openai_tools;
use crate::parse_command::parse_command;
use crate::plan_tool::handle_update_plan;
use crate::project_doc::get_user_instructions;
use crate::protocol::AgentMessageDeltaEvent;
use crate::protocol::AgentReasoningDeltaEvent;
use crate::protocol::AgentReasoningRawContentDeltaEvent;
use crate::protocol::AgentReasoningSectionBreakEvent;
use crate::protocol::ApplyPatchApprovalRequestEvent;
use crate::protocol::AskForApproval;
use crate::protocol::BackgroundEventEvent;
use crate::protocol::ErrorEvent;
use crate::protocol::Event;
use crate::protocol::EventMsg;
use crate::protocol::ExecApprovalRequestEvent;
use crate::protocol::ExecCommandBeginEvent;
use crate::protocol::ExecCommandEndEvent;
use crate::protocol::FileChange;
use crate::protocol::InputItem;
use crate::protocol::ListCustomPromptsResponseEvent;
use crate::protocol::Op;
use crate::protocol::PatchApplyBeginEvent;
use crate::protocol::PatchApplyEndEvent;
use crate::protocol::ReviewDecision;
use crate::protocol::ReviewOutputEvent;
use crate::protocol::SandboxPolicy;
use crate::protocol::SessionConfiguredEvent;
use crate::protocol::StreamErrorEvent;
use crate::protocol::Submission;
use crate::protocol::TaskCompleteEvent;
use crate::protocol::TokenUsage;
use crate::protocol::TokenUsageInfo;
use crate::protocol::TurnDiffEvent;
use crate::protocol::WebSearchBeginEvent;
use crate::rollout::RolloutRecorder;
use crate::rollout::RolloutRecorderParams;
use crate::safety::SafetyCheck;
use crate::safety::assess_command_safety;
use crate::safety::assess_safety_for_untrusted_command;
use crate::shell;
use crate::turn_diff_tracker::TurnDiffTracker;
use crate::unified_exec::UnifiedExecSessionManager;
use crate::user_instructions::UserInstructions;
use crate::user_notification::UserNotification;
use crate::util::backoff;
use crate::working_set;
use codex_protocol::config_types::ReasoningEffort as ReasoningEffortConfig;
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::custom_prompts::CustomPrompt;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::LocalShellAction;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::ShellToolCallParams;
use codex_protocol::protocol::InitialHistory;

mod compact;
use self::compact::build_compacted_history;
use self::compact::collect_user_messages;

// A convenience extension trait for acquiring mutex locks where poisoning is
// unrecoverable and should abort the program. This avoids scattered `.unwrap()`
// calls on `lock()` while still surfacing a clear panic message when a lock is
// poisoned.
trait MutexExt<T> {
    fn lock_unchecked(&self) -> MutexGuard<'_, T>;
}

impl<T> MutexExt<T> for Mutex<T> {
    fn lock_unchecked(&self) -> MutexGuard<'_, T> {
        #[expect(clippy::expect_used)]
        self.lock().expect("poisoned lock")
    }
}

/// The high-level interface to the Codex system.
/// It operates as a queue pair where you send submissions and receive events.
pub struct Codex {
    next_id: AtomicU64,
    tx_sub: Sender<Submission>,
    rx_event: Receiver<Event>,
}

/// Wrapper returned by [`Codex::spawn`] containing the spawned [`Codex`],
/// the submission id for the initial `ConfigureSession` request and the
/// unique session id.
pub struct CodexSpawnOk {
    pub codex: Codex,
    pub conversation_id: ConversationId,
}

pub(crate) const INITIAL_SUBMIT_ID: &str = "";
pub(crate) const SUBMISSION_CHANNEL_CAPACITY: usize = 64;

// Model-formatting limits: clients get full streams; oonly content sent to the model is truncated.
pub(crate) const MODEL_FORMAT_MAX_BYTES: usize = 10 * 1024; // 10 KiB
pub(crate) const MODEL_FORMAT_MAX_LINES: usize = 256; // lines
pub(crate) const MODEL_FORMAT_HEAD_LINES: usize = MODEL_FORMAT_MAX_LINES / 2;
pub(crate) const MODEL_FORMAT_TAIL_LINES: usize = MODEL_FORMAT_MAX_LINES - MODEL_FORMAT_HEAD_LINES; // 128
pub(crate) const MODEL_FORMAT_HEAD_BYTES: usize = MODEL_FORMAT_MAX_BYTES / 2;

impl Codex {
    /// Spawn a new [`Codex`] and initialize the session.
    pub async fn spawn(
        config: Config,
        auth_manager: Arc<AuthManager>,
        conversation_history: InitialHistory,
    ) -> CodexResult<CodexSpawnOk> {
        let (tx_sub, rx_sub) = async_channel::bounded(SUBMISSION_CHANNEL_CAPACITY);
        let (tx_event, rx_event) = async_channel::unbounded();

        let user_instructions = get_user_instructions(&config).await;

        let config = Arc::new(config);

        let configure_session = ConfigureSession {
            provider: config.model_provider.clone(),
            model: config.model.clone(),
            model_reasoning_effort: config.model_reasoning_effort,
            model_reasoning_summary: config.model_reasoning_summary,
            user_instructions,
            base_instructions: config.base_instructions.clone(),
            approval_policy: config.approval_policy,
            sandbox_policy: config.sandbox_policy.clone(),
            notify: config.notify.clone(),
            cwd: config.cwd.clone(),
        };

        // Generate a unique ID for the lifetime of this Codex session.
        let (session, turn_context) = Session::new(
            configure_session,
            config.clone(),
            auth_manager.clone(),
            tx_event.clone(),
            conversation_history,
        )
        .await
        .map_err(|e| {
            error!("Failed to create session: {e:#}");
            CodexErr::InternalAgentDied
        })?;
        let conversation_id = session.conversation_id;

        // This task will run until Op::Shutdown is received.
        tokio::spawn(submission_loop(session, turn_context, config, rx_sub));
        let codex = Codex {
            next_id: AtomicU64::new(0),
            tx_sub,
            rx_event,
        };

        Ok(CodexSpawnOk {
            codex,
            conversation_id,
        })
    }

    /// Submit the `op` wrapped in a `Submission` with a unique ID.
    pub async fn submit(&self, op: Op) -> CodexResult<String> {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            .to_string();
        let sub = Submission { id: id.clone(), op };
        self.submit_with_id(sub).await?;
        Ok(id)
    }

    /// Use sparingly: prefer `submit()` so Codex is responsible for generating
    /// unique IDs for each submission.
    pub async fn submit_with_id(&self, sub: Submission) -> CodexResult<()> {
        self.tx_sub
            .send(sub)
            .await
            .map_err(|_| CodexErr::InternalAgentDied)?;
        Ok(())
    }

    pub async fn next_event(&self) -> CodexResult<Event> {
        let event = self
            .rx_event
            .recv()
            .await
            .map_err(|_| CodexErr::InternalAgentDied)?;
        Ok(event)
    }
}

/// Mutable state of the agent
#[derive(Default)]
struct State {
    approved_commands: HashSet<Vec<String>>,
    current_task: Option<AgentTask>,
    pending_approvals: HashMap<String, oneshot::Sender<ReviewDecision>>,
    pending_input: Vec<ResponseInputItem>,
    history: ConversationHistory,
    token_info: Option<TokenUsageInfo>,
    /// Epoch identity of the working set selected for the most recent user
    /// request, used for cache-aware cost comparison (never a token budget).
    last_working_set_epoch: Option<String>,
    /// 1-based ordinal of the most recent user request whose working set was
    /// frozen. Recorded in the selection sidecar; never a model input.
    working_set_request_index: u64,
    next_internal_sub_id: u64,
}

/// Context for an initialized model agent
///
/// A session has at most 1 running task at a time, and can be interrupted by user input.
pub(crate) struct Session {
    conversation_id: ConversationId,
    tx_event: Sender<Event>,

    /// Manager for external MCP servers/tools.
    mcp_connection_manager: McpConnectionManager,
    session_manager: ExecSessionManager,
    unified_exec_manager: UnifiedExecSessionManager,

    /// External notifier command (will be passed as args to exec()). When
    /// `None` this feature is disabled.
    notify: Option<Vec<String>>,

    /// Optional rollout recorder for persisting the conversation transcript so
    /// sessions can be replayed or inspected later.
    rollout: Mutex<Option<RolloutRecorder>>,
    /// Synthetic pricing catalog injected by the internal test constructor only.
    /// Production rates are resolved per request through
    /// [`resolved_pricing_catalog`]; this field does not exist in a production
    /// build, so synthetic rates cannot enter one.
    #[cfg(test)]
    working_set_test_catalog: Option<working_set::PricingCatalog>,
    state: Mutex<State>,
    codex_linux_sandbox_exe: Option<PathBuf>,
    user_shell: shell::Shell,
    show_raw_agent_reasoning: bool,
}

/// The context needed for a single turn of the conversation.
#[derive(Debug)]
pub(crate) struct TurnContext {
    pub(crate) client: ModelClient,
    /// The session's current working directory. All relative paths provided by
    /// the model as well as sandbox policies are resolved against this path
    /// instead of `std::env::current_dir()`.
    pub(crate) cwd: PathBuf,
    pub(crate) base_instructions: Option<String>,
    pub(crate) user_instructions: Option<String>,
    pub(crate) approval_policy: AskForApproval,
    pub(crate) sandbox_policy: SandboxPolicy,
    pub(crate) shell_environment_policy: ShellEnvironmentPolicy,
    pub(crate) tools_config: ToolsConfig,
    pub(crate) is_review_mode: bool,
}

impl TurnContext {
    fn resolve_path(&self, path: Option<String>) -> PathBuf {
        path.as_ref()
            .map(PathBuf::from)
            .map_or_else(|| self.cwd.clone(), |p| self.cwd.join(p))
    }
}

/// Configure the model session.
struct ConfigureSession {
    /// Provider identifier ("openai", "openrouter", ...).
    provider: ModelProviderInfo,

    /// If not specified, server will use its default model.
    model: String,

    model_reasoning_effort: Option<ReasoningEffortConfig>,
    model_reasoning_summary: ReasoningSummaryConfig,

    /// Model instructions that are appended to the base instructions.
    user_instructions: Option<String>,

    /// Base instructions override.
    base_instructions: Option<String>,

    /// When to escalate for approval for execution
    approval_policy: AskForApproval,
    /// How to sandbox commands executed in the system
    sandbox_policy: SandboxPolicy,

    /// Optional external notifier command tokens. Present only when the
    /// client wants the agent to spawn a program after each completed
    /// turn.
    notify: Option<Vec<String>>,

    /// Working directory that should be treated as the *root* of the
    /// session. All relative paths supplied by the model as well as the
    /// execution sandbox are resolved against this directory **instead**
    /// of the process-wide current working directory. CLI front-ends are
    /// expected to expand this to an absolute path before sending the
    /// `ConfigureSession` operation so that the business-logic layer can
    /// operate deterministically.
    cwd: PathBuf,
}

impl Session {
    async fn new(
        configure_session: ConfigureSession,
        config: Arc<Config>,
        auth_manager: Arc<AuthManager>,
        tx_event: Sender<Event>,
        initial_history: InitialHistory,
    ) -> anyhow::Result<(Arc<Self>, TurnContext)> {
        let ConfigureSession {
            provider,
            model,
            model_reasoning_effort,
            model_reasoning_summary,
            user_instructions,
            base_instructions,
            approval_policy,
            sandbox_policy,
            notify,
            cwd,
        } = configure_session;
        debug!("Configuring session: model={model}; provider={provider:?}");
        if !cwd.is_absolute() {
            return Err(anyhow::anyhow!("cwd is not absolute: {cwd:?}"));
        }

        let (conversation_id, rollout_params) = match &initial_history {
            InitialHistory::New | InitialHistory::Forked(_) => {
                let conversation_id = ConversationId::default();
                (
                    conversation_id,
                    RolloutRecorderParams::new(conversation_id, user_instructions.clone()),
                )
            }
            InitialHistory::Resumed(resumed_history) => (
                resumed_history.conversation_id,
                RolloutRecorderParams::resume(resumed_history.rollout_path.clone()),
            ),
        };

        // Error messages to dispatch after SessionConfigured is sent.
        let mut post_session_configured_error_events = Vec::<Event>::new();

        // Kick off independent async setup tasks in parallel to reduce startup latency.
        //
        // - initialize RolloutRecorder with new or resumed session info
        // - spin up MCP connection manager
        // - perform default shell discovery
        // - load history metadata
        let rollout_fut = RolloutRecorder::new(&config, rollout_params);

        let mcp_fut = McpConnectionManager::new(config.mcp_servers.clone());
        let default_shell_fut = shell::default_user_shell();
        let history_meta_fut = crate::message_history::history_metadata(&config);

        // Join all independent futures.
        let (rollout_recorder, mcp_res, default_shell, (history_log_id, history_entry_count)) =
            tokio::join!(rollout_fut, mcp_fut, default_shell_fut, history_meta_fut);

        let rollout_recorder = rollout_recorder.map_err(|e| {
            error!("failed to initialize rollout recorder: {e:#}");
            anyhow::anyhow!("failed to initialize rollout recorder: {e:#}")
        })?;
        let rollout_path = rollout_recorder.rollout_path.clone();
        // Create the mutable state for the Session.
        let state = State {
            history: ConversationHistory::new(),
            ..Default::default()
        };

        // Handle MCP manager result and record any startup failures.
        let (mcp_connection_manager, failed_clients) = match mcp_res {
            Ok((mgr, failures)) => (mgr, failures),
            Err(e) => {
                let message = format!("Failed to create MCP connection manager: {e:#}");
                error!("{message}");
                post_session_configured_error_events.push(Event {
                    id: INITIAL_SUBMIT_ID.to_owned(),
                    msg: EventMsg::Error(ErrorEvent { message }),
                });
                (McpConnectionManager::default(), Default::default())
            }
        };

        // Surface individual client start-up failures to the user.
        if !failed_clients.is_empty() {
            for (server_name, err) in failed_clients {
                let message = format!("MCP client for `{server_name}` failed to start: {err:#}");
                error!("{message}");
                post_session_configured_error_events.push(Event {
                    id: INITIAL_SUBMIT_ID.to_owned(),
                    msg: EventMsg::Error(ErrorEvent { message }),
                });
            }
        }

        // Now that the conversation id is final (may have been updated by resume),
        // construct the model client.
        let client = ModelClient::new(
            config.clone(),
            Some(auth_manager.clone()),
            provider.clone(),
            model_reasoning_effort,
            model_reasoning_summary,
            conversation_id,
        );
        let turn_context = TurnContext {
            client,
            tools_config: ToolsConfig::new(&ToolsConfigParams {
                model_family: &config.model_family,
                approval_policy,
                sandbox_policy: sandbox_policy.clone(),
                include_plan_tool: config.include_plan_tool,
                include_apply_patch_tool: config.include_apply_patch_tool,
                include_web_search_request: config.tools_web_search_request,
                use_streamable_shell_tool: config.use_experimental_streamable_shell_tool,
                include_view_image_tool: config.include_view_image_tool,
                experimental_unified_exec_tool: config.use_experimental_unified_exec_tool,
            }),
            user_instructions,
            base_instructions,
            approval_policy,
            sandbox_policy,
            shell_environment_policy: config.shell_environment_policy.clone(),
            cwd,
            is_review_mode: false,
        };
        let sess = Arc::new(Session {
            conversation_id,
            tx_event: tx_event.clone(),
            mcp_connection_manager,
            session_manager: ExecSessionManager::default(),
            unified_exec_manager: UnifiedExecSessionManager::default(),
            notify,
            state: Mutex::new(state),
            rollout: Mutex::new(Some(rollout_recorder)),
            #[cfg(test)]
            working_set_test_catalog: None,
            codex_linux_sandbox_exe: config.codex_linux_sandbox_exe.clone(),
            user_shell: default_shell,
            show_raw_agent_reasoning: config.show_raw_agent_reasoning,
        });

        // Dispatch the SessionConfiguredEvent first and then report any errors.
        // If resuming, include converted initial messages in the payload so UIs can render them immediately.
        let initial_messages = initial_history.get_event_msgs();
        sess.record_initial_history(&turn_context, initial_history)
            .await;

        let events = std::iter::once(Event {
            id: INITIAL_SUBMIT_ID.to_owned(),
            msg: EventMsg::SessionConfigured(SessionConfiguredEvent {
                session_id: conversation_id,
                model,
                reasoning_effort: model_reasoning_effort,
                history_log_id,
                history_entry_count,
                initial_messages,
                rollout_path,
            }),
        })
        .chain(post_session_configured_error_events.into_iter());
        for event in events {
            sess.send_event(event).await;
        }

        Ok((sess, turn_context))
    }

    pub fn set_task(&self, task: AgentTask) {
        let mut state = self.state.lock_unchecked();
        if let Some(current_task) = state.current_task.take() {
            current_task.abort(TurnAbortReason::Replaced);
        }
        state.current_task = Some(task);
    }

    pub fn remove_task(&self, sub_id: &str) {
        let mut state = self.state.lock_unchecked();
        if let Some(task) = &state.current_task
            && task.sub_id == sub_id
        {
            state.current_task.take();
        }
    }

    fn next_internal_sub_id(&self) -> String {
        let mut state = self.state.lock_unchecked();
        let id = state.next_internal_sub_id;
        state.next_internal_sub_id += 1;
        format!("auto-compact-{id}")
    }

    async fn record_initial_history(
        &self,
        turn_context: &TurnContext,
        conversation_history: InitialHistory,
    ) {
        match conversation_history {
            InitialHistory::New => {
                // Build and record initial items (user instructions + environment context)
                let items = self.build_initial_context(turn_context);
                self.record_conversation_items(&items).await;
            }
            InitialHistory::Resumed(_) | InitialHistory::Forked(_) => {
                let rollout_items = conversation_history.get_rollout_items();
                let persist = matches!(conversation_history, InitialHistory::Forked(_));

                // Always add response items to conversation history
                let reconstructed_history =
                    self.reconstruct_history_from_rollout(turn_context, &rollout_items);
                if !reconstructed_history.is_empty() {
                    self.record_into_history(&reconstructed_history);
                }
                // Cache-aware accounting needs the epoch the previous request
                // actually sent; it lives in the sidecar, not the rollout.
                self.restore_working_set_epoch().await;

                // If persisting, persist all rollout items as-is (recorder filters)
                if persist && !rollout_items.is_empty() {
                    self.persist_rollout_items(&rollout_items).await;
                }
            }
        }
    }

    /// Persist the event to rollout and send it to clients.
    pub(crate) async fn send_event(&self, event: Event) {
        // Persist the event into rollout (recorder filters as needed)
        let rollout_items = vec![RolloutItem::EventMsg(event.msg.clone())];
        self.persist_rollout_items(&rollout_items).await;
        if let Err(e) = self.tx_event.send(event).await {
            error!("failed to send tool call event: {e}");
        }
    }

    pub async fn request_command_approval(
        &self,
        sub_id: String,
        call_id: String,
        command: Vec<String>,
        cwd: PathBuf,
        reason: Option<String>,
    ) -> oneshot::Receiver<ReviewDecision> {
        // Add the tx_approve callback to the map before sending the request.
        let (tx_approve, rx_approve) = oneshot::channel();
        let event_id = sub_id.clone();
        let prev_entry = {
            let mut state = self.state.lock_unchecked();
            state.pending_approvals.insert(sub_id, tx_approve)
        };
        if prev_entry.is_some() {
            warn!("Overwriting existing pending approval for sub_id: {event_id}");
        }

        let event = Event {
            id: event_id,
            msg: EventMsg::ExecApprovalRequest(ExecApprovalRequestEvent {
                call_id,
                command,
                cwd,
                reason,
            }),
        };
        self.send_event(event).await;
        rx_approve
    }

    pub async fn request_patch_approval(
        &self,
        sub_id: String,
        call_id: String,
        action: &ApplyPatchAction,
        reason: Option<String>,
        grant_root: Option<PathBuf>,
    ) -> oneshot::Receiver<ReviewDecision> {
        // Add the tx_approve callback to the map before sending the request.
        let (tx_approve, rx_approve) = oneshot::channel();
        let event_id = sub_id.clone();
        let prev_entry = {
            let mut state = self.state.lock_unchecked();
            state.pending_approvals.insert(sub_id, tx_approve)
        };
        if prev_entry.is_some() {
            warn!("Overwriting existing pending approval for sub_id: {event_id}");
        }

        let event = Event {
            id: event_id,
            msg: EventMsg::ApplyPatchApprovalRequest(ApplyPatchApprovalRequestEvent {
                call_id,
                changes: convert_apply_patch_to_protocol(action),
                reason,
                grant_root,
            }),
        };
        self.send_event(event).await;
        rx_approve
    }

    pub fn notify_approval(&self, sub_id: &str, decision: ReviewDecision) {
        let entry = {
            let mut state = self.state.lock_unchecked();
            state.pending_approvals.remove(sub_id)
        };
        match entry {
            Some(tx_approve) => {
                tx_approve.send(decision).ok();
            }
            None => {
                warn!("No pending approval found for sub_id: {sub_id}");
            }
        }
    }

    pub fn add_approved_command(&self, cmd: Vec<String>) {
        let mut state = self.state.lock_unchecked();
        state.approved_commands.insert(cmd);
    }

    /// Records input items: always append to conversation history and
    /// persist these response items to rollout.
    async fn record_conversation_items(&self, items: &[ResponseItem]) {
        self.record_into_history(items);
        self.persist_rollout_response_items(items).await;
    }

    fn reconstruct_history_from_rollout(
        &self,
        turn_context: &TurnContext,
        rollout_items: &[RolloutItem],
    ) -> Vec<ResponseItem> {
        let mut history = ConversationHistory::new();
        // Compact-on-demand: the newest persisted token-usage record is restored
        // into the session so the working-set selector has usage evidence after a
        // process boundary. Without it `expected_output_tokens` is unknown, cost
        // confidence stays `low`, and selection always falls back to canonical
        // history. The record is already written by `should_persist_event_msg`
        // and reloaded by the recorder; this only reads it back.
        let mut restored_usage: Option<TokenUsageInfo> = None;
        for item in rollout_items {
            match item {
                RolloutItem::ResponseItem(response_item) => {
                    history.record_items(std::iter::once(response_item));
                }
                RolloutItem::Compacted(compacted) => {
                    let snapshot = history.contents();
                    let user_messages = collect_user_messages(&snapshot);
                    let rebuilt = build_compacted_history(
                        self.build_initial_context(turn_context),
                        &user_messages,
                        &compacted.message,
                    );
                    history.replace(rebuilt);
                }
                RolloutItem::EventMsg(EventMsg::TokenCount(event)) => {
                    // Last record wins: usage is cumulative, so the newest entry
                    // describes the session as it stands now.
                    if let Some(info) = event.info.as_ref() {
                        restored_usage = Some(info.clone());
                    }
                }
                _ => {}
            }
        }
        if let Some(info) = restored_usage {
            let mut state = self.state.lock_unchecked();
            if state.token_info.is_none() {
                state.token_info = Some(info);
            }
        }
        history.contents()
    }

    /// Append ResponseItems to the in-memory conversation history only.
    fn record_into_history(&self, items: &[ResponseItem]) {
        self.state
            .lock_unchecked()
            .history
            .record_items(items.iter());
    }

    async fn persist_rollout_response_items(&self, items: &[ResponseItem]) {
        let rollout_items: Vec<RolloutItem> = items
            .iter()
            .cloned()
            .map(RolloutItem::ResponseItem)
            .collect();
        self.persist_rollout_items(&rollout_items).await;
    }

    fn build_initial_context(&self, turn_context: &TurnContext) -> Vec<ResponseItem> {
        let mut items = Vec::<ResponseItem>::with_capacity(2);
        if let Some(user_instructions) = turn_context.user_instructions.as_deref() {
            items.push(UserInstructions::new(user_instructions.to_string()).into());
        }
        items.push(ResponseItem::from(EnvironmentContext::new(
            Some(turn_context.cwd.clone()),
            Some(turn_context.approval_policy),
            Some(turn_context.sandbox_policy.clone()),
            Some(self.user_shell.clone()),
        )));
        items
    }

    async fn persist_rollout_items(&self, items: &[RolloutItem]) {
        let recorder = {
            let guard = self.rollout.lock_unchecked();
            guard.as_ref().cloned()
        };
        if let Some(rec) = recorder
            && let Err(e) = rec.record_items(items).await
        {
            error!("failed to record rollout items: {e:#}");
        }
    }

    fn update_token_usage_info(
        &self,
        turn_context: &TurnContext,
        token_usage: &Option<TokenUsage>,
    ) -> Option<TokenUsageInfo> {
        let mut state = self.state.lock_unchecked();
        let info = TokenUsageInfo::new_or_append(
            &state.token_info,
            token_usage,
            turn_context.client.get_model_context_window(),
        );
        state.token_info = info.clone();
        info
    }

    /// Record a user input item to conversation history and also persist a
    /// corresponding UserMessage EventMsg to rollout.
    async fn record_input_and_rollout_usermsg(&self, response_input: &ResponseInputItem) {
        let response_item: ResponseItem = response_input.clone().into();
        // Add to conversation history and persist response item to rollout
        self.record_conversation_items(std::slice::from_ref(&response_item))
            .await;

        // Derive user message events and persist only UserMessage to rollout
        let msgs =
            map_response_item_to_event_messages(&response_item, self.show_raw_agent_reasoning);
        let user_msgs: Vec<RolloutItem> = msgs
            .into_iter()
            .filter_map(|m| match m {
                EventMsg::UserMessage(ev) => Some(RolloutItem::EventMsg(EventMsg::UserMessage(ev))),
                _ => None,
            })
            .collect();
        if !user_msgs.is_empty() {
            self.persist_rollout_items(&user_msgs).await;
        }
    }

    async fn on_exec_command_begin(
        &self,
        turn_diff_tracker: &mut TurnDiffTracker,
        exec_command_context: ExecCommandContext,
    ) {
        let ExecCommandContext {
            sub_id,
            call_id,
            command_for_display,
            cwd,
            apply_patch,
        } = exec_command_context;
        let msg = match apply_patch {
            Some(ApplyPatchCommandContext {
                user_explicitly_approved_this_action,
                changes,
            }) => {
                turn_diff_tracker.on_patch_begin(&changes);

                EventMsg::PatchApplyBegin(PatchApplyBeginEvent {
                    call_id,
                    auto_approved: !user_explicitly_approved_this_action,
                    changes,
                })
            }
            None => EventMsg::ExecCommandBegin(ExecCommandBeginEvent {
                call_id,
                command: command_for_display.clone(),
                cwd,
                parsed_cmd: parse_command(&command_for_display)
                    .into_iter()
                    .map(Into::into)
                    .collect(),
            }),
        };
        let event = Event {
            id: sub_id.to_string(),
            msg,
        };
        self.send_event(event).await;
    }

    async fn on_exec_command_end(
        &self,
        turn_diff_tracker: &mut TurnDiffTracker,
        sub_id: &str,
        call_id: &str,
        output: &ExecToolCallOutput,
        is_apply_patch: bool,
    ) {
        let ExecToolCallOutput {
            stdout,
            stderr,
            aggregated_output,
            duration,
            exit_code,
            timed_out: _,
        } = output;
        // Send full stdout/stderr to clients; do not truncate.
        let stdout = stdout.text.clone();
        let stderr = stderr.text.clone();
        let formatted_output = format_exec_output_str(output);
        let aggregated_output: String = aggregated_output.text.clone();

        let msg = if is_apply_patch {
            EventMsg::PatchApplyEnd(PatchApplyEndEvent {
                call_id: call_id.to_string(),
                stdout,
                stderr,
                success: *exit_code == 0,
            })
        } else {
            EventMsg::ExecCommandEnd(ExecCommandEndEvent {
                call_id: call_id.to_string(),
                stdout,
                stderr,
                aggregated_output,
                exit_code: *exit_code,
                duration: *duration,
                formatted_output,
            })
        };

        let event = Event {
            id: sub_id.to_string(),
            msg,
        };
        self.send_event(event).await;

        // If this is an apply_patch, after we emit the end patch, emit a second event
        // with the full turn diff if there is one.
        if is_apply_patch {
            let unified_diff = turn_diff_tracker.get_unified_diff();
            if let Ok(Some(unified_diff)) = unified_diff {
                let msg = EventMsg::TurnDiff(TurnDiffEvent { unified_diff });
                let event = Event {
                    id: sub_id.into(),
                    msg,
                };
                self.send_event(event).await;
            }
        }
    }
    /// Runs the exec tool call and emits events for the begin and end of the
    /// command even on error.
    ///
    /// Returns the output of the exec tool call.
    async fn run_exec_with_events<'a>(
        &self,
        turn_diff_tracker: &mut TurnDiffTracker,
        begin_ctx: ExecCommandContext,
        exec_args: ExecInvokeArgs<'a>,
    ) -> crate::error::Result<ExecToolCallOutput> {
        let is_apply_patch = begin_ctx.apply_patch.is_some();
        let sub_id = begin_ctx.sub_id.clone();
        let call_id = begin_ctx.call_id.clone();

        self.on_exec_command_begin(turn_diff_tracker, begin_ctx.clone())
            .await;

        let result = process_exec_tool_call(
            exec_args.params,
            exec_args.sandbox_type,
            exec_args.sandbox_policy,
            exec_args.codex_linux_sandbox_exe,
            exec_args.stdout_stream,
        )
        .await;

        let output_stderr;
        let borrowed: &ExecToolCallOutput = match &result {
            Ok(output) => output,
            Err(CodexErr::Sandbox(SandboxErr::Timeout { output })) => output,
            Err(e) => {
                output_stderr = ExecToolCallOutput {
                    exit_code: -1,
                    stdout: StreamOutput::new(String::new()),
                    stderr: StreamOutput::new(get_error_message_ui(e)),
                    aggregated_output: StreamOutput::new(get_error_message_ui(e)),
                    duration: Duration::default(),
                    timed_out: false,
                };
                &output_stderr
            }
        };
        self.on_exec_command_end(
            turn_diff_tracker,
            &sub_id,
            &call_id,
            borrowed,
            is_apply_patch,
        )
        .await;

        result
    }

    /// Helper that emits a BackgroundEvent with the given message. This keeps
    /// the call‑sites terse so adding more diagnostics does not clutter the
    /// core agent logic.
    async fn notify_background_event(&self, sub_id: &str, message: impl Into<String>) {
        let event = Event {
            id: sub_id.to_string(),
            msg: EventMsg::BackgroundEvent(BackgroundEventEvent {
                message: message.into(),
            }),
        };
        self.send_event(event).await;
    }

    async fn notify_stream_error(&self, sub_id: &str, message: impl Into<String>) {
        let event = Event {
            id: sub_id.to_string(),
            msg: EventMsg::StreamError(StreamErrorEvent {
                message: message.into(),
            }),
        };
        self.send_event(event).await;
    }

    /// Build the full turn input by concatenating the current conversation
    /// history with additional items for this turn.
    pub fn turn_input_with_history(&self, extra: Vec<ResponseItem>) -> Vec<ResponseItem> {
        [self.state.lock_unchecked().history.contents(), extra].concat()
    }

    /// Provider-reported usage of the most recent response, when any. Recorded
    /// data only: used as cost accounting input, never as a selection predicate.
    fn last_token_usage(&self) -> Option<TokenUsage> {
        self.state
            .lock_unchecked()
            .token_info
            .as_ref()
            .map(|info| info.last_token_usage.clone())
    }

    /// Epoch identity of the working set sent for the most recent user request.
    fn last_working_set_epoch(&self) -> Option<String> {
        self.state.lock_unchecked().last_working_set_epoch.clone()
    }

    fn set_last_working_set_epoch(&self, epoch_id: String) {
        self.state.lock_unchecked().last_working_set_epoch = Some(epoch_id);
    }

    /// Advances and returns the 1-based ordinal of a user request whose working
    /// set is being frozen. Used only for sidecar records, never as model input.
    fn next_working_set_request_index(&self) -> u64 {
        let mut state = self.state.lock_unchecked();
        state.working_set_request_index += 1;
        state.working_set_request_index
    }

    /// Recorded pricing catalog for the exact wire model of a request, when one
    /// exists.
    ///
    /// Production rate data is resolved per request through
    /// [`resolved_pricing_catalog`], which also validates the recorded snapshot
    /// against the fixed reference TTL; a synthetic catalog can only be injected
    /// by the internal test constructor and does not exist in a production build.
    fn working_set_pricing_catalog(&self, model: &str) -> Option<working_set::PricingCatalog> {
        if let Some(catalog) = resolved_pricing_catalog(model, Utc::now()) {
            return Some(catalog);
        }
        #[cfg(test)]
        {
            if let Some(catalog) = self
                .working_set_test_catalog
                .as_ref()
                .filter(|catalog| catalog.model == model)
            {
                return Some(catalog.clone());
            }
        }
        None
    }

    /// Appends one structural record to the append-only selection sidecar next to
    /// the session rollout (see [`working_set::selection_sidecar_path`]).
    ///
    /// Persistence is observational: a failure is logged and never changes the
    /// model input, the canonical history or the task outcome.
    async fn record_working_set_sidecar(&self, record: &working_set::SidecarRecord) {
        let sidecar_path = {
            let guard = self.rollout.lock_unchecked();
            guard
                .as_ref()
                .map(|recorder| working_set::selection_sidecar_path(&recorder.get_rollout_path()))
        };
        let Some(path) = sidecar_path else {
            debug!("working-set sidecar skipped: rollout recorder is disabled");
            return;
        };
        if let Err(err) = append_sidecar_record(&path, record).await {
            warn!(
                path = %path.display(),
                "failed to append working-set selection record: {err:#}"
            );
        }
    }

    /// Restores the previous working-set epoch identity from the append-only
    /// selection sidecar so a resumed process keeps cache-aware cost accounting.
    ///
    /// Without it `previous_epoch_id` is `None` after a process boundary, the
    /// already-warm baseline prefix is mispriced as a fresh cache write, and the
    /// selector can wrongly prefer a projection over a cheaper cached baseline.
    /// The newest `turn.epoch_id` (falling back to the newest `selection`
    /// `manifest.epoch.epoch_id`) is the epoch the previous request actually
    /// sent, so it is the correct comparison point.
    ///
    /// Best effort: a missing, unreadable or malformed sidecar leaves the
    /// conservative default in place and never changes the task outcome.
    async fn restore_working_set_epoch(&self) {
        let sidecar_path = {
            let guard = self.rollout.lock_unchecked();
            guard
                .as_ref()
                .map(|recorder| working_set::selection_sidecar_path(&recorder.get_rollout_path()))
        };
        let Some(path) = sidecar_path else {
            return;
        };
        let Ok(contents) = tokio::fs::read_to_string(&path).await else {
            return;
        };
        let mut restored: Option<String> = None;
        for line in contents.lines() {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let candidate = match value.get("record").and_then(|record| record.as_str()) {
                Some("turn") => value.get("epoch_id").and_then(|id| id.as_str()),
                Some("selection") => value
                    .get("manifest")
                    .and_then(|manifest| manifest.get("epoch"))
                    .and_then(|epoch| epoch.get("epoch_id"))
                    .and_then(|id| id.as_str()),
                _ => None,
            };
            // A file is append-only, so the last usable record wins.
            if let Some(id) = candidate {
                restored = Some(id.to_string());
            }
        }
        if let Some(epoch_id) = restored {
            let mut state = self.state.lock_unchecked();
            if state.last_working_set_epoch.is_none() {
                state.last_working_set_epoch = Some(epoch_id);
            }
        }
    }

    /// Returns the input if there was no task running to inject into
    pub fn inject_input(&self, input: Vec<InputItem>) -> Result<(), Vec<InputItem>> {
        let mut state = self.state.lock_unchecked();
        if state.current_task.is_some() {
            state.pending_input.push(input.into());
            Ok(())
        } else {
            Err(input)
        }
    }

    pub fn get_pending_input(&self) -> Vec<ResponseInputItem> {
        let mut state = self.state.lock_unchecked();
        if state.pending_input.is_empty() {
            Vec::with_capacity(0)
        } else {
            let mut ret = Vec::new();
            std::mem::swap(&mut ret, &mut state.pending_input);
            ret
        }
    }

    pub async fn call_tool(
        &self,
        server: &str,
        tool: &str,
        arguments: Option<serde_json::Value>,
        timeout: Option<Duration>,
    ) -> anyhow::Result<CallToolResult> {
        self.mcp_connection_manager
            .call_tool(server, tool, arguments, timeout)
            .await
    }

    fn interrupt_task(&self) {
        info!("interrupt received: abort current task, if any");
        let mut state = self.state.lock_unchecked();
        state.pending_approvals.clear();
        state.pending_input.clear();
        if let Some(task) = state.current_task.take() {
            task.abort(TurnAbortReason::Interrupted);
        }
    }

    /// Spawn the configured notifier (if any) with the given JSON payload as
    /// the last argument. Failures are logged but otherwise ignored so that
    /// notification issues do not interfere with the main workflow.
    fn maybe_notify(&self, notification: UserNotification) {
        let Some(notify_command) = &self.notify else {
            return;
        };

        if notify_command.is_empty() {
            return;
        }

        let Ok(json) = serde_json::to_string(&notification) else {
            error!("failed to serialise notification payload");
            return;
        };

        let mut command = std::process::Command::new(&notify_command[0]);
        if notify_command.len() > 1 {
            command.args(&notify_command[1..]);
        }
        command.arg(json);

        // Fire-and-forget – we do not wait for completion.
        if let Err(e) = command.spawn() {
            warn!("failed to spawn notifier '{}': {e}", notify_command[0]);
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.interrupt_task();
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ExecCommandContext {
    pub(crate) sub_id: String,
    pub(crate) call_id: String,
    pub(crate) command_for_display: Vec<String>,
    pub(crate) cwd: PathBuf,
    pub(crate) apply_patch: Option<ApplyPatchCommandContext>,
}

#[derive(Clone, Debug)]
pub(crate) struct ApplyPatchCommandContext {
    pub(crate) user_explicitly_approved_this_action: bool,
    pub(crate) changes: HashMap<PathBuf, FileChange>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum AgentTaskKind {
    Regular,
    Review,
    Compact,
}

/// A series of Turns in response to user input.
pub(crate) struct AgentTask {
    sess: Arc<Session>,
    sub_id: String,
    handle: AbortHandle,
    kind: AgentTaskKind,
}

impl AgentTask {
    fn spawn(
        sess: Arc<Session>,
        turn_context: Arc<TurnContext>,
        sub_id: String,
        input: Vec<InputItem>,
    ) -> Self {
        let handle = {
            let sess = sess.clone();
            let sub_id = sub_id.clone();
            let tc = Arc::clone(&turn_context);
            tokio::spawn(async move { run_task(sess, tc, sub_id, input).await }).abort_handle()
        };
        Self {
            sess,
            sub_id,
            handle,
            kind: AgentTaskKind::Regular,
        }
    }

    fn review(
        sess: Arc<Session>,
        turn_context: Arc<TurnContext>,
        sub_id: String,
        input: Vec<InputItem>,
    ) -> Self {
        let handle = {
            let sess = sess.clone();
            let sub_id = sub_id.clone();
            let tc = Arc::clone(&turn_context);
            tokio::spawn(async move { run_task(sess, tc, sub_id, input).await }).abort_handle()
        };
        Self {
            sess,
            sub_id,
            handle,
            kind: AgentTaskKind::Review,
        }
    }

    fn compact(
        sess: Arc<Session>,
        turn_context: Arc<TurnContext>,
        sub_id: String,
        input: Vec<InputItem>,
        compact_instructions: String,
    ) -> Self {
        let handle = {
            let sess = sess.clone();
            let sub_id = sub_id.clone();
            let tc = Arc::clone(&turn_context);
            tokio::spawn(async move {
                compact::run_compact_task(sess, tc, sub_id, input, compact_instructions).await
            })
            .abort_handle()
        };
        Self {
            sess,
            sub_id,
            handle,
            kind: AgentTaskKind::Compact,
        }
    }

    fn abort(self, reason: TurnAbortReason) {
        // TOCTOU?
        if !self.handle.is_finished() {
            if self.kind == AgentTaskKind::Review {
                let sess = self.sess.clone();
                let sub_id = self.sub_id.clone();
                tokio::spawn(async move {
                    exit_review_mode(sess, sub_id, None).await;
                });
            }
            self.handle.abort();
            let event = Event {
                id: self.sub_id,
                msg: EventMsg::TurnAborted(TurnAbortedEvent { reason }),
            };
            let sess = self.sess;
            tokio::spawn(async move {
                sess.send_event(event).await;
            });
        }
    }
}

async fn submission_loop(
    sess: Arc<Session>,
    turn_context: TurnContext,
    config: Arc<Config>,
    rx_sub: Receiver<Submission>,
) {
    // Wrap once to avoid cloning TurnContext for each task.
    let mut turn_context = Arc::new(turn_context);
    // To break out of this loop, send Op::Shutdown.
    while let Ok(sub) = rx_sub.recv().await {
        debug!(?sub, "Submission");
        match sub.op {
            Op::Interrupt => {
                sess.interrupt_task();
            }
            Op::OverrideTurnContext {
                cwd,
                approval_policy,
                sandbox_policy,
                model,
                effort,
                summary,
            } => {
                // Recalculate the persistent turn context with provided overrides.
                let prev = Arc::clone(&turn_context);
                let provider = prev.client.get_provider();

                // Effective model + family
                let (effective_model, effective_family) = if let Some(ref m) = model {
                    let fam =
                        find_family_for_model(m).unwrap_or_else(|| config.model_family.clone());
                    (m.clone(), fam)
                } else {
                    (prev.client.get_model(), prev.client.get_model_family())
                };

                // Effective reasoning settings
                let effective_effort = effort.unwrap_or(prev.client.get_reasoning_effort());
                let effective_summary = summary.unwrap_or(prev.client.get_reasoning_summary());

                let auth_manager = prev.client.get_auth_manager();

                // Build updated config for the client
                let mut updated_config = (*config).clone();
                updated_config.model = effective_model.clone();
                updated_config.model_family = effective_family.clone();
                if let Some(model_info) = get_model_info(&effective_family) {
                    updated_config.model_context_window = Some(model_info.context_window);
                }

                let client = ModelClient::new(
                    Arc::new(updated_config),
                    auth_manager,
                    provider,
                    effective_effort,
                    effective_summary,
                    sess.conversation_id,
                );

                let new_approval_policy = approval_policy.unwrap_or(prev.approval_policy);
                let new_sandbox_policy = sandbox_policy
                    .clone()
                    .unwrap_or(prev.sandbox_policy.clone());
                let new_cwd = cwd.clone().unwrap_or_else(|| prev.cwd.clone());

                let tools_config = ToolsConfig::new(&ToolsConfigParams {
                    model_family: &effective_family,
                    approval_policy: new_approval_policy,
                    sandbox_policy: new_sandbox_policy.clone(),
                    include_plan_tool: config.include_plan_tool,
                    include_apply_patch_tool: config.include_apply_patch_tool,
                    include_web_search_request: config.tools_web_search_request,
                    use_streamable_shell_tool: config.use_experimental_streamable_shell_tool,
                    include_view_image_tool: config.include_view_image_tool,
                    experimental_unified_exec_tool: config.use_experimental_unified_exec_tool,
                });

                let new_turn_context = TurnContext {
                    client,
                    tools_config,
                    user_instructions: prev.user_instructions.clone(),
                    base_instructions: prev.base_instructions.clone(),
                    approval_policy: new_approval_policy,
                    sandbox_policy: new_sandbox_policy.clone(),
                    shell_environment_policy: prev.shell_environment_policy.clone(),
                    cwd: new_cwd.clone(),
                    is_review_mode: false,
                };

                // Install the new persistent context for subsequent tasks/turns.
                turn_context = Arc::new(new_turn_context);

                // Optionally persist changes to model / effort
                if cwd.is_some() || approval_policy.is_some() || sandbox_policy.is_some() {
                    sess.record_conversation_items(&[ResponseItem::from(EnvironmentContext::new(
                        cwd,
                        approval_policy,
                        sandbox_policy,
                        // Shell is not configurable from turn to turn
                        None,
                    ))])
                    .await;
                }
            }
            Op::UserInput { items } => {
                // attempt to inject input into current task
                if let Err(items) = sess.inject_input(items) {
                    // no current task, spawn a new one
                    let task =
                        AgentTask::spawn(sess.clone(), Arc::clone(&turn_context), sub.id, items);
                    sess.set_task(task);
                }
            }
            Op::UserTurn {
                items,
                cwd,
                approval_policy,
                sandbox_policy,
                model,
                effort,
                summary,
            } => {
                // attempt to inject input into current task
                if let Err(items) = sess.inject_input(items) {
                    // Derive a fresh TurnContext for this turn using the provided overrides.
                    let provider = turn_context.client.get_provider();
                    let auth_manager = turn_context.client.get_auth_manager();

                    // Derive a model family for the requested model; fall back to the session's.
                    let model_family = find_family_for_model(&model)
                        .unwrap_or_else(|| config.model_family.clone());

                    // Create a per‑turn Config clone with the requested model/family.
                    let mut per_turn_config = (*config).clone();
                    per_turn_config.model = model.clone();
                    per_turn_config.model_family = model_family.clone();
                    if let Some(model_info) = get_model_info(&model_family) {
                        per_turn_config.model_context_window = Some(model_info.context_window);
                    }

                    // Build a new client with per‑turn reasoning settings.
                    // Reuse the same provider and session id; auth defaults to env/API key.
                    let client = ModelClient::new(
                        Arc::new(per_turn_config),
                        auth_manager,
                        provider,
                        effort,
                        summary,
                        sess.conversation_id,
                    );

                    let fresh_turn_context = TurnContext {
                        client,
                        tools_config: ToolsConfig::new(&ToolsConfigParams {
                            model_family: &model_family,
                            approval_policy,
                            sandbox_policy: sandbox_policy.clone(),
                            include_plan_tool: config.include_plan_tool,
                            include_apply_patch_tool: config.include_apply_patch_tool,
                            include_web_search_request: config.tools_web_search_request,
                            use_streamable_shell_tool: config
                                .use_experimental_streamable_shell_tool,
                            include_view_image_tool: config.include_view_image_tool,
                            experimental_unified_exec_tool: config
                                .use_experimental_unified_exec_tool,
                        }),
                        user_instructions: turn_context.user_instructions.clone(),
                        base_instructions: turn_context.base_instructions.clone(),
                        approval_policy,
                        sandbox_policy,
                        shell_environment_policy: turn_context.shell_environment_policy.clone(),
                        cwd,
                        is_review_mode: false,
                    };
                    // TODO: record the new environment context in the conversation history
                    // no current task, spawn a new one with the per‑turn context
                    let task =
                        AgentTask::spawn(sess.clone(), Arc::new(fresh_turn_context), sub.id, items);
                    sess.set_task(task);
                }
            }
            Op::ExecApproval { id, decision } => match decision {
                ReviewDecision::Abort => {
                    sess.interrupt_task();
                }
                other => sess.notify_approval(&id, other),
            },
            Op::PatchApproval { id, decision } => match decision {
                ReviewDecision::Abort => {
                    sess.interrupt_task();
                }
                other => sess.notify_approval(&id, other),
            },
            Op::AddToHistory { text } => {
                let id = sess.conversation_id;
                let config = config.clone();
                tokio::spawn(async move {
                    if let Err(e) = crate::message_history::append_entry(&text, &id, &config).await
                    {
                        warn!("failed to append to message history: {e}");
                    }
                });
            }

            Op::GetHistoryEntryRequest { offset, log_id } => {
                let config = config.clone();
                let sess_clone = sess.clone();
                let sub_id = sub.id.clone();

                tokio::spawn(async move {
                    // Run lookup in blocking thread because it does file IO + locking.
                    let entry_opt = tokio::task::spawn_blocking(move || {
                        crate::message_history::lookup(log_id, offset, &config)
                    })
                    .await
                    .unwrap_or(None);

                    let event = Event {
                        id: sub_id,
                        msg: EventMsg::GetHistoryEntryResponse(
                            crate::protocol::GetHistoryEntryResponseEvent {
                                offset,
                                log_id,
                                entry: entry_opt.map(|e| {
                                    codex_protocol::message_history::HistoryEntry {
                                        conversation_id: e.session_id,
                                        ts: e.ts,
                                        text: e.text,
                                    }
                                }),
                            },
                        ),
                    };

                    sess_clone.send_event(event).await;
                });
            }
            Op::ListMcpTools => {
                let sub_id = sub.id.clone();

                // This is a cheap lookup from the connection manager's cache.
                let tools = sess.mcp_connection_manager.list_all_tools();
                let event = Event {
                    id: sub_id,
                    msg: EventMsg::McpListToolsResponse(
                        crate::protocol::McpListToolsResponseEvent { tools },
                    ),
                };
                sess.send_event(event).await;
            }
            Op::ListCustomPrompts => {
                let sub_id = sub.id.clone();

                let custom_prompts: Vec<CustomPrompt> =
                    if let Some(dir) = crate::custom_prompts::default_prompts_dir() {
                        crate::custom_prompts::discover_prompts_in(&dir).await
                    } else {
                        Vec::new()
                    };

                let event = Event {
                    id: sub_id,
                    msg: EventMsg::ListCustomPromptsResponse(ListCustomPromptsResponseEvent {
                        custom_prompts,
                    }),
                };
                sess.send_event(event).await;
            }
            Op::Compact => {
                // Attempt to inject input into current task
                if let Err(items) = sess.inject_input(vec![InputItem::Text {
                    text: compact::COMPACT_TRIGGER_TEXT.to_string(),
                }]) {
                    compact::spawn_compact_task(
                        sess.clone(),
                        Arc::clone(&turn_context),
                        sub.id,
                        items,
                    );
                }
            }
            Op::Shutdown => {
                info!("Shutting down Codex instance");

                // Gracefully flush and shutdown rollout recorder on session end so tests
                // that inspect the rollout file do not race with the background writer.
                let recorder_opt = sess.rollout.lock_unchecked().take();
                if let Some(rec) = recorder_opt
                    && let Err(e) = rec.shutdown().await
                {
                    warn!("failed to shutdown rollout recorder: {e}");
                    let event = Event {
                        id: sub.id.clone(),
                        msg: EventMsg::Error(ErrorEvent {
                            message: "Failed to shutdown rollout recorder".to_string(),
                        }),
                    };
                    sess.send_event(event).await;
                }

                let event = Event {
                    id: sub.id.clone(),
                    msg: EventMsg::ShutdownComplete,
                };
                sess.send_event(event).await;
                break;
            }
            Op::GetPath => {
                let sub_id = sub.id.clone();
                // Flush rollout writes before returning the path so readers observe a consistent file.
                let (path, rec_opt) = {
                    let guard = sess.rollout.lock_unchecked();
                    match guard.as_ref() {
                        Some(rec) => (rec.get_rollout_path(), Some(rec.clone())),
                        None => {
                            error!("rollout recorder not found");
                            continue;
                        }
                    }
                };
                if let Some(rec) = rec_opt
                    && let Err(e) = rec.flush().await
                {
                    warn!("failed to flush rollout recorder before GetHistory: {e}");
                }
                let event = Event {
                    id: sub_id.clone(),
                    msg: EventMsg::ConversationPath(ConversationPathResponseEvent {
                        conversation_id: sess.conversation_id,
                        path,
                    }),
                };
                sess.send_event(event).await;
            }
            Op::Review { review_request } => {
                spawn_review_thread(
                    sess.clone(),
                    config.clone(),
                    turn_context.clone(),
                    sub.id,
                    review_request,
                )
                .await;
            }
            _ => {
                // Ignore unknown ops; enum is non_exhaustive to allow extensions.
            }
        }
    }
    debug!("Agent loop exited");
}

/// Spawn a review thread using the given prompt.
async fn spawn_review_thread(
    sess: Arc<Session>,
    config: Arc<Config>,
    parent_turn_context: Arc<TurnContext>,
    sub_id: String,
    review_request: ReviewRequest,
) {
    let model = config.review_model.clone();
    let review_model_family = find_family_for_model(&model)
        .unwrap_or_else(|| parent_turn_context.client.get_model_family());
    let tools_config = ToolsConfig::new(&ToolsConfigParams {
        model_family: &review_model_family,
        approval_policy: parent_turn_context.approval_policy,
        sandbox_policy: parent_turn_context.sandbox_policy.clone(),
        include_plan_tool: false,
        include_apply_patch_tool: config.include_apply_patch_tool,
        include_web_search_request: false,
        use_streamable_shell_tool: false,
        include_view_image_tool: false,
        experimental_unified_exec_tool: config.use_experimental_unified_exec_tool,
    });

    let base_instructions = Some(REVIEW_PROMPT.to_string());
    let provider = parent_turn_context.client.get_provider();
    let auth_manager = parent_turn_context.client.get_auth_manager();
    let model_family = review_model_family.clone();

    // Build per‑turn client with the requested model/family.
    let mut per_turn_config = (*config).clone();
    per_turn_config.model = model.clone();
    per_turn_config.model_family = model_family.clone();
    if let Some(model_info) = get_model_info(&model_family) {
        per_turn_config.model_context_window = Some(model_info.context_window);
    }

    let client = ModelClient::new(
        Arc::new(per_turn_config),
        auth_manager,
        provider,
        parent_turn_context.client.get_reasoning_effort(),
        parent_turn_context.client.get_reasoning_summary(),
        sess.conversation_id,
    );

    let review_turn_context = TurnContext {
        client,
        tools_config,
        user_instructions: None,
        base_instructions,
        approval_policy: parent_turn_context.approval_policy,
        sandbox_policy: parent_turn_context.sandbox_policy.clone(),
        shell_environment_policy: parent_turn_context.shell_environment_policy.clone(),
        cwd: parent_turn_context.cwd.clone(),
        is_review_mode: true,
    };

    // Seed the child task with the review prompt as the initial user message.
    let input: Vec<InputItem> = vec![InputItem::Text {
        text: review_request.prompt.clone(),
    }];
    let tc = Arc::new(review_turn_context);

    // Clone sub_id for the upcoming announcement before moving it into the task.
    let sub_id_for_event = sub_id.clone();
    let task = AgentTask::review(sess.clone(), tc.clone(), sub_id, input);
    sess.set_task(task);

    // Announce entering review mode so UIs can switch modes.
    sess.send_event(Event {
        id: sub_id_for_event,
        msg: EventMsg::EnteredReviewMode(review_request),
    })
    .await;
}

/// Recorded provenance of the one reference-list price record, kept as a single
/// string so every amount can name the snapshot it came from: the explicit price
/// basis, the source URL, the recorded revision/date, the response-body digest
/// and the docs-schema route/digest. It carries no rate, no invoice and no
/// savings claim.
const LUNA_REFERENCE_PRICE_SOURCE: &str = concat!(
    "basis=openrouter_reference_list not_uos_invoice ",
    "url=https://openrouter.ai/api/v1/models ",
    "revision=openrouter-2026-09-19-5ecb4fe3 ",
    "date=2026-09-19T00:12:35.624Z ",
    "body_sha256=5ecb4fe389783c8f2bf49772ae6d405e230172f26401bc9ff06f45ad4c70ab74 ",
    "docs_schema=https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties.md ",
    "docs_schema_sha256=e973433327496ad6b12720c85c9c78cc28ee94b8af01c0ac33ad5ca120792a7c",
);

/// Production integration point for authoritative recorded rate data (M0 §4.1).
///
/// Returns the recorded, versioned pricing catalog for the exact wire model
/// slug, or `None` when no record exists (every other model, including
/// `gpt-6-astra`). The one record is the immutable OpenRouter public-catalog
/// snapshot captured 2026-09-19 (`openrouter-2026-09-19-5ecb4fe3`): OpenRouter
/// reference list prices in USD, recorded in the catalog's `per_1m_tokens`
/// unit. It is not a UOS invoice price and nothing here measures or claims
/// savings.
///
/// The snapshot's `web_search` price is charged per call, not per token, so it
/// is deliberately absent from this token-rate record. Its long-context tier
/// (strict `prompt_tokens_total > 272000`) prices the terms separately and is
/// recorded as per-term rates ([`working_set::LongContextTermRates`]): above the
/// threshold each recorded rate overrides that term's base rate and a term the
/// tier does not record inherits its base rate.
///
/// The record itself claims no freshness: [`resolved_pricing_catalog`] validates
/// the snapshot age against the fixed reference TTL when it is resolved.
fn recorded_pricing_catalog(model: &str) -> Option<working_set::PricingCatalog> {
    match model {
        // Exact string equality on the wire slug only: no namespace, alias,
        // `:batch`/`:free` suffix, `-pro` variant or fuzzy match resolves here.
        "gpt-5.6-luna" => Some(working_set::PricingCatalog {
            catalog_version: "openrouter-2026-09-19-5ecb4fe3".to_string(),
            model: "gpt-5.6-luna".to_string(),
            provider: "openrouter".to_string(),
            currency: "USD".to_string(),
            unit: "per_1m_tokens".to_string(),
            price_source: Some(LUNA_REFERENCE_PRICE_SOURCE.to_string()),
            fetched_at: Some("2026-09-19T00:12:35.624Z".to_string()),
            // Freshness is established when the record is resolved against the
            // fixed reference TTL; the immutable record never claims it.
            freshness: working_set::Freshness::Unknown,
            // Snapshot USD-per-token rates x 1e6, in the catalog's per-1M unit.
            uncached_input_rate: Some(0.2),
            cached_input_rate: Some(0.02),
            cache_write_rate: Some(0.25),
            output_rate: Some(1.2),
            // The snapshot's `completion` price covers generated output, but the
            // row does not record whether internal reasoning is billed as output,
            // so no value is guessed here.
            reasoning_in_output: None,
            // Not recorded in the snapshot: the selector degrades only the
            // requests that need a minimum prefix (`cache_ineligible`) instead
            // of inventing one.
            min_cacheable_prefix_tokens: None,
            // Cache read and write rates are recorded, so the eligibility rules
            // are usable; the minimum-prefix requirement above stays a
            // per-request unknown.
            cache_eligibility_known: true,
            long_context: Some(working_set::LongContextRule {
                threshold_tokens: 272_000,
                // Unused while `term_rates` is recorded: the tier prices each
                // term with its own rate, so no uniform multiplier applies.
                multiplier: 1.0,
                term_rates: Some(working_set::LongContextTermRates {
                    // Snapshot USD-per-token tier rates x 1e6.
                    uncached_input_rate: Some(0.4),
                    cached_input_rate: Some(0.04),
                    cache_write_rate: Some(0.5),
                    output_rate: Some(1.8),
                }),
            }),
            // Every rate the selector consumes is populated; the min-prefix
            // absence is a per-request `cache_ineligible` fallback, not a
            // catalog-level unknown that would degrade every request.
            unknown_fields: Vec::new(),
        }),
        _ => None,
    }
}

/// Fixed conservative TTL of a recorded reference-list snapshot. It is not a
/// product knob: an older snapshot must never drive cost, so it resolves stale.
const REFERENCE_PRICE_TTL_DAYS: i64 = 7;

/// Freshness of a recorded snapshot at `now`, against the fixed reference TTL.
///
/// Stale, future-dated and unparseable/missing fetch dates are all `Stale`: the
/// record cannot be verified as current, so the selector falls back with an
/// explicit `pricing_stale` reason instead of costing from it.
fn reference_price_freshness(
    fetched_at: Option<&str>,
    now: DateTime<Utc>,
) -> working_set::Freshness {
    let Some(fetched_at) = fetched_at else {
        return working_set::Freshness::Stale;
    };
    let Ok(fetched_at) = DateTime::parse_from_rfc3339(fetched_at) else {
        return working_set::Freshness::Stale;
    };
    let fetched_at = fetched_at.with_timezone(&Utc);
    if fetched_at > now || fetched_at < now - chrono::Duration::days(REFERENCE_PRICE_TTL_DAYS) {
        return working_set::Freshness::Stale;
    }
    working_set::Freshness::Fresh
}

/// Resolves the recorded catalog for an exact wire model slug at `now`, checking
/// the snapshot age against the fixed reference TTL. Pure given `(model, now)`
/// so the frozen selection stays deterministic: `freeze` reads the recorded
/// freshness and never a clock.
fn resolved_pricing_catalog(
    model: &str,
    now: DateTime<Utc>,
) -> Option<working_set::PricingCatalog> {
    let mut catalog = recorded_pricing_catalog(model)?;
    catalog.freshness = reference_price_freshness(catalog.fetched_at.as_deref(), now);
    Some(catalog)
}

/// Appends one JSON line to the append-only selection sidecar (create + append,
/// flushed before returning). Callers log failures and continue: persistence
/// never affects the model input or the canonical history.
async fn append_sidecar_record(
    path: &std::path::Path,
    record: &working_set::SidecarRecord,
) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;

    let line = serde_json::to_string(record)
        .map_err(|err| std::io::Error::other(format!("serialize sidecar record: {err}")))?;
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    file.write_all(line.as_bytes()).await?;
    file.write_all(b"\n").await?;
    file.flush().await?;
    Ok(())
}

/// Takes a user message as input and runs a loop where, at each turn, the model
/// replies with either:
///
/// - requested function calls
/// - an assistant message
///
/// While it is possible for the model to return multiple of these items in a
/// single turn, in practice, we generally one item per turn:
///
/// - If the model requests a function call, we execute it and send the output
///   back to the model in the next turn.
/// - If the model sends only an assistant message, we record it in the
///   conversation history and consider the task complete.
///
/// Review mode: when `turn_context.is_review_mode` is true, the turn runs in an
/// isolated in-memory thread without the parent session's prior history or
/// user_instructions. Emits ExitedReviewMode upon final review message.
async fn run_task(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    sub_id: String,
    input: Vec<InputItem>,
) {
    if input.is_empty() {
        return;
    }
    let event = Event {
        id: sub_id.clone(),
        msg: EventMsg::TaskStarted(TaskStartedEvent {
            model_context_window: turn_context.client.get_model_context_window(),
        }),
    };
    sess.send_event(event).await;

    let initial_input_for_turn: ResponseInputItem = ResponseInputItem::from(input);
    // For review threads, keep an isolated in-memory history so the
    // model sees a fresh conversation without the parent session's history.
    // For normal turns, continue recording to the session history as before.
    let is_review_mode = turn_context.is_review_mode;
    let mut review_thread_history: Vec<ResponseItem> = Vec::new();
    // Compact-on-demand: freeze the working set once per user request, before the
    // first model turn. Internal tool follow-up turns below reuse the frozen
    // projection and append the freshly recorded items (active tool chain)
    // verbatim. Review threads keep their isolated in-memory history and are
    // excluded from selection in this slice.
    let mut request_working_set: Option<working_set::FrozenSelection> = None;
    let mut working_set_request_index: u64 = 0;
    let mut turn_index: u64 = 0;
    if is_review_mode {
        review_thread_history.push(initial_input_for_turn.into());
    } else {
        working_set_request_index = sess.next_working_set_request_index();
        // Snapshot the canonical history before the request input is appended so
        // the request itself is pinned as `current_request` rather than history.
        let mut freeze_input = sess.turn_input_with_history(Vec::new());
        let current_request_start = freeze_input.len();
        sess.record_input_and_rollout_usermsg(&initial_input_for_turn)
            .await;
        freeze_input.push(initial_input_for_turn.into());
        let previous_epoch = sess.last_working_set_epoch();
        let usage = sess
            .last_token_usage()
            .as_ref()
            .map(working_set::UsageEvidence::from_token_usage);
        // The effective instructions and tool set are part of the epoch identity.
        // Their digests are recorded; the text and definitions never are.
        let tools = get_openai_tools(
            &turn_context.tools_config,
            Some(sess.mcp_connection_manager.list_all_tools()),
        );
        let instructions = Prompt {
            input: Vec::new(),
            tools: tools.clone(),
            base_instructions_override: turn_context.base_instructions.clone(),
        }
        .get_full_instructions(&turn_context.client.get_model_family())
        .into_owned();
        let model = turn_context.client.get_model();
        let catalog = sess.working_set_pricing_catalog(&model);
        let frozen = working_set::freeze(&working_set::FreezeInputs {
            model,
            reasoning_effort: turn_context
                .client
                .get_reasoning_effort()
                .map(|effort| effort.to_string())
                .unwrap_or_else(|| "unset".to_string()),
            instructions_digest: working_set::digest_text(&instructions),
            tools_digest: working_set::digest_serialized(&tools),
            canonical: &freeze_input,
            current_request_start,
            explicit_pins: &[],
            // Authoritative recorded rates are resolved per request by
            // `recorded_pricing_catalog` (the production integration point). The
            // one recorded OpenRouter reference-list record covers the exact
            // `gpt-5.6-luna` wire slug; every other model resolves to `None` and
            // the selector records `pricing_unknown` and sends the canonical
            // input unchanged (I6/I10/I12).
            catalog: catalog.as_ref(),
            usage: usage.as_ref(),
            previous_epoch_id: previous_epoch.as_deref(),
            horizon: working_set::Horizon::default(),
        });
        let manifest = frozen.manifest();
        debug!(
            model = %manifest.model,
            canonical_len = manifest.canonical.log_len,
            projection_len = manifest.projection.len,
            projection_digest = %manifest.projection.digest,
            manifest_digest = %frozen.manifest_digest(),
            confidence = manifest.cost.confidence.as_str(),
            applied = manifest.decision.applied,
            fallback = manifest.fallback.reason.unwrap_or("none"),
            "compact-on-demand working set frozen for user request"
        );
        sess.set_last_working_set_epoch(frozen.epoch_id().to_string());
        request_working_set = Some(frozen);
        // Persist the structural selection manifest once per user request. The
        // canonical digest before the request was appended and the canonical input
        // the selection was derived from are both recorded; selection never
        // mutates the canonical history, so the latter equals the manifest digest.
        let canonical_digest_before_request =
            working_set::digest_items(&freeze_input[..current_request_start]);
        if let Some(frozen) = request_working_set.as_ref() {
            let record = working_set::SidecarRecord::selection(
                working_set_request_index,
                frozen,
                canonical_digest_before_request,
            );
            sess.record_working_set_sidecar(&record).await;
        }
    }

    let mut last_agent_message: Option<String> = None;
    // Although from the perspective of codex.rs, TurnDiffTracker has the lifecycle of a Task which contains
    // many turns, from the perspective of the user, it is a single turn.
    let mut turn_diff_tracker = TurnDiffTracker::new();
    let mut auto_compact_recently_attempted = false;

    loop {
        // Note that pending_input would be something like a message the user
        // submitted through the UI while the model was running. Though the UI
        // may support this, the model might not.
        let pending_input = sess
            .get_pending_input()
            .into_iter()
            .map(ResponseItem::from)
            .collect::<Vec<ResponseItem>>();

        // Construct the input that we will send to the model.
        //
        // - For review threads, use the isolated in-memory history so the
        //   model sees a fresh conversation (no parent history/user_instructions).
        //
        // - For normal turns, use the session's full history. When using the
        //   chat completions API (or ZDR clients), the model needs the full
        //   conversation history on each turn. The rollout file, however, should
        //   only record the new items that originated in this turn so that it
        //   represents an append-only log without duplicates.
        let turn_input: Vec<ResponseItem> = if is_review_mode {
            if !pending_input.is_empty() {
                review_thread_history.extend(pending_input);
            }
            review_thread_history.clone()
        } else {
            sess.record_conversation_items(&pending_input).await;
            sess.turn_input_with_history(pending_input)
        };

        let turn_input_messages: Vec<String> = turn_input
            .iter()
            .filter_map(|item| match item {
                ResponseItem::Message { content, .. } => Some(content),
                _ => None,
            })
            .flat_map(|content| {
                content.iter().filter_map(|item| match item {
                    ContentItem::OutputText { text } => Some(text.clone()),
                    _ => None,
                })
            })
            .collect();
        // The working-set projection replaces only `Prompt.input`. The freshly
        // recorded items after the frozen prefix are appended verbatim, and any
        // mismatch falls back to the canonical input for this turn. Every model
        // turn is recorded in the sidecar with the canonical and wire digests it
        // actually used.
        turn_index += 1;
        let canonical_digest = working_set::digest_items(&turn_input);
        let derived_input = request_working_set
            .as_ref()
            .map(|frozen| frozen.derived_input(&turn_input));
        if let (Some(frozen), Some(derived)) =
            (request_working_set.as_ref(), derived_input.as_ref())
        {
            let (decision, reason, wire_len, wire_digest) = match derived {
                working_set::DerivedInput::Projected(items) => (
                    working_set::SIDECAR_DECISION_PROJECTED,
                    working_set::SIDECAR_DECISION_PROJECTED,
                    items.len(),
                    working_set::digest_items(items),
                ),
                working_set::DerivedInput::Canonical(reason) => {
                    debug!(
                        reason = reason.as_str(),
                        "compact-on-demand working set canonical fallback"
                    );
                    (
                        working_set::SIDECAR_DECISION_CANONICAL,
                        reason.as_str(),
                        turn_input.len(),
                        canonical_digest.clone(),
                    )
                }
            };
            let record = working_set::SidecarRecord::Turn {
                request_index: working_set_request_index,
                turn_index,
                decision,
                reason,
                epoch_id: frozen.epoch_id().to_string(),
                canonical_len: turn_input.len(),
                canonical_digest,
                wire_len,
                wire_digest,
            };
            sess.record_working_set_sidecar(&record).await;
        }
        let model_input = match derived_input {
            Some(working_set::DerivedInput::Projected(items)) => items,
            _ => turn_input,
        };
        match run_turn(
            &sess,
            turn_context.as_ref(),
            &mut turn_diff_tracker,
            sub_id.clone(),
            model_input,
        )
        .await
        {
            Ok(turn_output) => {
                let TurnRunResult {
                    processed_items,
                    total_token_usage,
                } = turn_output;
                let limit = turn_context
                    .client
                    .get_auto_compact_token_limit()
                    .unwrap_or(i64::MAX);
                let total_usage_tokens = total_token_usage
                    .as_ref()
                    .map(|usage| usage.tokens_in_context_window());
                let token_limit_reached = total_usage_tokens
                    .map(|tokens| (tokens as i64) >= limit)
                    .unwrap_or(false);
                let mut items_to_record_in_conversation_history = Vec::<ResponseItem>::new();
                let mut responses = Vec::<ResponseInputItem>::new();
                for processed_response_item in processed_items {
                    let ProcessedResponseItem { item, response } = processed_response_item;
                    match (&item, &response) {
                        (ResponseItem::Message { role, .. }, None) if role == "assistant" => {
                            // If the model returned a message, we need to record it.
                            items_to_record_in_conversation_history.push(item);
                        }
                        (
                            ResponseItem::LocalShellCall { .. },
                            Some(ResponseInputItem::FunctionCallOutput { call_id, output }),
                        ) => {
                            items_to_record_in_conversation_history.push(item);
                            items_to_record_in_conversation_history.push(
                                ResponseItem::FunctionCallOutput {
                                    call_id: call_id.clone(),
                                    output: output.clone(),
                                },
                            );
                        }
                        (
                            ResponseItem::FunctionCall { .. },
                            Some(ResponseInputItem::FunctionCallOutput { call_id, output }),
                        ) => {
                            items_to_record_in_conversation_history.push(item);
                            items_to_record_in_conversation_history.push(
                                ResponseItem::FunctionCallOutput {
                                    call_id: call_id.clone(),
                                    output: output.clone(),
                                },
                            );
                        }
                        (
                            ResponseItem::CustomToolCall { .. },
                            Some(ResponseInputItem::CustomToolCallOutput { call_id, output }),
                        ) => {
                            items_to_record_in_conversation_history.push(item);
                            items_to_record_in_conversation_history.push(
                                ResponseItem::CustomToolCallOutput {
                                    call_id: call_id.clone(),
                                    output: output.clone(),
                                },
                            );
                        }
                        (
                            ResponseItem::FunctionCall { .. },
                            Some(ResponseInputItem::McpToolCallOutput { call_id, result }),
                        ) => {
                            items_to_record_in_conversation_history.push(item);
                            let output = match result {
                                Ok(call_tool_result) => {
                                    convert_call_tool_result_to_function_call_output_payload(
                                        call_tool_result,
                                    )
                                }
                                Err(err) => FunctionCallOutputPayload {
                                    content: err.clone(),
                                    success: Some(false),
                                },
                            };
                            items_to_record_in_conversation_history.push(
                                ResponseItem::FunctionCallOutput {
                                    call_id: call_id.clone(),
                                    output,
                                },
                            );
                        }
                        (
                            ResponseItem::Reasoning {
                                id,
                                summary,
                                content,
                                encrypted_content,
                            },
                            None,
                        ) => {
                            items_to_record_in_conversation_history.push(ResponseItem::Reasoning {
                                id: id.clone(),
                                summary: summary.clone(),
                                content: content.clone(),
                                encrypted_content: encrypted_content.clone(),
                            });
                        }
                        _ => {
                            warn!("Unexpected response item: {item:?} with response: {response:?}");
                        }
                    };
                    if let Some(response) = response {
                        responses.push(response);
                    }
                }

                // Only attempt to take the lock if there is something to record.
                if !items_to_record_in_conversation_history.is_empty() {
                    if is_review_mode {
                        review_thread_history
                            .extend(items_to_record_in_conversation_history.clone());
                    } else {
                        sess.record_conversation_items(&items_to_record_in_conversation_history)
                            .await;
                    }
                }

                if token_limit_reached {
                    if auto_compact_recently_attempted {
                        let limit_str = limit.to_string();
                        let current_tokens = total_usage_tokens
                            .map(|tokens| tokens.to_string())
                            .unwrap_or_else(|| "unknown".to_string());
                        let event = Event {
                            id: sub_id.clone(),
                            msg: EventMsg::Error(ErrorEvent {
                                message: format!(
                                    "Conversation is still above the token limit after automatic summarization (limit {limit_str}, current {current_tokens}). Please start a new session or trim your input."
                                ),
                            }),
                        };
                        sess.send_event(event).await;
                        break;
                    }
                    auto_compact_recently_attempted = true;
                    compact::run_inline_auto_compact_task(sess.clone(), turn_context.clone()).await;
                    continue;
                }

                auto_compact_recently_attempted = false;

                if responses.is_empty() {
                    last_agent_message = get_last_assistant_message_from_turn(
                        &items_to_record_in_conversation_history,
                    );
                    sess.maybe_notify(UserNotification::AgentTurnComplete {
                        turn_id: sub_id.clone(),
                        input_messages: turn_input_messages,
                        last_assistant_message: last_agent_message.clone(),
                    });
                    break;
                }
                continue;
            }
            Err(e) => {
                info!("Turn error: {e:#}");
                let event = Event {
                    id: sub_id.clone(),
                    msg: EventMsg::Error(ErrorEvent {
                        message: e.to_string(),
                    }),
                };
                sess.send_event(event).await;
                // let the user continue the conversation
                break;
            }
        }
    }

    // If this was a review thread and we have a final assistant message,
    // try to parse it as a ReviewOutput.
    //
    // If parsing fails, construct a minimal ReviewOutputEvent using the plain
    // text as the overall explanation. Else, just exit review mode with None.
    //
    // Emits an ExitedReviewMode event with the parsed review output.
    if turn_context.is_review_mode {
        exit_review_mode(
            sess.clone(),
            sub_id.clone(),
            last_agent_message.as_deref().map(parse_review_output_event),
        )
        .await;
    }

    sess.remove_task(&sub_id);
    let event = Event {
        id: sub_id,
        msg: EventMsg::TaskComplete(TaskCompleteEvent { last_agent_message }),
    };
    sess.send_event(event).await;
}

/// Parse the review output; when not valid JSON, build a structured
/// fallback that carries the plain text as the overall explanation.
///
/// Returns: a ReviewOutputEvent parsed from JSON or a fallback populated from text.
fn parse_review_output_event(text: &str) -> ReviewOutputEvent {
    // Try direct parse first
    if let Ok(ev) = serde_json::from_str::<ReviewOutputEvent>(text) {
        return ev;
    }
    // If wrapped in markdown fences or extra prose, attempt to extract the first JSON object
    if let (Some(start), Some(end)) = (text.find('{'), text.rfind('}'))
        && start < end
        && let Some(slice) = text.get(start..=end)
        && let Ok(ev) = serde_json::from_str::<ReviewOutputEvent>(slice)
    {
        return ev;
    }
    // Not JSON – return a structured ReviewOutputEvent that carries
    // the plain text as the overall explanation.
    ReviewOutputEvent {
        overall_explanation: text.to_string(),
        ..Default::default()
    }
}

async fn run_turn(
    sess: &Session,
    turn_context: &TurnContext,
    turn_diff_tracker: &mut TurnDiffTracker,
    sub_id: String,
    input: Vec<ResponseItem>,
) -> CodexResult<TurnRunResult> {
    let tools = get_openai_tools(
        &turn_context.tools_config,
        Some(sess.mcp_connection_manager.list_all_tools()),
    );

    let prompt = Prompt {
        input,
        tools,
        base_instructions_override: turn_context.base_instructions.clone(),
    };

    let mut retries = 0;
    loop {
        match try_run_turn(sess, turn_context, turn_diff_tracker, &sub_id, &prompt).await {
            Ok(output) => return Ok(output),
            Err(CodexErr::Interrupted) => return Err(CodexErr::Interrupted),
            Err(CodexErr::EnvVar(var)) => return Err(CodexErr::EnvVar(var)),
            Err(e @ (CodexErr::UsageLimitReached(_) | CodexErr::UsageNotIncluded)) => {
                return Err(e);
            }
            Err(e) => {
                // Use the configured provider-specific stream retry budget.
                let max_retries = turn_context.client.get_provider().stream_max_retries();
                if retries < max_retries {
                    retries += 1;
                    let delay = match e {
                        CodexErr::Stream(_, Some(delay)) => delay,
                        _ => backoff(retries),
                    };
                    warn!(
                        "stream disconnected - retrying turn ({retries}/{max_retries} in {delay:?})...",
                    );

                    // Surface retry information to any UI/front‑end so the
                    // user understands what is happening instead of staring
                    // at a seemingly frozen screen.
                    sess.notify_stream_error(
                        &sub_id,
                        format!(
                            "stream error: {e}; retrying {retries}/{max_retries} in {delay:?}…"
                        ),
                    )
                    .await;

                    tokio::time::sleep(delay).await;
                } else {
                    return Err(e);
                }
            }
        }
    }
}

/// When the model is prompted, it returns a stream of events. Some of these
/// events map to a `ResponseItem`. A `ResponseItem` may need to be
/// "handled" such that it produces a `ResponseInputItem` that needs to be
/// sent back to the model on the next turn.
#[derive(Debug)]
struct ProcessedResponseItem {
    item: ResponseItem,
    response: Option<ResponseInputItem>,
}

#[derive(Debug)]
struct TurnRunResult {
    processed_items: Vec<ProcessedResponseItem>,
    total_token_usage: Option<TokenUsage>,
}

async fn try_run_turn(
    sess: &Session,
    turn_context: &TurnContext,
    turn_diff_tracker: &mut TurnDiffTracker,
    sub_id: &str,
    prompt: &Prompt,
) -> CodexResult<TurnRunResult> {
    // call_ids that are part of this response.
    let completed_call_ids = prompt
        .input
        .iter()
        .filter_map(|ri| match ri {
            ResponseItem::FunctionCallOutput { call_id, .. } => Some(call_id),
            ResponseItem::LocalShellCall {
                call_id: Some(call_id),
                ..
            } => Some(call_id),
            ResponseItem::CustomToolCallOutput { call_id, .. } => Some(call_id),
            _ => None,
        })
        .collect::<Vec<_>>();

    // call_ids that were pending but are not part of this response.
    // This usually happens because the user interrupted the model before we responded to one of its tool calls
    // and then the user sent a follow-up message.
    let missing_calls = {
        prompt
            .input
            .iter()
            .filter_map(|ri| match ri {
                ResponseItem::FunctionCall { call_id, .. } => Some(call_id),
                ResponseItem::LocalShellCall {
                    call_id: Some(call_id),
                    ..
                } => Some(call_id),
                ResponseItem::CustomToolCall { call_id, .. } => Some(call_id),
                _ => None,
            })
            .filter_map(|call_id| {
                if completed_call_ids.contains(&call_id) {
                    None
                } else {
                    Some(call_id.clone())
                }
            })
            .map(|call_id| ResponseItem::CustomToolCallOutput {
                call_id,
                output: "aborted".to_string(),
            })
            .collect::<Vec<_>>()
    };
    let prompt: Cow<Prompt> = if missing_calls.is_empty() {
        Cow::Borrowed(prompt)
    } else {
        // Add the synthetic aborted missing calls to the beginning of the input to ensure all call ids have responses.
        let input = [missing_calls, prompt.input.clone()].concat();
        Cow::Owned(Prompt {
            input,
            ..prompt.clone()
        })
    };

    let rollout_item = RolloutItem::TurnContext(TurnContextItem {
        cwd: turn_context.cwd.clone(),
        approval_policy: turn_context.approval_policy,
        sandbox_policy: turn_context.sandbox_policy.clone(),
        model: turn_context.client.get_model(),
        effort: turn_context.client.get_reasoning_effort(),
        summary: turn_context.client.get_reasoning_summary(),
    });
    sess.persist_rollout_items(&[rollout_item]).await;
    let mut stream = turn_context.client.clone().stream(&prompt).await?;

    let mut output = Vec::new();

    loop {
        // Poll the next item from the model stream. We must inspect *both* Ok and Err
        // cases so that transient stream failures (e.g., dropped SSE connection before
        // `response.completed`) bubble up and trigger the caller's retry logic.
        let event = stream.next().await;
        let Some(event) = event else {
            // Channel closed without yielding a final Completed event or explicit error.
            // Treat as a disconnected stream so the caller can retry.
            return Err(CodexErr::Stream(
                "stream closed before response.completed".into(),
                None,
            ));
        };

        let event = match event {
            Ok(ev) => ev,
            Err(e) => {
                // Propagate the underlying stream error to the caller (run_turn), which
                // will apply the configured `stream_max_retries` policy.
                return Err(e);
            }
        };

        match event {
            ResponseEvent::Created => {}
            ResponseEvent::OutputItemDone(item) => {
                let response = handle_response_item(
                    sess,
                    turn_context,
                    turn_diff_tracker,
                    sub_id,
                    item.clone(),
                )
                .await?;
                output.push(ProcessedResponseItem { item, response });
            }
            ResponseEvent::WebSearchCallBegin { call_id } => {
                let _ = sess
                    .tx_event
                    .send(Event {
                        id: sub_id.to_string(),
                        msg: EventMsg::WebSearchBegin(WebSearchBeginEvent { call_id }),
                    })
                    .await;
            }
            ResponseEvent::Completed {
                response_id: _,
                token_usage,
            } => {
                let info = sess.update_token_usage_info(turn_context, &token_usage);
                let _ = sess
                    .send_event(Event {
                        id: sub_id.to_string(),
                        msg: EventMsg::TokenCount(crate::protocol::TokenCountEvent { info }),
                    })
                    .await;

                let unified_diff = turn_diff_tracker.get_unified_diff();
                if let Ok(Some(unified_diff)) = unified_diff {
                    let msg = EventMsg::TurnDiff(TurnDiffEvent { unified_diff });
                    let event = Event {
                        id: sub_id.to_string(),
                        msg,
                    };
                    sess.send_event(event).await;
                }

                let result = TurnRunResult {
                    processed_items: output,
                    total_token_usage: token_usage.clone(),
                };

                return Ok(result);
            }
            ResponseEvent::OutputTextDelta(delta) => {
                // In review child threads, suppress assistant text deltas; the
                // UI will show a selection popup from the final ReviewOutput.
                if !turn_context.is_review_mode {
                    let event = Event {
                        id: sub_id.to_string(),
                        msg: EventMsg::AgentMessageDelta(AgentMessageDeltaEvent { delta }),
                    };
                    sess.send_event(event).await;
                } else {
                    trace!("suppressing OutputTextDelta in review mode");
                }
            }
            ResponseEvent::ReasoningSummaryDelta(delta) => {
                let event = Event {
                    id: sub_id.to_string(),
                    msg: EventMsg::AgentReasoningDelta(AgentReasoningDeltaEvent { delta }),
                };
                sess.send_event(event).await;
            }
            ResponseEvent::ReasoningSummaryPartAdded => {
                let event = Event {
                    id: sub_id.to_string(),
                    msg: EventMsg::AgentReasoningSectionBreak(AgentReasoningSectionBreakEvent {}),
                };
                sess.send_event(event).await;
            }
            ResponseEvent::ReasoningContentDelta(delta) => {
                if sess.show_raw_agent_reasoning {
                    let event = Event {
                        id: sub_id.to_string(),
                        msg: EventMsg::AgentReasoningRawContentDelta(
                            AgentReasoningRawContentDeltaEvent { delta },
                        ),
                    };
                    sess.send_event(event).await;
                }
            }
        }
    }
}

async fn handle_response_item(
    sess: &Session,
    turn_context: &TurnContext,
    turn_diff_tracker: &mut TurnDiffTracker,
    sub_id: &str,
    item: ResponseItem,
) -> CodexResult<Option<ResponseInputItem>> {
    debug!(?item, "Output item");
    let output = match item {
        ResponseItem::FunctionCall {
            name,
            arguments,
            call_id,
            ..
        } => {
            info!("FunctionCall: {name}({arguments})");
            Some(
                handle_function_call(
                    sess,
                    turn_context,
                    turn_diff_tracker,
                    sub_id.to_string(),
                    name,
                    arguments,
                    call_id,
                )
                .await,
            )
        }
        ResponseItem::LocalShellCall {
            id,
            call_id,
            status: _,
            action,
        } => {
            let LocalShellAction::Exec(action) = action;
            tracing::info!("LocalShellCall: {action:?}");
            let params = ShellToolCallParams {
                command: action.command,
                workdir: action.working_directory,
                timeout_ms: action.timeout_ms,
                with_escalated_permissions: None,
                justification: None,
            };
            let effective_call_id = match (call_id, id) {
                (Some(call_id), _) => call_id,
                (None, Some(id)) => id,
                (None, None) => {
                    error!("LocalShellCall without call_id or id");
                    return Ok(Some(ResponseInputItem::FunctionCallOutput {
                        call_id: "".to_string(),
                        output: FunctionCallOutputPayload {
                            content: "LocalShellCall without call_id or id".to_string(),
                            success: None,
                        },
                    }));
                }
            };

            let exec_params = to_exec_params(params, turn_context);
            Some(
                handle_container_exec_with_params(
                    exec_params,
                    sess,
                    turn_context,
                    turn_diff_tracker,
                    sub_id.to_string(),
                    effective_call_id,
                )
                .await,
            )
        }
        ResponseItem::CustomToolCall {
            id: _,
            call_id,
            name,
            input,
            status: _,
        } => Some(
            handle_custom_tool_call(
                sess,
                turn_context,
                turn_diff_tracker,
                sub_id.to_string(),
                name,
                input,
                call_id,
            )
            .await,
        ),
        ResponseItem::FunctionCallOutput { .. } => {
            debug!("unexpected FunctionCallOutput from stream");
            None
        }
        ResponseItem::CustomToolCallOutput { .. } => {
            debug!("unexpected CustomToolCallOutput from stream");
            None
        }
        ResponseItem::Message { .. }
        | ResponseItem::Reasoning { .. }
        | ResponseItem::WebSearchCall { .. } => {
            // In review child threads, suppress assistant message events but
            // keep reasoning/web search.
            let msgs = match &item {
                ResponseItem::Message { .. } if turn_context.is_review_mode => {
                    trace!("suppressing assistant Message in review mode");
                    Vec::new()
                }
                _ => map_response_item_to_event_messages(&item, sess.show_raw_agent_reasoning),
            };
            for msg in msgs {
                let event = Event {
                    id: sub_id.to_string(),
                    msg,
                };
                sess.send_event(event).await;
            }
            None
        }
        ResponseItem::Other => None,
    };
    Ok(output)
}

async fn handle_unified_exec_tool_call(
    sess: &Session,
    call_id: String,
    session_id: Option<String>,
    arguments: Vec<String>,
    timeout_ms: Option<u64>,
) -> ResponseInputItem {
    let parsed_session_id = if let Some(session_id) = session_id {
        match session_id.parse::<i32>() {
            Ok(parsed) => Some(parsed),
            Err(output) => {
                return ResponseInputItem::FunctionCallOutput {
                    call_id: call_id.to_string(),
                    output: FunctionCallOutputPayload {
                        content: format!("invalid session_id: {session_id} due to error {output}"),
                        success: Some(false),
                    },
                };
            }
        }
    } else {
        None
    };

    let request = crate::unified_exec::UnifiedExecRequest {
        session_id: parsed_session_id,
        input_chunks: &arguments,
        timeout_ms,
    };

    let result = sess.unified_exec_manager.handle_request(request).await;

    let output_payload = match result {
        Ok(value) => {
            #[derive(Serialize)]
            struct SerializedUnifiedExecResult<'a> {
                session_id: Option<String>,
                output: &'a str,
            }

            match serde_json::to_string(&SerializedUnifiedExecResult {
                session_id: value.session_id.map(|id| id.to_string()),
                output: &value.output,
            }) {
                Ok(serialized) => FunctionCallOutputPayload {
                    content: serialized,
                    success: Some(true),
                },
                Err(err) => FunctionCallOutputPayload {
                    content: format!("failed to serialize unified exec output: {err}"),
                    success: Some(false),
                },
            }
        }
        Err(err) => FunctionCallOutputPayload {
            content: format!("unified exec failed: {err}"),
            success: Some(false),
        },
    };

    ResponseInputItem::FunctionCallOutput {
        call_id,
        output: output_payload,
    }
}

async fn handle_function_call(
    sess: &Session,
    turn_context: &TurnContext,
    turn_diff_tracker: &mut TurnDiffTracker,
    sub_id: String,
    name: String,
    arguments: String,
    call_id: String,
) -> ResponseInputItem {
    match name.as_str() {
        "container.exec" | "shell" => {
            let params = match parse_container_exec_arguments(arguments, turn_context, &call_id) {
                Ok(params) => params,
                Err(output) => {
                    return *output;
                }
            };
            handle_container_exec_with_params(
                params,
                sess,
                turn_context,
                turn_diff_tracker,
                sub_id,
                call_id,
            )
            .await
        }
        "unified_exec" => {
            #[derive(Deserialize)]
            struct UnifiedExecArgs {
                input: Vec<String>,
                #[serde(default)]
                session_id: Option<String>,
                #[serde(default)]
                timeout_ms: Option<u64>,
            }

            let args = match serde_json::from_str::<UnifiedExecArgs>(&arguments) {
                Ok(args) => args,
                Err(err) => {
                    return ResponseInputItem::FunctionCallOutput {
                        call_id,
                        output: FunctionCallOutputPayload {
                            content: format!("failed to parse function arguments: {err}"),
                            success: Some(false),
                        },
                    };
                }
            };

            handle_unified_exec_tool_call(
                sess,
                call_id,
                args.session_id,
                args.input,
                args.timeout_ms,
            )
            .await
        }
        "view_image" => {
            #[derive(serde::Deserialize)]
            struct SeeImageArgs {
                path: String,
            }
            let args = match serde_json::from_str::<SeeImageArgs>(&arguments) {
                Ok(a) => a,
                Err(e) => {
                    return ResponseInputItem::FunctionCallOutput {
                        call_id,
                        output: FunctionCallOutputPayload {
                            content: format!("failed to parse function arguments: {e}"),
                            success: Some(false),
                        },
                    };
                }
            };
            let abs = turn_context.resolve_path(Some(args.path));
            let output = match sess.inject_input(vec![InputItem::LocalImage { path: abs }]) {
                Ok(()) => FunctionCallOutputPayload {
                    content: "attached local image path".to_string(),
                    success: Some(true),
                },
                Err(_) => FunctionCallOutputPayload {
                    content: "unable to attach image (no active task)".to_string(),
                    success: Some(false),
                },
            };
            ResponseInputItem::FunctionCallOutput { call_id, output }
        }
        "apply_patch" => {
            let args = match serde_json::from_str::<ApplyPatchToolArgs>(&arguments) {
                Ok(a) => a,
                Err(e) => {
                    return ResponseInputItem::FunctionCallOutput {
                        call_id,
                        output: FunctionCallOutputPayload {
                            content: format!("failed to parse function arguments: {e}"),
                            success: None,
                        },
                    };
                }
            };
            let exec_params = ExecParams {
                command: vec!["apply_patch".to_string(), args.input.clone()],
                cwd: turn_context.cwd.clone(),
                timeout_ms: None,
                env: HashMap::new(),
                with_escalated_permissions: None,
                justification: None,
            };
            handle_container_exec_with_params(
                exec_params,
                sess,
                turn_context,
                turn_diff_tracker,
                sub_id,
                call_id,
            )
            .await
        }
        "update_plan" => handle_update_plan(sess, arguments, sub_id, call_id).await,
        EXEC_COMMAND_TOOL_NAME => {
            // TODO(mbolin): Sandbox check.
            let exec_params = match serde_json::from_str::<ExecCommandParams>(&arguments) {
                Ok(params) => params,
                Err(e) => {
                    return ResponseInputItem::FunctionCallOutput {
                        call_id,
                        output: FunctionCallOutputPayload {
                            content: format!("failed to parse function arguments: {e}"),
                            success: Some(false),
                        },
                    };
                }
            };
            let result = sess
                .session_manager
                .handle_exec_command_request(exec_params)
                .await;
            let function_call_output = crate::exec_command::result_into_payload(result);
            ResponseInputItem::FunctionCallOutput {
                call_id,
                output: function_call_output,
            }
        }
        WRITE_STDIN_TOOL_NAME => {
            let write_stdin_params = match serde_json::from_str::<WriteStdinParams>(&arguments) {
                Ok(params) => params,
                Err(e) => {
                    return ResponseInputItem::FunctionCallOutput {
                        call_id,
                        output: FunctionCallOutputPayload {
                            content: format!("failed to parse function arguments: {e}"),
                            success: Some(false),
                        },
                    };
                }
            };
            let result = sess
                .session_manager
                .handle_write_stdin_request(write_stdin_params)
                .await;
            let function_call_output: FunctionCallOutputPayload =
                crate::exec_command::result_into_payload(result);
            ResponseInputItem::FunctionCallOutput {
                call_id,
                output: function_call_output,
            }
        }
        _ => {
            match sess.mcp_connection_manager.parse_tool_name(&name) {
                Some((server, tool_name)) => {
                    // TODO(mbolin): Determine appropriate timeout for tool call.
                    let timeout = None;
                    handle_mcp_tool_call(
                        sess, &sub_id, call_id, server, tool_name, arguments, timeout,
                    )
                    .await
                }
                None => {
                    // Unknown function: reply with structured failure so the model can adapt.
                    ResponseInputItem::FunctionCallOutput {
                        call_id,
                        output: FunctionCallOutputPayload {
                            content: format!("unsupported call: {name}"),
                            success: None,
                        },
                    }
                }
            }
        }
    }
}

async fn handle_custom_tool_call(
    sess: &Session,
    turn_context: &TurnContext,
    turn_diff_tracker: &mut TurnDiffTracker,
    sub_id: String,
    name: String,
    input: String,
    call_id: String,
) -> ResponseInputItem {
    info!("CustomToolCall: {name} {input}");
    match name.as_str() {
        "apply_patch" => {
            let exec_params = ExecParams {
                command: vec!["apply_patch".to_string(), input.clone()],
                cwd: turn_context.cwd.clone(),
                timeout_ms: None,
                env: HashMap::new(),
                with_escalated_permissions: None,
                justification: None,
            };
            let resp = handle_container_exec_with_params(
                exec_params,
                sess,
                turn_context,
                turn_diff_tracker,
                sub_id,
                call_id,
            )
            .await;

            // Convert function-call style output into a custom tool call output
            match resp {
                ResponseInputItem::FunctionCallOutput { call_id, output } => {
                    ResponseInputItem::CustomToolCallOutput {
                        call_id,
                        output: output.content,
                    }
                }
                // Pass through if already a custom tool output or other variant
                other => other,
            }
        }
        _ => {
            debug!("unexpected CustomToolCall from stream");
            ResponseInputItem::CustomToolCallOutput {
                call_id,
                output: format!("unsupported custom tool call: {name}"),
            }
        }
    }
}

fn to_exec_params(params: ShellToolCallParams, turn_context: &TurnContext) -> ExecParams {
    ExecParams {
        command: params.command,
        cwd: turn_context.resolve_path(params.workdir.clone()),
        timeout_ms: params.timeout_ms,
        env: create_env(&turn_context.shell_environment_policy),
        with_escalated_permissions: params.with_escalated_permissions,
        justification: params.justification,
    }
}

fn parse_container_exec_arguments(
    arguments: String,
    turn_context: &TurnContext,
    call_id: &str,
) -> Result<ExecParams, Box<ResponseInputItem>> {
    // parse command
    match serde_json::from_str::<ShellToolCallParams>(&arguments) {
        Ok(shell_tool_call_params) => Ok(to_exec_params(shell_tool_call_params, turn_context)),
        Err(e) => {
            // allow model to re-sample
            let output = ResponseInputItem::FunctionCallOutput {
                call_id: call_id.to_string(),
                output: FunctionCallOutputPayload {
                    content: format!("failed to parse function arguments: {e}"),
                    success: None,
                },
            };
            Err(Box::new(output))
        }
    }
}

pub struct ExecInvokeArgs<'a> {
    pub params: ExecParams,
    pub sandbox_type: SandboxType,
    pub sandbox_policy: &'a SandboxPolicy,
    pub codex_linux_sandbox_exe: &'a Option<PathBuf>,
    pub stdout_stream: Option<StdoutStream>,
}

fn maybe_translate_shell_command(
    params: ExecParams,
    sess: &Session,
    turn_context: &TurnContext,
) -> ExecParams {
    let should_translate = matches!(sess.user_shell, crate::shell::Shell::PowerShell(_))
        || turn_context.shell_environment_policy.use_profile;

    if should_translate
        && let Some(command) = sess
            .user_shell
            .format_default_shell_invocation(params.command.clone())
    {
        return ExecParams { command, ..params };
    }
    params
}

async fn handle_container_exec_with_params(
    params: ExecParams,
    sess: &Session,
    turn_context: &TurnContext,
    turn_diff_tracker: &mut TurnDiffTracker,
    sub_id: String,
    call_id: String,
) -> ResponseInputItem {
    // check if this was a patch, and apply it if so
    let apply_patch_exec = match maybe_parse_apply_patch_verified(&params.command, &params.cwd) {
        MaybeApplyPatchVerified::Body(changes) => {
            match apply_patch::apply_patch(sess, turn_context, &sub_id, &call_id, changes).await {
                InternalApplyPatchInvocation::Output(item) => return item,
                InternalApplyPatchInvocation::DelegateToExec(apply_patch_exec) => {
                    Some(apply_patch_exec)
                }
            }
        }
        MaybeApplyPatchVerified::CorrectnessError(parse_error) => {
            // It looks like an invocation of `apply_patch`, but we
            // could not resolve it into a patch that would apply
            // cleanly. Return to model for resample.
            return ResponseInputItem::FunctionCallOutput {
                call_id,
                output: FunctionCallOutputPayload {
                    content: format!("error: {parse_error:#}"),
                    success: None,
                },
            };
        }
        MaybeApplyPatchVerified::ShellParseError(error) => {
            trace!("Failed to parse shell command, {error:?}");
            None
        }
        MaybeApplyPatchVerified::NotApplyPatch => None,
    };

    let (params, safety, command_for_display) = match &apply_patch_exec {
        Some(ApplyPatchExec {
            action: ApplyPatchAction { patch, cwd, .. },
            user_explicitly_approved_this_action,
        }) => {
            let path_to_codex = std::env::current_exe()
                .ok()
                .map(|p| p.to_string_lossy().to_string());
            let Some(path_to_codex) = path_to_codex else {
                return ResponseInputItem::FunctionCallOutput {
                    call_id,
                    output: FunctionCallOutputPayload {
                        content: "failed to determine path to codex executable".to_string(),
                        success: None,
                    },
                };
            };

            let params = ExecParams {
                command: vec![
                    path_to_codex,
                    CODEX_APPLY_PATCH_ARG1.to_string(),
                    patch.clone(),
                ],
                cwd: cwd.clone(),
                timeout_ms: params.timeout_ms,
                env: HashMap::new(),
                with_escalated_permissions: params.with_escalated_permissions,
                justification: params.justification.clone(),
            };
            let safety = if *user_explicitly_approved_this_action {
                SafetyCheck::AutoApprove {
                    sandbox_type: SandboxType::None,
                }
            } else {
                assess_safety_for_untrusted_command(
                    turn_context.approval_policy,
                    &turn_context.sandbox_policy,
                    params.with_escalated_permissions.unwrap_or(false),
                )
            };
            (
                params,
                safety,
                vec!["apply_patch".to_string(), patch.clone()],
            )
        }
        None => {
            let safety = {
                let state = sess.state.lock_unchecked();
                assess_command_safety(
                    &params.command,
                    turn_context.approval_policy,
                    &turn_context.sandbox_policy,
                    &state.approved_commands,
                    params.with_escalated_permissions.unwrap_or(false),
                )
            };
            let command_for_display = params.command.clone();
            (params, safety, command_for_display)
        }
    };

    let sandbox_type = match safety {
        SafetyCheck::AutoApprove { sandbox_type } => sandbox_type,
        SafetyCheck::AskUser => {
            let rx_approve = sess
                .request_command_approval(
                    sub_id.clone(),
                    call_id.clone(),
                    params.command.clone(),
                    params.cwd.clone(),
                    params.justification.clone(),
                )
                .await;
            match rx_approve.await.unwrap_or_default() {
                ReviewDecision::Approved => (),
                ReviewDecision::ApprovedForSession => {
                    sess.add_approved_command(params.command.clone());
                }
                ReviewDecision::Denied | ReviewDecision::Abort => {
                    return ResponseInputItem::FunctionCallOutput {
                        call_id,
                        output: FunctionCallOutputPayload {
                            content: "exec command rejected by user".to_string(),
                            success: None,
                        },
                    };
                }
            }
            // No sandboxing is applied because the user has given
            // explicit approval. Often, we end up in this case because
            // the command cannot be run in a sandbox, such as
            // installing a new dependency that requires network access.
            SandboxType::None
        }
        SafetyCheck::Reject { reason } => {
            return ResponseInputItem::FunctionCallOutput {
                call_id,
                output: FunctionCallOutputPayload {
                    content: format!("exec command rejected: {reason}"),
                    success: None,
                },
            };
        }
    };

    let exec_command_context = ExecCommandContext {
        sub_id: sub_id.clone(),
        call_id: call_id.clone(),
        command_for_display: command_for_display.clone(),
        cwd: params.cwd.clone(),
        apply_patch: apply_patch_exec.map(
            |ApplyPatchExec {
                 action,
                 user_explicitly_approved_this_action,
             }| ApplyPatchCommandContext {
                user_explicitly_approved_this_action,
                changes: convert_apply_patch_to_protocol(&action),
            },
        ),
    };

    let params = maybe_translate_shell_command(params, sess, turn_context);
    let output_result = sess
        .run_exec_with_events(
            turn_diff_tracker,
            exec_command_context.clone(),
            ExecInvokeArgs {
                params: params.clone(),
                sandbox_type,
                sandbox_policy: &turn_context.sandbox_policy,
                codex_linux_sandbox_exe: &sess.codex_linux_sandbox_exe,
                stdout_stream: if exec_command_context.apply_patch.is_some() {
                    None
                } else {
                    Some(StdoutStream {
                        sub_id: sub_id.clone(),
                        call_id: call_id.clone(),
                        tx_event: sess.tx_event.clone(),
                    })
                },
            },
        )
        .await;

    match output_result {
        Ok(output) => {
            let ExecToolCallOutput { exit_code, .. } = &output;

            let is_success = *exit_code == 0;
            let content = format_exec_output(&output);
            ResponseInputItem::FunctionCallOutput {
                call_id: call_id.clone(),
                output: FunctionCallOutputPayload {
                    content,
                    success: Some(is_success),
                },
            }
        }
        Err(CodexErr::Sandbox(error)) => {
            handle_sandbox_error(
                turn_diff_tracker,
                params,
                exec_command_context,
                error,
                sandbox_type,
                sess,
                turn_context,
            )
            .await
        }
        Err(e) => ResponseInputItem::FunctionCallOutput {
            call_id: call_id.clone(),
            output: FunctionCallOutputPayload {
                content: format!("execution error: {e}"),
                success: None,
            },
        },
    }
}

async fn handle_sandbox_error(
    turn_diff_tracker: &mut TurnDiffTracker,
    params: ExecParams,
    exec_command_context: ExecCommandContext,
    error: SandboxErr,
    sandbox_type: SandboxType,
    sess: &Session,
    turn_context: &TurnContext,
) -> ResponseInputItem {
    let call_id = exec_command_context.call_id.clone();
    let sub_id = exec_command_context.sub_id.clone();
    let cwd = exec_command_context.cwd.clone();

    if let SandboxErr::Timeout { output } = &error {
        let content = format_exec_output(output);
        return ResponseInputItem::FunctionCallOutput {
            call_id,
            output: FunctionCallOutputPayload {
                content,
                success: Some(false),
            },
        };
    }

    // Early out if either the user never wants to be asked for approval, or
    // we're letting the model manage escalation requests. Otherwise, continue
    match turn_context.approval_policy {
        AskForApproval::Never | AskForApproval::OnRequest => {
            return ResponseInputItem::FunctionCallOutput {
                call_id,
                output: FunctionCallOutputPayload {
                    content: format!(
                        "failed in sandbox {sandbox_type:?} with execution error: {error}"
                    ),
                    success: Some(false),
                },
            };
        }
        AskForApproval::UnlessTrusted | AskForApproval::OnFailure => (),
    }

    // Note that when `error` is `SandboxErr::Denied`, it could be a false
    // positive. That is, it may have exited with a non-zero exit code, not
    // because the sandbox denied it, but because that is its expected behavior,
    // i.e., a grep command that did not match anything. Ideally we would
    // include additional metadata on the command to indicate whether non-zero
    // exit codes merit a retry.

    // For now, we categorically ask the user to retry without sandbox and
    // emit the raw error as a background event.
    sess.notify_background_event(&sub_id, format!("Execution failed: {error}"))
        .await;

    let rx_approve = sess
        .request_command_approval(
            sub_id.clone(),
            call_id.clone(),
            params.command.clone(),
            cwd.clone(),
            Some("command failed; retry without sandbox?".to_string()),
        )
        .await;

    match rx_approve.await.unwrap_or_default() {
        ReviewDecision::Approved | ReviewDecision::ApprovedForSession => {
            // Persist this command as pre‑approved for the
            // remainder of the session so future
            // executions skip the sandbox directly.
            // TODO(ragona): Isn't this a bug? It always saves the command in an | fork?
            sess.add_approved_command(params.command.clone());
            // Inform UI we are retrying without sandbox.
            sess.notify_background_event(&sub_id, "retrying command without sandbox")
                .await;

            // This is an escalated retry; the policy will not be
            // examined and the sandbox has been set to `None`.
            let retry_output_result = sess
                .run_exec_with_events(
                    turn_diff_tracker,
                    exec_command_context.clone(),
                    ExecInvokeArgs {
                        params,
                        sandbox_type: SandboxType::None,
                        sandbox_policy: &turn_context.sandbox_policy,
                        codex_linux_sandbox_exe: &sess.codex_linux_sandbox_exe,
                        stdout_stream: if exec_command_context.apply_patch.is_some() {
                            None
                        } else {
                            Some(StdoutStream {
                                sub_id: sub_id.clone(),
                                call_id: call_id.clone(),
                                tx_event: sess.tx_event.clone(),
                            })
                        },
                    },
                )
                .await;

            match retry_output_result {
                Ok(retry_output) => {
                    let ExecToolCallOutput { exit_code, .. } = &retry_output;

                    let is_success = *exit_code == 0;
                    let content = format_exec_output(&retry_output);

                    ResponseInputItem::FunctionCallOutput {
                        call_id: call_id.clone(),
                        output: FunctionCallOutputPayload {
                            content,
                            success: Some(is_success),
                        },
                    }
                }
                Err(e) => ResponseInputItem::FunctionCallOutput {
                    call_id: call_id.clone(),
                    output: FunctionCallOutputPayload {
                        content: format!("retry failed: {e}"),
                        success: None,
                    },
                },
            }
        }
        ReviewDecision::Denied | ReviewDecision::Abort => {
            // Fall through to original failure handling.
            ResponseInputItem::FunctionCallOutput {
                call_id,
                output: FunctionCallOutputPayload {
                    content: "exec command rejected by user".to_string(),
                    success: None,
                },
            }
        }
    }
}

fn format_exec_output_str(exec_output: &ExecToolCallOutput) -> String {
    let ExecToolCallOutput {
        aggregated_output, ..
    } = exec_output;

    // Head+tail truncation for the model: show the beginning and end with an elision.
    // Clients still receive full streams; only this formatted summary is capped.

    let mut s = &aggregated_output.text;
    let prefixed_str: String;

    if exec_output.timed_out {
        prefixed_str = format!(
            "command timed out after {} milliseconds\n",
            exec_output.duration.as_millis()
        ) + s;
        s = &prefixed_str;
    }

    let total_lines = s.lines().count();
    if s.len() <= MODEL_FORMAT_MAX_BYTES && total_lines <= MODEL_FORMAT_MAX_LINES {
        return s.to_string();
    }

    let lines: Vec<&str> = s.lines().collect();
    let head_take = MODEL_FORMAT_HEAD_LINES.min(lines.len());
    let tail_take = MODEL_FORMAT_TAIL_LINES.min(lines.len().saturating_sub(head_take));
    let omitted = lines.len().saturating_sub(head_take + tail_take);

    // Join head and tail blocks (lines() strips newlines; reinsert them)
    let head_block = lines
        .iter()
        .take(head_take)
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");
    let tail_block = if tail_take > 0 {
        lines[lines.len() - tail_take..].join("\n")
    } else {
        String::new()
    };
    let marker = format!("\n[... omitted {omitted} of {total_lines} lines ...]\n\n");

    // Byte budgets for head/tail around the marker
    let mut head_budget = MODEL_FORMAT_HEAD_BYTES.min(MODEL_FORMAT_MAX_BYTES);
    let tail_budget = MODEL_FORMAT_MAX_BYTES.saturating_sub(head_budget + marker.len());
    if tail_budget == 0 && marker.len() >= MODEL_FORMAT_MAX_BYTES {
        // Degenerate case: marker alone exceeds budget; return a clipped marker
        return take_bytes_at_char_boundary(&marker, MODEL_FORMAT_MAX_BYTES).to_string();
    }
    if tail_budget == 0 {
        // Make room for the marker by shrinking head
        head_budget = MODEL_FORMAT_MAX_BYTES.saturating_sub(marker.len());
    }

    // Enforce line-count cap by trimming head/tail lines
    let head_lines_text = head_block;
    let tail_lines_text = tail_block;
    // Build final string respecting byte budgets
    let head_part = take_bytes_at_char_boundary(&head_lines_text, head_budget);
    let mut result = String::with_capacity(MODEL_FORMAT_MAX_BYTES.min(s.len()));

    result.push_str(head_part);
    result.push_str(&marker);

    let remaining = MODEL_FORMAT_MAX_BYTES.saturating_sub(result.len());
    let tail_budget_final = remaining;
    let tail_part = take_last_bytes_at_char_boundary(&tail_lines_text, tail_budget_final);
    result.push_str(tail_part);

    result
}

// Truncate a &str to a byte budget at a char boundary (prefix)
#[inline]
fn take_bytes_at_char_boundary(s: &str, maxb: usize) -> &str {
    if s.len() <= maxb {
        return s;
    }
    let mut last_ok = 0;
    for (i, ch) in s.char_indices() {
        let nb = i + ch.len_utf8();
        if nb > maxb {
            break;
        }
        last_ok = nb;
    }
    &s[..last_ok]
}

// Take a suffix of a &str within a byte budget at a char boundary
#[inline]
fn take_last_bytes_at_char_boundary(s: &str, maxb: usize) -> &str {
    if s.len() <= maxb {
        return s;
    }
    let mut start = s.len();
    let mut used = 0usize;
    for (i, ch) in s.char_indices().rev() {
        let nb = ch.len_utf8();
        if used + nb > maxb {
            break;
        }
        start = i;
        used += nb;
        if start == 0 {
            break;
        }
    }
    &s[start..]
}

/// Exec output is a pre-serialized JSON payload
fn format_exec_output(exec_output: &ExecToolCallOutput) -> String {
    let ExecToolCallOutput {
        exit_code,
        duration,
        ..
    } = exec_output;

    #[derive(Serialize)]
    struct ExecMetadata {
        exit_code: i32,
        duration_seconds: f32,
    }

    #[derive(Serialize)]
    struct ExecOutput<'a> {
        output: &'a str,
        metadata: ExecMetadata,
    }

    // round to 1 decimal place
    let duration_seconds = ((duration.as_secs_f32()) * 10.0).round() / 10.0;

    let formatted_output = format_exec_output_str(exec_output);

    let payload = ExecOutput {
        output: &formatted_output,
        metadata: ExecMetadata {
            exit_code: *exit_code,
            duration_seconds,
        },
    };

    #[expect(clippy::expect_used)]
    serde_json::to_string(&payload).expect("serialize ExecOutput")
}

pub(super) fn get_last_assistant_message_from_turn(responses: &[ResponseItem]) -> Option<String> {
    responses.iter().rev().find_map(|item| {
        if let ResponseItem::Message { role, content, .. } = item {
            if role == "assistant" {
                content.iter().rev().find_map(|ci| {
                    if let ContentItem::OutputText { text } = ci {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
            } else {
                None
            }
        } else {
            None
        }
    })
}
fn convert_call_tool_result_to_function_call_output_payload(
    call_tool_result: &CallToolResult,
) -> FunctionCallOutputPayload {
    let CallToolResult {
        content,
        is_error,
        structured_content,
    } = call_tool_result;

    // In terms of what to send back to the model, we prefer structured_content,
    // if available, and fallback to content, otherwise.
    let mut is_success = is_error != &Some(true);
    let content = if let Some(structured_content) = structured_content
        && structured_content != &serde_json::Value::Null
        && let Ok(serialized_structured_content) = serde_json::to_string(&structured_content)
    {
        serialized_structured_content
    } else {
        match serde_json::to_string(&content) {
            Ok(serialized_content) => serialized_content,
            Err(err) => {
                // If we could not serialize either content or structured_content to
                // JSON, flag this as an error.
                is_success = false;
                err.to_string()
            }
        }
    };

    FunctionCallOutputPayload {
        content,
        success: Some(is_success),
    }
}

/// Emits an ExitedReviewMode Event with optional ReviewOutput.
async fn exit_review_mode(
    session: Arc<Session>,
    task_sub_id: String,
    review_output: Option<ReviewOutputEvent>,
) {
    let event = Event {
        id: task_sub_id,
        msg: EventMsg::ExitedReviewMode(ExitedReviewModeEvent { review_output }),
    };
    session.send_event(event).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ConfigOverrides;
    use crate::config::ConfigToml;
    use crate::protocol::CompactedItem;
    use crate::protocol::InitialHistory;
    use crate::protocol::ResumedHistory;
    use codex_protocol::models::ContentItem;
    use mcp_types::ContentBlock;
    use mcp_types::TextContent;
    use pretty_assertions::assert_eq;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration as StdDuration;

    #[test]
    fn reconstruct_history_matches_live_compactions() {
        let (session, turn_context) = make_session_and_context();
        let (rollout_items, expected) = sample_rollout(&session, &turn_context);

        let reconstructed = session.reconstruct_history_from_rollout(&turn_context, &rollout_items);

        assert_eq!(expected, reconstructed);
    }

    #[test]
    fn record_initial_history_reconstructs_resumed_transcript() {
        let (session, turn_context) = make_session_and_context();
        let (rollout_items, expected) = sample_rollout(&session, &turn_context);

        tokio_test::block_on(session.record_initial_history(
            &turn_context,
            InitialHistory::Resumed(ResumedHistory {
                conversation_id: ConversationId::default(),
                history: rollout_items,
                rollout_path: PathBuf::from("/tmp/resume.jsonl"),
            }),
        ));

        let actual = session.state.lock_unchecked().history.contents();
        assert_eq!(expected, actual);
    }

    #[test]
    fn record_initial_history_reconstructs_forked_transcript() {
        let (session, turn_context) = make_session_and_context();
        let (rollout_items, expected) = sample_rollout(&session, &turn_context);

        tokio_test::block_on(
            session.record_initial_history(&turn_context, InitialHistory::Forked(rollout_items)),
        );

        let actual = session.state.lock_unchecked().history.contents();
        assert_eq!(expected, actual);
    }

    #[test]
    fn prefers_structured_content_when_present() {
        let ctr = CallToolResult {
            // Content present but should be ignored because structured_content is set.
            content: vec![text_block("ignored")],
            is_error: None,
            structured_content: Some(json!({
                "ok": true,
                "value": 42
            })),
        };

        let got = convert_call_tool_result_to_function_call_output_payload(&ctr);
        let expected = FunctionCallOutputPayload {
            content: serde_json::to_string(&json!({
                "ok": true,
                "value": 42
            }))
            .unwrap(),
            success: Some(true),
        };

        assert_eq!(expected, got);
    }

    #[test]
    fn model_truncation_head_tail_by_lines() {
        // Build 400 short lines so line-count limit, not byte budget, triggers truncation
        let lines: Vec<String> = (1..=400).map(|i| format!("line{i}")).collect();
        let full = lines.join("\n");

        let exec = ExecToolCallOutput {
            exit_code: 0,
            stdout: StreamOutput::new(String::new()),
            stderr: StreamOutput::new(String::new()),
            aggregated_output: StreamOutput::new(full),
            duration: StdDuration::from_secs(1),
            timed_out: false,
        };

        let out = format_exec_output_str(&exec);

        // Expect elision marker with correct counts
        let omitted = 400 - MODEL_FORMAT_MAX_LINES; // 144
        let marker = format!("\n[... omitted {omitted} of 400 lines ...]\n\n");
        assert!(out.contains(&marker), "missing marker: {out}");

        // Validate head and tail
        let parts: Vec<&str> = out.split(&marker).collect();
        assert_eq!(parts.len(), 2, "expected one marker split");
        let head = parts[0];
        let tail = parts[1];

        let expected_head: String = (1..=MODEL_FORMAT_HEAD_LINES)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(head.starts_with(&expected_head), "head mismatch");

        let expected_tail: String = ((400 - MODEL_FORMAT_TAIL_LINES + 1)..=400)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(tail.ends_with(&expected_tail), "tail mismatch");
    }

    #[test]
    fn model_truncation_respects_byte_budget() {
        // Construct a large output (about 100kB) so byte budget dominates
        let big_line = "x".repeat(100);
        let full = std::iter::repeat_n(big_line, 1000)
            .collect::<Vec<_>>()
            .join("\n");

        let exec = ExecToolCallOutput {
            exit_code: 0,
            stdout: StreamOutput::new(String::new()),
            stderr: StreamOutput::new(String::new()),
            aggregated_output: StreamOutput::new(full.clone()),
            duration: StdDuration::from_secs(1),
            timed_out: false,
        };

        let out = format_exec_output_str(&exec);
        assert!(out.len() <= MODEL_FORMAT_MAX_BYTES, "exceeds byte budget");
        assert!(out.contains("omitted"), "should contain elision marker");

        // Ensure head and tail are drawn from the original
        assert!(full.starts_with(out.chars().take(8).collect::<String>().as_str()));
        assert!(
            full.ends_with(
                out.chars()
                    .rev()
                    .take(8)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect::<String>()
                    .as_str()
            )
        );
    }

    #[test]
    fn includes_timed_out_message() {
        let exec = ExecToolCallOutput {
            exit_code: 0,
            stdout: StreamOutput::new(String::new()),
            stderr: StreamOutput::new(String::new()),
            aggregated_output: StreamOutput::new("Command output".to_string()),
            duration: StdDuration::from_secs(1),
            timed_out: true,
        };

        let out = format_exec_output_str(&exec);

        assert_eq!(
            out,
            "command timed out after 1000 milliseconds\nCommand output"
        );
    }

    #[test]
    fn falls_back_to_content_when_structured_is_null() {
        let ctr = CallToolResult {
            content: vec![text_block("hello"), text_block("world")],
            is_error: None,
            structured_content: Some(serde_json::Value::Null),
        };

        let got = convert_call_tool_result_to_function_call_output_payload(&ctr);
        let expected = FunctionCallOutputPayload {
            content: serde_json::to_string(&vec![text_block("hello"), text_block("world")])
                .unwrap(),
            success: Some(true),
        };

        assert_eq!(expected, got);
    }

    #[test]
    fn success_flag_reflects_is_error_true() {
        let ctr = CallToolResult {
            content: vec![text_block("unused")],
            is_error: Some(true),
            structured_content: Some(json!({ "message": "bad" })),
        };

        let got = convert_call_tool_result_to_function_call_output_payload(&ctr);
        let expected = FunctionCallOutputPayload {
            content: serde_json::to_string(&json!({ "message": "bad" })).unwrap(),
            success: Some(false),
        };

        assert_eq!(expected, got);
    }

    #[test]
    fn success_flag_true_with_no_error_and_content_used() {
        let ctr = CallToolResult {
            content: vec![text_block("alpha")],
            is_error: Some(false),
            structured_content: None,
        };

        let got = convert_call_tool_result_to_function_call_output_payload(&ctr);
        let expected = FunctionCallOutputPayload {
            content: serde_json::to_string(&vec![text_block("alpha")]).unwrap(),
            success: Some(true),
        };

        assert_eq!(expected, got);
    }

    fn text_block(s: &str) -> ContentBlock {
        ContentBlock::TextContent(TextContent {
            annotations: None,
            text: s.to_string(),
            r#type: "text".to_string(),
        })
    }

    fn make_session_and_context() -> (Session, TurnContext) {
        let (tx_event, _rx_event) = async_channel::unbounded();
        let codex_home = tempfile::tempdir().expect("create temp dir");
        let config = Config::load_from_base_config_with_overrides(
            ConfigToml::default(),
            ConfigOverrides::default(),
            codex_home.path().to_path_buf(),
        )
        .expect("load default test config");
        let config = Arc::new(config);
        let conversation_id = ConversationId::default();
        let client = ModelClient::new(
            config.clone(),
            None,
            config.model_provider.clone(),
            config.model_reasoning_effort,
            config.model_reasoning_summary,
            conversation_id,
        );
        let tools_config = ToolsConfig::new(&ToolsConfigParams {
            model_family: &config.model_family,
            approval_policy: config.approval_policy,
            sandbox_policy: config.sandbox_policy.clone(),
            include_plan_tool: config.include_plan_tool,
            include_apply_patch_tool: config.include_apply_patch_tool,
            include_web_search_request: config.tools_web_search_request,
            use_streamable_shell_tool: config.use_experimental_streamable_shell_tool,
            include_view_image_tool: config.include_view_image_tool,
            experimental_unified_exec_tool: config.use_experimental_unified_exec_tool,
        });
        let turn_context = TurnContext {
            client,
            cwd: config.cwd.clone(),
            base_instructions: config.base_instructions.clone(),
            user_instructions: config.user_instructions.clone(),
            approval_policy: config.approval_policy,
            sandbox_policy: config.sandbox_policy.clone(),
            shell_environment_policy: config.shell_environment_policy.clone(),
            tools_config,
            is_review_mode: false,
        };
        let session = Session {
            conversation_id,
            tx_event,
            mcp_connection_manager: McpConnectionManager::default(),
            session_manager: ExecSessionManager::default(),
            unified_exec_manager: UnifiedExecSessionManager::default(),
            notify: None,
            rollout: Mutex::new(None),
            working_set_test_catalog: None,
            state: Mutex::new(State {
                history: ConversationHistory::new(),
                ..Default::default()
            }),
            codex_linux_sandbox_exe: None,
            user_shell: shell::Shell::Unknown,
            show_raw_agent_reasoning: config.show_raw_agent_reasoning,
        };
        (session, turn_context)
    }

    fn sample_rollout(
        session: &Session,
        turn_context: &TurnContext,
    ) -> (Vec<RolloutItem>, Vec<ResponseItem>) {
        let mut rollout_items = Vec::new();
        let mut live_history = ConversationHistory::new();

        let initial_context = session.build_initial_context(turn_context);
        for item in &initial_context {
            rollout_items.push(RolloutItem::ResponseItem(item.clone()));
        }
        live_history.record_items(initial_context.iter());

        let user1 = ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "first user".to_string(),
            }],
        };
        live_history.record_items(std::iter::once(&user1));
        rollout_items.push(RolloutItem::ResponseItem(user1.clone()));

        let assistant1 = ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: "assistant reply one".to_string(),
            }],
        };
        live_history.record_items(std::iter::once(&assistant1));
        rollout_items.push(RolloutItem::ResponseItem(assistant1.clone()));

        let summary1 = "summary one";
        let snapshot1 = live_history.contents();
        let user_messages1 = collect_user_messages(&snapshot1);
        let rebuilt1 = build_compacted_history(
            session.build_initial_context(turn_context),
            &user_messages1,
            summary1,
        );
        live_history.replace(rebuilt1);
        rollout_items.push(RolloutItem::Compacted(CompactedItem {
            message: summary1.to_string(),
        }));

        let user2 = ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "second user".to_string(),
            }],
        };
        live_history.record_items(std::iter::once(&user2));
        rollout_items.push(RolloutItem::ResponseItem(user2.clone()));

        let assistant2 = ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: "assistant reply two".to_string(),
            }],
        };
        live_history.record_items(std::iter::once(&assistant2));
        rollout_items.push(RolloutItem::ResponseItem(assistant2.clone()));

        let summary2 = "summary two";
        let snapshot2 = live_history.contents();
        let user_messages2 = collect_user_messages(&snapshot2);
        let rebuilt2 = build_compacted_history(
            session.build_initial_context(turn_context),
            &user_messages2,
            summary2,
        );
        live_history.replace(rebuilt2);
        rollout_items.push(RolloutItem::Compacted(CompactedItem {
            message: summary2.to_string(),
        }));

        let user3 = ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "third user".to_string(),
            }],
        };
        live_history.record_items(std::iter::once(&user3));
        rollout_items.push(RolloutItem::ResponseItem(user3.clone()));

        let assistant3 = ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: "assistant reply three".to_string(),
            }],
        };
        live_history.record_items(std::iter::once(&assistant3));
        rollout_items.push(RolloutItem::ResponseItem(assistant3.clone()));

        (rollout_items, live_history.contents())
    }

    /// Synthetic catalog: structural test data only, not any provider's price.
    fn synthetic_pricing_catalog(model: &str) -> working_set::PricingCatalog {
        working_set::PricingCatalog {
            catalog_version: "synthetic-test-0".to_string(),
            model: model.to_string(),
            provider: "synthetic".to_string(),
            currency: "USD".to_string(),
            unit: "per_1m_tokens".to_string(),
            price_source: Some("synthetic://test-only".to_string()),
            fetched_at: Some("2026-09-18T00:00:00Z".to_string()),
            freshness: working_set::Freshness::Fresh,
            uncached_input_rate: Some(1.0),
            cached_input_rate: Some(0.05),
            cache_write_rate: Some(2.0),
            output_rate: Some(1.0),
            reasoning_in_output: Some(true),
            min_cacheable_prefix_tokens: Some(1),
            cache_eligibility_known: true,
            long_context: None,
            unknown_fields: Vec::new(),
        }
    }

    /// The production hook resolves one exact recorded reference-list record and
    /// nothing else. Rate values are asserted only where they are recorded (the
    /// catalog itself); this test pins identity, provenance and basis.
    #[test]
    fn recorded_pricing_catalog_is_exact_luna_reference_list_only() {
        let catalog = recorded_pricing_catalog("gpt-5.6-luna").expect("luna record");
        assert_eq!(catalog.catalog_version, "openrouter-2026-09-19-5ecb4fe3");
        assert_eq!(catalog.model, "gpt-5.6-luna");
        assert_eq!(catalog.provider, "openrouter");
        assert_eq!(catalog.currency, "USD");
        assert_eq!(catalog.unit, "per_1m_tokens");
        // The immutable record claims no freshness of its own; the resolver
        // validates it against the fixed TTL (see the freshness test below).
        assert_eq!(catalog.freshness, working_set::Freshness::Unknown);
        // The snapshot does not record whether reasoning is billed as output, so
        // no derivation is guessed.
        assert_eq!(catalog.reasoning_in_output, None);
        assert_eq!(
            catalog.fetched_at.as_deref(),
            Some("2026-09-19T00:12:35.624Z")
        );
        let source = catalog
            .price_source
            .as_deref()
            .expect("recorded price source");
        for expected in [
            "basis=openrouter_reference_list",
            "https://openrouter.ai/api/v1/models",
            "revision=openrouter-2026-09-19-5ecb4fe3",
            "date=2026-09-19T00:12:35.624Z",
            "body_sha256=5ecb4fe389783c8f2bf49772ae6d405e230172f26401bc9ff06f45ad4c70ab74",
            "docs_schema_sha256=e973433327496ad6b12720c85c9c78cc28ee94b8af01c0ac33ad5ca120792a7c",
        ] {
            assert!(source.contains(expected), "price source lacks {expected}");
        }
        assert!(catalog.unknown_fields.is_empty());
        // Every rate the selector consumes is populated, so the empty
        // `unknown_fields` does not hide a missing rate.
        assert!(catalog.uncached_input_rate.is_some());
        assert!(catalog.cached_input_rate.is_some());
        assert!(catalog.cache_write_rate.is_some());
        assert!(catalog.output_rate.is_some());
        assert!(catalog.cache_eligibility_known);
        assert_eq!(
            catalog.long_context.map(|rule| rule.threshold_tokens),
            Some(272_000)
        );
        assert!(
            catalog
                .long_context
                .is_some_and(|rule| rule.term_rates.is_some()),
            "the recorded tier prices terms separately"
        );
        // Exact string equality only: aliases, neighboring variants and other
        // slugs (including gpt-6-astra) must not resolve a record.
        for model in [
            "gpt-6-astra",
            "openai/gpt-5.6-luna",
            "gpt-5.6-luna-pro",
            "gpt-5.6-luna:batch",
            "gpt-5.6-luna-20260709",
            "gpt-5.6",
            "",
        ] {
            assert!(
                recorded_pricing_catalog(model).is_none(),
                "{model} must not resolve a recorded catalog"
            );
        }
    }

    /// The recorded snapshot is fresh at its own recorded date and stale once it
    /// is older than the fixed 7-day reference TTL, future-dated, unreadable or
    /// missing.
    #[test]
    fn reference_price_freshness_uses_the_fixed_seven_day_ttl() {
        let now = DateTime::parse_from_rfc3339("2026-09-19T00:12:35.624Z")
            .expect("rfc3339 timestamp")
            .with_timezone(&Utc);
        assert_eq!(
            reference_price_freshness(Some("2026-09-19T00:12:35.624Z"), now),
            working_set::Freshness::Fresh
        );
        // Exactly seven days old is still inside the TTL; one second older is not.
        assert_eq!(
            reference_price_freshness(Some("2026-09-12T00:12:35.624Z"), now),
            working_set::Freshness::Fresh
        );
        assert_eq!(
            reference_price_freshness(Some("2026-09-12T00:12:34.624Z"), now),
            working_set::Freshness::Stale
        );
        // Future-dated, unreadable and missing fetch dates cannot be verified.
        assert_eq!(
            reference_price_freshness(Some("2026-09-19T00:12:35.625Z"), now),
            working_set::Freshness::Stale
        );
        assert_eq!(
            reference_price_freshness(Some("not-a-date"), now),
            working_set::Freshness::Stale
        );
        assert_eq!(
            reference_price_freshness(None, now),
            working_set::Freshness::Stale
        );
    }

    /// Regression on the production resolution path: the real recorded Luna
    /// reference record (no synthetic injection) must resolve fresh and cost a
    /// real selection, proving the production lookup does not always fall back.
    ///
    /// "Now" is derived from the recorded snapshot date, so the regression is
    /// deterministic and never depends on wall-clock time. The unrecorded
    /// `min_cacheable_prefix_tokens` is never needed here: the changed-epoch path
    /// bills the recorded cache-write rate instead of cached tokens.
    #[test]
    fn resolved_luna_record_applies_a_real_projection_without_a_synthetic_catalog() {
        let fetched_at = recorded_pricing_catalog("gpt-5.6-luna")
            .expect("recorded luna record")
            .fetched_at
            .expect("recorded fetch date");
        let now = DateTime::parse_from_rfc3339(&fetched_at)
            .expect("recorded fetch date is rfc3339")
            .with_timezone(&Utc)
            + chrono::Duration::hours(1);
        let catalog = resolved_pricing_catalog("gpt-5.6-luna", now).expect("resolved catalog");
        assert_eq!(catalog.freshness, working_set::Freshness::Fresh);

        let user = |text: &str| ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: text.to_string(),
            }],
        };
        let canonical = vec![
            user("earlier note about src/legacy.rs"),
            user("unrelated chatter about lunch"),
            user("now fix src/legacy.rs"),
        ];
        let usage = working_set::UsageEvidence {
            input_tokens: 1_000,
            cached_input_tokens: 0,
            output_tokens: 50,
        };
        let frozen = working_set::freeze(&working_set::FreezeInputs {
            model: "gpt-5.6-luna".to_string(),
            reasoning_effort: "none".to_string(),
            instructions_digest: working_set::digest_text("INSTRUCTIONS_TEXT"),
            tools_digest: working_set::digest_serialized(&json!([])),
            canonical: &canonical,
            current_request_start: 2,
            explicit_pins: &[],
            catalog: Some(&catalog),
            usage: Some(&usage),
            previous_epoch_id: None,
            horizon: working_set::Horizon::default(),
        });
        let manifest = frozen.manifest();
        assert_eq!(manifest.cost.confidence, working_set::Confidence::High);
        assert!(manifest.cost.unknown_fields.is_empty());
        assert!(manifest.decision.applied);
        assert_eq!(manifest.decision.reason, "lower_expected_cost");
        assert!(!manifest.fallback.used);
        // The projection really shrinks the wire input.
        assert!(manifest.projection.len < manifest.canonical.log_len);
        match frozen.derived_input(&canonical) {
            working_set::DerivedInput::Projected(items) => {
                assert!(items.len() < canonical.len());
            }
            other => panic!("expected a projected input, got {other:?}"),
        }
        // The amount is computed from the recorded rates, not a fallback: the
        // changed epoch bills the recorded cache-write rate and the output term
        // uses the recorded output rate.
        let baseline = manifest
            .cost
            .baseline
            .breakdown
            .expect("recorded catalog is costed");
        assert_eq!(baseline.long_context_multiplier, 1.0);
        assert_eq!(
            baseline.cache_write,
            manifest.cost.baseline.tokens.cache_write as f64 / 1_000_000.0 * 0.25
        );
        assert_eq!(baseline.output, 50.0 / 1_000_000.0 * 1.2);
        assert_eq!(baseline.total, baseline.cache_write + baseline.output);
    }

    /// Real core-loop integration test at the existing mock-server seam.
    ///
    /// A synthetic high-confidence catalog is injected through the internal
    /// session constructor only, then the actual `run_task` path must:
    /// (1) reduce the wire input for the second user request, (2) leave the
    /// canonical history identical (omitted items are still recorded), (3) carry
    /// the fresh tool continuation verbatim on the follow-up turn, (4) never
    /// re-dispatch a historical tool call, and (5) persist one structural
    /// selection record per user request plus one record per model turn in the
    /// append-only sidecar.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn working_set_projection_reduces_wire_and_keeps_canonical_history() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::atomic::Ordering;

        use crate::auth::AuthManager;
        use crate::auth::CodexAuth;
        use codex_protocol::models::FunctionCallOutputPayload;
        use wiremock::Mock;
        use wiremock::MockServer;
        use wiremock::Respond;
        use wiremock::ResponseTemplate;
        use wiremock::matchers::method;
        use wiremock::matchers::path;

        fn sse(events: Vec<serde_json::Value>) -> String {
            let mut body = String::new();
            for event in events {
                let kind = event["type"].as_str().expect("sse event type");
                body.push_str(&format!("event: {kind}\n"));
                body.push_str(&format!("data: {event}\n\n"));
            }
            body
        }

        fn completed(id: &str, input_tokens: u64) -> serde_json::Value {
            json!({
                "type": "response.completed",
                "response": {
                    "id": id,
                    "usage": {
                        "input_tokens": input_tokens,
                        "input_tokens_details": null,
                        "output_tokens": 10,
                        "output_tokens_details": null,
                        "total_tokens": input_tokens + 10
                    }
                }
            })
        }

        fn assistant_message(id: &str, text: &str) -> serde_json::Value {
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "id": id,
                    "content": [{"type": "output_text", "text": text}]
                }
            })
        }

        fn function_call(call_id: &str, name: &str, arguments: &str) -> serde_json::Value {
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments
                }
            })
        }

        #[derive(Clone)]
        struct SequencedSseResponder {
            bodies: Arc<Vec<String>>,
            requests: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
            calls: Arc<AtomicUsize>,
        }

        impl Respond for SequencedSseResponder {
            fn respond(&self, request: &wiremock::Request) -> ResponseTemplate {
                let index = self.calls.fetch_add(1, Ordering::SeqCst);
                let body: serde_json::Value =
                    serde_json::from_slice(&request.body).expect("request body is JSON");
                self.requests.lock().expect("requests lock").push(body);
                let response = self
                    .bodies
                    .get(index)
                    .unwrap_or_else(|| panic!("unexpected model request {index}"))
                    .clone();
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_raw(response, "text/event-stream")
            }
        }

        // Request 1 finishes with a message (and records token usage); request 2
        // asks for the plan tool; request 3 is the follow-up turn.
        let responder = SequencedSseResponder {
            bodies: Arc::new(vec![
                sse(vec![
                    assistant_message("m1", "FIRST_REPLY"),
                    completed("r1", 1_000),
                ]),
                sse(vec![
                    function_call(
                        "call_fresh",
                        "update_plan",
                        r#"{"plan":[{"step":"wire step","status":"in_progress"}]}"#,
                    ),
                    completed("r2", 1_100),
                ]),
                sse(vec![
                    assistant_message("m3", "SECOND_REPLY"),
                    completed("r3", 1_200),
                ]),
            ]),
            requests: Arc::new(std::sync::Mutex::new(Vec::new())),
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(responder.clone())
            .mount(&server)
            .await;

        let codex_home = tempfile::tempdir().expect("codex home");
        let cwd = tempfile::tempdir().expect("cwd");
        let mut config = Config::load_from_base_config_with_overrides(
            ConfigToml::default(),
            ConfigOverrides::default(),
            codex_home.path().to_path_buf(),
        )
        .expect("load default test config");
        config.cwd = cwd.path().to_path_buf();
        config.model_provider = ModelProviderInfo {
            base_url: Some(format!("{}/v1", server.uri())),
            ..crate::built_in_model_providers()["openai"].clone()
        };
        config.include_plan_tool = true;
        config.user_instructions = None;
        let config = Arc::new(config);
        let catalog = synthetic_pricing_catalog(&config.model);

        let conversation_id = ConversationId::default();
        let rollout_recorder =
            RolloutRecorder::new(&config, RolloutRecorderParams::new(conversation_id, None))
                .await
                .expect("create rollout recorder");
        let rollout_path = rollout_recorder.get_rollout_path();

        let client = ModelClient::new(
            config.clone(),
            Some(AuthManager::from_auth_for_testing(CodexAuth::from_api_key(
                "Test API Key",
            ))),
            config.model_provider.clone(),
            config.model_reasoning_effort,
            config.model_reasoning_summary,
            conversation_id,
        );
        let tools_config = ToolsConfig::new(&ToolsConfigParams {
            model_family: &config.model_family,
            approval_policy: config.approval_policy,
            sandbox_policy: config.sandbox_policy.clone(),
            include_plan_tool: config.include_plan_tool,
            include_apply_patch_tool: config.include_apply_patch_tool,
            include_web_search_request: config.tools_web_search_request,
            use_streamable_shell_tool: config.use_experimental_streamable_shell_tool,
            include_view_image_tool: config.include_view_image_tool,
            experimental_unified_exec_tool: config.use_experimental_unified_exec_tool,
        });
        let turn_context = Arc::new(TurnContext {
            client,
            cwd: config.cwd.clone(),
            base_instructions: config.base_instructions.clone(),
            user_instructions: config.user_instructions.clone(),
            approval_policy: config.approval_policy,
            sandbox_policy: config.sandbox_policy.clone(),
            shell_environment_policy: config.shell_environment_policy.clone(),
            tools_config,
            is_review_mode: false,
        });
        let (tx_event, rx_event) = async_channel::unbounded();
        let session = Arc::new(Session {
            conversation_id,
            tx_event,
            mcp_connection_manager: McpConnectionManager::default(),
            session_manager: ExecSessionManager::default(),
            unified_exec_manager: UnifiedExecSessionManager::default(),
            notify: None,
            rollout: Mutex::new(Some(rollout_recorder)),
            working_set_test_catalog: Some(catalog),
            state: Mutex::new(State {
                history: ConversationHistory::new(),
                ..Default::default()
            }),
            codex_linux_sandbox_exe: None,
            user_shell: shell::Shell::Unknown,
            show_raw_agent_reasoning: false,
        });
        session
            .record_initial_history(&turn_context, InitialHistory::New)
            .await;
        // Canonical history: one anchored message, one unanchored exchange and one
        // historical (already completed) tool pair that must never run again.
        let seeded_history = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "the earlier note mentions src/legacy.rs".to_string(),
                }],
            },
            ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "noted".to_string(),
                }],
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "update_plan".to_string(),
                arguments: r#"{"plan":[{"step":"historical step","status":"completed"}]}"#
                    .to_string(),
                call_id: "call_hist".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call_hist".to_string(),
                output: FunctionCallOutputPayload {
                    content: "Plan updated".to_string(),
                    success: Some(true),
                },
            },
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "unrelated chatter about lunch".to_string(),
                }],
            },
            ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "ok".to_string(),
                }],
            },
        ];
        session.record_conversation_items(&seeded_history).await;

        // Request 1 anchors `src/legacy.rs` and establishes provider usage
        // evidence; its own selection still falls back because no usage is
        // recorded yet.
        run_task(
            session.clone(),
            turn_context.clone(),
            "sub-1".to_string(),
            vec![InputItem::Text {
                text: "first request about src/legacy.rs".to_string(),
            }],
        )
        .await;
        // Request 2 anchors `src/legacy.rs` and runs one tool turn.
        run_task(
            session.clone(),
            turn_context.clone(),
            "sub-2".to_string(),
            vec![InputItem::Text {
                text: "now fix src/legacy.rs".to_string(),
            }],
        )
        .await;

        let requests = responder.requests.lock().expect("requests lock").clone();
        assert_eq!(
            requests.len(),
            3,
            "historical tool calls must not be re-dispatched"
        );
        let request_1_input = requests[0]["input"].as_array().expect("request 1 input");
        let request_2_input = requests[1]["input"].as_array().expect("request 2 input");
        let follow_up_input = requests[2]["input"].as_array().expect("request 3 input");
        assert!(
            request_2_input.len() < request_1_input.len(),
            "wire input must be reduced relative to canonical history"
        );
        assert!(
            request_2_input
                .iter()
                .any(|item| item.to_string().contains("src/legacy.rs")),
            "anchored history must be retained"
        );
        assert!(
            !request_2_input
                .iter()
                .any(|item| item.to_string().contains("unrelated chatter about lunch")),
            "unanchored history must be omitted from the wire"
        );
        assert!(
            !request_2_input
                .iter()
                .any(|item| item.to_string().contains("call_hist")),
            "historical calls must not be re-sent as pending work"
        );
        assert!(
            follow_up_input.len() > request_2_input.len(),
            "the fresh tool continuation is appended to the projection"
        );
        let fresh_call = &follow_up_input[follow_up_input.len() - 2];
        let fresh_output = &follow_up_input[follow_up_input.len() - 1];
        assert_eq!(fresh_call["type"], json!("function_call"));
        assert_eq!(fresh_call["call_id"], json!("call_fresh"));
        assert!(
            fresh_call["arguments"]
                .as_str()
                .expect("arguments")
                .contains("wire step")
        );
        assert_eq!(fresh_output["type"], json!("function_call_output"));
        assert_eq!(fresh_output["call_id"], json!("call_fresh"));
        assert!(
            !follow_up_input
                .iter()
                .any(|item| item.to_string().contains("call_hist")),
            "historical calls must not reappear after the overlay"
        );

        // Canonical history is untouched: every omitted item is still recorded.
        let canonical_history = session.turn_input_with_history(Vec::new());
        assert_eq!(canonical_history.len(), 13);
        assert!(canonical_history.iter().any(|item| {
            serde_json::to_string(item)
                .expect("item json")
                .contains("unrelated chatter about lunch")
        }));
        assert!(canonical_history.iter().any(|item| {
            serde_json::to_string(item)
                .expect("item json")
                .contains("call_hist")
        }));
        let mut plan_updates = 0;
        while let Ok(event) = rx_event.try_recv() {
            if matches!(event.msg, EventMsg::PlanUpdate(_)) {
                plan_updates += 1;
            }
        }
        assert_eq!(plan_updates, 1, "only the fresh tool call may execute");

        // Sidecar contract: one selection record per user request, one turn record
        // per model turn, structural digests only.
        let sidecar_path = working_set::selection_sidecar_path(&rollout_path);
        let sidecar = std::fs::read_to_string(&sidecar_path).expect("selection sidecar exists");
        let records: Vec<serde_json::Value> = sidecar
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("sidecar line is JSON"))
            .collect();
        let selections: Vec<&serde_json::Value> = records
            .iter()
            .filter(|record| record["record"] == json!("selection"))
            .collect();
        let turns: Vec<&serde_json::Value> = records
            .iter()
            .filter(|record| record["record"] == json!("turn"))
            .collect();
        assert_eq!(selections.len(), 2, "one selection record per user request");
        assert_eq!(turns.len(), 3, "one turn record per model turn");
        assert_eq!(selections[0]["request_index"], json!(1));
        assert_eq!(selections[0]["applied"], json!(false));
        assert_eq!(selections[0]["fallback_reason"], json!("pricing_unknown"));
        assert_eq!(selections[1]["request_index"], json!(2));
        assert_eq!(selections[1]["applied"], json!(true));
        assert_eq!(
            selections[1]["canonical_digest_after_selection"],
            selections[1]["manifest"]["canonical"]["digest"]
        );
        assert_ne!(
            selections[1]["canonical_digest_before_request"],
            selections[1]["canonical_digest_after_selection"]
        );
        assert_eq!(turns[0]["request_index"], json!(1));
        assert_eq!(turns[0]["decision"], json!("canonical"));
        assert_eq!(turns[0]["reason"], json!("pricing_unknown"));
        assert_eq!(turns[1]["request_index"], json!(2));
        assert_eq!(turns[1]["decision"], json!("projected"));
        assert!(
            turns[1]["wire_len"].as_u64().expect("wire_len")
                < turns[1]["canonical_len"].as_u64().expect("canonical_len")
        );
        assert_eq!(turns[2]["turn_index"], json!(2));
        assert_eq!(turns[2]["decision"], json!("projected"));
        for private in [
            "src/legacy.rs",
            "unrelated chatter about lunch",
            "historical step",
            "wire step",
        ] {
            assert!(
                !sidecar.contains(private),
                "selection sidecar leaked private value {private}"
            );
        }
    }
}
