//! Per-user-request working-set selection (compact-on-demand, module `m01-client`).
//!
//! The canonical conversation history is append-only and remains the source of
//! truth. This module derives, **once per user request**, a projection of the
//! history that keeps what the request needs and omits the rest, plus a
//! structural manifest. Runtime integration is in `codex.rs`:
//!
//! * the projection is frozen before the first model turn of a user request,
//! * internal tool follow-up turns reuse the frozen projection and append the
//!   fresh stream-recorded items (`active_tool_chain` overlay) verbatim,
//! * selection never dispatches a tool; dispatch stays driven by the live stream.
//!
//! Contract implemented here (M0/M1, corrected by the 2026-09-18 handoff):
//!
//! * relevance first: pins (current request, carried session context, explicit,
//!   closure, opaque variants) and deterministic anchor retrieval form the
//!   candidate; cost may only rank candidates that already satisfy coverage, never
//!   add/remove/reorder items.
//! * no token target or budget: token counts are accounting inputs plus the
//!   explicit "approximately equal cost, fewer tokens" user tie-break only. This
//!   module has no access to the auto-compact limit or the model context window.
//! * pairs are whole: a `function_call`/`custom_tool_call`/`local_shell_call` and
//!   its output are kept or dropped together, and an unresolved historical orphan
//!   falls back instead of being sent unpaired (C6/C9).
//! * conservative fallback: no anchors, non-selective retrieval, changed canonical
//!   prefix, orphaned overlay or unknown/stale pricing/cache data all send the
//!   canonical input unchanged and record a reason. No rate is ever invented.
//! * deterministic: identical semantic inputs produce identical projections, item
//!   ids and manifest digests. No clocks, randomness, map-order dependence, or
//!   provider traffic.
//! * privacy: the manifest and its log line carry structural counts, reason codes,
//!   model slug, item ids and digests only. Item text, anchor values, tool
//!   definitions and instruction text never leave this module.
//!
//! Deliberate scope notes:
//!
//! * The wire projection is applied whenever the catalog inputs are complete and
//!   fresh (`Confidence::High`). Recorded rate data is resolved by
//!   `codex::recorded_pricing_catalog`, which returns the immutable recorded
//!   OpenRouter reference-list catalog for the exact `gpt-5.6-luna` wire slug and
//!   `None` for every other model (M0 §4.1); without a record the selector
//!   records `pricing_unknown` and sends canonical history. Above a recorded
//!   long-context threshold each term is priced with its recorded override and
//!   falls back to that term's base rate when the tier records none; a tier with
//!   only one uniform multiplier scales every term. This is not a product flag.
//! * Structural per-request manifests are also appended to an append-only sidecar
//!   next to the session rollout (see [`selection_sidecar_path`]); the sidecar is
//!   never read back and never replaces canonical history.
//! * Review threads keep their isolated in-memory history and are excluded.
//! * The native auto-compact trigger and `Op::Compact` summarization are
//!   untouched. Implications: when auto-compaction replaces history mid-request the
//!   frozen prefix no longer matches and this path falls back to canonical for the
//!   rest of the request; and an applied projection lowers the provider-reported
//!   input tokens, so the token-limit check observes the projected request.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;

use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use regex_lite::Regex;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;

use crate::protocol::InputMessageKind;

/// Manifest schema version. Matches the M1 selection-manifest contract.
pub const SCHEMA_VERSION: u32 = 2;
/// Manifest kind, as recorded in the M1 contract.
pub const MANIFEST_KIND: &str = "compact-on-demand/selection-manifest";
/// Digest algorithm used for every digest in the manifest and its sidecar.
///
/// SHA-256, from the already-locked `sha2` dependency. Digests are
/// identity/privacy tokens, not signatures.
pub const DIGEST_ALGORITHM: &str = "sha256";

/// Fixed count cap on anchors extracted from the current request (C7 allows a
/// fixed count cap; token-derived caps are forbidden).
const MAX_ANCHORS: usize = 64;
/// Fixed count cap on retrieval hits. Above it retrieval is not selective enough
/// to justify omitting anything, so the request conservatively falls back.
const MAX_RETRIEVAL_HITS: usize = 64;
/// Supported costs within this relative tolerance are "approximately equal" and
/// the explicit user tie-break prefers the candidate with fewer input tokens.
const APPROX_EQUAL_COST_RELATIVE_TOLERANCE: f64 = 0.01;
/// Deterministic token estimate convention shared with `truncate.rs`: 4 bytes/token.
const BYTES_PER_TOKEN_ESTIMATE: u64 = 4;
/// Per-million-token catalog unit.
const CATALOG_UNIT: &str = "per_1m_tokens";
/// No savings may be claimed without provider usage evidence (I15).
const SAVINGS_CLAIM_NONE: &str = "none";

/// Why the canonical input was sent instead of a projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    /// Reserved for the M1 closed set; no Jev client exists in this slice.
    JevError,
    /// Reserved for the M1 closed set; no Jev client exists in this slice.
    JevTimeout,
    /// Reserved for the M1 closed set; no Jev client exists in this slice.
    JevMalformed,
    /// The canonical input could not be represented (e.g. empty input).
    SchemaInvalid,
    /// A historical call/result pair is incomplete or duplicated (C6).
    ClosureViolation,
    /// Reserved for the M1 closed set; nothing here aborts on overflow.
    HardContextOverflow,
    /// Reserved for the M1 closed set; selection stages do not panic.
    InternalError,
    /// A required rate or recorded input is missing; nothing was invented (I10).
    PricingUnknown,
    /// The recorded catalog is stale; no cost-driven change is allowed (I12).
    PricingStale,
    /// Cache eligibility or cached-token evidence is unknown (I13).
    CacheIneligible,
    /// No anchors, no anchor hits, or too many hits to be selective (C7).
    RetrievalInsufficient,
    /// The frozen canonical prefix changed (e.g. auto-compaction rewrote history).
    CanonicalPrefixChanged,
}

impl FallbackReason {
    /// Stable machine-readable reason code (also the manifest / log token).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::JevError => "jev_error",
            Self::JevTimeout => "jev_timeout",
            Self::JevMalformed => "jev_malformed",
            Self::SchemaInvalid => "schema_invalid",
            Self::ClosureViolation => "closure_violation",
            Self::HardContextOverflow => "hard_context_overflow",
            Self::InternalError => "internal_error",
            Self::PricingUnknown => "pricing_unknown",
            Self::PricingStale => "pricing_stale",
            Self::CacheIneligible => "cache_ineligible",
            Self::RetrievalInsufficient => "retrieval_insufficient",
            Self::CanonicalPrefixChanged => "canonical_prefix_changed",
        }
    }
}

/// Why the runtime sends the canonical input for a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalReason {
    /// A selection-stage failure; recorded in `fallback`.
    Fallback(FallbackReason),
    /// Cost comparison legitimately preferred the no-selection baseline.
    BaselinePreferred,
}

impl CanonicalReason {
    /// Stable machine-readable reason code.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fallback(reason) => reason.as_str(),
            Self::BaselinePreferred => "baseline_preferred",
        }
    }

    /// The fallback reason, when this decision is an actual fallback.
    pub fn fallback_reason(self) -> Option<FallbackReason> {
        match self {
            Self::Fallback(reason) => Some(reason),
            Self::BaselinePreferred => None,
        }
    }
}

/// Input to [`freeze`]. Everything is derived from recorded state; no clocks.
#[derive(Debug)]
pub struct FreezeInputs<'a> {
    /// Exact wire model slug for the request.
    pub model: String,
    /// Configured reasoning effort identity (already stringified; `unset` when
    /// the runtime has no explicit effort). Part of the epoch identity.
    pub reasoning_effort: String,
    /// Digest of the effective instruction text. Never the text itself.
    pub instructions_digest: String,
    /// Digest of the effective tool definitions. Never the definitions themselves.
    pub tools_digest: String,
    /// Canonical history including the current request's items.
    pub canonical: &'a [ResponseItem],
    /// Index of the first item that belongs to the current user request.
    pub current_request_start: usize,
    /// Items recorded as explicitly pinned in the canonical log (none at this pin).
    pub explicit_pins: &'a [usize],
    /// Recorded, versioned pricing catalog for this model, when one exists.
    pub catalog: Option<&'a PricingCatalog>,
    /// Provider-reported usage recorded from the previous response, when any.
    pub usage: Option<&'a UsageEvidence>,
    /// Epoch identity of the input actually sent for the previous user request.
    pub previous_epoch_id: Option<&'a str>,
    /// Cost horizon; defaults to the conservative `H1` with `p = 0`.
    pub horizon: Horizon,
}

/// Provider-reported usage of the previous request (recorded data only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsageEvidence {
    /// Provider-reported input tokens of the previous request.
    pub input_tokens: u64,
    /// Provider-reported cached input tokens of the previous request.
    pub cached_input_tokens: u64,
    /// Provider-reported output tokens of the previous request.
    pub output_tokens: u64,
}

impl UsageEvidence {
    /// Uses the recorded provider usage of the previous response.
    pub fn from_token_usage(usage: &crate::protocol::TokenUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            cached_input_tokens: usage.cached_input_tokens,
            output_tokens: usage.output_tokens,
        }
    }
}

/// Recorded freshness of a catalog record. Freshness is validated when the record
/// is recorded; the selector never reads a clock (I5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// Re-verified against its `price_source` when recorded.
    Fresh,
    /// Known stale; cost must not drive selection.
    Stale,
    /// Freshness was never established.
    Unknown,
}

/// Declared long-context price rule from the recorded catalog.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct LongContextRule {
    /// Token count the request's total prompt tokens must strictly exceed for
    /// the tier rates to apply (the recorded condition is `>`, never `>=`).
    pub threshold_tokens: u64,
    /// Multiplier applied to every cost term when the tier applies and no
    /// separate per-term rates are recorded.
    pub multiplier: f64,
    /// Absolute per-term rates of the tier, when the recorded rule prices the
    /// terms separately. Each recorded rate overrides that term's base rate for
    /// requests strictly above the threshold; `None` inherits the base rate.
    pub term_rates: Option<LongContextTermRates>,
}

/// Recorded per-term rates of a long-context tier, in the catalog unit (per 1M
/// tokens). `None` means the term inherits its base rate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct LongContextTermRates {
    /// Rate per 1M uncached input tokens above the threshold.
    pub uncached_input_rate: Option<f64>,
    /// Rate per 1M cached input tokens above the threshold.
    pub cached_input_rate: Option<f64>,
    /// Rate per 1M cache-write tokens above the threshold.
    pub cache_write_rate: Option<f64>,
    /// Rate per 1M output tokens above the threshold.
    pub output_rate: Option<f64>,
}

/// One recorded, immutable pricing catalog record (M0 §4.1).
///
/// `None` rates are unknown, never zero and never free. `cache_write_rate: None`
/// with caching otherwise declared means the provider has no separate cache-write
/// line item; the write is then charged at the known uncached rate.
#[derive(Debug, Clone, PartialEq)]
pub struct PricingCatalog {
    /// Version of the immutable record.
    pub catalog_version: String,
    /// Exact model slug this record applies to; never an alias.
    pub model: String,
    /// Provider identity.
    pub provider: String,
    /// Currency of the rates.
    pub currency: String,
    /// Unit of the rates (`per_1m_tokens`).
    pub unit: String,
    /// Where the rates were read from, when recorded.
    pub price_source: Option<String>,
    /// When the rates were fetched, when recorded.
    pub fetched_at: Option<String>,
    /// Recorded freshness of this record.
    pub freshness: Freshness,
    /// Rate per 1M uncached input tokens.
    pub uncached_input_rate: Option<f64>,
    /// Rate per 1M cached input tokens.
    pub cached_input_rate: Option<f64>,
    /// Rate per 1M cache-write tokens, when the provider has that line item.
    pub cache_write_rate: Option<f64>,
    /// Rate per 1M output tokens (reasoning included when billed as output).
    pub output_rate: Option<f64>,
    /// Whether reasoning tokens are billed as output, when recorded.
    pub reasoning_in_output: Option<bool>,
    /// Minimum cacheable prefix in tokens, when recorded.
    pub min_cacheable_prefix_tokens: Option<u64>,
    /// Whether the cache eligibility rules are recorded and usable.
    pub cache_eligibility_known: bool,
    /// Long-context price rule, when recorded.
    pub long_context: Option<LongContextRule>,
    /// Names of fields this record declares unknown.
    pub unknown_fields: Vec<String>,
}

/// Request horizon for expected cost. The default is conservative: one request
/// (`H1`) and zero reuse probability, so cache writes are never amortized.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Horizon {
    /// Number of requests over which cache writes are amortized (`k` in `H2(k)`).
    pub requests: u32,
    /// Declared reuse probability `p` in `[0, 1]`.
    pub reuse_probability: f64,
}

impl Default for Horizon {
    fn default() -> Self {
        Self {
            requests: 1,
            reuse_probability: 0.0,
        }
    }
}

impl Horizon {
    fn normalized(self) -> Self {
        Self {
            requests: self.requests.max(1),
            reuse_probability: self.reuse_probability.clamp(0.0, 1.0),
        }
    }

    /// Expected share of a cache write charged to this request.
    fn cache_write_amortization(self) -> f64 {
        let requests = f64::from(self.requests.saturating_sub(1));
        1.0 / (1.0 + self.reuse_probability * requests)
    }
}

/// Result of freezing the working set for one user request.
#[derive(Debug)]
pub struct FrozenSelection {
    prefix_len: usize,
    prefix_digest: String,
    projected: Vec<ResponseItem>,
    applied: bool,
    canonical_reason: CanonicalReason,
    manifest: SelectionManifest,
    manifest_digest: String,
}

impl FrozenSelection {
    /// Structural manifest for the frozen request.
    pub fn manifest(&self) -> &SelectionManifest {
        &self.manifest
    }

    /// Digest over the canonical JSON of the manifest.
    pub fn manifest_digest(&self) -> &str {
        &self.manifest_digest
    }

    /// Epoch identity of the input selected for this request.
    pub fn epoch_id(&self) -> &str {
        &self.manifest.epoch.epoch_id
    }

    /// Whether the runtime may send the projection instead of canonical history.
    pub fn applied(&self) -> bool {
        self.applied
    }

    /// Input for a model turn of this request.
    ///
    /// `canonical` is the append-only canonical input for the turn. The frozen
    /// prefix must still be byte-identical; items appended after it (the live
    /// tool chain, recorded by the run loop) are appended verbatim. Any mismatch
    /// or unpaired overlay falls back to the canonical input.
    pub fn derived_input(&self, canonical: &[ResponseItem]) -> DerivedInput {
        if !self.applied {
            return DerivedInput::Canonical(self.canonical_reason);
        }
        if canonical.len() < self.prefix_len
            || digest_items(&canonical[..self.prefix_len]) != self.prefix_digest
        {
            return DerivedInput::Canonical(CanonicalReason::Fallback(
                FallbackReason::CanonicalPrefixChanged,
            ));
        }
        let mut derived = self.projected.clone();
        derived.extend_from_slice(&canonical[self.prefix_len..]);
        if build_pairs(&derived).is_err() {
            return DerivedInput::Canonical(CanonicalReason::Fallback(
                FallbackReason::ClosureViolation,
            ));
        }
        DerivedInput::Projected(derived)
    }
}

/// Model input for one turn.
#[derive(Debug, PartialEq)]
pub enum DerivedInput {
    /// Selected items plus the verbatim active-chain overlay.
    Projected(Vec<ResponseItem>),
    /// Canonical input, with the reason it was sent instead of a projection.
    Canonical(CanonicalReason),
}

/// Freezes the working set for one user request.
///
/// Pure and deterministic: the same semantic inputs produce the same projection,
/// item ids and manifest digest. It performs no I/O, reads no clock, dispatches no
/// tool, and never mutates the canonical history.
pub fn freeze(inputs: &FreezeInputs<'_>) -> FrozenSelection {
    let canonical = inputs.canonical;
    let prefix_digest = digest_items(canonical);
    let item_jsons = item_json_strings(canonical);
    let item_tokens: Vec<u64> = item_jsons
        .iter()
        .map(|json| bytes_to_tokens(json.len() as u64))
        .collect();
    let item_digests: Vec<String> = item_jsons.iter().map(|json| digest_text(json)).collect();
    let canonical_tokens: u64 = item_tokens.iter().sum();
    let len = canonical.len();

    if canonical.is_empty() {
        return FrozenSelection::fallback(
            inputs,
            prefix_digest,
            &item_jsons,
            &item_tokens,
            &item_digests,
            canonical_tokens,
            FallbackReason::SchemaInvalid,
        );
    }

    let pairs = match build_pairs(canonical) {
        Ok(pairs) => pairs,
        Err(reason) => {
            return FrozenSelection::fallback(
                inputs,
                prefix_digest,
                &item_jsons,
                &item_tokens,
                &item_digests,
                canonical_tokens,
                reason,
            );
        }
    };

    let request_start = inputs.current_request_start.min(len);
    let mut pins: Vec<Option<(PinClass, &'static str)>> = vec![None; len];
    for pin in pins.iter_mut().skip(request_start) {
        *pin = Some((PinClass::CurrentRequest, "current_user_request"));
    }
    for &seq in inputs.explicit_pins {
        if seq < len && pins[seq].is_none() {
            pins[seq] = Some((PinClass::Explicit, "recorded_explicit_pin"));
        }
    }
    // The carried session context (user instructions and environment context
    // envelopes) is a pinned prefix: dropping it would change what the session is.
    for (seq, item) in canonical.iter().enumerate() {
        if pins[seq].is_none() && is_carried_context(item) {
            pins[seq] = Some((PinClass::Explicit, "carried_session_context"));
        }
    }
    for (seq, item) in canonical.iter().enumerate() {
        if pins[seq].is_none() && is_opaque(item) {
            pins[seq] = Some((PinClass::OpaqueVariant, opaque_reason(item)));
        }
    }
    apply_closure(&mut pins, &pairs);

    // Deterministic anchor retrieval from the current request only.
    let request_text = canonical[request_start..]
        .iter()
        .map(item_text)
        .collect::<Vec<String>>()
        .join("\n");
    let anchors = extract_anchors(&request_text);
    if anchors.is_empty() {
        return FrozenSelection::fallback(
            inputs,
            prefix_digest,
            &item_jsons,
            &item_tokens,
            &item_digests,
            canonical_tokens,
            FallbackReason::RetrievalInsufficient,
        );
    }
    let mut hit_scores: BTreeMap<usize, usize> = BTreeMap::new();
    for seq in 0..request_start {
        let text = item_text(&canonical[seq]).to_lowercase();
        let score = anchors
            .iter()
            .filter(|anchor| text.contains(anchor.as_str()))
            .count();
        if score > 0 {
            hit_scores.insert(seq, score);
        }
    }
    if hit_scores.is_empty() || hit_scores.len() > MAX_RETRIEVAL_HITS {
        return FrozenSelection::fallback(
            inputs,
            prefix_digest,
            &item_jsons,
            &item_tokens,
            &item_digests,
            canonical_tokens,
            FallbackReason::RetrievalInsufficient,
        );
    }
    for seq in hit_scores.keys() {
        if let Some(&counterpart) = pairs.get(seq)
            && pins[counterpart].is_none()
        {
            pins[counterpart] = Some((PinClass::DependencyClosure, "anchor_pair_closure"));
        }
    }

    let selected: Vec<usize> = (0..len)
        .filter(|seq| pins[*seq].is_some() || hit_scores.contains_key(seq))
        .collect();
    let projection_items: Vec<ResponseItem> =
        selected.iter().map(|&seq| canonical[seq].clone()).collect();
    let projection_tokens: u64 = selected.iter().map(|&seq| item_tokens[seq]).sum();
    let leading_selected = (0..len)
        .take_while(|&seq| pins[seq].is_some() || hit_scores.contains_key(&seq))
        .count();

    let horizon = inputs.horizon.normalized();
    let candidate_basis = Basis::new(
        inputs,
        &item_jsons,
        &item_tokens,
        projection_tokens,
        leading_selected,
    );
    let baseline_basis = Basis::new(inputs, &item_jsons, &item_tokens, canonical_tokens, len);
    let mut unknown_fields = catalog_unknown_fields(inputs);
    let candidate = cost_basis(inputs, &candidate_basis, horizon, &mut unknown_fields);
    let baseline = cost_basis(inputs, &baseline_basis, horizon, &mut unknown_fields);
    let unknown_fields: Vec<String> = unknown_fields.into_iter().collect();
    let confidence = if unknown_fields.is_empty() {
        Confidence::High
    } else {
        Confidence::Low
    };

    // `canonical_reason` is only read when `applied` is false.
    let (winner, decision_reason, applied, canonical_reason) = if confidence == Confidence::Low {
        let reason = classify_unknown(&unknown_fields, inputs.catalog);
        (
            Winner::Unknown,
            reason.as_str(),
            false,
            CanonicalReason::Fallback(reason),
        )
    } else {
        match (candidate.amount, baseline.amount) {
            (Some(candidate_amount), Some(baseline_amount)) => {
                if approximately_equal(candidate_amount, baseline_amount) {
                    if candidate.tokens.total < baseline.tokens.total {
                        (
                            Winner::Candidate,
                            "approximately_equal_cost_fewer_tokens",
                            true,
                            CanonicalReason::BaselinePreferred,
                        )
                    } else {
                        (
                            Winner::Baseline,
                            "approximately_equal_cost_baseline_tokens",
                            false,
                            CanonicalReason::BaselinePreferred,
                        )
                    }
                } else if candidate_amount < baseline_amount {
                    (
                        Winner::Candidate,
                        "lower_expected_cost",
                        true,
                        CanonicalReason::BaselinePreferred,
                    )
                } else {
                    (
                        Winner::Baseline,
                        "lower_expected_cost_baseline",
                        false,
                        CanonicalReason::BaselinePreferred,
                    )
                }
            }
            _ => (
                Winner::Unknown,
                FallbackReason::PricingUnknown.as_str(),
                false,
                CanonicalReason::Fallback(FallbackReason::PricingUnknown),
            ),
        }
    };

    let sent = if applied { &candidate } else { &baseline };
    let net = match (candidate.amount, baseline.amount) {
        (Some(candidate_amount), Some(baseline_amount)) => Some(baseline_amount - candidate_amount),
        _ => None,
    };
    let manifest = SelectionManifest {
        schema_version: SCHEMA_VERSION,
        kind: MANIFEST_KIND,
        digest_algorithm: DIGEST_ALGORITHM,
        model: inputs.model.clone(),
        reasoning_effort: inputs.reasoning_effort.clone(),
        canonical: CanonicalStats {
            log_len: len,
            digest: prefix_digest.clone(),
        },
        projection: ProjectionStats {
            len: selected.len(),
            omitted: len - selected.len(),
            digest: digest_item_json(&item_jsons, &selected),
            items: selected
                .iter()
                .map(|&seq| item_id(seq, &item_digests[seq]))
                .collect(),
        },
        epoch: EpochStats {
            epoch_id: sent.epoch_id.clone(),
            stable_prefix_digest: sent.stable_prefix_digest.clone(),
            candidate_epoch_id: candidate.epoch_id.clone(),
            baseline_epoch_id: baseline.epoch_id.clone(),
            changed: inputs.previous_epoch_id != Some(sent.epoch_id.as_str()),
        },
        cache_disposition: CacheDisposition {
            cached_eligible: sent.tokens.cached > 0,
            reason: sent.cache_reason,
        },
        pins: (0..len)
            .filter_map(|seq| {
                pins[seq].map(|(pin_class, reason)| PinRecord {
                    item_id: item_id(seq, &item_digests[seq]),
                    pin_class,
                    reason,
                })
            })
            .collect(),
        retrieval: {
            let mut hits: Vec<(usize, usize)> = hit_scores
                .iter()
                .map(|(seq, score)| (*seq, *score))
                .collect();
            hits.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
            hits.into_iter()
                .map(|(seq, score)| RetrievalRecord {
                    item_id: item_id(seq, &item_digests[seq]),
                    rule: "anchor",
                    anchor_count: score,
                    tie_break: seq,
                })
                .collect()
        },
        omissions: (0..len)
            .filter(|seq| pins[*seq].is_none() && !hit_scores.contains_key(seq))
            .map(|seq| OmissionRecord {
                item_id: item_id(seq, &item_digests[seq]),
                decision_source: "deterministic",
                reason: "no_anchor_match",
            })
            .collect(),
        cost: CostRecord {
            catalog_version: inputs
                .catalog
                .map(|catalog| catalog.catalog_version.clone()),
            currency: inputs.catalog.map(|catalog| catalog.currency.clone()),
            unit: inputs.catalog.map(|catalog| catalog.unit.clone()),
            horizon,
            confidence,
            unknown_fields,
            expected_output_tokens: inputs.usage.map(|usage| usage.output_tokens),
            candidate,
            baseline,
            net,
            savings_claim: SAVINGS_CLAIM_NONE,
        },
        decision: DecisionRecord {
            winner,
            confidence,
            applied,
            reason: decision_reason,
        },
        fallback: FallbackRecord {
            used: canonical_reason.fallback_reason().is_some(),
            reason: canonical_reason
                .fallback_reason()
                .map(FallbackReason::as_str),
        },
        telemetry: Telemetry {
            canonical_tokens,
            projection_tokens,
            note: "accounting_only_not_a_selection_input",
        },
    };
    let manifest_digest = manifest.digest();
    FrozenSelection {
        prefix_len: len,
        prefix_digest,
        projected: projection_items,
        applied,
        canonical_reason,
        manifest,
        manifest_digest,
    }
}

impl FrozenSelection {
    /// Builds the conservative fallback result: the canonical input is sent and
    /// the failure is recorded. No selection is applied, so every canonical item
    /// is recorded as not selected with `decision_source: "fallback"`.
    #[allow(clippy::too_many_arguments)]
    fn fallback(
        inputs: &FreezeInputs<'_>,
        prefix_digest: String,
        item_jsons: &[String],
        item_tokens: &[u64],
        item_digests: &[String],
        canonical_tokens: u64,
        reason: FallbackReason,
    ) -> Self {
        let len = inputs.canonical.len();
        let horizon = inputs.horizon.normalized();
        let baseline_basis = Basis::new(inputs, item_jsons, item_tokens, canonical_tokens, len);
        let mut unknown_fields = catalog_unknown_fields(inputs);
        let baseline = cost_basis(inputs, &baseline_basis, horizon, &mut unknown_fields);
        let unknown_fields: Vec<String> = unknown_fields.into_iter().collect();
        let manifest = SelectionManifest {
            schema_version: SCHEMA_VERSION,
            kind: MANIFEST_KIND,
            digest_algorithm: DIGEST_ALGORITHM,
            model: inputs.model.clone(),
            reasoning_effort: inputs.reasoning_effort.clone(),
            canonical: CanonicalStats {
                log_len: len,
                digest: prefix_digest.clone(),
            },
            projection: ProjectionStats {
                len: 0,
                omitted: len,
                digest: digest_item_json(item_jsons, &[]),
                items: Vec::new(),
            },
            epoch: EpochStats {
                epoch_id: baseline.epoch_id.clone(),
                stable_prefix_digest: baseline.stable_prefix_digest.clone(),
                candidate_epoch_id: baseline.epoch_id.clone(),
                baseline_epoch_id: baseline.epoch_id.clone(),
                changed: inputs.previous_epoch_id != Some(baseline.epoch_id.as_str()),
            },
            cache_disposition: CacheDisposition {
                cached_eligible: baseline.tokens.cached > 0,
                reason: baseline.cache_reason,
            },
            pins: Vec::new(),
            retrieval: Vec::new(),
            omissions: (0..len)
                .map(|seq| OmissionRecord {
                    item_id: item_id(seq, &item_digests[seq]),
                    decision_source: "fallback",
                    reason: reason.as_str(),
                })
                .collect(),
            cost: CostRecord {
                catalog_version: inputs
                    .catalog
                    .map(|catalog| catalog.catalog_version.clone()),
                currency: inputs.catalog.map(|catalog| catalog.currency.clone()),
                unit: inputs.catalog.map(|catalog| catalog.unit.clone()),
                horizon,
                confidence: Confidence::Low,
                unknown_fields,
                expected_output_tokens: inputs.usage.map(|usage| usage.output_tokens),
                candidate: baseline.clone(),
                baseline,
                net: None,
                savings_claim: SAVINGS_CLAIM_NONE,
            },
            decision: DecisionRecord {
                winner: Winner::Unknown,
                confidence: Confidence::Low,
                applied: false,
                reason: reason.as_str(),
            },
            fallback: FallbackRecord {
                used: true,
                reason: Some(reason.as_str()),
            },
            telemetry: Telemetry {
                canonical_tokens,
                projection_tokens: 0,
                note: "accounting_only_not_a_selection_input",
            },
        };
        let manifest_digest = manifest.digest();
        Self {
            prefix_len: len,
            prefix_digest,
            projected: Vec::new(),
            applied: false,
            canonical_reason: CanonicalReason::Fallback(reason),
            manifest,
            manifest_digest,
        }
    }
}

// ---------------------------------------------------------------------------
// Manifest (structural only: counts, reason codes, ids and digests)
// ---------------------------------------------------------------------------

/// Structural per-request record. Never contains item text, anchor values, tool
/// definitions or instruction text.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectionManifest {
    /// Contract version.
    pub schema_version: u32,
    /// Contract kind.
    pub kind: &'static str,
    /// Digest algorithm for every digest below.
    pub digest_algorithm: &'static str,
    /// Exact wire model slug.
    pub model: String,
    /// Reasoning effort identity used for the request.
    pub reasoning_effort: String,
    /// Canonical history accounting.
    pub canonical: CanonicalStats,
    /// Projection accounting (the would-be projection, even on fallback).
    pub projection: ProjectionStats,
    /// Cache epoch identity.
    pub epoch: EpochStats,
    /// Cache disposition of the selected input.
    pub cache_disposition: CacheDisposition,
    /// Items pinned (never omitted).
    pub pins: Vec<PinRecord>,
    /// Deterministic retrieval hits.
    pub retrieval: Vec<RetrievalRecord>,
    /// Items not selected in the projection, all accounted for (C12).
    pub omissions: Vec<OmissionRecord>,
    /// Cost estimate and confidence.
    pub cost: CostRecord,
    /// Ranking/decision record.
    pub decision: DecisionRecord,
    /// Fallback record.
    pub fallback: FallbackRecord,
    /// Observational token accounting.
    pub telemetry: Telemetry,
}

impl SelectionManifest {
    /// Digest over the canonical JSON of the manifest.
    pub fn digest(&self) -> String {
        digest_serialized(self)
    }
}

/// Canonical history accounting.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CanonicalStats {
    /// Number of canonical items the selection was derived from.
    pub log_len: usize,
    /// Digest of the canonical items.
    pub digest: String,
}

/// Projection accounting (the would-be projection, even on fallback).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProjectionStats {
    /// Number of selected items.
    pub len: usize,
    /// Number of items not selected.
    pub omitted: usize,
    /// Digest of the selected items.
    pub digest: String,
    /// Stable item ids of the selected items, in canonical order.
    pub items: Vec<String>,
}

/// Epoch identity of the selected input.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EpochStats {
    /// Epoch of the input actually selected.
    pub epoch_id: String,
    /// Stable-prefix digest of the input actually selected.
    pub stable_prefix_digest: String,
    /// Epoch the projection would have.
    pub candidate_epoch_id: String,
    /// Epoch of the no-selection baseline.
    pub baseline_epoch_id: String,
    /// Whether the selected epoch differs from the previously recorded one.
    pub changed: bool,
}

/// Cache disposition of the selected input.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CacheDisposition {
    /// Whether the selected input may bill cached tokens.
    pub cached_eligible: bool,
    /// Structural reason code.
    pub reason: &'static str,
}

/// Pinned item.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PinRecord {
    /// Stable item id.
    pub item_id: String,
    /// Pin class.
    pub pin_class: PinClass,
    /// Structural reason code.
    pub reason: &'static str,
}

/// Why an item can never be omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PinClass {
    /// Belongs to the current user request.
    CurrentRequest,
    /// Appended by the live tool chain after the freeze (overlay, not frozen).
    ActiveToolChain,
    /// Marked pinned in the canonical log.
    Explicit,
    /// Required by a selected call/result pair (C6).
    DependencyClosure,
    /// Variant whose semantics are unknown; retained conservatively.
    OpaqueVariant,
}

/// Deterministic retrieval hit.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RetrievalRecord {
    /// Stable item id.
    pub item_id: String,
    /// Retrieval rule (always `anchor` here).
    pub rule: &'static str,
    /// Number of distinct anchors matched.
    pub anchor_count: usize,
    /// Ascending-sequence tie-break.
    pub tie_break: usize,
}

/// Item not selected in the projection.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OmissionRecord {
    /// Stable item id.
    pub item_id: String,
    /// Decision source (`deterministic`, or `fallback` when no selection ran).
    pub decision_source: &'static str,
    /// Structural reason code.
    pub reason: &'static str,
}

/// Cost estimate and confidence.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CostRecord {
    /// Catalog version in force, when a catalog record existed.
    pub catalog_version: Option<String>,
    /// Currency of the amounts.
    pub currency: Option<String>,
    /// Unit of the rates, when a catalog record existed.
    pub unit: Option<String>,
    /// Horizon used for the comparison.
    pub horizon: Horizon,
    /// Cost confidence.
    pub confidence: Confidence,
    /// Fields that were unknown; non-empty forces `Confidence::Low`.
    pub unknown_fields: Vec<String>,
    /// Expected output tokens (recorded usage, never invented).
    pub expected_output_tokens: Option<u64>,
    /// Projection estimate.
    pub candidate: CostBasisRecord,
    /// No-selection baseline estimate.
    pub baseline: CostBasisRecord,
    /// `baseline.amount - candidate.amount`, when both are supported.
    pub net: Option<f64>,
    /// Always `none`: savings require provider usage evidence (I15).
    pub savings_claim: &'static str,
}

/// Cost record for one basis (candidate or baseline).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CostBasisRecord {
    /// Amount, `None` when any required rate is unknown.
    pub amount: Option<f64>,
    /// Line-item breakdown, when the amount is supported.
    pub breakdown: Option<Breakdown>,
    /// Token accounting.
    pub tokens: TokenRecord,
    /// Epoch identity of this basis.
    pub epoch_id: String,
    /// Stable-prefix digest of this basis.
    pub stable_prefix_digest: String,
    /// Structural cache reason code.
    pub cache_reason: &'static str,
}

/// Token accounting per class. Accounting only: never a selection predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TokenRecord {
    /// Total input tokens of the basis.
    pub total: u64,
    /// Tokens in the retained stable prefix.
    pub stable_prefix: u64,
    /// Tokens billed as cached.
    pub cached: u64,
    /// Tokens billed as cache write.
    pub cache_write: u64,
    /// Tokens billed as uncached input.
    pub uncached: u64,
    /// Expected output tokens.
    pub output: u64,
}

/// Line-item cost breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Breakdown {
    /// Uncached input term.
    pub uncached_input: f64,
    /// Cached input term.
    pub cached_input: f64,
    /// Cache-write term, amortized over the horizon.
    pub cache_write: f64,
    /// Output term (reasoning included when billed as output).
    pub output: f64,
    /// Long-context multiplier applied to every term (`1.0` when none applies or
    /// when the tier records separate per-term rates).
    pub long_context_multiplier: f64,
    /// Deterministic retrieval term (`0.0`: not monetized, no rate is invented).
    pub retrieval: f64,
    /// Jev term (`0.0`: deterministic retrieval only, no Jev in this slice).
    pub jev: f64,
    /// Sum of the terms after the multiplier.
    pub total: f64,
}

/// Cost confidence (M0 §4.4 / I12; `medium` collapses into `low` here).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    /// All required inputs recorded, fresh and comparable; cost may inform selection.
    High,
    /// Anything unknown or stale; no cost-driven change is allowed.
    Low,
}

impl Confidence {
    /// Stable machine-readable token.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Low => "low",
        }
    }
}

/// Ranking result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Winner {
    /// Projection ranked best.
    Candidate,
    /// No-selection baseline ranked best.
    Baseline,
    /// No supported comparison (low confidence).
    Unknown,
}

/// Decision record.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecisionRecord {
    /// Ranking winner.
    pub winner: Winner,
    /// Cost confidence.
    pub confidence: Confidence,
    /// Whether the runtime sends the projection.
    pub applied: bool,
    /// Structural reason code.
    pub reason: &'static str,
}

/// Fallback record (C9).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FallbackRecord {
    /// True only for an actual selection-stage failure.
    pub used: bool,
    /// Closed-set reason code.
    pub reason: Option<&'static str>,
}

/// Observational telemetry.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Telemetry {
    /// Canonical input tokens.
    pub canonical_tokens: u64,
    /// Projection input tokens (the would-be projection, even on fallback).
    pub projection_tokens: u64,
    /// Fixed note.
    pub note: &'static str,
}

// ---------------------------------------------------------------------------
// Item classification
// ---------------------------------------------------------------------------

/// A call/result pair edge: `seq -> counterpart seq`.
type PairEdges = BTreeMap<usize, usize>;

/// Builds the whole-pair edges and rejects unresolved or duplicated pairs.
///
/// Any call without its output, any output without its call, or any duplicated
/// `call_id` is a `closure_violation`: the canonical input is sent instead.
fn build_pairs(items: &[ResponseItem]) -> Result<PairEdges, FallbackReason> {
    let mut calls: BTreeMap<String, usize> = BTreeMap::new();
    let mut outputs: BTreeMap<String, usize> = BTreeMap::new();
    for (seq, item) in items.iter().enumerate() {
        match item {
            ResponseItem::FunctionCall { call_id, .. }
            | ResponseItem::CustomToolCall { call_id, .. }
            | ResponseItem::LocalShellCall {
                call_id: Some(call_id),
                ..
            } => {
                if calls.insert(call_id.clone(), seq).is_some() {
                    return Err(FallbackReason::ClosureViolation);
                }
            }
            ResponseItem::FunctionCallOutput { call_id, .. }
            | ResponseItem::CustomToolCallOutput { call_id, .. } => {
                if outputs.insert(call_id.clone(), seq).is_some() {
                    return Err(FallbackReason::ClosureViolation);
                }
            }
            _ => {}
        }
    }
    let mut edges = PairEdges::new();
    for (call_id, call_seq) in &calls {
        let Some(&output_seq) = outputs.get(call_id) else {
            return Err(FallbackReason::ClosureViolation);
        };
        edges.insert(*call_seq, output_seq);
        edges.insert(output_seq, *call_seq);
    }
    if outputs.keys().any(|call_id| !calls.contains_key(call_id)) {
        return Err(FallbackReason::ClosureViolation);
    }
    Ok(edges)
}

fn apply_closure(pins: &mut [Option<(PinClass, &'static str)>], pairs: &PairEdges) {
    for seq in 0..pins.len() {
        if pins[seq].is_none() {
            continue;
        }
        if let Some(&counterpart) = pairs.get(&seq)
            && pins[counterpart].is_none()
        {
            pins[counterpart] = Some((PinClass::DependencyClosure, "call_result_pair"));
        }
    }
}

/// Session-context envelopes (`<user_instructions>` / `<environment_context>`)
/// carried at the head of the canonical history.
fn is_carried_context(item: &ResponseItem) -> bool {
    match item {
        ResponseItem::Message { role, content, .. } => content.iter().any(|part| match part {
            ContentItem::InputText { text } => !matches!(
                InputMessageKind::from((role.as_str(), text.as_str())),
                InputMessageKind::Plain
            ),
            _ => false,
        }),
        _ => false,
    }
}

/// Variants whose dependencies are unknown or provider-referenced.
fn is_opaque(item: &ResponseItem) -> bool {
    match item {
        ResponseItem::Reasoning { .. }
        | ResponseItem::WebSearchCall { .. }
        | ResponseItem::Other => true,
        ResponseItem::LocalShellCall { call_id: None, .. } => true,
        _ => false,
    }
}

fn opaque_reason(item: &ResponseItem) -> &'static str {
    match item {
        ResponseItem::Reasoning { .. } => "reasoning_item_referenced_by_provider",
        ResponseItem::WebSearchCall { .. } => "web_search_call",
        ResponseItem::Other => "unknown_variant",
        ResponseItem::LocalShellCall { call_id: None, .. } => "call_without_call_id",
        _ => "opaque",
    }
}

/// Plain text of an item, used only for in-process anchor matching.
fn item_text(item: &ResponseItem) -> String {
    match item {
        ResponseItem::Message { content, .. } => content
            .iter()
            .filter_map(|part| match part {
                ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                    Some(text.clone())
                }
                ContentItem::InputImage { .. } => None,
            })
            .collect::<Vec<String>>()
            .join("\n"),
        ResponseItem::Reasoning {
            summary, content, ..
        } => {
            let mut parts: Vec<String> = summary
                .iter()
                .map(|part| match part {
                    ReasoningItemReasoningSummary::SummaryText { text } => text.clone(),
                })
                .collect();
            if let Some(content) = content {
                parts.extend(content.iter().map(|part| match part {
                    ReasoningItemContent::ReasoningText { text }
                    | ReasoningItemContent::Text { text } => text.clone(),
                }));
            }
            parts.join("\n")
        }
        ResponseItem::FunctionCall { .. }
        | ResponseItem::CustomToolCall { .. }
        | ResponseItem::LocalShellCall { .. }
        | ResponseItem::WebSearchCall { .. } => serde_json::to_string(item).unwrap_or_default(),
        ResponseItem::FunctionCallOutput { output, .. } => output.content.clone(),
        ResponseItem::CustomToolCallOutput { output, .. } => output.clone(),
        ResponseItem::Other => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Anchors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnchorKind {
    /// Any match of the pattern is a specific anchor.
    Specific,
    /// A bare identifier; only retained when it has a specific shape.
    Identifier,
}

static ANCHOR_PATTERNS: LazyLock<Vec<(Regex, AnchorKind)>> = LazyLock::new(|| {
    [
        ("https?://[^\\s\"'`<>()\\[\\]{}]+", AnchorKind::Specific),
        (r#""([^"\n]{2,120})""#, AnchorKind::Specific),
        (r"'([^'\n]{2,120})'", AnchorKind::Specific),
        (r"`([^`\n]{2,120})`", AnchorKind::Specific),
        (
            r"(?:\.\.?/)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+",
            AnchorKind::Specific,
        ),
        (r"\b[A-Za-z_][A-Za-z0-9_]*\b", AnchorKind::Identifier),
        (r"\b[A-Z]{2,}[_-]?[0-9]{2,}\b", AnchorKind::Specific),
        (r"\bE[0-9]{3,}\b", AnchorKind::Specific),
        (r"\b0x[0-9a-fA-F]{4,}\b", AnchorKind::Specific),
    ]
    .into_iter()
    .filter_map(|(pattern, kind)| Regex::new(pattern).ok().map(|regex| (regex, kind)))
    .collect()
});

/// Deterministic anchor extraction, in first-appearance order, lowercased for
/// case-insensitive matching and capped at [`MAX_ANCHORS`].
fn extract_anchors(text: &str) -> Vec<String> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut anchors: Vec<String> = Vec::new();
    for (regex, kind) in ANCHOR_PATTERNS.iter() {
        for captures in regex.captures_iter(text) {
            let Some(raw) = captures
                .get(1)
                .or_else(|| captures.get(0))
                .map(|matched| matched.as_str())
            else {
                continue;
            };
            let candidate = raw.trim();
            if candidate.len() < 2 {
                continue;
            }
            match kind {
                AnchorKind::Specific => {
                    // "and/or"-style tokens are not file paths.
                    if candidate.contains('/')
                        && !candidate.contains('.')
                        && !candidate.starts_with('.')
                        && !candidate.starts_with('/')
                    {
                        continue;
                    }
                }
                AnchorKind::Identifier => {
                    if !is_specific_identifier(candidate) {
                        continue;
                    }
                }
            }
            let lowered = candidate.to_lowercase();
            if seen.insert(lowered.clone()) {
                anchors.push(lowered);
                if anchors.len() >= MAX_ANCHORS {
                    return anchors;
                }
            }
        }
    }
    anchors
}

/// Identifiers are only anchors when shaped specifically (separator or inner
/// uppercase), so ordinary prose words never select history.
fn is_specific_identifier(candidate: &str) -> bool {
    if candidate.len() < 4 {
        return false;
    }
    let mut chars = candidate.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    if !candidate
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return false;
    }
    let has_separator = candidate.contains('_');
    let has_inner_uppercase = candidate.chars().skip(1).any(|c| c.is_ascii_uppercase());
    has_separator || has_inner_uppercase
}

// ---------------------------------------------------------------------------
// Cost
// ---------------------------------------------------------------------------

struct Basis {
    total: u64,
    stable_prefix: u64,
    stable_prefix_digest: String,
    epoch_id: String,
}

impl Basis {
    fn new(
        inputs: &FreezeInputs<'_>,
        item_jsons: &[String],
        item_tokens: &[u64],
        total: u64,
        stable_prefix_len: usize,
    ) -> Self {
        let stable_prefix: u64 = item_tokens[..stable_prefix_len].iter().sum();
        let stable_prefix_digest =
            digest_item_json(item_jsons, &(0..stable_prefix_len).collect::<Vec<usize>>());
        let epoch_id = epoch_id(inputs, &stable_prefix_digest);
        Self {
            total,
            stable_prefix,
            stable_prefix_digest,
            epoch_id,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Charge {
    cached: u64,
    cache_write: u64,
    uncached: u64,
    reason: &'static str,
}

fn cache_charge(
    inputs: &FreezeInputs<'_>,
    basis: &Basis,
    unknown: &mut BTreeSet<String>,
) -> Charge {
    let Some(catalog) = inputs.catalog else {
        unknown.insert("catalog".to_string());
        return Charge {
            cached: 0,
            cache_write: 0,
            uncached: basis.total,
            reason: "catalog_missing",
        };
    };
    let caching_declared =
        catalog.cached_input_rate.is_some() || catalog.cache_write_rate.is_some();
    if !caching_declared {
        return Charge {
            cached: 0,
            cache_write: 0,
            uncached: basis.total,
            reason: "provider_cache_not_declared",
        };
    }
    if !catalog.cache_eligibility_known {
        unknown.insert("cache_eligibility".to_string());
        return Charge {
            cached: 0,
            cache_write: 0,
            uncached: basis.total,
            reason: "cache_eligibility_unknown",
        };
    }
    if inputs.previous_epoch_id == Some(basis.epoch_id.as_str()) {
        let Some(usage) = inputs.usage else {
            unknown.insert("cached_input_tokens".to_string());
            return Charge {
                cached: 0,
                cache_write: 0,
                uncached: basis.total,
                reason: "cached_token_evidence_missing",
            };
        };
        if catalog.cached_input_rate.is_none() {
            return write_or_uncached(catalog, basis.total);
        }
        let Some(min_prefix) = catalog.min_cacheable_prefix_tokens else {
            unknown.insert("min_cacheable_prefix_tokens".to_string());
            return Charge {
                cached: 0,
                cache_write: 0,
                uncached: basis.total,
                reason: "min_cacheable_prefix_unknown",
            };
        };
        if basis.stable_prefix < min_prefix {
            return Charge {
                cached: 0,
                cache_write: 0,
                uncached: basis.total,
                reason: "stable_prefix_below_min_cacheable",
            };
        }
        let cached = usage
            .cached_input_tokens
            .min(basis.stable_prefix)
            .min(basis.total);
        return Charge {
            cached,
            cache_write: 0,
            uncached: basis.total - cached,
            reason: "epoch_unchanged_cached_prefix",
        };
    }
    // Changed or unknown epoch: never bill cached tokens (I13).
    write_or_uncached(catalog, basis.total)
}

fn write_or_uncached(catalog: &PricingCatalog, total: u64) -> Charge {
    if catalog.cache_write_rate.is_some() {
        Charge {
            cached: 0,
            cache_write: total,
            uncached: 0,
            reason: "changed_epoch_cache_write",
        }
    } else {
        Charge {
            cached: 0,
            cache_write: 0,
            uncached: total,
            reason: "changed_epoch_uncached_no_write_rate",
        }
    }
}

fn cost_basis(
    inputs: &FreezeInputs<'_>,
    basis: &Basis,
    horizon: Horizon,
    unknown: &mut BTreeSet<String>,
) -> CostBasisRecord {
    let charge = cache_charge(inputs, basis, unknown);
    let breakdown = expected_amount(inputs, basis, &charge, horizon, unknown);
    CostBasisRecord {
        amount: breakdown.map(|breakdown| breakdown.total),
        breakdown,
        tokens: TokenRecord {
            total: basis.total,
            stable_prefix: basis.stable_prefix,
            cached: charge.cached,
            cache_write: charge.cache_write,
            uncached: charge.uncached,
            output: inputs.usage.map_or(0, |usage| usage.output_tokens),
        },
        epoch_id: basis.epoch_id.clone(),
        stable_prefix_digest: basis.stable_prefix_digest.clone(),
        cache_reason: charge.reason,
    }
}

/// How the recorded long-context rule applies to one basis.
enum LongContextTier {
    /// No recorded rule applies; every term uses its base rate.
    Base,
    /// One uniform multiplier applies to every term.
    UniformMultiplier(f64),
    /// The recorded rule prices the terms separately: each recorded rate
    /// overrides that term's base rate and `None` inherits it.
    TermRates(LongContextTermRates),
}

/// Applies the recorded long-context rule strictly: the condition is
/// `total prompt tokens > threshold`, so exactly the threshold stays at base
/// rates.
fn long_context_tier(catalog: &PricingCatalog, total_tokens: u64) -> LongContextTier {
    match catalog.long_context {
        Some(rule) if total_tokens > rule.threshold_tokens => match rule.term_rates {
            Some(term_rates) => LongContextTier::TermRates(term_rates),
            None => LongContextTier::UniformMultiplier(rule.multiplier),
        },
        _ => LongContextTier::Base,
    }
}

fn expected_amount(
    inputs: &FreezeInputs<'_>,
    basis: &Basis,
    charge: &Charge,
    horizon: Horizon,
    unknown: &mut BTreeSet<String>,
) -> Option<Breakdown> {
    let catalog = inputs.catalog?;
    let (long_context_multiplier, term_rates) = match long_context_tier(catalog, basis.total) {
        LongContextTier::Base => (1.0, None),
        LongContextTier::UniformMultiplier(multiplier) => (multiplier, None),
        LongContextTier::TermRates(term_rates) => (1.0, Some(term_rates)),
    };
    // A recorded per-term rate overrides that term's base rate; a term the tier
    // does not record inherits the base rate (I10: nothing is invented).
    let uncached_rate = term_rates
        .and_then(|rates| rates.uncached_input_rate)
        .or(catalog.uncached_input_rate);
    let Some(uncached_rate) = uncached_rate else {
        unknown.insert("uncached_input_rate".to_string());
        return None;
    };
    let mut cached_rate = 0.0;
    if charge.cached > 0 {
        let Some(rate) = term_rates
            .and_then(|rates| rates.cached_input_rate)
            .or(catalog.cached_input_rate)
        else {
            unknown.insert("cached_input_rate".to_string());
            return None;
        };
        cached_rate = rate;
    }
    let output_rate = term_rates
        .and_then(|rates| rates.output_rate)
        .or(catalog.output_rate);
    let Some(output_rate) = output_rate else {
        unknown.insert("output_rate".to_string());
        return None;
    };
    let Some(usage) = inputs.usage else {
        unknown.insert("expected_output_tokens".to_string());
        return None;
    };
    // No cache-write line item means the provider bills the write as ordinary
    // input; the known uncached rate is reused rather than inventing a rate (I10).
    let cache_write_rate = term_rates
        .and_then(|rates| rates.cache_write_rate)
        .or(catalog.cache_write_rate)
        .unwrap_or(uncached_rate);
    let per_million = 1_000_000.0;
    let uncached_input = charge.uncached as f64 / per_million * uncached_rate;
    let cached_input = charge.cached as f64 / per_million * cached_rate;
    let cache_write = charge.cache_write as f64 / per_million
        * cache_write_rate
        * horizon.cache_write_amortization();
    let output = usage.output_tokens as f64 / per_million * output_rate;
    let subtotal = uncached_input + cached_input + cache_write + output;
    Some(Breakdown {
        uncached_input,
        cached_input,
        cache_write,
        output,
        long_context_multiplier,
        retrieval: 0.0,
        jev: 0.0,
        total: subtotal * long_context_multiplier,
    })
}

fn catalog_unknown_fields(inputs: &FreezeInputs<'_>) -> BTreeSet<String> {
    let mut unknown: BTreeSet<String> = BTreeSet::new();
    match inputs.catalog {
        Some(catalog) => {
            if catalog.model != inputs.model {
                unknown.insert("catalog.model".to_string());
            }
            if catalog.unit != CATALOG_UNIT {
                unknown.insert("unit".to_string());
            }
            if catalog.price_source.is_none() {
                unknown.insert("price_source".to_string());
            }
            if catalog.freshness != Freshness::Fresh {
                unknown.insert("fetched_at".to_string());
            }
            for field in &catalog.unknown_fields {
                unknown.insert(field.clone());
            }
        }
        None => {
            unknown.insert("catalog".to_string());
        }
    }
    unknown
}

fn classify_unknown(unknown: &[String], catalog: Option<&PricingCatalog>) -> FallbackReason {
    if catalog.is_some_and(|catalog| catalog.freshness != Freshness::Fresh) {
        return FallbackReason::PricingStale;
    }
    let cache_fields = [
        "cache_eligibility",
        "min_cacheable_prefix_tokens",
        "cached_input_tokens",
    ];
    if !unknown.is_empty()
        && unknown
            .iter()
            .all(|field| cache_fields.contains(&field.as_str()))
    {
        return FallbackReason::CacheIneligible;
    }
    FallbackReason::PricingUnknown
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().min(right.abs());
    (left - right).abs() <= APPROX_EQUAL_COST_RELATIVE_TOLERANCE * scale
}

fn epoch_id(inputs: &FreezeInputs<'_>, stable_prefix_digest: &str) -> String {
    let mut identity: BTreeMap<&str, String> = BTreeMap::new();
    identity.insert("model", inputs.model.clone());
    identity.insert("reasoning_effort", inputs.reasoning_effort.clone());
    identity.insert("instructions_digest", inputs.instructions_digest.clone());
    identity.insert("tools_digest", inputs.tools_digest.clone());
    identity.insert("stable_prefix_digest", stable_prefix_digest.to_string());
    identity.insert(
        "catalog_version",
        inputs
            .catalog
            .map(|catalog| catalog.catalog_version.clone())
            .unwrap_or_else(|| "unknown".to_string()),
    );
    digest_serialized(&identity)
}

// ---------------------------------------------------------------------------
// Digest + token helpers
// ---------------------------------------------------------------------------

/// Digest of a text value (identity/privacy token, not a signature).
pub fn digest_text(text: &str) -> String {
    digest_bytes(text.as_bytes())
}

/// Digest of any serializable value over its canonical JSON.
pub fn digest_serialized<T: Serialize + ?Sized>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(value) => digest_text(&canonical_json(&value)),
        Err(_) => digest_text("<unserializable>"),
    }
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{DIGEST_ALGORITHM}:{:x}", Sha256::digest(bytes))
}

/// Digest over a list of items with length framing.
pub fn digest_items(items: &[ResponseItem]) -> String {
    let jsons = item_json_strings(items);
    digest_item_json(&jsons, &(0..jsons.len()).collect::<Vec<usize>>())
}

fn digest_item_json(item_jsons: &[String], selected: &[usize]) -> String {
    let mut hasher = Sha256::new();
    for &seq in selected {
        let Some(json) = item_jsons.get(seq) else {
            continue;
        };
        hasher.update((json.len() as u64).to_le_bytes());
        hasher.update(json.as_bytes());
    }
    format!("{DIGEST_ALGORITHM}:{:x}", hasher.finalize())
}

fn item_json_strings(items: &[ResponseItem]) -> Vec<String> {
    items
        .iter()
        .map(|item| serde_json::to_string(item).unwrap_or_default())
        .collect()
}

fn bytes_to_tokens(bytes: u64) -> u64 {
    bytes.div_ceil(BYTES_PER_TOKEN_ESTIMATE)
}

/// Stable derived item id: sequence plus a short content digest.
fn item_id(seq: usize, item_digest: &str) -> String {
    let short = item_digest
        .strip_prefix(DIGEST_ALGORITHM)
        .and_then(|rest| rest.strip_prefix(':'))
        .unwrap_or(item_digest);
    let short = short.get(..12).unwrap_or(short);
    format!("i{seq:06}-{short}")
}

/// Canonical JSON: object keys sorted, no insignificant whitespace.
fn canonical_json(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical_json(value, &mut out);
    out
}

fn write_canonical_json(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(boolean) => {
            out.push_str(if *boolean { "true" } else { "false" });
        }
        serde_json::Value::Number(number) => out.push_str(&number.to_string()),
        serde_json::Value::String(text) => {
            out.push_str(&serde_json::to_string(text).unwrap_or_default());
        }
        serde_json::Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            let sorted: BTreeMap<&String, &serde_json::Value> = map.iter().collect();
            out.push('{');
            for (index, (key, item)) in sorted.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).unwrap_or_default());
                out.push(':');
                write_canonical_json(item, out);
            }
            out.push('}');
        }
    }
}

// ---------------------------------------------------------------------------
// Append-only selection sidecar (structural records only)
// ---------------------------------------------------------------------------

/// Extension used by the selection sidecar next to a rollout file.
pub const SIDECAR_EXTENSION: &str = "working-set.jsonl";

/// Path of the append-only selection sidecar for a rollout file.
///
/// The sidecar is adjacent to the canonical rollout JSONL and carries the same
/// session identity (the rollout file name already contains the session
/// timestamp and conversation id): `rollout-<timestamp>-<conversation_id>.jsonl`
/// becomes `rollout-<timestamp>-<conversation_id>.working-set.jsonl`. It is a
/// separate append-only log, never the canonical history, and nothing reads it
/// back into the agent.
pub fn selection_sidecar_path(rollout_path: &Path) -> PathBuf {
    rollout_path.with_extension(SIDECAR_EXTENSION)
}

/// Decision token: the selected projection was sent on the wire.
pub const SIDECAR_DECISION_PROJECTED: &str = "projected";
/// Decision token: the canonical input was sent on the wire.
pub const SIDECAR_DECISION_CANONICAL: &str = "canonical";

/// One JSON line of the append-only selection sidecar.
///
/// Structural only: counts, reason codes, model slug, item ids and digests.
/// Item text, anchor values, tool definitions, instruction text and tokens never
/// appear here.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "record", rename_all = "snake_case")]
pub enum SidecarRecord {
    /// Exactly one per user request, written before its first model turn.
    Selection {
        /// Manifest schema version.
        schema_version: u32,
        /// Contract kind.
        kind: &'static str,
        /// 1-based ordinal of the user request within the session.
        request_index: u64,
        /// Full structural manifest of the frozen selection.
        manifest: SelectionManifest,
        /// Digest over the canonical JSON of `manifest`.
        manifest_digest: String,
        /// Whether the runtime may send the projection for this request.
        applied: bool,
        /// Digest of the canonical history snapshot before the request item was
        /// appended (the input to selection).
        canonical_digest_before_request: String,
        /// Digest of the canonical input selection was derived from, including
        /// the request item. Selection never mutates it.
        canonical_digest_after_selection: String,
        /// Ranking/selection reason code.
        selection_reason: &'static str,
        /// Fallback reason code, when canonical history was sent because of a
        /// selection-stage failure.
        fallback_reason: Option<&'static str>,
    },
    /// One per model turn of the request: which input the wire carried.
    Turn {
        /// Ordinal of the user request this turn belongs to.
        request_index: u64,
        /// 1-based ordinal of the model turn within the request.
        turn_index: u64,
        /// [`SIDECAR_DECISION_PROJECTED`] or [`SIDECAR_DECISION_CANONICAL`].
        decision: &'static str,
        /// `projected`, or the canonical reason code for a canonical turn.
        reason: &'static str,
        /// Epoch identity of the frozen selection.
        epoch_id: String,
        /// Number of canonical items for the turn.
        canonical_len: usize,
        /// Digest of the canonical input for the turn.
        canonical_digest: String,
        /// Number of items actually sent.
        wire_len: usize,
        /// Digest of the items actually sent.
        wire_digest: String,
    },
}

impl SidecarRecord {
    /// Builds the once-per-user-request selection record for a frozen selection.
    pub fn selection(
        request_index: u64,
        frozen: &FrozenSelection,
        canonical_digest_before_request: String,
    ) -> Self {
        let manifest = frozen.manifest();
        Self::Selection {
            schema_version: SCHEMA_VERSION,
            kind: MANIFEST_KIND,
            request_index,
            manifest: manifest.clone(),
            manifest_digest: frozen.manifest_digest().to_string(),
            applied: frozen.applied(),
            canonical_digest_before_request,
            canonical_digest_after_selection: manifest.canonical.digest.clone(),
            selection_reason: manifest.decision.reason,
            fallback_reason: manifest.fallback.reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::FunctionCallOutputPayload;
    use pretty_assertions::assert_eq;

    const MODEL: &str = "test-model";

    fn message(role: &str, text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: role.to_string(),
            content: vec![ContentItem::InputText {
                text: text.to_string(),
            }],
        }
    }

    fn call(call_id: &str, name: &str, arguments: &str) -> ResponseItem {
        ResponseItem::FunctionCall {
            id: None,
            name: name.to_string(),
            arguments: arguments.to_string(),
            call_id: call_id.to_string(),
        }
    }

    fn output(call_id: &str, content: &str) -> ResponseItem {
        ResponseItem::FunctionCallOutput {
            call_id: call_id.to_string(),
            output: FunctionCallOutputPayload {
                content: content.to_string(),
                success: Some(true),
            },
        }
    }

    fn reasoning() -> ResponseItem {
        ResponseItem::Reasoning {
            id: "rs_test".to_string(),
            summary: Vec::new(),
            content: None,
            encrypted_content: Some("OPAQUE_ENCRYPTED_BLOB".to_string()),
        }
    }

    /// Synthetic catalog: structural test data only, not any provider's price.
    fn synthetic_catalog() -> PricingCatalog {
        PricingCatalog {
            catalog_version: "synthetic-test-0".to_string(),
            model: MODEL.to_string(),
            provider: "synthetic".to_string(),
            currency: "USD".to_string(),
            unit: CATALOG_UNIT.to_string(),
            price_source: Some("synthetic://test-only".to_string()),
            fetched_at: Some("2026-09-18T00:00:00Z".to_string()),
            freshness: Freshness::Fresh,
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

    fn usage() -> UsageEvidence {
        UsageEvidence {
            input_tokens: 1_000,
            cached_input_tokens: 400,
            output_tokens: 100,
        }
    }

    struct Fixture<'a> {
        canonical: Vec<ResponseItem>,
        request_start: usize,
        catalog: Option<&'a PricingCatalog>,
        usage: Option<&'a UsageEvidence>,
        previous_epoch_id: Option<&'a str>,
        explicit_pins: Vec<usize>,
    }

    impl<'a> Fixture<'a> {
        fn freeze(&self) -> FrozenSelection {
            freeze(&FreezeInputs {
                model: MODEL.to_string(),
                reasoning_effort: "none".to_string(),
                instructions_digest: digest_text("INSTRUCTIONS_TEXT"),
                tools_digest: digest_text("TOOLS_JSON"),
                canonical: &self.canonical,
                current_request_start: self.request_start,
                explicit_pins: &self.explicit_pins,
                catalog: self.catalog,
                usage: self.usage,
                previous_epoch_id: self.previous_epoch_id,
                horizon: Horizon::default(),
            })
        }
    }

    /// History with one anchored pair, one unanchored pair, opaque items and a
    /// current request that anchors `src/app.rs`.
    fn standard_fixture<'a>(
        catalog: Option<&'a PricingCatalog>,
        usage: Option<&'a UsageEvidence>,
        previous_epoch_id: Option<&'a str>,
    ) -> Fixture<'a> {
        let canonical = vec![
            message("user", "please update src/app.rs and keep src/lib.rs"),
            message("assistant", "on it"),
            reasoning(),
            call("call_1", "shell", r#"{"command":["cat","src/app.rs"]}"#),
            output("call_1", "fn main() {}"),
            message("user", "unrelated chatter about lunch"),
            message("assistant", "noted"),
            call("call_2", "shell", r#"{"command":["ls"]}"#),
            output("call_2", "file list"),
            message("assistant", "done"),
            message(
                "user",
                "now fix the bug in src/app.rs reported as E1234 in `bubble`",
            ),
        ];
        Fixture {
            canonical,
            request_start: 10,
            catalog,
            usage,
            previous_epoch_id,
            explicit_pins: Vec::new(),
        }
    }

    fn selected_seqs(manifest: &SelectionManifest) -> Vec<usize> {
        manifest
            .projection
            .items
            .iter()
            .map(|id| id[1..7].parse::<usize>().expect("sequence prefix"))
            .collect()
    }

    fn omitted_seqs(manifest: &SelectionManifest) -> Vec<usize> {
        manifest
            .omissions
            .iter()
            .map(|record| {
                record.item_id[1..7]
                    .parse::<usize>()
                    .expect("sequence prefix")
            })
            .collect()
    }

    #[test]
    fn pairs_are_whole_and_unanchored_pairs_are_dropped_together() {
        let fixture = standard_fixture(None, None, None);
        let frozen = fixture.freeze();
        let manifest = frozen.manifest();
        // Selected: 0 (anchor hit), 2 (opaque reasoning), 3+4 (anchored pair with
        // closure), 10 (current request). Omitted: 1, 5, 6, 7+8 (whole pair), 9.
        assert_eq!(manifest.projection.len, 5);
        assert_eq!(manifest.projection.omitted, 6);
        assert_eq!(selected_seqs(manifest), vec![0, 2, 3, 4, 10]);
        assert_eq!(omitted_seqs(manifest), vec![1, 5, 6, 7, 8, 9]);
        // Every canonical item is accounted for exactly once (C12).
        assert_eq!(
            manifest.pins.len() + manifest.retrieval.len() + manifest.omissions.len(),
            manifest.canonical.log_len
        );
    }

    #[test]
    fn unresolved_historical_orphan_falls_back_to_canonical() {
        let mut fixture = standard_fixture(None, None, None);
        fixture
            .canonical
            .insert(6, call("call_orphan", "shell", "{}"));
        let frozen = fixture.freeze();
        assert!(!frozen.applied());
        assert!(frozen.manifest().fallback.used);
        assert_eq!(frozen.manifest().fallback.reason, Some("closure_violation"));
        assert_eq!(
            frozen.derived_input(&fixture.canonical),
            DerivedInput::Canonical(CanonicalReason::Fallback(FallbackReason::ClosureViolation))
        );
    }

    #[test]
    fn duplicated_call_id_falls_back_to_canonical() {
        let mut fixture = standard_fixture(None, None, None);
        fixture.canonical.push(call("call_2", "shell", "{}"));
        let frozen = fixture.freeze();
        assert_eq!(frozen.manifest().fallback.reason, Some("closure_violation"));
    }

    #[test]
    fn request_without_anchors_falls_back_conservatively() {
        let mut fixture = standard_fixture(None, None, None);
        fixture.canonical[10] = message("user", "please continue");
        let frozen = fixture.freeze();
        assert_eq!(
            frozen.manifest().fallback.reason,
            Some("retrieval_insufficient")
        );
        assert!(!frozen.applied());
    }

    #[test]
    fn anchors_without_hits_fall_back_conservatively() {
        let mut fixture = standard_fixture(None, None, None);
        fixture.canonical[10] = message("user", "now check src/missing_file.rs");
        let frozen = fixture.freeze();
        assert_eq!(
            frozen.manifest().fallback.reason,
            Some("retrieval_insufficient")
        );
    }

    #[test]
    fn opaque_variants_are_retained_conservatively() {
        let mut fixture = standard_fixture(None, None, None);
        fixture.canonical.insert(5, ResponseItem::Other);
        fixture.request_start = 11;
        let frozen = fixture.freeze();
        let manifest = frozen.manifest();
        let opaque: Vec<&PinRecord> = manifest
            .pins
            .iter()
            .filter(|pin| pin.pin_class == PinClass::OpaqueVariant)
            .collect();
        assert_eq!(opaque.len(), 2);
        assert_eq!(selected_seqs(manifest), vec![0, 2, 3, 4, 5, 11]);
    }

    #[test]
    fn explicit_pins_are_retained() {
        let mut fixture = standard_fixture(None, None, None);
        fixture.explicit_pins = vec![6];
        let frozen = fixture.freeze();
        assert!(selected_seqs(frozen.manifest()).contains(&6));
        assert!(frozen.manifest().pins.iter().any(|pin| {
            pin.pin_class == PinClass::Explicit && pin.item_id.starts_with("i000006-")
        }));
    }

    #[test]
    fn carried_session_context_envelopes_are_pinned() {
        let mut fixture = standard_fixture(None, None, None);
        fixture.canonical.insert(
            0,
            message(
                "user",
                "<environment_context>\n  <cwd>/tmp</cwd>\n</environment_context>",
            ),
        );
        fixture.request_start = 11;
        let frozen = fixture.freeze();
        let manifest = frozen.manifest();
        assert!(selected_seqs(manifest).contains(&0));
        assert!(manifest.pins.iter().any(|pin| {
            pin.item_id.starts_with("i000000-") && pin.reason == "carried_session_context"
        }));
        // A plain message is never treated as a carried envelope.
        assert!(!is_carried_context(&message("user", "plain note")));
    }

    #[test]
    fn identical_inputs_produce_identical_digests_and_manifest() {
        let first = standard_fixture(None, None, None).freeze();
        let second = standard_fixture(None, None, None).freeze();
        assert_eq!(first.manifest_digest(), second.manifest_digest());
        assert_eq!(
            first.manifest().projection.digest,
            second.manifest().projection.digest
        );
        assert_eq!(
            first.manifest().projection.items,
            second.manifest().projection.items
        );
    }

    #[test]
    fn item_ids_are_stable_and_ordered() {
        let frozen = standard_fixture(None, None, None).freeze();
        let items = &frozen.manifest().projection.items;
        assert_eq!(items.len(), 5);
        for (index, id) in items.iter().enumerate() {
            assert!(id.starts_with('i'));
            assert_eq!(id.len(), 20);
            assert!(id[1..7].chars().all(|c| c.is_ascii_digit()));
            assert!(id[8..].chars().all(|c| c.is_ascii_hexdigit()));
            if index > 0 {
                assert!(items[index - 1] < *id, "ids ascend by sequence");
            }
        }
    }

    #[test]
    fn manifest_never_carries_private_text_anchors_or_tools() {
        let catalog = synthetic_catalog();
        let usage = usage();
        let frozen = standard_fixture(Some(&catalog), Some(&usage), None).freeze();
        let manifest_json = serde_json::to_string(frozen.manifest()).expect("manifest serializes");
        for private in [
            "src/app.rs",
            "src/lib.rs",
            "E1234",
            "bubble",
            "unrelated chatter about lunch",
            "OPAQUE_ENCRYPTED_BLOB",
            "INSTRUCTIONS_TEXT",
            "TOOLS_JSON",
            "shell",
            "file list",
        ] {
            assert!(
                !manifest_json.contains(private),
                "manifest leaked private value {private}"
            );
        }
        assert!(manifest_json.contains("sha256:"));
    }

    #[test]
    fn high_confidence_catalog_applies_the_projection_on_the_wire() {
        let catalog = synthetic_catalog();
        let usage = usage();
        let fixture = standard_fixture(Some(&catalog), Some(&usage), None);
        let frozen = fixture.freeze();
        let manifest = frozen.manifest();
        assert_eq!(manifest.cost.confidence, Confidence::High);
        assert_eq!(manifest.decision.winner, Winner::Candidate);
        assert!(manifest.decision.applied);
        assert!(!manifest.fallback.used);
        assert_eq!(manifest.cost.savings_claim, "none");
        assert!(manifest.cost.unknown_fields.is_empty());
        assert_eq!(manifest.decision.reason, "lower_expected_cost");
        let expected: Vec<ResponseItem> = vec![0usize, 2, 3, 4, 10]
            .into_iter()
            .map(|seq| fixture.canonical[seq].clone())
            .collect();
        assert_eq!(
            frozen.derived_input(&fixture.canonical),
            DerivedInput::Projected(expected)
        );
    }

    #[test]
    fn missing_catalog_falls_back_with_pricing_unknown_and_no_invented_amounts() {
        let frozen = standard_fixture(None, None, None).freeze();
        let manifest = frozen.manifest();
        assert_eq!(manifest.cost.confidence, Confidence::Low);
        assert_eq!(manifest.fallback.reason, Some("pricing_unknown"));
        assert!(manifest.fallback.used);
        assert!(!manifest.decision.applied);
        assert_eq!(manifest.cost.candidate.amount, None);
        assert_eq!(manifest.cost.baseline.amount, None);
        assert_eq!(manifest.cost.net, None);
        assert!(!manifest.cost.unknown_fields.is_empty());
        assert_eq!(manifest.cost.savings_claim, "none");
        assert_eq!(manifest.cost.currency, None);
        assert_eq!(manifest.cost.unit, None);
    }

    #[test]
    fn stale_catalog_falls_back_with_pricing_stale() {
        let mut catalog = synthetic_catalog();
        catalog.freshness = Freshness::Stale;
        let usage = usage();
        let frozen = standard_fixture(Some(&catalog), Some(&usage), None).freeze();
        assert_eq!(frozen.manifest().fallback.reason, Some("pricing_stale"));
        assert!(!frozen.manifest().decision.applied);
    }

    #[test]
    fn unknown_cache_eligibility_falls_back_with_cache_ineligible() {
        let mut catalog = synthetic_catalog();
        catalog.cache_eligibility_known = false;
        let usage = usage();
        let frozen = standard_fixture(Some(&catalog), Some(&usage), None).freeze();
        assert_eq!(frozen.manifest().fallback.reason, Some("cache_ineligible"));
    }

    /// Synthetic per-term tier rates: deliberately not any provider's rates.
    fn synthetic_term_rates() -> LongContextTermRates {
        LongContextTermRates {
            uncached_input_rate: Some(5.0),
            cached_input_rate: Some(0.6),
            cache_write_rate: Some(6.0),
            output_rate: Some(7.0),
        }
    }

    #[test]
    fn long_context_threshold_is_strict() {
        let mut catalog = synthetic_catalog();
        catalog.long_context = Some(LongContextRule {
            threshold_tokens: 272_000,
            multiplier: 3.0,
            term_rates: None,
        });
        // The recorded condition is `>`, so exactly the threshold stays base.
        assert!(matches!(
            long_context_tier(&catalog, 272_000),
            LongContextTier::Base
        ));
        assert!(matches!(
            long_context_tier(&catalog, 272_001),
            LongContextTier::UniformMultiplier(multiplier) if multiplier == 3.0
        ));
        // The same strict boundary holds when the tier records per-term rates.
        let mut per_term = synthetic_catalog();
        per_term.long_context = Some(LongContextRule {
            threshold_tokens: 272_000,
            multiplier: 1.0,
            term_rates: Some(synthetic_term_rates()),
        });
        assert!(matches!(
            long_context_tier(&per_term, 272_000),
            LongContextTier::Base
        ));
        assert!(matches!(
            long_context_tier(&per_term, 272_001),
            LongContextTier::TermRates(rates) if rates == synthetic_term_rates()
        ));
    }

    #[test]
    fn uniform_long_context_multiplier_scales_the_whole_amount() {
        let mut catalog = synthetic_catalog();
        catalog.long_context = Some(LongContextRule {
            threshold_tokens: 1,
            multiplier: 3.0,
            term_rates: None,
        });
        let usage = usage();
        let frozen = standard_fixture(Some(&catalog), Some(&usage), None).freeze();
        let breakdown = frozen
            .manifest()
            .cost
            .candidate
            .breakdown
            .expect("uniform tier is costed");
        assert_eq!(breakdown.long_context_multiplier, 3.0);
        let subtotal = breakdown.uncached_input
            + breakdown.cached_input
            + breakdown.cache_write
            + breakdown.output;
        assert_eq!(breakdown.total, subtotal * 3.0);
    }

    /// Synthetic catalog whose recorded tier prices the terms separately.
    fn per_term_catalog(term_rates: LongContextTermRates) -> PricingCatalog {
        let mut catalog = synthetic_catalog();
        catalog.long_context = Some(LongContextRule {
            threshold_tokens: 1,
            multiplier: 1.0,
            term_rates: Some(term_rates),
        });
        catalog
    }

    #[test]
    fn per_term_long_context_rates_override_each_term_above_the_threshold() {
        let catalog = per_term_catalog(synthetic_term_rates());
        let usage = usage();
        // Reuse the recorded candidate epoch so the candidate bills cached and
        // uncached input while the baseline bills a cache write.
        let first = standard_fixture(Some(&catalog), Some(&usage), None).freeze();
        let candidate_epoch = first.manifest().epoch.candidate_epoch_id.clone();
        let frozen =
            standard_fixture(Some(&catalog), Some(&usage), Some(candidate_epoch.as_str())).freeze();
        let manifest = frozen.manifest();
        assert_eq!(manifest.cost.confidence, Confidence::High);
        assert!(!manifest.fallback.used);
        for basis in [&manifest.cost.candidate, &manifest.cost.baseline] {
            let breakdown = basis.breakdown.expect("per-term tier is costed");
            assert_eq!(breakdown.long_context_multiplier, 1.0);
            assert_eq!(
                breakdown.uncached_input,
                basis.tokens.uncached as f64 / 1_000_000.0 * 5.0
            );
            assert_eq!(
                breakdown.cached_input,
                basis.tokens.cached as f64 / 1_000_000.0 * 0.6
            );
            assert_eq!(
                breakdown.cache_write,
                basis.tokens.cache_write as f64 / 1_000_000.0 * 6.0
            );
            assert_eq!(
                breakdown.output,
                basis.tokens.output as f64 / 1_000_000.0 * 7.0
            );
            let subtotal = breakdown.uncached_input
                + breakdown.cached_input
                + breakdown.cache_write
                + breakdown.output;
            assert_eq!(breakdown.total, subtotal);
        }
    }

    #[test]
    fn per_term_long_context_rates_inherit_base_rates_for_unrecorded_terms() {
        let catalog = per_term_catalog(LongContextTermRates {
            uncached_input_rate: Some(5.0),
            cached_input_rate: None,
            cache_write_rate: None,
            output_rate: None,
        });
        let usage = usage();
        let first = standard_fixture(Some(&catalog), Some(&usage), None).freeze();
        let candidate_epoch = first.manifest().epoch.candidate_epoch_id.clone();
        let frozen =
            standard_fixture(Some(&catalog), Some(&usage), Some(candidate_epoch.as_str())).freeze();
        let manifest = frozen.manifest();
        assert_eq!(manifest.cost.confidence, Confidence::High);
        let candidate = manifest
            .cost
            .candidate
            .breakdown
            .expect("per-term tier is costed");
        assert_eq!(candidate.long_context_multiplier, 1.0);
        // The one recorded override applies to its own term only.
        assert_eq!(
            candidate.uncached_input,
            manifest.cost.candidate.tokens.uncached as f64 / 1_000_000.0 * 5.0
        );
        // Every other term inherits its recorded base rate (0.05 and 1.0).
        assert_eq!(
            candidate.cached_input,
            manifest.cost.candidate.tokens.cached as f64 / 1_000_000.0 * 0.05
        );
        assert_eq!(
            candidate.output,
            manifest.cost.candidate.tokens.output as f64 / 1_000_000.0 * 1.0
        );
    }

    #[test]
    fn per_term_long_context_rates_below_the_threshold_use_base_rates() {
        let mut catalog = per_term_catalog(synthetic_term_rates());
        catalog.long_context = Some(LongContextRule {
            threshold_tokens: 10_000_000,
            multiplier: 1.0,
            term_rates: Some(synthetic_term_rates()),
        });
        let usage = usage();
        let frozen = standard_fixture(Some(&catalog), Some(&usage), None).freeze();
        let manifest = frozen.manifest();
        assert_eq!(manifest.cost.confidence, Confidence::High);
        let baseline = manifest
            .cost
            .baseline
            .breakdown
            .expect("base tier is costed");
        assert_eq!(baseline.long_context_multiplier, 1.0);
        // Below the threshold no tier rate applies: the base rates are used.
        assert_eq!(
            baseline.cache_write,
            manifest.cost.baseline.tokens.cache_write as f64 / 1_000_000.0 * 2.0
        );
        assert_eq!(
            baseline.output,
            manifest.cost.baseline.tokens.output as f64 / 1_000_000.0 * 1.0
        );
    }

    #[test]
    fn catalog_for_another_model_is_unknown_pricing() {
        let mut catalog = synthetic_catalog();
        catalog.model = "some-other-model".to_string();
        let usage = usage();
        let frozen = standard_fixture(Some(&catalog), Some(&usage), None).freeze();
        assert_eq!(frozen.manifest().fallback.reason, Some("pricing_unknown"));
        assert!(
            frozen
                .manifest()
                .cost
                .unknown_fields
                .contains(&"catalog.model".to_string())
        );
    }

    #[test]
    fn changed_epoch_charges_cache_write_and_never_bills_cached_tokens() {
        let catalog = synthetic_catalog();
        let usage = usage();
        let first = standard_fixture(Some(&catalog), Some(&usage), None).freeze();
        let candidate_epoch = first.manifest().epoch.epoch_id.clone();
        let second =
            standard_fixture(Some(&catalog), Some(&usage), Some(candidate_epoch.as_str())).freeze();
        let manifest = second.manifest();
        assert!(!manifest.epoch.changed);
        assert!(manifest.cost.candidate.tokens.cached > 0);
        assert_eq!(manifest.cost.candidate.tokens.cache_write, 0);
        assert_eq!(
            manifest.cost.candidate.cache_reason,
            "epoch_unchanged_cached_prefix"
        );
        // The baseline's stable prefix differs, so it is a new epoch and is
        // charged a cache write rather than cached tokens (I13).
        assert_eq!(manifest.cost.baseline.tokens.cached, 0);
        assert_eq!(
            manifest.cost.baseline.cache_reason,
            "changed_epoch_cache_write"
        );
    }

    #[test]
    fn cache_aware_cost_can_prefer_a_large_cached_baseline_over_a_small_candidate() {
        let catalog = synthetic_catalog();
        let usage = usage();
        let first = standard_fixture(Some(&catalog), Some(&usage), None).freeze();
        let baseline_epoch = first.manifest().epoch.baseline_epoch_id.clone();
        let fixture = standard_fixture(Some(&catalog), Some(&usage), Some(baseline_epoch.as_str()));
        let second = fixture.freeze();
        let manifest = second.manifest();
        // The candidate has far fewer tokens yet loses: a token-count ranking
        // would pick the opposite winner.
        assert!(manifest.cost.candidate.tokens.total < manifest.cost.baseline.tokens.total);
        assert!(manifest.cost.baseline.tokens.cached > 0);
        assert_eq!(manifest.decision.winner, Winner::Baseline);
        assert!(!manifest.decision.applied);
        assert_eq!(manifest.decision.reason, "lower_expected_cost_baseline");
        assert!(!manifest.fallback.used);
        assert_eq!(
            second.derived_input(&fixture.canonical),
            DerivedInput::Canonical(CanonicalReason::BaselinePreferred)
        );
    }

    #[test]
    fn approximately_equal_costs_prefer_fewer_tokens_only_within_tolerance() {
        // A large anchored item plus one small unanchored item: dropping the small
        // item is a sub-1% cost change, so the explicit user tie-break applies.
        let big = "a".repeat(20_000);
        let canonical = vec![
            message("user", &format!("context for src/app.rs {big}")),
            message("assistant", "small unanchored note"),
            message("user", "fix src/app.rs"),
        ];
        let catalog = synthetic_catalog();
        let usage = usage();
        let fixture = Fixture {
            canonical,
            request_start: 2,
            catalog: Some(&catalog),
            usage: Some(&usage),
            previous_epoch_id: None,
            explicit_pins: Vec::new(),
        };
        let frozen = fixture.freeze();
        let manifest = frozen.manifest();
        assert_eq!(
            manifest.decision.reason,
            "approximately_equal_cost_fewer_tokens"
        );
        assert!(manifest.decision.applied);
        assert!(manifest.cost.candidate.tokens.total < manifest.cost.baseline.tokens.total);
    }

    #[test]
    fn a_projection_that_omits_nothing_prefers_the_baseline_without_a_fallback() {
        let canonical = vec![
            message("user", "check src/app.rs"),
            message("user", "fix src/app.rs"),
        ];
        let catalog = synthetic_catalog();
        let usage = usage();
        let fixture = Fixture {
            canonical,
            request_start: 1,
            catalog: Some(&catalog),
            usage: Some(&usage),
            previous_epoch_id: None,
            explicit_pins: Vec::new(),
        };
        let frozen = fixture.freeze();
        let manifest = frozen.manifest();
        assert_eq!(manifest.projection.omitted, 0);
        assert_eq!(manifest.decision.winner, Winner::Baseline);
        assert!(!manifest.decision.applied);
        assert!(!manifest.fallback.used);
        assert_eq!(
            frozen.derived_input(&fixture.canonical),
            DerivedInput::Canonical(CanonicalReason::BaselinePreferred)
        );
    }

    #[test]
    fn frozen_selection_appends_the_live_tool_chain_verbatim() {
        let catalog = synthetic_catalog();
        let usage = usage();
        let fixture = standard_fixture(Some(&catalog), Some(&usage), None);
        let frozen = fixture.freeze();
        assert!(frozen.applied());
        let mut turn_input = fixture.canonical.clone();
        let appended_call = call("call_9", "shell", r#"{"command":["pwd"]}"#);
        let appended_output = output("call_9", "/tmp");
        turn_input.push(appended_call.clone());
        turn_input.push(appended_output.clone());
        let mut expected: Vec<ResponseItem> = vec![0usize, 2, 3, 4, 10]
            .into_iter()
            .map(|seq| fixture.canonical[seq].clone())
            .collect();
        expected.push(appended_call);
        expected.push(appended_output);
        assert_eq!(
            frozen.derived_input(&turn_input),
            DerivedInput::Projected(expected)
        );
    }

    #[test]
    fn frozen_selection_rejects_a_changed_or_shrunk_prefix() {
        let catalog = synthetic_catalog();
        let usage = usage();
        let fixture = standard_fixture(Some(&catalog), Some(&usage), None);
        let frozen = fixture.freeze();
        let mut rewritten = fixture.canonical.clone();
        rewritten[1] = message("assistant", "rewritten history");
        assert_eq!(
            frozen.derived_input(&rewritten),
            DerivedInput::Canonical(CanonicalReason::Fallback(
                FallbackReason::CanonicalPrefixChanged
            ))
        );
        let shrunk = fixture.canonical[..5].to_vec();
        assert_eq!(
            frozen.derived_input(&shrunk),
            DerivedInput::Canonical(CanonicalReason::Fallback(
                FallbackReason::CanonicalPrefixChanged
            ))
        );
    }

    #[test]
    fn frozen_selection_rejects_an_unpaired_overlay_item() {
        let catalog = synthetic_catalog();
        let usage = usage();
        let fixture = standard_fixture(Some(&catalog), Some(&usage), None);
        let frozen = fixture.freeze();
        let mut turn_input = fixture.canonical.clone();
        turn_input.push(call("call_unpaired", "shell", "{}"));
        assert_eq!(
            frozen.derived_input(&turn_input),
            DerivedInput::Canonical(CanonicalReason::Fallback(FallbackReason::ClosureViolation))
        );
    }

    #[test]
    fn default_horizon_is_conservative() {
        let horizon = Horizon::default();
        assert_eq!(horizon.requests, 1);
        assert_eq!(horizon.reuse_probability, 0.0);
        assert_eq!(horizon.normalized().cache_write_amortization(), 1.0);
        let optimistic = Horizon {
            requests: 3,
            reuse_probability: 1.0,
        };
        assert_eq!(
            optimistic.normalized().cache_write_amortization(),
            1.0 / 3.0
        );
    }

    #[test]
    fn anchor_extraction_keeps_specific_shapes_only() {
        let anchors = extract_anchors(
            "please fix src/app.rs and codex-rs/core/src/codex.rs because E1234 broke \
             `bubble_sort` in https://example.invalid/issue/7 (see \"quoted thing\")",
        );
        assert!(anchors.contains(&"src/app.rs".to_string()));
        assert!(anchors.contains(&"codex-rs/core/src/codex.rs".to_string()));
        assert!(anchors.contains(&"e1234".to_string()));
        assert!(anchors.contains(&"bubble_sort".to_string()));
        assert!(anchors.contains(&"https://example.invalid/issue/7".to_string()));
        assert!(anchors.contains(&"quoted thing".to_string()));
        assert!(!anchors.contains(&"please".to_string()));
        assert!(!anchors.contains(&"because".to_string()));
    }

    #[test]
    fn selection_sidecar_is_rollout_adjacent_and_keeps_session_identity() {
        let rollout =
            Path::new("/tmp/sessions/2026/09/18/rollout-2026-09-18T00-00-00-000-abc.jsonl");
        assert_eq!(
            selection_sidecar_path(rollout),
            PathBuf::from(
                "/tmp/sessions/2026/09/18/rollout-2026-09-18T00-00-00-000-abc.working-set.jsonl"
            )
        );
    }

    #[test]
    fn sidecar_selection_record_is_structural_and_matches_the_manifest() {
        let catalog = synthetic_catalog();
        let usage = usage();
        let fixture = standard_fixture(Some(&catalog), Some(&usage), None);
        let frozen = fixture.freeze();
        assert!(frozen.applied());
        let record = SidecarRecord::selection(1, &frozen, digest_text("CANONICAL_BEFORE_REQUEST"));
        let json = serde_json::to_string(&record).expect("sidecar record serializes");
        assert!(json.contains("\"record\":\"selection\""));
        assert!(json.contains("\"manifest_digest\":\"sha256:"));
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid sidecar json");
        assert_eq!(
            value["canonical_digest_after_selection"],
            serde_json::json!(frozen.manifest().canonical.digest)
        );
        assert_ne!(
            value["canonical_digest_before_request"],
            value["canonical_digest_after_selection"]
        );
        assert_eq!(value["applied"], serde_json::json!(true));
        assert_eq!(
            value["selection_reason"],
            serde_json::json!(frozen.manifest().decision.reason)
        );
        for private in [
            "src/app.rs",
            "unrelated chatter about lunch",
            "shell",
            "E1234",
            "INSTRUCTIONS_TEXT",
        ] {
            assert!(
                !json.contains(private),
                "sidecar record leaked private value {private}"
            );
        }
    }
}
