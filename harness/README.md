# `harness/` — Per-invocation runtime for skill execution and verification

Spec: [`PLAN-HARNESS`](../plans/05-harness/PLAN-HARNESS.md), [`PLAN-COMPONENTS-IMPLEMENTATION`](../plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md) §4 (Phase A).

The harness is the *frozen verifier* in the role split (root README §"Three-agent role split" / §"Architecture"):

> The Skill Bank provides candidates, the **Harness narrows + may veto**, the Actor decides, the Orchestrator handles offline promotion.

Concretely, the harness's job for one skill invocation is six steps:

1. **Filter** the bank's candidate set to a domain- and contract-eligible subset (`select_eligible_skills`).
2. **Score** every kept skill with `fit_score` + `risk_score` and per-check booleans (`binding_ok / precondition_ok / evidence_ok / adapter_ok`). _Currently absent — see [§9](#9-online-surface-api-gaps-validate_invocation-scoring-intentionactive_skill-inputs)._
3. **Validate the invocation** — once the Actor has bound slots and proposed a skill, the harness re-checks and may veto (`validate_invocation`). _Currently absent — see [§9](#9-online-surface-api-gaps-validate_invocation-scoring-intentionactive_skill-inputs)._
4. **Execute** one chosen skill via the right `SkillAdapter` (`run_skill`).
5. **Record** everything as a `SkillEpisode` + reward-log entry — this is where invariant **G0 (evidence-driven)** bites via `SkillEpisode.finalize()`.
6. **Provide replay validation** for the gate (Stage 1, [`PLAN-UNIFIED-SKILL-GATE`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) §7).

The harness has **two surfaces** sharing this module:

| Surface | When it runs | Public API |
|---|---|---|
| **Online runtime** | Per actor decision (steps 1–5 above). Inputs: `(schema_state, intention, retrieved_skills, active_skill, local_reasoning_trace)`. Outputs: `eligible_skills` + `invocation_veto` + `SkillEpisode`. | `select_eligible_skills`, _`validate_invocation` (pending)_, `run_skill` |
| **Offline `GateRunner`** | Per Crafter `BankMutationProposal`. Inputs: `(proposal, candidate_skill, replay_seeds, shadow_log/rollout_batch, target_domains, FewShotDemo[], frozen_eval_suite)`. Outputs: per-stage `GateVerdictPayload` + roll-up `SkillEvaluationRecord`. | Today composed in `orchestrator/gate_service.py` — a harness-side aggregator (`harness/gate_runner.py`) is owed (see [§3](#3-the-six-gate-runner-and-the-transfer-manager-dont-exist-yet)) |

It does **NOT**:
- Choose which skill to commit to — that's the Actor.
- Mutate `SkillRecord.status` — only `SkillLifecycleManager` may.
- Read/write a "memory" buffer — invariant 2 (no-memory).
- Train policies / call the teacher — that's the offline Orchestrator + Crafter.

```python
from harness import (
    SkillHarness, HarnessConfig,
    AdapterRegistry,
    SkillAdapter, AdapterRunContext, AdapterRunResult,
    EligibilityFilter, EligibleSkill,
    ReplayValidator, ReplayResult,
    RewardLogger,
    FewShotAdapter, FewShotDemo, AdaptResult,
)
```

---

## Module map

| File | Role |
|---|---|
| `skill_harness.py` | `SkillHarness` — the public entry point. `select_eligible_skills(state, candidates)` and `run_skill(skill, state, env, …)`. Owns the per-invocation lifecycle: eligibility filter → adapter run → tracing → `SkillEpisode` → replay validation |
| `eligibility.py` | `EligibilityFilter` — matches `SkillRecord.contract` (typed input slots, pre-conditions, domain) against the current `StateSchema`; returns `EligibleSkill[]` annotated with the slot bindings the adapter will need |
| `skill_adapter.py` | `SkillAdapter` base class + `AdapterRunContext` / `AdapterRunResult`. The contract every per-domain adapter must implement |
| `adapter_registry.py` | `AdapterRegistry` — domain → `SkillAdapter` lookup. Adapters self-register on `register(adapter)` |
| `adapters/` | Per-domain implementations: `gymv_adapter.py` (Gym-V games), `browser_adapter.py` (BrowserGym), `osworld_adapter.py` (OSWorld desktop), `video_adapter.py`, `visual_reasoning_adapter.py`. Plus `_common.py` (shared slot-binding helpers) and `_stub_base.py` (test scaffolding) |
| `replay_validator.py` | `ReplayValidator` — replays a `SkillEpisode` against a frozen environment snapshot; returns `ReplayResult` for Stage 1 of the gate. Currently a Phase-A stub; the full deterministic-replay path lands in Phase D |
| `reward_logger.py` | `RewardLogger` — append-only JSONL of per-step `r_env / r_follow / r_cost / r_total`. The orchestrator's `BudgetController` reads `r_cost` to enforce per-episode budgets |
| `few_shot_adapter.py` | `FewShotAdapter.adapt(skill, demos)` — Stage 3a (transfer) of the gate. Takes K target-domain demos, runs them, and emits the `eligible_domains` list that gets mirrored into `SkillRecord.verified_domains` (invariant 8). Currently the only path that legitimately produces `verified_domains` evidence |

---

## Adapter contract

Every concrete adapter under `adapters/` inherits `SkillAdapter` and implements:

```python
class SkillAdapter(ABC):
    domain: str          # one of common.enums.DOMAINS
    name: str

    def can_handle(self, skill: SkillRecord, state: StateSchema) -> bool: ...
    def bind_slots(self, skill, state) -> SlotBinding: ...
    def run(self, ctx: AdapterRunContext) -> AdapterRunResult: ...
```

`AdapterRunResult` carries the executed action sequence, the produced evidence (mapped to the skill's `expected_evidence_roles`), and a `success: bool` flag. The harness wraps the run in a `SkillEpisode`; if any non-`ACTION` step has empty evidence, `SkillEpisode.finalize()` raises and the episode is marked `failed_contract` (invariant G0).

---

## Phase boundaries

| Phase | What this package contains | Status |
|---|---|---|
| A (MVP) | `SkillHarness`, eligibility filter, gymv + browser adapters, reward log, replay-validator stub | **Delivered** — covered by `tests/test_smoke.py` and `tests/test_invariants.py` |
| D (transfer + replay) | Full deterministic `ReplayValidator`; six-gate `GateRunner` (G0–G5); `transfer_manager.py`; osworld / video / visual_reasoning adapters | Pending — see root README §"Pending" |
| F (trainable extensions) | LoRA heads `skill_select`, `continue_vs_switch`, `accept_transfer`, `adapter_refine` consumed by `SkillHarness.select_eligible_skills` | Pending |

---

## What's missing today (pending work in this package)

Grouped by impact. Items 1–4 are blocking the "harness as gate verifier" story; items 5–8 are smaller correctness / completeness gaps inside the existing files.

### 1. Transfer-target adapters are deterministic stubs

Only `gymv_adapter.py` is real. `browser`, `osworld`, `video`, and `visual_reasoning` all inherit `StubTransferTargetAdapter` (`adapters/_stub_base.py`) and run `make_deterministic_executor`, which never touches a real environment — it echoes the action back and emits one synthetic `GATHER` evidence ref per hop just to keep G0 satisfied. Real env binding is owed by `vlm_wrapper/<domain>_adapter.py` and must be plugged in via `adapter.set_executor(real_executor)`.

### 2. `ReplayValidator` is a "dry-run rerun", not a real replay

`replay_validator.py` calls the adapter again from the seed's *initial* state with `dry_run=True`. It never:

- walks `seed.steps` action-by-action,
- compares post-states or evidence locators across the original and replayed runs,
- consults `orchestrator/SnapshotManager` for a frozen environment snapshot.

The PASS threshold is also hardcoded (`0.8` in `as_stage_verdict`) and there is no curated **held-out replay-seed corpus** — `validate(seeds=...)` requires the caller to bring its own.

### 3. The six-gate runner and the transfer manager don't exist yet

Two files named in the plan are not in the package:

| Missing file | Owes |
|---|---|
| `harness/gate_runner.py` | Unified `GateRunner` walking G0–G5 against `gate_service` stages; today only `orchestrator/gate_service.py` composes the stages, and there is no harness-side aggregator that owns the final verdict |
| `harness/transfer_manager.py` | Two-phase **shadow → active** protocol layered on top of Stage 3a (`FewShotAdapter`); without it a transfer pass goes straight to "verified" with no shadow-deploy quarantine |

### 4. `FewShotAdapter` runs, but the scorer is a placeholder

`few_shot_adapter.py` itself is wired correctly. What it doesn't ship:

- **Domain-aware scorer.** `default_success_fn` only checks `episode.outcome.success and contract_satisfied`; it never compares `episode.outcome.answer` to `demo.expected`. The docstring explicitly defers this to "the orchestrator's transfer-eval driver", which has not yet plugged in a real `success_fn`.
- **Real demo corpus.** When `demos` is empty, `adapt(...)` falls back to a synthetic empty-state probe and tags the verdict `target_domain_demo_unavailable`. There is no curated `FewShotDemo` set per target domain.
- **One-sided cost stop.** The K-shot loop only breaks on `cost_tokens > _max_tokens`; `cost_ms` is tracked but never used as a stop condition.

### 5. Trainable selection heads (Phase F) — completely absent

`select_eligible_skills` is purely the deterministic `EligibilityFilter` (contract + slot-binding match). The Phase-F LoRA heads — `skill_select`, `continue_vs_switch`, `accept_transfer`, `adapter_refine` — that are supposed to consume the harness's candidate set are not implemented.

### 6. Budget is collected but not enforced by the harness itself

`SkillHarness._effective_budget` builds `{tokens, hops, ms}` and passes it into `AdapterRunContext`, but the only place that actually checks budget is `BudgetGuard` in `_stub_base.py`, and only for `hops` and `ms`. The harness never:

- compares `result.cost` against the budget after the run,
- aborts mid-run on token overrun,
- reads `r_cost` back from `RewardLogger` to feed the orchestrator's `BudgetController` (today the path is writer-only).

### 7. `_record_failure` always blames the last step

```python
last_step = episode.steps[-1] if episode.steps else None
trace = FailureTrace(..., failed_step_index=last_step.step_index if last_step else None, ...)
```

In a multi-hop adapter run the failing step isn't necessarily the tail. The adapter result has no slot to communicate "step N failed", so the harness has no way to record it accurately.

### 8. `HarnessConfig.fail_on_missing_adapter` is a no-op

Both branches of the `if self._config.fail_on_missing_adapter:` guard inside `run_skill` currently `return episode`. Either the flag should drive distinct behavior (e.g. raise vs. return a failed episode) or be removed.

---

## Spec-contract gaps (audit: 2026-04-30)

Items 1–8 above are runtime/execution holes inside files that already exist. Items 9–14 below are gaps in the **API surface and persisted-artefact shapes** between [`PLAN-HARNESS`](../plans/05-harness/PLAN-HARNESS.md) / [`PLAN-UNIFIED-SKILL-GATE`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) and what the live code actually exposes. They were surfaced while scoping a `labeling_supplement/dump_harness_io_gpt54.py` driver that exercises the live harness against the cold-start corpus.

### 9. Online-surface API gaps (`validate_invocation`, scoring, intention/active_skill inputs)

The README intro lists the harness as "narrows + may veto", but only the narrowing exists. Concretely:

  - **`SkillHarness.validate_invocation` is missing entirely.** A `grep validate_invocation` across the package returns zero hits in any `.py` file (only plan documents). `run_skill` (`skill_harness.py:82`) goes straight from `(skill, state, bindings)` to `adapter.run` with no second-pass check. Slot-binding errors (e.g. `MERGE` adapter on non-adjacent tiles, `VERIFY_EVIDENCE` without an evidence span) only surface as adapter-level `abort_reason` — too late to refuse the invocation.
  - **No scoring path.** `EligibilityFilter` is explicit at `eligibility.py:16` ("we never *score* skills here"). Spec asks for `fit_score` + `risk_score` per kept skill; neither exists.
  - **`select_eligible_skills(candidates, state, *, skill_type_hint)` (`skill_harness.py:73`) ships only a strict subset of the spec'd inputs:**

| Spec input | Current parameter | Status |
|---|---|---|
| `schema_state` | `state: StateSchema` | ✓ |
| `retrieved_skills` | `candidates` | ✓ |
| `intention` | — | missing |
| `active_skill` | — | missing |
| `local_reasoning_trace` | — | missing |

  - **`EligibleSkill.to_json()` (`eligibility.py:44–52`) emits only** `{skill_id, skill_name, skill_status, adapter_name, shadow_only, reasons}` — missing the per-skill check booleans (`binding_ok / precondition_ok / evidence_ok / adapter_ok`) and the rejected-skill `veto / veto_reason` channel. Rejected candidates are silently dropped today, so the actor cannot reason about why a skill was excluded.

### 10. `SkillEpisode` artefact field gaps

The harness's most visible artefact (`data_structure/extensions/skill_episode.py`) is missing fields that the gate, the Crafter, and any future I/O dump need:

| Spec field | Status | Gap |
|---|---|---|
| `evidence_role` | ✓ | on `SkillEpisodeOutcome.evidence_role` (line 67) |
| `evidence_in / evidence_out` split | ❌ | only one uni-directional `SkillEpisodeStep.evidence: List[EvidenceRef]` (line 37). Crafter / GateRunner cannot tell "what did the skill consume?" from "what did it produce?" |
| `evidence_warrant` / `verify_verdict` / `reason_warrant` | ❌ | no citation slots on outcome or step |
| `protocol_trace` (mapping `episode.steps[i] → skill.protocol[k]`) | ❌ | `steps` is the adapter's raw step record, not a structured trace against `skill.protocol[]`. The Repairer can therefore only patch in the dark — this is the §7.1 mismatch #2 in [`../implementation_notes/crafter-harness-orchestrator-roles.md`](../implementation_notes/crafter-harness-orchestrator-roles.md) made concrete |
| `contract_progress` (per-key) | ❌ | only `outcome.contract_satisfied: bool` (line 65). No per-key (effects_add fired, effects_del fired, expected_evidence_role fired) granularity |
| `reward_components` | ❌ | only scalar `outcome.score: Optional[float]` (line 69). `cost: Dict[str, float]` (line 114) carries token/hop/ms but is not the multi-component reward the spec implies |
| `shadow` flag on episode | 🟡 | `EligibleSkill.shadow_only: bool` (`eligibility.py:43`) is set at filter time but never propagates into `SkillEpisode` or `RewardLogEntry`. Stage 2 therefore cannot distinguish shadow-mode failures from real-mode failures when reading back the log |
| `diagnostic_labels` (list) | 🟡 | only a single `transfer_label: Optional[str]` (line 115). G0-violation tagging is currently routed to a separate `FailureTrace` via `SkillHarness._record_failure` (line 219), not to the episode itself |

### 11. `SkillEvaluationRecord` reproducibility-anchor gaps

The roll-up the orchestrator reads (`data_structure/extensions/skill_evaluation.py`) does not pin the run to a reproducible context:

| Spec field | Status | Gap |
|---|---|---|
| `skill_id`, `final_decision`, `decision_reason`, per-stage payloads | ✓ | via `verdict.{stages, final_verdict, rationale}` |
| `version` | 🟡 | only `skill_content_hash` (a fingerprint, not a version string) |
| `status_before` / `status_after` | ❌ | not recorded |
| `approved_domains` | 🟡 | recorded as `verdict.eligible_domains` (renamed) |
| `rejected_domains` | ❌ | not recorded; consumer must compute `target_domains \ eligible_domains` |
| `rollback_target` | ❌ | not recorded |
| `bank_snapshot_id` | ❌ | **the most important gap** — the gate evaluation is not snapshot-pinned. `SnapshotManager` exists (`orchestrator/snapshot_manager.py`) and `RunRelease.bank_snapshot_path` is recorded *on promotion* (`orchestrator/promotion_orchestrator.py:146`), but two evaluations against different snapshots are indistinguishable on disk today |
| `eval_suite_id`, `adapter_versions`, `ontology_version` | ❌ | no record of which eval suite, which adapter versions, or which schema/ontology version the verdict was emitted against — blocks reproducible audit |
| `diagnostic_labels` (flat list) | 🟡 | only `transfer_labels: Dict[str, int]` (a histogram). Per-stage `StageVerdict.failures: List[str]` is the closest stand-in |

### 12. `GateService` stage I/O signatures don't match the spec

`orchestrator/gate_service.py:91 GateService.evaluate(...)` composes all five stages, but two diverge from what the spec calls out:

  - **Stage 2 shadow** (`_run_shadow`, line 193) takes `Optional[RewardLogger]` — *not* a `rollout_batch[]`. It reads via `log.filter(skill_id=...)` from the in-process logger.
  - **Stage 4 non-regression** (`_run_non_regression`, line 350) takes scalar `baseline_score` / `post_score` — *not* a frozen `eval_suite[]` reference. There is no `eval_suite_id` recorded anywhere.

Both are additive fixes (overload to accept the new shapes), but they need to be acknowledged or any I/O-dump driver will trip on them.

### 13. `GateService` lives under `orchestrator/`, not `harness/` — naming mismatch with the spec

The spec calls it the "Harness `GateRunner`". The live composition lives in [`../orchestrator/gate_service.py`](../orchestrator/gate_service.py); the harness only owns the leaf primitives (`ReplayValidator`, `FewShotAdapter`). Item 3 above already names `harness/gate_runner.py` as missing. The architectural choice is defensible (the orchestrator owns stage composition), but consumers reading the spec will look in `harness/` and find nothing — the rename / relocate is the remaining cosmetic fix once items 9–12 land.

### 14. No I/O dump driver — live harness behaviour against the cold-start corpus is unverified

Nothing today drives the live `SkillHarness` against `labeling/skill_actions_out/` to validate that:

  - the eligibility narrowing agrees with the cold-start actor's bound `skill_query.selected_skill_id`,
  - `run_skill` produces non-empty `SkillEpisode`s on the gymv adapter for the existing `(corpus, source)` pairs,
  - `replay_validate` round-trips synthesized seeds from `labeling/skill_bank_out/.../sub_episodes.json`.

Verifying these is the prerequisite for the actor rewire (`HarnessSkillProvider` per [`../IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §"Not yet delivered"): we need numerical evidence that the harness behaves as designed before it goes in front of the actor.

The natural fit is a sibling driver under [`../labeling_supplement/`](../labeling_supplement/) that mirrors the existing `decide_skill_crafting_gpt54.py` / `reflect_per_episode_gpt54.py` pattern. Data plumbing per surface:

| Driver surface | Bank | Per-step / per-rollout | Proposals |
|---|---|---|---|
| **Online dump** (eligibility, validate_invocation, run_skill, SkillEpisode) | `labeling/skill_bank_out/run_<ts>/<corpus>/<source>/skill_bank.jsonl` | `labeling/skill_actions_out/run_<ts>/<corpus>/<source>/episode_*.json` (state, intention, retrieved_skills, active_skill, raw_intentions, ground-truth `skill_query.selected_skill_id` for agreement metric) | n/a |
| **Offline GateRunner dump** (Stage 0–4, `SkillEvaluationRecord`) | `labeling/skill_bank_out/...` (skill the proposal mutates) | `labeling/skill_actions_out/...` (replay seeds + shadow log + cross-source few-shot demos + `_skill_actions_summary.json` non-regression baseline) | `labeling_supplement/crafter_proposals_out/run_<ts>/...` and `labeling_supplement/episode_reflections_out/run_<ts>/...` |

Note the layering: `labeling_supplement/` only contains *what to verify* (Crafter proposals); the rollouts the harness consumes — replay seeds, shadow logs, transfer demos, non-regression baselines — all still live in `labeling/`.

### Outside this package, but blocking its value

- **Actor rewire.** `decision_agents.skill_interface.SkillBankProvider` still queries the bank directly. Until it is replaced by a `HarnessSkillProvider` that wraps `SkillHarness.select_eligible_skills`, the "harness narrows + may veto" rule is not in force at runtime. Tracked in [`../IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §"Not yet delivered".
- **Real domain executors** under `vlm_wrapper/<domain>_adapter.py` (item 1's other half).

### Suggested work-order

Smallest cost / highest value first. The reordering below puts the audit's [§9–§14](#spec-contract-gaps-audit-2026-04-30) items ahead of items 1–8 because they are pure additive contract fixes (no behavioural break) and they unblock the I/O-dump driver that validates everything else.

  1. Add `SkillHarness.validate_invocation(skill, state, bindings, *, intention, reasoning_trace) -> {veto, veto_reason, diagnostic_labels}` and propagate `EligibleSkill.shadow_only` into `SkillEpisode` / `RewardLogEntry` ([§9](#9-online-surface-api-gaps-validate_invocation-scoring-intentionactive_skill-inputs), [§10](#10-skillepisode-artefact-field-gaps) shadow row).
  2. Extend `EligibleSkill` with the per-skill check booleans (`binding_ok / precondition_ok / evidence_ok / adapter_ok`), `fit_score`, `risk_score`, and `veto / veto_reason` for rejected candidates ([§9](#9-online-surface-api-gaps-validate_invocation-scoring-intentionactive_skill-inputs)). Pure additive on `eligibility.py`.
  3. Extend `SkillEpisode` with `evidence_in / evidence_out` split, `evidence_warrant / verify_verdict / reason_warrant`, `protocol_trace` (index from `episode.steps[i]` to `skill.protocol[k]`), per-key `contract_progress`, structured `reward_components`, and a list-typed `diagnostic_labels` ([§10](#10-skillepisode-artefact-field-gaps)). Pure additive on `data_structure/extensions/skill_episode.py`.
  4. Extend `SkillEvaluationRecord` with the reproducibility anchors `bank_snapshot_id`, `eval_suite_id`, `adapter_versions`, `ontology_version`, plus `status_before / status_after / rejected_domains / rollback_target` ([§11](#11-skillevaluationrecord-reproducibility-anchor-gaps)). Wire `bank_snapshot_id` through `GateService.evaluate(...)`.
  5. Stand up `labeling_supplement/dump_harness_io_gpt54.py` (online surface only) per [§14](#14-no-io-dump-driver--live-harness-behaviour-against-the-cold-start-corpus-is-unverified). Doubles as the integration test for items 1–4.
  6. Plug a real domain-aware `success_fn` into `FewShotAdapter` (existing item 4).
  7. Wire the legacy actor to `HarnessSkillProvider`.
  8. Implement action-level `ReplayValidator` (walk `seed.steps`, compare actions + evidence) — existing item 2.
  9. Stand up `harness/gate_runner.py` over `gate_service` stages — existing item 3 + [§13](#13-gateservice-lives-under-orchestrator-not-harness--naming-mismatch-with-the-spec) relocate.
  10. Extend the dump driver to the offline GateRunner surface (Stage 0–4 + `SkillEvaluationRecord` per Crafter proposal) — second half of [§14](#14-no-io-dump-driver--live-harness-behaviour-against-the-cold-start-corpus-is-unverified).
  11. Add `rollout_batch[]` overload on `_run_shadow` and `eval_suite[]` overload on `_run_non_regression` ([§12](#12-gateservice-stage-io-signatures-dont-match-the-spec)).
  12. Add `harness/transfer_manager.py` for shadow → active — existing item 3 second half.
  13. Wire `vlm_wrapper/<domain>_adapter.py` executors via `set_executor()` — `browser` → `osworld` → `video` → `visual_reasoning` (existing item 1).
  14. Phase-F LoRA heads in `select_eligible_skills` (existing item 5).

---

## Cross-references

- Root [`readme.md`](../readme.md) §"Architecture" — where the harness sits in the four-stage pipeline.
- [`../plans/05-harness/PLAN-HARNESS.md`](../plans/05-harness/PLAN-HARNESS.md) — full spec, gate stack G0–G5.
- [`../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) — `GateRunner` stages and aggregation contract that drives [§11–§13](#11-skillevaluationrecord-reproducibility-anchor-gaps).
- [`../skill_bank/README.md`](../skill_bank/README.md) — invariant 8 ties `FewShotAdapter` to `SkillLifecycleManager.record_transfer_verification`.
- [`../implementation_notes/crafter-harness-orchestrator-roles.md`](../implementation_notes/crafter-harness-orchestrator-roles.md) — three-role I/O contract; §3 cheat sheet enumerates the artefact families [§10–§11](#10-skillepisode-artefact-field-gaps) extend; §7.1 mismatch #2 motivates [§10](#10-skillepisode-artefact-field-gaps)'s `protocol_trace` row.
- [`../labeling_supplement/`](../labeling_supplement/) — sibling location for the `dump_harness_io_gpt54.py` driver in [§14](#14-no-io-dump-driver--live-harness-behaviour-against-the-cold-start-corpus-is-unverified).
- [`../tests/test_smoke.py`](../tests/test_smoke.py) — runnable end-to-end wiring example.
