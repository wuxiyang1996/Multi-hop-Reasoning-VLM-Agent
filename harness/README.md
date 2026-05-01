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
| `protocol_trace` (mapping `episode.steps[i] → skill.protocol[k]`) | ❌ | `steps` is the adapter's raw step record, not a structured trace against `skill.protocol[]`. The Repairer can therefore only patch in the dark — this is the §7.1 mismatch #2 in [`../implementation_notes/crafter-harness-orchestrator-roles.md`](../implementation_notes/crafter-harness-orchestrator-roles.md) made concrete. **Also depends on [§21](#21-cold-start-protocol-is-natural-language-prose-not-typed-hops)** — there is no `skill.protocol[k]` to index against until cold-start protocols are lifted to typed hops |
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

> **Status (2026-04-30, Day-1 of intra-gymv transfer milestone):** Online-surface dump driver landed at [`../labeling_supplement/dump_harness_io_gpt54.py`](../labeling_supplement/dump_harness_io_gpt54.py) with bash dispatcher [`../labeling_supplement/run_dump_harness_io.sh`](../labeling_supplement/run_dump_harness_io.sh). First Phase-0 baseline (twenty_forty_eight + tetris, 6 episodes / 277 steps, online surface only — no `--run-skill`) recorded in [`../labeling_supplement/harness_io_out/run_phase0_20260501_012106/_run_summary.json`](../labeling_supplement/harness_io_out/run_phase0_20260501_012106/) and analysed in [`../labeling_supplement/harness_io_out/_phase0_report.md`](../labeling_supplement/harness_io_out/_phase0_report.md).
>
> One real driver-shape limitation surfaced: the dump driver feeds the actor's per-step `retrieved_skill_ids` (cold-start RAG) as candidates to `select_eligible_skills`. That RAG was built per-game, so cross-game IDs never reach the filter — the dump driver alone cannot exhibit the §22 failure mode. A standalone bypass probe ([`../labeling_supplement/_phase0_cross_eligibility_probe.py`](../labeling_supplement/_phase0_cross_eligibility_probe.py)) feeds all skills in a fused bank as candidates and produces the §22 measurement; it's hardened against `safe_skill_id` collisions across games (e.g. tetris and super_mario both ship `INSPECT/SETUP`). Offline-surface dump and replay-seed round-trip remain unverified.

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

### 21. Cold-start `protocol` is natural-language prose, not typed hops

> **Status (2026-04-30):** Day-1 design lock for the lift landed at [`../implementation_notes/protocol-lift-design.md`](../implementation_notes/protocol-lift-design.md). Two-loaded-shapes nuance discovered: the audit's "zero hops" framing is correct only for the direct-from-jsonl load path. The dump-driver / probe path already passes prose through [`../labeling_supplement/_harness_io_helpers.py:_wrap_protocol_steps`](../labeling_supplement/_harness_io_helpers.py), which emits `[{"action": "EXEC", "payload": {}, "notes": "<prose>"}, …]` — `iter_hops` yields N hops per skill, but every hop normalises to `"EXEC"` with empty payload. **The shape gap is closed via this workaround; the semantic gap (real verbs, populated `${slot}` placeholders, typed effects) is open.** Empirical sweep across all 80 prose steps in `run_20260430_030637`: a 21-verb gymv-only taxonomy mined from the schema's `<affordances>` block plus subordinator-stripping + downstream-walk classifier covers **74 / 80 = 92.5 %** of prose steps. Implementation lands in [`../labeling/_decorate_skill_records.py`](../labeling/_decorate_skill_records.py) (Day 2). One small pre-existing bug in the schema parser surfaced en route — [`../labeling_supplement/_harness_io_helpers.py:parse_schema_canonical`](../labeling_supplement/_harness_io_helpers.py) does not parse the `<attributes>` block, so `state.facts` only carries `{"goal": ...}`; the success-fn (Day 4–5) needs either a parser extension (Option A) or a direct read from `step.state` (Option B). Tracked in §5.1 of the design doc.

Every adapter — gymv and the four transfer-target stubs — walks `skill.protocol` via `iter_hops`:

```40:44:Multi-hop-Reasoning-VLM-Agent/harness/adapters/_common.py
def iter_hops(skill: SkillRecord) -> Iterator[Tuple[int, Dict[str, Any]]]:
    for i, hop in enumerate(skill.protocol):
        if not isinstance(hop, dict):
            continue
        yield i, hop
```

The contract is `protocol: List[Dict[str, Any]]` (`data_structure/extensions/skill_record.py:89` — *"# ordered hop list"*), where each hop is a dict carrying `action / op / type` plus `${slot}` placeholders the binder can resolve.

The cold-start labeling pipeline produces a different shape. From `labeling/skill_bank_out/run_<ts>/env_wrappers/twenty_forty_eight/skill_bank.jsonl`, skill `COMMIT/MERGE`:

```json
"protocol": {
  "preconditions": [...],
  "steps": [
    "Read the current board and enumerate the four possible slide directions: up, down, left, right.",
    "Determine which directions are legal by checking whether sliding in that direction would move any tile or merge any adjacent equal tiles.",
    "Select the intended direction according to the controlling policy or plan.",
    "Execute a single slide of the full board in the selected direction.",
    ...
  ],
  "success_criteria": [...],
  "abort_criteria": [...],
  "expected_duration": 10
}
```

Two structural problems:

1. **Wrong outer shape.** `protocol` is a *dict-of-string-lists*, not a *list-of-dicts*. When passed through `iter_hops`, the for-loop iterates the dict's keys (strings), each of which fails the `isinstance(hop, dict)` check. **Zero hops are yielded.** No executor call ever happens.
2. **Wrong inner content even if the shape is fixed.** Each step is a natural-language sentence (*"Slides the board in a direction so matching adjacent tiles combine"*), not `{"action": "SLIDE", "args": {"direction": "${dir}"}}`. There is no abstract verb, no typed slot, no `${binding}` placeholder. The contract's `eff_add` / `eff_del` arrays are also empty — there is no formal effect specification to test "did the skill actually do what it promised."

Why this is masked today: every gymv-and-transfer adapter falls back to `_deterministic_executor`, which echoes any input action and emits one synthetic `GATHER` evidence ref ([§1](#1-transfer-target-adapters-are-deterministic-stubs)). The episode that comes back has the right *shape* but no real grounding. The moment a real executor is wired ([§16.1](#161-adapter-executors-are-stubs-so-run_skill-is-a-black-hole)), the prose-protocol input has nowhere to go and `harness.run_skill(...)` will return a no-op episode — even on a same-game state.

This is the upstream blocker for several of the audit items already on the list:

  - [§1](#1-transfer-target-adapters-are-deterministic-stubs) — real adapters can only consume typed hops.
  - [§4](#4-fewshotadapter-runs-but-the-scorer-is-a-placeholder) — `default_success_fn` cannot test "skill did the right thing in target domain" if the skill has no executable semantics in any domain.
  - [§10](#10-skillepisode-artefact-field-gaps) `protocol_trace` row — there is no `skill.protocol[k]` to index `episode.steps[i]` against.

The fix is **not** in `harness/`. It belongs in the cold-start labeling pipeline (`labeling/_decorate_skill_records.py` or a sibling Crafter-style transformer) that lifts `protocol.steps: List[str]` into `protocol: List[Dict]` with abstract verbs and types `eff_add` / `eff_del` from the schema-canonical evidence. The harness inherits the contract; it cannot repair it.

### 22. `feasible_domains` granularity collapses gymv games into a single bucket

> **Status (2026-04-30):** Empirically confirmed at 100 % cross-contamination on `twenty_forty_eight ↔ tetris`. Probe at [`../labeling_supplement/_phase0_cross_eligibility_probe.py`](../labeling_supplement/_phase0_cross_eligibility_probe.py); fused 9-skill bank, 277 steps total. Result: every cold-start `COMMIT` / `GATHER`-typed skill from the *other* game is admitted on every step (`6/6` tetris skills on every 2048 step; `2/3` 2048 skills — both `ACTION`-typed — on every tetris step). Only `COMPARE/MERGE` (2048) is filtered, and only because it's `evidence_role=REASON → SkillType.REASONING` and no `(gymv, REASONING)` adapter is registered — orthogonal to §22. The probe measures **admission only**; the agreement-impact follow-up (does cross-game admission steal the actor's pick?) is a Day-2 extension noted in [`../labeling_supplement/harness_io_out/_phase0_report.md`](../labeling_supplement/harness_io_out/_phase0_report.md) §4. Fix lands as the additive contract change called out below.

`SOURCE_DOMAINS = ("gymv",)` has no per-game cell:

```30:36:Multi-hop-Reasoning-VLM-Agent/common/enums.py
SOURCE_DOMAINS: Tuple[str, ...] = ("gymv",)
TRANSFER_TARGET_DOMAINS: Tuple[str, ...] = (
    "browser",
    "osworld",
    "video",
    "visual_reasoning",
)
```

Every cold-start skill is tagged `applicable_domains=["gymv"]`; per-game info lives only in two free-form metadata fields:

  - `state.task = "make_gaming_env/<game>"` (parsed from `metadata.schema_canonical`),
  - `skill.provenance.source_name = "<game>"` (set by the cold-start ingester).

Neither is read by `EligibilityFilter` or `FewShotAdapter`. Three consequences:

  - **A.** `EligibilityFilter` admits a 2048-mined skill into a tetris episode — both have `state.domain == "gymv"`, the filter never reads `state.task`.
  - **B.** `FewShotAdapter.adapt(skill, target_domain="gymv", demos=[tetris_demos])` is semantically undefined — the skill *already* claims `feasible_domains=["gymv"]`, so the gate has nothing to verify. The Stage 3a transfer machinery only models cross-domain transitions.
  - **C.** `verified_domains` has domain granularity. There is no field to record "verified on tetris specifically." A skill that passes a real intra-gymv probe on tetris cannot have that fact persisted.

This blocks intra-gymv cross-game transfer experiments (2048 → tetris, candy_crush → 2048, …), which are otherwise the cheapest first transfer milestone — see the callout below.

The fix is an additive contract change: add `SkillRecord.feasible_tasks: List[str]` and `verified_tasks: List[str]`, plumb a `target_task: Optional[str] = None` parameter through `FewShotAdapter.adapt(...)` and `GateService._run_transfer(...)`, and have `EligibilityFilter` honour both `domain` and (when set) `task`. Cold-start ingestion seeds `feasible_tasks=[provenance.source_name]`.

### Intra-gymv transfer is the right first milestone

[§21](#21-cold-start-protocol-is-natural-language-prose-not-typed-hops) and [§22](#22-feasible_domains-granularity-collapses-gymv-games-into-a-single-bucket) together suggest a smaller first transfer experiment than gymv → browser/osworld. Pinning the first transfer cycle inside gymv (2048 → tetris → candy_crush → …) shrinks four of the six cost axes:

| Cost axis | gymv → browser/osworld/video/vr | gymv → gymv (cross-game) |
|---|---|---|
| Adapter executors to wire ([§1](#1-transfer-target-adapters-are-deterministic-stubs) / [§16.1](#161-adapter-executors-are-stubs-so-run_skill-is-a-black-hole)) | 4 transfer + 1 source = 5 | **1** (`GymvAdapter`) |
| New env bindings | full browser DOM, VM control, frame indexer, MCQ resolver | **none** — `cold_start/generate_cold_start_actor*.py` already drives the envs; expose its `step()` |
| Demo corpus ([§4](#4-fewshotadapter-runs-but-the-scorer-is-a-placeholder)) | doesn't exist | **already on disk** — every `labeling/skill_actions_out/.../<game>/episode_*.json` is a real rollout with state, action, intention, ground-truth `skill_query.selected_skill_id` |
| Domain-aware `success_fn` ([§4](#4-fewshotadapter-runs-but-the-scorer-is-a-placeholder)) | per target (DOM diff, screen diff, video QA, …) | **single gymv-shape scorer** keyed on consecutive `schema_canonical` blocks + `cumulative_reward` |
| Slot-binding ontology | cross-modal — `tile → DOM_node`, `direction → click`, … | gymv-internal — abstract verbs over `selectable_entity` / `container_entity` / `direction` |
| Protocol lift ([§21](#21-cold-start-protocol-is-natural-language-prose-not-typed-hops)) | needed | needed (same lift, but only over gymv-shaped skills) |

Five out of six axes are dramatically smaller. The protocol lift ([§21](#21-cold-start-protocol-is-natural-language-prose-not-typed-hops)) is the same cost either way and is the hard prerequisite — without it, `harness.run_skill(skill, state)` produces a no-op episode even on a *same-game* state. So the order is: first prove `harness.run_skill(COMMIT/MERGE, twenty_forty_eight_state)` actually executes (lift + gymv executor); then add [§22](#22-feasible_domains-granularity-collapses-gymv-games-into-a-single-bucket)'s task axis; only then run cross-game transfer probes; only then attempt cross-domain. Intra-gymv is the first place to discover real binding-failure patterns ([`PLAN-SKILL-BANK §4.3b`](../plans/04-skill-bank/PLAN-SKILL-BANK.md)) on actual data.

### Outside this package, but blocking its value

- **Actor rewire.** `decision_agents.skill_interface.SkillBankProvider` still queries the bank directly. Until it is replaced by a `HarnessSkillProvider` that wraps `SkillHarness.select_eligible_skills`, the "harness narrows + may veto" rule is not in force at runtime. Tracked in [`../IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §"Not yet delivered".
- **Real domain executors** under `vlm_wrapper/<domain>_adapter.py` (item 1's other half).

### Suggested work-order

Smallest cost / highest value first. The reordering below puts the audit's [§9–§22](#spec-contract-gaps-audit-2026-04-30) items ahead of items 1–8 because they are pure additive contract fixes (no behavioural break) and they unblock the I/O-dump driver that validates everything else.

  1. Add `SkillHarness.validate_invocation(skill, state, bindings, *, intention, reasoning_trace) -> {veto, veto_reason, diagnostic_labels}` and propagate `EligibleSkill.shadow_only` into `SkillEpisode` / `RewardLogEntry` ([§9](#9-online-surface-api-gaps-validate_invocation-scoring-intentionactive_skill-inputs), [§10](#10-skillepisode-artefact-field-gaps) shadow row).
  2. Extend `EligibleSkill` with the per-skill check booleans (`binding_ok / precondition_ok / evidence_ok / adapter_ok`), `fit_score`, `risk_score`, and `veto / veto_reason` for rejected candidates ([§9](#9-online-surface-api-gaps-validate_invocation-scoring-intentionactive_skill-inputs)). Pure additive on `eligibility.py`.
  3. Extend `SkillEpisode` with `evidence_in / evidence_out` split, `evidence_warrant / verify_verdict / reason_warrant`, `protocol_trace` (index from `episode.steps[i]` to `skill.protocol[k]`), per-key `contract_progress`, structured `reward_components`, and a list-typed `diagnostic_labels` ([§10](#10-skillepisode-artefact-field-gaps)). Pure additive on `data_structure/extensions/skill_episode.py`.
  4. Extend `SkillEvaluationRecord` with the reproducibility anchors `bank_snapshot_id`, `eval_suite_id`, `adapter_versions`, `ontology_version`, plus `status_before / status_after / rejected_domains / rollback_target` ([§11](#11-skillevaluationrecord-reproducibility-anchor-gaps)). Wire `bank_snapshot_id` through `GateService.evaluate(...)`.
  5. **[done — Day 1, 2026-04-30]** Stand up `labeling_supplement/dump_harness_io_gpt54.py` (online surface only) per [§14](#14-no-io-dump-driver--live-harness-behaviour-against-the-cold-start-corpus-is-unverified). Doubles as the integration test for items 1–4. Companion bypass probe `_phase0_cross_eligibility_probe.py` shipped alongside, since the dump driver inherits the actor's per-game RAG and cannot exhibit §22 on its own. Phase-0 baseline numbers in `labeling_supplement/harness_io_out/_phase0_report.md`.
  6. **[design locked — Day 1; impl Day 2]** Lift cold-start `protocol.steps` from prose to typed hops in the labeling pipeline — abstract verbs + `${slot}` placeholders + populated `eff_add` / `eff_del` derived from `metadata.schema_canonical` ([§21](#21-cold-start-protocol-is-natural-language-prose-not-typed-hops)). Gymv-only ontology is sufficient for the first pass. Hard prerequisite for items 7, 13, and any meaningful Stage 3a transfer. Design lock + 21-verb taxonomy + 92.5 % coverage measurement: [`../implementation_notes/protocol-lift-design.md`](../implementation_notes/protocol-lift-design.md). Implementation site: [`../labeling/_decorate_skill_records.py`](../labeling/_decorate_skill_records.py).
  7. **[Day 2]** Add the task axis — `SkillRecord.feasible_tasks` / `verified_tasks`, `FewShotAdapter.adapt(..., target_task)`, `EligibilityFilter` honours both `domain` and `task` ([§22](#22-feasible_domains-granularity-collapses-gymv-games-into-a-single-bucket)). Cold-start ingestion seeds `feasible_tasks=[provenance.source_name]`. Pure additive contract change. Empirical motivation (100 % cross-contamination on 2048 ↔ tetris) recorded in `labeling_supplement/harness_io_out/_phase0_report.md`.
  8. **[Day 3–4]** First intra-gymv transfer cycle — wire `GymvAdapter.set_executor(real_step)`, build `FewShotDemo`s from existing `labeling/skill_actions_out/.../<game>/episode_*.json`, plug a gymv-shape `success_fn` keyed on consecutive `schema_canonical` blocks. Run `(2048 ↔ tetris ↔ candy_crush)` transfer probes through Stage 3a. First end-to-end transfer signal on real data.
  9. Plug a real domain-aware `success_fn` into `FewShotAdapter` for cross-domain (existing item 4) — generalises item 8's gymv scorer.
  10. Wire the legacy actor to `HarnessSkillProvider`.
  11. Implement action-level `ReplayValidator` (walk `seed.steps`, compare actions + evidence) — existing item 2.
  12. Stand up `harness/gate_runner.py` over `gate_service` stages — existing item 3 + [§13](#13-gateservice-lives-under-orchestrator-not-harness--naming-mismatch-with-the-spec) relocate.
  13. Extend the dump driver to the offline GateRunner surface (Stage 0–4 + `SkillEvaluationRecord` per Crafter proposal) — second half of [§14](#14-no-io-dump-driver--live-harness-behaviour-against-the-cold-start-corpus-is-unverified).
  14. Add `rollout_batch[]` overload on `_run_shadow` and `eval_suite[]` overload on `_run_non_regression` ([§12](#12-gateservice-stage-io-signatures-dont-match-the-spec)).
  15. Add `harness/transfer_manager.py` for shadow → active — existing item 3 second half.
  16. Wire `vlm_wrapper/<domain>_adapter.py` executors via `set_executor()` — `browser` → `osworld` → `video` → `visual_reasoning` (existing item 1). Cross-domain transfer follows the path validated by item 8.
  17. Phase-F LoRA heads in `select_eligible_skills` (existing item 5).

---

## Wire-up status (audit: 2026-04-30)

[§9–§22](#spec-contract-gaps-audit-2026-04-30) catalogue what's missing on the harness's *API surface*. This section answers the orthogonal question: **given the live process today, can the harness actually be wired into the framework?** It was scoped while standing up `labeling_supplement/dump_harness_io_gpt54.py` and grepping every caller of `EpisodeRunner(`, `SkillHarness(`, `ActorAgent(`, and `PromotionOrchestrator.promote(`.

Short answer: **no for the live online runtime, yes for the offline promotion loop.** The asymmetry is structural — see [§17](#17-the-keystone-bankrunnable-is-empty-until-the-offline-loop-fires-once) for why the offline loop is a hard prerequisite for the online one.

### 15. Topology — what's already connected vs not

The wiring code I expected to be missing is already there. `EpisodeRunner.run` is an end-to-end harness driver:

```112:129:Multi-hop-Reasoning-VLM-Agent/orchestrator/runner.py
                eligible = self._harness.select_eligible_skills(
                    self._bank.runnable(),
                    state,
                )
                choice = self._actor.choose_action(state, eligible)
                if choice is None:
                    next_state, done = self._env.step(None)
                else:
                    budget.add_skill_invocation()
                    last_episode = self._harness.run_skill(
                        choice.skill,
                        state,
                        parent_run_id=run_id,
                        bindings=choice.bindings,
                    )
                    skill_eps.append(last_episode)
                    self._store.put_skill_episode(last_episode)
                    next_state, done = self._env.step(last_episode)
```

`GateService.evaluate(...)` likewise drives `harness.replay_validate` (Stage 1) and `FewShotAdapter.adapt → harness.run_skill` (Stage 3a). The connectors exist; the question is whether the *runtime* uses them.

| Component | Production caller? | Wired to `SkillHarness`? |
|---|---|---|
| `orchestrator.EpisodeRunner` | None — only [`../tests/test_smoke.py`](../tests/test_smoke.py) | yes (in code) |
| `decision_agents.ActorAgent` | `cold_start/generate_cold_start_actor*.py`, `decision_agents.run_actor_episode` | **no** — uses `SkillBankProvider` over legacy `skill_agents.SkillBankMVP` |
| `orchestrator.PromotionOrchestrator.promote` | None — only [`../tests/test_smoke.py`](../tests/test_smoke.py) | n/a |
| `crafter.SkillCrafterService.cycle` | None — only the dump driver and unit tests | n/a |

There is also a **name collision** in the codebase: `decision_agents/core/harness.py::Harness` is an *env wrapper* (gym-style `reset`/`step`), totally unrelated to `harness.SkillHarness`. The live `ActorAgent` already takes a `harness=` kwarg, and that kwarg means the env. Any wiring work has to disambiguate, and the spec's "harness" (this package) should probably end up renamed at the call site to keep readers from conflating the two.

### 16. Hard blockers — would silently break the runtime if flipped today

#### 16.1 Adapter executors are stubs, so `run_skill` is a black hole

All five registered adapters fall back to `_deterministic_executor` from `adapters/_stub_base.py` when no real env-step callback is registered. `gymv_wrapper` exposes `set_executor(...)` but **nothing in production calls it**. If the live actor is wired to `harness.run_skill` today, the env doesn't advance — the adapter fabricates a plausible `SkillEpisode` and returns. This would catastrophically regress every cold-start rollout. Same root cause as [§1](#1-transfer-target-adapters-are-deterministic-stubs); restated here because it's the #1 risk for *online* wiring specifically.

#### 16.2 `EnvLike` protocol mismatch with production gym envs

`EpisodeRunner.run` requires `env.step(episode: Optional[SkillEpisode]) -> (StateSchema, bool)` — i.e. the env consumes a typed `SkillEpisode` and returns a typed `StateSchema`. None of the production envs under `env_wrappers/` match that shape; they take primitive actions and return text observations. The smoke test passes only because the smoke env is bespoke. A real wire-up needs an `EnvLike` shim per env (gymv first).

#### 16.3 Bank-pointer mismatch

The live actor's `decision_agents.skill_interface.SkillBankProvider` reads `skill_agents.SkillBankMVP` (legacy four-stage bank). `SkillHarness` reads `skill_bank.SkillRepository` (new typed four-store). The planned `skill_bank/legacy_bridge.py` (status: not yet delivered, see [`../IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §"Not yet delivered") is what closes that gap. Without it, even a correctly-wired `HarnessSkillProvider` would point at an empty new-bank while the legacy bank does the real work.

#### 16.4 `ActorLike` protocol mismatch with `decision_agents.ActorAgent`

`EpisodeRunner` expects `ActorLike.choose_action(state: StateSchema, eligible: List[EligibleSkill]) -> Optional[ActorChoice]` (`orchestrator/runner.py:38–46`), where `ActorChoice` carries a typed `SkillRecord` + `bindings`. The current `decision_agents.ActorAgent.run` returns text actions and consumes text observations. They'd need an adapter wrapper before the runner can drive the live actor.

#### 16.5 `record_outcome` shape mismatch with `RewardLogger`

The actor-side `SkillProvider.record_outcome(skill_id, *, outcome, reward, steps_taken, info)` (`decision_agents/skill_interface.py:179–196`) is per-attempt scalar feedback. `RewardLogger` and the gate's Stage 2 expect a full `SkillEpisode`. Loss of fidelity unless the actor produces a real `SkillEpisode` itself — which depends on [§10](#10-skillepisode-artefact-field-gaps) landing first.

### 17. The keystone — `bank.runnable()` is empty until the offline loop fires once

`SkillRepository.runnable()` reads from the `active` store only:

```51:57:Multi-hop-Reasoning-VLM-Agent/skill_bank/repository.py
    def runnable(self, *, include_shadow: bool = True) -> List[SkillRecord]:
        out: List[SkillRecord] = []
        for r in self.active.all():
            if r.status == SkillStatus.SHADOW and not include_shadow:
                continue
            out.append(r)
        return out
```

The active store holds statuses `ACTIVE` and `SHADOW`. Cold-start banks ingest skills as `CANDIDATE` (in the `candidate` store). Until the offline promotion loop fires *at least once* and graduates a skill across `CANDIDATE → SHADOW`, `EpisodeRunner` sees `[]`, the eligibility filter has nothing to filter, and every online step is a no-op.

This is the most consequential finding from the audit: **the offline promotion path is a hard prerequisite for the online runtime**, not the other way around. Even if [§16](#16-hard-blockers--would-silently-break-the-runtime-if-flipped-today) were solved tomorrow, the runner would still be skill-starved.

The lifecycle's `feasible_domains ≥ 2` check (`skill_bank/lifecycle.py:253`) only fires on transitions to `ACTIVE`. Single-domain cold-start skills can therefore still reach `SHADOW` — sufficient to populate `bank.runnable(include_shadow=True)` — even before Phase D transfer arenas exist. So one offline cycle on the dump driver's outputs really is enough to unblock the online path *for shadow-mode rollouts*.

### 18. The one wire-up that's safe today — offline promotion

```
labeling_supplement/dump_harness_io_out/<run>/<corpus>/<source>/proposal_*/evaluation.json
        ↓ (new driver, ~150 LOC)
PromotionPlan: [(skill, target_status_from_verdict, evaluation, rationale), ...]
        ↓
PromotionOrchestrator.promote(plan, repository=test_repo)
        ↓
test_repo now has SHADOW / PROVISIONAL skills
        ↓
bank.runnable() returns non-empty
        ↓
EpisodeRunner has something to filter
```

Properties:

  - Reads files only; doesn't touch the live actor or any env.
  - Operates on a test `SkillRepository`, not production state.
  - Surfaces real orchestrator-side breakage (`content_hash` drift, atomic-write conflicts, snapshot-reference issues) on actual data, not stubbed records.
  - Failure mode is loud: a malformed promotion raises `LifecycleError`; nothing in production changes.
  - Is the **only** sequence that makes the online loop meaningful afterward, per [§17](#17-the-keystone-bankrunnable-is-empty-until-the-offline-loop-fires-once).

The natural location is a sibling to the dump driver under [`../labeling_supplement/`](../labeling_supplement/) — e.g. `promote_evaluations_gpt54.py` — mirroring the same dispatcher pattern.

### 19. Recommended sequencing to a fully-wired online runtime

| Stage | What | Risk | Rough effort | Unblocks |
|---|---|---|---|---|
| **0** | Offline promotion driver — consume `evaluation.json` → `PromotionOrchestrator.promote` against a test repo ([§18](#18-the-one-wire-up-thats-safe-today--offline-promotion)) | Low | 1–2 days | Skills reach SHADOW; runner has data ([§17](#17-the-keystone-bankrunnable-is-empty-until-the-offline-loop-fires-once)) |
| **1** | Additive contract fixes — `SkillEpisode` ([§10](#10-skillepisode-artefact-field-gaps)), `SkillEvaluationRecord` reproducibility anchors ([§11](#11-skillevaluationrecord-reproducibility-anchor-gaps)), `EligibleSkill` scoring ([§9](#9-online-surface-api-gaps-validate_invocation-scoring-intentionactive_skill-inputs)) | Low | 5–7 days | Online wiring is *meaningful* when it lands |
| **2** | Additive harness API — `validate_invocation` (§9.1), threaded `intention / active_skill / local_reasoning_trace` (§9.2) | Low | 4–5 days | Online actor can pre-flight check |
| **3** | Real adapter executor for `gymv` via `gymv_wrapper.set_executor(...)`; replace deterministic-stub fallback with explicit `ABORT` so silent stubs can't escape ([§16.1](#161-adapter-executors-are-stubs-so-run_skill-is-a-black-hole)). Pre-requires the protocol lift ([§21](#21-cold-start-protocol-is-natural-language-prose-not-typed-hops)) — without it `harness.run_skill` produces no-op episodes even on same-game state. Pairs with the task axis ([§22](#22-feasible_domains-granularity-collapses-gymv-games-into-a-single-bucket)) to enable the intra-gymv transfer milestone before cross-domain ([Intra-gymv transfer is the right first milestone](#intra-gymv-transfer-is-the-right-first-milestone)). | Medium | 7–10 days | `run_skill` actually moves the env |
| **4** | `EnvLike` shim for gymv ([§16.2](#162-envlike-protocol-mismatch-with-production-gym-envs)); `skill_bank/legacy_bridge.py` for the bank-pointer flip ([§16.3](#163-bank-pointer-mismatch)) | Medium | 4–5 days | Live actor reads the new bank; runner can step real envs |
| **5** | `HarnessSkillProvider` + actor wired in **shadow mode** (legacy provider still authoritative; harness disagreements logged, not acted on) | Low | 4–5 days | Diagnostic signal pre-flip |
| **6** | Actor flip — `HarnessSkillProvider` becomes authoritative; cold-start scripts call `EpisodeRunner` directly; actor produces real `SkillEpisode` ([§16.4](#164-actorlike-protocol-mismatch-with-decision_agentsactoragent), [§16.5](#165-record_outcome-shape-mismatch-with-rewardlogger)) | Higher | 7–10 days | Live online wiring |

Total: ~6–8 weeks. Stage 0 alone gives end-to-end signal in a day or two and is the prerequisite for everything that depends on `bank.runnable()` returning non-empty.

The "Suggested work-order" in [§Suggested work-order](#suggested-work-order) above is harness-internal; this table layers the runtime-topology dimension on top of it. Stage 0 here corresponds to no item there (it's an orchestrator-side driver). Stages 1–2 here ≈ items 1–4 there. Stages 3–6 here are mostly items 7 + 13 there but explicitly sequenced.

### 20. Bottom line

  - **The hard part of harness wiring is already in [`../orchestrator/runner.py`](../orchestrator/runner.py) and [`../orchestrator/gate_service.py`](../orchestrator/gate_service.py).** Those code paths are correct in shape, exercised by [`../tests/test_smoke.py`](../tests/test_smoke.py) and the dump driver, and were the riskiest thing to get right.
  - **What's missing is not the wiring but the things on either end** — a real adapter executor (so `run_skill` advances the env), a populated `active` store (so `bank.runnable()` returns anything), and a typed `EnvLike` shim per environment.
  - **The offline promotion loop is the keystone.** Until it fires once on real data, the online runtime is *structurally* skill-starved. Flipping the harness in today would just make every actor step a no-op (best case) or a stub-driven black hole (worst case, if `run_skill` is invoked).

---

## Cross-references

- Root [`readme.md`](../readme.md) §"Architecture" — where the harness sits in the four-stage pipeline.
- [`../plans/05-harness/PLAN-HARNESS.md`](../plans/05-harness/PLAN-HARNESS.md) — full spec, gate stack G0–G5.
- [`../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) — `GateRunner` stages and aggregation contract that drives [§11–§13](#11-skillevaluationrecord-reproducibility-anchor-gaps).
- [`../skill_bank/README.md`](../skill_bank/README.md) — invariant 8 ties `FewShotAdapter` to `SkillLifecycleManager.record_transfer_verification`.
- [`../implementation_notes/crafter-harness-orchestrator-roles.md`](../implementation_notes/crafter-harness-orchestrator-roles.md) — three-role I/O contract; §3 cheat sheet enumerates the artefact families [§10–§11](#10-skillepisode-artefact-field-gaps) extend; §7.1 mismatch #2 motivates [§10](#10-skillepisode-artefact-field-gaps)'s `protocol_trace` row.
- [`../labeling_supplement/`](../labeling_supplement/) — sibling location for the `dump_harness_io_gpt54.py` driver in [§14](#14-no-io-dump-driver--live-harness-behaviour-against-the-cold-start-corpus-is-unverified).
- [`../tests/test_smoke.py`](../tests/test_smoke.py) — runnable end-to-end wiring example.
