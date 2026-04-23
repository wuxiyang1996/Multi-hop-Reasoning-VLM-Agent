# `harness/` — Per-invocation runtime for skill execution and verification

Spec: [`PLAN-HARNESS`](../plans/05-harness/PLAN-HARNESS.md), [`PLAN-COMPONENTS-IMPLEMENTATION`](../plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md) §4 (Phase A).

The harness is the *frozen verifier* in the role split (root README §"Three-agent role split" / §"Architecture"):

> The Skill Bank provides candidates, the **Harness narrows + may veto**, the Actor decides, the Orchestrator handles offline promotion.

Concretely, the harness's job for one skill invocation is four steps:

1. **Filter** the bank's candidate set to a domain- and contract-eligible subset (`select_eligible_skills`).
2. **Execute** one chosen skill via the right `SkillAdapter` (`run_skill`).
3. **Record** everything as a `SkillEpisode` + reward-log entry — this is where invariant **G0 (evidence-driven)** bites via `SkillEpisode.finalize()`.
4. **Provide replay validation** for the gate (Stage 1, [`PLAN-UNIFIED-SKILL-GATE`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) §7).

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

### Outside this package, but blocking its value

- **Actor rewire.** `decision_agents.skill_interface.SkillBankProvider` still queries the bank directly. Until it is replaced by a `HarnessSkillProvider` that wraps `SkillHarness.select_eligible_skills`, the "harness narrows + may veto" rule is not in force at runtime. Tracked in [`../IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §"Not yet delivered".
- **Real domain executors** under `vlm_wrapper/<domain>_adapter.py` (item 1's other half).

### Suggested work-order

Smallest cost / highest value first:

1. Plug a real domain-aware `success_fn` into `FewShotAdapter`.
2. Wire the legacy actor to `HarnessSkillProvider`.
3. Implement action-level `ReplayValidator` (walk `seed.steps`, compare actions + evidence).
4. Stand up `harness/gate_runner.py` over `gate_service` stages.
5. Add `harness/transfer_manager.py` for shadow → active.
6. Wire `vlm_wrapper/<domain>_adapter.py` executors via `set_executor()` — `browser` → `osworld` → `video` → `visual_reasoning`.
7. Phase-F LoRA heads in `select_eligible_skills`.

---

## Cross-references

- Root [`readme.md`](../readme.md) §"Architecture" — where the harness sits in the four-stage pipeline.
- [`../plans/05-harness/PLAN-HARNESS.md`](../plans/05-harness/PLAN-HARNESS.md) — full spec, gate stack G0–G5.
- [`../skill_bank/README.md`](../skill_bank/README.md) — invariant 8 ties `FewShotAdapter` to `SkillLifecycleManager.record_transfer_verification`.
- [`../tests/test_smoke.py`](../tests/test_smoke.py) — runnable end-to-end wiring example.
