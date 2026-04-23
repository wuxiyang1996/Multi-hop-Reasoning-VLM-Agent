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

## Cross-references

- Root [`readme.md`](../readme.md) §"Architecture" — where the harness sits in the four-stage pipeline.
- [`../plans/05-harness/PLAN-HARNESS.md`](../plans/05-harness/PLAN-HARNESS.md) — full spec, gate stack G0–G5.
- [`../skill_bank/README.md`](../skill_bank/README.md) — invariant 8 ties `FewShotAdapter` to `SkillLifecycleManager.record_transfer_verification`.
- [`../tests/test_smoke.py`](../tests/test_smoke.py) — runnable end-to-end wiring example.
