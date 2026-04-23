# `orchestrator/` — System control plane

Spec: [`PLAN-PIPELINE-ORCHESTRATOR`](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md), [`PLAN-COMPONENTS-IMPLEMENTATION`](../plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md) §4 (Phase B).

Single top-level runner that drives outer-env episodes, persists artefacts, runs the gate, and atomically promotes / rolls back skills. The orchestrator is the **only** module that touches all of `harness/`, `crafter/`, and `skill_bank/` at once — and it's the only place where a skill's lifecycle moves forward.

```python
from orchestrator import (
    EpisodeRunner, EpisodeResult,
    ArtifactStore,
    BudgetController, BudgetExceeded,
    GateService, NonRegressionResult,
    PromotionOrchestrator, PromotionPlan, PromotionResult,
    SnapshotManager,
    OrchestratorConfig, BudgetLimits, GateThresholds,
    TeacherConfig, JudgeConfig, FewShotConfig,
)
```

---

## Module map

| File | Role |
|---|---|
| `runner.py` | `EpisodeRunner` — drives one outer-env episode end-to-end: resets the env, repeatedly asks the Actor to pick from `SkillHarness.select_eligible_skills`, executes via the harness, logs `SkillEpisode` to `ArtifactStore`. Returns `EpisodeResult(outcome, episodes, evidence_summary, budget_used)` |
| `artifact_store.py` | `ArtifactStore` — atomic JSONL writes for every artefact type: `episodes/`, `proposals/`, `failures/`, `evaluations/`, `releases/`. Atomic = write-to-tmp + rename, with per-stream advisory locks. The orchestrator's only side-effect on disk |
| `budget.py` | `BudgetController` — caps tokens, dollars, wallclock, and inner-hop count per episode. Reads the per-step `r_cost` written by `harness.RewardLogger`. Raises `BudgetExceeded` from inside the runner so partial work is still flushed to the artefact store |
| `gate_service.py` | `GateService` — composes the seven canonical stages from [`PLAN-UNIFIED-SKILL-GATE`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) §7: `Stage 0 static` (contract + invariant checks), `Stage 1 replay` (deterministic re-execution), `Stage 2 shadow` (parallel run alongside ACTIVE), `Stage 3a transfer` (calls `harness.FewShotAdapter`), `Stage 3b non-regression` (eval-suite delta), `Stage 4 provisional`, `Stage 5 active`. Emits `GateVerdictPayload` with per-stage `StageVerdict`s and an `eligible_domains` list (the `verified_domains` source of truth) |
| `promotion_orchestrator.py` | `PromotionOrchestrator` — atomic promotion / rollback. `promote(plan: PromotionPlan)` (1) snapshots the current bank+adapter+config via `SnapshotManager`, (2) refuses on FAIL or content-hash drift, (3) refuses ACTIVE on `LIMITED_PASS` (invariant 5), (4) calls `lifecycle.record_transfer_verification(...)` *before* the status transition (invariant 8 — `verified_domains` is gate-owned), (5) invokes `lifecycle.promote(...)` to physically move the JSON. Rollback restores the snapshot atomically |
| `snapshot_manager.py` | `SnapshotManager` — packs `(bank state ⊕ adapter weights ⊕ config)` into a `RunRelease`. Used by the promotion orchestrator on every promotion to enable atomic rollback, and by eval drivers to pin a specific frozen system version |
| `config.py` | `OrchestratorConfig` and its sub-configs: `BudgetLimits`, `GateThresholds`, `TeacherConfig` (Synthesis-Reflection / Crafter teacher — currently `gpt-4o`), `JudgeConfig` (eval-driver judge), `FewShotConfig` (K, max-iters for `FewShotAdapter`). Single source of truth for tunable knobs; loaded from YAML/JSON or constructed directly in tests |

---

## Episode → promotion data flow

```
EpisodeRunner.run(env, actor, budget)
  ├── for step in env:
  │     ├── eligible = harness.select_eligible_skills(state)
  │     ├── skill    = actor.choose(state, eligible)
  │     ├── result   = harness.run_skill(skill, state, env)        # SkillEpisode
  │     ├── artifact_store.put_episode(result.episode)
  │     └── budget.charge(result.cost)  # raises BudgetExceeded
  └── return EpisodeResult(...)

# Offline (separate run, slow timescale):
PromotionOrchestrator.promote(plan)
  ├── snapshot_manager.create_release(bank, adapter, config)
  ├── verdict = gate_service.evaluate(skill, eval_suite_id)
  │     ├── Stage 0  static
  │     ├── Stage 1  replay   (harness.ReplayValidator over recorded episodes)
  │     ├── Stage 2  shadow   (parallel runs)
  │     ├── Stage 3a transfer (harness.FewShotAdapter; emits eligible_domains)
  │     ├── Stage 3b non-regression (eval suite delta)
  │     └── Stage 4 / 5
  ├── if verdict.fail: rollback; return REJECTED
  ├── if verdict.limited_pass and target == ACTIVE: refuse; return BLOCKED
  ├── lifecycle.record_transfer_verification(skill, verdict.eligible_domains)
  ├── lifecycle.promote(skill, target_status)             # physical store move
  └── artifact_store.put_evaluation(SkillEvaluationRecord)
```

The `record_transfer_verification → lifecycle.promote` ordering is **load-bearing**: it's the one place invariant 8 (`verified_domains` is gate-owned) becomes a runtime guarantee. The `tests/test_invariants.py` suite has a dedicated test for this ordering.

---

## Phase boundaries

| Phase | What this package contains | Status |
|---|---|---|
| B (MVP) | `EpisodeRunner`, atomic `ArtifactStore`, `BudgetController`, `GateService` (stages 0–4), `PromotionOrchestrator`, `SnapshotManager` | **Delivered** — covered by `tests/test_smoke.py::test_smoke_end_to_end` |
| D (transfer + replay) | Full Stage 1 deterministic replay; Stage 3a wired to all four target-domain adapters; Stage 3b non-regression with `eval_suite_id` | Pending |
| E (eval suite + dashboards) | `eval_suite.py`, slice/label dashboards, `eval_suite_id` wiring across releases | Pending |

---

## Cross-references

- Root [`readme.md`](../readme.md) §"Architecture" / §"Implementation status".
- [`../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md`](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) §0a — the actor / harness / skill-bank / orchestrator boundary.
- [`../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) §7 — canonical gate-stage spec.
- [`../skill_bank/README.md`](../skill_bank/README.md) — invariants this orchestrator must respect on every `promote(...)`.
