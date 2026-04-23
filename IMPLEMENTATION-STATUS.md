# Implementation status — P0, Phase A, B, C MVP

Last updated: 2026-04-21.

This document tracks what has been implemented from
[`plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md`](plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md)
and where each module lives. It coexists with — and does not replace —
the legacy code under `decision_agents/`, `skill_agents/`,
`vlm_wrapper/`, and `data_structure/`.

## Delivered

| Phase | Module | Files |
| --- | --- | --- |
| Common | Shared types and enums | `common/{enums,ids,state_schema,typing,models}.py` |
| P0 | Extension records | `data_structure/extensions/{skill_episode,skill_record,gate_verdict,skill_evaluation,bank_mutation_proposal,failure_trace,run_release}.py` |
| Skill bank | Split-storage + lifecycle | `skill_bank/{stores,repository,lifecycle}.py` |
| Phase A | Harness MVP | `harness/{skill_adapter,adapter_registry,eligibility,reward_logger,replay_validator,skill_harness}.py`, `harness/adapters/{_common,gymv_adapter,browser_adapter}.py` |
| Phase B | Orchestrator MVP | `orchestrator/{config,artifact_store,budget,snapshot_manager,gate_service,promotion_orchestrator,runner}.py` |
| Phase C | Crafter MVP | `crafter/{failure_memory,failure_diagnoser,composer,generalizer,hypothesizer,service}.py` |
| Crafter Phase D | `Repairer` + `PatchProposal` plumbing (`SkillCrafterService.propose_repair`, repair-first dispatch in `cycle()`) | `crafter/repairer.py`, `crafter/service.py` |
| Crafter Phase F | Frozen Qwen3-VL-32B / 235B-A22B teacher registry + `SkillCrafterService.{with_qwen3_vl_teacher, from_env, set_teacher_model}` (default still GPT-4o) | `common/models.py`, `crafter/service.py` |
| Tests | Invariant + smoke + backbone-model + crafter Phase D/F | `tests/{conftest,test_invariants,test_smoke,test_backbone_model,test_crafter_repair,test_few_shot_transfer}.py` (60 passing) |

### Backbone model

Single source of truth: `common/models.py`.

  - `BACKBONE_MODEL = "gpt-4o"` — actor / policy / harness default.
  - `BACKBONE_TEACHER_MODEL = "gpt-4o"` — crafter / Synthesis-Reflection
    Agent default.
  - `BACKBONE_JUDGE_MODEL = "gpt-4o"` — eval driver judge default.

The 8B / 32B / 72B Qwen tracks (LoRA / GRPO / frozen-teacher) are
**deferred**. They remain reachable through dedicated entrypoints:

  - `scripts/qwen3_decision_agent.py`
  - `scripts/qwen3_skillbank_agent.py`
  - `inference/run_qwen3_8b_eval.py`
  - `inference/run_academic_benchmarks.py`
  - `skill_agents/lora/` (training-time only)

No library default points at them. To re-enable a deferred track,
either pass `model="..."` explicitly or set
`VLM_AGENT_BACKBONE_MODEL` / `VLM_AGENT_BACKBONE_TEACHER_MODEL` /
`VLM_AGENT_BACKBONE_JUDGE_MODEL`.

Live defaults flipped in this pass:

  - `decision_agents/agent.py` `VLMDecisionAgent.DEFAULT_MODEL`: `gpt-4o-mini` → `gpt-4o`
  - `decision_agents/actor_agent.py` `DEFAULT_MODEL`: `gpt-4o-mini` → `gpt-4o`
  - `decision_agents/agent_helper.py` `DEFAULT_LLM_MODEL`: `gpt-4o-mini` → `gpt-4o`
  - `decision_agents/dummy_agent.py` example/default args: `gpt-4o-mini` → `gpt-4o`
  - `orchestrator/config.py` `TeacherConfig.model_name`: `None` → `BACKBONE_TEACHER_MODEL`
  - `orchestrator/config.py` `JudgeConfig` (new) and `OrchestratorConfig.backbone_model` (new): default `BACKBONE_MODEL`
  - `crafter/service.py` `SkillCrafterService._teacher`: `None` → `BACKBONE_TEACHER_MODEL`

### Invariants enforced (mechanical, with tests)

1. **G0 — evidence-driven** (`SkillEpisode.finalize` rejects evidence-free
   non-action successes; `SkillLifecycleManager` rejects ACTIVE
   promotion when `expected_evidence_roles` is empty).
2. **No-memory** (`SkillEpisodeStep.__post_init__` rejects any action
   type starting with `QUERY_MEM` or `WRITE_MEM`).
3. **General-protocol** (`SkillLifecycleManager` rejects ACTIVE
   promotion when `feasible_domains < 2`; `GateService.Stage 0` flags
   it during evaluation).
4. **Bank-write isolation** (`SkillStore.put`/`remove` raise
   `StoreLockedError` unless called via the lifecycle manager).
5. **Gate-bound promotion** (`PromotionOrchestrator.promote` rejects
   `FAIL` verdicts and content-hash drift; refuses ACTIVE promotion on
   `LIMITED_PASS`).
6. **Crafter scope** (`SkillCrafterService` materialises new skills only
   as DRAFT records; never touches `active_store`).

## Not yet delivered (next sessions)

- **P1 — Visual-grounding stabilisation** under `vlm_wrapper/grounding.py`.
- **P2 — Eval E0 driver** (`evaluation/driver.py`,
  `evaluation/answer_evaluator.py`, slice/report scaffolds).
- **Phase D / E / F** of `PLAN-COMPONENTS-IMPLEMENTATION.md`
  (training cadence, multi-domain rollout, full eval).
- **Actor rewire**: replace `decision_agents.skill_interface
  .SkillBankProvider` with a `HarnessSkillProvider` that wraps
  `SkillHarness.select_eligible_skills`.
- **Legacy bridge**: one-way migration of `skill_agents/skill_bank`
  Stage-3 records into the new `SkillRecord` (planned in
  `skill_bank/legacy_bridge.py`).
- **Live frozen-teacher inference for Qwen3-VL** — the `SkillCrafterService`
  surface for swapping to `Qwen/Qwen3-VL-{32B,235B-A22B}` is wired
  (`with_qwen3_vl_teacher`, `from_env`, `set_teacher_model`), but the
  actual model invocation path through `API_func.ask_model` for these
  IDs still needs provider-side routing (vLLM / HF endpoint).

## How to run the tests

```bash
cd Multi-hop-Reasoning-VLM-Agent
python -m pytest tests/ -v
```

Expected: `29 passed`.

## Module dependency graph

```
common/                  # leaf
  ↑
data_structure/extensions/   # leaf (uses common)
  ↑
skill_bank/             # uses common + extensions
  ↑                ↑
harness/             |  # uses common + extensions
  ↑                ↑
orchestrator/        |  # uses common + extensions + skill_bank + harness
  ↑
crafter/                # uses common + extensions + skill_bank + orchestrator (only ArtifactStore)
```

The crafter never imports `harness` (it operates on traces, not live
adapters) and never imports `skill_bank.stores` (only the lifecycle
manager). The harness never imports `orchestrator` or `crafter`.
