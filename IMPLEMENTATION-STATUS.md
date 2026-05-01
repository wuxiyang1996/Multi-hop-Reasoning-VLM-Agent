# Implementation status — P0, Phase A, B, C MVP

Last updated: 2026-05-01 (S0 sprint — lane-(a) decision shipped, SFT
checkpoints reconciled, pre-flight tooling landed).

This document tracks what has been implemented from
[`plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md`](plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md)
and where each module lives. It coexists with — and does not replace —
the legacy code under `decision_agents/`, `skill_agents/`,
`vlm_wrapper/`, and `data_structure/`.

> **Cross-references for the current readiness audit**:
> - [`implementation_notes/pre-training-readiness-audit.md`](implementation_notes/pre-training-readiness-audit.md) — open / closed items by sprint (S0–S4).
> - [`implementation_notes/skill-lane-decision.md`](implementation_notes/skill-lane-decision.md) — T1.3 closed: skill = retrieval payload (lane (a)).
> - [`implementation_notes/single-vs-two-mdp-tradeoff.md`](implementation_notes/single-vs-two-mdp-tradeoff.md) — T3.6 decided + shipped: single-MDP (no `hop_select` LoRA, no `inner_mdp.py`).
> - [`runs/sft_coldstart/sft_summary_all.json`](runs/sft_coldstart/sft_summary_all.json) — run-wide manifest of all six trained SFT adapters (T2.9).

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
| Crafter Phase F | Frozen Qwen3-VL-32B / 235B-A22B teacher registry + `SkillCrafterService.{with_qwen3_vl_teacher, from_env, set_teacher_model}` (default `Qwen/Qwen3.5-35B-A3B`; Qwen3-VL is an opt-in upgrade path) | `common/models.py`, `crafter/service.py` |
| Harness Day 7-9 | `GateRunner` (offline gate surface), action-level `ReplayValidator`, `validate_invocation`, `RejectedSkillSink`, expanded `SkillEpisode*` / `SkillEvaluationRecord` fields, `PromotionOrchestrator` reproducibility anchors, `--persist` writeback for the Stage-3a transfer cycle | `harness/{gate_runner,replay_validator,skill_harness,eligibility,rejected_skill_sink}.py`, `data_structure/extensions/{skill_episode,skill_evaluation}.py`, `orchestrator/promotion_orchestrator.py`, `labeling_supplement/_phase4_transfer_cycle.py` |
| Trainer Day 10 | `SkillHarnessHook` wires the harness's two LLM-free surfaces into the co-evolution loop: pre-LLM eligibility filter + post-LLM `validate_invocation` veto in Phase A, and `RejectedSkillSink → record_false_binding_pattern` drain in Phase B′. Opt-in via `--harness-enabled` / `crafter_promotion_enabled`. | `trainer/coevolution/{_harness_hook,_crafter_hook,episode_runner,rollout_collector,orchestrator,config}.py`, `scripts/run_coevolution.py` |
| Lane-(a) flag (T1.3a, S0) | `SkillCrafterService(enable_protocol_patching=False)` default — Repairer / `PatchProposal` mint path is parked under the lane-(a) decision; dispatcher's existing `_STATUS_NO_OP` → Hypothesizer fall-through carries the failure signal. Threaded through `_crafter_hook → CoEvolutionConfig → run_coevolution.py --enable-protocol-patching`. | `crafter/service.py`, `trainer/coevolution/{_crafter_hook,config,orchestrator}.py`, `scripts/run_coevolution.py` |
| Threshold YAMLs (T2.5, S0) | `configs/skill_gate.yaml` (single source of truth for G0–G5 thresholds + drift annotations) and `configs/failure_routing.yaml` (FailureClass → Crafter mode dispatch table, including the new lane-(a) `BANK_GAP / RETRIEVAL_MISLEAD / STALE_DESCRIPTION` taxonomy). Both carry `policy_version` for audit-log drift attribution. | `configs/{skill_gate,failure_routing}.yaml` |
| SFT manifest + tools (T2.9 / T2.10 / T1.1′, S0) | Run-wide manifest for all six trained adapters; load-smoke and exact-match probes for pre-flight verification before launching co-evolution. | `scripts/build_sft_manifest.py`, `runs/sft_coldstart/sft_summary_all.json`, `evaluation/{smoke_load_sft_adapters,probe_schema_gen_exact_match}.py` |
| Offline promotion cycle (T1.2, S1) | One-shot wrapper for the §17 keystone — drives `decide_promotion_gpt54.py` + `legacy_writeback.writeback_promotion` once to flip cold-start banks from CANDIDATE to ACTIVE/PROVISIONAL/SHADOW so `bank.runnable() != []`. Asserts the post-condition before exiting. | `scripts/run_offline_promotion_cycle.sh`, `labeling_supplement/decide_promotion_gpt54.py`, `skill_bank/legacy_writeback.py` |
| Tests | Invariant + smoke + backbone-model + crafter Phase D/F + lane-(a) flag + harness Day 7-10 + trainer Day 10 hook | `tests/{conftest,test_invariants,test_smoke,test_backbone_model,test_crafter_*,test_few_shot_transfer,test_gate_runner,test_replay_validator_action_walk,test_lifecycle_*,test_validate_invocation,test_skill_episode_field_expansion,test_promotion_orchestrator_anchors,test_phase4_persist,test_stub_executor_typed_hops,test_rejected_skill_sink,test_trainer_harness_hook,test_crafter_lane_a_flag}.py` (433 passing as of S0 close; one pre-existing unrelated failure in `test_schema_predicates::test_extra_whitespace_tolerated`) |

### Backbone models — three-tier stack

Single source of truth: `common/models.py`.

  - `BACKBONE_MODEL = "Qwen/Qwen3.5-9B"` — actor (`decision_agents`) +
    skill-bank (`skill_agents`) trained policy. LoRA-adapter target for
    `trainer/SFT/` and `trainer/coevolution/`.
  - `BACKBONE_TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"` — frozen
    control-plane backbone shared by the crafter, harness, and
    orchestrator. Served separately via
    `inference/serve_qwen35_35b_a3b.sh`.
  - `BACKBONE_JUDGE_MODEL = "gpt-5.5"` — eval-driver judge / validation.
  - `BACKBONE_SFT_TEACHER_MODEL = "gpt-5.5"` — cold-start data
    generation (`cold_start/`, `labeling/`) consumed by `trainer/SFT/`.

The Qwen3-VL Phase-F teachers (`Qwen/Qwen3-VL-32B`,
`Qwen/Qwen3-VL-235B-A22B`) and the older 8B / 32B / 72B Qwen tracks are
**deferred**. They remain reachable through dedicated entrypoints:

  - `scripts/qwen3_decision_agent.py`
  - `scripts/qwen3_skillbank_agent.py`
  - `inference/run_qwen3_8b_eval.py`
  - `inference/run_academic_benchmarks.py`
  - `skill_agents/lora/` (training-time only)
  - `SkillCrafterService.with_qwen3_vl_teacher(...)`

No library default points at them. To re-enable a deferred track,
either pass `model="..."` explicitly or set one of
`VLM_AGENT_BACKBONE_MODEL` / `VLM_AGENT_BACKBONE_TEACHER_MODEL` /
`VLM_AGENT_BACKBONE_JUDGE_MODEL` / `VLM_AGENT_BACKBONE_SFT_TEACHER_MODEL`
at process start.

Live defaults flipped in the 2026-04-28 model-stack migration:

  - `decision_agents/agent.py` `VLMDecisionAgent.DEFAULT_MODEL`: `gpt-4o` → `Qwen/Qwen3.5-9B`
  - `decision_agents/actor_agent.py` `DEFAULT_MODEL`: `gpt-4o` → `Qwen/Qwen3.5-9B`
  - `decision_agents/agent_helper.py` `DEFAULT_LLM_MODEL`: `gpt-4o` → `Qwen/Qwen3.5-9B`
  - `API_func.ask_model(model=None, ...)` default: `gpt-4o` → `Qwen/Qwen3.5-9B`
  - `orchestrator/config.py` `TeacherConfig.model_name`: `gpt-4o` → `Qwen/Qwen3.5-35B-A3B`
  - `orchestrator/config.py` `JudgeConfig.model_name`: `gpt-4o` → `gpt-5.5`
  - `crafter/service.py` `SkillCrafterService._teacher`: `gpt-4o` → `Qwen/Qwen3.5-35B-A3B`
  - `cold_start/` and `labeling/` `MODEL_GPT54` / `--label_model`: `gpt-5.4` / `gpt-5-mini` → `gpt-5.5`
  - `skill_agents/lora/config.py` `MultiLoraConfig.base_model_name_or_path`: `Qwen/Qwen3-8B` → `Qwen/Qwen3.5-9B`

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

> Single source of truth for sprint sequencing:
> [`implementation_notes/pre-training-readiness-audit.md`](implementation_notes/pre-training-readiness-audit.md)
> §6 — five-sprint plan (S0–S4). The list below mirrors the
> still-open items there; flip an entry to "Delivered" *and* tick
> the audit row when you ship.

### S0 — pre-flight (✅ shipped 2026-05-01; outstanding = GPU-bound execution only)

- ☑ **T1.3a** — `SkillCrafterService(enable_protocol_patching=False)` default,
  threaded through `_crafter_hook → CoEvolutionConfig → run_coevolution.py
  --enable-protocol-patching`. Tests in `tests/test_crafter_lane_a_flag.py`.
- ☑ **T2.5** — `configs/{skill_gate,failure_routing}.yaml` shipped (policy
  v1.0.0; lane-(a) failure taxonomy with `lane_b_primary_mode` overrides;
  drift annotations).
- ☑ **T2.9** — `runs/sft_coldstart/sft_summary_all.json` regenerable via
  `python scripts/build_sft_manifest.py`.
- ☑ **T2.6** — This file refreshed to point at the Day-7→10 hooks, the
  `runs/sft_*` corpus, the lane-(a) flag, the threshold YAMLs, the SFT
  manifest, and the offline-promotion driver.
- ☑ **T1.3e + T1.3f + T3.6** — Lane-(a) banner blocks shipped on
  `harness/README.md` §22, `implementation_notes/crafter-harness-orchestrator-roles.md`
  §7.3 / §7.5 (with new "Post-decision rule" sub-section), and the four
  PLAN docs (PLAN-SKILL-CRAFTER, PLAN-SKILL-BANK, PLAN-HARNESS,
  PLAN-COMPONENTS-IMPLEMENTATION). Each banner explicitly marks
  `hop_select` / `inner_mdp` references obsolete and points at
  `single-vs-two-mdp-tradeoff.md`.

Outstanding = run-the-script-on-a-GPU only:

- ⏳ **T1.1′ (script ready)** — Run `evaluation/probe_schema_gen_exact_match.py`
  against `runs/sft_schema_gen/schema_gen_20260430_091831` to confirm
  PLAN-VISUAL-GROUNDING-MILESTONES §13 thresholds (field-acc ≥0.85,
  Path-A ≥0.70). The script exits non-zero on miss.
- ⏳ **T2.10 (script ready)** — Run `evaluation/smoke_load_sft_adapters.py`
  once on a GPU node to confirm none of the six adapters has a torn
  `*.partial_*` shard.

### S1 — fire offline once (the §17 keystone) — ✅ wrapper shipped 2026-05-01

- ⏳ **T1.2 (wrapper ready)** — Run `bash scripts/run_offline_promotion_cycle.sh`
  once to convert `bank.runnable() == []` into non-empty. The wrapper drives
  `decide_promotion_gpt54.py` + an inline call to
  `skill_bank.legacy_writeback.writeback_promotion`, then asserts the
  `bank.runnable() != []` post-condition before exiting.

### S2 / S3 / S4 (later sprints)

- **S2 — Live + audit guards.** Enable `crafter_promotion_enabled=True`,
  begin one-game GRPO smoke, wire `curator_weight` early-stage knob (T2.7),
  add `policy_version` audit drift rows.
- **S3 — Curriculum + scale-out.** Multi-game GRPO; integrate the audit's
  curriculum-graduation thresholds.
- **S4 — Full eval + co-evolution.** Eval E0 / E1 / E2 drivers; cross-task
  transfer using the `evaluation_dataset/` pool + holdout manifests.

### Cross-cutting items not yet sprinted

- **P1 — Visual-grounding stabilisation** under `vlm_wrapper/grounding.py`.
- **P2 — Eval E0 driver** (`evaluation/driver.py`,
  `evaluation/answer_evaluator.py`, slice/report scaffolds).
- **Phase D / E / F** of `PLAN-COMPONENTS-IMPLEMENTATION.md`
  (training cadence, multi-domain rollout, full eval).
- **Actor rewire**: replace `decision_agents.skill_interface
  .SkillBankProvider` with a `HarnessSkillProvider` that wraps
  `SkillHarness.select_eligible_skills`. (Note: the Day-10
  `SkillHarnessHook` already covers this for the trainer's
  co-evolution loop, but the `decision_agents` library API has not
  yet been ported.)
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

Expected as of S0 close: `433 passed, 1 deselected` (the deselected test is
the pre-existing `test_schema_predicates::test_extra_whitespace_tolerated`
which is unrelated to the readiness audit and tracked separately).

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
