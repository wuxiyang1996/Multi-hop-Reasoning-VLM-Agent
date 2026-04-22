# Multi-hop Reasoning VLM Agent

**A skill-centric, evidence-driven, gate-bound visual agent.** This repository builds a Visual Language Model (VLM) agent that converts pixels into a structured `<state>` schema and acts on it through a four-stage pipeline of *Visual Grounding → Action → Skill Bank → Skill Crafter*, governed by three operational components (*Skill Harness*, *Pipeline Orchestrator*, *Unified Skill Gate*).

The system learns **transferable reasoning, grounding, and control skills as general protocols feasible across game, webagent, os-agent, video-understanding, and visual reasoning tasks**. The first concrete arena is **short-video evidence-grounded reasoning** (Video-Holmes-style); cross-domain generalization is a hard, mechanically-enforced invariant of the skill bank, not an aspiration.

This repo supersedes the COS-PLAY codebase that lives alongside it under `decision_agents/`, `skill_agents/`, `vlm_wrapper/`, and `data_structure/legacy/`. Those modules remain importable as a reference for the legacy single-domain GRPO loop; the new build under `common/`, `harness/`, `orchestrator/`, `crafter/`, `skill_bank/`, and `data_structure/extensions/` implements the canonical plan from [`plans/`](plans/README.md).

---

## Table of Contents

- [Why this project](#why-this-project)
- [Architecture](#architecture)
- [Mechanically-enforced invariants](#mechanically-enforced-invariants)
- [Backbone model — GPT-4o for now](#backbone-model--gpt-4o-for-now)
- [Repository layout](#repository-layout)
- [Implementation status](#implementation-status)
- [Quick start](#quick-start)
- [Plans index](#plans-index)
- [Legacy COS-PLAY notes](#legacy-cos-play-notes)
- [Citation and license](#citation-and-license)

---

## Why this project

Modern VLMs reason well on a single image but break down on multi-hop visual tasks where evidence must be gathered, verified, chained, and committed across hops. Existing skill-based agents either (a) bind their skills to one domain (game-only or browser-only) or (b) treat skills as opaque function calls with no evidence interface. Both choices block transfer and hide failures.

This project takes the opposite stance:

1. **Every skill is a general protocol** — it must declare adapter bindings to all five domains (game, webagent, os-agent, video, visual reasoning). Single-domain skills are rejected at promotion time. See [Skill Bank §0.1](plans/03-skill-bank/PLAN-SKILL-BANK.md#01-general-protocol-invariant-no-domain-specific-skill-families).
2. **Every skill is evidence-driven** — it must declare a role from `{GATHER, VERIFY, REASON, COMMIT}` and record a non-empty evidence interface on every successful episode. Opaque skills are rejected at Gate G0. See [Skill Bank §0.3](plans/03-skill-bank/PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills).
3. **Every promotion is gate-bound** — no proposal reaches `ACTIVE` without passing the canonical gate stack (`static → replay → shadow → transfer → non-regression`). See [Unified Skill Gate](plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md).
4. **The Actor is the policy, the Harness is a frozen verifier.** The Skill Bank provides candidates, the Harness narrows + may veto, the Actor decides, the Orchestrator handles offline promotion. The frozen large model never silently becomes the policy. See [Pipeline Orchestrator §0a](plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#0a-actor-harness-skill-bank-orchestrator-boundary).

The first benchmark target is short-video multi-hop reasoning, but the skill ontology is fixed across phases — short-video is the first **arena** in which already-defined general protocols (e.g. `collect_evidence_chain`, `disambiguate_target`, `locate_filter_select`, `actor_action_binding`, `verify_constraint`) earn their `verified_domains` entry.

---

## Architecture

### Pipeline (canonical four stages)

```
Pixels (game frame / screenshot / video / image)
    ↓
(1) Visual Grounding   — VLM parser → structured <state> schema
    ↓
(2) Action Agent       — two-level MDP: inner reasoning hops → environment actions
    ↓                       ↑
(3) Skill Bank         — segmentation → contracts → cross-domain retrieval
    ↓                       ↑
(4) Skill Crafter      — compose / generalize / hypothesize / repair
    ↓
    └──→ proposals enter the Unified Gate → bank as DRAFT → CANDIDATE → ACTIVE
```

### Operational components

| Component | Role | Module |
| --- | --- | --- |
| **Skill Harness** | Per-invocation runtime: eligibility filter → adapter run → tracing → `SkillEpisode` → replay validation | [`harness/`](harness/) |
| **Pipeline Orchestrator** | System control plane: `EpisodeRunner`, `ArtifactStore`, `BudgetController`, `GateService`, `PromotionOrchestrator`, `SnapshotManager` | [`orchestrator/`](orchestrator/) |
| **Unified Skill Gate** | Canonical `SkillStatus` / `SkillSourceType` / `SkillRecord` / `GateVerdict` lifecycle, split-storage (`draft / candidate / active / archive`), `SkillLifecycleManager` as the *only* writer | [`skill_bank/`](skill_bank/) |
| **Skill Crafter** | Slow-timescale typed proposal layer (composition, generalization, hypothesis, repair, retire) — outputs are *candidates only* | [`crafter/`](crafter/) |

### Two-level MDP

The agent runs an **outer** environment loop with a **lightweight typed inner loop** of inner actions `{GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE}`, capped at 0–3 hops. Skills capture *how to think* (typed hop chains) — not just *what to do*. Heavy reasoning (failure diagnosis, composition, transfer, hypothesis generation) is strictly offline. See [Action Agent §5](plans/02-action-agent/PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control).

### Three-agent role split

| Agent | Default model (current phase) | Role | Update timescale |
| --- | --- | --- | --- |
| **Actor / Decision Agent** | `gpt-4o` | Online action execution, hop selection, skill selection, protocol following | Fast |
| **Skill-Use / Operational** | `gpt-4o` | Skill retrieval, segmentation, contract learning, curation | Medium |
| **Synthesis-Reflection (Teacher)** | `gpt-4o` (will swap to frozen 32B/72B later) | Failure reflection, composition, hypothesis, transfer, judging | Slow, gated |

See [Action Agent §2](plans/02-action-agent/PLAN-ACTION-AGENT.md#three-agent-role-split) for the canonical specification.

---

## Mechanically-enforced invariants

The plan calls out six invariants that must hold across *every* phase. Each is enforced by code, not by reviewer discipline, and is exercised by `tests/test_invariants.py`.

| # | Invariant | Enforcement point |
| --- | --- | --- |
| 1 | **G0 — evidence-driven**: every successful non-`ACTION` skill records a non-empty `evidence_in` / `evidence_out` / `evidence_warrant`. | `SkillEpisode.finalize` raises on empty evidence; `SkillLifecycleManager` rejects ACTIVE promotion when `expected_evidence_roles` is empty. |
| 2 | **No-memory**: no cross-episode storage layer. | `SkillEpisodeStep.__post_init__` rejects any action type starting with `QUERY_MEM` / `WRITE_MEM`. |
| 3 | **General-protocol**: every active skill must be feasible in ≥ 2 domains. | `SkillLifecycleManager` rejects ACTIVE promotion when `feasible_domains < 2`; `GateService.Stage 0` flags it during evaluation. |
| 4 | **Bank-write isolation**: only `SkillLifecycleManager` may mutate any skill store. | `SkillStore.put` / `remove` raise `StoreLockedError` unless called via the lifecycle manager. |
| 5 | **Gate-bound promotion**: no skill reaches `ACTIVE` without a passing `GateVerdictPayload` and stable content hash. | `PromotionOrchestrator.promote` rejects `FAIL` verdicts and content-hash drift; refuses ACTIVE on `LIMITED_PASS`. |
| 6 | **Crafter scope**: the crafter materialises only `DRAFT` records and never touches `active_store`. | `SkillCrafterService._persist` calls `SkillLifecycleManager.ingest_draft`; static dependency rule keeps `crafter/` from importing `skill_bank/stores`. |

---

## Backbone model — GPT-4o for now

The single source of truth is [`common/models.py`](common/models.py). For the current phase, every library default points at GPT-4o:

```python
from common.models import (
    BACKBONE_MODEL,          # "gpt-4o" — actor / policy / harness default
    BACKBONE_TEACHER_MODEL,  # "gpt-4o" — crafter / Synthesis-Reflection default
    BACKBONE_JUDGE_MODEL,    # "gpt-4o" — eval-driver judge default
)
```

The **8B / 32B / 72B Qwen tracks** (LoRA, GRPO, frozen-teacher) referenced throughout the plans are **deferred**. They remain reachable through dedicated entrypoints — `scripts/qwen3_*.py`, `inference/run_qwen3_8b_eval.py`, `inference/run_academic_benchmarks.py`, `skill_agents/lora/` — but no library default points at them. Override at process start with one of:

```bash
export VLM_AGENT_BACKBONE_MODEL=...           # actor / harness
export VLM_AGENT_BACKBONE_TEACHER_MODEL=...   # crafter
export VLM_AGENT_BACKBONE_JUDGE_MODEL=...     # eval driver
```

Test coverage for the GPT-4o pin lives in [`tests/test_backbone_model.py`](tests/test_backbone_model.py) (13 tests).

---

## Repository layout

### New modules (canonical build, this plan)

```
common/                   # canonical enums, ID helpers, <state> schema, BACKBONE_MODEL
data_structure/extensions/   # SkillEpisode, SkillRecord, GateVerdict, SkillEvaluationRecord,
                          # BankMutationProposal (Compose/Generalize/Hypothesis/Patch/Retire),
                          # FailureTrace, FailureDiagnosis, RunRelease
skill_bank/               # split-storage (draft/candidate/active/archive) + SkillLifecycleManager
                          # + SkillRepository — bank-write isolation invariant
harness/                  # SkillHarness, AdapterRegistry, EligibilityFilter, ReplayValidator,
                          # RewardLogger, SkillAdapter base + adapters/{gymv,browser}
orchestrator/             # EpisodeRunner, ArtifactStore, BudgetController, GateService,
                          # PromotionOrchestrator, SnapshotManager, OrchestratorConfig
                          # (TeacherConfig + JudgeConfig + backbone_model)
crafter/                  # FailureMemory, FailureDiagnoser, Composer, Generalizer,
                          # Hypothesizer, SkillCrafterService — proposals only
tests/                    # test_invariants.py (14), test_smoke.py (2),
                          # test_backbone_model.py (13)
```

### Legacy modules (COS-PLAY, kept for reference and incremental migration)

```
decision_agents/          # legacy COS-PLAY decision agent, intention/skill/reward
skill_agents/             # legacy skill-bank pipeline + GRPO training (LoRA, segmentation)
vlm_wrapper/              # legacy visual-grounding parsers and benchmark loaders
data_structure/legacy/    # legacy Episode / Experience records (extensions/ supersedes)
inference/                # legacy inference scripts (incl. Qwen3-8B / vLLM entrypoints)
trainer/                  # legacy SFT / GRPO / FSDP training infrastructure
env_wrappers/             # NL wrappers, Gymnasium adapters, game configs
labeling/, cold_start/    # cold-start labeling and seed-trajectory generation
```

### Plans

```
plans/
├── 00-system/        north-star scoreboard, eval-first target, role walkthrough
├── 01-visual-grounding/ Stage 1 — VLM parser + milestones
├── 02-action-agent/  Stage 2 — two-level MDP decision agent
├── 03-skill-bank/    Stage 3 — cross-task skill bank, retrieval, contracts
├── 04-skill-crafter/ Stage 4 — compose / generalize / hypothesize
├── 05-harness/       per-invocation runtime + gate stack
├── 06-orchestrator/  system control plane (DAG, promotion / rollback)
├── 07-skill-gate/    canonical lifecycle and gate spec
├── 08-cross-cutting/ failure routing, uncertainty calibration, experience ext.
├── 09-implementation/ Cursor-ready build sheet — Phase A → F + invariants
├── 10-edits/         already-applied refactor edit plans
└── 99-archive/       superseded discussions kept for provenance
```

See [`plans/README.md`](plans/README.md) for the full index.

---

## Implementation status

A more detailed view lives in [`IMPLEMENTATION-STATUS.md`](IMPLEMENTATION-STATUS.md). Headline:

### Delivered (29 tests passing)

- **Common** — enums, IDs, `<state>` schema, backbone-model registry.
- **P0** — extension records (7 dataclasses) under `data_structure/extensions/`.
- **Skill bank** — split-storage with `SkillLifecycleManager` enforcing the unified gate.
- **Phase A — Harness MVP** — `SkillHarness`, eligibility filter, adapters (`gymv`, `browser`), reward log, replay validator stub.
- **Phase B — Orchestrator MVP** — `EpisodeRunner`, atomic `ArtifactStore`, `BudgetController`, `GateService` (stages 0–4), `PromotionOrchestrator`, `SnapshotManager`.
- **Phase C — Crafter MVP** — failure memory + diagnoser, composer, generalizer, hypothesizer, `SkillCrafterService`.
- **Backbone model** — GPT-4o pinned across actor / teacher / judge with env-var override path.
- **Invariants** — six invariants mechanically enforced and tested.

### Pending (next sessions)

| Track | Item | Owning module |
| --- | --- | --- |
| **P1 — Visual Grounding** | Lightweight grounding stabilisation; routing policy A/B/C | `vlm_wrapper/grounding.py` |
| **P2 — Eval E0 driver** | JSONL driver, MCQ answer evaluator, LLM judge, easy/medium/hard slices, headline triple report | `evaluation/{driver,answer_evaluator,llm_judge,slices,report}.py` |
| **Phase D — Transfer + Replay** | Two-phase shadow → active transfer; full six-gate `GateRunner` (G0–G5); held-out replay; adapters for `osworld`, `video`, `visual_reasoning` | `harness/{transfer_manager,gate_runner,replay_validator}.py` + adapters |
| **Phase E — Eval suite + dashboards** | Frozen eval suite for non-regression; slice / label dashboards; `eval_suite_id` wiring | `orchestrator/eval_suite.py` |
| **Phase F — Trainable extensions** | LoRA heads `skill_select`, `continue_vs_switch`, `accept_transfer`, `adapter_refine` | TBD |
| **Actor rewire** | `HarnessSkillProvider` so the Actor consumes `SkillHarness.select_eligible_skills` instead of querying the bank directly | `decision_agents/skill_interface.py` |
| **Legacy bridge** | One-way migration of `skill_agents/skill_bank` Stage-3 records into `SkillRecord` | `skill_bank/legacy_bridge.py` |
| **Repair plumbing** | `SkillCrafterService.propose_repair` exposing the existing `PatchProposal` type | `crafter/service.py` |

Phases A → B → C → D → E → F are strict; do not start a phase before its predecessor's acceptance criteria are green.

---

## Quick start

### Install

```bash
cd Multi-hop-Reasoning-VLM-Agent
conda create -n vlm-agent python=3.11 -y
conda activate vlm-agent
pip install -e .
```

For full setup (CUDA toolchain, vLLM for the deferred Qwen tracks, game environments) see [`install/README.md`](install/README.md). The new build (`harness/`, `orchestrator/`, `crafter/`, `skill_bank/`, `tests/`) only requires the standard scientific Python stack and an OpenAI / OpenRouter API key for GPT-4o.

### Configure the backbone

```bash
cp .env.example .env
# minimum: OPENAI_API_KEY=...  (or OPENROUTER_API_KEY=...)
set -a && source .env && set +a
```

The default backbone is `gpt-4o`; nothing else needs to be set.

### Run the test suite

```bash
python -m pytest tests/ -v
```

Expected: `29 passed`. The suite covers the six invariants, an end-to-end `EpisodeRunner` smoke run, the crafter's failure → DRAFT proposal cycle, and the GPT-4o backbone pin.

### Smoke-run the orchestrator

```python
from common import StateSchema
from harness import AdapterRegistry, HarnessConfig, SkillHarness
from harness.adapters import GymvAdapter
from orchestrator import (
    ArtifactStore,
    BudgetController,
    EpisodeRunner,
    OrchestratorConfig,
)
from skill_bank import SkillLifecycleManager, SkillRepository
from skill_bank.stores import SkillStore, StoreName

# wire the bank
repo = SkillRepository(
    draft_store=SkillStore(StoreName.DRAFT, "_bank/draft"),
    candidate_store=SkillStore(StoreName.CANDIDATE, "_bank/candidate"),
    active_store=SkillStore(StoreName.ACTIVE, "_bank/active"),
    archive_store=SkillStore(StoreName.ARCHIVE, "_bank/archive"),
)
lifecycle = SkillLifecycleManager(repo)

# wire the harness
registry = AdapterRegistry()
registry.register(GymvAdapter())
harness = SkillHarness(adapter_registry=registry, config=HarnessConfig())

# run an episode (fake env / actor — see tests/test_smoke.py for a full example)
artifacts = ArtifactStore("_artifacts")
runner = EpisodeRunner(
    config=OrchestratorConfig(),
    harness=harness,
    repository=repo,
    artifact_store=artifacts,
)
result = runner.run(
    budget=BudgetController(),
    env=...,        # any object exposing reset() / step()
    actor=...,      # any object exposing choose(state, eligible) -> action
    initial_state=StateSchema(domain="gymv", task="demo", goal="demo", step=0),
)
print(result.outcome)
```

See [`tests/test_smoke.py::test_smoke_end_to_end`](tests/test_smoke.py) for the full runnable example.

---

## Plans index

The full plan corpus is in [`plans/`](plans/README.md). Recommended reading order:

1. [`plans/00-system/PLAN-SYSTEM-NORTHSTAR.md`](plans/00-system/PLAN-SYSTEM-NORTHSTAR.md) — single canonical scoreboard and stop/go rules.
2. [`plans/00-system/PLAN-EVAL-FIRST-TARGET.md`](plans/00-system/PLAN-EVAL-FIRST-TARGET.md) — Joint Success Rate contract and the `E0 → E1 → E2` rollout.
3. [`plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md`](plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md) and [milestones](plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md).
4. [`plans/02-action-agent/PLAN-ACTION-AGENT.md`](plans/02-action-agent/PLAN-ACTION-AGENT.md) — two-level MDP, three-agent role split.
5. [`plans/03-skill-bank/PLAN-SKILL-BANK.md`](plans/03-skill-bank/PLAN-SKILL-BANK.md) — invariants §0.1 / §0.3, cross-domain retrieval.
6. [`plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md`](plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md) — frozen-teacher proposal layer.
7. [`plans/05-harness/PLAN-HARNESS.md`](plans/05-harness/PLAN-HARNESS.md) — per-invocation runtime, gates G0–G5.
8. [`plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md`](plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) — control plane, episode-local evidence contract.
9. [`plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) — canonical lifecycle.
10. [`plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md`](plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md) — Cursor build sheet, Phase A → F.

Cross-cutting (read on demand): [`plans/08-cross-cutting/`](plans/08-cross-cutting/) — failure routing, uncertainty calibration, experience extension.

---

## Legacy COS-PLAY notes

The `decision_agents/`, `skill_agents/`, `vlm_wrapper/`, `trainer/`, `inference/`, and `env_wrappers/` directories carry the COS-PLAY codebase that this build supersedes. COS-PLAY is the **co-evolution framework over Qwen3-8B for game agents** described in:

> *COS-PLAY: Co-Evolving LLM Decision and Skill Bank Agents for Long-Horizon Game Play.*

The COS-PLAY entrypoints (`scripts/run_coevolution.py`, `scripts/qwen3_*.py`, `inference/run_qwen3_8b_eval.py`, `bash scripts/run_2048.sh`, etc.) and the corresponding documentation in their per-module READMEs remain valid — they are retained as the **deferred 8B/32B/72B Qwen tracks**. They do not run by default; the new build under `common/`, `harness/`, `orchestrator/`, `crafter/`, `skill_bank/` defaults to GPT-4o end-to-end.

Two pieces of glue are scheduled to land in the next sessions:

- A `decision_agents.skill_interface.HarnessSkillProvider` so the COS-PLAY Actor can consume `SkillHarness.select_eligible_skills` instead of querying the legacy bank directly.
- A one-way `skill_bank/legacy_bridge.py` that migrates Stage-3 `skill_agents/skill_bank` records into the new `SkillRecord` format.

---

## Citation and license

If you use this codebase, please cite both the multi-hop reasoning agent (this build) and the COS-PLAY paper:

```bibtex
@misc{multihop-vlm-agent,
  title={Multi-hop Reasoning VLM Agent: A Skill-Centric, Evidence-Driven, Gate-Bound Visual Agent},
  author={Wu, Xiyang and others},
  year={2026},
  note={Codebase: \url{https://github.com/wuxiyang1996/Multi-hop-Reasoning-VLM-Agent}}
}

@inproceedings{wu2026cosplay,
  title={Co-Evolving {LLM} Decision and Skill Bank Agents for Long-Horizon Game Play},
  author={Wu, Xiyang and others},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026}
}
```

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
