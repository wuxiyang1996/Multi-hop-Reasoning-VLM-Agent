# Multi-hop Reasoning VLM Agent

**A skill-centric, evidence-driven, gate-bound visual agent.** This repository builds a Visual Language Model (VLM) agent that converts pixels into a structured `<state>` schema and acts on it through a four-stage pipeline of *Visual Grounding → Action → Skill Bank → Skill Crafter*, governed by three operational components (*Skill Harness*, *Pipeline Orchestrator*, *Unified Skill Gate*).

The system learns **transferable reasoning, grounding, and control skills as general protocols feasible across game, webagent, os-agent, video-understanding, and visual reasoning tasks**. The first concrete arena is **short-video evidence-grounded reasoning** (Video-Holmes-style); cross-domain generalization is a hard, mechanically-enforced invariant of the skill bank, not an aspiration.

This repo supersedes the COS-PLAY codebase that lives alongside it under `decision_agents/`, `skill_agents/`, `vlm_wrapper/`, and `data_structure/legacy/`. Those modules remain importable as a reference for the legacy single-domain GRPO loop; the new build under `common/`, `harness/`, `orchestrator/`, `crafter/`, `skill_bank/`, and `data_structure/extensions/` implements the canonical plan from [`plans/`](plans/README.md).

---

## Table of Contents

- [Why this project](#why-this-project)
- [Architecture](#architecture)
- [Mechanically-enforced invariants](#mechanically-enforced-invariants)
- [Skill transfer layer](#skill-transfer-layer)
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

**Core thesis — games as the skill foundry, other domains as few-shot transfer targets.** Games are uniquely well-suited to *learn* multi-hop reasoning, grounding, and control skills: they expose dense, cheap, verifiable rewards; deterministic resets; and rich multi-hop structure (gather → verify → chain → commit) over a controllable visual state. This project uses games as the **source domain** in which skills are discovered, gated, and hardened, and then **transfers those same skills to webagent, os-agent, short-video understanding, and visual reasoning via few-shot adaptation** — a handful of target-domain episodes are enough to bind a game-learned protocol to a new adapter, because the skill is a typed protocol over evidence, not a domain-specific policy.

This project takes the opposite stance to opaque, single-domain skill agents:

1. **Every skill is a general protocol, learned in games and few-shot adapted elsewhere.** A skill must declare adapter bindings to all five domains (game, webagent, os-agent, video, visual reasoning); the *game* binding is where it is first discovered and stress-tested, and the remaining bindings are earned via few-shot transfer trials at the gate. Single-domain skills are rejected at promotion time. See [Skill Bank §0.1](plans/03-skill-bank/PLAN-SKILL-BANK.md#01-general-protocol-invariant-no-domain-specific-skill-families).
2. **Every skill is evidence-driven** — it must declare a role from `{GATHER, VERIFY, REASON, COMMIT}` and record a non-empty evidence interface on every successful episode. Opaque skills are rejected at Gate G0. The evidence interface is what makes few-shot transfer tractable: only the adapter bindings change across domains, while the typed evidence contract stays fixed. See [Skill Bank §0.3](plans/03-skill-bank/PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills).
3. **Every promotion is gate-bound, with a dedicated transfer gate.** No proposal reaches `ACTIVE` without passing the canonical gate stack (`static → replay → shadow → transfer → non-regression`). The `transfer` stage is the few-shot adaptation check: a game-learned skill must succeed on a small budget of episodes in at least one non-game domain before its `verified_domains` entry is granted. See [Unified Skill Gate](plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md).
4. **The Actor is the policy, the Harness is a frozen verifier.** The Skill Bank provides candidates, the Harness narrows + may veto, the Actor decides, the Orchestrator handles offline promotion. The frozen large model never silently becomes the policy. See [Pipeline Orchestrator §0a](plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#0a-actor-harness-skill-bank-orchestrator-boundary).

The first benchmark target outside the game foundry is short-video multi-hop reasoning, but the skill ontology is fixed across phases — short-video is the first **transfer arena** in which game-learned general protocols (e.g. `collect_evidence_chain`, `disambiguate_target`, `locate_filter_select`, `actor_action_binding`, `verify_constraint`) earn their `verified_domains` entry through few-shot adaptation. Webagent, os-agent, and visual reasoning follow as additional transfer arenas under the same gate.

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
| 7 | **Source-domain / transfer-target asymmetry**: every `ACTIVE` skill must declare at least one `source_domains` entry from `SOURCE_DOMAINS` (currently `{"gymv"}`) **and** at least one `verified_domains` entry from `TRANSFER_TARGET_DOMAINS` (`{browser, osworld, video, visual_reasoning}`). Game-only or transfer-only skills cannot be active. | `SkillLifecycleManager._validate_invariants` raises on ACTIVE promotion when either side is empty (additive to invariant 3, back-compat for legacy records). |
| 8 | **`verified_domains` is gate-owned**: the `verified_domains` field (and the matching `adapter_history` entries) reflect Stage 3a outcomes only. The pipeline is: `GateService._run_transfer` runs `FewShotAdapter.adapt()` on K demos, the verified targets are emitted in `GateVerdictPayload.eligible_domains`, and `PromotionOrchestrator.promote(...)` mirrors them into the `SkillRecord` via `SkillLifecycleManager.record_transfer_verification(...)` *before* the status transition. The lifecycle manager is the only writer; any other component is forbidden. | `SkillLifecycleManager.record_transfer_verification` is the sole mutator of `verified_domains` / `adapter_history`; `PromotionOrchestrator._record_transfer_verifications` calls it from the pre-transition step. |

---

## Skill transfer layer

The "games as foundry, other domains as transfer targets" thesis from [§Why this project](#why-this-project) is implemented end-to-end as a stack of seven concrete components. None of them is a placeholder — they all run today as part of the gate stack and the test suite (`tests/test_few_shot_transfer.py`).

### Source / target asymmetry — the substrate

Two hard-coded tuples in [`common/enums.py`](common/enums.py) encode the asymmetry. They are imported everywhere a domain string is interpreted, so "is this a foundry domain?" / "is this a transfer arena?" is one constant lookup, never a free string:

```30:36:common/enums.py
SOURCE_DOMAINS: Tuple[str, ...] = ("gymv",)
TRANSFER_TARGET_DOMAINS: Tuple[str, ...] = (
    "browser",
    "osworld",
    "video",
    "visual_reasoning",
)
```

A skill that only ever ran in `gymv` carries `source_domains={"gymv"}` and an empty `verified_domains`; it cannot reach `ACTIVE` (invariants 7 & 8). A skill that earned a transfer-gate pass on, say, `video` gets `"video"` appended to `verified_domains` *and* a `TransferAdapterEvent` appended to `adapter_history` — both writes go through `SkillLifecycleManager.record_transfer_verification(...)` and nowhere else.

### Generalizer — proposing transferable templates

[`crafter/generalizer.py`](crafter/generalizer.py) consumes per-domain skills and emits `GeneralizeProposal` records that strip away domain-specific predicates, replacing them with the canonical inner-MDP vocabulary (`GROUND / CHECK / RETRIEVE / COMMIT / EXECUTE`) and shared schema slots. The output is always a `DRAFT` record with `source_domains={"gymv"}` and `verified_domains=()` — the gate stack is the only path that can extend either field.

The bridging types `TransferableSkill`, `SlotBinding`, `ReasoningProtocol`, `HopStep`, `AbstractPredicate`, and the `FAMILY_PROTOCOLS` registry live in [`skill_agents/skill_template.py`](skill_agents/skill_template.py); the discovery pipeline that produces them from legacy episodes is [`skill_agents/extract_transferable.py`](skill_agents/extract_transferable.py). See [`skill_agents/README.md`](skill_agents/README.md) for the legacy-bridge details.

### Few-Shot Adapter — the transferability probe

[`harness/few_shot_adapter.py`](harness/few_shot_adapter.py) is the single place where "does this skill bind to a new domain?" is decided. It is stateless: given a `SkillRecord`, a target domain, and a sequence of `FewShotDemo`s, it

1. Validates eligibility (`SkillType` ∈ adapter's supported types, source-domain check),
2. Coerces each demo's `state` into a target-domain `StateSchema`,
3. Runs the skill via `SkillHarness.run_skill(...)` against the registered adapter for the target domain,
4. Scores each shot with a pluggable `success_fn` (default: episode outcome + contract satisfaction),
5. Returns an `AdaptResult{success_rate, n_success, n_total, episodes, …}`.

Critically it never mutates the bank — it is a read-only probe used by the gate stack and by ad-hoc transferability tests.

### Stub adapters — common scaffolding for the four target domains

[`harness/adapters/_stub_base.py`](harness/adapters/_stub_base.py) defines `StubTransferTargetAdapter`, the shared hop-loop scaffolding for `browser`, `osworld`, `video`, and `visual_reasoning`. Each concrete adapter (e.g. [`harness/adapters/browser.py`](harness/adapters/browser.py)) is just a name + a default `HopExecutor`. The executor is pluggable (`adapter.set_executor(real_executor)`), so today's deterministic stubs let the gate stack be exercised end-to-end while the real `vlm_wrapper/<domain>_adapter.py` executors are wired up independently.

### Gate stage 3a — wiring the probe into promotion

[`orchestrator/gate_service.py`](orchestrator/gate_service.py) composes the canonical gate stack `static → replay → shadow → transfer → non-regression`. Stage 3a (`_run_transfer`) is the only stage that calls `FewShotAdapter.adapt(...)`:

```219:316:orchestrator/gate_service.py
def _run_transfer(
    self,
    skill: SkillRecord,
    *,
    few_shot_demos: Optional[Mapping[str, Sequence[FewShotDemo]]] = None,
) -> tuple[StageVerdict, List[str]]:
    # ... infer targets from TRANSFER_TARGET_DOMAINS ∩ adapter registry ...
    adapt_results: List[AdaptResult] = []
    for tgt in targets:
        shots = (few_shot_demos or {}).get(tgt, ())
        adapt_results.append(
            self._few_shot.adapt(skill=skill, target_domain=tgt, demos=shots)
        )
    # ...
    if len(verified) >= thresholds.transfer_min_target_domains_verified:
        verdict = GateVerdict.PASS
    elif len(verified) >= 1:
        verdict = GateVerdict.LIMITED_PASS
    else:
        verdict = GateVerdict.FAIL
```

The verified target-domain list is emitted as `GateVerdictPayload.eligible_domains` and passed back to the orchestrator. The verdict is `PASS` if ≥ `transfer_min_target_domains_verified` targets succeed, `LIMITED_PASS` if at least one does, otherwise `FAIL`.

### Promotion Orchestrator — mirroring results into the bank

[`orchestrator/promotion_orchestrator.py`](orchestrator/promotion_orchestrator.py) extracts the Stage-3a verdict from the `SkillEvaluationRecord`, converts it into per-target metrics, and calls the lifecycle manager **before** the status transition:

```208:254:orchestrator/promotion_orchestrator.py
self._lifecycle.record_transfer_verification(
    skill.skill_id,
    verified_targets=verified,
    evaluation_id=evaluation.evaluation_id,
    per_target_metrics=per_target_metrics,
    rationale=f"stage_3a:{rationale}" if rationale else "stage_3a",
)
```

This is the only call site outside the lifecycle manager itself that touches the transfer-verification path.

### Lifecycle Manager — the only writer + the ACTIVE-promotion invariant

[`skill_bank/lifecycle.py`](skill_bank/lifecycle.py) is the sanctioned writer. `record_transfer_verification(...)` extends `verified_domains` and appends a `TransferAdapterEvent` to `adapter_history` atomically; `_validate_invariants(...)` then refuses any `ACTIVE` promotion of a source-tagged skill that has not yet earned at least one verified target domain:

```272:283:skill_bank/lifecycle.py
if to_status == SkillStatus.ACTIVE and record.source_domains:
    if not any(d in SOURCE_DOMAINS for d in record.source_domains):
        raise LifecycleError(
            f"Cannot promote {record.skill_id!r} to ACTIVE: "
            f"source-domain (game-foundry) lineage required, …"
        )
    if not any(d in TRANSFER_TARGET_DOMAINS for d in record.verified_domains):
        raise LifecycleError(
            f"Cannot promote {record.skill_id!r} to ACTIVE: "
            f"few-shot transfer gate (G3a) requires ≥1 verified target "
            f"domain, …"
        )
```

This is the mechanical realisation of invariants 7 and 8: source-domain lineage and gate-owned `verified_domains` are not conventions — they are preconditions on the only function that can mark a skill `ACTIVE`.

### TL;DR call graph

```
crafter/generalizer.py        → DRAFT SkillRecord (source_domains={"gymv"}, verified_domains=())
        ↓
orchestrator/gate_service._run_transfer
        ↓ for each target ∈ TRANSFER_TARGET_DOMAINS ∩ AdapterRegistry
harness/few_shot_adapter.adapt
        ↓ runs skill via SkillHarness.run_skill against StubTransferTargetAdapter
        ↓ scores K demos → AdaptResult
        ↓
orchestrator/gate_service       → GateVerdictPayload(eligible_domains=verified[, …])
        ↓
orchestrator/promotion_orchestrator._record_transfer_verifications
        ↓
skill_bank/lifecycle.record_transfer_verification
        ↓ extends verified_domains + appends adapter_history
skill_bank/lifecycle._validate_invariants
        ↓ allows ACTIVE iff source ∩ SOURCE_DOMAINS ≠ ∅ and verified ∩ TRANSFER_TARGET_DOMAINS ≠ ∅
ACTIVE SkillRecord (transferable, evidence-driven, gate-bound)
```

End-to-end coverage lives in [`tests/test_few_shot_transfer.py`](tests/test_few_shot_transfer.py): it exercises the `SkillRecord` persistence path, the lifecycle invariants, the `FewShotAdapter` execution loop, and the `GateService` verdict shape in a single fixture set.

### What's still pending (transfer-side)

| Item | Owning module |
| --- | --- |
| Two-phase **shadow → active** transfer protocol on top of Stage 3a | `harness/transfer_manager.py` |
| Unified six-gate runner (`G0–G5`) consuming `gate_service` stages | `harness/gate_runner.py` |
| Real (non-stub) executors for each transfer-target adapter | `vlm_wrapper/<domain>_adapter.py` |
| Held-out replay seeds for Stage 1 | `harness/replay_validator.py` |

These are tracked under Phase D in [`IMPLEMENTATION-STATUS.md`](IMPLEMENTATION-STATUS.md) and in the [Pending](#pending-next-sessions) row above.

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

### Schema → predicate bridge ([`skill_agents/schema_predicates.py`](skill_agents/schema_predicates.py))

The unified visual grounding head ([`visual_grounding_tests/generate_gymv_image_schema.py`](visual_grounding_tests/generate_gymv_image_schema.py)) emits `<state>...</state>` blocks built by [`vlm_wrapper.schema`](vlm_wrapper/schema.py). The legacy Stage-3 contract learner ([`skill_agents/stage3_mvp/`](skill_agents/stage3_mvp)) consumes a flat predicate-probability map (`Dict[str, float]`). [`skill_agents/schema_predicates.py`](skill_agents/schema_predicates.py) is the pure-Python adapter between them — it lets game and `env_wrappers` rollouts produced by the unified grounding head feed the legacy COS-PLAY skill-mining pipeline (boundary proposal → segmentation → contract learning → bank maintenance) without re-implementing per-domain extractors.

Predicate keys are flat strings encoding the source section, so the cross-domain skill templates in [`skill_agents/skill_template.py`](skill_agents/skill_template.py) match them directly:

| Section | Predicate key shape |
| --- | --- |
| `<entities>` | `entity:<eid>:exists`, `entity:<eid>:type:<t>`, `entity:<eid>:ontology:<role>` |
| `<attributes>` | `attr:<eid>:<key>=<value>` |
| `<affordances>` | `afford:<eid>:<verb>` |
| `<relations>` | `rel:<verb>:<eA>:<eB>[:<eC>...]` |
| `<state_flags>` | `flag:<name>` (boolean) / `flag:<name>=<bucket>` (categorical) |
| `<targets>` | `target:eid=<eid>`, `target:blocker=<eid>`, `target:candidate:<eid>`, `target:history_anchor=<eid>` |

The `<uncertainty>` block attenuates per-(eid, field) probabilities to `0.4` for `high`, `0.6` for `medium`, `1.0` for `low`. Stage 3 booleanises predicates at ~0.5, so a `high`-uncertainty entity flips to *absent* in the contract learner — exactly the desired behaviour.

Plug it in as a source on the existing predicate extractor:

```python
from skill_agents.stage3_mvp.extract_predicates import CompositePredicateExtractor
from skill_agents.stage3_mvp.predicate_vocab import PredicateVocab
from skill_agents.schema_predicates import schema_to_predicates

vocab = PredicateVocab()
extractor = CompositePredicateExtractor(vocab)
extractor.add_source(schema_to_predicates)

preds = extractor(experience.summary_state)  # accepts str / dict / Experience-like
```

Coverage lives in [`tests/test_schema_predicates.py`](tests/test_schema_predicates.py) (entity / attribute / affordance / relation / target / state-flag / uncertainty / input-coercion / robustness — including a real `CompositePredicateExtractor` integration check).

### Other glue scheduled to land in the next sessions

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
