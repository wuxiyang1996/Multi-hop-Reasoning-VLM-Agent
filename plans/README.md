# Multi-hop Visual Reasoning Agent — System Plan

**Goal:** Train a VLM as a visual parser that converts pixels into structured summaries, and build a full skill-based agent pipeline on top of it — from visual grounding through skill-guided action selection.

**Primary transfer objective.** This project learns **transferable reasoning, grounding, and control skills across game, webagent, os-agent, video-understanding, and visual reasoning tasks**. Two invariants are enforced at the Harness gate, not just in prose:

1. **General-protocol invariant.** Every skill in the bank is a general protocol feasible across all five domains — no domain-specific skill families, no "short-video-only" skills. See [Skill Bank §0.1](PLAN-SKILL-BANK.md#01-general-protocol-invariant-no-domain-specific-skill-families).
2. **Evidence-driven invariant.** Every skill is evidence-driven and exists only to assist reasoning / decision-making. Every skill declares one of four roles — `GATHER | VERIFY | REASON | COMMIT` — and records a non-empty evidence interface (`evidence_in` or `evidence_out`, plus role-specific fields like `evidence_warrant` for `COMMIT`) on every successful episode. Opaque skills (empty evidence on both sides) are rejected by Gate G0. See [Skill Bank §0.3](PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills) and [Harness §10](PLAN-HARNESS.md#10-promotion-gates).

See [Visual Skills](PLAN-VISUAL-SKILLS.md) and [Harness](PLAN-HARNESS.md) for the transfer machinery that enforces both invariants.

**Near-term execution focus.** The first evaluation and deployment target is **short-video evidence-grounded reasoning** (Video-Holmes-style). Short-video is the first **arena** in which general protocols are exercised — it is not a separate class of skills. See [Current execution focus](#current-execution-focus) for what this repo does and does not include.

---

## Pipeline overview

```
Pixels (game frame / screenshot / video)
    ↓
(1) Visual Grounding     — VLM parser → structured <state> schema
    ↓
(2) Action Agent         — two-level MDP: inner reasoning hops → environment actions
    ↓                        ↑
(3) Skill Bank           — trajectory segmentation → skill contracts → retrieval
    ↓                        ↑
(4) Skill Crafter        — compose / generalize / hypothesize new skills
    ↓
    └──→ feeds back into Skill Bank
```

---

## Plan documents

| # | Plan | Scope |
|---|------|-------|
| 1 | **[Visual Grounding](PLAN-VISUAL-GROUNDING.md)** | VLM parser, canonical schema, grounding heads (heuristic, vision, OmniParser, tool loop), domain adapters (Gym-V, BrowserGym, OSWorld), benchmark evaluation (CLEVR, GQA, SIV-Bench, Video-Holmes), schema completeness guarantee (§12), Qwen3-VL-8B training |
| 1b | **[Visual Grounding Milestones](PLAN-VISUAL-GROUNDING-MILESTONES.md)** | Concrete execution plan: 5-stage inference pipeline, routing policy (Path A/B/C), training phases 0–4, week-by-week schedule, 7 ablations, success criteria |
| 2 | **[Action Agent](PLAN-ACTION-AGENT.md)** | Two-level MDP (outer env + inner reasoning hops), decision loop, GROUND/CHECK/RETRIEVE/COMMIT/EXECUTE inner actions, **7B/8B capability assessment (§2)**, **three-agent role split (§2)**, **co-evolution & GRPO decomposition (§6)**, **training schedule & timescale separation (§6)**, uncertainty-driven GROUND triggering (§10), tiered model architecture (Tier 0/1/2), reward shaping (r_env + r_follow + r_cost) |
| 3 | **[Skill Bank](PLAN-SKILL-BANK.md)** | **Cross-task** skill bank for reasoning and control across games, web, video, visual reasoning, and embodied tasks. Skill as structured-state program (§0.5), shared inner primitives + adapter-based binding (§1.5), unified structured state interface with entity ontology (§3), typed slots + domain adapters in data model (§4), 5-stage pipeline, effect families + 3-layer hierarchy (§8), 6 transferable skill families (§9), **asymmetric GRPO co-evolution with acceptance gates (§7)**, phase detection across domains (§5), query/select API with cross-domain retrieval (§6) |
| 4 | **[Skill Crafter](PLAN-SKILL-CRAFTER.md)** | Skill composition (effect chaining + hop protocol chaining), cross-domain generalization (schema-slot transfer), transferable skill families (4 cross-domain families), novel skill hypothesis, **frozen teacher design with phased adaptation policy (§2)**, **frozen teacher improvement channels (§2)**, failure reflection & counterfactual reasoning, integration with visual grounding tool traces |
| 5 | **[Visual Skills](PLAN-VISUAL-SKILLS.md)** *(optional)* | Transferable visual grounding strategies as skills — unified skill format for cross-domain transfer (preconditions/effects/slots/adapters), two kinds of effects (world vs belief/grounding), effect families, cross-domain entity ontology, three-layer skill bank hierarchy, automatic skill discovery from state transitions |
| 6 | **[Pipeline Orchestrator](PLAN-PIPELINE-ORCHESTRATOR.md)** | End-to-end harness — rollout DAG (online + offline), unified artifact/log schema, centralized acceptance gate with promotion/rollback, **episode-local evidence & trace bookkeeping contracts**, training cadence by timescale, full-system evaluation matrix, budget controller, failure escalation and human audit points |
| 7 | **[Skill Harness](PLAN-HARNESS.md)** | Per-invocation runtime for skill use, validation, and transfer — `SkillEpisode`, `SkillHarness`, `AdapterRegistry`, `TransferManager`, `ReplayValidator`, `RewardLogger`; semantic-skill vs. domain-adapter separation; two-phase shadow→active transfer protocol; six-gate promotion (G0 evidence-driven / binding / adapter / replay / shadow / non-regression); Phase 0+1 as the immediate implementation target; composes with Pipeline Orchestrator (macro DAG) as its micro runtime |
| 8 | **[Unified Skill Gate](PLAN-UNIFIED-SKILL-GATE.md)** | Canonical lifecycle and promotion specification spanning Skill Bank, Harness, and Pipeline Orchestrator. Defines `SkillStatus` (`draft → candidate → shadow → provisional → active`, plus `deprecated / rejected / rolled_back`), `SkillSourceType`, `SkillRecord`, `SkillEvaluationRecord`, `GateVerdict`. Pins ownership: Skill Bank owns lifecycle + writes via `SkillLifecycleManager`; Harness owns gate execution via `GateRunner`; Orchestrator owns promotion / rollback transactions via `PromotionOrchestrator`. Storage split (`draft_store / candidate_store / active_store / archive_store`) makes "no promotion without gate" mechanically enforceable. All skill sources — mined, crafted, repaired, transferred, teacher-proposed (frozen 32B/72B), human-seeded — pass the same gate stack: static → replay → shadow → transfer → non-regression |
| 9 | **[Components Implementation](PLAN-COMPONENTS-IMPLEMENTATION.md)** | Cursor-ready build sheet for `harness/`, `crafter/`, `orchestrator/`. Pins target repo layout (`src/harness/ src/crafter/ src/orchestrator/ src/skill_bank/ src/common/`), the strict phase order **A (Harness MVP) → B (Orchestrator MVP) → C (Crafter MVP) → D (Transfer + Replay) → E (Eval + dashboards) → F (optional trainable extensions)** with per-phase acceptance criteria, the architectural boundaries that must hold across phases (Harness ≠ Crafter ≠ Orchestrator), and a paste-ready Cursor prompt encoding all required invariants (no promotion without gate; replay before promote; G0 evidence enforcement; atomic snapshot move; Crafter outputs are candidates only; Harness never selects the final skill; Orchestrator never executes per-invocation logic). Does not duplicate canonical specs — links into `PLAN-HARNESS.md`, `PLAN-SKILL-CRAFTER.md`, `PLAN-PIPELINE-ORCHESTRATOR.md`, `PLAN-UNIFIED-SKILL-GATE.md` for the *why* and *what*; this plan owns the *how* and *when* |

**Design reference:** [`LONG_HORIZON_REASONING.md`](../LONG_HORIZON_REASONING.md) — the two-level MDP framing that unifies the component plans (grounding through bank/crafter); the [Pipeline Orchestrator](PLAN-PIPELINE-ORCHESTRATOR.md) specifies cross-cutting execution and gates.

---

## Key shared concepts

### Canonical `<state>` schema

All four components share the same structured schema format (defined in [Visual Grounding §3](PLAN-VISUAL-GROUNDING.md)):

```text
<state>
domain={browser|gymv}
task={task_id}
goal={goal string}
step={int}

<entities>
e1[type=..., label=..., pos=x,y,w,h]

<attributes>
e1.state=visible,clickable

<relations>
adjacent(e1,e2)

<state_flags>
progress=0.3
phase=mid

<targets>
target=e1
blocker=null

<actions>
a1=click(e1)
</state>
```

### Shared slot names (cross-domain transfer)

`target`, `blocker`, `constraint`, `candidate_set`, `history_anchor` — used by all four plans for downstream skill transfer.

### Two-level MDP (with a lightweight inner loop)

Multi-hop visual reasoning is framed as a **two-level MDP** (see [`LONG_HORIZON_REASONING.md`](../LONG_HORIZON_REASONING.md)). The outer level is the environment; the inner level is a **lightweight typed control loop** — not a free-form reasoning generator:

```
┌── OUTER MDP (environment level) ──────────────────────┐
│  State: screenshot + task    Action: click/type/move   │
│                                                        │
│  ┌── INNER MDP (typed local control, ≤3 hops) ────┐  │
│  │  State: <state> schema + short typed trace      │  │
│  │  Actions: GROUND | CHECK | RETRIEVE | COMMIT    │  │
│  │           | EXECUTE (exits inner loop)           │  │
│  └─────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

Online hop depth is capped at 0–2 by default and ≤3 under uncertainty. `hop_select` is a constrained typed next-hop router (not a mini planner). Skills still capture *how to think* (short typed hop chains), not just *what to do*. GRPO optimizes actor heads end-to-end, but heavy reasoning — failure diagnosis, skill composition, transfer, new skill invention — is strictly offline (32B/72B). See [Action Agent §5](PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control) for the full design.

### Three-agent role split & model convention

The system decomposes into three logical agents with distinct model assignments and update timescales. See [Action Agent §2](PLAN-ACTION-AGENT.md#three-agent-role-split) for the full specification.

| Agent | Model | Role | Update timescale |
|-------|-------|------|------------------|
| **Agent 1: Actor / Decision Agent** | Qwen3-8B (GRPO-trained) | Online action execution, hop selection, skill selection, protocol following | Fast — every training iteration |
| **Agent 2: Skill-Use / Operational** | Qwen3-8B (GRPO-trained) + rules | RAG skill retrieval, _SkillTracker lifecycle, segmentation, contract learning, curation | Medium — every few iterations |
| **Agent 3: Synthesis-Reflection** | Qwen3-32B/72B (inference-only, **frozen**) | Failure reflection, skill composition, skill hypothesis, cross-domain transfer, cold-start trajectories, verification/judging | Slow — proposals every N episodes, gated |

**Additional models:**
- **GPT-5.4** — training-free cold-start and labeling (Tier 0)
- **Qwen3-VL-8B** — visual grounding VLM (distilled from GPT-4o labels)

**Key design principles:**
- The 32B/72B teacher is frozen first. Its outputs are *candidate proposals*, admitted only after multi-pass verification and held-out replay checks.
- Co-evolution is asymmetric: fast GRPO for the actor, selective GRPO for skill-side operational decisions, slow verified large-model reflection for synthesis.
- The frozen teacher improves via better input data, better **evidence organization**, better **replay validation**, better **transfer routing**, better inference procedures, better verification, and optional distillation — not weight updates (initially).
- See [Action Agent §6](PLAN-ACTION-AGENT.md#6-co-evolution--grpo-decomposition) for the full co-evolution design and training schedule.

### Unified principle for Actor and frozen Harness

In this project, the Actor and Harness serve **different purposes** and must not collapse into each other.

- The **Actor** is the trainable online decision-making policy. It follows the COS-PLAY Decision Agent pattern and operates over schema-based state. At each step, it must decide whether to continue a skill, switch skills, act without a skill, emit a typed reasoning step, or take a primitive action. See [PLAN-ACTION-AGENT.md §1a](PLAN-ACTION-AGENT.md#1a-actor-role-and-boundary).
- The **Harness** is a frozen 72B runtime support layer. It does **not** replace the Actor's policy role. Instead, it acts as a high-capacity **verifier, candidate filter, veto layer, and teacher-like advisor**. It narrows the candidate space, validates runtime feasibility, and blocks invalid skill invocations, but **the final online decision remains with the Actor**. See [PLAN-HARNESS.md §1a](PLAN-HARNESS.md#1a-harness-role-as-frozen-72b-runtime-layer).
- The **Skill Bank** provides candidates; the **Harness** constrains the candidate space; the **Actor** decides; the **Orchestrator** handles offline promotion and rollback. See [PLAN-PIPELINE-ORCHESTRATOR.md §0a](PLAN-PIPELINE-ORCHESTRATOR.md#0a-actor-harness-skill-bank-orchestrator-boundary) for the canonical four-way boundary and the online skill invocation interface.

> Since the Harness is a frozen 72B model, it should not replace the Actor as the online policy. Instead, it should serve as a high-capacity runtime verifier, candidate filter, veto layer, and teacher-like advisor. The Actor, following the COS-PLAY Decision Agent pattern, remains the final decision maker over skill continuation, skill switching, no-skill fallback, typed reasoning steps, and primitive actions. The Skill Bank provides candidates, the Harness constrains the candidate space, and the Orchestrator handles offline promotion and rollback.

This separation keeps the architecture consistent, preserves the learning burden of the Actor, and prevents the frozen large model from silently becoming the true policy.

### LoRA adapter layout

| Adapter | Agent | Purpose | Plan |
|---------|-------|---------|------|
| `schema_gen` | Agent 1 | Screenshot → `<state>` schema | [Visual Grounding](PLAN-VISUAL-GROUNDING.md) |
| `hop_select` | Agent 1 | Typed next-hop router: schema + short typed trace → `(NEXT_HOP, TARGET)` from `{GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE}`; 0–3 hop cap | [Action Agent](PLAN-ACTION-AGENT.md) |
| `skill_select` | Agent 1 | Schema → which reasoning skill to invoke | [Action Agent](PLAN-ACTION-AGENT.md) / [Skill Bank](PLAN-SKILL-BANK.md) |
| `segment` | Agent 2 | Trajectory → skill boundary detection | [Skill Bank](PLAN-SKILL-BANK.md) |
| `contract` | Agent 2 | Segment → effects contract | [Skill Bank](PLAN-SKILL-BANK.md) |

### Co-evolution overview

```
┌─ Fast timescale ────────────────────────────────────────────┐
│  Actor (8B): GRPO on action_execute, skill_select, hop_select│
│  Updates every training iteration                            │
└──────────────────────────┬───────────────────────────────────┘
                           ↕ experience trajectories
┌─ Medium timescale ──────────────────────────────────────────┐
│  Skill Bank (8B): GRPO on SEGMENT, CONTRACT, CURATOR        │
│  Selective GRPO on continue/switch, accept/reject, merge/split│
│  Updates every few iterations                                │
└──────────────────────────┬───────────────────────────────────┘
                           ↕ candidate skills, patches
┌─ Slow timescale ────────────────────────────────────────────┐
│  Synthesis-Reflection (32B/72B frozen):                      │
│    failure reflection, composition, hypothesis, transfer     │
│  Proposals every N episodes → acceptance gate → bank         │
└─────────────────────────────────────────────────────────────┘
```

---

## Current execution focus

The skill ontology is cross-domain and transferable across **game / webagent / os-agent / video-understanding / visual reasoning**, and **every skill in the bank is a general protocol feasible across all five** ([Skill Bank §0.1](PLAN-SKILL-BANK.md#01-general-protocol-invariant-no-domain-specific-skill-families)). The current repo narrows the *execution target* — not the skill ontology — to keep scope tractable:

- **Episode-local state only.** The agent's only state-keeping surface within an episode is the structured `<state>`, the short typed hop trace, an intermediate belief state, and adapter-provided within-episode evidence references. There is no cross-episode storage layer.
- **No long-video reasoning in this repo.** Hour-scale video and multi-session continuity are out of scope.
- **First benchmark target:** **short-video multi-hop reasoning** with evidence chaining (Video-Holmes-style).
- **What short-video exercises:** general protocols such as `collect_evidence_chain`, `disambiguate_target`, `locate_filter_select`, `actor_action_binding`, and `verify_constraint` — each of which carries five-domain adapter bindings from day one. Short-video is the first arena where these protocols earn their `verified_domains` entry; it is not a separate skill family.
- **Skill ontology is fixed across phases:** every protocol's adapters for game / webagent / os-agent / visual reasoning exist alongside the video adapter and are exercised in a staggered order, but the protocols themselves are the same.

See [PLAN-PIPELINE-ORCHESTRATOR.md §4](PLAN-PIPELINE-ORCHESTRATOR.md#4-episode-local-evidence--trace-bookkeeping) for the episode-local evidence & trace bookkeeping contract that defines what the agent actually keeps within an episode.

---

## One-sentence framing

This project builds a skill-centric agent — operating over an episode-local trajectory of structured `<state>`, short typed hop trace, intermediate belief state, and within-episode evidence references — that learns **transferable reasoning, grounding, and control skills as general protocols feasible across game, webagent, os-agent, video-understanding, and visual reasoning**, with **short-video evidence-grounded reasoning** as the first arena where those protocols are exercised and verified. Online execution uses 8B models with GRPO; offline skill synthesis and reflection use a frozen 32B/72B teacher whose outputs are verified before admission.
