# Multi-hop Visual Reasoning Agent — System Plan

**Goal:** Train a VLM as a visual parser that converts pixels into structured summaries, and build a full skill-based agent pipeline on top of it — from visual grounding through skill-guided action selection.

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
| 6 | **[Pipeline Orchestrator](PLAN-PIPELINE-ORCHESTRATOR.md)** | End-to-end harness — rollout DAG (online + offline), unified artifact/log schema, centralized acceptance gate with promotion/rollback, memory integration contracts, training cadence by timescale, full-system evaluation matrix, budget controller, failure escalation and human audit points |
| 7 | **[Skill Harness](PLAN-HARNESS.md)** | Per-invocation runtime for skill use, validation, and transfer — `SkillEpisode`, `SkillHarness`, `AdapterRegistry`, `TransferManager`, `ReplayValidator`, `RewardLogger`; semantic-skill vs. domain-adapter separation; two-phase shadow→active transfer protocol; five-gate promotion (binding / adapter / replay / shadow / non-regression); Phase 0+1 as the immediate implementation target; composes with Pipeline Orchestrator (macro DAG) as its micro runtime |

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
- The frozen teacher improves via better input data, better memory, better inference procedures, better verification, and optional distillation — not weight updates (initially).
- See [Action Agent §6](PLAN-ACTION-AGENT.md#6-co-evolution--grpo-decomposition) for the full co-evolution design and training schedule.

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

## One-sentence framing

We train a VLM as a visual parser — supervised by the text state that game and web environments already provide for free — that converts pixels into structured summaries for a skill-based agent pipeline, and learns to call environment APIs for information it cannot see. Online execution uses 8B models with GRPO; offline skill synthesis and reflection use a frozen 32B/72B teacher whose outputs are verified before admission.
