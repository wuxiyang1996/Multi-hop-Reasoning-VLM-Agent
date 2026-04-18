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
| 2 | **[Action Agent](PLAN-ACTION-AGENT.md)** | Two-level MDP (outer env + inner reasoning hops), decision loop, GROUND/CHECK/RETRIEVE/CONCLUDE/EXECUTE inner actions, **7B/8B capability assessment (§2)**, **three-agent role split (§2)**, **co-evolution & GRPO decomposition (§6)**, **training schedule & timescale separation (§6)**, uncertainty-driven GROUND triggering (§10), tiered model architecture (Tier 0/1/2), reward shaping (r_env + r_follow + r_cost) |
| 3 | **[Skill Bank](PLAN-SKILL-BANK.md)** | 5-stage pipeline (boundary proposal → segmentation → contract learning → bank maintenance → quality eval), **asymmetric GRPO co-evolution with acceptance gates (§7)**, transferable skill extraction (§10), reasoning skill discovery (hop chain templates), phase detection, proto-skill staging, query/select API |
| 4 | **[Skill Crafter](PLAN-SKILL-CRAFTER.md)** | Skill composition (effect chaining + hop protocol chaining), cross-domain generalization (schema-slot transfer), transferable skill families (4 cross-domain families), novel skill hypothesis, **frozen teacher design with phased adaptation policy (§2)**, **frozen teacher improvement channels (§2)**, failure reflection & counterfactual reasoning, integration with visual grounding tool traces |

**Design reference:** [`LONG_HORIZON_REASONING.md`](../LONG_HORIZON_REASONING.md) — the two-level MDP framing that unifies all four plans.

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

### Two-level MDP (long-horizon reasoning)

Multi-hop visual reasoning is reframed as a **long-horizon interaction** problem (see [`LONG_HORIZON_REASONING.md`](../LONG_HORIZON_REASONING.md)). Each reasoning hop becomes an explicit step in an inner MDP:

```
┌── OUTER MDP (environment level) ──────────────────────┐
│  State: screenshot + task    Action: click/type/move   │
│                                                        │
│  ┌── INNER MDP (reasoning level) ──────────────────┐  │
│  │  State: <state> schema + hop trace              │  │
│  │  Actions: GROUND | CHECK | RETRIEVE | CONCLUDE  │  │
│  │           | EXECUTE (exits inner loop)           │  │
│  └─────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

The agent **learns when to stop reasoning and act**. Skills capture *how to think* (reasoning hop chains), not just *what to do*. GRPO optimizes the full reasoning chain end-to-end. See [Action Agent §5](PLAN-ACTION-AGENT.md) for the full design.

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
| `hop_select` | Agent 1 | Schema + trace → next reasoning action | [Action Agent](PLAN-ACTION-AGENT.md) |
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
