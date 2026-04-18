# TODO-VLM: Multi-hop Visual Reasoning Agent

**Goal:** Train a VLM as a visual parser that converts pixels into structured summaries, and build a full skill-based agent pipeline on top of it — from visual grounding through skill-guided action selection.

This document has been split into four focused plans in `plans/`. See below for the full pipeline and links.

---

## Pipeline overview

```
Pixels (game frame / screenshot / video)
    ↓
(1) Visual Grounding     — VLM parser → structured <state> schema
    ↓
(2) Skill Bank           — trajectory segmentation → skill contracts → retrieval
    ↓
(3) Action Agent         — skill-guided decision making → environment actions
    ↓
(4) Skill Crafter        — compose / generalize / hypothesize new skills
    ↓
    └──→ feeds back into Skill Bank
```

---

## Plan documents

| # | Plan | Scope | File |
|---|------|-------|------|
| 1 | **[Visual Grounding](plans/PLAN-VISUAL-GROUNDING.md)** | VLM parser, canonical schema, grounding heads (heuristic, vision, OmniParser, tool loop), domain adapters (Gym-V, BrowserGym, OSWorld), benchmark evaluation (CLEVR, GQA, ToolVQA, SIV-Bench, Video-Holmes), Qwen3-VL-8B training | `plans/PLAN-VISUAL-GROUNDING.md` |
| 2 | **[Action Agent](plans/PLAN-ACTION-AGENT.md)** | Two-level MDP (outer env + inner reasoning hops), decision loop, GROUND/CHECK/RETRIEVE/CONCLUDE/EXECUTE inner actions, two model backends (GPT-5.4, Qwen3-8B), reward shaping (r_env + r_follow + r_cost) | `plans/PLAN-ACTION-AGENT.md` |
| 3 | **[Skill Bank](plans/PLAN-SKILL-BANK.md)** | 5-stage pipeline (boundary proposal → segmentation → contract learning → bank maintenance → quality eval), GRPO co-evolution (3 LoRA adapters), **reasoning skill discovery** (hop chain templates), phase detection, proto-skill staging, query/select API | `plans/PLAN-SKILL-BANK.md` |
| 4 | **[Skill Crafter](plans/PLAN-SKILL-CRAFTER.md)** | Skill composition (effect chaining + **hop protocol chaining**), cross-domain generalization (schema-slot transfer), **transferable skill families** (4 cross-domain families), novel skill hypothesis, integration with visual grounding tool traces | `plans/PLAN-SKILL-CRAFTER.md` |

---

## Key shared concepts

### Canonical `<state>` schema

All four components share the same structured schema format (defined in PLAN-VISUAL-GROUNDING §3):

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

Multi-hop visual reasoning is reframed as a **long-horizon interaction** problem (see `LONG_HORIZON_REASONING.md`). Each reasoning hop becomes an explicit step in an inner MDP:

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

The agent **learns when to stop reasoning and act**. Skills capture *how to think* (reasoning hop chains), not just *what to do*. GRPO optimizes the full reasoning chain end-to-end. See [Action Agent §5](plans/PLAN-ACTION-AGENT.md) for the full design.

### Model convention

- **Qwen3-8B** — GRPO-trained with LoRA adapters for all pipeline components
- **GPT-5.4** — training-free cold-start and labeling
- **Qwen3-VL-8B** — visual grounding VLM (distilled from GPT-4o labels)

### LoRA adapter layout

| Adapter | Purpose | Plan |
|---------|---------|------|
| `schema_gen` | Screenshot → `<state>` schema | Visual Grounding |
| `hop_select` | Schema + trace → next reasoning action | Action Agent |
| `skill_select` | Schema → which reasoning skill to invoke | Action Agent / Skill Bank |
| `segment` | Trajectory → skill boundary detection | Skill Bank |
| `contract` | Segment → effects contract | Skill Bank |

---

## One-sentence framing

We train a VLM as a visual parser — supervised by the text state that game and web environments already provide for free — that converts pixels into structured summaries for a skill-based agent pipeline, and learns to call environment APIs for information it cannot see.
