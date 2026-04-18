# PLAN: Skill Bank Agent

**Scope:** Build and maintain a Skill Bank from long-horizon game trajectories — segment trajectories into skills, learn symbolic contracts (effects), serve queries for the [Action Agent](PLAN-ACTION-AGENT.md), and co-evolve with GRPO training. Skills are the reusable strategic building blocks that bridge perception (from [Visual Grounding](PLAN-VISUAL-GROUNDING.md)) to action.

**Upstream:** Episode trajectories from the Action Agent; structured schemas from Visual Grounding.
**Downstream:** Skill guidance consumed by the Action Agent; skill contracts used for reward shaping.
**Co-evolves with:** [Skill Crafter](PLAN-SKILL-CRAFTER.md) (which composes and creates new skills).

---

## 1. Architecture overview

The Skill Bank pipeline has five stages that form a closed loop:

```
Episode trajectories
    ↓
Stage 1: Boundary Proposal       — candidate cut points from signals + phase transitions
    ↓
Stage 2: Segmentation            — decode segments + compound skill labels via preference learning
    ↓
Stage 3: Contract Learning       — learn/verify/refine effects contracts per skill
    ↓
Stage 4: Bank Maintenance         — propose → filter (CURATOR) → execute (split/merge/refine/promote)
    ↓
Stage 4.5: Quality Evaluation    — score sub-episodes on outcome, follow-through, consistency
    ↓
Query / Select API               — decision agent retrieves skills with structured guidance
```

---

## 2. Core components

| Component | Purpose |
|-----------|---------|
| **SkillBankAgent** | Full pipeline orchestrator: ingest → segment → learn → maintain → query |
| **SkillQueryEngine** | Rich retrieval + selection: keyword, effect-based, and state-aware search (RAG + applicability + pass rate) |
| **SkillBankMVP** | Persistent JSONL storage for skill contracts and verification reports |
| **NewPoolManager** | Tracking of `__NEW__` segments; clusters graduate via proto-skill staging |
| **ProtoSkillManager** | Staging area: materialize → verify → promote to real bank |

---

## 3. Data model

Each skill has two logical parts:

### Protocol store (what the action agent sees)

- `name`, `strategic_description`, `tags`
- `protocol`: steps, preconditions, success_criteria, abort_criteria, expected_duration
- `confidence`
- Used by `skill_bank_to_text()`, `query_skill_bank()`, and to set `active_skill_plan`

### Contract (what the reward system uses)

- `eff_add`: predicates that should become true
- `eff_del`: predicates that should become false
- `eff_event`: event predicates
- Used for segmentation verification, reward shaping (r_follow), and stage 2↔3 feedback

**The agent plans from protocols and is rewarded for making progress on the contract's eff_add predicates.**

### Reasoning skills (from long-horizon reframe)

Under the two-level MDP (see [Action Agent §5](PLAN-ACTION-AGENT.md#5-two-level-mdp-long-horizon-reasoning)), skills capture *how to think*, not just *what to do*. Each reasoning skill is a multi-step policy over inner MDP actions (GROUND, CHECK, RETRIEVE, CONCLUDE, EXECUTE).

**Example reasoning skill:**

```
skill: constraint_satisfaction
trigger: state_flags.error != null OR target.blocker != null
protocol:
  hop1: GROUND(blocker entity)
  hop2: CHECK(what constraint is violated)
  hop3: RETRIEVE(similar past resolution)
  hop4: CONCLUDE(subgoal = resolve blocker first)
  hop5: EXECUTE(action addressing blocker)
```

**Mapping COS-PLAY components to the unified system:**

| COS-PLAY Component | Role in Reasoning Skills |
|---------------------|--------------------------|
| Skill Bank | Stores reusable **hop chain templates** (not just action patterns) |
| Skill protocols (trigger, steps, abort/success) | Become **reasoning protocols** — when to ground, when to check, when to conclude |
| GRPO training | Optimizes the full reasoning chain end-to-end |
| Co-evolution loop | Discovers new reasoning patterns from trajectories |
| RAG + embeddings | Retrieves relevant reasoning templates for current visual state |

**Trajectory format:** Under the two-level MDP, episode trajectories include both inner hops and outer actions:

```
(schema_0, GROUND, schema_0') → (schema_0', CHECK, schema_0'') → (schema_0'', EXECUTE(click), schema_1) → ...
```

These trajectories are segmentable by Stage 1–2 — reasoning hop chains become discoverable skills alongside action-level skills.

---

## 4. Stage details

### Stage 1: Boundary Proposal

Extract signals from trajectories → propose candidate cut points **C**. Phase transitions are injected as boundary events. Not GRPO-wrapped (algorithmic from predicates + signals).

- **Implementation:** `skill_agents/boundary_proposal/`

### Stage 2: Segmentation

Decode over **C** with preference-learned scorer → segments + compound skill labels (including `__NEW__`).

**Scoring formula (6 terms):**

```
Score(i, j, k | k_prev) =
    1.0 * behavior_fit       [LLM preferences — Bradley-Terry]
  + 2.0 * intention_fit      [per-step phase:tag compound labels]
  + 0.3 * duration_prior     [Gaussian]
  + 1.0 * transition_prior   [LLM preferences — Bradley-Terry]
  + 0.0 * contract_compat    [Stage 3 feedback, off by default]
  + 0.5 * boundary_preference
```

**Compound skill labels:** Phase detector + intention tags → `"endgame:MERGE"`, `"opening:POSITION"`, etc. Same tactic in different game phases becomes a distinct skill.

**GRPO:** SEGMENT LoRA wraps `collect_segment_preferences()`; reward = `SegmentationDiagnostics`.

- **Implementation:** `skill_agents/infer_segmentation/`

### Stage 3: Contract Learning

For each non-NEW skill: learn effects contract, verify against holdout instances, refine. Contracts feed back into Stage 2 via `compat_fn`.

**GRPO:** CONTRACT LoRA wraps `llm_summarize_contract()`; reward = `verify_effects_contract().overall_pass_rate`.

- **Implementation:** `skill_agents/stage3_mvp/`

### Stage 4: Bank Maintenance

**Propose → Filter → Execute** flow:

1. **Propose:** Build `SkillProfile` per skill, propose candidates (refine, merge, split, materialize, promote).
2. **Filter:** CURATOR LoRA — approve/veto/defer per candidate.
3. **Execute:** Approved actions run (refine = weaken + strengthen; merge/split with alias map).

New skills enter through **proto-skill staging:** `__NEW__` clusters → materialize → proto-skill → verify → promote.

**GRPO:** CURATOR LoRA wraps `filter_candidates()`; reward = `bank_quality_delta`.

- **Implementation:** `skill_agents/bank_maintenance/`

### Stage 4.5: Quality Evaluation

Score sub-episodes on: outcome_reward, follow_through, consistency, compactness. Low-quality segments can be dropped before bank maintenance.

- **Implementation:** `skill_agents/quality/`

---

## 5. Phase detection preprocessor

Per-step intention tags capture tactical intent; the phase detector adds strategic context:

```
raw tag:      MERGE          MERGE          MERGE
phase:        opening        midgame        endgame
compound:     opening:MERGE  midgame:MERGE  endgame:MERGE   ← 3 distinct skills
```

### Game-specific extractors

| Game | State Feature | Phases |
|------|--------------|--------|
| 2048 | Board occupancy + highest tile | opening, midgame, endgame |
| Tetris | Board fill ratio | opening, midgame, endgame |
| Super Mario | Mario x-position | early_level, mid_level, late_level |
| Avalon | Round signals | team_building, quest, endgame |
| Diplomacy | Turn/season signals | opening, orders, retreat, adjustment |
| Candy Crush | Temporal position | early, mid, late |

**Impact:** 2048 goes from 1 skill (MERGE) to 9 compound skills. Super Mario from 1 to 4.

---

## 6. Query / Select API

### Simple retrieval

```python
results = agent.query_skill("propose team and vote on quest", top_k=3)
```

### Rich skill selection (preferred for action agent)

```python
results = agent.select_skill(
    query="propose team for quest",
    current_state={"is_leader": True, "quest_round": 2},
    top_k=3,
)
# Each result has: skill_id, relevance, applicability, confidence,
# matched_effects, missing_effects, protocol, micro_plan, failure_modes
```

### Effect-based retrieval

```python
results = agent.query_by_effects(
    desired_add={"team_proposed", "quest_active"},
    desired_del={"waiting_for_leader"},
    top_k=3,
)
```

---

## 7. GRPO co-evolution

Three LoRA adapters on Qwen3-8B, trained during co-evolution:

| Adapter | Stage | Wrapped function | Reward |
|---------|-------|------------------|--------|
| **CONTRACT** (P0) | 3 | `llm_summarize_contract()` | `verify_effects_contract().overall_pass_rate` |
| **CURATOR** (P1) | 4 | `filter_candidates()` | `bank_quality_delta` |
| **SEGMENT** (P1) | 2 | `collect_segment_preferences()` | `SegmentationDiagnostics` |

**Two-phase architecture:**
1. **Phase 1 (Rollout):** Pipeline calls LLM as normal → GRPO wrapper generates G samples → reward per sample → best returned to pipeline + all stored in buffer.
2. **Phase 2 (Training):** Read buffer → recompute log_probs → group-normalized rewards → GRPO policy gradient → update LoRA adapter → clear buffer.

---

## 8. Tool-call reward (agentic RL)

For QUERY_SKILL, QUERY_MEM, CALL_SKILL the reward includes:

**r_total = w_relevance × r_relevance + w_utility × r_utility**

- **r_relevance**: RAG retrieval score × relevance_scale
- **r_utility**: fraction of eff_add predicates satisfied in outcome × utility_per_predicate, plus completion bonus

---

## 9. Cold-start I/O recording

Every LLM call that will be replaced by GRPO records its prompt/response:
- `teacher_io_coldstart.jsonl` — Stage 2 teacher calls
- `coldstart_io_all.jsonl` — all other stages

Used for: SFT cold-start for Qwen3-8B, reference outputs for GRPO reward comparison.

---

## 10. Transferable skill extraction

Skills discovered in one game/domain can transfer to others.  The extraction pipeline analyses per-game banks and produces domain-agnostic templates.

### Skill template format (`TransferableSkill`)

Each transferable skill wraps the concrete `Skill` schema with three cross-domain abstractions:

| Component | Purpose |
|-----------|---------|
| **SlotBinding** | Maps domain predicates to shared schema slots (`target`, `blocker`, `constraint`, `candidate_set`, `history_anchor`) |
| **AbstractPredicate** | Parameterised eff_add/eff_del using `$slot` placeholders, with per-domain instantiations |
| **ReasoningProtocol** | Hop chain template using inner MDP actions (GROUND → CHECK → RETRIEVE → CONCLUDE → EXECUTE) |

### Four transferable skill families

| Family | Hop chain | Game | Browser | Visual QA |
|--------|-----------|------|---------|-----------|
| **locate_filter_select** | GROUND → CHECK → CONCLUDE → EXECUTE | Candidates → best legal move | UI elements → relevant control | Objects → attributes → answer |
| **blocker_prerequisite_replan** | GROUND → CHECK → RETRIEVE → CONCLUDE → EXECUTE | Deadlock → resolve prerequisite | Disabled control → fill missing field | Weak evidence → gather anchor |
| **history_hidden_state_act** | RETRIEVE → CHECK → GROUND → CONCLUDE → EXECUTE | Dialogue → infer alliance → act | Prior pages → session state → next step | Prior frames → disambiguate |
| **compare_under_constraint** | GROUND → CHECK → CHECK → CONCLUDE → EXECUTE | Move preserving structure | Path minimising risk | Candidate consistent with constraints |

### Extraction pipeline

```
Per-game skill banks
    ↓
Stage A: Predicate normalisation     — map game predicates to slots via regex heuristics
    ↓
Stage B: Structural clustering       — agglomerative clustering by role signatures (cross-game)
    ↓
Stage C: Template abstraction        — produce TransferableSkill with protocol + slot bindings
    ↓
Stage D: Transferability scoring     — domain coverage × slot coverage × protocol quality × evidence
    ↓
Stage E: Export                      — transferable_skills.jsonl + transfer_index.json + families
```

### Usage

```python
# From the SkillBankAgent
templates = agent.extract_transferable_skills(
    other_banks={"tetris": bank_tetris, "avalon": bank_avalon},
    output_dir="output/transferable",
)

# Import a template into a new game
agent.import_transferable_skill(templates[0], slot_map={
    "target": "focused_element",
    "blocker": "validation_error",
})

# Or standalone
from skill_agents.extract_transferable import extract_transferable_skills
templates = extract_transferable_skills(
    banks={"2048": bank_2048, "tetris": bank_tetris},
    output_dir="output/transferable",
)
```

---

## 11. Implementation

| Directory | Purpose |
|-----------|---------|
| `skill_agents/pipeline.py` | SkillBankAgent orchestrator |
| `skill_agents/query.py` | SkillQueryEngine + SkillSelectionResult |
| `skill_agents/skill_template.py` | TransferableSkill, SlotBinding, ReasoningProtocol, AbstractPredicate |
| `skill_agents/extract_transferable.py` | Cross-domain extraction pipeline (normalise → cluster → abstract → score → export) |
| `skill_agents/boundary_proposal/` | Stage 1 |
| `skill_agents/infer_segmentation/` | Stage 2 |
| `skill_agents/stage3_mvp/` | Stage 3 |
| `skill_agents/bank_maintenance/` | Stage 4 |
| `skill_agents/quality/` | Stage 4.5 |
| `skill_agents/skill_bank/` | Persistent storage + NEW pool |
| `skill_agents/grpo/` | GRPO infrastructure |
| `skill_agents/lora/` | Multi-LoRA model |
| `skill_agents/extract_skillbank/` | Extraction scripts |
| `skill_agents/tool_call_reward.py` | Agentic RL reward |
| `skill_agents/coldstart_io.py` | Cold-start I/O recording |

---

## 12. TODO

| Task | Priority | Status |
|------|----------|--------|
| Transferable skill template + extraction pipeline | P0 | **Done** |
| Extend segmentation to inner MDP hop traces | P1 | Not started |
| Reasoning skill discovery (hop chain templates) | P1 | Not started |
| Inner hop reward signal for GRPO (hop quality + outer reward) | P1 | Not started |
| Reasoning protocol contracts (trigger → hops → EXECUTE) | P2 | Not started |
| RAG retrieval over reasoning templates (not just action skills) | P2 | Not started |
| LLM-based slot binding (replace regex heuristics with LLM inference) | P2 | Not started |
| Cross-domain transfer evaluation harness (transfer success rate metric) | P2 | Not started |
