# PLAN: Cross-Task Skill Bank for Reasoning and Control

**Scope:** Build and maintain a Skill Bank from long-horizon structured trajectories across **games, web agents, long videos, visual reasoning, and embodied tasks**. The Skill Bank discovers reusable skills, learns their protocols and symbolic contracts, and exposes them to the [Action Agent](PLAN-ACTION-AGENT.md) through retrieval and selection APIs.

**Upstream:** Structured episode trajectories from the Action Agent; structured schemas from [Visual Grounding](PLAN-VISUAL-GROUNDING.md); intermediate reasoning traces; visual grounding outputs; memory retrieval traces; execution outcomes / verification signals. These inputs may come from multiple domains, but are converted into a shared typed structured representation before skill discovery and maintenance.
**Downstream:** Skill guidance consumed by the Action Agent and the Reasoning Agent; skill contracts used for reward shaping; bank curation consumed by GRPO-based training loops.
**Co-evolves with:** [Skill Crafter](PLAN-SKILL-CRAFTER.md) (which composes and creates new skills); [Visual Skills](PLAN-VISUAL-SKILLS.md) (optional grounding strategy layer).

---

## 0. Goal

Build and maintain a **Skill Bank** from long-horizon structured trajectories across **games, web agents, long videos, visual reasoning, and embodied tasks**. The Skill Bank should discover reusable skills, learn their protocols and symbolic contracts, and expose them to the Action Agent through retrieval and selection APIs.

The key design goal is to store **transferable reasoning-and-control skills**, not only environment-specific action motifs. A skill should be reusable across tasks through shared state abstractions, shared inner primitives, and verifiable outcome contracts.

---

## 0.5. Core design principle: skill as a structured-state program

A skill is a **reusable reasoning-and-control program over a typed structured state**.

Instead of defining skills as raw environment-dependent action macros, we define them as:

- a typed applicability condition over structured state
- an executable protocol composed of shared reasoning/control primitives
- a symbolic outcome contract describing expected state changes
- a transfer interface that binds the abstract skill to different domains via adapters

This lets the same high-level skill transfer across:
- games
- web/UI environments
- long-video reasoning settings
- visual reasoning tasks
- embodied / robotics tasks

In this plan, the Skill Bank stores **reasoning templates + contracts**, while environment-specific parsing, grounding, and action realization are delegated to adapters.

---

## 1. Architecture overview

### High-level loop

```
Structured episode trajectories (from any domain)
    ↓
Stage 1: Boundary Proposal       — candidate cut points from signals + state deltas
    ↓
Stage 2: Segmentation            — decode segments + skill labels via preference learning
    ↓
Stage 3: Contract Learning       — learn/verify/refine symbolic contracts per skill
    ↓
Stage 4: Bank Maintenance        — propose → filter (CURATOR) → execute (split/merge/refine/promote)
    ↓
Stage 4.5: Quality Evaluation   — score on effectiveness, transferability, contract validity
    ↓
Query / Select API               — action agent retrieves skills with structured guidance
```

The bank co-evolves with the policy: improved skills improve policy rollouts, and improved rollouts improve the bank.

### Upstream inputs

The Skill Bank consumes:
- structured episode trajectories
- intermediate reasoning traces (inner MDP hop chains)
- visual grounding outputs (entities, relations, evidence)
- memory retrieval traces
- execution outcomes / verification signals

These inputs may come from multiple domains, but are converted into a shared typed structured representation before skill discovery and maintenance.

### Downstream consumers

The Skill Bank serves:
- the Action Agent (skill selection, protocol following, active skill tracking)
- the Reasoning Agent (hop chain templates, evidence strategies)
- the Memory / Retrieval controller (skill-relevant memory queries)
- bank curation and GRPO-based training loops

---

## 1.5. Cross-task transfer objective

The bank should support **cross-task transfer** through three mechanisms:

### 1. Shared state abstraction

Different environments are mapped to a common typed state interface (§3). The canonical `<state>` schema (defined in [Visual Grounding §3](PLAN-VISUAL-GROUNDING.md#3-canonical-schema)) provides the shared representation: entities, attributes, relations, state_flags, targets, uncertainty. All skill preconditions and effects are written over this shared schema.

### 2. Shared inner primitives

Skills are written using reusable reasoning/control primitives from the inner MDP action vocabulary (see [Action Agent §5](PLAN-ACTION-AGENT.md#5-two-level-mdp-long-horizon-reasoning)):

| Primitive | Purpose |
|-----------|---------|
| `GROUND` | Locate / bind entities from visual or structured input |
| `CHECK` | Verify a relation, attribute, or constraint |
| `RETRIEVE` | Query skill bank, memory, or prior observations |
| `CONCLUDE` | Commit an intermediate result or subgoal |
| `ACT` / `EXECUTE` | Emit an environment action (exits inner loop) |
| `VERIFY` | Confirm that expected effects hold after execution |

### 3. Adapter-based binding

Each environment provides an adapter that:
- parses observations into structured state
- grounds entities / relations / evidence
- binds abstract actions to concrete actions
- reports verification signals back to the bank

This means skills transfer through **state + protocol + contract**, not through raw action strings. See [Visual Skills §6](PLAN-VISUAL-SKILLS.md#6-separating-semantics-from-execution) for the full semantic/execution separation and the abstract-operator-to-domain mapping table.

---

## 2. Five-stage bank pipeline

All stages operate on **typed structured trajectories** rather than only task-local action traces.

### Stage 1: Boundary Proposal

Given a long trajectory, propose candidate boundaries for reusable skill segments.

Signals may include:
- abrupt changes in local reward / progress
- predicate flips or state deltas
- phase transitions (see §5)
- action-mode switches (grounding → reasoning → action)
- intention / subgoal changes
- memory handoffs
- evidence handoffs
- grounding failures or uncertainty spikes reported by adapters
- tool usage pattern changes

Not GRPO-wrapped (algorithmic from predicates + signals).

- **Implementation:** `skill_agents/boundary_proposal/`

### Stage 2: Segmentation / skill decoding

Given candidate segments, decode them into known skills or propose `__NEW__` skills.

Segmentation operates over:
- typed state transitions
- reasoning traces (inner MDP hops)
- action traces (outer MDP actions)
- grounding / evidence traces

The decoder may produce:
- action-level skills
- reasoning-level skills (hop chain templates)
- mixed reasoning-and-control skills
- grounding-level skills (multi-step perception strategies, see [Visual Skills §7](PLAN-VISUAL-SKILLS.md#7-grounding-skill-bank))

If no existing skill fits well, assign `__NEW__` and forward to contract learning / bank maintenance.

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

**Compound skill labels:** Phase detector + intention tags → `"endgame:MERGE"`, `"opening:POSITION"`, etc. Same tactic in different phases becomes a distinct skill.

**GRPO:** SEGMENT LoRA wraps `collect_segment_preferences()`; reward = `SegmentationDiagnostics`.

- **Implementation:** `skill_agents/infer_segmentation/`

### Stage 3: Contract Learning

For each candidate skill, infer:
- what preconditions tend to hold (over the typed structured state)
- what effects are consistently produced (world-effects and belief/grounding-effects)
- what evidence or verification signals support those effects
- what parts are cross-domain invariants vs domain-specific bindings

The goal is to learn a compact symbolic contract that can be used for:
- retrieval (effect-based search)
- reward shaping (r_follow from eff_add predicates)
- verification (contract pass rate)
- transfer (shared predicates across domains)

Contracts feed back into Stage 2 via `compat_fn`.

**GRPO:** CONTRACT LoRA wraps `llm_summarize_contract()`; reward = `verify_effects_contract().overall_pass_rate`.

- **Implementation:** `skill_agents/stage3_mvp/`

### Stage 4: Bank Maintenance

**Propose → Filter → Execute** flow:

1. **Propose:** Build `SkillProfile` per skill, propose candidates (refine, merge, split, materialize, promote, retire).
2. **Filter:** CURATOR LoRA — approve/veto/defer per candidate.
3. **Execute:** Approved actions run (refine = weaken + strengthen; merge/split with alias map).

Maintenance decisions should consider:
- protocol similarity
- contract similarity (shared effect predicates)
- signature compatibility (typed slot overlap)
- transferability across domains (domain coverage)
- verification consistency (pass rate stability)
- usage frequency / utility (selection rate by action agent)

New skills enter through **proto-skill staging:** `__NEW__` clusters → materialize → proto-skill → verify → promote.

**GRPO:** CURATOR LoRA wraps `filter_candidates()`; reward = `bank_quality_delta`.

- **Implementation:** `skill_agents/bank_maintenance/`

### Stage 4.5: Quality Evaluation

Evaluate candidate or updated skills on:
- **In-domain effectiveness:** outcome_reward, follow_through, consistency, compactness
- **Cross-domain transferability:** domain coverage, slot coverage, protocol quality
- **Contract validity:** pass rate on holdout instances
- **Verification consistency:** stability across episodes
- **Grounding cost:** tool calls and latency per skill invocation
- **Evidence sufficiency:** whether evidence_required fields are satisfiable
- **Downstream utility:** selection rate and reward improvement for the action agent

Low-quality segments can be dropped before bank maintenance.

- **Implementation:** `skill_agents/quality/`

---

## 3. Unified structured state interface

All tasks should be mapped into a typed structured state representation. The canonical `<state>` schema (defined in [Visual Grounding §3](PLAN-VISUAL-GROUNDING.md#3-canonical-schema)) provides the shared format. Skills are defined over this schema, not over raw observations.

### 3.1. State schema

```yaml
state:
  domain: str                    # gymv | browser | desktop | image_qa | video_qa | embodied
  task: str
  goal: str
  step: int

  entities: list[Entity]         # typed objects with attributes
  attributes: dict[eid, dict]    # per-entity key-value attributes
  relations: list[Relation]      # spatial, functional, temporal, grouping
  events: list[Event]            # state changes, temporal markers
  goals: list[Goal]              # active subgoals and completion status

  state_flags:
    progress: float              # 0–1
    phase: str                   # early | mid | late | domain-specific
    error: str | null
    dialog_open: bool
    input_pending: bool

  targets:
    target: eid | null
    blocker: eid | null
    constraint: str | null
    candidate_set: list[eid]
    history_anchor: eid | null

  uncertainty: dict[eid, str]    # per-entity confidence (high | medium | low)
  memory_refs: list[str]         # references to episodic memory entries
  evidence_refs: list[str]       # references to collected evidence chains
  action_candidates: list[str]   # valid actions in the current state
```

### 3.2. Entity types (cross-domain ontology)

For skills to transfer, domain-specific objects must map into shared types. See [Visual Skills §5](PLAN-VISUAL-SKILLS.md#5-cross-domain-entity-ontology) for the full ontology.

| Ontology type | Purpose | Example domains |
|---------------|---------|-----------------|
| `selectable_entity` | Objects that can be targeted / selected | Buttons, icons, tiles, items, people in video |
| `interactive_entity` | Objects whose state can be changed | Form fields, levers, doors, switches |
| `container_entity` | Objects that contain other objects | Dropdowns, folders, chests, rooms, scenes |
| `textual_anchor` | Text that identifies or locates other entities | Labels, titles, score displays, subtitles |
| `navigable_region` | Areas that can be moved to / focused on | Page sections, desktop areas, map zones, timeline segments |
| `tracked_entity` | Objects that persist across time | Session elements, persistent windows, moving characters |
| `goal_indicator` | Objects that signal goal progress | Success messages, score thresholds, answer evidence |
| `blocking_entity` | Objects that prevent actions | Modals, disabled states, walls, occlusions |

### 3.3. Two kinds of effect predicates

Skills produce two kinds of state changes, both expressed as predicates over the schema:

| Effect type | Changes | Example predicates |
|-------------|---------|-------------------|
| **World-effects** | External environment state | `selected(target)=true`, `opened(container)=true`, `distance(agent,target) decreases` |
| **Belief/grounding-effects** | Internal binding / evidence / confidence state | `binding(target)=resolved`, `confidence(target)≥τ`, `candidate_count=1`, `evidence_collected=true` |

World-effects are the existing `eff_add` / `eff_del` / `eff_event` predicates. Belief/grounding-effects are new — they describe changes to the agent's understanding rather than changes to the environment. Both types use the same contract format.

---

## 4. Skill data model

Each skill has three logical parts:

### 4.1. Protocol store (what the action agent sees)

- `skill_id`, `name`, `strategic_description`, `tags`
- `skill_type`: `"reasoning"` | `"action"` | `"grounding"` | `"mixed"`
- `category`: effect family (see §8)
- `protocol`: steps (using shared inner primitives), preconditions, success_criteria, abort_criteria, expected_duration
- `typed_slots`: slot variables with ontology types (§3.2)
- `confidence`
- Used by `skill_bank_to_text()`, `query_skill_bank()`, and to set `active_skill_plan`

### 4.2. Contract (what the reward system uses)

- `eff_add`: predicates that should become true (world-effects + belief-effects)
- `eff_del`: predicates that should become false
- `eff_event`: event predicates
- `evidence_required`: what evidence must be collected for the effects to be verifiable
- Used for segmentation verification, reward shaping (r_follow), and stage 2↔3 feedback

**The agent plans from protocols and is rewarded for making progress on the contract's eff_add predicates.**

### 4.3. Transfer interface (what enables cross-domain reuse)

- `slot_bindings`: maps typed slots to domain-specific schema fields (see [Visual Skills §3b](PLAN-VISUAL-SKILLS.md#3b-typed-slot-variables))
- `abstract_predicates`: parameterised eff_add/eff_del using `$slot` placeholders, with per-domain instantiations
- `domain_adapters`: per-domain execution realizations (how abstract operators map to concrete tools/actions)
- `transfer_hints`: domains where this skill has been validated
- `reasoning_protocol`: hop chain template using inner MDP actions

### 4.4. Reasoning skills (inner MDP hop chain templates)

Under the two-level MDP (see [Action Agent §5](PLAN-ACTION-AGENT.md#5-two-level-mdp-long-horizon-reasoning)), skills capture *how to think*, not just *what to do*. Each reasoning skill is a multi-step policy over inner MDP actions (GROUND, CHECK, RETRIEVE, CONCLUDE, EXECUTE).

**Example reasoning skill:**

```
skill: constraint_satisfaction
skill_type: reasoning
category: verification
trigger: state_flags.error != null OR target.blocker != null
slots:
  blocker: blocking_entity
  constraint: str
protocol:
  hop1: GROUND(blocker entity)
  hop2: CHECK(what constraint is violated)
  hop3: RETRIEVE(similar past resolution)
  hop4: CONCLUDE(subgoal = resolve blocker first)
  hop5: EXECUTE(action addressing blocker)
effects:
  eff_add: [blocker_resolved, constraint_satisfied]
  eff_del: [blocker_active]
domain_adapters:
  game: EXECUTE → game-specific action
  browser: EXECUTE → click/type/navigate
  video: EXECUTE → temporal navigation + evidence collection
```

**Example cross-domain skill:**

```
skill: locate_filter_select
skill_type: mixed
category: acquisition
trigger: candidate_set is non-empty AND target is unresolved
slots:
  target: selectable_entity
  candidate_set: list[selectable_entity]
  filter_criterion: str
protocol:
  hop1: GROUND(candidate_set)
  hop2: CHECK(filter_criterion against each candidate)
  hop3: CONCLUDE(best candidate)
  hop4: EXECUTE(select best)
effects:
  eff_add: [target_selected, candidate_resolved]
  eff_del: [target_unresolved]
domain_adapters:
  game: candidates = legal moves → best legal move
  browser: candidates = UI elements → relevant control
  image_qa: candidates = detected objects → answer target
  video_qa: candidates = temporal moments → key frame
  embodied: candidates = reachable objects → grasp target
```

**Mapping COS-PLAY components to the unified system:**

| COS-PLAY Component | Role in Cross-Task Skills |
|---------------------|--------------------------|
| Skill Bank | Stores reusable **hop chain templates** with typed slots and domain adapters |
| Skill protocols (trigger, steps, abort/success) | Become **reasoning protocols** — when to ground, when to check, when to conclude |
| GRPO training | Optimizes the full reasoning chain end-to-end |
| Co-evolution loop | Discovers new reasoning patterns from trajectories across all domains |
| RAG + embeddings | Retrieves relevant reasoning templates for current visual/structured state |

**Trajectory format:** Under the two-level MDP, episode trajectories include both inner hops and outer actions:

```
(schema_0, GROUND, schema_0') → (schema_0', CHECK, schema_0'') → (schema_0'', EXECUTE(click), schema_1) → ...
```

These trajectories are segmentable by Stage 1–2 — reasoning hop chains become discoverable skills alongside action-level skills.

---

## 5. Phase and context detection

Per-step intention tags capture tactical intent; the phase detector adds strategic context. Phase detection generalizes across domains — not just games.

```
raw tag:      MERGE          MERGE          MERGE
phase:        opening        midgame        endgame
compound:     opening:MERGE  midgame:MERGE  endgame:MERGE   ← 3 distinct skills
```

### Domain-specific extractors

| Domain | Environment | State Feature | Phases |
|--------|------------|--------------|--------|
| Game | 2048 | Board occupancy + highest tile | opening, midgame, endgame |
| Game | Tetris | Board fill ratio | opening, midgame, endgame |
| Game | Super Mario | Mario x-position | early_level, mid_level, late_level |
| Game | Avalon | Round signals | team_building, quest, endgame |
| Game | Diplomacy | Turn/season signals | opening, orders, retreat, adjustment |
| Game | Candy Crush | Temporal position | early, mid, late |
| Browser | WebArena | Task completion % + page depth | exploration, form_filling, verification |
| Browser | MiniWoB++ | Element count + interaction history | identification, interaction, confirmation |
| Desktop | OSWorld | Window/app state | navigation, configuration, verification |
| Image QA | CLEVR/GQA | Evidence chain length | grounding, reasoning, answering |
| Video QA | SIV-Bench | Timeline position + evidence count | scanning, focusing, concluding |
| Video QA | Video-Holmes | Clue chain length + scene coverage | exploration, investigation, synthesis |

**Impact:** Phase detection makes the same abstract reasoning pattern (e.g., "filter candidates") produce distinct skills when applied in different strategic contexts. 2048 goes from 1 skill (MERGE) to 9 compound skills. Browser tasks gain phase-aware exploration vs. verification skills.

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
    domain="game",
    top_k=3,
)
# Each result has: skill_id, skill_type, relevance, applicability, confidence,
# matched_effects, missing_effects, protocol, micro_plan, failure_modes,
# domain_adapter (if available for current domain)
```

### Effect-based retrieval

```python
results = agent.query_by_effects(
    desired_add={"team_proposed", "quest_active"},
    desired_del={"waiting_for_leader"},
    top_k=3,
)
```

### Cross-domain transfer retrieval

```python
results = agent.query_transferable(
    source_domain="game",
    target_domain="browser",
    effect_family="acquisition",
    top_k=3,
)
# Returns skills with validated adapters for the target domain,
# or skills with high structural similarity that need adapter creation
```

### Scoring

Each candidate skill is scored on three axes:

| Component | Weight | Source |
|-----------|--------|--------|
| Retrieval relevance | 40% | RAG embedding cosine similarity + keyword Jaccard |
| Execution applicability | 35% | Effect compatibility against current state predicates + slot coverage check |
| Historical pass rate | 25% | Success rate from past executions (domain-aware when available) |

---

## 7. GRPO co-evolution

### Asymmetric co-evolution framework

The Skill Bank sits at the **medium timescale** within the three-agent co-evolution framework (see [Action Agent §6](PLAN-ACTION-AGENT.md#6-co-evolution--grpo-decomposition)). Its operational components (retrieval, scoring, tracking) update more frequently than the synthesis-reflection agent but less frequently than the actor.

**Co-evolution loop for the Skill Bank:**

```
Actor rolls out with current bank (fast timescale)
        ↓
Collect experience trajectories + skill-use statistics (from all domains)
        ↓
Skill Bank operational update (medium timescale):
  • Segmentation of new trajectories (typed structured)
  • Contract learning/refinement (world + belief effects)
  • Bank maintenance (propose → filter → execute)
  • Quality evaluation (including cross-domain transferability)
        ↓
Synthesis-reflection agent proposes (slow timescale):
  • New skills from failure reflection
  • Compositions from effect chaining
  • Cross-domain transfers (new adapters for existing skills)
  • "New skill vs. new adapter" decisions
        ↓
Acceptance gate (before anything enters the bank):
  1. Contract completeness checks against schema
  2. Retrieval compatibility with current bank entries
  3. Replay or held-out verification on stored trajectories
  4. Non-regression filtering (must not lower pass rate on prior successes)
  5. Cross-domain consistency (shared predicates must hold across validated domains)
        ↓
Accepted artifacts enter bank → actor uses updated bank
```

**Update cadence:** 5–10 actor GRPO update cycles, then 1 offline skill-bank update cycle. Skill-bank refinement runs after the actor's update converges enough that traces are meaningful. Bank updates should not run every iteration — that creates a moving target the actor chases too aggressively.

### LoRA adapters (8B, trained)

Three LoRA adapters on Qwen3-8B, trained during co-evolution:

| Adapter | Stage | Wrapped function | Reward |
|---------|-------|------------------|--------|
| **CONTRACT** (P0) | 3 | `llm_summarize_contract()` | `verify_effects_contract().overall_pass_rate` |
| **CURATOR** (P1) | 4 | `filter_candidates()` | `bank_quality_delta` |
| **SEGMENT** (P1) | 2 | `collect_segment_preferences()` | `SegmentationDiagnostics` |

These adapters belong to the **skill-use / operational agent** (Agent 2 in the [three-agent split](PLAN-ACTION-AGENT.md#three-agent-role-split)). They handle the sequential bank-management decisions that benefit from GRPO: segmentation, contract quality, and curation. Simple retrieval, applicability scoring, pass-rate lookup, and `_SkillTracker` lifecycle logic remain algorithmic — GRPO is not applied to these.

**Additional skill-use GRPO targets** (selective, for sequential decisions only):

| Decision | Policy output | Reward |
|---|---|---|
| Continue / switch active skill | binary | Downstream reward improvement, reduced stall |
| Accept / reject candidate segment as skill instance | binary | Contract satisfaction, skill reuse rate |
| Merge / split / retire / keep skill | categorical | Bank compactness regularization, downstream actor improvement |
| Protocol revision choice from candidate set | categorical | Follow-through rate, skill pass rate delta |
| Accept / reject cross-domain transfer proposal | binary | Transfer success rate, adapter validation pass rate |

### Two-phase architecture

1. **Phase 1 (Rollout):** Pipeline calls LLM as normal → GRPO wrapper generates G samples → reward per sample → best returned to pipeline + all stored in buffer.
2. **Phase 2 (Training):** Read buffer → recompute log_probs → group-normalized rewards → GRPO policy gradient → update LoRA adapter → clear buffer.

### Synthesis-reflection outputs (from frozen 32B/72B)

The Skill Bank receives candidate artifacts from the synthesis-reflection agent ([Skill Crafter](PLAN-SKILL-CRAFTER.md)):
- New skill proposals (from Composer, Hypothesizer, Generalizer)
- Revised protocols (from Failure Reflector recovery actions)
- Contract patches (precondition strengthening, effect updates)
- Cross-domain transfer mappings (new adapters for existing abstract skills)
- "New skill vs. new adapter" decisions (see [Visual Skills §11](PLAN-VISUAL-SKILLS.md#11-how-the-synthesis-reflection-agent-helps-with-transfer))

All of these are treated as **candidate proposals**, not ground truth. They enter the bank only after passing the acceptance gate. The Skill Bank does not blindly trust the 32B/72B — it verifies, replays, and gates every output. See [Skill Crafter §2](PLAN-SKILL-CRAFTER.md#2-architecture) for the frozen teacher design rationale.

---

## 8. Effect families and skill hierarchy

### Effect families

Skills organized by the kind of state change they create — the primary axis for cross-domain transfer. See [Visual Skills §4](PLAN-VISUAL-SKILLS.md#4-effect-families) for the full taxonomy.

| Family | State change | Game example | Browser example | Video example | Embodied example |
|--------|-------------|--------------|-----------------|---------------|------------------|
| **Acquisition** | Bring target into focus / possession | Select best move | Click target element | Lock onto tracked person | Grasp object |
| **Navigation** | Move attention or agent to region | Move to area | Scroll / navigate to section | Seek to temporal region | Move to location |
| **Inspection** | Reveal hidden information | Observe board state | Read tooltip / expand details | Sample frames for evidence | Look at / examine object |
| **Manipulation** | Change target state | Merge tiles | Fill form / toggle switch | — | Push / rotate / assemble |
| **Verification** | Test whether desired state holds | Check goal condition | Verify form submission | Confirm answer evidence | Check grasp stability |
| **Disambiguation** | Resolve multiple candidates | Choose between moves | Select from similar controls | Identify correct person | Choose correct object |
| **Tracking** | Maintain entity identity over time | Track moving objects | Maintain session state | Follow person across cuts | Track moving target |
| **Recovery** | Restore after failure / loss | Undo / reposition | Go back / retry | Re-scan for lost entity | Re-grasp / reposition |

### Three-layer skill hierarchy

The bank organizes into three layers to support both transfer and domain-specific robustness. See [Visual Skills §8](PLAN-VISUAL-SKILLS.md#8-three-layer-skill-bank-hierarchy) for the full design.

**Layer 1: Abstract transferable skills** — shared across all domains, defined by semantic contracts only.

```
acquire_target, inspect_region, navigate_to_goal, verify_condition,
disambiguate_candidate, track_entity, open_reveal_interact,
recover_after_loss, collect_evidence, localize_temporal_event,
constraint_satisfaction, blocker_prerequisite_replan,
history_hidden_state_act, compare_under_constraint
```

**Layer 2: Domain adapters** — per-domain execution realizations.

```
game.acquire_target, browser.acquire_target, video.acquire_target, embodied.acquire_target
```

**Layer 3: Environment-specific tactics** — concrete low-level wrappers.

```
game.acquire_target.walk_then_interact
browser.acquire_target.button_click
video.acquire_target.find_then_track
embodied.acquire_target.reach_and_grasp
```

### How the actor uses the hierarchy

The actor does not directly choose from a flat bank of environment-specific skills. The decision process is three-step:

1. **What effect is needed?** → select abstract skill (Layer 1) based on current state + desired state change
2. **Which entities fill the slots?** → bind typed slots from current grounded state
3. **Which adapter executes it here?** → select domain adapter (Layer 2), fall back to specific tactic (Layer 3)

---

## 9. Transferable skill extraction

Skills discovered in one domain can transfer to others. The extraction pipeline analyses per-domain banks and produces domain-agnostic templates.

### Skill template format (`TransferableSkill`)

Each transferable skill wraps the concrete `Skill` schema with cross-domain abstractions:

| Component | Purpose |
|-----------|---------|
| **SlotBinding** | Maps domain predicates to shared ontology slots (`target`, `blocker`, `constraint`, `candidate_set`, `history_anchor`) |
| **AbstractPredicate** | Parameterised eff_add/eff_del using `$slot` placeholders, with per-domain instantiations |
| **ReasoningProtocol** | Hop chain template using inner MDP actions (GROUND → CHECK → RETRIEVE → CONCLUDE → EXECUTE) |
| **DomainAdapters** | Per-domain execution realizations (tool calls, action bindings) |

### Transferable skill families

| Family | Hop chain | Game | Browser | Visual QA | Video | Embodied |
|--------|-----------|------|---------|-----------|-------|----------|
| **locate_filter_select** | GROUND → CHECK → CONCLUDE → EXECUTE | Candidates → best legal move | UI elements → relevant control | Objects → attributes → answer | Frames → key moment | Objects → grasp target |
| **blocker_prerequisite_replan** | GROUND → CHECK → RETRIEVE → CONCLUDE → EXECUTE | Deadlock → resolve prerequisite | Disabled control → fill missing field | Weak evidence → gather anchor | Missing context → scan earlier | Obstacle → clear path |
| **history_hidden_state_act** | RETRIEVE → CHECK → GROUND → CONCLUDE → EXECUTE | Dialogue → infer alliance → act | Prior pages → session state → next step | Prior frames → disambiguate | Earlier scenes → identify person | Prior interactions → predict affordance |
| **compare_under_constraint** | GROUND → CHECK → CHECK → CONCLUDE → EXECUTE | Move preserving structure | Path minimising risk | Candidate consistent with constraints | Moment consistent with timeline | Action within force/reach limits |
| **disambiguate_target** | GROUND → RETRIEVE → CHECK → CONCLUDE | Multiple game objects → correct one | Similar UI elements → correct control | Ambiguous objects → correct match | Multiple people → correct track | Cluttered objects → correct grasp |
| **collect_evidence_chain** | GROUND → CHECK → GROUND → CHECK → CONCLUDE | Multi-step board analysis | Multi-page form verification | Multi-hop visual reasoning | Multi-scene clue chaining | Multi-step task verification |

### Extraction pipeline

```
Per-domain skill banks
    ↓
Stage A: Predicate normalisation     — map domain predicates to ontology slots via regex + LLM
    ↓
Stage B: Structural clustering       — agglomerative clustering by role signatures (cross-domain)
    ↓
Stage C: Template abstraction        — produce TransferableSkill with protocol + slot bindings + adapters
    ↓
Stage D: Transferability scoring     — domain coverage × slot coverage × protocol quality × evidence
    ↓
Stage E: Export                      — transferable_skills.jsonl + transfer_index.json + families
```

**Extension to grounding skills:** The extraction pipeline also applies to visual grounding strategies — multi-step grounding patterns (disambiguation, target recovery, evidence collection) can be extracted as transferable grounding skills with belief/binding-effect contracts. See [Visual Skills](PLAN-VISUAL-SKILLS.md) for the full grounding skill format and how grounding segments integrate into Stages A–D.

### Usage

```python
# From the SkillBankAgent
templates = agent.extract_transferable_skills(
    other_banks={"tetris": bank_tetris, "webarena": bank_webarena, "clevr": bank_clevr},
    output_dir="output/transferable",
)

# Import a template into a new domain
agent.import_transferable_skill(templates[0], slot_map={
    "target": "focused_element",
    "blocker": "validation_error",
}, domain="browser")

# Or standalone
from skill_agents.extract_transferable import extract_transferable_skills
templates = extract_transferable_skills(
    banks={"2048": bank_2048, "tetris": bank_tetris, "webarena": bank_webarena},
    output_dir="output/transferable",
)
```

---

## 10. Tool-call reward (agentic RL)

For QUERY_SKILL, QUERY_MEM, CALL_SKILL the reward includes:

**r_total = w_relevance × r_relevance + w_utility × r_utility**

- **r_relevance**: RAG retrieval score × relevance_scale
- **r_utility**: fraction of eff_add predicates satisfied in outcome × utility_per_predicate, plus completion bonus

---

## 11. Cold-start I/O recording

Every LLM call that will be replaced by GRPO records its prompt/response:
- `teacher_io_coldstart.jsonl` — Stage 2 teacher calls
- `coldstart_io_all.jsonl` — all other stages

Used for: SFT cold-start for Qwen3-8B, reference outputs for GRPO reward comparison.

---

## 12. Core components

| Component | Purpose |
|-----------|---------|
| **SkillBankAgent** | Full pipeline orchestrator: ingest → segment → learn → maintain → query |
| **SkillQueryEngine** | Rich retrieval + selection: keyword, effect-based, state-aware, and cross-domain search (RAG + applicability + pass rate) |
| **SkillBankMVP** | Persistent JSONL storage for skill contracts, transfer interfaces, and verification reports |
| **NewPoolManager** | Tracking of `__NEW__` segments; clusters graduate via proto-skill staging |
| **ProtoSkillManager** | Staging area: materialize → verify → promote to real bank |
| **TransferManager** | Cross-domain transfer: extract → score → import → validate adapters |

---

## 13. Implementation

| Directory | Purpose |
|-----------|---------|
| `skill_agents/pipeline.py` | SkillBankAgent orchestrator |
| `skill_agents/query.py` | SkillQueryEngine + SkillSelectionResult |
| `skill_agents/skill_template.py` | TransferableSkill, SlotBinding, ReasoningProtocol, AbstractPredicate, DomainAdapter |
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

## 14. TODO

| Task | Priority | Status |
|------|----------|--------|
| Transferable skill template + extraction pipeline | P0 | **Done** |
| Acceptance gate pipeline (contract check, replay verification, non-regression filter) | P0 | Not started |
| Implement timescale separation for bank update cadence (medium timescale) | P0 | Not started |
| Extend segmentation to inner MDP hop traces | P1 | Not started |
| Reasoning skill discovery (hop chain templates) | P1 | Not started |
| Inner hop reward signal for GRPO (hop quality + outer reward) | P1 | Not started |
| Skill-use GRPO: continue/switch, accept/reject, merge/split decisions | P1 | Not started |
| Integration with synthesis-reflection agent outputs (gated candidate ingestion) | P1 | Not started |
| Unified structured state interface (§3) across all domains | P1 | Not started |
| Cross-domain entity ontology mapping (§3.2, heuristic + LLM) | P1 | Not started |
| Belief/grounding-effect contracts (§3.3) | P1 | Not started |
| Typed slot variables in skill data model (§4.1, §4.3) | P1 | Not started |
| Domain adapter registry (abstract operator → concrete tool call) | P1 | Not started |
| Effect family taxonomy and skill hierarchy (§8) | P1 | Not started |
| Extend skill families from 4 to 6 (§9, +disambiguate_target, +collect_evidence_chain) | P1 | Not started |
| Phase detection for non-game domains (browser, video, embodied) | P1 | Not started |
| Cross-domain transfer retrieval API (§6) | P2 | Not started |
| Cross-domain transfer acceptance GRPO target (§7) | P2 | Not started |
| Reasoning protocol contracts (trigger → hops → EXECUTE) | P2 | Not started |
| RAG retrieval over reasoning templates (not just action skills) | P2 | Not started |
| LLM-based slot binding (replace regex heuristics with LLM inference) | P2 | Not started |
| Cross-domain transfer evaluation harness (transfer success rate metric) | P2 | Not started |
| TransferManager component for cross-domain skill import/export | P2 | Not started |
| Embodied task adapter (observation → schema, abstract action → motor command) | P3 | Not started |
