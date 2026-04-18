# PLAN: Skill Crafter Agent

**Scope:** Compose, create, and refine new skills from existing Skill Bank primitives. The Skill Crafter is the creative layer that discovers higher-order strategies by combining existing skills, generalizing across games/domains, and proposing novel skill hypotheses that the [Skill Bank](PLAN-SKILL-BANK.md) can test and adopt.

**Upstream:** Existing skill bank (contracts, protocols, execution traces); structured schemas from [Visual Grounding](PLAN-VISUAL-GROUNDING.md); episode trajectories from [Action Agent](PLAN-ACTION-AGENT.md).
**Downstream:** New/refined skills injected into the Skill Bank; cross-domain skill transfer proposals.

---

## 1. Motivation

The Skill Bank (Stage 1–4) discovers skills *bottom-up* from observed trajectories — it segments what the agent actually did. But some valuable skills are never observed because:

1. **Composition gap:** The agent never chains two skills in the right order (e.g., "build corner" in 2048 = `POSITION` → `MERGE` → `POSITION`).
2. **Transfer gap:** A skill discovered in one game (e.g., "clear bottom row" in Tetris) has a structural analogue in another game (e.g., "clear bottom boxes" in Sokoban) but the surface-level observations differ.
3. **Hypothesis gap:** Some strategies are known from game theory or human play but never appear in the agent's rollouts because the agent hasn't explored that part of the strategy space.

The Skill Crafter addresses these gaps by operating *top-down*: it proposes new skills, the Skill Bank tests them, and the Action Agent tries them.

---

## 2. Architecture

### Model tier assignment

The Skill Crafter runs entirely on **Tier 1 (Qwen3-32B/72B, inference-only)** — see [Action Agent §2](PLAN-ACTION-AGENT.md#2-tiered-model-architecture) for the full tiered architecture rationale. All three creation modes (Composer, Generalizer, Hypothesizer) and the Failure Reflector require multi-step counterfactual reasoning, cross-domain analogy, and structured diagnosis that exceed the 8B reasoning ceiling. Because these components run offline (between episodes, not per-step), the larger model adds no latency to the Action Agent's decision loop.

**Multi-run reasoning requirement:** Even at 32B/72B scale, single-pass inference is insufficient for the Skill Crafter's tasks. Each creation or reflection task requires multiple reasoning passes:

- **Composer:** Pass 1 — propose candidate compositions; Pass 2 — verify effect chain validity per pair; Pass 3 — generate protocol + test expectations.
- **Generalizer:** Pass 1 — identify shared structural slots; Pass 2 — propose mapping candidates; Pass 3 — instantiate and sanity-check with target-domain examples.
- **Hypothesizer:** Best-of-N sampling (N=3–5) — generate N proposals, score by contract completeness + novelty, keep top-K.
- **Failure Reflector:** Pass 1 — identify symptom step; Pass 2 — re-evaluate each prior hop with targeted prompts; Pass 3 — confirm root cause via counterfactual.

This multi-run design costs ~3–5× the tokens of a single 32B/72B call per task, but remains negligible compared to GRPO rollout costs since these tasks run once per episode batch, not per step.

### Pipeline overview

```
                     Skill Bank (existing skills)
                            ↓
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
    Skill Composer    Skill Generalizer   Skill Hypothesizer
              ↓             ↓             ↓
              └─────────────┼─────────────┘
                            ↓
                    Proto-Skill Proposals
                            ↓
                    Skill Bank (Stage 4: verify → promote)
                            ↓
                    Action Agent (trial execution)
                            ↓ (on failure)
                    ┌───────────────┐
                    │  Failure       │
                    │  Reflector     │
                    │  (§6)         │
                    └───────┬───────┘
                            ↓
              ┌─────────────┼──────────────┐
              ↓             ↓              ↓
         Localize       Diagnose       Recover
         (where?)       (why?)         (how to fix?)
              ↓             ↓              ↓
              └─────────────┼──────────────┘
                            ↓
              ┌─────────────┼─────────────┐
              ↓                           ↓
        Skill Bank                  Skill Crafter
        (patch existing)            (compose/hypothesize new)
              ↓                           ↓
              └─────────┬─────────────────┘
                        ↓
                  Retry with improved skills
```

### Three creation modes

| Mode | Input | Output | Method |
|------|-------|--------|--------|
| **Composer** | 2+ existing skills | Compound skill (chain/branch/loop) | Sequence planning + effect chaining |
| **Generalizer** | 1 skill + different domain | Transferred skill with adapted contract | Structural analogy via shared schema slots |
| **Hypothesizer** | Game description + failure patterns | Novel skill proposal | LLM reasoning over rules + failure analysis |

---

## 3. Skill Composer

### What it does

Takes two or more existing skills and proposes a compound skill that chains them together with explicit transition conditions.

### Composition operators

| Operator | Semantics | Example |
|----------|-----------|---------|
| `sequence(A, B)` | Execute A until success, then B | `sequence(POSITION_corner, MERGE_large)` |
| `if_then_else(cond, A, B)` | Check condition, branch | `if_then_else(stack_high, CLEAR_row, BUILD_tetris)` |
| `repeat_until(A, cond)` | Loop skill A | `repeat_until(MERGE_small, highest_tile >= 512)` |
| `fallback(A, B)` | Try A, on abort try B | `fallback(NAVIGATE_direct, NAVIGATE_detour)` |

### Effect chaining

A composition is valid when the postconditions of skill A satisfy the preconditions of skill B:

```
Skill A: eff_add = {corner_built, high_tile_positioned}
Skill B: preconditions = {corner_built}
         eff_add = {large_merge_completed}

Composed: sequence(A, B)
  preconditions = A.preconditions
  eff_add = A.eff_add ∪ B.eff_add
  eff_del = A.eff_del ∪ B.eff_del
```

### Discovery algorithm

1. For each pair of skills (A, B) in the bank:
   - Check: do A's eff_add predicates overlap with B's preconditions?
   - If yes: propose `sequence(A, B)` as a candidate compound skill.
2. For skills with high abort rates:
   - Find skills whose eff_add addresses the abort condition.
   - Propose `fallback(original, recovery)`.
3. For skills that are often repeated:
   - Check if repetition correlates with progress.
   - Propose `repeat_until(skill, progress_threshold)`.
4. Submit proposals to Skill Bank Stage 4 (proto-skill staging → verify → promote).

---

## 4. Skill Generalizer

### What it does

Takes a skill learned in one game/domain and proposes an analogue for a different game/domain by mapping through the shared schema.

### Transfer via shared schema slots

The canonical schema (from Visual Grounding) uses shared slot names across domains:

| Slot | Game example | Browser example | Video example |
|------|-------------|-----------------|---------------|
| `target` | tile to merge | button to click | person to track |
| `blocker` | wall preventing push | modal blocking click | occlusion blocking view |
| `constraint` | cannot pull boxes | form validation | temporal ordering |
| `candidate_set` | movable tiles | clickable elements | visible people |
| `progress` | score / boxes solved | form fields completed | clues found |

**Key insight:** Because the schema format is the same across domains, a skill defined in terms of schema slots can transfer to any domain that populates those slots.

### Transfer algorithm

1. For a source skill with contract `{eff_add, eff_del, preconditions}`:
   - Express predicates in terms of schema slots (e.g., `target.value >= 512` → `target.value >= threshold`).
   - Parameterize domain-specific constants.
2. In the target domain:
   - Find schema instances where the parameterized predicates can be instantiated.
   - Propose the transferred skill with adapted contract.
3. Validate:
   - Run the transferred skill in the target domain.
   - Verify eff_add predicates are achievable.
   - If pass rate > threshold → promote to bank.

### Cross-domain examples

| Source (game) | Target (browser) | Shared structure |
|--------------|-------------------|-----------------|
| 2048: `MERGE` (combine adjacent tiles) | Form: `FILL_ADJACENT` (fill adjacent fields) | `adjacent(target, e_next)` + `target.value → combined` |
| Sokoban: `PUSH_TO_GOAL` (push box to target) | Shopping: `ADD_TO_CART` (navigate item to cart) | `navigate(target) → apply_action(target)` |
| Tetris: `CLEAR_ROW` (complete and clear) | Todo: `COMPLETE_SECTION` (fill all fields in section) | `fill(candidate_set) → clear(candidate_set)` |

### Cross-domain examples (video benchmarks)

| Source (SIV-Bench) | Target (Video-Holmes) | Shared structure |
|--------------------|----------------------|-----------------|
| `TRACK_INTERACTION` (follow social exchange) | `TRACK_CLUE` (follow clue across scenes) | `track_object(query) → temporal chain of evidence` |
| `DETECT_EMOTION_CHANGE` (find expression shift) | `DETECT_PLOT_TWIST` (find narrative shift) | `detect_scene_changes() → compare_elements(before, after)` |

---

## 5. Skill Hypothesizer

### What it does

Proposes entirely new skills that don't exist in the bank, based on:
- Game rules / task descriptions
- Common failure patterns (high-abort skills, low-reward trajectories)
- Human strategy knowledge (via LLM)

### Hypothesis generation

1. **Failure analysis** (informed by Failure Reflector §6)**:**
   - Query the Failure Memory (§6.7) for skills with high abort rates, recurring failure patterns, or retired skills.
   - Use `FailureDiagnosis` records — specifically the `violated_assumption` and `suggested_fix` fields — as structured input rather than raw failure counts.
   - Ask LLM: "Given these diagnosed failures and their root causes, what strategy would avoid them?"
   - Propose the LLM's suggestion as a proto-skill, linking back to the failure patterns it addresses.

2. **Rule-based reasoning:**
   - Parse game description (`env.description`) for strategic implications.
   - Ask LLM: "What skills would be useful for a game with these rules?"
   - Cross-reference against existing bank skills.
   - Propose skills that cover gaps.

3. **Archetype matching:**
   - Maintain a library of game-theoretic archetypes (minimax, resource management, spatial planning, social deduction, etc.).
   - Match game characteristics to archetypes.
   - Instantiate archetype-specific skill templates.

### Hypothesis format

```python
ProtoSkill(
    name="corner_defense",
    strategic_description="Maintain highest tile in corner while building merge chains",
    proposed_by="hypothesizer:failure_analysis",
    source_evidence=["skill:MERGE has 40% abort rate when corner is lost"],
    proposed_contract=SkillEffectsContract(
        eff_add={"corner_maintained", "merge_chain_intact"},
        eff_del={"corner_lost"},
    ),
    proposed_protocol=Protocol(
        steps=["Position highest tile in corner", "Build descending chain along edge", "Merge from opposite end"],
        preconditions=["highest_tile >= 128"],
        success_criteria=["corner_maintained after 5+ merges"],
        abort_criteria=["highest_tile displaced from corner"],
    ),
    confidence=0.3,  # low — needs verification
)
```

Submitted to Skill Bank Stage 4 for proto-skill staging → verify → promote/reject.

---

## 6. Failure Reflection & Reasoning Recovery

### Motivation

When the agent executes a multi-hop reasoning chain and it fails — wrong answer, timeout, degraded reward, aborted skill — the failure is rarely random. It originates at a specific step in the chain, for a specific reason. Without the ability to **locate**, **diagnose**, and **learn from** these failures, the Skill Crafter proposes blind fixes. This section defines a structured failure reflection loop that closes that gap.

### 6.1 Failure Trace Capture

Every reasoning chain (hop sequence) produces an execution trace. On failure, the trace is captured as a `FailureTrace`:

```python
FailureTrace(
    episode_id="ep_0042",
    skill_name="MERGE_chain",
    hop_sequence=[
        HopRecord(step=0, action="GROUND", input="locate highest tile", output="tile_256 at (0,0)", status="ok"),
        HopRecord(step=1, action="CHECK",  input="adjacent mergeable?", output="tile_128 at (0,1)", status="ok"),
        HopRecord(step=2, action="EXECUTE", input="merge (0,0)+(0,1)", output="blocked by tile_64", status="FAIL"),
        HopRecord(step=3, action="CONCLUDE", input="—", output="—", status="skipped"),
    ],
    failure_step=2,
    failure_type="precondition_violated",
    context_snapshot={...},  # full state at failure point
    expected_outcome="tile_384 at (0,0)",
    actual_outcome="no merge; board unchanged",
)
```

Key fields:
- **`failure_step`** — the exact index in the hop sequence where things went wrong.
- **`failure_type`** — classified category (see §6.2).
- **`context_snapshot`** — the full `<state>` schema at the moment of failure, so the reflector can reason over what the agent *actually saw*.

### 6.2 Failure Classification Taxonomy

Not all failures are the same. The reflector classifies each failure into a category that determines the recovery strategy:

| Category | Description | Example | Recovery bias |
|----------|-------------|---------|---------------|
| **grounding_error** | Entity was misidentified or not found | VLM returned wrong tile position | Re-ground with refined query |
| **precondition_violated** | Step assumed a condition that wasn't true | Tried to merge but path was blocked | Backtrack and re-check preconditions |
| **stale_context** | Relied on outdated state information | Board changed between GROUND and EXECUTE | Re-observe before acting |
| **logical_error** | Reasoning step drew wrong conclusion from correct inputs | Chose wrong merge direction despite seeing the board correctly | Revise reasoning rule / protocol |
| **missing_information** | Needed data the agent didn't have | Couldn't infer hidden tile from history | Add RETRIEVE hop or request more context |
| **cascading_failure** | Earlier soft error amplified into hard failure at later step | Slightly wrong grounding → wrong CHECK → wrong EXECUTE | Trace back to root cause step |
| **resource_exhaustion** | Ran out of hops / time / tokens before completion | Complex chain exceeded hop budget | Simplify protocol or increase budget |

### 6.3 Failure Localization (Where)

The reflector walks backward through the hop sequence to find the **root cause step** — which may differ from the step that visibly failed.

**Algorithm: Backward Trace Analysis**

```
Input: FailureTrace T
Output: root_cause_step, root_cause_type

1. Start at T.failure_step (the step that raised the error).
2. For each prior step i = failure_step-1 ... 0:
   a. Re-evaluate step i's output against the context_snapshot at step i.
   b. If step i's output was already degraded (wrong entity, stale state, 
      incorrect inference):
      - Mark i as candidate root cause.
      - Classify the degradation type.
   c. If step i's output was correct given its inputs → stop.
      The root cause is the most recent candidate (or failure_step itself 
      if no prior degradation found).
3. Return (root_cause_step, root_cause_type).
```

This distinguishes between:
- **Direct failures** — the failing step itself is the problem (e.g., wrong EXECUTE action).
- **Cascading failures** — an earlier step silently produced bad output that propagated forward.

### 6.4 Failure Diagnosis (Why)

Once the root cause step is located, the reflector generates a structured diagnosis by prompting the LLM with the failure context:

```
Prompt template (failure_diagnosis):
───────────────────────────────────
You are analyzing a reasoning failure in a multi-hop chain.

**Skill:** {skill_name}
**Task:** {task_description}
**Full hop trace:** {hop_sequence}
**Root cause step:** Step {root_cause_step} — action: {action}, input: {input}
**Expected output:** {expected}
**Actual output:** {actual}
**State at failure:** {context_snapshot}

1. Why did step {root_cause_step} produce the wrong output?
2. What assumption was violated?
3. Was the input to this step correct? If not, trace the error further.
4. What specific change to the skill's protocol would prevent this failure?
───────────────────────────────────
```

The diagnosis output is a structured `FailureDiagnosis`:

```python
FailureDiagnosis(
    root_cause_step=2,
    root_cause_type="precondition_violated",
    explanation="Step 2 attempted merge without checking that the path between tiles was clear. "
                "The GROUND step correctly found the tiles, but the CHECK step only verified "
                "value compatibility, not spatial accessibility.",
    violated_assumption="Adjacent tiles with matching values can always merge",
    suggested_fix="Add a CHECK_PATH hop between GROUND and EXECUTE that verifies no blocking "
                  "tiles exist on the merge path.",
    confidence=0.75,
)
```

### 6.5 Recovery Strategies (How to Improve)

Based on the diagnosis, the reflector proposes one or more recovery actions. These aren't just retries — they are structured improvements that feed back into the Skill Crafter's three creation modes:

| Recovery strategy | When to apply | What it produces | Feeds into |
|-------------------|---------------|------------------|------------|
| **Protocol patch** | Logical error or missing check | Revised protocol with added/modified hop | Skill Bank (update existing skill) |
| **Precondition strengthening** | Precondition violated | Tighter precondition predicates on the skill contract | Skill Bank (contract update) |
| **Fallback injection** | Grounding error or stale context | `fallback(original_step, recovery_step)` composition | Composer (§3) |
| **Hop insertion** | Missing information | New RETRIEVE or CHECK hop inserted into protocol | Skill Bank (protocol update) |
| **Skill decomposition** | Cascading failure across many steps | Break monolithic skill into smaller, independently verifiable sub-skills | Composer (§3) |
| **Re-grounding trigger** | Stale context | Add re-observe checkpoints at key protocol boundaries | Skill Bank (protocol update) |
| **Skill retirement** | Persistent failure despite multiple fixes | Demote skill; let Hypothesizer (§5) propose replacement | Hypothesizer (§5) |

### 6.6 Reflection Loop Integration

The failure reflection loop runs as a post-episode process and connects to the rest of the Skill Crafter pipeline:

```
Episode execution (Action Agent)
        ↓ (on failure)
  FailureTrace capture
        ↓
  Failure Localization (§6.3)  →  root_cause_step
        ↓
  Failure Diagnosis (§6.4)     →  FailureDiagnosis
        ↓
  Recovery Proposal (§6.5)     →  RecoveryAction[]
        ↓
  ┌─────┴──────────────────┐
  ↓                        ↓
Skill Bank               Skill Crafter
(patch existing)         (compose/hypothesize new)
  ↓                        ↓
  └────────┬───────────────┘
           ↓
  Action Agent (retry with improved skills)
           ↓
  Evaluate: did the fix work?
           ↓
  ┌────────┴────────┐
  ↓                 ↓
 Yes: promote      No: escalate
 fix + update       (deeper reflection
 confidence)        or retire skill)
```

### 6.7 Failure Memory & Pattern Aggregation

Individual failure diagnoses are stored in a **Failure Memory** that enables pattern-level learning:

- **Recurrence detection:** If the same `(skill, failure_type, root_cause_step)` tuple appears N+ times, escalate from "patch" to "redesign."
- **Cross-skill patterns:** If multiple skills fail with the same `failure_type` (e.g., many skills have `stale_context` errors), propose a systemic fix (e.g., add re-grounding checkpoints to all long-chain protocols).
- **Failure clustering:** Group failures by shared violated assumptions. Each cluster may point to a missing primitive skill or a flawed schema mapping.
- **Improvement tracking:** For each recovery action applied, track whether it reduced the failure rate. Recovery actions with low success rates are themselves subject to reflection.

```python
FailureMemory(
    entries=[FailureDiagnosis, ...],
    
    # Aggregated patterns
    recurrence_counts={("MERGE_chain", "precondition_violated", 2): 7},
    cross_skill_patterns={"stale_context": ["MERGE_chain", "POSITION_corner", "NAVIGATE_path"]},
    recovery_effectiveness={
        "add_CHECK_PATH_hop": {"applied": 5, "resolved": 4, "rate": 0.80},
        "strengthen_precondition": {"applied": 3, "resolved": 1, "rate": 0.33},
    },
)
```

### 6.8 Escalation Policy

Not every failure warrants the same level of response. The escalation policy determines how much effort to invest:

| Occurrence count | Response level | Action |
|------------------|---------------|--------|
| 1st occurrence | **Log** | Store diagnosis, no immediate action |
| 2nd occurrence (same pattern) | **Patch** | Apply lightest recovery strategy |
| 3rd–5th occurrence | **Redesign** | Decompose or rewrite the skill protocol |
| 6+ occurrences | **Retire & replace** | Demote skill, ask Hypothesizer for alternative |
| Cross-skill pattern (3+ skills) | **Systemic fix** | Propose architectural change (new primitive, schema update) |

---

## 7. Transferable skill families (long-horizon reasoning)

> **Note:** Failure reflection (§6) applies to reasoning chains within these skill families — when a locate→filter→select chain fails at the "filter" step, the reflector localizes that step, diagnoses why the filter criteria were wrong, and proposes a protocol patch or fallback.

Under the two-level MDP (see [Action Agent §5](PLAN-ACTION-AGENT.md#5-two-level-mdp-long-horizon-reasoning)), the Skill Crafter composes and transfers *reasoning policies* — not single-call chain-of-thought templates, but actual multi-step policies that can be trained, composed, and transferred across domains.

### Cross-domain skill families

| Family | Game | Web | Visual Reasoning |
|--------|------|-----|------------------|
| **Locate → filter → select** | Candidate moves → best legal | UI candidates → relevant control | Objects → attributes → answer target |
| **Blocker → prerequisite → replan** | Deadlock → missing setup | Disabled control → missing field | Weak evidence → gather anchor |
| **History → hidden state → act** | Dialogue → alliance/threat | Prior pages → next step | Prior frames → disambiguate |
| **Compare under future constraint** | Move preserving structure | Path lowering risk/steps | Candidate consistent with constraints |

Each family is a reusable multi-step reasoning policy whose protocol maps to inner MDP actions (GROUND, CHECK, RETRIEVE, CONCLUDE, EXECUTE).

### Composition under the inner MDP

Skill composition (§2) gains a new dimension: composing *reasoning hops* rather than just environment actions.

- **Sequence composition** now chains hop protocols: Skill A's CONCLUDE feeds Skill B's GROUND trigger.
- **Fallback composition** tries alternative reasoning strategies: if GROUND fails to locate the entity, fall back to RETRIEVE from memory.
- **Nested composition**: an inner RETRIEVE hop can invoke a sub-skill's entire reasoning protocol.

### Transfer via shared reasoning vocabulary

Because all domains share the same inner action vocabulary (GROUND, CHECK, RETRIEVE, CONCLUDE, EXECUTE) and the same `<state>` schema structure, transfer becomes a matter of **schema-slot mapping** rather than domain-specific engineering:

1. **Source domain** skill: `Locate → filter → select` over entities {piece, obstacle, board_position} in a game.
2. **Target domain** mapping: {piece → form_field, obstacle → validation_error, board_position → form_section} in browser.
3. **Reasoning protocol** is unchanged: GROUND(target) → CHECK(constraints) → CONCLUDE(best) → EXECUTE(action).

---

## 8. Integration with Visual Grounding & Failure Reflection

The Skill Crafter leverages multi-hop visual reasoning from the vlm_wrapper:

### For Composition

- Use `spatial_query` to verify that composed skills' preconditions/postconditions are visually grounded.
- Use `detect_objects` to confirm entity existence before proposing entity-dependent compositions.

### For Generalization

- Use the shared schema as the transfer medium — both source and target domains produce `<state>` schemas with the same entity/relation/target structure.
- Visual grounding ensures transferred skills map to real visual entities in the target domain.

### For Hypothesis generation

- Use `classify_scene` to characterize the game state and match to archetypes.
- Use tool traces from failed episodes to identify where the agent's visual understanding broke down.

### For Failure Reflection

- Use `detect_objects` and `spatial_query` during backward trace analysis (§6.3) to re-evaluate whether GROUND steps produced correct visual grounding at the failure point.
- Compare the VLM's current perception against the `context_snapshot` stored in the `FailureTrace` to detect grounding drift.
- Feed visual grounding errors into the failure classification taxonomy (§6.2) as `grounding_error` — the most common failure type in visual reasoning domains.

---

## 9. Evaluation

### Metrics for crafted skills

| Metric | How measured | Threshold |
|--------|-------------|-----------|
| Contract pass rate | Stage 3 verification on trial episodes | ≥ 0.7 for promotion |
| Improvement over base | Reward delta when using crafted skill vs. existing | > 0 |
| Transfer success rate | % of cross-domain proposals that pass verification | Track |
| Composition utility | % of composed skills selected by action agent | Track |
| Novelty | Jaccard distance from nearest existing skill | ≥ 0.25 |

### Metrics for failure reflection

| Metric | How measured | Threshold |
|--------|-------------|-----------|
| Localization accuracy | % of root cause steps confirmed correct by manual review | ≥ 0.7 |
| Recovery success rate | % of applied recovery actions that resolved the failure pattern | ≥ 0.5 |
| Mean diagnoses to fix | Average number of reflection cycles before a failure pattern is resolved | ≤ 3 |
| Failure recurrence rate | % of fixed failures that reappear within N episodes | ≤ 0.15 |
| Systemic fix coverage | % of cross-skill patterns addressed by systemic fixes | Track |

### Evaluation protocol

1. Crafter proposes N proto-skills per iteration.
2. Each proto-skill enters Stage 4 staging → verification.
3. Promoted skills are tried by the action agent for K episodes.
4. Skills that improve reward are kept; others are demoted or removed.

---

## 10. Rollout order

**Phase 1 — Skill Composer (within-game)**
1. Implement effect chaining algorithm.
2. Propose sequence/fallback compositions for existing game skills.
3. Verify and promote via Stage 4.
4. Measure: do composed skills improve episode reward?

**Phase 2 — Skill Hypothesizer (within-game)**
1. Implement failure analysis pipeline.
2. Propose novel skills based on failure patterns + game rules.
3. Verify and promote via Stage 4.
4. Measure: do hypothesized skills cover previously unaddressed situations?

**Phase 3 — Skill Generalizer (cross-domain)**
1. Implement schema-slot-based transfer.
2. Transfer skills from games → browser (or vice versa).
3. Verify transferred skills in target domain.
4. Measure: does transfer reduce cold-start time in new domains?

**Phase 4 — Failure Reflection Loop**
1. Implement `FailureTrace` capture and `FailureMemory` storage.
2. Implement backward trace analysis for failure localization.
3. Implement LLM-based failure diagnosis with structured output.
4. Implement recovery strategy selection and application.
5. Implement escalation policy and pattern aggregation.
6. Measure: do reflected fixes reduce failure recurrence?

**Phase 5 — Cross-modality transfer (image ↔ video)**
1. Transfer visual reasoning skills from image benchmarks to video.
2. Transfer temporal reasoning patterns from video benchmarks to interactive environments.

---

## 11. TODO

| Task | Priority | Status |
|------|----------|--------|
| Effect chaining algorithm for skill composition | P0 | Not started |
| Composition operators (sequence, fallback, repeat_until) | P0 | Not started |
| Failure analysis pipeline for hypothesis generation | P1 | Not started |
| Schema-slot transfer algorithm | P1 | Not started |
| FailureTrace capture & HopRecord serialization | P0 | Not started |
| Backward trace analysis (failure localization) | P0 | Not started |
| Failure classification taxonomy implementation | P1 | Not started |
| LLM-based failure diagnosis with structured output | P1 | Not started |
| Recovery strategy selector & applicator | P1 | Not started |
| FailureMemory store & pattern aggregation | P1 | Not started |
| Escalation policy engine | P2 | Not started |
| Archetype library for hypothesis matching | P2 | Not started |
| Cross-domain transfer evaluation harness | P2 | Not started |
| Integration with vlm_wrapper tool traces | P2 | Not started |
| Hop-chain composition operators (inner MDP) | P1 | Not started |
| Cross-domain reasoning protocol transfer | P2 | Not started |
| Failure reflection metrics & dashboarding | P2 | Not started |

---

## 12. Implementation (planned)

| File (planned) | Purpose |
|----------------|---------|
| `skill_agents/crafter/composer.py` | Skill composition via effect chaining |
| `skill_agents/crafter/generalizer.py` | Cross-domain skill transfer |
| `skill_agents/crafter/hypothesizer.py` | Novel skill proposal from failures + rules |
| `skill_agents/crafter/archetypes.py` | Game-theoretic archetype library |
| `skill_agents/crafter/evaluation.py` | Crafted-skill quality metrics |
| `skill_agents/crafter/failure_trace.py` | FailureTrace / HopRecord data structures and capture |
| `skill_agents/crafter/failure_reflector.py` | Backward trace analysis, failure localization & classification |
| `skill_agents/crafter/failure_diagnosis.py` | LLM-based diagnosis prompt construction & structured output |
| `skill_agents/crafter/recovery.py` | Recovery strategy selection, application, and feedback |
| `skill_agents/crafter/failure_memory.py` | Persistent failure memory, pattern aggregation, escalation |

Currently no implementation exists — this is the newest component of the pipeline.
