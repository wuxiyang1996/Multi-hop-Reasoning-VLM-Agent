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

1. **Failure analysis:**
   - Identify skills with high abort rates or low pass rates.
   - Ask LLM: "Given these failures, what strategy would avoid them?"
   - Propose the LLM's suggestion as a proto-skill.

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

## 6. Transferable skill families (long-horizon reasoning)

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

## 7. Integration with Visual Grounding

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

---

## 8. Evaluation

### Metrics for crafted skills

| Metric | How measured | Threshold |
|--------|-------------|-----------|
| Contract pass rate | Stage 3 verification on trial episodes | ≥ 0.7 for promotion |
| Improvement over base | Reward delta when using crafted skill vs. existing | > 0 |
| Transfer success rate | % of cross-domain proposals that pass verification | Track |
| Composition utility | % of composed skills selected by action agent | Track |
| Novelty | Jaccard distance from nearest existing skill | ≥ 0.25 |

### Evaluation protocol

1. Crafter proposes N proto-skills per iteration.
2. Each proto-skill enters Stage 4 staging → verification.
3. Promoted skills are tried by the action agent for K episodes.
4. Skills that improve reward are kept; others are demoted or removed.

---

## 9. Rollout order

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

**Phase 4 — Cross-modality transfer (image ↔ video)**
1. Transfer visual reasoning skills from image benchmarks to video.
2. Transfer temporal reasoning patterns from video benchmarks to interactive environments.

---

## 10. TODO

| Task | Priority | Status |
|------|----------|--------|
| Effect chaining algorithm for skill composition | P0 | Not started |
| Composition operators (sequence, fallback, repeat_until) | P0 | Not started |
| Failure analysis pipeline for hypothesis generation | P1 | Not started |
| Schema-slot transfer algorithm | P1 | Not started |
| Archetype library for hypothesis matching | P2 | Not started |
| Cross-domain transfer evaluation harness | P2 | Not started |
| Integration with vlm_wrapper tool traces | P2 | Not started |
| Hop-chain composition operators (inner MDP) | P1 | Not started |
| Cross-domain reasoning protocol transfer | P2 | Not started |

---

## 11. Implementation (planned)

| File (planned) | Purpose |
|----------------|---------|
| `skill_agents/crafter/composer.py` | Skill composition via effect chaining |
| `skill_agents/crafter/generalizer.py` | Cross-domain skill transfer |
| `skill_agents/crafter/hypothesizer.py` | Novel skill proposal from failures + rules |
| `skill_agents/crafter/archetypes.py` | Game-theoretic archetype library |
| `skill_agents/crafter/evaluation.py` | Crafted-skill quality metrics |

Currently no implementation exists — this is the newest component of the pipeline.
