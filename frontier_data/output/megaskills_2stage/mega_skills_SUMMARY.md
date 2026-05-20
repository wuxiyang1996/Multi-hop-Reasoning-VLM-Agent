# Mega-skills from LLM-judge clustering

- generated: 2026-05-20T09:08:12.593796+00:00
- judgments: `/workspace/Multi-hop-Reasoning-VLM-Agent/frontier_data/output/plan_level_similarity_judgments.json`
- threshold: ≥ 4
- source banks: 5 GRPO-validated games (82 skills)
- mega-skills: **10** (10 multi-task, 0 multi-domain)

## Mega-skills (sorted by member count)

### `mega.000.explore` — ACT → VERIFY

- members: **36** across 5 task(s): candy_crush, gymv_columns, gymv_streets_of_rage_2, gymv_strider, gymv_thunder_force_iii
- representative: `gymv_columns::__EXPLORE__` — Explore
  - description: The EXPLORE skill advances the game phase from opening to midgame while accumulating score and clearing initial state predicates.

- template steps (from representative):
  1. Scan the board to identify available move options and current state.
  2. Select the highest-value legal move based on immediate scoring potential.
  3. Execute the selected move to place the piece on the board.
  4. Verify the board state has updated and score has increased.
  5. Assess the new board configuration for emerging patterns or threats.
  6. If a high-value combo is detected, execute the follow-up move immediately.
  7. Confirm the transition to a stable midgame state with accumulated points.

- judge rationale (top-3 edges):
  - score=5: Select a high-value move, execute it, perform follow-up consolidations/combos, and verify midgame.
  - score=5: Position, execute high-value attack to increase score, verify increment, trigger and lock phase change.
  - score=5: Select phase-dependent action, execute it, verify score and deactivate initial predicates.

- ICL exemplar source: `gymv_columns::__EXPLORE__` (protocol_raw)

- members detail:
  - `candy_crush                   ::early:COMMIT/CLEAR            ` sig=`EVALUATE → ACT` n_steps=3
  - `candy_crush                   ::late:COMMIT/CLEAR             ` sig=`EVALUATE → ACT` n_steps=3
  - `gymv_columns                  ::COMMIT/CLEAR                  ` sig=`PERCEIVE → ACT → EVALUATE` n_steps=3
  - `gymv_columns                  ::COMMIT/EXECUTE                ` sig=`ACT → EVALUATE → NAVIGATE` n_steps=5
  - `gymv_columns                  ::COMMIT/SETUP                  ` sig=`ACT → VERIFY → EVALUATE` n_steps=4
  - `gymv_columns                  ::INSPECT/SETUP                 ` sig=`PERCEIVE → VERIFY → ACT` n_steps=3
  - `gymv_columns                  ::__EXPLORE__                   ` sig=`PERCEIVE → DECIDE → NAVIGATE → EVALUATE → ACT → VERIFY` n_steps=7 ★
  - `gymv_columns                  ::early:SETUP                   ` sig=`EVALUATE → ACT → VERIFY` n_steps=3
  - `gymv_columns                  ::late:COMMIT/CLEAR             ` sig=`PERCEIVE → EVALUATE → VERIFY` n_steps=4
  - `gymv_columns                  ::late:INSPECT/SETUP            ` sig=`PERCEIVE → EVALUATE` n_steps=3
  - `gymv_columns                  ::mid:EXECUTE                   ` sig=`ACT → EVALUATE → PERCEIVE → VERIFY` n_steps=5
  - `gymv_columns                  ::mid:NAVIGATE                  ` sig=`ACT → NAVIGATE → ACT` n_steps=5
  - `gymv_columns                  ::mid:OPTIMIZE                  ` sig=`ACT → VERIFY` n_steps=3
  - `gymv_streets_of_rage_2        ::COMMIT/ATTACK                 ` sig=`ACT → VERIFY → EVALUATE` n_steps=4
  - `gymv_streets_of_rage_2        ::__EXPLORE__                   ` sig=`EVALUATE → PERCEIVE → VERIFY` n_steps=3
  - `gymv_streets_of_rage_2        ::early:COMMIT/ATTACK           ` sig=`ACT → VERIFY → PERCEIVE → EVALUATE → ACT` n_steps=7
  - `gymv_streets_of_rage_2        ::early:SETUP                   ` sig=`ACT → EVALUATE` n_steps=2
  - `gymv_streets_of_rage_2        ::late:COMMIT/ATTACK            ` sig=`NAVIGATE → ACT → VERIFY → EVALUATE → ACT` n_steps=6
  - `gymv_streets_of_rage_2        ::late:EXECUTE                  ` sig=`ACT → EVALUATE → ACT` n_steps=4
  - `gymv_strider                  ::COMMIT/ATTACK                 ` sig=`ACT → VERIFY → PERCEIVE` n_steps=4
  - `gymv_strider                  ::COMMIT/COLLECT                ` sig=`ACT → VERIFY → EVALUATE` n_steps=4
  - `gymv_strider                  ::COMMIT/NAVIGATE               ` sig=`ACT → VERIFY → NAVIGATE → PERCEIVE → EVALUATE` n_steps=6
  - `gymv_strider                  ::COMMIT/POSITION               ` sig=`ACT` n_steps=5
  - `gymv_strider                  ::COMPARE/ATTACK                ` sig=`ACT → VERIFY` n_steps=3
  - `gymv_strider                  ::early:COMMIT/NAVIGATE         ` sig=`ACT → PERCEIVE` n_steps=3
  - `gymv_strider                  ::early:SETUP                   ` sig=`ACT → VERIFY` n_steps=4
  - `gymv_strider                  ::late:COMMIT/ATTACK            ` sig=`VERIFY → EVALUATE → ACT → VERIFY` n_steps=4
  - `gymv_strider                  ::mid:NAVIGATE                  ` sig=`NAVIGATE → VERIFY` n_steps=3
  - `gymv_strider                  ::mid:OPTIMIZE                  ` sig=`EVALUATE → VERIFY → EVALUATE` n_steps=4
  - `gymv_thunder_force_iii        ::COMMIT/POSITION               ` sig=`ACT → PERCEIVE → VERIFY` n_steps=4
  - `gymv_thunder_force_iii        ::__EXPLORE__                   ` sig=`ACT → EVALUATE` n_steps=3
  - `gymv_thunder_force_iii        ::early:COMMIT/NAVIGATE         ` sig=`NAVIGATE → EVALUATE → VERIFY` n_steps=3
  - `gymv_thunder_force_iii        ::early:EXPLORE                 ` sig=`DECIDE → ACT` n_steps=4
  - `gymv_thunder_force_iii        ::early:SETUP                   ` sig=`ACT → VERIFY → EVALUATE → ACT → PERCEIVE` n_steps=6
  - `gymv_thunder_force_iii        ::late:OPTIMIZE                 ` sig=`ACT → EVALUATE` n_steps=3
  - `gymv_thunder_force_iii        ::mid:NAVIGATE                  ` sig=`ACT → VERIFY` n_steps=3

### `mega.001.late_recover_survive` — EVALUATE → ACT → PERCEIVE → ACT

- members: **13** across 5 task(s): candy_crush, gymv_columns, gymv_streets_of_rage_2, gymv_strider, gymv_thunder_force_iii
- representative: `gymv_thunder_force_iii::late:RECOVER/SURVIVE` — Late Recover/Survive
  - description: The skill triggers the transition from midgame to endgame, decreasing life count by one and raising the final score.

- template steps (from representative):
  1. Evaluate best available action
  2. Execute chosen action
  3. Observe result
  4. Achieve: world.lives=3, world.phase=endgame, world.score=80
  5. Remove: world.lives=4, world.phase=midgame, world.score=60

- judge rationale (top-3 edges):
  - score=5: Select best action → execute it → observe result and update world state.
  - score=5: Select best action → execute it → observe result and update world state.
  - score=5: Select best action → execute it → observe result and update world state (including lives).

- ICL exemplar source: `gymv_thunder_force_iii::late:RECOVER/SURVIVE` (protocol_raw)

- members detail:
  - `candy_crush                   ::early:CLEAR                   ` sig=`PERCEIVE → ACT → EVALUATE → ACT` n_steps=4
  - `gymv_columns                  ::early:RECOVER/SURVIVE         ` sig=`EVALUATE → ACT → PERCEIVE → ACT` n_steps=4
  - `gymv_streets_of_rage_2        ::COMMIT/EXECUTE                ` sig=`EVALUATE → ACT → PERCEIVE → ACT` n_steps=4
  - `gymv_streets_of_rage_2        ::late:COMMIT/EXECUTE           ` sig=`EVALUATE → ACT → PERCEIVE → ACT` n_steps=5
  - `gymv_streets_of_rage_2        ::late:OPTIMIZE                 ` sig=`ACT → VERIFY → ACT` n_steps=4
  - `gymv_strider                  ::late:COMMIT/CLEAR             ` sig=`ACT → EVALUATE → VERIFY → PERCEIVE` n_steps=4
  - `gymv_strider                  ::mid:EXECUTE                   ` sig=`EVALUATE → ACT → PERCEIVE → ACT` n_steps=4
  - `gymv_thunder_force_iii        ::COMMIT/CLEAR                  ` sig=`EVALUATE → ACT → PERCEIVE → ACT` n_steps=5
  - `gymv_thunder_force_iii        ::COMMIT/NAVIGATE               ` sig=`EVALUATE → ACT → PERCEIVE → ACT` n_steps=5
  - `gymv_thunder_force_iii        ::late:COMMIT/ATTACK            ` sig=`ACT → PERCEIVE` n_steps=3
  - `gymv_thunder_force_iii        ::late:COMMIT/EXECUTE           ` sig=`EVALUATE → ACT → PERCEIVE → EVALUATE` n_steps=5
  - `gymv_thunder_force_iii        ::late:RECOVER/DEFEND           ` sig=`EVALUATE → ACT → PERCEIVE → EVALUATE` n_steps=5
  - `gymv_thunder_force_iii        ::late:RECOVER/SURVIVE          ` sig=`EVALUATE → ACT → PERCEIVE → EVALUATE` n_steps=5 ★

### `mega.002.recover_survive` — ACT → VERIFY → EVALUATE

- members: **7** across 3 task(s): gymv_streets_of_rage_2, gymv_strider, gymv_thunder_force_iii
- representative: `gymv_strider::RECOVER/SURVIVE` — Recover/Survive
  - description: Survive a room transition into a new hazard-filled area, remaining alive while enemies, fire columns, and UI/scene elements change.

- template steps (from representative):
  1. Immediately execute a movement action (e.g., 'jump' or 'dash') to increase distance from the nearest hazard or enemy.
  2. If an enemy is within attack range, execute a defensive or evasive action (e.g., 'roll' or 'block') to avoid damage.
  3. Maintain continuous movement away from static hazards (fire columns) until the room transition completes or the threat is neutralized.
  4. Observe the environment for new threats and adjust movement vector accordingly.

- judge rationale (top-3 edges):
  - score=5: Evasive movement away from threats, verify safety, then monitor/adjust for new hazards.
  - score=5: Lateral evasive maneuver away from fire, then monitor/adjust heading until threat passes.
  - score=5: Detect imminent hazard, execute evasive movement to relieve pressure, maintain until safe

- ICL exemplar source: `gymv_strider::RECOVER/SURVIVE` (protocol_raw)

- members detail:
  - `gymv_streets_of_rage_2        ::RECOVER/EVADE                 ` sig=`ACT → VERIFY → EVALUATE` n_steps=3
  - `gymv_streets_of_rage_2        ::late:SURVIVE                  ` sig=`PERCEIVE → ACT` n_steps=3
  - `gymv_strider                  ::COMMIT/EXPLORE                ` sig=`NAVIGATE → VERIFY` n_steps=3
  - `gymv_strider                  ::RECOVER/SURVIVE               ` sig=`ACT → PERCEIVE` n_steps=4 ★
  - `gymv_thunder_force_iii        ::RECOVER/EVADE                 ` sig=`ACT → VERIFY → ACT` n_steps=4
  - `gymv_thunder_force_iii        ::RECOVER/SURVIVE               ` sig=`ACT → EVALUATE` n_steps=3
  - `gymv_thunder_force_iii        ::mid:EXECUTE                   ` sig=`ACT → NAVIGATE → ACT` n_steps=3

### `mega.003.recover_reshuffle` — EVALUATE → ACT → PERCEIVE

- members: **4** across 2 task(s): candy_crush, gymv_streets_of_rage_2
- representative: `candy_crush::skill-770a9b2393` — Recover/Reshuffle
  - description: When no valid match exists and the board has been reshuffled, take the highest-value match-3 that is now available (the reshuffle guarantees at least one).

- template steps (from representative):
  1. Evaluate best available action
  2. Execute chosen action
  3. Observe result

- judge rationale (top-3 edges):
  - score=5: Evaluate best action → execute it → observe result
  - score=5: Evaluate best action, execute it, then observe the result.
  - score=5: Evaluate best action → execute it → observe the result (same 3-step loop).

- ICL exemplar source: `candy_crush::mid:EXECUTE` (protocol_raw)

- members detail:
  - `candy_crush                   ::mid:EXECUTE                   ` sig=`EVALUATE → ACT → PERCEIVE` n_steps=3
  - `candy_crush                   ::skill-2d9b4140e3              ` sig=`EVALUATE → ACT → PERCEIVE` n_steps=3
  - `candy_crush                   ::skill-770a9b2393              ` sig=`EVALUATE → ACT → PERCEIVE` n_steps=3 ★
  - `gymv_streets_of_rage_2        ::early:COMMIT/COLLECT          ` sig=`EVALUATE → ACT → PERCEIVE` n_steps=3

### `mega.004.inspect_setup` — EVALUATE → ACT → PERCEIVE → EVALUATE → ACT

- members: **4** across 4 task(s): gymv_columns, gymv_streets_of_rage_2, gymv_strider, gymv_thunder_force_iii
- representative: `gymv_strider::INSPECT/SETUP` — Inspect/Setup
  - description: Transitions from an initial mixed/menu-like scene into the active stage setup, relabeling the player as hero and spawning the level background, HUD, and visible enemies/items.

- template steps (from representative):
  1. Issue the 'Start' or 'Enter' action command to dismiss the initial menu scene.
  2. Verify the transition by checking if the visual state changes from a menu to the game world.
  3. Confirm the presence of the HUD (Heads-Up Display) elements such as health bars or score indicators.
  4. Validate that the level background and at least one enemy or item entity are visible on the screen.
  5. Ensure the player entity is now labeled as 'hero' and positioned within the active play area.

- judge rationale (top-3 edges):
  - score=5: Evaluate→Act→Observe; set concrete score values and clear initial score appearance events.
  - score=5: Trigger scene setup then verify HUD, background, entities, and player labeling.
  - score=5: Evaluate an action, execute it, observe result, produce a score state and clear startup events.

- ICL exemplar source: `gymv_strider::INSPECT/SETUP` (protocol_raw)

- members detail:
  - `gymv_columns                  ::early:INSPECT/SETUP           ` sig=`EVALUATE → ACT → PERCEIVE → EVALUATE → ACT` n_steps=5
  - `gymv_streets_of_rage_2        ::INSPECT/SETUP                 ` sig=`ACT → PERCEIVE` n_steps=4
  - `gymv_strider                  ::INSPECT/SETUP                 ` sig=`ACT → VERIFY → EVALUATE → VERIFY` n_steps=5 ★
  - `gymv_thunder_force_iii        ::early:INSPECT/SETUP           ` sig=`EVALUATE → ACT → PERCEIVE → EVALUATE → ACT` n_steps=5

### `mega.005.early_recover_survive` — EVALUATE → ACT → PERCEIVE → ACT

- members: **3** across 2 task(s): candy_crush, gymv_thunder_force_iii
- representative: `gymv_thunder_force_iii::early:RECOVER/SURVIVE` — Early Recover/Survive
  - description: Recovers status by advancing game phase to midgame and triggering score changes.

- template steps (from representative):
  1. Evaluate best available action
  2. Execute chosen action
  3. Observe result
  4. Achieve: event.score_changed, world.phase=midgame, world.score=50
  5. Remove: event.lives_appeared, event.score_appeared, world.phase=opening

- judge rationale (top-3 edges):
  - score=5: Evaluate→execute→observe, then achieve score_changed and transition phase while removing opening events.
  - score=5: Evaluate→execute→observe, then achieve midgame and remove opening phase predicate.
  - score=5: Evaluate→act→observe→achieve midgame and remove opening.

- ICL exemplar source: `gymv_thunder_force_iii::early:RECOVER/SURVIVE` (protocol_raw)

- members detail:
  - `candy_crush                   ::COMMIT/CLEAR                  ` sig=`EVALUATE → ACT → PERCEIVE → ACT` n_steps=5
  - `gymv_thunder_force_iii        ::early:COMMIT/ATTACK           ` sig=`EVALUATE → ACT → PERCEIVE → ACT` n_steps=5
  - `gymv_thunder_force_iii        ::early:RECOVER/SURVIVE         ` sig=`EVALUATE → ACT → PERCEIVE → EVALUATE → ACT` n_steps=5 ★

### `mega.006.commit_explore` — EVALUATE → ACT → PERCEIVE → ACT

- members: **3** across 3 task(s): candy_crush, gymv_columns, gymv_thunder_force_iii
- representative: `gymv_columns::COMMIT/EXPLORE` — Commit/Explore
  - description: Commits the falling column in the left well, advancing piece/jewel identities and setting the subgoal to clear a match with the current falling column.

- template steps (from representative):
  1. Identify the specific piece/jewel identity currently falling in the left well.
  2. Execute the drop action to commit the falling column to the board grid.
  3. Verify the board transformation reflects the new piece placement.
  4. Initiate a search for adjacent matching pieces triggered by the new placement.
  5. Confirm the execution of a match-clearing action if a match is found.
  6. Observe the removal of matched pieces and the update of the score counter.

- judge rationale (top-3 edges):
  - score=5: Identify piece, commit it, verify board transform, search/confirm match and score update.
  - score=5: Commit a placement/move, verify board transformation and that a match was scored.
  - score=5: Commit action to change world, observe new pieces/board and verify score update

- ICL exemplar source: `gymv_columns::COMMIT/EXPLORE` (protocol_raw)

- members detail:
  - `candy_crush                   ::skill-37c95b1014              ` sig=`EVALUATE → ACT → PERCEIVE → ACT` n_steps=4
  - `gymv_columns                  ::COMMIT/EXPLORE                ` sig=`PERCEIVE → ACT → VERIFY → ACT → VERIFY → PERCEIVE` n_steps=6 ★
  - `gymv_thunder_force_iii        ::COMMIT/EXPLORE                ` sig=`ACT → EVALUATE` n_steps=3

### `mega.007.mid_execute` — EVALUATE → ACT → PERCEIVE → EVALUATE

- members: **3** across 3 task(s): gymv_columns, gymv_streets_of_rage_2, gymv_thunder_force_iii
- representative: `gymv_streets_of_rage_2::mid:EXECUTE` — Mid Execute
  - description: Transitions the game phase from midgame to endgame, triggers a score change event, and results in a final score of 450.

- template steps (from representative):
  1. Evaluate best available action
  2. Execute chosen action
  3. Observe result
  4. Remove: world.score=200

- judge rationale (top-3 edges):
  - score=5: Evaluate→Execute→Observe then remove a specific world.score flag
  - score=5: Evaluate→Execute→Observe then remove a specific world.score flag
  - score=5: Evaluate → execute → observe → remove a specific world.score.

- ICL exemplar source: `gymv_streets_of_rage_2::mid:EXECUTE` (protocol_raw)

- members detail:
  - `gymv_columns                  ::early:COMMIT/CLEAR            ` sig=`EVALUATE → ACT → PERCEIVE → EVALUATE` n_steps=4
  - `gymv_streets_of_rage_2        ::mid:EXECUTE                   ` sig=`EVALUATE → ACT → PERCEIVE → EVALUATE` n_steps=4 ★
  - `gymv_thunder_force_iii        ::late:COMMIT/CLEAR             ` sig=`EVALUATE → ACT → PERCEIVE → EVALUATE` n_steps=4

### `mega.008.explore.f63b` — EVALUATE → VERIFY

- members: **3** across 2 task(s): gymv_streets_of_rage_2, gymv_strider
- representative: `gymv_strider::__EXPLORE__` — Explore
  - description: No state changes observed; skill has no effect on game state.

- template steps (from representative):
  1. Move the agent one grid cell in a random valid direction
  2. Wait for the movement animation to complete and the new tile to render
  3. Check the visual input for new entities, items, or terrain features not seen in the previous frame
  4. If no new visual information is detected, repeat the move action in a different direction
  5. Continue moving until a new object is detected or a maximum step limit is reached

- judge rationale (top-3 edges):
  - score=5: Perform movement to change position and verify the game state updated/observed.
  - score=5: Execute movement/navigation and verify position change and observed state.
  - score=4: Move action then verify movement/observation of state change.

- ICL exemplar source: `gymv_strider::__EXPLORE__` (protocol_raw)

- members detail:
  - `gymv_streets_of_rage_2        ::mid:NAVIGATE                  ` sig=`EVALUATE → VERIFY` n_steps=3
  - `gymv_strider                  ::__EXPLORE__                   ` sig=`ACT → VERIFY → ACT` n_steps=5 ★
  - `gymv_strider                  ::early:EXPLORE                 ` sig=`ACT → VERIFY` n_steps=3

### `mega.009.commit_evade` — ACT → VERIFY → ACT

- members: **3** across 2 task(s): gymv_strider, gymv_thunder_force_iii
- representative: `gymv_thunder_force_iii::COMMIT/EVADE` — Commit/Evade
  - description: The player ship commits to an evasive move into a safer cave lane/path, transitioning from generic terrain-wall surroundings to a distinct upper/lower route while remaining visible.

- template steps (from representative):
  1. Rotate ship orientation to align with the entrance of the target upper or lower cave lane.
  2. Initiate maximum thrust to accelerate into the cave lane, transitioning terrain from generic walls to the distinct route.
  3. Maintain forward momentum through the cave lane while keeping the ship within the visible bounds of the path.
  4. Verify that the ship has fully entered the safer cave lane and is no longer exposed in the generic open area.

- judge rationale (top-3 edges):
  - score=5: Align to lane entrance, thrust into lane, maintain momentum, verify safe entry.
  - score=5: Align to target lane, initiate movement (thrust), maintain momentum, verify safe entry.
  - score=5: Input directional+recovery to evade, sustain vector until reaching safe landing.

- ICL exemplar source: `gymv_thunder_force_iii::COMMIT/EVADE` (protocol_raw)

- members detail:
  - `gymv_strider                  ::COMMIT/EVADE                  ` sig=`ACT → VERIFY → ACT` n_steps=3
  - `gymv_strider                  ::RECOVER/EVADE                 ` sig=`ACT` n_steps=4
  - `gymv_thunder_force_iii        ::COMMIT/EVADE                  ` sig=`ACT → VERIFY` n_steps=4 ★
