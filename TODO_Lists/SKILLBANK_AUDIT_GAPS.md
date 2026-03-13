# Skill Bank Audit — Issues & Gaps

**Created:** 2026-03-13  
**Source:** `labeling/output/gpt54_skillbank/` full audit  
**Status:** Open  

---

## Context

The `gpt54_skillbank` extraction ran GPT-5.4 over 8 games (avalon, candy_crush, diplomacy, pokemon_red, sokoban, super_mario, tetris, twenty_forty_eight), producing 48 skills, 456 sub-episodes, and 13 cross-game archetypes. The bank is structurally complete: all fields consumed by `skill_bank_to_text()`, `select_skill_from_bank()`, and `RewardComputer` are populated. The three-tier query path (tag filtering → RAG similarity → protocol execution) works end-to-end.

This file tracks the remaining issues and improvement areas found during the audit.

---

## Issues

### 1. `quality_score` is 0.0 everywhere

**Severity:** Medium  
**Where:** Every `sub_episodes[].quality_score` in all `skill_bank.jsonl` files  

All 456 sub-episodes have `quality_score: 0.0`. Either the quality scoring pass was never run, or the extraction pipeline doesn't compute it yet. Quality scores would let `select_skill_from_bank()` prefer higher-quality exemplars and allow the skill agent to prune low-quality evidence during refinement.

**Action:**
- [ ] Add a quality scoring pass to `label_episodes_gpt54.py` or as a post-processing step
- [ ] Score based on: outcome (success > partial > fail), cumulative_reward relative to skill average, segment length vs expected_duration, whether success_criteria were met

---

### 2. `episode_id` is empty string on all sub-episodes

**Severity:** Low-Medium  
**Where:** Every `sub_episodes[].episode_id` in all `skill_bank.jsonl` files  

All sub-episodes have `episode_id: ""`. This makes it impossible to trace a sub-episode back to its source episode file for debugging, replay, or provenance tracking.

**Action:**
- [ ] Populate `episode_id` during extraction from the source episode filename or metadata
- [ ] Format: `"{game}_episode_{idx}"` or use the actual filename stem

---

### 3. Single episode per game limits skill diversity

**Severity:** Medium-High  
**Where:** `extraction_batch_summary.json` shows `episodes_processed: 1` for every game  

Only 1 episode was processed per game. This means:
- Skills are biased toward one playthrough's strategy
- Thin skills (e.g. SETUP for 2048 has only 4 instances) lack robustness
- No coverage of alternative strategies (e.g. different anchor corners in 2048, different openers in Diplomacy)

**Action:**
- [ ] Run extraction on additional episodes per game (target 3-5 episodes minimum)
- [ ] After re-extraction, check that skill merging in the skill agent handles duplicates across episodes
- [ ] Priority games: twenty_forty_eight (SETUP is thin), candy_crush (only 15 sub-episodes total), avalon (only 7 sub-episodes)

---

### 4. `expected_duration` may be under-calibrated

**Severity:** Low  
**Where:** `skill.protocol.expected_duration` across all skills  

For 2048, MERGE says `expected_duration: 3` but sub-episodes show segments spanning 5-15 steps. CLEAR, POSITION, SETUP, SURVIVE all say 1 step. These are rough per-invocation estimates rather than observed durations.

**Action:**
- [ ] Compute `median_duration` and `p90_duration` from sub-episode `(seg_end - seg_start)` values
- [ ] Use median as `expected_duration` or add separate `observed_median_duration` field
- [ ] Feed into `skill_abort_k` logic in the decision agent (abort after K × expected_duration)

---

### 5. `report` field is null on all skills

**Severity:** Low  
**Where:** Every skill entry in `skill_bank.jsonl` has `"report": null`  

Verification reports (pass rate, failure analysis) were not generated. Not blocking, but useful for confidence scoring and debugging.

**Action:**
- [ ] Run the skill verification stage (`stage5_verify`) if implemented
- [ ] Or compute lightweight stats: pass_rate = n_success / n_instances, common failure mode frequency

---

### 6. No normalized confidence score

**Severity:** Low  
**Where:** Skills store `n_instances` but no pre-computed `confidence` float  

`skill_bank_to_text()` currently derives confidence from `n_instances > 0` (binary). A normalized score (e.g. sigmoid over n_instances with midpoint at 10) would give more useful prompt text and help skill selection prioritize well-evidenced skills.

**Action:**
- [ ] Add `confidence` field during extraction: `confidence = min(1.0, n_instances / 20)` or similar
- [ ] Or compute on load in `SkillBankMVP.load()` as a derived property

---

### 7. Cross-game archetype grouping quirks

**Severity:** Low  
**Where:** `skill_archetypes.json` and `skill_rag_index.json`  

- The `archetype_merge` groups all 4 non-CLEAR 2048 skills (MERGE, POSITION, SETUP, SURVIVE) under one umbrella. While conceptually reasonable (they all serve the merge-around-anchor strategy), it means RAG queries for "survive" in 2048 land in the "merge" archetype rather than the "survive" archetype. The `archetype_survive` only has sokoban's "Step Away Safely."
- Some archetypes have truncated descriptions (ending mid-sentence), e.g. `archetype_collect.description` ends with "...through simple, ".

**Action:**
- [ ] Review archetype assignment — consider whether 2048 SURVIVE should also appear under `archetype_survive`
- [ ] Fix truncated description strings (likely a max_chars cutoff in the extraction prompt)

---

## What's Working Well (no action needed)

- **Protocol completeness:** All skills have full `preconditions`, `steps`, `success_criteria`, `abort_criteria` — decision agent gets actionable guidance
- **Contract completeness:** All `eff_add`, `eff_del`, `eff_event` populated — reward shaping works
- **Execution hints:** `common_preconditions`, `termination_cues`, `common_failure_modes`, `execution_description` all populated — `select_skill_from_bank()` returns rich guidance
- **RAG index format:** `text` fields use `game=... | skill=... | effects=... | context=...` matching `build_rag_summary()` output — embedding similarity will score correctly
- **Catalog consistency:** Per-game catalogs and `skill_catalog_all.json` are consistent with the JSONL bank entries
- **Tag coverage per game is appropriate:** Each game uses the tags that make sense for its mechanics (2048 doesn't need ATTACK/NAVIGATE, Sokoban doesn't need MERGE, etc.)
