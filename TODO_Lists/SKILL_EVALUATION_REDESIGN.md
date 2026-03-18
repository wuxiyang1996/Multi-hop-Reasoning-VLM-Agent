# Skill Evaluation Redesign

**Status**: Planned (not started)
**Goal**: Redesign skill evaluation to use a composite score of success rate, expected reward, and reusability, while adding a UCB-style exploration bonus to skill selection.

---

## Current State

Two separate scoring paths need updating:

1. **Skill quality scoring** (`quality/sub_episode_evaluator.py`) — evaluates sub-episodes with `0.3*reward + 0.3*follow_through + 0.2*compactness + 0.2*consistency`. Used for bank maintenance (retire, refine, promote).

2. **Skill selection confidence** (`query.py`) — ranks candidates at runtime with `0.4*relevance + 0.35*applicability + 0.25*pass_rate`. No exploration bonus exists.

Neither considers **expected reward** or **reusability**. There is **no exploration mechanism** — popular skills get selected more, starving new/niche skills.

---

## Part A: Composite Skill Evaluation Score

Add `compute_skill_score()` on the `Skill` dataclass in `skill_agents_grpo/stage3_mvp/schemas.py`:

```
skill_score = w_sr * success_rate
            + w_rw * normalized_expected_reward
            + w_re * reusability
```

- **success_rate** (weight 0.35): existing `self.success_rate` (successes / total sub-episodes)
- **expected_reward** (weight 0.35): mean `quality_score` across sub-episodes, evidence-discounted: `evidence_factor * mean_quality` where `evidence_factor = min(1.0, n / 10)`
- **reusability** (weight 0.30): how broadly applicable the skill is:
  - `context_diversity`: distinct `summary_state` contexts / n_instances
  - `transition_diversity`: distinct predecessor+successor skills
  - Combined: `min(1.0, 0.6 * context_diversity + 0.4 * transition_ratio)`

---

## Part B: Exploration Bonus (UCB-style)

Modify `_compute_confidence()` in `skill_agents_grpo/query.py`:

```
confidence = exploitation_score + exploration_bonus
```

- `exploitation_score = 0.35 * relevance + 0.30 * norm_applicability + 0.20 * pass_rate + 0.15 * skill_score`
- `exploration_bonus = c * sqrt(ln(N_total + 1) / (n_skill + 1))`
  - `N_total`: total selections across all skills this iteration
  - `n_skill`: times this skill was selected this iteration
  - `c`: exploration constant (default 0.15, configurable)

UCB1-inspired: under-explored skills get a higher bonus. As a skill is selected more, bonus shrinks and exploitation dominates.

---

## Part C: Selection Tracker

Add `SelectionTracker` to `skill_agents_grpo/query.py`:

- `increment(skill_id)`: called after each selection
- `get_counts(skill_id) -> (n_skill, n_total)`: returns counts for UCB
- `reset()`: called at start of each co-evolution iteration
- Stored as `Dict[str, int]` on `SkillQueryEngine`

---

## Part D: Use skill_score in Bank Maintenance

- **Retirement** (`sub_episode_evaluator.py`): retire when `skill_score < 0.2` AND `n_instances >= 5`
- **Protocol refinement** (`pipeline.py`): trigger on `skill_score < 0.35` instead of `success_rate < 0.4`
- **Curator prompt** (`llm_curator.py`): include `skill_score` in action details

---

## Part E: Recency Decay

Prevent stale skills from dominating:

- `recency_factor = min(1.0, n_recent / 3)` where `n_recent` = sub-episodes from last 2 iterations
- `effective_score = 0.8 * skill_score + 0.2 * recency_factor`

---

## Files to Modify

| File | Change |
|------|--------|
| `skill_agents_grpo/stage3_mvp/schemas.py` + mirror | Add `compute_skill_score()`, `expected_reward`, `reusability` properties |
| `skill_agents_grpo/query.py` + mirror | Add `SelectionTracker`, update `_compute_confidence()` with UCB |
| `skill_agents_grpo/quality/sub_episode_evaluator.py` | Update retirement logic to use `skill_score` |
| `skill_agents_grpo/pipeline.py` | Update `refine_low_pass_protocols()` threshold |
| `skill_agents_grpo/bank_maintenance/llm_curator.py` | Add `skill_score` to curator prompt |

## Backward Compatibility

- `compute_skill_score()` uses only existing data (sub_episodes, success_rate)
- `SelectionTracker` initializes empty — first iteration gives max exploration bonus to all skills
- Old skill banks load fine — falls back gracefully when fields are missing
- Weights are configurable
