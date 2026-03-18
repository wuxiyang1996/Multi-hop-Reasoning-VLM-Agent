# Skill Evaluation Redesign

**Status**: Implemented
**Goal**: Redesign skill evaluation and GRPO rewards. Keep it simple — each
adapter's reward matches its role, skill quality is measured by what matters,
new skills get a fair chance.

---

## Design Philosophy

- **skill_selection / action_taking** — reward = actual environment reward.
  Skill selection also encourages exploring under-used skills.
- **segment / contract** — reward = reliability (accurate output, parseable
  format). Already well-designed; no changes needed.
- **curator** — reward = does curation improve bank quality? Show skill quality
  info in the prompt so the LLM can make better decisions.

Skill quality has three factors:

1. **Reward contribution** — does using this skill lead to high env reward?
2. **Usage frequency** — how often does the agent call this skill?
3. **Exploration** — new skills should get a chance before being judged.

---

## Part A: Skill Score

Add `compute_skill_score()` on the `Skill` dataclass in
`skill_agents_grpo/stage3_mvp/schemas.py`:

```python
def compute_skill_score(self, n_selections: int = 0,
                        n_total_selections: int = 0,
                        n_bank_skills: int = 1) -> float:
    # (1) reward contribution: mean quality_score across sub-episodes
    #     quality_score is already min-max normalized to [0, 1] per skill
    #     by score_all_sub_episodes() in sub_episode_evaluator.py
    if not self.sub_episodes:
        return 0.5  # neutral default for brand-new skills
    r_reward = mean(ep.quality_score for ep in self.sub_episodes)

    # (2) usage: how often the agent picks this skill
    fair_share = max(1, n_total_selections) / max(1, n_bank_skills)
    r_usage = min(1.0, n_selections / max(1, fair_share))

    return 0.5 * r_reward + 0.5 * r_usage
```

Two components, equal weight, both in [0, 1]. `quality_score` is used instead
of raw `cumulative_reward` because env reward scales vary across games (Tetris
scores in thousands, Avalon is 0/1). `quality_score` is already min-max
normalized within each skill by `score_all_sub_episodes()` — no extra
normalization needed. New skills with no sub-episodes return 0.5 (benefit of
the doubt).

---

## Part B: Exploration in Skill Selection (UCB)

Modify `_compute_confidence()` in `skill_agents_grpo/query.py`:

```python
def _compute_confidence(self, skill, relevance, applicability, pass_rate,
                        n_skill=0, n_total=0):
    exploit = (0.40 * relevance
             + 0.30 * min(1.0, applicability)
             + 0.30 * pass_rate)

    # UCB exploration: under-selected skills get a bonus
    explore = 0.15 * sqrt(log(n_total + 1) / (n_skill + 1))

    return exploit + explore
```

Simple UCB1. No `skill_score` inside confidence — keep selection about
relevance/applicability/pass_rate for the current context, with an exploration
nudge. `skill_score` is used elsewhere (bank maintenance, curator).

---

## Part C: Selection Tracker

Lightweight counter on `SkillQueryEngine`:

```python
class SelectionTracker:
    def __init__(self):
        self._counts: Dict[str, int] = defaultdict(int)
        self._total: int = 0

    def increment(self, skill_id: str):
        self._counts[skill_id] += 1
        self._total += 1

    def get(self, skill_id: str) -> tuple[int, int]:
        return self._counts[skill_id], self._total

    def reset(self):
        self._counts.clear()
        self._total = 0
```

Called: `increment()` after each skill selection, `reset()` at iteration start.

---

## Part D: GRPO Reward Adjustments

### skill_selection — delayed reward at skill-switch time

Replace raw per-step `float(env_reward)` with a **delayed, per-skill reward**
assigned retroactively when the skill tracker triggers a reselection.

```
r_env_norm = min(1.0, max(0.0, cumulative_env_reward / steps_on_skill))
r_follow_norm = min(1.0, max(0.0, cumulative_r_follow / steps_on_skill))

r = 0.5 * r_env_norm + 0.5 * r_follow_norm
```

Two components, equal weight:
- `r_env_norm` — did env reward accumulate while this skill was active?
  (general progress signal, normalized per-step so it's game-agnostic)
- `r_follow_norm` — did the skill's contract predicates (`eff_add`) actually
  get satisfied? (skill-specific attribution via `RewardComputer.r_follow`)

`r_follow` is the key differentiator: it directly measures whether the skill
*itself* helped, not just whether good things happened to coincide. A skill
whose `eff_add` predicates get satisfied earns high `r_follow`; an irrelevant
skill earns ~0.

Both values are already tracked on `_SkillTracker` (`reward_on_skill`) and
`RewardComputer` (`_cumulative.r_follow`). At skill-switch time, read both
accumulators, normalize by steps, and assign to the skill_selection `GRPORecord`.

**Files**: `trainer/coevolution/episode_runner.py` (assign reward at switch),
`decision_agents/reward_func.py` (expose cumulative r_follow per skill period).

### action_taking — no change

Keep raw env reward. Correct signal for action selection.

### segment — no change

Current reward already handles reliability (margins, confidence) and penalizes
bad output (unparseable = 0 through fallback). Well-designed as-is.

### contract — no change

Current reward already handles reliability (pass rate, literal success,
sparsity) and penalizes bad format (null/empty = 0 or 0.05). Well-designed
as-is.

### curator — show skill_score in prompt, keep reward

Don't change the reward formula. Instead, add `skill_score` to the curator
prompt so the LLM has quality information when deciding approve/veto/defer.
The existing `quality_delta` reward already measures "did curation help?"

---

## Part E: Use skill_score in Bank Maintenance

Three simple substitutions:

- **Retirement** (`sub_episode_evaluator.py`): retire when `skill_score < 0.2`
  AND `n_sub_episodes >= 5`
- **Protocol refinement** (`pipeline.py`): trigger `refine_low_pass_protocols()`
  on `skill_score < 0.35` instead of `success_rate < 0.4`
- **Curator prompt** (`llm_curator.py`): append `Skill score: X.XX` to each
  action's detail block

---

## Files to Modify

| File | Change |
|------|--------|
| `skill_agents_grpo/stage3_mvp/schemas.py` + mirror | Add `compute_skill_score()` |
| `skill_agents_grpo/query.py` + mirror | Add `SelectionTracker`, add UCB term to `_compute_confidence()` |
| `trainer/coevolution/episode_runner.py` | Delayed skill_selection reward (r_env + r_follow) at skill-switch |
| `decision_agents/reward_func.py` | Expose per-skill-period cumulative r_follow |
| `skill_agents_grpo/quality/sub_episode_evaluator.py` | Retirement threshold uses `skill_score` |
| `skill_agents_grpo/pipeline.py` | Refinement threshold uses `skill_score` |
| `skill_agents_grpo/bank_maintenance/llm_curator.py` | Show `skill_score` in prompt |

## What was cut (intentionally)

- No `evidence_factor`, `reward_scale`, `novelty_cap` hyperparameters
- No format compliance terms for segment/contract (already handled by fallbacks)
- No curator reward formula change (just better prompt info)
- No recency decay (unnecessary layer)
- No `skill_score` inside `_compute_confidence()` (keep selection context-focused)
- New skills default to 0.5 instead of a decaying novelty bonus
- No `r_success` term in skill_selection reward (success signal is implicit in games)

## Future experiment (not in scope)

**Per-step skill selection with no-op**: run skill selection every step (or
every K steps) with "keep current skill" as an explicit candidate option.
Gives the model more training signal and lets it learn *when* to switch.
Trade-off: ~Kx more LLM calls per episode. Revisit once the basic feedback
loop is validated.
