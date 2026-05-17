"""Reward shaping for skill_selection GRPO.

Three mechanisms layered on top of the base ``skill_selection_reward``:

1. **Skill chain reward** — delayed credit assignment: after selecting a
   skill, track cumulative score delta over the next *K* steps and
   retroactively boost/penalise that decision.  Teaches the LoRA which
   skills *actually produce score* (not just which ones look good on paper).

2. **Exploration bonus** — counteracts the SFT positional bias (LoRA
   defaulting to SKILL: 1) by giving a small bonus when a non-default
   position is chosen AND produces positive env reward.

3. **Anti-collapse penalty** — penalises consecutive selection of the
   same candidate position, encouraging the LoRA to condition on state
   rather than memorising a fixed position.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

CHAIN_REWARD_HORIZON: int = 5
CHAIN_REWARD_GAMMA: float = 0.9
CHAIN_REWARD_WEIGHT: float = 0.20

EXPLORATION_BONUS: float = 0.08
DEFAULT_POSITION_PENALTY: float = 0.03

ANTI_COLLAPSE_THRESHOLD: int = 3
ANTI_COLLAPSE_PENALTY_PER_EXTRA: float = 0.04
ANTI_COLLAPSE_MAX_PENALTY: float = 0.15

SKILL_DIVERSITY_WINDOW: int = 12
SKILL_DIVERSITY_THRESHOLD: float = 0.70
SKILL_DIVERSITY_PENALTY: float = 0.12
SKILL_DIVERSITY_BONUS: float = 0.06

# ── Telemetry ──────────────────────────────────────────────────────────

_SHAPING_STATS: Dict[str, int] = {
    "chain_applied": 0,
    "chain_positive": 0,
    "chain_negative": 0,
    "exploration_bonus": 0,
    "default_penalty": 0,
    "anti_collapse": 0,
}


def reset_shaping_stats() -> None:
    for k in _SHAPING_STATS:
        _SHAPING_STATS[k] = 0


def get_shaping_stats() -> Dict[str, int]:
    return dict(_SHAPING_STATS)


# ── Skill Chain Reward ─────────────────────────────────────────────────

@dataclass
class _PendingChainReward:
    """A skill-selection decision awaiting delayed chain reward."""
    grpo_record_idx: int
    step_selected: int
    score_at_selection: float
    remaining_horizon: int = CHAIN_REWARD_HORIZON


class SkillChainTracker:
    """Track pending chain rewards across episode steps.

    Usage inside episode loop::

        tracker = SkillChainTracker()

        # When a skill is selected (GRPO record appended):
        tracker.register(grpo_idx=len(grpo_records)-1, step=step_count, current_score=total_reward)

        # Every step:
        tracker.observe_step(current_score=total_reward)

        # At episode end:
        tracker.finalize(grpo_records, current_score=total_reward)
    """

    def __init__(self, horizon: int = CHAIN_REWARD_HORIZON, gamma: float = CHAIN_REWARD_GAMMA):
        self._horizon = horizon
        self._gamma = gamma
        self._pending: List[_PendingChainReward] = []
        self._step_scores: List[float] = []

    def register(self, grpo_idx: int, step: int, current_score: float) -> None:
        self._pending.append(_PendingChainReward(
            grpo_record_idx=grpo_idx,
            step_selected=step,
            score_at_selection=current_score,
        ))

    def observe_step(self, current_score: float) -> None:
        self._step_scores.append(current_score)

    def finalize(
        self,
        grpo_records: list,
        current_score: float,
        weight: float = CHAIN_REWARD_WEIGHT,
    ) -> int:
        """Apply chain rewards to pending GRPO records. Returns count applied."""
        n_applied = 0
        for pending in self._pending:
            score_delta = current_score - pending.score_at_selection

            steps_after = len(self._step_scores) - pending.step_selected
            if steps_after <= 0:
                continue

            discounted = 0.0
            for t in range(min(steps_after, self._horizon)):
                step_idx = pending.step_selected + t
                if step_idx + 1 < len(self._step_scores):
                    delta_t = self._step_scores[step_idx + 1] - self._step_scores[step_idx]
                else:
                    delta_t = 0.0
                discounted += (self._gamma ** t) * delta_t

            max_abs = max(abs(discounted), 1.0)
            chain_r = discounted / max_abs
            chain_r = max(-1.0, min(1.0, chain_r))

            idx = pending.grpo_record_idx
            if 0 <= idx < len(grpo_records):
                rec = grpo_records[idx]
                old_reward = rec.reward
                bonus = weight * chain_r
                rec.reward = old_reward + bonus

                if rec.metadata is None:
                    rec.metadata = {}
                rec.metadata["chain_reward"] = round(bonus, 4)
                rec.metadata["chain_score_delta"] = round(float(score_delta), 2)

                _SHAPING_STATS["chain_applied"] += 1
                if bonus > 0:
                    _SHAPING_STATS["chain_positive"] += 1
                elif bonus < 0:
                    _SHAPING_STATS["chain_negative"] += 1
                n_applied += 1

        self._pending.clear()
        return n_applied


# ── Exploration Bonus ──────────────────────────────────────────────────

def exploration_bonus(
    chosen_idx: int,
    env_reward: float,
    n_candidates: int,
    bonus: float = EXPLORATION_BONUS,
    penalty: float = DEFAULT_POSITION_PENALTY,
) -> float:
    """Reward shaping to break SFT positional bias.

    - Non-default position (idx > 0) + positive env reward → small bonus
    - Default position (idx == 0) + zero env reward → small penalty
    - All other cases → 0.0
    """
    if n_candidates < 2:
        return 0.0

    if chosen_idx > 0 and env_reward > 0:
        _SHAPING_STATS["exploration_bonus"] += 1
        return bonus
    elif chosen_idx == 0 and env_reward <= 0:
        _SHAPING_STATS["default_penalty"] += 1
        return -penalty
    return 0.0


# ── Anti-Collapse Penalty ──────────────────────────────────────────────

class PositionCollapseTracker:
    """Track consecutive same-position selections and apply penalties."""

    def __init__(self, threshold: int = ANTI_COLLAPSE_THRESHOLD):
        self._threshold = threshold
        self._recent_positions: Deque[int] = deque(maxlen=10)

    def record(self, chosen_idx: int) -> None:
        self._recent_positions.append(chosen_idx)

    def penalty(self) -> float:
        """Compute anti-collapse penalty based on recent position history."""
        if len(self._recent_positions) < self._threshold:
            return 0.0

        tail = list(self._recent_positions)[-self._threshold:]
        if len(set(tail)) > 1:
            return 0.0

        consecutive = 0
        pos = self._recent_positions[-1]
        for p in reversed(self._recent_positions):
            if p == pos:
                consecutive += 1
            else:
                break

        if consecutive < self._threshold:
            return 0.0

        extra = consecutive - self._threshold
        pen = ANTI_COLLAPSE_PENALTY_PER_EXTRA * (1 + extra)
        pen = min(pen, ANTI_COLLAPSE_MAX_PENALTY)
        _SHAPING_STATS["anti_collapse"] += 1
        return -pen


# ── Multi-step Skill Continuation Gate ─────────────────────────────────

def premature_switch_penalty(
    protocol_completion_ratio: float,
    reselect_reason: str,
    min_completion: float = 0.5,
    penalty: float = 0.10,
) -> float:
    """Penalise switching away from a skill before it has a chance to work.

    Only applies when the switch reason is ``zero_reward_stall`` (the LoRA
    gave up too early) AND the protocol is less than *min_completion* done.
    Does NOT penalise switches triggered by abort criteria or success —
    those are legitimate lifecycle events.
    """
    if reselect_reason != "zero_reward_stall":
        return 0.0
    if protocol_completion_ratio >= min_completion:
        return 0.0
    return -penalty


# ── Skill-ID Diversity Bonus / Penalty ────────────────────────────────

class SkillDiversityTracker:
    """Penalise monopoly of a single skill_id; reward trying under-used skills.

    Solves the Candy Crush collapse where one discovered skill captured
    84% of all action steps.  Tracks a sliding window of recent skill_id
    selections and applies:
      - penalty when one skill exceeds ``threshold`` share of the window
      - bonus when a skill with < 20% historical share is selected
    """

    def __init__(
        self,
        window: int = SKILL_DIVERSITY_WINDOW,
        threshold: float = SKILL_DIVERSITY_THRESHOLD,
    ):
        self._window = window
        self._threshold = threshold
        self._recent: Deque[str] = deque(maxlen=window)

    def record_and_shape(self, skill_id: str) -> float:
        """Record *skill_id* and return a diversity shaping term."""
        if not skill_id:
            return 0.0

        from collections import Counter
        counts = Counter(self._recent)
        total = len(self._recent)

        self._recent.append(skill_id)

        if total < 4:
            return 0.0

        share = counts.get(skill_id, 0) / total
        if share >= self._threshold:
            _SHAPING_STATS["anti_collapse"] += 1
            return -SKILL_DIVERSITY_PENALTY
        elif share <= 0.15 and total >= self._window // 2:
            _SHAPING_STATS["exploration_bonus"] += 1
            return SKILL_DIVERSITY_BONUS
        return 0.0
