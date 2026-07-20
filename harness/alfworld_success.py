"""Outcome scorer for ALFWorld few-shot transfer probes."""

from __future__ import annotations

from typing import Any, Callable

from data_structure.extensions.skill_episode import SkillEpisode
from harness.gymv_success import register_success_fn


def make_alfworld_success_fn(
    *,
    pass_rate_threshold: float = 1.0,
    require_episode_success: bool = True,
) -> Callable[[SkillEpisode, Any], float]:
    """Require a real demo expectation and sufficient environment reward.

    Requiring ``demo.expected`` prevents a deterministic dry-run adapter from
    accidentally granting ALFWorld verification when no target-domain
    demonstration was supplied.
    """
    del pass_rate_threshold

    def _score(episode: SkillEpisode, demo: Any) -> float:
        outcome = episode.outcome
        if outcome is None:
            return 0.0
        if require_episode_success and (
            not outcome.success or not outcome.contract_satisfied
        ):
            return 0.0
        expected = getattr(demo, "expected", None) or {}
        if not expected:
            return 0.0
        min_reward = float(expected.get("min_reward", 1.0))
        score = float(outcome.score or 0.0)
        return 1.0 if score >= min_reward else 0.0

    return _score


register_success_fn("alfworld", make_alfworld_success_fn)


__all__ = ["make_alfworld_success_fn"]
