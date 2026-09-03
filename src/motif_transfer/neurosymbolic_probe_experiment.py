"""Leakage-resistant operational examples for a neural-symbolic probe pilot."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


FEATURE_NAMES = (
    "log_native_action_count",
    "step_over_100",
    "previous_admissible_set_changed",
    "previous_positive_native_reward",
    "previous_terminal",
    "recent_admissible_change_rate_3",
    "recent_positive_reward_rate_3",
    "previous_action_repeated",
    "history_unique_action_ratio",
)

LABEL_NAMES = (
    "admissible_set_changed",
    "positive_native_reward",
    "terminal",
)


@dataclass(frozen=True)
class OperationalTransition:
    episode_id: str
    step: int
    before_native_actions: tuple[str, ...]
    action: str
    after_native_actions: tuple[str, ...]
    reward: float
    terminal: bool

    def validate(self) -> None:
        if self.step < 0:
            raise ValueError("operational transition step must be non-negative")
        if self.action not in self.before_native_actions:
            raise ValueError("operational transition action was not native-admissible")
        if not math.isfinite(self.reward):
            raise ValueError("operational transition reward must be finite")


@dataclass(frozen=True)
class OperationalProbeExample:
    episode_id: str
    step: int
    features: tuple[float, ...]
    labels: tuple[int, ...]


def _effect(transition: OperationalTransition) -> tuple[int, int, int]:
    return (
        int(
            set(transition.before_native_actions)
            != set(transition.after_native_actions)
        ),
        int(transition.reward > 0),
        int(transition.terminal),
    )


def build_operational_probe_examples(
    transitions: Sequence[OperationalTransition],
) -> tuple[OperationalProbeExample, ...]:
    """Build pre-action features without domain, action-token or future leakage."""

    rows = tuple(sorted(transitions, key=lambda row: (row.episode_id, row.step)))
    by_episode: dict[str, list[OperationalTransition]] = {}
    for row in rows:
        row.validate()
        by_episode.setdefault(row.episode_id, []).append(row)
    examples = []
    for episode_id, episode_rows in sorted(by_episode.items()):
        steps = [row.step for row in episode_rows]
        if steps != list(range(steps[0], steps[0] + len(steps))):
            raise ValueError("operational episode steps must be contiguous")
        prior_effects: list[tuple[int, int, int]] = []
        prior_actions: list[str] = []
        for row in episode_rows:
            previous = prior_effects[-1] if prior_effects else (0, 0, 0)
            recent = prior_effects[-3:]
            recent_affordance = (
                sum(effect[0] for effect in recent) / len(recent)
                if recent else 0.0
            )
            recent_reward = (
                sum(effect[1] for effect in recent) / len(recent)
                if recent else 0.0
            )
            previous_repeated = int(
                len(prior_actions) >= 2
                and prior_actions[-1] == prior_actions[-2]
            )
            unique_ratio = (
                len(set(prior_actions)) / len(prior_actions)
                if prior_actions else 0.0
            )
            features = (
                math.log1p(len(row.before_native_actions)),
                min(row.step, 100) / 100.0,
                float(previous[0]),
                float(previous[1]),
                float(previous[2]),
                recent_affordance,
                recent_reward,
                float(previous_repeated),
                unique_ratio,
            )
            examples.append(OperationalProbeExample(
                episode_id, row.step, features, _effect(row),
            ))
            prior_effects.append(_effect(row))
            prior_actions.append(row.action)
    return tuple(examples)


def split_source_examples(
    examples: Sequence[OperationalProbeExample],
) -> dict[str, tuple[OperationalProbeExample, ...]]:
    """Match the repository's frozen sorted-episode round-robin split."""

    names = ("train", "validation", "source_held_out")
    episode_ids = sorted({row.episode_id for row in examples})
    split_by_episode = {
        episode_id: names[index % len(names)]
        for index, episode_id in enumerate(episode_ids)
    }
    return {
        name: tuple(
            row for row in examples if split_by_episode[row.episode_id] == name
        )
        for name in names
    }


__all__ = [
    "FEATURE_NAMES",
    "LABEL_NAMES",
    "OperationalProbeExample",
    "OperationalTransition",
    "build_operational_probe_examples",
    "split_source_examples",
]
