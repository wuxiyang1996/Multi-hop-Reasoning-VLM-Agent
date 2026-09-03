from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


EFFECT_FEATURE_NAMES = (
    "state_changed",
    "line_change_fraction",
    "character_length_delta_tanh",
    "available_action_count_delta_tanh",
    "action_repeated",
    "normalized_immediate_reward_tanh",
    "terminated",
)
CONTEXT_FEATURE_NAMES = (
    "episode_progress_fraction",
    "remaining_budget_fraction",
    "recent_normalized_reward_tanh",
)
VALUE_HORIZONS = (1, 2, 4, 8)


@dataclass(frozen=True)
class SourceOptionRow:
    game: str
    episode_id: str
    step_index: int
    effect_features: tuple[float, ...]
    context_features: tuple[float, ...]
    horizon_values: tuple[float, ...]


def _float_reward(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def reward_normalizer(episodes: Sequence[Sequence[Mapping[str, object]]]) -> tuple[float, float]:
    rewards = np.asarray(
        [_float_reward(step.get("reward")) for episode in episodes for step in episode],
        dtype=np.float64,
    )
    if rewards.size == 0:
        raise ValueError("cannot normalize an empty source split")
    return float(np.mean(rewards)), max(float(np.std(rewards)), 1e-6)


def extract_structural_episode(
    experiences: Sequence[Mapping[str, object]],
    *,
    game: str,
    episode_id: str,
    reward_mean: float,
    reward_scale: float,
) -> tuple[SourceOptionRow, ...]:
    if not experiences:
        return ()
    normalized_rewards = np.asarray(
        [(_float_reward(step.get("reward")) - reward_mean) / reward_scale for step in experiences],
        dtype=np.float64,
    )
    rows = []
    episode_length = len(experiences)
    for index, step in enumerate(experiences):
        state = str(step.get("state") or "")
        next_state = str(step.get("next_state") or "")
        state_lines = state.splitlines()
        next_lines = next_state.splitlines()
        line_count = max(len(state_lines), len(next_lines), 1)
        changed_lines = sum(
            (state_lines[line] if line < len(state_lines) else None)
            != (next_lines[line] if line < len(next_lines) else None)
            for line in range(line_count)
        )
        available = step.get("available_actions")
        available_count = len(available) if isinstance(available, list) else 0
        if index + 1 < episode_length:
            next_available = experiences[index + 1].get("available_actions")
            next_available_count = len(next_available) if isinstance(next_available, list) else 0
        else:
            next_available_count = available_count
        previous_action = experiences[index - 1].get("action") if index else None
        recent_reward = float(np.sum(normalized_rewards[max(0, index - 4) : index]))
        effect = (
            float(state != next_state),
            changed_lines / line_count,
            float(np.tanh((len(next_state) - len(state)) / 100.0)),
            float(np.tanh((next_available_count - available_count) / 5.0)),
            float(index > 0 and str(step.get("action")) == str(previous_action)),
            float(np.tanh(normalized_rewards[index] / 3.0)),
            float(bool(step.get("done"))),
        )
        values = tuple(
            float(sum(
                (0.97**offset) * normalized_rewards[index + offset]
                for offset in range(min(horizon, episode_length - index))
            ))
            for horizon in VALUE_HORIZONS
        )
        rows.append(SourceOptionRow(
            game=game,
            episode_id=episode_id,
            step_index=index,
            effect_features=effect,
            context_features=(
                index / max(1, episode_length - 1),
                (episode_length - index) / episode_length,
                float(np.tanh(recent_reward / 3.0)),
            ),
            horizon_values=values,
        ))
    return tuple(rows)


def fit_effect_clusters(
    rows: Sequence[SourceOptionRow],
    *,
    cluster_count: int,
    seed: int,
) -> tuple[StandardScaler, KMeans]:
    if not 3 <= cluster_count <= 8:
        raise ValueError("cluster_count must be between 3 and 8")
    matrix = np.asarray([row.effect_features for row in rows], dtype=np.float64)
    scaler = StandardScaler().fit(matrix)
    model = KMeans(n_clusters=cluster_count, random_state=seed, n_init=20)
    model.fit(scaler.transform(matrix))
    return scaler, model


def option_design(
    rows: Sequence[SourceOptionRow],
    *,
    scaler: StandardScaler,
    clusterer: KMeans,
    corruption: str | None = None,
) -> np.ndarray:
    effects = scaler.transform(np.asarray([row.effect_features for row in rows]))
    option_ids = clusterer.predict(effects)
    cluster_count = clusterer.n_clusters
    if corruption == "phase_permuted":
        option_ids = (option_ids + 1) % cluster_count
    elif corruption == "within_episode_shift":
        option_ids = option_ids.copy()
        start = 0
        while start < len(rows):
            end = start + 1
            while end < len(rows) and rows[end].episode_id == rows[start].episode_id:
                end += 1
            option_ids[start:end] = np.roll(
                option_ids[start:end], max(1, (end - start) // 3)
            )
            start = end
    elif corruption is not None:
        raise ValueError(f"unknown option corruption: {corruption}")

    previous_ids = []
    for index, row in enumerate(rows):
        same_episode = index > 0 and rows[index - 1].episode_id == row.episode_id
        previous_ids.append(int(option_ids[index - 1]) if same_episode else cluster_count)
    option_one_hot = np.eye(cluster_count, dtype=np.float64)[option_ids]
    previous_one_hot = np.eye(cluster_count + 1, dtype=np.float64)[previous_ids]
    contexts = np.asarray([row.context_features for row in rows], dtype=np.float64)
    return np.column_stack((contexts, option_one_hot, previous_one_hot))


def target_values(rows: Sequence[SourceOptionRow]) -> np.ndarray:
    return np.asarray([row.horizon_values for row in rows], dtype=np.float64)


def fit_value_model(design: np.ndarray, values: np.ndarray) -> Ridge:
    return Ridge(alpha=1.0).fit(design, values)


def mse_summary(expected: np.ndarray, predicted: np.ndarray) -> dict:
    per_horizon = {
        f"h{horizon}": float(mean_squared_error(expected[:, index], predicted[:, index]))
        for index, horizon in enumerate(VALUE_HORIZONS)
    }
    return {
        "aggregate": float(np.mean(list(per_horizon.values()))),
        "per_horizon": per_horizon,
    }

