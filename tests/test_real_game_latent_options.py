from __future__ import annotations

from motif_transfer.real_game_latent_options import (
    extract_structural_episode,
    reward_normalizer,
)


def _episode(reasoning: str) -> list[dict]:
    return [
        {
            "state": "A\nB",
            "action": "left",
            "reward": 0,
            "next_state": "A\nC",
            "done": False,
            "available_actions": ["left", "right"],
            "skill_reasoning": reasoning,
            "intentions": reasoning,
        },
        {
            "state": "A\nC",
            "action": "left",
            "reward": 2,
            "next_state": "A\nD",
            "done": True,
            "available_actions": ["left"],
            "skill_reasoning": reasoning,
            "intentions": reasoning,
        },
    ]


def test_semantic_fields_cannot_change_structural_rows() -> None:
    first = _episode("secret source semantics")
    second = _episode("completely different target-like text")
    mean, scale = reward_normalizer([first])
    first_rows = extract_structural_episode(
        first, game="game", episode_id="episode", reward_mean=mean, reward_scale=scale
    )
    second_rows = extract_structural_episode(
        second, game="game", episode_id="episode", reward_mean=mean, reward_scale=scale
    )
    assert first_rows == second_rows


def test_structural_effect_and_horizon_values_are_mechanical() -> None:
    episode = _episode("ignored")
    mean, scale = reward_normalizer([episode])
    rows = extract_structural_episode(
        episode, game="game", episode_id="episode", reward_mean=mean, reward_scale=scale
    )
    assert len(rows) == 2
    assert rows[0].effect_features[0] == 1.0
    assert rows[1].effect_features[4] == 1.0
    assert rows[1].effect_features[-1] == 1.0
    assert rows[0].horizon_values[0] < rows[0].horizon_values[1]
