from __future__ import annotations

from scripts.run_webshop_intervention_forks_v7 import _discover_opportunities


def _source() -> dict:
    return {
        "artifact_sha256": "source",
        "cluster_count": 2,
        "effect_feature_names": [
            "state_changed",
            "line_change_fraction",
            "character_length_delta_tanh",
            "available_action_count_delta_tanh",
            "action_repeated",
            "normalized_immediate_reward_tanh",
            "terminated",
        ],
        "effect_scaler": {"mean": [0] * 7, "scale": [1] * 7},
        "cluster_centers": [
            [0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ],
        "value_model": {
            "coefficients": [
                [0] * 8,
                [0] * 8,
                [0] * 8,
                [0, 0, 0, 0, 1, 0, 0, 0],
            ],
            "intercept": [0, 0, 0, 0],
        },
    }


def test_discovers_matched_repeat_recovery_opportunity() -> None:
    receipt = {
        "task_id": "webshop.28",
        "maximum_steps": 12,
        "steps": [{
            "step": 0,
            "before_hash": "state",
            "candidates": ["repeat", "recover"],
            "predicted_effects": [
                [0, 0, 0, 0, 0.9, 0, 0],
                [1, 0, 0, 0, 0.1, 0, 0],
            ],
            "selected_index": 0,
            "selected_action": "repeat",
            "reward": 0.0,
        }],
    }
    opportunities = _discover_opportunities(receipt, _source(), _source())
    assert len(opportunities) == 1
    assert opportunities[0]["selected_indices"]["target_only"] == 0
    assert opportunities[0]["selected_indices"]["selective_minimum_repeat"] == 1
