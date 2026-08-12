from __future__ import annotations

import pytest

from motif_transfer.relation_edge_value_v13 import (
    FEATURE_NAMES,
    extract_relation_edge_features,
    fit_ridge_value_head,
    fork_utility,
    predict_relation_edge_value,
)


def _decision() -> dict:
    return {
        "action": "move potato 2 to fridge 1",
        "fallback_action": "close fridge 1",
        "fallback_effect": "POSITION",
        "target_policy_ratio": 0.9,
        "best_realization_score": 0.8,
        "slot_state": {
            "remaining_slots": 1,
            "completed_count": 1,
        },
    }


def _grounded() -> dict:
    return {
        "move potato 2 to fridge 1": {
            "policy": 0.9,
            "completion": 0.8,
            "binding": 0.95,
            "applicability": 0.85,
        },
        "close fridge 1": {
            "policy": 1.0,
            "completion": 0.2,
            "binding": 0.1,
            "applicability": 0.4,
        },
    }


def test_features_are_pre_action_and_fixed_order() -> None:
    features = extract_relation_edge_features(
        decision=_decision(),
        grounded=_grounded(),
        ledger={"required_count": 2},
        step=12,
        max_steps=60,
        native_action_count=10,
    )
    assert tuple(features) == FEATURE_NAMES
    assert features["step_fraction"] == pytest.approx(0.2)
    assert features["remaining_slots_fraction"] == pytest.approx(0.5)
    assert features["policy_margin"] == pytest.approx(-0.1)
    assert features["fallback_position"] == 1.0


def test_success_dominates_all_tie_break_terms() -> None:
    rescued = fork_utility(
        source_success=True,
        control_success=False,
        source_steps=60,
        control_steps=1,
        source_completed_fraction=0.0,
        control_completed_fraction=1.0,
        max_steps=60,
    )
    harmed = fork_utility(
        source_success=False,
        control_success=True,
        source_steps=1,
        control_steps=60,
        source_completed_fraction=1.0,
        control_completed_fraction=0.0,
        max_steps=60,
    )
    assert rescued > 0.0
    assert harmed < 0.0


def test_ridge_head_learns_feature_direction_deterministically() -> None:
    rows = []
    for value in (-2.0, -1.0, 1.0, 2.0):
        features = {name: 0.0 for name in FEATURE_NAMES}
        features["policy_margin"] = value
        rows.append({"features": features, "utility": value})
    first = fit_ridge_value_head(rows)
    second = fit_ridge_value_head(rows)
    assert first == second
    positive = dict(rows[-1]["features"])
    negative = dict(rows[0]["features"])
    assert predict_relation_edge_value(first, positive) > 0.0
    assert predict_relation_edge_value(first, negative) < 0.0
