from __future__ import annotations

import pytest

from motif_transfer.real_source_relation_causal_v20 import (
    ACTION_FEATURE_NAMES,
    UTILITY_FEATURE_NAMES,
    action_causal_features,
    linear_probability,
    linear_value,
    score_relation_decision,
    utility_features,
)
from motif_transfer.relation_edge_value_v13 import FEATURE_NAMES
from scripts.train_real_source_relation_causal_v20 import _partition_calibration


def _ledger() -> dict:
    return {
        "goal_spec": {
            "goal_object_type": "apple",
            "target_receptacle_type": "cabinet",
            "required_count": 2,
        },
        "completed_objects": ["apple 1"],
    }


def test_causal_action_features_are_entity_conditioned_and_pre_action() -> None:
    values = action_causal_features(
        action="move apple 2 to cabinet 1",
        grounded_scores={
            "policy": 0.4,
            "completion": 0.8,
            "binding": 0.9,
            "applicability": 0.7,
        },
        ledger=_ledger(),
        history=("take apple 2 from table 1",),
        step=12,
        max_steps=60,
    )
    assert tuple(values) == ACTION_FEATURE_NAMES
    assert values["remaining_slots_fraction"] == 0.5
    assert values["goal_object_token_match"] == 1.0
    assert values["goal_receptacle_token_match"] == 1.0
    assert values["verb_move"] == 1.0


def test_non_goal_action_has_no_entity_binding() -> None:
    values = action_causal_features(
        action="close drawer 1",
        grounded_scores={},
        ledger=_ledger(),
        history=(),
        step=0,
        max_steps=60,
    )
    assert values["goal_object_token_match"] == 0.0
    assert values["goal_receptacle_token_match"] == 0.0
    assert values["verb_close"] == 1.0


def test_utility_features_include_causal_effect_margin() -> None:
    action = action_causal_features(
        action="move apple 2 to cabinet 1",
        grounded_scores={}, ledger=_ledger(), history=(), step=1, max_steps=60,
    )
    fallback = action_causal_features(
        action="look",
        grounded_scores={}, ledger=_ledger(), history=(), step=1, max_steps=60,
    )
    base = {name: 0.0 for name in FEATURE_NAMES}
    values = utility_features(
        base_features=base,
        source_effect_probability=0.9,
        fallback_effect_probability=0.2,
        source_action_features=action,
        fallback_action_features=fallback,
    )
    assert tuple(values) == UTILITY_FEATURE_NAMES
    assert values["causal_effect_margin"] == pytest.approx(0.7)


def test_serialized_linear_heads_score_without_sklearn_runtime() -> None:
    probability_features = {name: 0.0 for name in ACTION_FEATURE_NAMES}
    probability_model = {
        "feature_names": list(ACTION_FEATURE_NAMES),
        "means": [0.0] * len(ACTION_FEATURE_NAMES),
        "scales": [1.0] * len(ACTION_FEATURE_NAMES),
        "intercept": 0.0,
        "weights": [0.0] * len(ACTION_FEATURE_NAMES),
    }
    assert linear_probability(probability_model, probability_features) == 0.5
    value_features = {name: 0.0 for name in UTILITY_FEATURE_NAMES}
    value_model = {
        "feature_names": list(UTILITY_FEATURE_NAMES),
        "means": [0.0] * len(UTILITY_FEATURE_NAMES),
        "scales": [1.0] * len(UTILITY_FEATURE_NAMES),
        "intercept": 0.25,
        "weights": [0.0] * len(UTILITY_FEATURE_NAMES),
    }
    assert linear_value(value_model, value_features) == 0.25


def test_calibration_partition_is_disjoint_and_outcome_blind() -> None:
    rows = [
        {"fork_id": f"fork-{index}", "utility": float(index % 3)}
        for index in range(12)
    ]
    conformal, qualification = _partition_calibration(rows)
    changed = [dict(row, utility=-999.0) for row in reversed(rows)]
    changed_conformal, changed_qualification = _partition_calibration(changed)
    assert {row["fork_id"] for row in conformal}.isdisjoint(
        row["fork_id"] for row in qualification
    )
    assert {row["fork_id"] for row in conformal} == {
        row["fork_id"] for row in changed_conformal
    }
    assert {row["fork_id"] for row in qualification} == {
        row["fork_id"] for row in changed_qualification
    }


def test_live_relation_score_uses_serialized_causal_and_utility_heads() -> None:
    probability_model = {
        "feature_names": list(ACTION_FEATURE_NAMES),
        "means": [0.0] * len(ACTION_FEATURE_NAMES),
        "scales": [1.0] * len(ACTION_FEATURE_NAMES),
        "intercept": 0.0,
        "weights": [0.0] * len(ACTION_FEATURE_NAMES),
    }
    value_model = {
        "feature_names": list(UTILITY_FEATURE_NAMES),
        "means": [0.0] * len(UTILITY_FEATURE_NAMES),
        "scales": [1.0] * len(UTILITY_FEATURE_NAMES),
        "intercept": 0.25,
        "weights": [0.0] * len(UTILITY_FEATURE_NAMES),
    }
    decision = {
        "action": "move apple 2 to cabinet 1",
        "fallback_action": "look",
        "fallback_effect": "OTHER",
        "slot_state": {"remaining_slots": 1, "completed_count": 1},
        "target_policy_ratio": 2.0,
        "best_realization_score": 0.8,
    }
    grounded = {
        decision["action"]: {
            "policy": 0.8, "completion": 0.9,
            "binding": 0.9, "applicability": 0.9,
        },
        decision["fallback_action"]: {
            "policy": 0.4, "completion": 0.1,
            "binding": 0.1, "applicability": 0.2,
        },
    }
    result = score_relation_decision(
        candidate={
            "target_causal_effect_head": probability_model,
            "target_incremental_utility_head": value_model,
            "conformal": {"overprediction_error_quantile": 0.1},
        },
        decision=decision,
        grounded=grounded,
        ledger=_ledger(),
        history=(),
        step=12,
        max_steps=60,
        native_action_count=2,
    )
    assert result["source_causal_effect_probability"] == 0.5
    assert result["fallback_causal_effect_probability"] == 0.5
    assert result["conformal_lower_bound"] == pytest.approx(0.15)
    assert result["admitted"] is True


def test_live_relation_score_prefers_selective_risk_threshold() -> None:
    probability_model = {
        "feature_names": list(ACTION_FEATURE_NAMES),
        "means": [0.0] * len(ACTION_FEATURE_NAMES),
        "scales": [1.0] * len(ACTION_FEATURE_NAMES),
        "intercept": 0.0,
        "weights": [0.0] * len(ACTION_FEATURE_NAMES),
    }
    value_model = {
        "feature_names": list(UTILITY_FEATURE_NAMES),
        "means": [0.0] * len(UTILITY_FEATURE_NAMES),
        "scales": [1.0] * len(UTILITY_FEATURE_NAMES),
        "intercept": 0.04,
        "weights": [0.0] * len(UTILITY_FEATURE_NAMES),
    }
    decision = {
        "action": "move apple 2 to cabinet 1",
        "fallback_action": "look",
        "fallback_effect": "OTHER",
        "slot_state": {"remaining_slots": 1, "completed_count": 1},
        "target_policy_ratio": 2.0,
        "best_realization_score": 0.8,
    }
    grounded = {
        decision["action"]: {
            "policy": 0.8, "completion": 0.9,
            "binding": 0.9, "applicability": 0.9,
        },
        decision["fallback_action"]: {
            "policy": 0.4, "completion": 0.1,
            "binding": 0.1, "applicability": 0.2,
        },
    }
    result = score_relation_decision(
        candidate={
            "target_causal_effect_head": probability_model,
            "target_incremental_utility_head": value_model,
            "conformal": {"overprediction_error_quantile": 1.0},
            "selective_risk_calibration": {"admission_threshold": 0.025},
        },
        decision=decision,
        grounded=grounded,
        ledger=_ledger(),
        history=(),
        step=12,
        max_steps=60,
        native_action_count=2,
    )
    assert result["admission_threshold"] == 0.025
    assert result["admission_authority"] == (
        "OUT_OF_FOLD_SELECTIVE_RISK_THRESHOLD"
    )
    assert result["admitted"] is True
