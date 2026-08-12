from __future__ import annotations

import pytest

from motif_transfer.real_source_relation_neural_risk_v24 import (
    neural_value,
    score_neural_risk,
)


def test_serialized_relu_mlp() -> None:
    model = {
        "feature_names": ["x", "y"],
        "means": [1.0, 2.0],
        "scales": [2.0, 1.0],
        "input_weights": [[2.0, -1.0], [1.0, 3.0]],
        "hidden_bias": [0.0, -1.0],
        "output_weights": [0.5, 2.0],
        "output_bias": 0.25,
    }
    # standardized input [1, 1] -> hidden relu [3, 1] -> 3.75
    assert neural_value(model, {"x": 3.0, "y": 3.0}) == pytest.approx(3.75)


def test_admission_requires_neural_value_and_causal_margin() -> None:
    candidate = {
        "target_neural_utility_mlp": {
            "feature_names": ["x"],
            "means": [0.0],
            "scales": [1.0],
            "input_weights": [[1.0]],
            "hidden_bias": [0.0],
            "output_weights": [1.0],
            "output_bias": 0.0,
        },
        "neural_risk_calibration": {
            "admission_threshold": 0.5,
            "minimum_causal_effect_margin": 0.2,
        },
    }
    admitted = score_neural_risk(
        candidate=candidate,
        base_score={"utility_features": {"x": 1.0}, "causal_effect_margin": 0.8},
    )
    rejected = score_neural_risk(
        candidate=candidate,
        base_score={"utility_features": {"x": 1.0}, "causal_effect_margin": 0.1},
    )
    assert admitted["admitted"] is True
    assert rejected["admitted"] is False
    assert admitted["outcome_fields_consumed"] is False
