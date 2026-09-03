"""Serialized target-native neural utility scorer for V24 relation transfer."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def neural_value(
    model: Mapping[str, Any], features: Mapping[str, float]
) -> float:
    """Evaluate a serialized one-hidden-layer MLP without sklearn at inference."""
    names = tuple(map(str, model["feature_names"]))
    vector = np.asarray([float(features[name]) for name in names], dtype=np.float64)
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.asarray(model["scales"], dtype=np.float64)
    input_weights = np.asarray(model["input_weights"], dtype=np.float64)
    hidden_bias = np.asarray(model["hidden_bias"], dtype=np.float64)
    output_weights = np.asarray(model["output_weights"], dtype=np.float64)
    if not (len(vector) == len(means) == len(scales) == input_weights.shape[0]):
        raise ValueError("V24 MLP input dimensions mismatch")
    hidden = np.maximum(((vector - means) / scales) @ input_weights + hidden_bias, 0.0)
    return float(hidden @ output_weights + float(model["output_bias"]))


def score_neural_risk(
    *,
    candidate: Mapping[str, Any],
    base_score: Mapping[str, Any],
) -> dict[str, Any]:
    """Add the frozen V24 neural risk decision to a pre-action V20 score."""
    predicted = neural_value(
        candidate["target_neural_utility_mlp"], base_score["utility_features"]
    )
    threshold = float(candidate["neural_risk_calibration"]["admission_threshold"])
    minimum_margin = float(
        candidate["neural_risk_calibration"]["minimum_causal_effect_margin"]
    )
    causal_margin = float(base_score["causal_effect_margin"])
    return {
        "predicted_neural_incremental_utility": predicted,
        "admission_threshold": threshold,
        "minimum_causal_effect_margin": minimum_margin,
        "causal_effect_margin": causal_margin,
        "admitted": bool(predicted > threshold and causal_margin > minimum_margin),
        "outcome_fields_consumed": False,
    }


__all__ = ["neural_value", "score_neural_risk"]
