"""Frozen neural paired-uplift ensembles over CLEVRER proof receipts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .clevrer_proof_receipts import PROOF_FEATURE_NAMES
from .video_recovery_cate import FEATURE_NAMES


V14_FEATURE_NAMES = FEATURE_NAMES + PROOF_FEATURE_NAMES


@dataclass(frozen=True)
class FrozenTanhModel:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    input_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[float, ...]
    output_bias: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], feature_count: int) -> "FrozenTanhModel":
        model = cls(
            tuple(map(float, value["feature_mean"])),
            tuple(map(float, value["feature_scale"])),
            tuple(tuple(map(float, row)) for row in value["input_weights"]),
            tuple(map(float, value["hidden_bias"])),
            tuple(map(float, value["output_weights"])),
            float(value["output_bias"]),
        )
        model.validate(feature_count)
        return model

    def validate(self, feature_count: int) -> None:
        if len(self.feature_mean) != feature_count or len(self.feature_scale) != feature_count:
            raise ValueError("proof grounder scaler shape mismatch")
        matrix = np.asarray(self.input_weights, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != feature_count:
            raise ValueError("proof grounder input shape mismatch")
        if matrix.shape[1] != len(self.hidden_bias) or matrix.shape[1] != len(self.output_weights):
            raise ValueError("proof grounder hidden shape mismatch")
        if any(value <= 0 or not math.isfinite(value) for value in self.feature_scale):
            raise ValueError("proof grounder feature scales must be positive and finite")

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        feature_count = len(self.feature_mean)
        self.validate(feature_count)
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != feature_count:
            raise ValueError("proof grounder inference shape mismatch")
        standardized = (
            matrix - np.asarray(self.feature_mean, dtype=np.float64)
        ) / np.asarray(self.feature_scale, dtype=np.float64)
        hidden = np.tanh(
            standardized @ np.asarray(self.input_weights, dtype=np.float64)
            + np.asarray(self.hidden_bias, dtype=np.float64)
        )
        return hidden @ np.asarray(self.output_weights, dtype=np.float64) + self.output_bias


@dataclass(frozen=True)
class FrozenTanhEnsemble:
    models: tuple[FrozenTanhModel, ...]
    feature_count: int

    @classmethod
    def from_list(
        cls, values: Sequence[Mapping[str, Any]], feature_count: int,
    ) -> "FrozenTanhEnsemble":
        if not values:
            raise ValueError("proof grounder ensemble must be nonempty")
        return cls(
            tuple(FrozenTanhModel.from_dict(value, feature_count) for value in values),
            feature_count,
        )

    def predict_heads(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        return np.vstack([model.predict(features) for model in self.models])

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        return self.predict_heads(features).mean(axis=0)


def artifact_content_hash(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "artifact_sha256"}
    payload = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_v14_artifact(value: Mapping[str, Any]) -> tuple[
    FrozenTanhEnsemble, FrozenTanhEnsemble, FrozenTanhEnsemble, float
]:
    if value.get("status") != "FROZEN_CLEVRER_PROOF_PAIRED_UPLIFT_ENSEMBLE":
        raise ValueError("unexpected V14 proof artifact status")
    if tuple(value.get("feature_names", ())) != V14_FEATURE_NAMES:
        raise ValueError("V14 proof feature schema mismatch")
    if int(value.get("base_feature_count", -1)) != len(FEATURE_NAMES):
        raise ValueError("V14 base feature boundary mismatch")
    if artifact_content_hash(value) != value.get("artifact_sha256"):
        raise ValueError("V14 artifact content hash mismatch")
    threshold = float(value["decision_threshold"])
    if not math.isfinite(threshold):
        raise ValueError("V14 decision threshold must be finite")
    expected_heads = len(value["model_seeds"])
    groups = (
        ("proof_models", len(V14_FEATURE_NAMES)),
        ("base_only_control_models", len(FEATURE_NAMES)),
        ("permuted_uplift_control_models", len(V14_FEATURE_NAMES)),
    )
    ensembles = []
    for key, feature_count in groups:
        models = value[key]
        if len(models) != expected_heads:
            raise ValueError(f"V14 ensemble width mismatch: {key}")
        ensembles.append(FrozenTanhEnsemble.from_list(models, feature_count))
    return ensembles[0], ensembles[1], ensembles[2], threshold


__all__ = [
    "FrozenTanhEnsemble",
    "FrozenTanhModel",
    "V14_FEATURE_NAMES",
    "artifact_content_hash",
    "validate_v14_artifact",
]
