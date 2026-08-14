"""Target-native paired-uplift grounding for video recovery decisions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "is_explanatory",
    "is_predictive",
    "is_counterfactual",
    "choice_count",
    "choice_count_fraction",
    "question_has_negate",
    "question_has_counterfactual",
    "question_has_exit",
    "question_collision_count",
    "question_program_length_fraction",
    "explicit_error_fraction",
    "explicit_yes_fraction",
    "trajectory_yes_fraction",
    "answer_disagreement_fraction",
    "explicit_bit_0",
    "explicit_bit_1",
    "explicit_bit_2",
    "explicit_bit_3",
    "trajectory_bit_0",
    "trajectory_bit_1",
    "trajectory_bit_2",
    "trajectory_bit_3",
    "disagreement_bit_0",
    "disagreement_bit_1",
    "disagreement_bit_2",
    "disagreement_bit_3",
    "mean_choice_collision_count",
    "mean_choice_program_length_fraction",
)


def build_features(
    *,
    family: str,
    question_program: Sequence[str],
    choice_programs: Sequence[Sequence[str]],
    explicit_answer: str,
    trajectory_answer: str,
    explicit_error_count: int,
) -> tuple[float, ...]:
    if family not in {"explanatory", "predictive", "counterfactual"}:
        raise ValueError(f"unsupported CLEVRER causal family: {family}")
    if not explicit_answer or len(explicit_answer) != len(trajectory_answer):
        raise ValueError("paired native answers must have equal nonzero length")
    if any(value not in "01" for value in explicit_answer + trajectory_answer):
        raise ValueError("paired native answers must be binary vectors")
    count = len(explicit_answer)
    if len(choice_programs) != count or not 0 <= explicit_error_count <= count:
        raise ValueError("choice programs and executor errors must align")
    explicit_bits = [int(index < count and explicit_answer[index] == "1") for index in range(4)]
    trajectory_bits = [int(index < count and trajectory_answer[index] == "1") for index in range(4)]
    disagreement_bits = [
        int(index < count and explicit_answer[index] != trajectory_answer[index])
        for index in range(4)
    ]
    features = (
        float(family == "explanatory"),
        float(family == "predictive"),
        float(family == "counterfactual"),
        float(count),
        count / 4.0,
        float("negate" in question_program),
        float("filter_counterfact" in question_program),
        float("filter_out" in question_program),
        float(question_program.count("filter_collision")),
        len(question_program) / 25.0,
        explicit_error_count / count,
        explicit_answer.count("1") / count,
        trajectory_answer.count("1") / count,
        sum(a != b for a, b in zip(explicit_answer, trajectory_answer)) / count,
        *map(float, explicit_bits),
        *map(float, trajectory_bits),
        *map(float, disagreement_bits),
        sum(choice.count("filter_collision") for choice in choice_programs) / count,
        sum(len(choice) for choice in choice_programs) / (count * 20.0),
    )
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError("video recovery CATE feature contract drift")
    return tuple(features)


@dataclass(frozen=True)
class FrozenTanhRegressor:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    input_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[float, ...]
    output_bias: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenTanhRegressor":
        model = cls(
            tuple(map(float, value["feature_mean"])),
            tuple(map(float, value["feature_scale"])),
            tuple(tuple(map(float, row)) for row in value["input_weights"]),
            tuple(map(float, value["hidden_bias"])),
            tuple(map(float, value["output_weights"])),
            float(value["output_bias"]),
        )
        model.validate()
        return model

    def validate(self) -> None:
        size = len(FEATURE_NAMES)
        if len(self.feature_mean) != size or len(self.feature_scale) != size:
            raise ValueError("CATE scaler shape mismatch")
        matrix = np.asarray(self.input_weights, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != size:
            raise ValueError("CATE input weight shape mismatch")
        if matrix.shape[1] != len(self.hidden_bias) or matrix.shape[1] != len(self.output_weights):
            raise ValueError("CATE hidden width mismatch")
        if any(value <= 0 for value in self.feature_scale):
            raise ValueError("CATE feature scales must be positive")

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        self.validate()
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError("CATE inference feature shape mismatch")
        standardized = (
            matrix - np.asarray(self.feature_mean, dtype=np.float64)
        ) / np.asarray(self.feature_scale, dtype=np.float64)
        hidden = np.tanh(
            standardized @ np.asarray(self.input_weights, dtype=np.float64)
            + np.asarray(self.hidden_bias, dtype=np.float64)
        )
        return (
            hidden @ np.asarray(self.output_weights, dtype=np.float64)
            + self.output_bias
        )


def artifact_content_hash(value: Mapping[str, Any]) -> str:
    """Hash a frozen artifact body independently of filesystem formatting."""

    body = {key: item for key, item in value.items() if key != "artifact_sha256"}
    payload = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_frozen_artifact(value: Mapping[str, Any]) -> tuple[
    FrozenTanhRegressor, FrozenTanhRegressor, float
]:
    """Validate lineage-sensitive fields and instantiate both frozen heads."""

    if value.get("status") != "FROZEN_TARGET_NATIVE_PAIRED_UPLIFT_GROUNDER":
        raise ValueError("unexpected CATE artifact status")
    if tuple(value.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("CATE feature schema mismatch")
    expected_hash = str(value.get("artifact_sha256") or "")
    if not expected_hash or artifact_content_hash(value) != expected_hash:
        raise ValueError("CATE artifact content hash mismatch")
    threshold = float(value["decision_threshold"])
    if not math.isfinite(threshold):
        raise ValueError("CATE decision threshold must be finite")
    authentic = FrozenTanhRegressor.from_dict(value["model"])
    permuted = FrozenTanhRegressor.from_dict(value["permuted_control_model"])
    return authentic, permuted, threshold


__all__ = [
    "FEATURE_NAMES",
    "FrozenTanhRegressor",
    "artifact_content_hash",
    "build_features",
    "validate_frozen_artifact",
]
