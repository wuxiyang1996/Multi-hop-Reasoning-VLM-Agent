"""Active video QA bridge for intervention-grounded TEST/COMMIT transfer.

The source model never receives pixels, target timestamps, or answer text.  A
target-native visual model produces evidence receipts and a small calibration
head maps those receipts to a belief over the benchmark's native answer slots.
Only the abstract decision to TEST again or COMMIT is delegated to the source
value model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .controlled_exploration_transfer import (
    FEATURE_NAMES,
    MatchedValueExample,
    ValueEnsemble,
    _build_domains_and_examples,
    fit_value_ensemble,
    marginal_value_control,
    shuffled_value_control,
)


ANSWER_SLOTS = ("A", "B", "C", "D", "E", "F")
VIDEO_CONDITIONS = (
    "target_only",
    "authentic_source_plus_target",
    "shuffled_source_plus_target",
    "source_marginal_plus_target",
    "target_native_information_gain",
)


def stable_hash(value: object) -> str:
    import json

    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def normalized_probabilities(values: Mapping[str, Any]) -> np.ndarray:
    vector = np.asarray([float(values.get(slot, 0.0)) for slot in ANSWER_SLOTS])
    if not np.all(np.isfinite(vector)) or np.any(vector < 0):
        raise ValueError("answer probabilities must be finite and nonnegative")
    total = float(np.sum(vector))
    if total <= 0:
        raise ValueError("answer probabilities have no mass")
    vector = np.clip(vector / total, 1e-6, 1.0)
    return vector / np.sum(vector)


def normalized_entropy(belief: Sequence[float]) -> float:
    vector = np.clip(np.asarray(belief, dtype=np.float64), 1e-12, 1.0)
    vector /= np.sum(vector)
    return float(-np.sum(vector * np.log(vector)) / math.log(len(vector)))


@dataclass(frozen=True)
class CalibrationRow:
    sample_id: str
    prefix_length: int
    max_tests: int
    mean_planner_score: float
    raw_probabilities: tuple[float, ...]
    answer_index: int

    def features(self) -> np.ndarray:
        probabilities = np.clip(
            np.asarray(self.raw_probabilities, dtype=np.float64), 1e-6, 1.0,
        )
        logits = np.log(probabilities)
        logits -= np.mean(logits)
        return np.concatenate((
            logits,
            np.asarray([
                self.prefix_length / max(1, self.max_tests),
                self.mean_planner_score,
            ]),
        ))


@dataclass(frozen=True)
class SoftmaxCalibrationHead:
    """Slot-symmetric neural temperature head.

    The head may soften or sharpen the target VLM belief as a function of the
    visible-evidence budget, but it cannot permute A--F or memorize a
    target-label mapping.
    """

    temperature_weights: tuple[float, float, float]

    def predict(self, features: Sequence[float]) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (len(ANSWER_SLOTS) + 2,):
            raise ValueError("calibration feature shape mismatch")
        context = np.asarray([1.0, values[-2], values[-1]])
        raw_temperature = float(context @ np.asarray(self.temperature_weights))
        temperature = 0.25 + float(np.logaddexp(0.0, raw_temperature))
        logits = values[:len(ANSWER_SLOTS)] / temperature
        logits -= np.max(logits)
        probabilities = np.exp(np.clip(logits, -30.0, 30.0))
        return probabilities / np.sum(probabilities)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "target_native_slot_symmetric_temperature_head",
            "temperature_weights": list(self.temperature_weights),
            "cannot_permute_answer_slots": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SoftmaxCalibrationHead":
        return cls(tuple(map(float, payload["temperature_weights"])))


def fit_calibration_head(
    rows: Sequence[CalibrationRow],
    *,
    seed: int,
    epochs: int = 2000,
    learning_rate: float = 0.02,
    l2: float = 0.01,
) -> SoftmaxCalibrationHead:
    """Fit a deterministic slot-symmetric target-only temperature layer."""

    if not rows:
        raise ValueError("calibration rows cannot be empty")
    matrix = np.asarray([row.features() for row in rows], dtype=np.float64)
    labels = np.asarray([row.answer_index for row in rows], dtype=np.int64)
    raw_logits = matrix[:, :len(ANSWER_SLOTS)]
    context = np.column_stack((
        np.ones(len(matrix)), matrix[:, -2], matrix[:, -1],
    ))
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=3)
    first = np.zeros_like(weights)
    second = np.zeros_like(weights)
    beta1, beta2 = 0.9, 0.999
    for epoch in range(1, epochs + 1):
        raw_temperature = context @ weights
        temperature = 0.25 + np.logaddexp(0.0, raw_temperature)
        logits = raw_logits / temperature[:, None]
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(np.clip(logits, -30.0, 30.0))
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        logit_gradient = probabilities
        logit_gradient[np.arange(len(labels)), labels] -= 1.0
        temperature_gradient = np.sum(
            logit_gradient * (-raw_logits / temperature[:, None] ** 2), axis=1,
        )
        softplus_gradient = 1.0 / (1.0 + np.exp(-np.clip(
            raw_temperature, -30.0, 30.0,
        )))
        gradient = context.T @ (temperature_gradient * softplus_gradient)
        gradient = gradient / len(labels) + l2 * weights
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient**2
        weights -= learning_rate * (first / (1.0 - beta1**epoch)) / (
            np.sqrt(second / (1.0 - beta2**epoch)) + 1e-8
        )
    return SoftmaxCalibrationHead(tuple(map(float, weights)))


@dataclass(frozen=True)
class GainRow:
    sample_id: str
    current_belief: tuple[float, ...]
    next_planner_score: float
    prefix_fraction: float
    information_gain: float
    confidence_gain: float

    def features(self) -> np.ndarray:
        return np.asarray([
            normalized_entropy(self.current_belief),
            max(self.current_belief),
            self.next_planner_score,
            self.prefix_fraction,
            1.0 - self.prefix_fraction,
        ], dtype=np.float64)


@dataclass(frozen=True)
class NeuralGainGrounder:
    input_mean: tuple[float, ...]
    input_scale: tuple[float, ...]
    hidden_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[tuple[float, ...], ...]
    output_bias: tuple[float, ...]

    def predict(
        self,
        belief: Sequence[float],
        *,
        next_planner_score: float,
        prefix_fraction: float,
    ) -> tuple[float, float]:
        row = GainRow(
            "runtime", tuple(map(float, belief)), float(next_planner_score),
            float(prefix_fraction), 0.0, 0.0,
        )
        values = (row.features() - np.asarray(self.input_mean)) / np.asarray(
            self.input_scale
        )
        hidden = np.tanh(
            values @ np.asarray(self.hidden_weights) + np.asarray(self.hidden_bias)
        )
        output = hidden @ np.asarray(self.output_weights) + np.asarray(self.output_bias)
        return tuple(map(float, np.clip(output, 0.0, 1.0)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "target_native_mlp_expected_evidence_gain",
            "input_mean": list(self.input_mean),
            "input_scale": list(self.input_scale),
            "hidden_weights": [list(row) for row in self.hidden_weights],
            "hidden_bias": list(self.hidden_bias),
            "output_weights": [list(row) for row in self.output_weights],
            "output_bias": list(self.output_bias),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NeuralGainGrounder":
        return cls(
            tuple(map(float, payload["input_mean"])),
            tuple(map(float, payload["input_scale"])),
            tuple(tuple(map(float, row)) for row in payload["hidden_weights"]),
            tuple(map(float, payload["hidden_bias"])),
            tuple(tuple(map(float, row)) for row in payload["output_weights"]),
            tuple(map(float, payload["output_bias"])),
        )


def fit_gain_grounder(
    rows: Sequence[GainRow],
    *,
    seed: int,
    hidden_units: int = 16,
    epochs: int = 1800,
    learning_rate: float = 0.02,
    l2: float = 0.01,
) -> NeuralGainGrounder:
    if not rows:
        raise ValueError("gain rows cannot be empty")
    matrix = np.asarray([row.features() for row in rows], dtype=np.float64)
    labels = np.asarray([
        (max(0.0, row.information_gain), max(0.0, row.confidence_gain))
        for row in rows
    ])
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-6] = 1.0
    inputs = (matrix - mean) / scale
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, math.sqrt(2 / inputs.shape[1]), (inputs.shape[1], hidden_units))
    b1 = np.zeros(hidden_units)
    w2 = rng.normal(0.0, math.sqrt(2 / hidden_units), (hidden_units, 2))
    b2 = np.zeros(2)
    parameters = [w1, b1, w2, b2]
    first = [np.zeros_like(value) for value in parameters]
    second = [np.zeros_like(value) for value in parameters]
    beta1, beta2 = 0.9, 0.999
    for epoch in range(1, epochs + 1):
        hidden = np.tanh(inputs @ parameters[0] + parameters[1])
        predictions = hidden @ parameters[2] + parameters[3]
        error = (predictions - labels) * (2.0 / len(labels))
        gradients = [
            inputs.T @ ((error @ parameters[2].T) * (1.0 - hidden**2))
            + l2 * parameters[0],
            np.sum((error @ parameters[2].T) * (1.0 - hidden**2), axis=0),
            hidden.T @ error + l2 * parameters[2],
            np.sum(error, axis=0),
        ]
        for index, gradient in enumerate(gradients):
            first[index] = beta1 * first[index] + (1.0 - beta1) * gradient
            second[index] = beta2 * second[index] + (1.0 - beta2) * gradient**2
            parameters[index] -= learning_rate * (
                first[index] / (1.0 - beta1**epoch)
            ) / (np.sqrt(second[index] / (1.0 - beta2**epoch)) + 1e-8)
    return NeuralGainGrounder(
        tuple(map(float, mean)), tuple(map(float, scale)),
        tuple(tuple(map(float, row)) for row in parameters[0]),
        tuple(map(float, parameters[1])),
        tuple(tuple(map(float, row)) for row in parameters[2]),
        tuple(map(float, parameters[3])),
    )


def build_source_value_models(
    controlled_config: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, ValueEnsemble]:
    """Rebuild frozen V3 source-only models without target receipts."""

    source = controlled_config["source"]
    _, _, examples = _build_domains_and_examples(
        source["train_domain_seeds"],
        surface="game",
        domain_config=controlled_config["domain"],
        state_count=int(source["states_per_domain"]),
        calibration_seed_namespace="source-train",
    )
    controls: dict[str, Sequence[MatchedValueExample]] = {
        "authentic_source_plus_target": examples,
        "shuffled_source_plus_target": shuffled_value_control(
            examples, seed=int(controlled_config["model"]["control_seed"]),
        ),
        "source_marginal_plus_target": marginal_value_control(examples),
    }
    output = {}
    for index, (condition, rows) in enumerate(controls.items()):
        model = fit_value_ensemble(
            rows, (), seed=seed + index,
            ensemble_size=int(controlled_config["model"]["ensemble_size"]),
            alpha=float(controlled_config["model"]["ridge_alpha"]),
            target_mass=1.0,
        )
        assert model is not None
        output[condition] = model
    return output


def video_action_features(
    belief: Sequence[float],
    *,
    prefix_length: int,
    max_tests: int,
    next_planner_score: float,
    gain_grounder: NeuralGainGrounder,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Return one target TEST feature and six native COMMIT features."""

    vector = np.asarray(belief, dtype=np.float64)
    vector /= np.sum(vector)
    entropy = normalized_entropy(vector)
    information_gain, confidence_gain = gain_grounder.predict(
        vector,
        next_planner_score=next_planner_score,
        prefix_fraction=prefix_length / max(1, max_tests),
    )
    test = (
        1.0,
        information_gain,
        confidence_gain,
        1.0 - 2.0 * abs(float(next_planner_score) - 0.5),
        float(np.max(vector)),
        entropy,
        (max_tests - prefix_length) / max_tests,
        0.0,
        0.0,
    )
    commits = tuple(
        (
            0.0, 0.0, 0.0, 0.0, float(np.max(vector)), entropy,
            (max_tests - prefix_length) / max_tests, float(probability), 0.0,
        )
        for probability in vector
    )
    if len(test) != len(FEATURE_NAMES):
        raise AssertionError("video feature contract drift")
    return test, commits


@dataclass(frozen=True)
class VideoPolicyDecision:
    kind: str
    answer_index: int | None
    source_abstained: bool
    predicted_test_value: float | None
    predicted_commit_value: float | None


def choose_video_action(
    belief: Sequence[float],
    *,
    condition: str,
    prefix_length: int,
    max_tests: int,
    next_planner_score: float,
    gain_grounder: NeuralGainGrounder,
    source_models: Mapping[str, ValueEnsemble],
    fallback_commit_threshold: float,
    uncertainty_scale: float,
    decision_margin: float,
    information_gain_threshold: float,
) -> VideoPolicyDecision:
    vector = np.asarray(belief, dtype=np.float64)
    best_answer = int(np.argmax(vector))
    if prefix_length >= max_tests:
        return VideoPolicyDecision("COMMIT", best_answer, False, None, None)
    test, commits = video_action_features(
        vector, prefix_length=prefix_length, max_tests=max_tests,
        next_planner_score=next_planner_score, gain_grounder=gain_grounder,
    )
    fallback_test = float(np.max(vector)) < fallback_commit_threshold
    if condition == "target_only":
        return VideoPolicyDecision(
            "TEST" if fallback_test else "COMMIT",
            None if fallback_test else best_answer,
            True, None, None,
        )
    if condition == "target_native_information_gain":
        take_test = test[1] > information_gain_threshold
        return VideoPolicyDecision(
            "TEST" if take_test else "COMMIT",
            None if take_test else best_answer,
            False, test[1], float(np.max(vector)),
        )
    model = source_models[condition]
    features = (test,) + commits
    means, deviations = model.predict(features)
    best_commit_offset = int(np.argmax(means[1:]))
    best_commit_index = best_commit_offset + 1
    gap = float(means[0] - means[best_commit_index])
    uncertainty = uncertainty_scale * math.sqrt(
        float(deviations[0] ** 2 + deviations[best_commit_index] ** 2)
    )
    if gap - uncertainty > decision_margin:
        return VideoPolicyDecision(
            "TEST", None, False, float(means[0]), float(means[best_commit_index]),
        )
    if -gap - uncertainty > decision_margin:
        return VideoPolicyDecision(
            "COMMIT", best_commit_offset, False,
            float(means[0]), float(means[best_commit_index]),
        )
    return VideoPolicyDecision(
        "TEST" if fallback_test else "COMMIT",
        None if fallback_test else best_answer,
        True, float(means[0]), float(means[best_commit_index]),
    )


def exact_binomial_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(n, index) for index in range(tail + 1)) / 2**n
    return min(1.0, 2.0 * probability)


__all__ = [
    "ANSWER_SLOTS",
    "VIDEO_CONDITIONS",
    "CalibrationRow",
    "GainRow",
    "NeuralGainGrounder",
    "SoftmaxCalibrationHead",
    "VideoPolicyDecision",
    "build_source_value_models",
    "choose_video_action",
    "exact_binomial_two_sided",
    "fit_calibration_head",
    "fit_gain_grounder",
    "normalized_entropy",
    "normalized_probabilities",
    "stable_hash",
    "video_action_features",
]
