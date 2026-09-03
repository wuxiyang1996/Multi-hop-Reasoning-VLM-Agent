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
    ResidualValueEnsemble,
    ValueEnsemble,
    _build_domains_and_examples,
    fit_value_ensemble,
    fit_source_prior_residual_ensemble,
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


def normalized_probabilities(
    values: Mapping[str, Any],
    *,
    answer_slots: Sequence[str] = ANSWER_SLOTS,
) -> np.ndarray:
    slots = tuple(map(str, answer_slots))
    if len(slots) < 2 or len(set(slots)) != len(slots):
        raise ValueError("answer_slots must contain at least two unique slots")
    vector = np.asarray([float(values.get(slot, 0.0)) for slot in slots])
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
    answer_slot_count: int = len(ANSWER_SLOTS)

    def predict(self, features: Sequence[float]) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (self.answer_slot_count + 2,):
            raise ValueError("calibration feature shape mismatch")
        context = np.asarray([1.0, values[-2], values[-1]])
        raw_temperature = float(context @ np.asarray(self.temperature_weights))
        temperature = 0.25 + float(np.logaddexp(0.0, raw_temperature))
        logits = values[:self.answer_slot_count] / temperature
        logits -= np.max(logits)
        probabilities = np.exp(np.clip(logits, -30.0, 30.0))
        return probabilities / np.sum(probabilities)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "target_native_slot_symmetric_temperature_head",
            "temperature_weights": list(self.temperature_weights),
            "answer_slot_count": self.answer_slot_count,
            "cannot_permute_answer_slots": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SoftmaxCalibrationHead":
        return cls(
            tuple(map(float, payload["temperature_weights"])),
            int(payload.get("answer_slot_count", len(ANSWER_SLOTS))),
        )


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
    answer_slot_count = len(rows[0].raw_probabilities)
    if answer_slot_count < 2 or any(
        len(row.raw_probabilities) != answer_slot_count for row in rows
    ):
        raise ValueError("calibration rows must share at least two answer slots")
    if any(not 0 <= row.answer_index < answer_slot_count for row in rows):
        raise ValueError("calibration answer index is outside the answer slots")
    matrix = np.asarray([row.features() for row in rows], dtype=np.float64)
    labels = np.asarray([row.answer_index for row in rows], dtype=np.int64)
    raw_logits = matrix[:, :answer_slot_count]
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
    return SoftmaxCalibrationHead(
        tuple(map(float, weights)), answer_slot_count=answer_slot_count,
    )


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


CANDIDATE_DESCRIPTOR_WIDTH = 8


@dataclass(frozen=True)
class CandidateEffectRow:
    """One matched target-native evidence intervention.

    ``descriptor`` is target-native and action-token free from the source
    model's point of view.  The first two outputs retain the controlled
    source interface (information/confidence gain); ``answer_quality_gain``
    is used only by the target-native candidate selector and preflight.
    """

    sample_id: str
    candidate_id: str
    current_belief: tuple[float, ...]
    planner_score: float
    descriptor: tuple[float, ...]
    information_gain: float
    confidence_gain: float
    answer_quality_gain: float

    def features(self) -> np.ndarray:
        if not self.descriptor:
            raise ValueError("candidate descriptor cannot be empty")
        return np.asarray([
            normalized_entropy(self.current_belief),
            max(self.current_belief),
            float(self.planner_score),
            *map(float, self.descriptor),
        ], dtype=np.float64)


@dataclass(frozen=True)
class CandidateEffectGrounder:
    """Small target-native neural model for candidate intervention uplift."""

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
        planner_score: float,
        descriptor: Sequence[float],
    ) -> tuple[float, float, float]:
        expected_width = len(self.input_mean) - 3
        if len(descriptor) != expected_width:
            raise ValueError(
                f"candidate descriptor width mismatch: expected "
                f"{expected_width}, got {len(descriptor)}"
            )
        row = CandidateEffectRow(
            sample_id="runtime",
            candidate_id="runtime",
            current_belief=tuple(map(float, belief)),
            planner_score=float(planner_score),
            descriptor=tuple(map(float, descriptor)),
            information_gain=0.0,
            confidence_gain=0.0,
            answer_quality_gain=0.0,
        )
        values = (row.features() - np.asarray(self.input_mean)) / np.asarray(
            self.input_scale
        )
        hidden = np.tanh(
            values @ np.asarray(self.hidden_weights) + np.asarray(self.hidden_bias)
        )
        output = hidden @ np.asarray(self.output_weights) + np.asarray(self.output_bias)
        return (
            float(np.clip(output[0], 0.0, 1.0)),
            float(np.clip(output[1], 0.0, 1.0)),
            float(np.clip(output[2], -1.0, 1.0)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "target_native_candidate_intervention_uplift_mlp",
            "descriptor_width": len(self.input_mean) - 3,
            "input_mean": list(self.input_mean),
            "input_scale": list(self.input_scale),
            "hidden_weights": [list(row) for row in self.hidden_weights],
            "hidden_bias": list(self.hidden_bias),
            "output_weights": [list(row) for row in self.output_weights],
            "output_bias": list(self.output_bias),
            "outputs": [
                "expected_information_gain",
                "expected_map_confidence_gain",
                "expected_answer_quality_gain",
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateEffectGrounder":
        model = cls(
            tuple(map(float, payload["input_mean"])),
            tuple(map(float, payload["input_scale"])),
            tuple(tuple(map(float, row)) for row in payload["hidden_weights"]),
            tuple(map(float, payload["hidden_bias"])),
            tuple(tuple(map(float, row)) for row in payload["output_weights"]),
            tuple(map(float, payload["output_bias"])),
        )
        if int(payload["descriptor_width"]) != len(model.input_mean) - 3:
            raise ValueError("candidate grounder descriptor width drift")
        return model


def fit_candidate_effect_grounder(
    rows: Sequence[CandidateEffectRow],
    *,
    seed: int,
    hidden_units: int = 16,
    epochs: int = 1800,
    learning_rate: float = 0.02,
    l2: float = 0.01,
) -> CandidateEffectGrounder:
    """Fit a deterministic MLP to matched target intervention deltas."""

    if not rows:
        raise ValueError("candidate effect rows cannot be empty")
    descriptor_widths = {len(row.descriptor) for row in rows}
    if len(descriptor_widths) != 1:
        raise ValueError("candidate descriptors must have one aligned width")
    matrix = np.asarray([row.features() for row in rows], dtype=np.float64)
    labels = np.asarray([
        (
            max(0.0, row.information_gain),
            max(0.0, row.confidence_gain),
            float(np.clip(row.answer_quality_gain, -1.0, 1.0)),
        )
        for row in rows
    ], dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-6] = 1.0
    inputs = (matrix - mean) / scale
    rng = np.random.default_rng(seed)
    w1 = rng.normal(
        0.0, math.sqrt(2 / inputs.shape[1]), (inputs.shape[1], hidden_units),
    )
    b1 = np.zeros(hidden_units)
    w2 = rng.normal(0.0, math.sqrt(2 / hidden_units), (hidden_units, 3))
    b2 = np.zeros(3)
    parameters = [w1, b1, w2, b2]
    first = [np.zeros_like(value) for value in parameters]
    second = [np.zeros_like(value) for value in parameters]
    beta1, beta2 = 0.9, 0.999
    for epoch in range(1, epochs + 1):
        hidden = np.tanh(inputs @ parameters[0] + parameters[1])
        predictions = hidden @ parameters[2] + parameters[3]
        error = (predictions - labels) * (2.0 / len(labels))
        hidden_gradient = (error @ parameters[2].T) * (1.0 - hidden**2)
        gradients = [
            inputs.T @ hidden_gradient + l2 * parameters[0],
            np.sum(hidden_gradient, axis=0),
            hidden.T @ error + l2 * parameters[2],
            np.sum(error, axis=0),
        ]
        for index, gradient in enumerate(gradients):
            first[index] = beta1 * first[index] + (1.0 - beta1) * gradient
            second[index] = beta2 * second[index] + (1.0 - beta2) * gradient**2
            parameters[index] -= learning_rate * (
                first[index] / (1.0 - beta1**epoch)
            ) / (np.sqrt(second[index] / (1.0 - beta2**epoch)) + 1e-8)
    return CandidateEffectGrounder(
        tuple(map(float, mean)),
        tuple(map(float, scale)),
        tuple(tuple(map(float, row)) for row in parameters[0]),
        tuple(map(float, parameters[1])),
        tuple(tuple(map(float, row)) for row in parameters[2]),
        tuple(map(float, parameters[3])),
    )


@dataclass(frozen=True)
class GroundedCandidateIntervention:
    candidate_id: str
    planner_score: float
    predicted_information_gain: float
    predicted_confidence_gain: float
    predicted_answer_quality_gain: float
    predicted_outcome_balance: float
    repeat_fraction: float = 0.0


@dataclass(frozen=True)
class CandidatePolicyDecision:
    kind: str
    candidate_id: str | None
    answer_index: int | None
    source_abstained: bool
    predicted_test_value: float | None
    predicted_commit_value: float | None


def candidate_action_features(
    belief: Sequence[float],
    *,
    candidates: Sequence[GroundedCandidateIntervention],
    remaining_test_fraction: float,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Ground parameterized TEST(candidate) actions plus native commits."""

    vector = np.asarray(belief, dtype=np.float64)
    vector /= np.sum(vector)
    entropy = normalized_entropy(vector)
    tests = tuple((
        1.0,
        float(candidate.predicted_information_gain),
        float(candidate.predicted_confidence_gain),
        float(candidate.predicted_outcome_balance),
        float(np.max(vector)),
        entropy,
        float(remaining_test_fraction),
        0.0,
        float(candidate.repeat_fraction),
    ) for candidate in candidates)
    commits = tuple((
        0.0, 0.0, 0.0, 0.0, float(np.max(vector)), entropy,
        float(remaining_test_fraction), float(probability), 0.0,
    ) for probability in vector)
    if any(len(row) != len(FEATURE_NAMES) for row in tests + commits):
        raise AssertionError("candidate feature contract drift")
    return tests, commits


def choose_candidate_action(
    belief: Sequence[float],
    *,
    condition: str,
    candidates: Sequence[GroundedCandidateIntervention],
    source_models: Mapping[str, ValueEnsemble],
    uncertainty_scale: float,
    decision_margin: float,
    fallback_commit_threshold: float,
    target_quality_threshold: float,
    information_gain_threshold: float,
) -> CandidatePolicyDecision:
    """Choose among parameterized TEST candidates and native COMMITs.

    The source model only sees the nine token-free causal features.  Candidate
    IDs and target tool names remain target-native and are returned only after
    the abstract source value comparison.
    """

    if not candidates:
        raise ValueError("candidate intervention set cannot be empty")
    vector = np.asarray(belief, dtype=np.float64)
    vector /= np.sum(vector)
    best_answer = int(np.argmax(vector))
    if condition == "target_only":
        best = max(candidates, key=lambda row: row.planner_score)
        take_test = float(np.max(vector)) < fallback_commit_threshold
        return CandidatePolicyDecision(
            "TEST" if take_test else "COMMIT",
            best.candidate_id if take_test else None,
            None if take_test else best_answer,
            True,
            best.planner_score,
            float(np.max(vector)),
        )
    if condition == "target_native_information_gain":
        best = max(candidates, key=lambda row: row.predicted_information_gain)
        take_test = best.predicted_information_gain > information_gain_threshold
        return CandidatePolicyDecision(
            "TEST" if take_test else "COMMIT",
            best.candidate_id if take_test else None,
            None if take_test else best_answer,
            False,
            best.predicted_information_gain,
            float(np.max(vector)),
        )
    if condition == "target_native_candidate_uplift":
        best = max(candidates, key=lambda row: row.predicted_answer_quality_gain)
        take_test = best.predicted_answer_quality_gain > target_quality_threshold
        return CandidatePolicyDecision(
            "TEST" if take_test else "COMMIT",
            best.candidate_id if take_test else None,
            None if take_test else best_answer,
            False,
            best.predicted_answer_quality_gain,
            float(np.max(vector)),
        )
    tests, commits = candidate_action_features(
        vector, candidates=candidates, remaining_test_fraction=1.0,
    )
    model = source_models[condition]
    means, deviations = model.predict(tests + commits)
    test_index = int(np.argmax(means[:len(tests)]))
    commit_offset = int(np.argmax(means[len(tests):]))
    commit_index = len(tests) + commit_offset
    gap = float(means[test_index] - means[commit_index])
    uncertainty = uncertainty_scale * math.sqrt(float(
        deviations[test_index] ** 2 + deviations[commit_index] ** 2
    ))
    if gap - uncertainty > decision_margin:
        return CandidatePolicyDecision(
            "TEST", candidates[test_index].candidate_id, None, False,
            float(means[test_index]), float(means[commit_index]),
        )
    if -gap - uncertainty > decision_margin:
        return CandidatePolicyDecision(
            "COMMIT", None, commit_offset, False,
            float(means[test_index]), float(means[commit_index]),
        )
    best = max(candidates, key=lambda row: row.predicted_answer_quality_gain)
    take_test = best.predicted_answer_quality_gain > target_quality_threshold
    return CandidatePolicyDecision(
        "TEST" if take_test else "COMMIT",
        best.candidate_id if take_test else None,
        None if take_test else best_answer,
        True,
        float(means[test_index]),
        float(means[commit_index]),
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


def build_source_value_training_sets(
    controlled_config: Mapping[str, Any],
    *,
    objective_test_cost: float | None = None,
) -> dict[str, tuple[MatchedValueExample, ...]]:
    """Rebuild authentic/control source supervision without target receipts.

    ``objective_test_cost`` grounds the source program to the target metric.
    It changes neither target labels nor candidate outcomes: it only asks the
    same source active-identification domains to optimize the target's declared
    intervention cost.  Accuracy-only benchmarks therefore use zero, whereas
    latency/cost-aware targets may retain a positive value.
    """

    source = controlled_config["source"]
    domain_config = dict(controlled_config["domain"])
    if objective_test_cost is not None:
        if not 0.0 <= float(objective_test_cost) < 1.0:
            raise ValueError("objective_test_cost must be in [0, 1)")
        domain_config["test_cost"] = float(objective_test_cost)
    _, _, examples = _build_domains_and_examples(
        source["train_domain_seeds"],
        surface="game",
        domain_config=domain_config,
        state_count=int(source["states_per_domain"]),
        calibration_seed_namespace="source-train",
    )
    return {
        "authentic_source_plus_target": examples,
        "shuffled_source_plus_target": shuffled_value_control(
            examples, seed=int(controlled_config["model"]["control_seed"]),
        ),
        "source_marginal_plus_target": marginal_value_control(examples),
    }


def build_source_value_models(
    controlled_config: Mapping[str, Any],
    *,
    seed: int,
    objective_test_cost: float | None = None,
) -> dict[str, ValueEnsemble]:
    """Rebuild frozen V3 source-only models without target receipts."""

    controls = build_source_value_training_sets(
        controlled_config, objective_test_cost=objective_test_cost,
    )
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


def build_source_residual_value_models(
    controlled_config: Mapping[str, Any],
    target_examples: Sequence[MatchedValueExample],
    *,
    seed: int,
    objective_test_cost: float | None = None,
    residual_scale: float = 1.0,
) -> dict[str, Any]:
    """Cross-domain source priors with the same target residual supervision."""

    controls = build_source_value_training_sets(
        controlled_config, objective_test_cost=objective_test_cost,
    )
    model_config = controlled_config["model"]
    output = {}
    for index, (condition, rows) in enumerate(controls.items()):
        model = fit_source_prior_residual_ensemble(
            rows,
            target_examples,
            seed=seed + index,
            ensemble_size=int(model_config["ensemble_size"]),
            source_alpha=float(model_config["ridge_alpha"]),
            residual_alpha=float(model_config["residual_ridge_alpha"]),
            residual_scale=float(residual_scale),
        )
        assert model is not None
        output[condition] = model
    return output


def add_target_residual_to_source_models(
    controlled_config: Mapping[str, Any],
    source_models: Mapping[str, ValueEnsemble],
    target_examples: Sequence[MatchedValueExample],
    *,
    seed: int,
    residual_scale: float,
) -> dict[str, ValueEnsemble | ResidualValueEnsemble]:
    """Fit target residuals while keeping already-fitted source priors frozen."""

    if not target_examples or residual_scale <= 0:
        return dict(source_models)
    model_config = controlled_config["model"]
    output: dict[str, ValueEnsemble | ResidualValueEnsemble] = {}
    features = [row.features for row in target_examples]
    for index, (condition, source_model) in enumerate(source_models.items()):
        source_predictions, _ = source_model.predict(features)
        residual_rows = tuple(
            MatchedValueExample(
                state_id=row.state_id,
                action=row.action,
                features=row.features,
                value=float(row.value - prediction),
            )
            for row, prediction in zip(target_examples, source_predictions)
        )
        residual_model = fit_value_ensemble(
            (),
            residual_rows,
            seed=seed + index,
            ensemble_size=int(model_config["ensemble_size"]),
            alpha=float(model_config["residual_ridge_alpha"]),
            target_mass=1.0,
        )
        assert residual_model is not None
        output[condition] = ResidualValueEnsemble(
            source_model, residual_model, float(residual_scale),
        )
    return output


def source_test_feature_support(
    controlled_config: Mapping[str, Any],
    *,
    objective_test_cost: float | None = None,
) -> dict[str, float]:
    """Return empirical source support for continuous TEST features."""

    source = controlled_config["source"]
    domain_config = dict(controlled_config["domain"])
    if objective_test_cost is not None:
        domain_config["test_cost"] = float(objective_test_cost)
    _, _, examples = _build_domains_and_examples(
        source["train_domain_seeds"],
        surface="game",
        domain_config=domain_config,
        state_count=int(source["states_per_domain"]),
        calibration_seed_namespace="source-train",
    )
    tests = np.asarray([
        row.features for row in examples if row.action.kind == "TEST"
    ], dtype=np.float64)
    if not len(tests):
        raise ValueError("source training set contains no TEST examples")
    return {
        "maximum_information_gain": float(np.max(tests[:, 1])),
        "maximum_confidence_gain": float(np.max(tests[:, 2])),
    }


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
    "CANDIDATE_DESCRIPTOR_WIDTH",
    "VIDEO_CONDITIONS",
    "CalibrationRow",
    "CandidateEffectGrounder",
    "CandidateEffectRow",
    "CandidatePolicyDecision",
    "GainRow",
    "GroundedCandidateIntervention",
    "NeuralGainGrounder",
    "SoftmaxCalibrationHead",
    "VideoPolicyDecision",
    "build_source_value_models",
    "build_source_residual_value_models",
    "add_target_residual_to_source_models",
    "build_source_value_training_sets",
    "source_test_feature_support",
    "candidate_action_features",
    "choose_candidate_action",
    "choose_video_action",
    "exact_binomial_two_sided",
    "fit_calibration_head",
    "fit_candidate_effect_grounder",
    "fit_gain_grounder",
    "normalized_entropy",
    "normalized_probabilities",
    "stable_hash",
    "video_action_features",
]
