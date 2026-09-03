"""Controlled game-to-diagnosis transfer of an intervention-value program.

The source and target domains intentionally share no action tokens.  They share
only a latent active-identification problem: acquire evidence, update a belief,
and commit to one of several hypotheses.  Source supervision is made of matched
intervention values.  A target-native grounder estimates target observation
likelihoods from target calibration receipts before the source program is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "is_test",
    "expected_information_gain",
    "expected_map_confidence_gain",
    "predicted_outcome_balance",
    "current_map_confidence",
    "current_entropy",
    "remaining_test_fraction",
    "candidate_hypothesis_probability",
    "action_repeat_fraction",
)

CONDITIONS = (
    "target_only",
    "authentic_source_plus_target",
    "shuffled_source_plus_target",
    "source_marginal_plus_target",
)

SURFACE_VOCABULARY = {
    "game": ("unlock_vault", "pull_rune"),
    "diagnosis": ("diagnose_syndrome", "order_assay"),
    "software": ("repair_fault", "run_trace"),
    "science": ("identify_catalyst", "perform_assay"),
    "forensics": ("attribute_case", "inspect_clue"),
    "network": ("isolate_fault", "probe_link"),
}


def _stable_seed(payload: object) -> int:
    raw = repr(payload).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**32)


def _entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(values * np.log(values)))


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("belief mass must be positive and finite")
    return values / total


@dataclass(frozen=True)
class AbstractAction:
    kind: str
    index: int
    token: str

    def validate(self, hypothesis_count: int, test_count: int) -> None:
        if self.kind not in {"TEST", "COMMIT"}:
            raise ValueError("unknown abstract action kind")
        limit = test_count if self.kind == "TEST" else hypothesis_count
        if not 0 <= self.index < limit:
            raise ValueError("abstract action index is out of bounds")
        if not self.token:
            raise ValueError("action token must be non-empty")


@dataclass(frozen=True)
class ActiveIdentificationDomain:
    domain_id: str
    surface: str
    hypothesis_tokens: tuple[str, ...]
    test_tokens: tuple[str, ...]
    likelihood: tuple[tuple[float, ...], ...]
    max_tests: int
    test_cost: float

    def validate(self) -> None:
        if self.surface not in SURFACE_VOCABULARY:
            raise ValueError(f"unknown surface: {self.surface}")
        if len(self.hypothesis_tokens) < 2 or len(self.test_tokens) < 2:
            raise ValueError("domain requires multiple hypotheses and tests")
        if set(self.hypothesis_tokens) & set(self.test_tokens):
            raise ValueError("hypothesis and test tokens must be disjoint")
        matrix = self.likelihood_array
        expected = (len(self.hypothesis_tokens), len(self.test_tokens))
        if matrix.shape != expected:
            raise ValueError("likelihood matrix shape mismatch")
        if np.any(matrix <= 0) or np.any(matrix >= 1):
            raise ValueError("likelihoods must be strictly between zero and one")
        if self.max_tests <= 0:
            raise ValueError("max_tests must be positive")
        if not 0 <= self.test_cost < 1:
            raise ValueError("test_cost must be in [0, 1)")

    @property
    def likelihood_array(self) -> np.ndarray:
        return np.asarray(self.likelihood, dtype=np.float64)

    @property
    def actions(self) -> tuple[AbstractAction, ...]:
        tests = tuple(
            AbstractAction("TEST", index, token)
            for index, token in enumerate(self.test_tokens)
        )
        commits = tuple(
            AbstractAction("COMMIT", index, token)
            for index, token in enumerate(self.hypothesis_tokens)
        )
        return tests + commits


@dataclass(frozen=True)
class GroundedState:
    true_belief: tuple[float, ...]
    grounded_belief: tuple[float, ...]
    remaining_tests: int
    test_counts: tuple[int, ...]


@dataclass(frozen=True)
class MatchedValueExample:
    state_id: str
    action: AbstractAction
    features: tuple[float, ...]
    value: float


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    condition: str
    success: bool
    net_return: float
    test_count: int
    abstention_count: int
    committed_hypothesis: int


@dataclass(frozen=True)
class RidgeValueModel:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError("value-model feature shape mismatch")
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        design = np.column_stack(((matrix - mean) / scale, np.ones(len(matrix))))
        return design @ np.asarray(self.coefficients, dtype=np.float64)


@dataclass(frozen=True)
class ValueEnsemble:
    models: tuple[RidgeValueModel, ...]

    def predict(self, features: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise ValueError("cannot predict with an empty ensemble")
        predictions = np.asarray(
            [model.predict(features) for model in self.models], dtype=np.float64,
        )
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)


@dataclass(frozen=True)
class ResidualValueEnsemble:
    """A source value prior plus a target-native residual correction."""

    source: ValueEnsemble
    residual: ValueEnsemble
    residual_scale: float

    def predict(self, features: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
        source_mean, source_deviation = self.source.predict(features)
        residual_mean, residual_deviation = self.residual.predict(features)
        mean = source_mean + self.residual_scale * residual_mean
        deviation = np.sqrt(
            source_deviation**2
            + (self.residual_scale * residual_deviation) ** 2
        )
        return mean, deviation


def make_domain(
    *,
    seed: int,
    surface: str,
    hypothesis_count: int = 4,
    test_count: int = 5,
    max_tests: int = 4,
    test_cost: float = 0.025,
) -> ActiveIdentificationDomain:
    """Create an information-rich hidden-rule game or diagnosis task."""

    if hypothesis_count > 2**test_count:
        raise ValueError("test_count cannot distinguish all hypotheses")
    rng = np.random.default_rng(seed)
    codes: np.ndarray | None = None
    for _ in range(1000):
        integers = rng.choice(2**test_count, size=hypothesis_count, replace=False)
        candidate = np.asarray([
            [(int(value) >> bit) & 1 for bit in range(test_count)]
            for value in integers
        ], dtype=np.int64)
        columns_split = all(len(set(candidate[:, bit])) == 2 for bit in range(test_count))
        pair_distances = [
            int(np.sum(candidate[left] != candidate[right]))
            for left in range(hypothesis_count)
            for right in range(left + 1, hypothesis_count)
        ]
        if columns_split and min(pair_distances) >= 2:
            codes = candidate
            break
    if codes is None:
        raise RuntimeError("failed to construct a distinguishable domain")
    accuracies = rng.uniform(0.72, 0.92, size=test_count)
    likelihood = np.empty((hypothesis_count, test_count), dtype=np.float64)
    for hypothesis in range(hypothesis_count):
        for test in range(test_count):
            base = accuracies[test] if codes[hypothesis, test] else 1 - accuracies[test]
            likelihood[hypothesis, test] = float(np.clip(
                base + rng.uniform(-0.025, 0.025), 0.05, 0.95,
            ))
    if surface not in SURFACE_VOCABULARY:
        raise ValueError(f"unknown surface: {surface}")
    hypothesis_prefix, test_prefix = SURFACE_VOCABULARY[surface]
    hypotheses = tuple(
        f"{hypothesis_prefix}_{seed}_{index}" for index in range(hypothesis_count)
    )
    tests = tuple(f"{test_prefix}_{seed}_{index}" for index in range(test_count))
    domain = ActiveIdentificationDomain(
        domain_id=f"{surface}-{seed}",
        surface=surface,
        hypothesis_tokens=hypotheses,
        test_tokens=tests,
        likelihood=tuple(tuple(map(float, row)) for row in likelihood),
        max_tests=max_tests,
        test_cost=test_cost,
    )
    domain.validate()
    for action in domain.actions:
        action.validate(hypothesis_count, test_count)
    return domain


def calibrate_target_grounder(
    domain: ActiveIdentificationDomain,
    *,
    samples_per_cell: int,
    seed: int,
    beta_prior: float = 1.5,
) -> np.ndarray:
    """Estimate target-native outcome likelihoods from calibration receipts."""

    if samples_per_cell <= 0:
        raise ValueError("samples_per_cell must be positive")
    rng = np.random.default_rng(seed)
    successes = rng.binomial(samples_per_cell, domain.likelihood_array)
    estimate = (successes + beta_prior) / (samples_per_cell + 2 * beta_prior)
    return np.asarray(estimate, dtype=np.float64)


def calibrate_target_neural_grounder(
    domain: ActiveIdentificationDomain,
    *,
    samples_per_cell: int,
    seed: int,
    beta_prior: float = 1.5,
    hidden_units: int = 32,
    epochs: int = 1800,
    learning_rate: float = 0.03,
    l2: float = 1e-4,
) -> np.ndarray:
    """Fit a target-native MLP to anonymous intervention/outcome receipts.

    The network receives only local one-hot hypothesis and intervention IDs.
    Environment likelihoods are used to sample calibration outcomes, never as
    training labels or runtime inputs.
    """

    if samples_per_cell <= 0:
        raise ValueError("samples_per_cell must be positive")
    if hidden_units <= 0 or epochs <= 0 or learning_rate <= 0 or l2 < 0:
        raise ValueError("invalid neural grounder hyperparameters")
    hypothesis_count, test_count = domain.likelihood_array.shape
    rng = np.random.default_rng(seed)
    successes = rng.binomial(samples_per_cell, domain.likelihood_array)
    targets = (successes + beta_prior) / (samples_per_cell + 2 * beta_prior)
    inputs = np.zeros(
        (hypothesis_count * test_count, hypothesis_count + test_count),
        dtype=np.float64,
    )
    for hypothesis in range(hypothesis_count):
        for test in range(test_count):
            row = hypothesis * test_count + test
            inputs[row, hypothesis] = 1.0
            inputs[row, hypothesis_count + test] = 1.0
    labels = targets.reshape(-1)
    input_weights = rng.normal(
        0.0, math.sqrt(2.0 / inputs.shape[1]),
        size=(inputs.shape[1], hidden_units),
    )
    hidden_bias = np.zeros(hidden_units, dtype=np.float64)
    output_weights = rng.normal(
        0.0, math.sqrt(2.0 / hidden_units), size=hidden_units,
    )
    output_bias = 0.0
    parameters: list[np.ndarray] = [
        input_weights,
        hidden_bias,
        output_weights,
        np.asarray([output_bias], dtype=np.float64),
    ]
    first_moments = [np.zeros_like(value) for value in parameters]
    second_moments = [np.zeros_like(value) for value in parameters]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for epoch in range(1, epochs + 1):
        hidden = np.tanh(inputs @ parameters[0] + parameters[1])
        logits = hidden @ parameters[2] + parameters[3][0]
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        logit_gradient = (probabilities - labels) / len(labels)
        gradients = [
            inputs.T @ (
                (logit_gradient[:, None] * parameters[2][None, :])
                * (1.0 - hidden**2)
            ) + l2 * parameters[0],
            np.sum(
                (logit_gradient[:, None] * parameters[2][None, :])
                * (1.0 - hidden**2),
                axis=0,
            ),
            hidden.T @ logit_gradient + l2 * parameters[2],
            np.asarray([np.sum(logit_gradient)], dtype=np.float64),
        ]
        for index, gradient in enumerate(gradients):
            first_moments[index] = (
                beta1 * first_moments[index] + (1.0 - beta1) * gradient
            )
            second_moments[index] = (
                beta2 * second_moments[index] + (1.0 - beta2) * gradient**2
            )
            corrected_first = first_moments[index] / (1.0 - beta1**epoch)
            corrected_second = second_moments[index] / (1.0 - beta2**epoch)
            parameters[index] -= learning_rate * corrected_first / (
                np.sqrt(corrected_second) + epsilon
            )
    hidden = np.tanh(inputs @ parameters[0] + parameters[1])
    logits = hidden @ parameters[2] + parameters[3][0]
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return np.clip(probabilities.reshape(hypothesis_count, test_count), 0.03, 0.97)


def update_belief(
    belief: Sequence[float], likelihood: np.ndarray, test_index: int, outcome: int,
) -> np.ndarray:
    values = np.asarray(belief, dtype=np.float64)
    factors = likelihood[:, test_index] if outcome else 1 - likelihood[:, test_index]
    return _normalize(values * factors)


def _test_statistics(
    belief: np.ndarray, likelihood: np.ndarray, test_index: int,
) -> tuple[float, float, float]:
    current_entropy = _entropy(belief)
    current_max = float(np.max(belief))
    probability_one = float(np.dot(belief, likelihood[:, test_index]))
    expected_entropy = 0.0
    expected_max = 0.0
    for outcome, probability in ((1, probability_one), (0, 1 - probability_one)):
        posterior = update_belief(belief, likelihood, test_index, outcome)
        expected_entropy += probability * _entropy(posterior)
        expected_max += probability * float(np.max(posterior))
    information_gain = max(0.0, current_entropy - expected_entropy)
    confidence_gain = expected_max - current_max
    balance = 1.0 - 2.0 * abs(probability_one - 0.5)
    return information_gain, confidence_gain, balance


def action_features(
    domain: ActiveIdentificationDomain,
    state: GroundedState,
    grounded_likelihood: np.ndarray,
    action: AbstractAction,
) -> tuple[float, ...]:
    belief = np.asarray(state.grounded_belief, dtype=np.float64)
    entropy_scale = math.log(len(belief))
    if action.kind == "TEST":
        information_gain, confidence_gain, balance = _test_statistics(
            belief, grounded_likelihood, action.index,
        )
        candidate_probability = 0.0
        repeat_fraction = state.test_counts[action.index] / domain.max_tests
        is_test = 1.0
    else:
        information_gain = 0.0
        confidence_gain = 0.0
        balance = 0.0
        candidate_probability = float(belief[action.index])
        repeat_fraction = 0.0
        is_test = 0.0
    return (
        is_test,
        information_gain / entropy_scale,
        confidence_gain,
        balance,
        float(np.max(belief)),
        _entropy(belief) / entropy_scale,
        state.remaining_tests / domain.max_tests,
        candidate_probability,
        repeat_fraction,
    )


def _optimal_values(
    belief: np.ndarray,
    likelihood: np.ndarray,
    remaining_tests: int,
    test_cost: float,
    cache: dict[tuple[tuple[float, ...], int], float],
) -> tuple[np.ndarray, np.ndarray]:
    commit_values = np.asarray(belief, dtype=np.float64)
    if remaining_tests <= 0:
        return np.empty(0, dtype=np.float64), commit_values
    test_values = []
    for test_index in range(likelihood.shape[1]):
        probability_one = float(np.dot(belief, likelihood[:, test_index]))
        expected = -test_cost
        for outcome, probability in ((1, probability_one), (0, 1 - probability_one)):
            posterior = update_belief(belief, likelihood, test_index, outcome)
            key = (tuple(np.round(posterior, 10)), remaining_tests - 1)
            if key not in cache:
                nested_tests, nested_commits = _optimal_values(
                    posterior, likelihood, remaining_tests - 1, test_cost, cache,
                )
                cache[key] = float(max(
                    np.max(nested_commits),
                    np.max(nested_tests) if len(nested_tests) else -math.inf,
                ))
            expected += probability * cache[key]
        test_values.append(expected)
    return np.asarray(test_values, dtype=np.float64), commit_values


def matched_action_values(
    domain: ActiveIdentificationDomain, state: GroundedState,
) -> dict[tuple[str, int], float]:
    belief = np.asarray(state.true_belief, dtype=np.float64)
    tests, commits = _optimal_values(
        belief,
        domain.likelihood_array,
        state.remaining_tests,
        domain.test_cost,
        {},
    )
    values = {
        ("COMMIT", index): float(value)
        for index, value in enumerate(commits)
    }
    values.update({
        ("TEST", index): float(value)
        for index, value in enumerate(tests)
    })
    return values


def collect_matched_examples(
    domain: ActiveIdentificationDomain,
    grounded_likelihood: np.ndarray,
    *,
    state_count: int,
    seed: int,
) -> tuple[MatchedValueExample, ...]:
    """Collect same-state values for every native intervention."""

    if state_count <= 0:
        raise ValueError("state_count must be positive")
    rng = np.random.default_rng(seed)
    hypothesis_count = len(domain.hypothesis_tokens)
    rows: list[MatchedValueExample] = []
    for state_index in range(state_count):
        hidden = int(rng.integers(hypothesis_count))
        history_length = int(rng.integers(0, domain.max_tests + 1))
        true_belief = np.full(hypothesis_count, 1 / hypothesis_count)
        grounded_belief = true_belief.copy()
        counts = np.zeros(len(domain.test_tokens), dtype=np.int64)
        for _ in range(history_length):
            test_index = int(rng.integers(len(domain.test_tokens)))
            outcome = int(rng.random() < domain.likelihood_array[hidden, test_index])
            true_belief = update_belief(
                true_belief, domain.likelihood_array, test_index, outcome,
            )
            grounded_belief = update_belief(
                grounded_belief, grounded_likelihood, test_index, outcome,
            )
            counts[test_index] += 1
        state = GroundedState(
            tuple(map(float, true_belief)),
            tuple(map(float, grounded_belief)),
            domain.max_tests - history_length,
            tuple(map(int, counts)),
        )
        values = matched_action_values(domain, state)
        state_id = f"{domain.domain_id}:state-{state_index}"
        for action in domain.actions:
            key = (action.kind, action.index)
            if key not in values:
                continue
            rows.append(MatchedValueExample(
                state_id=state_id,
                action=action,
                features=action_features(domain, state, grounded_likelihood, action),
                value=values[key],
            ))
    return tuple(rows)


def shuffled_value_control(
    examples: Sequence[MatchedValueExample], *, seed: int,
) -> tuple[MatchedValueExample, ...]:
    """Shuffle values only within matched states, preserving state marginals."""

    grouped: dict[str, list[MatchedValueExample]] = {}
    for row in examples:
        grouped.setdefault(row.state_id, []).append(row)
    output = []
    for state_id, rows in sorted(grouped.items()):
        rng = np.random.default_rng(_stable_seed((seed, state_id, "shuffle")))
        values = np.asarray([row.value for row in rows], dtype=np.float64)
        values = values[rng.permutation(len(values))]
        output.extend(
            MatchedValueExample(
                row.state_id, row.action, row.features, float(value),
            )
            for row, value in zip(rows, values)
        )
    return tuple(output)


def marginal_value_control(
    examples: Sequence[MatchedValueExample],
) -> tuple[MatchedValueExample, ...]:
    means = {
        kind: float(np.mean([row.value for row in examples if row.action.kind == kind]))
        for kind in ("TEST", "COMMIT")
    }
    return tuple(
        MatchedValueExample(row.state_id, row.action, row.features, means[row.action.kind])
        for row in examples
    )


def _fit_ridge(
    examples: Sequence[MatchedValueExample],
    *,
    alpha: float,
    sample_weights: np.ndarray | None = None,
) -> RidgeValueModel:
    if not examples:
        raise ValueError("ridge model requires training examples")
    features = np.asarray([row.features for row in examples], dtype=np.float64)
    labels = np.asarray([row.value for row in examples], dtype=np.float64)
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack((standardized, np.ones(len(standardized))))
    weights = (
        np.ones(len(examples), dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    if weights.shape != (len(examples),) or np.any(weights <= 0):
        raise ValueError("sample weights must be positive and aligned")
    root = np.sqrt(weights)
    weighted_design = design * root[:, None]
    weighted_labels = labels * root
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_labels,
    )
    return RidgeValueModel(
        tuple(map(float, mean)),
        tuple(map(float, scale)),
        tuple(map(float, coefficients)),
    )


def fit_value_ensemble(
    source: Sequence[MatchedValueExample],
    target: Sequence[MatchedValueExample],
    *,
    seed: int,
    ensemble_size: int,
    alpha: float,
    target_mass: float,
) -> ValueEnsemble | None:
    if not source and not target:
        return None
    combined = tuple(source) + tuple(target)
    base_weights = np.ones(len(combined), dtype=np.float64)
    if source and target:
        base_weights[:len(source)] = 1.0 / len(source)
        base_weights[len(source):] = target_mass / len(target)
        base_weights *= len(combined) / np.sum(base_weights)
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(combined):
        grouped.setdefault(row.state_id, []).append(index)
    state_ids = sorted(grouped)
    models = []
    for member in range(ensemble_size):
        rng = np.random.default_rng(_stable_seed((seed, member, "ridge-bootstrap")))
        sampled_states = rng.choice(state_ids, size=len(state_ids), replace=True)
        indices = [index for state_id in sampled_states for index in grouped[str(state_id)]]
        boot = tuple(combined[index] for index in indices)
        weights = base_weights[np.asarray(indices, dtype=np.int64)]
        models.append(_fit_ridge(boot, alpha=alpha, sample_weights=weights))
    return ValueEnsemble(tuple(models))


def fit_source_prior_residual_ensemble(
    source: Sequence[MatchedValueExample],
    target: Sequence[MatchedValueExample],
    *,
    seed: int,
    ensemble_size: int,
    source_alpha: float,
    residual_alpha: float,
    residual_scale: float,
) -> ValueEnsemble | ResidualValueEnsemble | None:
    """Fit target corrections without allowing sparse target data to erase source."""

    if not source:
        return fit_value_ensemble(
            (), target, seed=seed, ensemble_size=ensemble_size,
            alpha=source_alpha, target_mass=1.0,
        )
    source_model = fit_value_ensemble(
        source, (), seed=seed, ensemble_size=ensemble_size,
        alpha=source_alpha, target_mass=1.0,
    )
    assert source_model is not None
    if not target or residual_scale <= 0:
        return source_model
    source_predictions, _ = source_model.predict([row.features for row in target])
    residual_rows = tuple(
        MatchedValueExample(
            state_id=row.state_id,
            action=row.action,
            features=row.features,
            value=float(row.value - prediction),
        )
        for row, prediction in zip(target, source_predictions)
    )
    residual_model = fit_value_ensemble(
        (), residual_rows,
        seed=_stable_seed((seed, "target-residual")),
        ensemble_size=ensemble_size,
        alpha=residual_alpha,
        target_mass=1.0,
    )
    assert residual_model is not None
    return ResidualValueEnsemble(source_model, residual_model, residual_scale)


def _fallback_action(
    domain: ActiveIdentificationDomain,
    state: GroundedState,
    *,
    commit_threshold: float,
) -> AbstractAction:
    belief = np.asarray(state.grounded_belief, dtype=np.float64)
    commits = [action for action in domain.actions if action.kind == "COMMIT"]
    tests = [action for action in domain.actions if action.kind == "TEST"]
    if state.remaining_tests <= 0 or float(np.max(belief)) >= commit_threshold:
        return commits[int(np.argmax(belief))]
    return min(tests, key=lambda action: (state.test_counts[action.index], action.index))


def choose_neurosymbolic_action(
    domain: ActiveIdentificationDomain,
    state: GroundedState,
    grounded_likelihood: np.ndarray,
    model: ValueEnsemble | ResidualValueEnsemble | None,
    *,
    uncertainty_scale: float,
    decision_margin: float,
    fallback_commit_threshold: float,
) -> tuple[AbstractAction, bool]:
    """Route TEST/COMMIT through a value comparator or abstain to fallback."""

    fallback = _fallback_action(
        domain, state, commit_threshold=fallback_commit_threshold,
    )
    if model is None or state.remaining_tests <= 0:
        return fallback, True
    actions = tuple(
        action for action in domain.actions
        if action.kind == "COMMIT" or state.remaining_tests > 0
    )
    features = [action_features(domain, state, grounded_likelihood, action) for action in actions]
    means, deviations = model.predict(features)
    test_indices = [index for index, action in enumerate(actions) if action.kind == "TEST"]
    commit_indices = [index for index, action in enumerate(actions) if action.kind == "COMMIT"]
    best_test = max(test_indices, key=lambda index: means[index])
    best_commit = max(commit_indices, key=lambda index: means[index])
    gap = float(means[best_test] - means[best_commit])
    uncertainty = uncertainty_scale * math.sqrt(
        float(deviations[best_test] ** 2 + deviations[best_commit] ** 2)
    )
    if gap - uncertainty > decision_margin:
        return actions[best_test], False
    if -gap - uncertainty > decision_margin:
        return actions[best_commit], False
    return fallback, True


def _matched_outcome(
    domain: ActiveIdentificationDomain,
    *,
    hidden_hypothesis: int,
    test_index: int,
    occurrence: int,
    episode_seed: int,
) -> int:
    rng = np.random.default_rng(_stable_seed((
        domain.domain_id, hidden_hypothesis, test_index, occurrence, episode_seed,
    )))
    return int(rng.random() < domain.likelihood_array[hidden_hypothesis, test_index])


def run_episode(
    domain: ActiveIdentificationDomain,
    grounded_likelihood: np.ndarray,
    model: ValueEnsemble | ResidualValueEnsemble | None,
    *,
    condition: str,
    episode_seed: int,
    uncertainty_scale: float,
    decision_margin: float,
    fallback_commit_threshold: float,
) -> EpisodeResult:
    hidden = _stable_seed((domain.domain_id, episode_seed, "hidden")) % len(
        domain.hypothesis_tokens
    )
    grounded_belief = np.full(len(domain.hypothesis_tokens), 1 / len(domain.hypothesis_tokens))
    true_belief = grounded_belief.copy()
    counts = np.zeros(len(domain.test_tokens), dtype=np.int64)
    abstentions = 0
    committed = -1
    for decision_index in range(domain.max_tests + 1):
        state = GroundedState(
            tuple(map(float, true_belief)),
            tuple(map(float, grounded_belief)),
            domain.max_tests - int(np.sum(counts)),
            tuple(map(int, counts)),
        )
        action, abstained = choose_neurosymbolic_action(
            domain,
            state,
            grounded_likelihood,
            model,
            uncertainty_scale=uncertainty_scale,
            decision_margin=decision_margin,
            fallback_commit_threshold=fallback_commit_threshold,
        )
        abstentions += int(abstained)
        if action.kind == "COMMIT":
            committed = action.index
            break
        occurrence = int(counts[action.index])
        outcome = _matched_outcome(
            domain,
            hidden_hypothesis=int(hidden),
            test_index=action.index,
            occurrence=occurrence,
            episode_seed=episode_seed,
        )
        true_belief = update_belief(
            true_belief, domain.likelihood_array, action.index, outcome,
        )
        grounded_belief = update_belief(
            grounded_belief, grounded_likelihood, action.index, outcome,
        )
        counts[action.index] += 1
        if decision_index == domain.max_tests:
            break
    if committed < 0:
        committed = int(np.argmax(grounded_belief))
    success = committed == hidden
    test_count = int(np.sum(counts))
    return EpisodeResult(
        episode_id=f"{domain.domain_id}:episode-{episode_seed}",
        condition=condition,
        success=success,
        net_return=float(success) - domain.test_cost * test_count,
        test_count=test_count,
        abstention_count=abstentions,
        committed_hypothesis=committed,
    )


def evaluate_model(
    domains: Sequence[ActiveIdentificationDomain],
    grounded_likelihoods: Mapping[str, np.ndarray],
    model: ValueEnsemble | ResidualValueEnsemble | None,
    *,
    condition: str,
    episode_seeds: Sequence[int],
    policy_config: Mapping[str, float],
) -> tuple[EpisodeResult, ...]:
    rows = []
    for domain in domains:
        grounded = grounded_likelihoods[domain.domain_id]
        for episode_seed in episode_seeds:
            rows.append(run_episode(
                domain,
                grounded,
                model,
                condition=condition,
                episode_seed=int(episode_seed),
                uncertainty_scale=float(policy_config["uncertainty_scale"]),
                decision_margin=float(policy_config["decision_margin"]),
                fallback_commit_threshold=float(policy_config["fallback_commit_threshold"]),
            ))
    return tuple(rows)


def summarize_episodes(rows: Sequence[EpisodeResult]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize empty episodes")
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row.success for row in rows])),
        "mean_net_return": float(np.mean([row.net_return for row in rows])),
        "mean_tests": float(np.mean([row.test_count for row in rows])),
        "mean_abstentions": float(np.mean([row.abstention_count for row in rows])),
    }


def paired_bootstrap_delta(
    treatment: Sequence[EpisodeResult],
    control: Sequence[EpisodeResult],
    *,
    metric: str,
    seed: int,
    samples: int,
) -> dict[str, float]:
    treatment_by_id = {row.episode_id: row for row in treatment}
    control_by_id = {row.episode_id: row for row in control}
    if set(treatment_by_id) != set(control_by_id):
        raise ValueError("paired bootstrap episode identities differ")
    episode_ids = sorted(treatment_by_id)
    if metric == "success":
        deltas = np.asarray([
            float(treatment_by_id[key].success) - float(control_by_id[key].success)
            for key in episode_ids
        ])
    elif metric == "net_return":
        deltas = np.asarray([
            treatment_by_id[key].net_return - control_by_id[key].net_return
            for key in episode_ids
        ])
    else:
        raise ValueError("unsupported paired metric")
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(deltas), size=len(deltas))
        boot[index] = float(np.mean(deltas[selected]))
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _select_states(
    rows: Sequence[MatchedValueExample], state_count: int,
) -> tuple[MatchedValueExample, ...]:
    selected_ids = sorted({row.state_id for row in rows})[:state_count]
    selected = set(selected_ids)
    return tuple(row for row in rows if row.state_id in selected)


def _build_domains_and_examples(
    seeds: Sequence[int],
    *,
    surface: str,
    domain_config: Mapping[str, Any],
    state_count: int,
    calibration_seed_namespace: str,
) -> tuple[tuple[ActiveIdentificationDomain, ...], dict[str, np.ndarray], tuple[MatchedValueExample, ...]]:
    domains = []
    grounders: dict[str, np.ndarray] = {}
    examples = []
    for seed in seeds:
        domain = make_domain(
            seed=int(seed),
            surface=surface,
            hypothesis_count=int(domain_config["hypothesis_count"]),
            test_count=int(domain_config["test_count"]),
            max_tests=int(domain_config["max_tests"]),
            test_cost=float(domain_config["test_cost"]),
        )
        grounder_config = domain_config.get("target_grounder", {})
        grounder_kind = str(grounder_config.get("kind", "beta_binomial"))
        common_grounder = {
            "samples_per_cell": int(domain_config["calibration_samples_per_cell"]),
            "seed": _stable_seed((calibration_seed_namespace, int(seed))),
            "beta_prior": float(domain_config["calibration_beta_prior"]),
        }
        if grounder_kind == "beta_binomial":
            grounded = calibrate_target_grounder(domain, **common_grounder)
        elif grounder_kind == "target_native_mlp":
            grounded = calibrate_target_neural_grounder(
                domain,
                **common_grounder,
                hidden_units=int(grounder_config["hidden_units"]),
                epochs=int(grounder_config["epochs"]),
                learning_rate=float(grounder_config["learning_rate"]),
                l2=float(grounder_config["l2"]),
            )
        else:
            raise ValueError(f"unsupported target grounder: {grounder_kind}")
        domains.append(domain)
        grounders[domain.domain_id] = grounded
        examples.extend(collect_matched_examples(
            domain,
            grounded,
            state_count=state_count,
            seed=_stable_seed((calibration_seed_namespace, int(seed), "states")),
        ))
    return tuple(domains), grounders, tuple(examples)


def _source_value_diagnostic(
    train: Sequence[MatchedValueExample],
    evaluation: Sequence[MatchedValueExample],
    model_config: Mapping[str, Any],
) -> dict[str, float]:
    reports = {}
    for name, rows in (
        ("authentic", train),
        ("shuffled", shuffled_value_control(train, seed=int(model_config["control_seed"]))),
        ("marginal", marginal_value_control(train)),
    ):
        model = fit_value_ensemble(
            rows,
            (),
            seed=int(model_config["seed"]),
            ensemble_size=int(model_config["ensemble_size"]),
            alpha=float(model_config["ridge_alpha"]),
            target_mass=float(model_config["target_mass"]),
        )
        assert model is not None
        features = [row.features for row in evaluation]
        predictions, _ = model.predict(features)
        labels = np.asarray([row.value for row in evaluation], dtype=np.float64)
        reports[name] = float(np.mean((predictions - labels) ** 2))
    return reports


def run_controlled_transfer(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run discovery or frozen controlled transfer splits from a JSON config."""

    domain_config = config["domain"]
    source_config = config["source"]
    target_config = config["target"]
    model_config = config["model"]
    policy_config = config["policy"]
    source_seed_sets = [
        set(map(int, source_config["train_domain_seeds"])),
        set(map(int, source_config["evaluation_domain_seeds"])),
    ]
    target_split_seeds = {
        name: set(map(int, seeds))
        for name, seeds in target_config["evaluation_domain_seeds"].items()
    }
    all_seed_sets = source_seed_sets + [
        set(map(int, target_config["support_domain_seeds"])),
        *target_split_seeds.values(),
    ]
    for left in range(len(all_seed_sets)):
        for right in range(left + 1, len(all_seed_sets)):
            if all_seed_sets[left] & all_seed_sets[right]:
                raise ValueError("source/target/split domain seeds must be disjoint")

    _, _, source_train = _build_domains_and_examples(
        source_config["train_domain_seeds"],
        surface="game",
        domain_config=domain_config,
        state_count=int(source_config["states_per_domain"]),
        calibration_seed_namespace="source-train",
    )
    _, _, source_evaluation = _build_domains_and_examples(
        source_config["evaluation_domain_seeds"],
        surface="game",
        domain_config=domain_config,
        state_count=int(source_config["states_per_domain"]),
        calibration_seed_namespace="source-evaluation",
    )
    _, _, target_support = _build_domains_and_examples(
        target_config["support_domain_seeds"],
        surface="diagnosis",
        domain_config=domain_config,
        state_count=int(target_config["support_states_per_domain"]),
        calibration_seed_namespace="target-support",
    )
    source_controls = {
        "authentic_source_plus_target": tuple(source_train),
        "shuffled_source_plus_target": shuffled_value_control(
            source_train, seed=int(model_config["control_seed"]),
        ),
        "source_marginal_plus_target": marginal_value_control(source_train),
    }
    source_diagnostic = _source_value_diagnostic(
        source_train, source_evaluation, model_config,
    )

    split_reports: dict[str, Any] = {}
    episode_rows: dict[str, Any] = {}
    for split, seeds in target_config["evaluation_domain_seeds"].items():
        domains, grounders, _ = _build_domains_and_examples(
            sorted(seeds),
            surface="diagnosis",
            domain_config=domain_config,
            state_count=1,
            calibration_seed_namespace=f"target-{split}",
        )
        split_report: dict[str, Any] = {}
        split_grounder_mse = float(np.mean([
            np.mean((grounders[domain.domain_id] - domain.likelihood_array) ** 2)
            for domain in domains
        ]))
        split_episode_rows: dict[str, Any] = {}
        for support_k in map(int, target_config["support_k"]):
            support = _select_states(target_support, support_k)
            support_state_count = len({row.state_id for row in support})
            residual_scale = 0.0
            if model_config["kind"] == "source_prior_target_residual_ensemble":
                warmup = float(model_config["residual_full_strength_states"])
                if warmup <= 0:
                    raise ValueError("residual_full_strength_states must be positive")
                residual_scale = float(model_config["maximum_residual_scale"]) * min(
                    1.0, support_state_count / warmup,
                )
            condition_rows: dict[str, tuple[EpisodeResult, ...]] = {}
            condition_summary: dict[str, Any] = {}
            for condition in CONDITIONS:
                source_rows = () if condition == "target_only" else source_controls[condition]
                model_seed = _stable_seed((
                    int(model_config["seed"]), split, support_k, condition,
                ))
                if (
                    model_config["kind"] == "source_prior_target_residual_ensemble"
                    and condition != "target_only"
                ):
                    model = fit_source_prior_residual_ensemble(
                        source_rows,
                        support,
                        seed=model_seed,
                        ensemble_size=int(model_config["ensemble_size"]),
                        source_alpha=float(model_config["ridge_alpha"]),
                        residual_alpha=float(model_config["residual_ridge_alpha"]),
                        residual_scale=residual_scale,
                    )
                else:
                    model = fit_value_ensemble(
                        source_rows,
                        support,
                        seed=model_seed,
                        ensemble_size=int(model_config["ensemble_size"]),
                        alpha=float(model_config["ridge_alpha"]),
                        target_mass=float(model_config["target_mass"]),
                    )
                rows = evaluate_model(
                    domains,
                    grounders,
                    model,
                    condition=condition,
                    episode_seeds=target_config["episode_seeds"],
                    policy_config=policy_config,
                )
                condition_rows[condition] = rows
                condition_summary[condition] = summarize_episodes(rows)
                split_episode_rows[f"k{support_k}:{condition}"] = [
                    row.__dict__ for row in rows
                ]
            comparisons = {}
            authentic = condition_rows["authentic_source_plus_target"]
            for control in (
                "target_only", "shuffled_source_plus_target", "source_marginal_plus_target",
            ):
                comparisons[control] = {
                    metric: paired_bootstrap_delta(
                        authentic,
                        condition_rows[control],
                        metric=metric,
                        seed=_stable_seed((split, support_k, control, metric, "bootstrap")),
                        samples=int(config["gate"]["bootstrap_samples"]),
                    )
                    for metric in ("success", "net_return")
                }
            split_report[f"k{support_k}"] = {
                "target_grounder_audit": {
                    "kind": str(domain_config.get("target_grounder", {}).get(
                        "kind", "beta_binomial"
                    )),
                    "mse_vs_hidden_environment_likelihood_audit_only": (
                        split_grounder_mse
                    ),
                    "hidden_environment_likelihood_consumed_by_policy": False,
                },
                "adaptation": {
                    "kind": model_config["kind"],
                    "target_support_states": support_state_count,
                    "residual_scale": residual_scale,
                },
                "conditions": condition_summary,
                "authentic_paired_deltas": comparisons,
            }
        split_reports[split] = split_report
        episode_rows[split] = split_episode_rows

    gate_rows = []
    gate_config = config["gate"]
    requirements = gate_config.get("requirements")
    if requirements is None:
        requirements = ({
            "name": "legacy_superiority",
            "required_splits": gate_config["required_splits"],
            "required_k": gate_config["required_k"],
            "controls": gate_config["authentic_must_beat"],
            "minimum_mean_net_return_delta": gate_config[
                "minimum_mean_net_return_delta"
            ],
            "minimum_ci95_net_return_delta": gate_config[
                "minimum_ci95_net_return_delta"
            ],
        },)
    for requirement in requirements:
        metric = str(requirement.get("metric", "net_return"))
        if metric not in {"net_return", "success"}:
            raise ValueError(f"unsupported gate metric: {metric}")
        minimum_mean = float(requirement.get(
            f"minimum_mean_{metric}_delta",
            requirement.get("minimum_mean_net_return_delta", -math.inf),
        ))
        minimum_ci95 = float(requirement.get(
            f"minimum_ci95_{metric}_delta",
            requirement.get("minimum_ci95_net_return_delta", -math.inf),
        ))
        for split in requirement["required_splits"]:
            for support_k in requirement["required_k"]:
                cell = split_reports[str(split)][f"k{int(support_k)}"]
                for control in requirement["controls"]:
                    delta = cell["authentic_paired_deltas"][str(control)][metric]
                    passed = (
                        delta["mean_delta"] > minimum_mean
                        and delta["ci95_low"] > minimum_ci95
                    )
                    gate_rows.append({
                        "requirement": str(requirement["name"]),
                        "split": split,
                        "support_k": int(support_k),
                        "control": control,
                        "metric": metric,
                        "minimum_mean_net_return_delta": minimum_mean,
                        "minimum_ci95_net_return_delta": minimum_ci95,
                        "passed": bool(passed),
                        **delta,
                    })
    comparison_gate_passed = bool(gate_rows) and all(
        row["passed"] for row in gate_rows
    )
    source_tokens = {
        token
        for seed in source_config["train_domain_seeds"]
        for token in make_domain(
            seed=int(seed), surface="game",
            hypothesis_count=int(domain_config["hypothesis_count"]),
            test_count=int(domain_config["test_count"]),
            max_tests=int(domain_config["max_tests"]),
            test_cost=float(domain_config["test_cost"]),
        ).hypothesis_tokens
        + make_domain(
            seed=int(seed), surface="game",
            hypothesis_count=int(domain_config["hypothesis_count"]),
            test_count=int(domain_config["test_count"]),
            max_tests=int(domain_config["max_tests"]),
            test_cost=float(domain_config["test_cost"]),
        ).test_tokens
    }
    target_seeds = list(map(int, target_config["support_domain_seeds"])) + [
        int(seed)
        for seeds in target_config["evaluation_domain_seeds"].values()
        for seed in seeds
    ]
    target_tokens = {
        token
        for seed in target_seeds
        for token in make_domain(
            seed=int(seed), surface="diagnosis",
            hypothesis_count=int(domain_config["hypothesis_count"]),
            test_count=int(domain_config["test_count"]),
            max_tests=int(domain_config["max_tests"]),
            test_cost=float(domain_config["test_cost"]),
        ).hypothesis_tokens
        + make_domain(
            seed=int(seed), surface="diagnosis",
            hypothesis_count=int(domain_config["hypothesis_count"]),
            test_count=int(domain_config["test_count"]),
            max_tests=int(domain_config["max_tests"]),
            test_cost=float(domain_config["test_cost"]),
        ).test_tokens
    }
    invariant_config = gate_config.get("invariants", {})
    grounder_mse_values = [
        float(cell[f"k{int(support_k)}"]["target_grounder_audit"][
            "mse_vs_hidden_environment_likelihood_audit_only"
        ])
        for cell in split_reports.values()
        for support_k in target_config["support_k"]
    ]
    invariant_rows = {
        "zero_shared_raw_tokens": (
            not (source_tokens & target_tokens)
            if invariant_config.get("require_zero_shared_raw_tokens") else True
        ),
        "target_grounder_kind": (
            str(domain_config.get("target_grounder", {}).get(
                "kind", "beta_binomial"
            )) == str(invariant_config["require_target_grounder_kind"])
            if "require_target_grounder_kind" in invariant_config else True
        ),
        "maximum_target_grounder_mse": (
            max(grounder_mse_values) <= float(
                invariant_config["maximum_target_grounder_mse"]
            )
            if "maximum_target_grounder_mse" in invariant_config else True
        ),
        "source_authentic_mse_less_than_controls": (
            all(
                source_diagnostic["authentic"] < source_diagnostic[str(control)]
                for control in invariant_config.get(
                    "source_authentic_mse_strictly_less_than", []
                )
            )
        ),
        "minimum_source_train_examples": (
            len(source_train) >= int(
                invariant_config.get("minimum_source_train_examples", 0)
            )
        ),
    }
    invariant_gate_passed = all(invariant_rows.values())
    gate_passed = comparison_gate_passed and invariant_gate_passed
    schema_version = int(config.get("schema_version", 1))
    return {
        "schema_version": schema_version,
        "experiment": f"CONTROLLED_INTERVENTION_GROUNDED_TRANSFER_V{schema_version}",
        "status": "SUPPORTED" if gate_passed else "NOT_SUPPORTED",
        "claim_boundary": config["claim_boundary"],
        "feature_names": list(FEATURE_NAMES),
        "conditions": list(CONDITIONS),
        "surface_audit": {
            "source_surface": "game",
            "target_surface": "diagnosis",
            "shared_raw_tokens": sorted(source_tokens & target_tokens),
        },
        "data_counts": {
            "source_train_examples": len(source_train),
            "source_evaluation_examples": len(source_evaluation),
            "target_support_examples": len(target_support),
            "target_support_states": len({row.state_id for row in target_support}),
        },
        "target_grounder": dict(domain_config.get(
            "target_grounder", {"kind": "beta_binomial"}
        )),
        "source_value_mse": source_diagnostic,
        "splits": split_reports,
        "gate": {
            "passed": gate_passed,
            "comparison_gate_passed": comparison_gate_passed,
            "invariant_gate_passed": invariant_gate_passed,
            "comparisons": gate_rows,
            "invariants": invariant_rows,
            "invariant_specification": dict(invariant_config),
        },
        "episode_rows": episode_rows,
    }


__all__ = [
    "CONDITIONS",
    "FEATURE_NAMES",
    "AbstractAction",
    "ActiveIdentificationDomain",
    "EpisodeResult",
    "GroundedState",
    "MatchedValueExample",
    "RidgeValueModel",
    "ResidualValueEnsemble",
    "ValueEnsemble",
    "action_features",
    "calibrate_target_grounder",
    "calibrate_target_neural_grounder",
    "choose_neurosymbolic_action",
    "collect_matched_examples",
    "evaluate_model",
    "fit_value_ensemble",
    "fit_source_prior_residual_ensemble",
    "make_domain",
    "marginal_value_control",
    "matched_action_values",
    "paired_bootstrap_delta",
    "run_controlled_transfer",
    "run_episode",
    "shuffled_value_control",
    "summarize_episodes",
    "update_belief",
]
