from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


OPTION_NAMES = ("SEARCH", "ACQUIRE", "TRANSFORM", "PLACE", "VERIFY")
FEATURE_NAMES = (
    *(f"option_{name.lower()}" for name in OPTION_NAMES),
    *(f"required_{name.lower()}" for name in OPTION_NAMES),
    "matches_required_option",
    "precondition_satisfied",
    "predicted_completion_probability",
    "goal_binding_probability",
    "remaining_budget_fraction",
    "workflow_progress_fraction",
    "action_repeat_fraction",
    "predicted_noop_probability",
    "stage_urgency",
    "match_x_completion",
    "match_x_binding",
    "match_x_precondition",
    "failure_cost",
)


def stable_seed(value: object) -> int:
    return int(hashlib.sha256(repr(value).encode()).hexdigest()[:16], 16) % (2**32)


@dataclass(frozen=True)
class HierarchicalValueExample:
    state_id: str
    option: str
    features: tuple[float, ...]
    value: float


@dataclass(frozen=True)
class HierarchicalRidgeModel:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError("hierarchical value-model feature shape mismatch")
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        design = np.column_stack(((matrix - mean) / scale, np.ones(len(matrix))))
        return design @ np.asarray(self.coefficients, dtype=np.float64)


@dataclass(frozen=True)
class HierarchicalValueEnsemble:
    models: tuple[HierarchicalRidgeModel, ...]

    def predict(self, features: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise ValueError("cannot predict with an empty ensemble")
        predictions = np.asarray([model.predict(features) for model in self.models])
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)


def option_features(
    *,
    option: str,
    required_option: str,
    precondition_satisfied: float,
    completion_probability: float,
    goal_binding_probability: float,
    remaining_budget_fraction: float,
    workflow_progress_fraction: float,
    action_repeat_fraction: float,
    noop_probability: float,
    stage_urgency: float,
    failure_cost: float,
) -> tuple[float, ...]:
    if option not in OPTION_NAMES or required_option not in OPTION_NAMES:
        raise ValueError("unknown hierarchical option")
    match = float(option == required_option)
    values = (
        *(float(option == name) for name in OPTION_NAMES),
        *(float(required_option == name) for name in OPTION_NAMES),
        match,
        float(precondition_satisfied),
        float(completion_probability),
        float(goal_binding_probability),
        float(remaining_budget_fraction),
        float(workflow_progress_fraction),
        float(action_repeat_fraction),
        float(noop_probability),
        float(stage_urgency),
        match * float(completion_probability),
        match * float(goal_binding_probability),
        match * float(precondition_satisfied),
        float(failure_cost),
    )
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError("hierarchical feature contract drift")
    return tuple(values)


def _workflow(rng: np.random.Generator) -> tuple[str, ...]:
    pattern = int(rng.integers(0, 5))
    if pattern == 0:
        return ("SEARCH", "ACQUIRE", "PLACE")
    if pattern == 1:
        return ("SEARCH", "ACQUIRE", "TRANSFORM", "PLACE")
    if pattern == 2:
        return ("SEARCH", "TRANSFORM", "VERIFY")
    if pattern == 3:
        return ("SEARCH", "ACQUIRE", "PLACE", "SEARCH", "ACQUIRE", "PLACE")
    return ("SEARCH", "ACQUIRE", "TRANSFORM", "PLACE", "VERIFY")


def _optimal_values(
    workflow: Sequence[str],
    completion: Mapping[str, float],
    costs: Mapping[str, float],
    maximum_budget: int,
    progress_reward: float,
    invalid_option_cost: float,
) -> tuple[tuple[float, ...], ...]:
    stage_count = len(workflow)
    values = np.zeros((stage_count + 1, maximum_budget + 1), dtype=np.float64)
    values[stage_count, :] = 1.0
    for budget in range(1, maximum_budget + 1):
        for stage in range(stage_count - 1, -1, -1):
            candidates = []
            for option in OPTION_NAMES:
                cost = float(costs[option])
                if option == workflow[stage]:
                    probability = float(completion[option])
                    candidate = (
                        -cost
                        + probability * (
                            progress_reward + values[stage + 1, budget - 1]
                        )
                        + (1.0 - probability) * values[stage, budget - 1]
                    )
                else:
                    candidate = -cost - invalid_option_cost + values[stage, budget - 1]
                candidates.append(candidate)
            values[stage, budget] = max(candidates)
    return tuple(tuple(map(float, row)) for row in values)


def collect_source_examples(
    *,
    surfaces: Sequence[str],
    domains_per_surface: int,
    states_per_domain: int,
    seed: int,
    minimum_budget: int,
    maximum_budget: int,
    completion_probability_range: Sequence[float],
    failure_cost_range: Sequence[float],
    progress_reward: float = 0.15,
    invalid_option_cost: float = 0.18,
) -> tuple[HierarchicalValueExample, ...]:
    rows: list[HierarchicalValueExample] = []
    for surface in surfaces:
        for domain_index in range(domains_per_surface):
            domain_seed = stable_seed((seed, surface, domain_index, "hierarchical-source"))
            rng = np.random.default_rng(domain_seed)
            workflow = _workflow(rng)
            completions = {
                option: float(rng.uniform(*completion_probability_range))
                for option in OPTION_NAMES
            }
            costs = {
                option: float(rng.uniform(*failure_cost_range))
                for option in OPTION_NAMES
            }
            table = _optimal_values(
                workflow,
                completions,
                costs,
                maximum_budget,
                progress_reward,
                invalid_option_cost,
            )
            for state_index in range(states_per_domain):
                stage = int(rng.integers(0, len(workflow)))
                budget = int(rng.integers(minimum_budget, maximum_budget + 1))
                state_id = f"{surface}:{domain_index}:{state_index}"
                for option in OPTION_NAMES:
                    repeat = float(rng.integers(0, 5)) / 4.0
                    # Target-native affordances must not leak the symbolic phase.
                    # The same seemingly executable option can be useful in one
                    # workflow state and a no-op with respect to progress in another.
                    binding = float(rng.beta(4, 3))
                    precondition = float(rng.beta(5, 3))
                    completion = completions[option] * (0.5 + 0.5 * precondition)
                    noop = 1.0 - completion
                    if option == workflow[stage]:
                        q_value = (
                            -costs[option]
                            + completion * (
                                progress_reward + table[stage + 1][budget - 1]
                            )
                            + (1.0 - completion) * table[stage][budget - 1]
                        )
                    else:
                        q_value = (
                            -costs[option]
                            - invalid_option_cost
                            + table[stage][budget - 1]
                        )
                    rows.append(HierarchicalValueExample(
                        state_id=state_id,
                        option=option,
                        features=option_features(
                            option=option,
                            required_option=workflow[stage],
                            precondition_satisfied=precondition,
                            completion_probability=completion,
                            goal_binding_probability=binding,
                            remaining_budget_fraction=budget / maximum_budget,
                            workflow_progress_fraction=stage / max(1, len(workflow) - 1),
                            action_repeat_fraction=repeat,
                            noop_probability=noop,
                            stage_urgency=(len(workflow) - stage) / budget,
                            failure_cost=costs[option],
                        ),
                        value=float(q_value),
                    ))
    return tuple(rows)


def _fit_ridge(
    rows: Sequence[HierarchicalValueExample], alpha: float
) -> HierarchicalRidgeModel:
    matrix = np.asarray([row.features for row in rows], dtype=np.float64)
    labels = np.asarray([row.value for row in rows], dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack(((matrix - mean) / scale, np.ones(len(matrix))))
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ labels)
    return HierarchicalRidgeModel(
        tuple(map(float, mean)),
        tuple(map(float, scale)),
        tuple(map(float, coefficients)),
    )


def fit_value_ensemble(
    rows: Sequence[HierarchicalValueExample],
    *,
    seed: int,
    ensemble_size: int,
    alpha: float,
) -> HierarchicalValueEnsemble:
    grouped: dict[str, list[HierarchicalValueExample]] = {}
    for row in rows:
        grouped.setdefault(row.state_id, []).append(row)
    state_ids = sorted(grouped)
    models = []
    for member in range(ensemble_size):
        rng = np.random.default_rng(stable_seed((seed, member, "hierarchical-bootstrap")))
        sampled = rng.choice(state_ids, size=len(state_ids), replace=True)
        boot = [row for state_id in sampled for row in grouped[str(state_id)]]
        models.append(_fit_ridge(boot, alpha))
    return HierarchicalValueEnsemble(tuple(models))


def shuffled_value_control(
    rows: Sequence[HierarchicalValueExample], *, seed: int
) -> tuple[HierarchicalValueExample, ...]:
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[HierarchicalValueExample]] = {}
    for row in rows:
        grouped.setdefault(row.state_id, []).append(row)
    result = []
    for state_id in sorted(grouped):
        group = grouped[state_id]
        values = rng.permutation([row.value for row in group])
        result.extend(
            HierarchicalValueExample(row.state_id, row.option, row.features, float(value))
            for row, value in zip(group, values)
        )
    return tuple(result)


def marginal_value_control(
    rows: Sequence[HierarchicalValueExample],
) -> tuple[HierarchicalValueExample, ...]:
    mean = float(np.mean([row.value for row in rows]))
    return tuple(
        HierarchicalValueExample(row.state_id, row.option, row.features, mean)
        for row in rows
    )


def phase_permuted_control(
    rows: Sequence[HierarchicalValueExample],
) -> tuple[HierarchicalValueExample, ...]:
    option_start = 0
    required_start = len(OPTION_NAMES)
    match_index = 2 * len(OPTION_NAMES)
    result = []
    for row in rows:
        values = list(row.features)
        required = values[required_start : required_start + len(OPTION_NAMES)]
        values[required_start : required_start + len(OPTION_NAMES)] = required[1:] + required[:1]
        option_index = int(np.argmax(values[option_start : option_start + len(OPTION_NAMES)]))
        required_index = int(np.argmax(values[required_start : required_start + len(OPTION_NAMES)]))
        match = float(option_index == required_index)
        values[match_index] = match
        values[match_index + 9] = match * values[match_index + 2]
        values[match_index + 10] = match * values[match_index + 3]
        values[match_index + 11] = match * values[match_index + 1]
        result.append(HierarchicalValueExample(
            row.state_id, row.option, tuple(values), row.value,
        ))
    return tuple(result)


def serialize_ensemble(model: HierarchicalValueEnsemble) -> dict[str, Any]:
    return {
        "kind": "hierarchical-ridge-value-ensemble-v2",
        "feature_names": list(FEATURE_NAMES),
        "models": [asdict(member) for member in model.models],
    }


def deserialize_ensemble(payload: Mapping[str, Any]) -> HierarchicalValueEnsemble:
    if tuple(payload["feature_names"]) != FEATURE_NAMES:
        raise ValueError("hierarchical feature contract mismatch")
    return HierarchicalValueEnsemble(tuple(
        HierarchicalRidgeModel(
            tuple(map(float, row["feature_mean"])),
            tuple(map(float, row["feature_scale"])),
            tuple(map(float, row["coefficients"])),
        )
        for row in payload["models"]
    ))


__all__ = [
    "FEATURE_NAMES",
    "OPTION_NAMES",
    "HierarchicalValueEnsemble",
    "HierarchicalValueExample",
    "collect_source_examples",
    "deserialize_ensemble",
    "fit_value_ensemble",
    "marginal_value_control",
    "option_features",
    "phase_permuted_control",
    "serialize_ensemble",
    "shuffled_value_control",
]
