from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .hierarchical_skill_transfer import (
    FEATURE_NAMES,
    HierarchicalValueExample,
    stable_seed,
)


PAIRWISE_FEATURE_NAMES = (
    *(f"delta_{name}" for name in FEATURE_NAMES),
    *(f"squared_delta_{name}" for name in FEATURE_NAMES),
)


@dataclass(frozen=True)
class PairwiseAdvantageExample:
    state_id: str
    surface: str
    domain_id: str
    left_option: str
    right_option: str
    features: tuple[float, ...]
    advantage: float


@dataclass(frozen=True)
class PairwiseRidgeModel:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(PAIRWISE_FEATURE_NAMES):
            raise ValueError("pairwise advantage feature shape mismatch")
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        design = np.column_stack(((matrix - mean) / scale, np.ones(len(matrix))))
        return design @ np.asarray(self.coefficients, dtype=np.float64)


@dataclass(frozen=True)
class PairwiseAdvantageEnsemble:
    models: tuple[PairwiseRidgeModel, ...]

    def predict(
        self, features: Sequence[Sequence[float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise ValueError("cannot predict with an empty pairwise ensemble")
        predictions = np.asarray([model.predict(features) for model in self.models])
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)


def pairwise_features(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, ...]:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != (len(FEATURE_NAMES),) or right_values.shape != (
        len(FEATURE_NAMES),
    ):
        raise ValueError("option feature shape mismatch")
    delta = left_values - right_values
    squared_delta = left_values**2 - right_values**2
    return tuple(map(float, np.concatenate((delta, squared_delta))))


def _state_groups(
    rows: Sequence[HierarchicalValueExample],
) -> dict[str, tuple[HierarchicalValueExample, ...]]:
    grouped: dict[str, list[HierarchicalValueExample]] = {}
    for row in rows:
        grouped.setdefault(row.state_id, []).append(row)
    return {
        state_id: tuple(sorted(group, key=lambda row: row.option))
        for state_id, group in grouped.items()
    }


def pairwise_examples(
    rows: Sequence[HierarchicalValueExample],
) -> tuple[PairwiseAdvantageExample, ...]:
    output = []
    for state_id, group in sorted(_state_groups(rows).items()):
        parts = state_id.split(":")
        if len(parts) not in {3, 4}:
            raise ValueError(f"unexpected source state identity: {state_id}")
        surface, domain_index, _state_index = parts[-3:]
        domain_id = f"{surface}:{domain_index}"
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                output.append(PairwiseAdvantageExample(
                    state_id=state_id,
                    surface=surface,
                    domain_id=domain_id,
                    left_option=left.option,
                    right_option=right.option,
                    features=pairwise_features(left.features, right.features),
                    advantage=float(left.value - right.value),
                ))
    return tuple(output)


def phase_blind_rows(
    rows: Sequence[HierarchicalValueExample],
) -> tuple[HierarchicalValueExample, ...]:
    required_start = 5
    match_index = 10
    output = []
    for row in rows:
        values = list(row.features)
        values[required_start:required_start + 5] = [0.0] * 5
        values[match_index] = 0.0
        values[match_index + 9:match_index + 12] = [0.0] * 3
        output.append(HierarchicalValueExample(
            state_id=row.state_id,
            option=row.option,
            features=tuple(values),
            value=row.value,
        ))
    return tuple(output)


def intervention_grounded_rows(
    rows: Sequence[HierarchicalValueExample],
    *,
    probe_trials: int,
    probe_seed: int,
) -> tuple[HierarchicalValueExample, ...]:
    if probe_trials < 2:
        raise ValueError("intervention probe requires at least two trials")
    output = []
    for row in rows:
        values = list(row.features)
        oracle_match = float(values[10])
        native_completion = float(values[12])
        causal_effect_probability = oracle_match * native_completion
        rng = np.random.default_rng(stable_seed((
            probe_seed, row.state_id, row.option, "matched-option-probes"
        )))
        successes = int(rng.binomial(probe_trials, causal_effect_probability))
        # Beta(1, 1) posterior mean is receipt-derived and remains finite for
        # options with no observed effects. The latent required phase is never
        # serialized into the returned feature vector.
        effect_rate = (successes + 1.0) / (probe_trials + 2.0)
        values[5:10] = [0.0] * 5
        values[10] = 0.0
        values[11] = effect_rate
        values[12] = effect_rate
        values[15] = 0.0
        values[17] = 1.0 - effect_rate
        values[18] = 0.0
        values[19:22] = [0.0] * 3
        output.append(HierarchicalValueExample(
            state_id=row.state_id,
            option=row.option,
            features=tuple(values),
            value=row.value,
        ))
    return tuple(output)


def effect_blind_rows(
    rows: Sequence[HierarchicalValueExample],
) -> tuple[HierarchicalValueExample, ...]:
    output = []
    for row in rows:
        values = list(row.features)
        values[11] = 0.0
        values[12] = 0.0
        values[17] = 0.0
        output.append(replace_example(row, tuple(values)))
    return tuple(output)


def replace_example(
    row: HierarchicalValueExample, features: tuple[float, ...]
) -> HierarchicalValueExample:
    return HierarchicalValueExample(
        state_id=row.state_id,
        option=row.option,
        features=features,
        value=row.value,
    )


def within_state_effect_permutation(
    rows: Sequence[HierarchicalValueExample],
) -> tuple[HierarchicalValueExample, ...]:
    output = []
    for _state_id, group in sorted(_state_groups(rows).items()):
        rotated = group[1:] + group[:1]
        for row, donor in zip(group, rotated):
            values = list(row.features)
            values[11] = donor.features[11]
            values[12] = donor.features[12]
            values[17] = donor.features[17]
            output.append(replace_example(row, tuple(values)))
    return tuple(output)


def _fit_ridge(
    rows: Sequence[PairwiseAdvantageExample], alpha: float
) -> PairwiseRidgeModel:
    matrix = np.asarray([row.features for row in rows], dtype=np.float64)
    labels = np.asarray([row.advantage for row in rows], dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack(((matrix - mean) / scale, np.ones(len(matrix))))
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ labels)
    return PairwiseRidgeModel(
        feature_mean=tuple(map(float, mean)),
        feature_scale=tuple(map(float, scale)),
        coefficients=tuple(map(float, coefficients)),
    )


def fit_pairwise_ensemble(
    rows: Sequence[PairwiseAdvantageExample],
    *,
    seed: int,
    ensemble_size: int,
    alpha: float,
) -> PairwiseAdvantageEnsemble:
    grouped: dict[str, list[PairwiseAdvantageExample]] = {}
    for row in rows:
        grouped.setdefault(row.state_id, []).append(row)
    state_ids = sorted(grouped)
    models = []
    for member in range(ensemble_size):
        rng = np.random.default_rng(stable_seed((seed, member, "pairwise-bootstrap")))
        sampled = rng.choice(state_ids, size=len(state_ids), replace=True)
        boot = [row for state_id in sampled for row in grouped[str(state_id)]]
        models.append(_fit_ridge(boot, alpha))
    return PairwiseAdvantageEnsemble(tuple(models))


def conformal_error_quantile(
    model: PairwiseAdvantageEnsemble,
    rows: Sequence[PairwiseAdvantageExample],
    *,
    alpha: float,
) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("conformal alpha must be between zero and one")
    predicted, _ = model.predict([row.features for row in rows])
    truth = np.asarray([row.advantage for row in rows], dtype=np.float64)
    overprediction = predicted - truth
    rank = min(
        len(overprediction) - 1,
        int(np.ceil((len(overprediction) + 1) * (1.0 - alpha))) - 1,
    )
    return float(np.sort(overprediction)[max(0, rank)])


def option_scores(
    model: PairwiseAdvantageEnsemble,
    option_features_by_name: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    options = sorted(option_features_by_name)
    scores = {option: 0.0 for option in options}
    for left_index, left in enumerate(options):
        for right in options[left_index + 1 :]:
            features = pairwise_features(
                option_features_by_name[left], option_features_by_name[right]
            )
            prediction, _ = model.predict([features])
            advantage = float(prediction[0])
            scores[left] += advantage
            scores[right] -= advantage
    return scores


def choose_option_against_fallback(
    model: PairwiseAdvantageEnsemble,
    option_features_by_name: Mapping[str, Sequence[float]],
    *,
    fallback_option: str,
    conformal_error: float,
) -> dict[str, Any]:
    if fallback_option not in option_features_by_name:
        raise ValueError("fallback option is not target-actionable")
    candidates = sorted(option_features_by_name)
    comparisons = []
    for option in candidates:
        if option == fallback_option:
            continue
        features = pairwise_features(
            option_features_by_name[option],
            option_features_by_name[fallback_option],
        )
        mean, deviation = model.predict([features])
        lower = float(mean[0] - conformal_error)
        comparisons.append({
            "option": option,
            "predicted_advantage": float(mean[0]),
            "ensemble_deviation": float(deviation[0]),
            "conformal_lower_bound": lower,
        })
    best = max(
        comparisons,
        key=lambda row: (row["conformal_lower_bound"], row["option"]),
    )
    admitted = best["conformal_lower_bound"] > 0.0
    return {
        "option": best["option"] if admitted else fallback_option,
        "fallback_option": fallback_option,
        "source_admitted": admitted,
        "comparison": best,
        "all_comparisons": comparisons,
    }


def serialize_pairwise_ensemble(model: PairwiseAdvantageEnsemble) -> dict[str, Any]:
    return {
        "kind": "pairwise-relative-option-advantage-ensemble-v16",
        "feature_names": list(PAIRWISE_FEATURE_NAMES),
        "models": [asdict(member) for member in model.models],
    }


def deserialize_pairwise_ensemble(
    payload: Mapping[str, Any],
) -> PairwiseAdvantageEnsemble:
    if tuple(payload["feature_names"]) != PAIRWISE_FEATURE_NAMES:
        raise ValueError("pairwise feature contract mismatch")
    return PairwiseAdvantageEnsemble(tuple(
        PairwiseRidgeModel(
            feature_mean=tuple(map(float, row["feature_mean"])),
            feature_scale=tuple(map(float, row["feature_scale"])),
            coefficients=tuple(map(float, row["coefficients"])),
        )
        for row in payload["models"]
    ))


__all__ = [
    "PAIRWISE_FEATURE_NAMES",
    "PairwiseAdvantageEnsemble",
    "PairwiseAdvantageExample",
    "choose_option_against_fallback",
    "conformal_error_quantile",
    "deserialize_pairwise_ensemble",
    "effect_blind_rows",
    "fit_pairwise_ensemble",
    "intervention_grounded_rows",
    "option_scores",
    "pairwise_examples",
    "pairwise_features",
    "phase_blind_rows",
    "serialize_pairwise_ensemble",
    "within_state_effect_permutation",
]
