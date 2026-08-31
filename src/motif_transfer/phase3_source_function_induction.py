"""Source-only induction of domain-specific functions in a shared IR.

The older typed-effect artifact selected one of four temporal measurements and
then delegated every source to the same attempt loop.  This module keeps the
shared measurement vocabulary but learns the *program body* from source
intervention sets:

* a sparse convex function over temporally typed effects;
* the observation horizon implied by that function;
* whether a second ranked intervention recovered discovery failures; and
* an explicit HIGH/LOW transition graph with fail-closed abstention.

The hypothesis class and executor contract are shared.  Coefficients, horizon,
retry edge, and qualification are source-content derived.  Game identity,
native action strings, and target data are never accepted as features.
"""

from __future__ import annotations

from itertools import permutations, product
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .contracts import stable_hash
from .phase3_typed_effect_induction import (
    IMMEDIATE_EFFECT,
    MEDIUM_EFFECT,
    PERSISTENCE_EFFECT,
    SHORT_EFFECT,
    TYPED_EFFECTS,
    TypedInterventionSet,
)


# Kept local to the source IR so target adapters do not author temporal
# semantics.  These are measurement endpoints already present in source
# receipts, not named policy templates.
EFFECT_ENDPOINTS = {
    IMMEDIATE_EFFECT: 1,
    SHORT_EFFECT: 4,
    MEDIUM_EFFECT: 8,
    PERSISTENCE_EFFECT: 8,
}
WEIGHT_DENOMINATOR = 4
QUALIFIED = "SOURCE_DOMAIN_FUNCTION_QUALIFIED"
ABSTAINING = "SOURCE_DOMAIN_FUNCTION_ABSTENTION_INDUCED"


def _weight_grid() -> tuple[tuple[float, ...], ...]:
    values = []
    for counts in product(range(WEIGHT_DENOMINATOR + 1), repeat=len(TYPED_EFFECTS)):
        if sum(counts) != WEIGHT_DENOMINATOR:
            continue
        values.append(tuple(value / WEIGHT_DENOMINATOR for value in counts))
    return tuple(values)


FUNCTION_WEIGHT_GRID = _weight_grid()


def _checked_effect_vector(row: Mapping[str, Any]) -> tuple[float, ...]:
    if set(row) != set(TYPED_EFFECTS):
        raise ValueError("candidate effects do not match shared function IR")
    values = []
    for effect_type in TYPED_EFFECTS:
        value = row[effect_type]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("candidate effect is not numeric")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("candidate effect is outside [0,1]")
        values.append(value)
    return tuple(values)


def function_weights(program: Mapping[str, Any]) -> tuple[float, ...]:
    function = program.get("source_function")
    if not isinstance(function, Mapping):
        raise ValueError("source-domain program omitted source_function")
    terms = function.get("terms")
    if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
        raise ValueError("source-domain function omitted terms")
    by_type = {str(row["effect_type"]): float(row["weight"]) for row in terms}
    if set(by_type) != {
        effect_type for effect_type, value in by_type.items() if value > 0
    }:
        raise ValueError("source-domain function has a nonpositive term")
    if not set(by_type) <= set(TYPED_EFFECTS):
        raise ValueError("source-domain function uses an unknown effect type")
    values = tuple(by_type.get(effect_type, 0.0) for effect_type in TYPED_EFFECTS)
    if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ValueError("source-domain function weights do not sum to one")
    return values


def score_effects(
    program: Mapping[str, Any], candidate_effects: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    weights = function_weights(program)
    return tuple(
        sum(weight * value for weight, value in zip(weights, _checked_effect_vector(row)))
        for row in candidate_effects
    )


def function_trial_order(
    program: Mapping[str, Any], candidate_effects: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, ...], str | None]:
    validate_source_function_program(program)
    if program.get("status") != QUALIFIED:
        return (), "SOURCE_DOMAIN_FUNCTION_NOT_QUALIFIED"
    if len(candidate_effects) < 2:
        return (), "TARGET_CANDIDATE_SET_HAS_FEWER_THAN_TWO"
    try:
        values = score_effects(program, candidate_effects)
    except ValueError as error:
        return (), f"TARGET_FUNCTION_EFFECT_SCHEMA_INVALID:{error}"
    maximum = max(values)
    if sum(math.isclose(value, maximum, abs_tol=1e-12) for value in values) != 1:
        return (), "TARGET_FUNCTION_ARGMAX_NOT_UNIQUE"
    margin = maximum - sorted(values, reverse=True)[1]
    required = float(program["abstention_rule"]["minimum_score_margin"])
    if margin + 1e-12 < required:
        return (), "TARGET_FUNCTION_MARGIN_BELOW_SOURCE_GUARD"
    return tuple(sorted(range(len(values)), key=lambda index: (-values[index], index))), None


def _permuted_candidate_effects(
    row: TypedInterventionSet,
) -> tuple[dict[str, float], ...]:
    values = [dict(candidate.effect_values) for candidate in row.candidates]
    count = len(values)
    if count < 2:
        return tuple(values)
    offset = 1 + int(stable_hash({
        "snapshot_sha256": row.snapshot_sha256,
        "control": "FULL_TYPED_EFFECT_VECTOR_BINDING_SHUFFLE_V1",
    })[:8], 16) % (count - 1)
    return tuple(values[(index + offset) % count] for index in range(count))


def evaluate_source_function(
    examples: Iterable[TypedInterventionSet], *, weights: Sequence[float],
    source_split: str, shuffled_effects: bool = False,
) -> dict[str, Any]:
    selected = [row for row in examples if row.source_split == source_split]
    correct = unique = varying = top2 = 0
    margins = []
    verified_ranks = []
    for row in selected:
        candidate_effects = (
            _permuted_candidate_effects(row) if shuffled_effects else
            tuple(dict(candidate.effect_values) for candidate in row.candidates)
        )
        values = [
            sum(float(weight) * value for weight, value in zip(
                weights, _checked_effect_vector(candidate),
            ))
            for candidate in candidate_effects
        ]
        ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
        maximum = values[ordered[0]]
        winners = [index for index, value in enumerate(values) if math.isclose(
            value, maximum, abs_tol=1e-12,
        )]
        correct += int(ordered[0] == row.verified_candidate_rank)
        top2 += int(row.verified_candidate_rank in ordered[:2])
        unique += int(len(winners) == 1)
        varying += int(max(values) > min(values))
        margins.append(maximum - values[ordered[1]])
        verified_ranks.append(ordered.index(row.verified_candidate_rank) + 1)
    total = len(selected)
    return {
        "examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "top2_correct": top2,
        "top2_accuracy": top2 / total if total else 0.0,
        "top2_recovery_gain": (top2 - correct) / total if total else 0.0,
        "unique_argmax_sets": unique,
        "unique_argmax_fraction": unique / total if total else 0.0,
        "varying_effect_sets": varying,
        "varying_effect_fraction": varying / total if total else 0.0,
        "mean_argmax_margin": sum(margins) / total if total else 0.0,
        "mean_verified_rank": sum(verified_ranks) / total if total else 0.0,
        "shuffled_effects": bool(shuffled_effects),
    }


def _dominant_effect(weights: Sequence[float]) -> str:
    return max(
        TYPED_EFFECTS,
        key=lambda effect_type: (
            float(weights[TYPED_EFFECTS.index(effect_type)]),
            -EFFECT_ENDPOINTS[effect_type],
            -TYPED_EFFECTS.index(effect_type),
        ),
    )


def _required_horizon(weights: Sequence[float]) -> int:
    return max(
        EFFECT_ENDPOINTS[effect_type]
        for effect_type, weight in zip(TYPED_EFFECTS, weights)
        if float(weight) > 0
    )


def induce_source_function_program(
    examples: Sequence[TypedInterventionSet], *, source_receipts_sha256: str,
    minimum_discovery_examples: int = 8,
    minimum_qualification_examples: int = 8,
    minimum_discovery_accuracy: float = 0.50,
    minimum_qualification_accuracy: float = 0.50,
    minimum_qualification_varying_fraction: float = 0.50,
    minimum_authentic_minus_shuffled: float = 0.25,
) -> dict[str, Any]:
    """Induce a sparse source-domain function and explicit transition graph."""

    discovery_rows = [row for row in examples if row.source_split == "discovery"]
    candidates = []
    for weights in FUNCTION_WEIGHT_GRID:
        metrics = evaluate_source_function(
            discovery_rows, weights=weights, source_split="discovery",
        )
        nonzero = sum(value > 0 for value in weights)
        horizon = _required_horizon(weights)
        # All structure selection is discovery-only.  MDL tie breaks prefer a
        # sparse and earlier-observable function, then a stable content hash.
        key = (
            metrics["correct"], metrics["unique_argmax_sets"],
            -nonzero, -horizon,
            stable_hash({"weights": list(weights)}),
        )
        candidates.append((key, weights, metrics))
    _, weights, discovery = max(candidates, key=lambda row: row[0])
    qualification = evaluate_source_function(
        examples, weights=weights, source_split="qualification",
    )
    shuffled = evaluate_source_function(
        examples, weights=weights, source_split="qualification",
        shuffled_effects=True,
    )
    gates = {
        "minimum_discovery_examples": discovery["examples"] >= minimum_discovery_examples,
        "minimum_qualification_examples": qualification["examples"] >= minimum_qualification_examples,
        "discovery_predictive_accuracy": discovery["accuracy"] >= minimum_discovery_accuracy,
        "qualification_predictive_accuracy": qualification["accuracy"] >= minimum_qualification_accuracy,
        "qualification_effect_is_observably_varying": (
            qualification["varying_effect_fraction"]
            >= minimum_qualification_varying_fraction
        ),
        "qualification_beats_shuffled_effect_binding": (
            qualification["accuracy"] - shuffled["accuracy"]
            >= minimum_authentic_minus_shuffled
        ),
    }
    qualified = all(gates.values())
    terms = [
        {"effect_type": effect_type, "weight": float(weight)}
        for effect_type, weight in zip(TYPED_EFFECTS, weights)
        if weight > 0
    ]
    required_horizon = _required_horizon(weights)
    retry_after_low = discovery["top2_recovery_gain"] > 0
    graph_core = {
        "entry_state": "RANKED_CANDIDATE_ABSENT",
        "required_observation_horizon": required_horizon,
        "transitions": [
            {
                "from": "RANKED_CANDIDATE_ABSENT",
                "guard": "UNIQUE_FUNCTION_ARGMAX",
                "to": "FUNCTION_CANDIDATE_ACTIVE",
            },
            {
                "from": "FUNCTION_CANDIDATE_ACTIVE",
                "guard": "OBSERVED_EFFECT_HIGH",
                "to": "TERMINAL",
            },
            {
                "from": "FUNCTION_CANDIDATE_ACTIVE",
                "guard": "OBSERVED_EFFECT_LOW",
                "to": (
                    "RANKED_CANDIDATE_ABSENT" if retry_after_low else "ABSTAIN"
                ),
            },
            {
                "from": "FUNCTION_CANDIDATE_ACTIVE",
                "guard": "OBSERVED_EFFECT_UNKNOWN",
                "to": "ABSTAIN",
            },
        ],
    }
    source_function = {
        "kind": "SPARSE_CONVEX_TEMPORAL_EFFECT_FUNCTION",
        "terms": terms,
        "dominant_effect_type": _dominant_effect(weights),
        "required_observation_horizon": required_horizon,
        "retry_after_low": retry_after_low,
        "maximum_trials": 2 if retry_after_low else 1,
        "discovery_top2_recovery_gain": discovery["top2_recovery_gain"],
        "selection_authority": (
            "SOURCE_DISCOVERY_INTERVENTION_TUPLES_ONLY;FINITE_SHARED_"
            "FUNCTION_CLASS;MDL_TIE_BREAK"
        ),
    }
    operator_core = {
        "preconditions": {
            "source_qualification_passed": qualified,
            "target_candidate_count_minimum": 2,
            "target_effect_types_required": list(TYPED_EFFECTS),
            "target_unique_function_argmax_required": True,
        },
        "score": {
            "kind": source_function["kind"],
            "terms": terms,
        },
        "state_delta": [
            ["active_candidate", "UNIQUE_FUNCTION_ARGMAX"],
            ["attempt_ledger", "MARK_ACTIVE_AS_TRIED"],
        ],
    }
    operator = {"operator_id": f"OP_{stable_hash(operator_core)[:16]}", **operator_core}
    thresholds = {
        "minimum_discovery_examples": minimum_discovery_examples,
        "minimum_qualification_examples": minimum_qualification_examples,
        "minimum_discovery_accuracy": minimum_discovery_accuracy,
        "minimum_qualification_accuracy": minimum_qualification_accuracy,
        "minimum_qualification_varying_fraction": minimum_qualification_varying_fraction,
        "minimum_authentic_minus_shuffled": minimum_authentic_minus_shuffled,
    }
    body = {
        "schema_version": "phase3-source-induced-domain-function-v4",
        "status": QUALIFIED if qualified else ABSTAINING,
        "source_receipts_sha256": str(source_receipts_sha256),
        "induction_authority": (
            "SOURCE_DISCOVERY_TRANSITION_TUPLES_ONLY;SOURCE_QUALIFICATION_"
            "CALIBRATION;NO_SOURCE_IDENTITY_FEATURE;NO_TARGET_DATA"
        ),
        "shared_ir": {
            "effect_types": list(TYPED_EFFECTS),
            "effect_endpoints": dict(EFFECT_ENDPOINTS),
            "function_class": "NONNEGATIVE_QUARTER_GRID_SUM_TO_ONE",
            "weight_denominator": WEIGHT_DENOMINATOR,
        },
        "source_function": source_function,
        "operators": [operator] if qualified else [],
        "transition_graph": graph_core,
        "discovery_metrics": discovery,
        "qualification_metrics": qualification,
        "qualification_shuffled_effect_metrics": shuffled,
        "qualification_gates": gates,
        "abstention_rule": {
            "source_not_qualified": "ABSTAIN",
            "missing_or_nonfinite_target_effect": "ABSTAIN",
            "candidate_set_smaller_than_two": "ABSTAIN",
            "nonunique_target_function_argmax": "ABSTAIN",
            "minimum_score_margin": 0.0,
            "unknown_observed_effect": "ABSTAIN",
        },
        "thresholds": thresholds,
        "target_data_read": False,
        "native_action_tokens_exported": False,
        "source_identity_used_as_feature": False,
        "forbidden_named_policy_tokens": [
            "EXPLORE_UNTRIED", "BACKTRACK_REPLAN", "COMMIT_VERIFY",
        ],
    }
    return body | {"program_sha256": stable_hash(body)}


def validate_source_function_program(program: Mapping[str, Any]) -> None:
    body = dict(program)
    claimed = body.pop("program_sha256", None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError("source-domain function hash mismatch")
    if program.get("schema_version") != "phase3-source-induced-domain-function-v4":
        raise ValueError("unsupported source-domain function schema")
    if program.get("status") not in {QUALIFIED, ABSTAINING}:
        raise ValueError("unsupported source-domain function status")
    if program.get("target_data_read") is not False:
        raise ValueError("source-domain function does not attest target isolation")
    if program.get("source_identity_used_as_feature") is not False:
        raise ValueError("source-domain function used source identity")
    weights = function_weights(program)
    function = program["source_function"]
    if int(function["required_observation_horizon"]) != _required_horizon(weights):
        raise ValueError("source-domain function horizon mismatch")
    if int(function["maximum_trials"]) not in {1, 2}:
        raise ValueError("source-domain function maximum_trials is invalid")
    expected_retry = bool(float(function["discovery_top2_recovery_gain"]) > 0)
    if bool(function["retry_after_low"]) != expected_retry:
        raise ValueError("source-domain retry edge is not source-derived")
    serialized = json.dumps(program, sort_keys=True)
    for token in program.get("forbidden_named_policy_tokens") or ():
        if serialized.count(str(token)) != 1:
            raise ValueError("named policy token leaked into source function")


def recalibrate_source_function_program(
    program: Mapping[str, Any], *, calibration_metrics: Mapping[str, Any],
    calibration_receipt_sha256: str,
    minimum_calibration_accuracy: float = 0.50,
    minimum_calibration_varying_fraction: float = 0.50,
    minimum_calibration_authentic_minus_shuffled: float = 0.25,
) -> dict[str, Any]:
    """Use an independent source batch only to retain or remove a function."""

    validate_source_function_program(program)
    authentic = calibration_metrics.get("authentic")
    shuffled = calibration_metrics.get("shuffled_effect_binding")
    if not isinstance(authentic, Mapping) or not isinstance(shuffled, Mapping):
        raise ValueError("source-function calibration omitted controls")
    gates = {
        "single_batch_source_qualification_passed": program.get("status") == QUALIFIED,
        "independent_calibration_accuracy": (
            float(authentic.get("accuracy", 0.0)) >= minimum_calibration_accuracy
        ),
        "independent_calibration_effect_varies": (
            float(authentic.get("varying_effect_fraction", 0.0))
            >= minimum_calibration_varying_fraction
        ),
        "independent_calibration_beats_shuffled": (
            float(authentic.get("accuracy", 0.0))
            - float(shuffled.get("accuracy", 0.0))
            >= minimum_calibration_authentic_minus_shuffled
        ),
    }
    admitted = all(gates.values())
    original = dict(program)
    original_sha = str(original.pop("program_sha256"))
    body = {
        **original,
        "status": QUALIFIED if admitted else ABSTAINING,
        "operators": list(program.get("operators") or ()) if admitted else [],
        "pre_recalibration_program_sha256": original_sha,
        "cross_batch_calibration": {
            "authority": (
                "INDEPENDENT_SOURCE_BATCH_ONLY;CAN_REMOVE_FUNCTION_BUT_CANNOT_"
                "CHANGE_TERMS_HORIZON_RETRY_OR_GRAPH;NO_TARGET_DATA"
            ),
            "calibration_receipt_sha256": str(calibration_receipt_sha256),
            "metrics": {
                "authentic": dict(authentic),
                "shuffled_effect_binding": dict(shuffled),
            },
            "thresholds": {
                "minimum_calibration_accuracy": minimum_calibration_accuracy,
                "minimum_calibration_varying_fraction": minimum_calibration_varying_fraction,
                "minimum_calibration_authentic_minus_shuffled": (
                    minimum_calibration_authentic_minus_shuffled
                ),
            },
            "gates": gates,
        },
    }
    return body | {"program_sha256": stable_hash(body)}


def maximum_source_function_contrast_derangement(
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Permute functions within admission strata for a structure control."""

    programs = {}
    for name, artifact in artifacts.items():
        body = dict(artifact)
        claimed = body.pop("artifact_sha256", None)
        if not claimed or claimed != stable_hash(body):
            raise ValueError(f"source-function artifact hash mismatch: {name}")
        program = artifact.get("source_function_program")
        if not isinstance(program, Mapping):
            raise ValueError(f"source-function artifact omitted program: {name}")
        validate_source_function_program(program)
        programs[str(name)] = program
    mapping = {}
    for status in sorted({str(row["status"]) for row in programs.values()}):
        names = tuple(sorted(
            name for name, row in programs.items() if str(row["status"]) == status
        ))
        if len(names) < 2:
            raise ValueError("each source-function status stratum needs two lineages")
        candidates = []
        for permuted in permutations(names):
            if any(left == right for left, right in zip(names, permuted)):
                continue
            distance = 0.0
            horizon_contrasts = retry_contrasts = 0
            for left, right in zip(names, permuted):
                left_program = programs[left]
                right_program = programs[right]
                distance += sum(abs(a - b) for a, b in zip(
                    function_weights(left_program), function_weights(right_program),
                ))
                horizon_contrasts += int(
                    left_program["source_function"]["required_observation_horizon"]
                    != right_program["source_function"]["required_observation_horizon"]
                )
                retry_contrasts += int(
                    bool(left_program["source_function"]["retry_after_low"])
                    != bool(right_program["source_function"]["retry_after_low"])
                )
            candidate = dict(zip(names, permuted))
            candidates.append((
                distance, horizon_contrasts, retry_contrasts,
                stable_hash(candidate), candidate,
            ))
        if not candidates:
            raise ValueError(f"no source-function derangement for status: {status}")
        mapping.update(max(candidates, key=lambda row: row[:4])[-1])
    return dict(sorted(mapping.items()))


__all__ = [
    "ABSTAINING", "EFFECT_ENDPOINTS", "FUNCTION_WEIGHT_GRID", "QUALIFIED",
    "evaluate_source_function", "function_trial_order", "function_weights",
    "induce_source_function_program", "maximum_source_function_contrast_derangement",
    "recalibrate_source_function_program", "score_effects",
    "validate_source_function_program",
]
