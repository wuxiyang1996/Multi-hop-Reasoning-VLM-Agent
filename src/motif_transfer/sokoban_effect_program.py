"""Effect-triggered symbolic skill distilled from Sokoban interventions.

V1 predicted the next option directly and consequently transferred Sokoban's
POSITION-heavy option occupancy to ALFWorld.  This module deliberately removes
that domain-specific duration prior.  The transferable object is a Boolean
guard over intervention effects:

    positive relational effect available -> COMMIT -> VERIFY
    otherwise                            -> POSITION -> REPLAN

Coordinates, action tokens, option frequencies, and concrete target actions
are outside the transfer contract.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .contracts import stable_hash
from .sokoban_commit_skill import (
    option_features,
    parse_state,
    shortest_solution,
    validate_plan,
)


ARTIFACT_VERSION = "SOKOBAN_EFFECT_PROGRAM_V2"
QUALIFICATION_VERSION = "SOKOBAN_EFFECT_PROGRAM_QUALIFICATION_V2"
CONDITIONS = (
    "authentic_effect_guard",
    "commit_availability_only",
    "inverted_effect_guard",
    "position_occupancy_prior",
)


def effect_predicates(features: tuple[float, ...]) -> dict[str, bool]:
    """Convert source-native intervention statistics to symbolic predicates."""

    return {
        "commit_available": features[4] > 0.0,
        "direct_progress_available": features[5] > 0.0,
        "assignment_improvement_available": features[8] > 0.0,
        "regression_observed": features[6] > 0.0,
        "deadlock_observed": features[7] > 0.0,
    }


def select_option(condition: str, predicates: Mapping[str, bool]) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported source effect condition: {condition}")
    positive_effect = bool(
        predicates["direct_progress_available"]
        or predicates["assignment_improvement_available"]
    )
    if condition == "authentic_effect_guard":
        commit = positive_effect
    elif condition == "commit_availability_only":
        commit = bool(predicates["commit_available"])
    elif condition == "inverted_effect_guard":
        commit = bool(predicates["commit_available"] and not positive_effect)
    else:
        commit = False
    return "COMMIT" if commit else "POSITION"


def _examples(plan: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    examples = []
    for row in validate_plan(plan):
        if str(row["split"]) != split:
            continue
        state = parse_state(str(row["state"]))
        solution = shortest_solution(state)
        if not solution:
            continue
        optimal = "COMMIT" if solution[0].startswith("push ") else "POSITION"
        features = option_features(state, "COMMIT")
        examples.append({
            "snapshot_id": str(row["snapshot_id"]),
            "episode_id": str(row["episode_id"]),
            "optimal_first_option": optimal,
            "commit_features": list(features),
            "predicates": effect_predicates(features),
        })
    return examples


def _condition_metrics(examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for condition in CONDITIONS:
        rows = []
        for row in examples:
            selected = select_option(condition, row["predicates"])
            rows.append({
                "snapshot_id": row["snapshot_id"],
                "optimal_first_option": row["optimal_first_option"],
                "selected_option": selected,
                "correct": selected == row["optimal_first_option"],
            })
        result[condition] = {
            "n": len(rows),
            "accuracy": sum(item["correct"] for item in rows) / len(rows),
            "selected_option_counts": dict(sorted(Counter(
                item["selected_option"] for item in rows
            ).items())),
            "predictions": rows,
        }
    return result


def build_effect_program(
    plan: Mapping[str, Any], *, development_receipt: Mapping[str, str],
) -> dict[str, Any]:
    examples = _examples(plan, "discovery")
    if len(examples) < 12:
        raise ValueError("too few source discovery examples")
    metrics = _condition_metrics(examples)
    authentic = metrics["authentic_effect_guard"]["accuracy"]
    if not all(
        authentic > metrics[name]["accuracy"]
        for name in CONDITIONS if name != "authentic_effect_guard"
    ):
        raise ValueError("effect guard does not dominate source discovery controls")
    body = {
        "artifact_version": ARTIFACT_VERSION,
        "lifecycle": "DISCOVERY_FROZEN_AWAITING_NEW_FRESH_SOURCE_CONFIRMATION",
        "claim_boundary": (
            "TRANSFER_BOOLEAN_INTERVENTION_EFFECT_STRUCTURE_ONLY; EXCLUDE_"
            "SOURCE_OPTION_OCCUPANCY_COORDINATES_ACTION_NAMES_AND_DURATIONS"
        ),
        "source_plan_sha256": str(plan["plan_sha256"]),
        "development_receipt": dict(development_receipt),
        "program": {
            "predicates": [
                "COMMIT_AVAILABLE",
                "DIRECT_PROGRESS_AVAILABLE",
                "ASSIGNMENT_IMPROVEMENT_AVAILABLE",
                "EXPECTED_EFFECT_OBSERVED",
            ],
            "rules": [
                {
                    "when": (
                        "DIRECT_PROGRESS_AVAILABLE_OR_"
                        "ASSIGNMENT_IMPROVEMENT_AVAILABLE"
                    ),
                    "select": "COMMIT",
                    "then": "VERIFY_EXPECTED_EFFECT",
                },
                {
                    "when": "NO_POSITIVE_COMMIT_EFFECT_AVAILABLE",
                    "select": "POSITION",
                    "then": "RECOMPUTE_EFFECT_PREDICATES",
                },
                {
                    "when": "EXPECTED_EFFECT_REFUTED",
                    "select": "REPLAN_OR_ABSTAIN",
                },
            ],
            "target_permission": (
                "TARGET_NATIVE_GROUNDER_MAY_GROUND_PREDICATES_AND_REALIZE_ONE_"
                "NATIVE_ACTION; SOURCE_PROGRAM_SELECTS_POSITION_OR_COMMIT_ONLY"
            ),
        },
        "discovery": {
            "eligible_examples": len(examples),
            "optimal_option_counts": dict(sorted(Counter(
                row["optimal_first_option"] for row in examples
            ).items())),
            "condition_metrics": {
                name: {key: value for key, value in row.items() if key != "predictions"}
                for name, row in metrics.items()
            },
            "source_snapshot_ids": [row["snapshot_id"] for row in examples],
        },
    }
    return body | {"artifact_sha256": stable_hash(body)}


def validate_effect_program(artifact: Mapping[str, Any]) -> None:
    body = dict(artifact)
    claimed = str(body.pop("artifact_sha256", ""))
    if stable_hash(body) != claimed:
        raise ValueError("source effect program hash mismatch")
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported source effect program version")
    rules = artifact.get("program", {}).get("rules", [])
    if [row.get("select") for row in rules[:2]] != ["COMMIT", "POSITION"]:
        raise ValueError("source effect control flow changed")


def qualify_effect_program(
    plan: Mapping[str, Any], artifact: Mapping[str, Any], *,
    minimum_examples_per_option: int = 6,
    minimum_accuracy: float = 0.90,
) -> dict[str, Any]:
    validate_effect_program(artifact)
    if plan.get("status") != "FROZEN_FRESH_CONFIRMATION_BEFORE_ARTIFACT_PREDICTIONS":
        raise ValueError("qualification requires a frozen fresh source plan")
    examples = _examples(plan, "held_out")
    counts = Counter(row["optimal_first_option"] for row in examples)
    metrics = _condition_metrics(examples)
    authentic = metrics["authentic_effect_guard"]
    coverage = bool(
        len(examples) >= 12
        and all(counts[option] >= minimum_examples_per_option
                for option in ("POSITION", "COMMIT"))
    )
    superiority = bool(coverage and all(
        authentic["accuracy"] > metrics[name]["accuracy"]
        for name in CONDITIONS if name != "authentic_effect_guard"
    ))
    passed = bool(
        coverage and authentic["accuracy"] >= minimum_accuracy and superiority
    )
    body = {
        "qualification_version": QUALIFICATION_VERSION,
        "status": (
            "SOURCE_EFFECT_PROGRAM_CONFIRMED"
            if passed else "SOURCE_EFFECT_PROGRAM_REJECTED"
        ),
        "claim_boundary": "NEW_FRESH_SOURCE_CONFIRMATION_ONLY_NO_TARGET_EVIDENCE",
        "source_plan_sha256": str(plan["plan_sha256"]),
        "artifact_sha256": str(artifact["artifact_sha256"]),
        "eligible_examples": len(examples),
        "optimal_option_counts": dict(sorted(counts.items())),
        "condition_metrics": metrics,
        "thresholds": {
            "minimum_examples_per_option": minimum_examples_per_option,
            "minimum_authentic_accuracy": minimum_accuracy,
            "strict_superiority_to_each_source_control": True,
        },
        "gates": {
            "coverage": coverage,
            "accuracy": bool(coverage and authentic["accuracy"] >= minimum_accuracy),
            "control_superiority": superiority,
        },
        "source_gate_passed": passed,
        "next_step": "FREEZE_NEW_TARGET_SPLIT" if passed else "STOP_BEFORE_TARGET",
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "ARTIFACT_VERSION",
    "CONDITIONS",
    "QUALIFICATION_VERSION",
    "build_effect_program",
    "effect_predicates",
    "qualify_effect_program",
    "select_option",
    "validate_effect_program",
]
