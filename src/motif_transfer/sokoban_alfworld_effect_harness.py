"""V2 ALFWorld Harness for an effect-triggered Sokoban symbolic program.

The authentic path never reads ALFWorld's hand-written ``required_option`` to
select an option or realize an action.  It receives target-native neural
applicability/completion scores, grounds the source predicate
POSITIVE_COMMIT_EFFECT_AVAILABLE, and executes the frozen source Boolean rule.
The required option is retained only for diagnostics and the explicitly named
target-native symbolic reference condition.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .sokoban_effect_program import select_option, validate_effect_program


CONDITIONS = (
    "target_only",
    "authentic_source_effect_harness",
    "commit_availability_control",
    "inverted_effect_control",
    "position_occupancy_control",
    "effect_group_permuted_control",
    "target_native_stage_reference",
)


def _target_group(row: Mapping[str, Any]) -> str:
    return "POSITION" if str(row["option"]) == "SEARCH" else "COMMIT"


def _required_group(grounded: Mapping[str, Mapping[str, Any]]) -> str:
    required = {str(row["required_option"]) for row in grounded.values()}
    if len(required) != 1:
        raise ValueError("target grounder returned inconsistent required options")
    return "POSITION" if next(iter(required)) == "SEARCH" else "COMMIT"


def _effect_score(row: Mapping[str, Any]) -> float:
    return float(row["applicability"]) * float(row["completion"])


def ground_effect_predicates(
    grounded: Mapping[str, Mapping[str, Any]], *, effect_threshold: float,
) -> dict[str, Any]:
    """Ground source predicates without consulting target workflow stage."""

    if not 0.0 < effect_threshold < 1.0:
        raise ValueError("effect threshold must be in (0, 1)")
    grouped = {
        option: [row for row in grounded.values() if _target_group(row) == option]
        for option in ("POSITION", "COMMIT")
    }
    best = {
        option: max((_effect_score(row) for row in rows), default=0.0)
        for option, rows in grouped.items()
    }
    return {
        "commit_available": bool(grouped["COMMIT"]),
        "direct_progress_available": best["COMMIT"] >= effect_threshold,
        "assignment_improvement_available": False,
        "regression_observed": bool(
            grouped["COMMIT"]
            and max(float(row["applicability"]) for row in grouped["COMMIT"]) < 0.5
        ),
        "deadlock_observed": False,
        "best_commit_effect_score": best["COMMIT"],
        "best_position_effect_score": best["POSITION"],
        "effect_threshold": effect_threshold,
    }


def _policy_score(
    action: str, row: Mapping[str, Any], history: Sequence[str],
) -> float:
    return float(row.get("policy", row["applicability"])) / (
        1.0 + history.count(action)
    )


def _commit_realization_score(
    action: str, row: Mapping[str, Any], history: Sequence[str],
) -> float:
    applicability = float(row["applicability"])
    completion = float(row["completion"])
    binding = float(row["binding"])
    return applicability * (0.20 + 0.80 * completion) * (
        0.25 + 0.75 * binding
    ) / (1.0 + history.count(action))


def choose_action(
    *, condition: str, grounded: Mapping[str, Mapping[str, Any]],
    history: Sequence[str], source_artifact: Mapping[str, Any],
    effect_threshold: float,
) -> dict[str, Any]:
    if condition not in CONDITIONS or not grounded:
        raise ValueError("unsupported condition or empty target grounding")
    validate_effect_program(source_artifact)
    actions = tuple(grounded)
    fallback = max(
        actions,
        key=lambda action: (_policy_score(action, grounded[action], history), action),
    )
    fallback_group = _target_group(grounded[fallback])
    required_group = _required_group(grounded)
    predicates = ground_effect_predicates(
        grounded, effect_threshold=effect_threshold,
    )
    if condition == "target_only":
        return {
            "action": fallback,
            "fallback_action": fallback,
            "fallback_option": fallback_group,
            "required_option_diagnostic_only": required_group,
            "source_selected_option": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_option": False,
            "diagnostic": "NULL_SKILL_TARGET_ONLY",
            "effect_predicates": predicates,
        }

    if condition == "authentic_source_effect_harness":
        source_condition = "authentic_effect_guard"
        selected_option = select_option(source_condition, predicates)
    elif condition == "commit_availability_control":
        source_condition = "commit_availability_only"
        selected_option = select_option(source_condition, predicates)
    elif condition == "inverted_effect_control":
        source_condition = "inverted_effect_guard"
        selected_option = select_option(source_condition, predicates)
    elif condition == "position_occupancy_control":
        source_condition = "position_occupancy_prior"
        selected_option = select_option(source_condition, predicates)
    elif condition == "effect_group_permuted_control":
        source_condition = "EFFECT_GROUP_PERMUTED"
        selected_option = (
            "COMMIT"
            if predicates["commit_available"]
            and predicates["best_position_effect_score"] >= effect_threshold
            else "POSITION"
        )
    else:
        source_condition = "TARGET_NATIVE_STAGE_REFERENCE"
        selected_option = required_group

    candidates = [
        action for action, row in grounded.items()
        if _target_group(row) == selected_option
    ]
    if not candidates:
        return {
            "action": fallback,
            "fallback_action": fallback,
            "fallback_option": fallback_group,
            "required_option_diagnostic_only": required_group,
            "source_selected_option": selected_option,
            "source_condition": source_condition,
            "source_admitted": False,
            "changed_action": False,
            "changed_option": False,
            "diagnostic": "TARGET_REALIZATION_UNAVAILABLE_REPLAN",
            "effect_predicates": predicates,
        }
    if selected_option == "POSITION":
        selected = max(
            candidates,
            key=lambda action: (
                _policy_score(action, grounded[action], history), action,
            ),
        )
    else:
        selected = max(
            candidates,
            key=lambda action: (
                _commit_realization_score(action, grounded[action], history), action,
            ),
        )
    return {
        "action": selected,
        "fallback_action": fallback,
        "fallback_option": fallback_group,
        "required_option_diagnostic_only": required_group,
        "source_selected_option": selected_option,
        "source_condition": source_condition,
        "source_admitted": True,
        "changed_action": selected != fallback,
        "changed_option": _target_group(grounded[selected]) != fallback_group,
        "diagnostic": (
            "SOURCE_EFFECT_OPTION_REALIZED_VERIFY_NEXT_STATE"
            if condition != "target_native_stage_reference"
            else "TARGET_NATIVE_STAGE_REFERENCE_REALIZED"
        ),
        "effect_predicates": predicates,
    }


__all__ = [
    "CONDITIONS",
    "choose_action",
    "ground_effect_predicates",
]
