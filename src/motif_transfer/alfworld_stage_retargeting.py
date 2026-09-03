"""Permission-bounded Sokoban effect retargeting for ALFWorld.

The frozen source program chooses only ``POSITION`` or ``COMMIT``.  In this
adapter ``COMMIT`` means "advance the current target-native stage", not "take
an arbitrary non-navigation action".  The ALFWorld neural policy retains sole
authority over the concrete action inside the externally selected group.

This is intentionally separate from the V2 effect Harness so old receipts stay
reproducible.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .sokoban_effect_program import select_option, validate_effect_program


CONDITIONS = (
    "raw_target_only",
    "null_skill_same_harness",
    "authentic_source_skill",
    "commit_availability_control",
    "inverted_effect_control",
    "position_prior_control",
    "target_oracle_skill",
)


def canonical_group(row: Mapping[str, Any]) -> str:
    """Map target-native stages to the source program's control level."""

    return "POSITION" if str(row["option"]) == "SEARCH" else "COMMIT"


def _policy_score(
    action: str,
    row: Mapping[str, Any],
    history: Sequence[str],
) -> float:
    # Repetition is target-native loop handling shared by every Harness arm.
    return float(row.get("policy", row["applicability"])) / (
        1.0 + history.count(action)
    )


def ground_source_predicates(
    grounded: Mapping[str, Mapping[str, Any]],
    *,
    effect_threshold: float,
) -> dict[str, Any]:
    """Ground the frozen Sokoban Boolean predicates from neural target scores."""

    if not 0.0 < effect_threshold < 1.0:
        raise ValueError("effect_threshold must be in (0, 1)")
    grouped = {
        group: [row for row in grounded.values() if canonical_group(row) == group]
        for group in ("POSITION", "COMMIT")
    }
    advance_scores = [
        float(row["applicability"]) * float(row["completion"])
        for row in grouped["COMMIT"]
    ]
    best_advance = max(advance_scores, default=0.0)
    return {
        "commit_available": bool(grouped["COMMIT"]),
        "direct_progress_available": best_advance >= effect_threshold,
        "assignment_improvement_available": False,
        "regression_observed": bool(
            grouped["COMMIT"]
            and max(float(row["applicability"]) for row in grouped["COMMIT"]) < 0.5
        ),
        "deadlock_observed": False,
        "best_commit_effect_score": best_advance,
        "effect_threshold": float(effect_threshold),
    }


def choose_action(
    *,
    condition: str,
    grounded: Mapping[str, Mapping[str, Any]],
    history: Sequence[str],
    source_artifact: Mapping[str, Any],
    effect_threshold: float,
) -> dict[str, Any]:
    """Select a source option, then realize it with the target neural policy."""

    if condition not in CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    if not grounded:
        raise ValueError("target grounding cannot be empty")
    validate_effect_program(source_artifact)
    actions = tuple(grounded)
    fallback = max(
        actions,
        key=lambda action: (_policy_score(action, grounded[action], history), action),
    )
    fallback_group = canonical_group(grounded[fallback])
    predicates = ground_source_predicates(
        grounded,
        effect_threshold=effect_threshold,
    )

    if condition in {"raw_target_only", "null_skill_same_harness"}:
        return {
            "action": fallback,
            "fallback_action": fallback,
            "fallback_option": fallback_group,
            "source_selected_option": None,
            "source_condition": "NULL",
            "source_admitted": False,
            "changed_action": False,
            "changed_option": False,
            "diagnostic": "NULL_SKILL_TARGET_FALLBACK",
            "effect_predicates": predicates,
        }

    if condition == "authentic_source_skill":
        source_condition = "authentic_effect_guard"
        selected_group = select_option(source_condition, predicates)
    elif condition == "commit_availability_control":
        source_condition = "commit_availability_only"
        selected_group = select_option(source_condition, predicates)
    elif condition == "inverted_effect_control":
        source_condition = "inverted_effect_guard"
        selected_group = select_option(source_condition, predicates)
    elif condition == "position_prior_control":
        source_condition = "position_occupancy_prior"
        selected_group = select_option(source_condition, predicates)
    else:
        # Explicit target-written upper bound.  This diagnostic is never a
        # source condition and may read the target workflow label.
        source_condition = "TARGET_ORACLE"
        required = {str(row["required_option"]) for row in grounded.values()}
        if len(required) != 1:
            raise ValueError("target oracle received inconsistent stage labels")
        selected_group = "POSITION" if next(iter(required)) == "SEARCH" else "COMMIT"

    candidates = [
        action for action in actions
        if canonical_group(grounded[action]) == selected_group
    ]
    if not candidates:
        return {
            "action": fallback,
            "fallback_action": fallback,
            "fallback_option": fallback_group,
            "source_selected_option": selected_group,
            "source_condition": source_condition,
            "source_admitted": False,
            "changed_action": False,
            "changed_option": False,
            "diagnostic": "SELECTED_OPTION_UNAVAILABLE_ABSTAIN",
            "effect_predicates": predicates,
        }

    # Critical authority boundary: source scores never rank target actions.
    selected = max(
        candidates,
        key=lambda action: (_policy_score(action, grounded[action], history), action),
    )
    return {
        "action": selected,
        "fallback_action": fallback,
        "fallback_option": fallback_group,
        "source_selected_option": selected_group,
        "source_condition": source_condition,
        "source_admitted": True,
        "changed_action": selected != fallback,
        "changed_option": canonical_group(grounded[selected]) != fallback_group,
        "diagnostic": "SOURCE_OPTION_TARGET_NEURAL_REALIZATION",
        "effect_predicates": predicates,
    }


__all__ = [
    "CONDITIONS",
    "canonical_group",
    "choose_action",
    "ground_source_predicates",
]
