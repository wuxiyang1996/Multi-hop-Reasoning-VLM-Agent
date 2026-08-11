"""Permission-bounded ALFWorld Harness for a frozen Sokoban option skill.

The source model may choose only POSITION or COMMIT.  The target Harness owns
target-native predicate grounding and realizes one concrete native action
inside the selected option.  A refuted COMMIT precondition always falls back
to the target-only action; source scores never directly rank ALFWorld actions.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .alfworld_hierarchical_grounder import workflow_status
from .contracts import stable_hash
from .sokoban_commit_skill import (
    OPTIONS,
    predict_option_value,
    validate_artifact as validate_source_artifact,
)


CONDITIONS = (
    "target_only",
    "authentic_source_plus_harness",
    "option_swap_source_plus_harness",
    "source_marginal_plus_harness",
    "phase_permuted_source_plus_harness",
    "target_oracle_option_plus_harness",
)


def _target_group(row: Mapping[str, Any]) -> str:
    return "POSITION" if str(row["option"]) == "SEARCH" else "COMMIT"


def _required_group(grounded: Mapping[str, Mapping[str, Any]]) -> str:
    required = {str(row["required_option"]) for row in grounded.values()}
    if len(required) != 1:
        raise ValueError("target grounder returned inconsistent required options")
    return "POSITION" if next(iter(required)) == "SEARCH" else "COMMIT"


def canonical_target_features(
    *, option: str, grounded: Mapping[str, Mapping[str, Any]],
    goal: str, history: Sequence[str],
) -> tuple[float, ...]:
    """Ground target-native neural scores into the frozen source ISA."""

    if option not in OPTIONS or not grounded:
        raise ValueError("invalid target canonicalization request")
    rows = [row for row in grounded.values() if _target_group(row) == option]
    total = max(1, len(grounded))
    count = max(1, len(rows))
    applicability = [float(row["applicability"]) for row in rows]
    completion = [float(row["completion"]) for row in rows]
    binding = [float(row["binding"]) for row in rows]
    status = workflow_status(goal, history)
    applicable_fraction = sum(value >= 0.5 for value in applicability) / count
    completion_mean = sum(completion) / count
    completion_max = max(completion, default=0.0)
    regression = (
        sum(1.0 - value for value in applicability) / count
        if option == "COMMIT" else 0.0
    )
    risk = sum(
        (1.0 - applicable) * (1.0 - complete)
        for applicable, complete in zip(applicability, completion)
    ) / count
    return (
        float(option == "POSITION"),
        float(option == "COMMIT"),
        applicable_fraction,
        completion_mean,
        completion_mean if option == "COMMIT" else 0.0,
        completion_max,
        regression,
        risk,
        completion_max,
        1.0 - status.progress_fraction,
        max(binding, default=0.0),
        len(rows) / total,
    )


def _phase_permuted(features: Sequence[float]) -> tuple[float, ...]:
    values = list(map(float, features))
    values[0], values[1] = values[1], values[0]
    values[2], values[4] = values[4], values[2]
    values[5], values[7] = values[7], values[5]
    return tuple(values)


def _realization_score(
    action: str, row: Mapping[str, Any], history: Sequence[str],
) -> float:
    applicability = float(row["applicability"])
    completion = float(row["completion"])
    binding = 1.0 if str(row["option"]) == "SEARCH" else float(row["binding"])
    repeat_discount = 1.0 + history.count(action)
    return applicability * (0.20 + 0.80 * completion) * (
        0.25 + 0.75 * binding
    ) / repeat_discount


def _target_policy_score(
    action: str, row: Mapping[str, Any], history: Sequence[str],
) -> float:
    return float(row.get("policy", row["applicability"])) / (
        1.0 + history.count(action)
    )


def choose_action(
    *, condition: str, grounded: Mapping[str, Mapping[str, Any]],
    goal: str, history: Sequence[str], source_artifact: Mapping[str, Any],
    identity: str,
) -> dict[str, Any]:
    if condition not in CONDITIONS or not grounded:
        raise ValueError("unsupported condition or empty target grounding")
    validate_source_artifact(source_artifact)
    actions = tuple(grounded)
    fallback = max(
        actions,
        key=lambda action: (
            _target_policy_score(action, grounded[action], history), action,
        ),
    )
    fallback_group = _target_group(grounded[fallback])
    required_group = _required_group(grounded)
    if condition == "target_only":
        return {
            "action": fallback,
            "fallback_action": fallback,
            "fallback_option": fallback_group,
            "required_option": required_group,
            "source_selected_option": None,
            "source_admitted": False,
            "changed_action": False,
            "changed_option": False,
            "diagnostic": "NULL_SKILL_TARGET_ONLY",
            "source_option_scores": {},
        }

    if condition == "target_oracle_option_plus_harness":
        selected_option = required_group
        option_scores: dict[str, float] = {}
    else:
        features = {
            option: canonical_target_features(
                option=option, grounded=grounded, goal=goal, history=history,
            )
            for option in OPTIONS
        }
        if condition == "authentic_source_plus_harness":
            model = source_artifact["models"]["authentic"]
            option_scores = {
                option: predict_option_value(model, row)
                for option, row in features.items()
            }
        elif condition == "option_swap_source_plus_harness":
            model = source_artifact["models"]["within_state_option_swap"]
            option_scores = {
                option: predict_option_value(model, row)
                for option, row in features.items()
            }
        elif condition == "phase_permuted_source_plus_harness":
            model = source_artifact["models"]["authentic"]
            option_scores = {
                option: predict_option_value(model, _phase_permuted(row))
                for option, row in features.items()
            }
        else:
            marginal = float(source_artifact["models"]["source_marginal"]["constant"])
            option_scores = {option: marginal for option in OPTIONS}
        selected_option = max(
            OPTIONS,
            key=lambda option: (
                option_scores[option], stable_hash((identity, condition, option)),
            ),
        )

    if selected_option == "COMMIT" and required_group != "COMMIT":
        return {
            "action": fallback,
            "fallback_action": fallback,
            "fallback_option": fallback_group,
            "required_option": required_group,
            "source_selected_option": selected_option,
            "source_admitted": False,
            "changed_action": False,
            "changed_option": False,
            "diagnostic": "COMMIT_PRECONDITION_REFUTED",
            "source_option_scores": option_scores,
        }

    candidates = [
        action for action, row in grounded.items()
        if _target_group(row) == selected_option
        and (
            selected_option != "COMMIT"
            or str(row["option"]) == str(row["required_option"])
        )
    ]
    if not candidates:
        return {
            "action": fallback,
            "fallback_action": fallback,
            "fallback_option": fallback_group,
            "required_option": required_group,
            "source_selected_option": selected_option,
            "source_admitted": False,
            "changed_action": False,
            "changed_option": False,
            "diagnostic": "TARGET_REALIZATION_UNAVAILABLE",
            "source_option_scores": option_scores,
        }
    selected = max(
        candidates,
        key=lambda action: (
            _realization_score(action, grounded[action], history), action,
        ),
    )
    return {
        "action": selected,
        "fallback_action": fallback,
        "fallback_option": fallback_group,
        "required_option": required_group,
        "source_selected_option": selected_option,
        "source_admitted": True,
        "changed_action": selected != fallback,
        "changed_option": _target_group(grounded[selected]) != fallback_group,
        "diagnostic": (
            "SOURCE_OPTION_REALIZED"
            if condition != "target_oracle_option_plus_harness"
            else "TARGET_ORACLE_OPTION_REALIZED"
        ),
        "source_option_scores": option_scores,
    }


def softmax_option_probability(scores: Mapping[str, float], option: str) -> float:
    maximum = max(scores.values())
    values = {key: math.exp(value - maximum) for key, value in scores.items()}
    return values[option] / sum(values.values())


__all__ = [
    "CONDITIONS",
    "canonical_target_features",
    "choose_action",
    "softmax_option_probability",
]
