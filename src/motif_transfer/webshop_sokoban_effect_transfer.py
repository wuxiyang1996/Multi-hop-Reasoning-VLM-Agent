"""Ground the frozen Sokoban effect program in WebShop-native predictions.

The source program was independently confirmed on matched Sokoban
interventions.  It transfers only a causal control relation:

    positive irreversible effect available -> COMMIT -> VERIFY
    otherwise                              -> POSITION/PREPARE

WebShop action tokens and observations never enter the source artifact.  A
target-native outcome MLP grounds the program predicates and realizes the
selected option with one native WebShop action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .sokoban_effect_program import select_option, validate_effect_program
from .webshop_neural_symbolic_v9 import OUTCOME_NAMES


TARGET_ONLY_CONDITIONS = ("target_only", "target_native_myopic")
SOURCE_CONDITION_MAP = {
    "authentic_sokoban_effect_plus_target": "authentic_effect_guard",
    "commit_availability_control_plus_target": "commit_availability_only",
    "inverted_effect_control_plus_target": "inverted_effect_guard",
    "position_prior_control_plus_target": "position_occupancy_prior",
}
CONDITIONS = TARGET_ONLY_CONDITIONS + tuple(SOURCE_CONDITION_MAP)


@dataclass(frozen=True)
class EffectTransferDecision:
    selected_index: int
    abstract_kind: str
    source_abstained: bool
    source_test_value: float | None
    source_commit_value: float | None
    reason: str


def validate_source_gate(
    artifact: Mapping[str, Any], confirmation: Mapping[str, Any],
) -> None:
    """Fail closed unless the exact source artifact passed fresh confirmation."""

    validate_effect_program(artifact)
    if confirmation.get("source_gate_passed") is not True:
        raise ValueError("Sokoban source effect program did not pass its gate")
    if confirmation.get("next_step") != "FREEZE_NEW_TARGET_SPLIT":
        raise ValueError("Sokoban source confirmation did not authorize a target")
    if confirmation.get("artifact_sha256") != artifact.get("artifact_sha256"):
        raise ValueError("Sokoban source artifact/confirmation mismatch")
    gates = confirmation.get("gates", {})
    if not all(gates.get(name) is True for name in (
        "coverage", "accuracy", "control_superiority",
    )):
        raise ValueError("Sokoban source confirmation gates are incomplete")


def grounded_effect_predicates(
    predictions: np.ndarray,
    *,
    commit_index: int,
    all_visible_constraints_satisfied: bool,
    probability_threshold: float,
) -> dict[str, bool]:
    """Map target-native causal predictions to source-program predicates."""

    changed, terminated, reward, _ = range(len(OUTCOME_NAMES))
    commit_changed = float(predictions[commit_index, changed])
    commit_reward = float(predictions[commit_index, reward])
    commit_termination = float(predictions[commit_index, terminated])
    commit_available = commit_changed > probability_threshold
    positive_effect = bool(
        all_visible_constraints_satisfied
        and commit_available
        and commit_reward > probability_threshold
    )
    return {
        "commit_available": commit_available,
        "direct_progress_available": positive_effect,
        "assignment_improvement_available": positive_effect,
        "regression_observed": bool(
            commit_termination > probability_threshold
            and commit_reward <= probability_threshold
        ),
        "deadlock_observed": bool(
            commit_termination > probability_threshold
            and not all_visible_constraints_satisfied
        ),
    }


def choose_sokoban_effect_action(
    *,
    condition: str,
    predictions: np.ndarray,
    semantics: Sequence[Mapping[str, Any]],
    source_models: Mapping[str, Any],
    visible_satisfied: bool,
    visible_unsatisfied: bool,
    prior_no_effect: bool,
    remaining_fraction: float,
    previous_action: str | None,
    candidates: Sequence[str],
    uncertainty_scale: float,
    decision_margin: float,
) -> EffectTransferDecision:
    """Select one WebShop action through the frozen Sokoban effect program.

    ``source_models`` contains the validated symbolic artifact for compatibility
    with the V9 execution harness; it is never fit on target receipts.  The last
    four scalar arguments are accepted under the same frozen harness interface,
    but source-specific duration, uncertainty, and margins are deliberately not
    transferred.
    """

    del remaining_fraction, previous_action, uncertainty_scale, decision_margin
    if condition not in CONDITIONS:
        raise ValueError(f"unknown Sokoban-WebShop condition: {condition}")
    if len(candidates) != len(semantics) or predictions.shape != (
        len(candidates), len(OUTCOME_NAMES)
    ):
        raise ValueError("candidate prediction alignment mismatch")
    if condition == "target_only":
        return EffectTransferDecision(0, "TARGET", True, None, None, "target_rank_zero")

    commit_indices = [index for index, row in enumerate(semantics) if row["is_commit"]]
    position_indices = [
        index for index, row in enumerate(semantics)
        if not row["is_commit"] and not row["is_noop"]
    ]
    if not commit_indices or not position_indices:
        return EffectTransferDecision(
            0, "TARGET", True, None, None, "missing_position_or_commit",
        )
    changed, terminated, reward, progress = range(len(OUTCOME_NAMES))
    best_commit = max(
        commit_indices,
        key=lambda index: (
            predictions[index, reward] - predictions[index, terminated], -index,
        ),
    )
    best_position = max(
        position_indices,
        key=lambda index: (
            predictions[index, progress],
            predictions[index, changed] - predictions[index, terminated],
            -index,
        ),
    )
    if condition == "target_native_myopic":
        selected = max(
            range(len(candidates)),
            key=lambda index: (
                predictions[index, reward], -predictions[index, terminated], -index,
            ),
        )
        return EffectTransferDecision(
            selected,
            "COMMIT" if semantics[selected]["is_commit"] else "POSITION",
            False,
            None,
            None,
            "maximum_predicted_immediate_reward",
        )

    all_visible_constraints_satisfied = visible_satisfied and not visible_unsatisfied
    # This is a target applicability gate, shared unchanged by every source
    # condition.  It permits source authority only at a verified ready state or
    # after an observed no-effect transition with a grounded preparation action.
    if not all_visible_constraints_satisfied and (
        not prior_no_effect
        or predictions[best_position, progress] <= 0.5
        or predictions[best_position, changed] <= 0.5
    ):
        return EffectTransferDecision(
            0, "TARGET", True, None, None, "no_grounded_position_effect",
        )

    artifact = source_models.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("validated Sokoban source artifact is missing")
    predicates = grounded_effect_predicates(
        predictions,
        commit_index=best_commit,
        all_visible_constraints_satisfied=all_visible_constraints_satisfied,
        probability_threshold=0.5,
    )
    source_condition = SOURCE_CONDITION_MAP[condition]
    selected_option = select_option(source_condition, predicates)
    selected = best_commit if selected_option == "COMMIT" else best_position
    return EffectTransferDecision(
        selected,
        selected_option,
        False,
        float(selected_option == "POSITION"),
        float(selected_option == "COMMIT"),
        f"sokoban_{source_condition}",
    )


__all__ = [
    "CONDITIONS",
    "SOURCE_CONDITION_MAP",
    "EffectTransferDecision",
    "choose_sokoban_effect_action",
    "grounded_effect_predicates",
    "validate_source_gate",
]
