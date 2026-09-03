from __future__ import annotations

import numpy as np
import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.webshop_sokoban_effect_transfer import (
    choose_sokoban_effect_action,
    grounded_effect_predicates,
    validate_source_gate,
)


def _artifact() -> dict:
    body = {
        "artifact_version": "SOKOBAN_EFFECT_PROGRAM_V2",
        "program": {
            "rules": [
                {"select": "COMMIT"},
                {"select": "POSITION"},
            ],
        },
    }
    return body | {"artifact_sha256": stable_hash(body)}


def _decision(condition: str, *, satisfied: bool, prior_no_effect: bool = True):
    # POSITION predicts state/prerequisite progress; COMMIT predicts state
    # change, termination, and reward.  Columns follow OUTCOME_NAMES.
    predictions = np.asarray([
        [0.90, 0.05, 0.05, 0.90],
        [0.95, 0.90, 0.90, 0.05],
    ])
    semantics = [
        {"is_commit": False, "is_noop": False},
        {"is_commit": True, "is_noop": False},
    ]
    return choose_sokoban_effect_action(
        condition=condition,
        predictions=predictions,
        semantics=semantics,
        source_models={"artifact": _artifact()},
        visible_satisfied=satisfied,
        visible_unsatisfied=not satisfied,
        prior_no_effect=prior_no_effect,
        remaining_fraction=0.5,
        previous_action=None,
        candidates=("click('constraint')", "click('buy')"),
        uncertainty_scale=0.0,
        decision_margin=0.0,
    )


def test_source_gate_fails_closed_and_accepts_exact_confirmed_artifact() -> None:
    artifact = _artifact()
    confirmation = {
        "source_gate_passed": True,
        "next_step": "FREEZE_NEW_TARGET_SPLIT",
        "artifact_sha256": artifact["artifact_sha256"],
        "gates": {"coverage": True, "accuracy": True, "control_superiority": True},
    }
    validate_source_gate(artifact, confirmation)
    with pytest.raises(ValueError, match="did not pass"):
        validate_source_gate(artifact, confirmation | {"source_gate_passed": False})
    with pytest.raises(ValueError, match="mismatch"):
        validate_source_gate(artifact, confirmation | {"artifact_sha256": "wrong"})


def test_authentic_effect_guard_is_state_dependent() -> None:
    ready = _decision("authentic_sokoban_effect_plus_target", satisfied=True)
    not_ready = _decision("authentic_sokoban_effect_plus_target", satisfied=False)
    assert (ready.abstract_kind, ready.selected_index) == ("COMMIT", 1)
    assert (not_ready.abstract_kind, not_ready.selected_index) == ("POSITION", 0)


def test_source_controls_create_semantic_action_contrast() -> None:
    availability = _decision(
        "commit_availability_control_plus_target", satisfied=False,
    )
    inverted_ready = _decision("inverted_effect_control_plus_target", satisfied=True)
    position_ready = _decision("position_prior_control_plus_target", satisfied=True)
    assert availability.abstract_kind == "COMMIT"
    assert inverted_ready.abstract_kind == "POSITION"
    assert position_ready.abstract_kind == "POSITION"


def test_target_applicability_gate_is_shared_and_can_abstain() -> None:
    decision = _decision(
        "authentic_sokoban_effect_plus_target",
        satisfied=False,
        prior_no_effect=False,
    )
    assert decision.source_abstained
    assert decision.reason == "no_grounded_position_effect"


def test_grounded_predicates_require_target_readiness_for_positive_effect() -> None:
    predictions = np.asarray([[0.9, 0.9, 0.9, 0.0]])
    not_ready = grounded_effect_predicates(
        predictions,
        commit_index=0,
        all_visible_constraints_satisfied=False,
        probability_threshold=0.5,
    )
    ready = grounded_effect_predicates(
        predictions,
        commit_index=0,
        all_visible_constraints_satisfied=True,
        probability_threshold=0.5,
    )
    assert not not_ready["direct_progress_available"]
    assert ready["direct_progress_available"]
