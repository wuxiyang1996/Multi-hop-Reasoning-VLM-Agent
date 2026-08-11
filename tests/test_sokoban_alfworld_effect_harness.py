from __future__ import annotations

import copy

from motif_transfer.contracts import stable_hash
from motif_transfer.sokoban_alfworld_effect_harness import (
    choose_action,
    ground_effect_predicates,
)


def _artifact() -> dict:
    body = {
        "artifact_version": "SOKOBAN_EFFECT_PROGRAM_V2",
        "program": {"rules": [{"select": "COMMIT"}, {"select": "POSITION"}]},
    }
    return body | {"artifact_sha256": stable_hash(body)}


def _grounded(required: str = "ACQUIRE") -> dict:
    return {
        "go to cabinet": {
            "option": "SEARCH", "required_option": required,
            "applicability": 0.9, "completion": 0.2, "binding": 0.1,
            "policy": 0.8,
        },
        "take apple from cabinet": {
            "option": "ACQUIRE", "required_option": required,
            "applicability": 0.9, "completion": 0.9, "binding": 0.9,
            "policy": 0.4,
        },
        "put apple in fridge": {
            "option": "PLACE", "required_option": required,
            "applicability": 0.2, "completion": 0.2, "binding": 0.9,
            "policy": 0.3,
        },
    }


def test_effect_predicate_uses_target_neural_scores() -> None:
    predicates = ground_effect_predicates(_grounded(), effect_threshold=0.25)
    assert predicates["commit_available"]
    assert predicates["direct_progress_available"]
    assert predicates["best_commit_effect_score"] == 0.81


def test_authentic_path_realizes_positive_commit() -> None:
    decision = choose_action(
        condition="authentic_source_effect_harness", grounded=_grounded(),
        history=(), source_artifact=_artifact(), effect_threshold=0.25,
    )
    assert decision["source_selected_option"] == "COMMIT"
    assert decision["action"] == "take apple from cabinet"
    assert decision["source_admitted"]


def test_authentic_path_is_invariant_to_required_option_field() -> None:
    first = choose_action(
        condition="authentic_source_effect_harness", grounded=_grounded("ACQUIRE"),
        history=(), source_artifact=_artifact(), effect_threshold=0.25,
    )
    changed = copy.deepcopy(_grounded("PLACE"))
    second = choose_action(
        condition="authentic_source_effect_harness", grounded=changed,
        history=(), source_artifact=_artifact(), effect_threshold=0.25,
    )
    assert first["source_selected_option"] == second["source_selected_option"]
    assert first["action"] == second["action"]


def test_no_positive_effect_repositions_with_target_policy() -> None:
    grounded = _grounded()
    grounded["take apple from cabinet"]["completion"] = 0.01
    decision = choose_action(
        condition="authentic_source_effect_harness", grounded=grounded,
        history=(), source_artifact=_artifact(), effect_threshold=0.25,
    )
    assert decision["source_selected_option"] == "POSITION"
    assert decision["action"] == "go to cabinet"
