from __future__ import annotations

from motif_transfer.contracts import stable_hash
from motif_transfer.sokoban_alfworld_harness import (
    canonical_target_features,
    choose_action,
)
from motif_transfer.sokoban_commit_skill import FEATURE_NAMES


def _model(position_weight: float, commit_weight: float) -> dict:
    coefficients = [0.0] * (len(FEATURE_NAMES) + 1)
    coefficients[1] = position_weight
    coefficients[2] = commit_weight
    return {
        "kind": "STANDARDIZED_RIDGE_OPTION_VALUE_V1",
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": [0.0] * len(FEATURE_NAMES),
        "feature_scale": [1.0] * len(FEATURE_NAMES),
        "coefficients": coefficients,
        "alpha": 0.5,
        "training_rows": 10,
    }


def _artifact() -> dict:
    body = {
        "artifact_version": "SOKOBAN_COMMIT_SKILL_V1",
        "lifecycle": "DISCOVERY_FROZEN_AWAITING_SOURCE_QUALIFICATION",
        "claim_boundary": "test",
        "plan_sha256": "a" * 64,
        "maximum_solver_nodes": 100,
        "ridge_alpha": 0.5,
        "minimum_discovery_examples_per_option": 1,
        "source_grounding": {},
        "transferable_program": {"options": ["POSITION", "COMMIT"]},
        "models": {
            "authentic": _model(0.0, 2.0),
            "within_state_option_swap": _model(2.0, 0.0),
            "source_marginal": {"constant": 0.5},
        },
        "raw_source_action_tokens_transferred": False,
        "raw_source_coordinates_transferred": False,
    }
    return body | {"artifact_sha256": stable_hash(body)}


def _grounded(required: str = "ACQUIRE") -> dict:
    return {
        "go to cabinet": {
            "option": "SEARCH", "required_option": required,
            "applicability": 0.8, "completion": 0.4, "binding": 0.2,
            "policy": 0.9,
        },
        "take apple from cabinet": {
            "option": "ACQUIRE", "required_option": required,
            "applicability": 0.9, "completion": 0.9, "binding": 0.9,
            "policy": 0.6,
        },
        "put apple in fridge": {
            "option": "PLACE", "required_option": required,
            "applicability": 0.4, "completion": 0.2, "binding": 0.9,
            "policy": 0.4,
        },
    }


def test_canonical_features_match_source_isa_width() -> None:
    grounded = _grounded()
    position = canonical_target_features(
        option="POSITION", grounded=grounded, goal="put apple in fridge",
        history=(),
    )
    commit = canonical_target_features(
        option="COMMIT", grounded=grounded, goal="put apple in fridge",
        history=(),
    )
    assert len(position) == len(FEATURE_NAMES)
    assert len(commit) == len(FEATURE_NAMES)
    assert position[:2] == (1.0, 0.0)
    assert commit[:2] == (0.0, 1.0)


def test_source_selects_option_while_harness_realizes_native_action() -> None:
    decision = choose_action(
        condition="authentic_source_plus_harness",
        grounded=_grounded(), goal="put apple in fridge", history=(),
        source_artifact=_artifact(), identity="task:0",
    )
    assert decision["source_selected_option"] == "COMMIT"
    assert decision["action"] == "take apple from cabinet"
    assert decision["source_admitted"]
    assert decision["changed_option"]


def test_harness_refutes_commit_when_target_precondition_is_not_ready() -> None:
    decision = choose_action(
        condition="authentic_source_plus_harness",
        grounded=_grounded(required="SEARCH"), goal="put apple in fridge",
        history=(), source_artifact=_artifact(), identity="task:1",
    )
    assert decision["source_selected_option"] == "COMMIT"
    assert decision["diagnostic"] == "COMMIT_PRECONDITION_REFUTED"
    assert not decision["source_admitted"]
    assert decision["action"] == decision["fallback_action"]
