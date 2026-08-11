from __future__ import annotations

import copy

import pytest

from motif_transfer.sokoban_effect_program import (
    effect_predicates,
    select_option,
    validate_effect_program,
)


def test_effect_guard_uses_positive_effect_not_commit_frequency() -> None:
    positive = effect_predicates((0, 1, 1, 1, 0.25, 0, 0, 0, 0.1, 1, 0.5, 0.5))
    merely_available = effect_predicates(
        (0, 1, 1, 1, 0.25, 0, 0, 0, 0, 1, 0.5, 0.5)
    )
    assert select_option("authentic_effect_guard", positive) == "COMMIT"
    assert select_option("authentic_effect_guard", merely_available) == "POSITION"
    assert select_option("commit_availability_only", merely_available) == "COMMIT"


def test_inverted_and_occupancy_controls_break_effect_semantics() -> None:
    predicates = {
        "commit_available": True,
        "direct_progress_available": True,
        "assignment_improvement_available": False,
        "regression_observed": False,
        "deadlock_observed": False,
    }
    assert select_option("inverted_effect_guard", predicates) == "POSITION"
    assert select_option("position_occupancy_prior", predicates) == "POSITION"


def test_effect_artifact_hash_is_fail_closed() -> None:
    artifact = {
        "artifact_version": "SOKOBAN_EFFECT_PROGRAM_V2",
        "program": {"rules": [{"select": "COMMIT"}, {"select": "POSITION"}]},
    }
    from motif_transfer.contracts import stable_hash

    artifact["artifact_sha256"] = stable_hash(artifact)
    validate_effect_program(artifact)
    broken = copy.deepcopy(artifact)
    broken["program"]["rules"][0]["select"] = "POSITION"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_effect_program(broken)
