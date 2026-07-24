from __future__ import annotations

import pytest

from motif_transfer.contracts import Lifecycle
from motif_transfer.transfer_matrix import (
    REQUIRED_TARGET_CONDITIONS,
    TargetConditionOutcome,
    TransferExperimentSpec,
    evaluate_transfer_matrix,
)


def _row(condition: str, *, ars: float = 0.2, support: int = 0) -> TargetConditionOutcome:
    return TargetConditionOutcome(
        sample_id="s1", condition=condition, initial_asset_hash="asset",
        decision_model_hash="decision", harness_model_hash="harness",
        judge_model_hash="judge", tool_contract_hash="tools", budget_hash="budget",
        apr_pass=False, ars=ars, tool_calls=3, invalid_calls=0,
        max_output_tokens=4096, temperature=0.0,
        source_advisories=int(condition in {"authentic_game_source", "renamed_game_source"}),
        live_supported_advisories=support,
    )


def test_confirmatory_spec_requires_source_supported() -> None:
    spec = TransferExperimentSpec(
        "e", "vtb", "manifest", "candidate", "sha", Lifecycle.CANDIDATE,
        "decision", "harness", "o4-mini", "tools", 20,
    )
    with pytest.raises(ValueError):
        spec.validate()
    spec.validate(diagnostic_only=True)


def test_matrix_requires_matched_six_conditions() -> None:
    with pytest.raises(ValueError):
        evaluate_transfer_matrix([_row("target_only")])
    rows = [_row(name) for name in REQUIRED_TARGET_CONDITIONS]
    rows[-1] = TargetConditionOutcome(**({**rows[-1].__dict__, "budget_hash": "different"}))
    with pytest.raises(ValueError):
        evaluate_transfer_matrix(rows)


def test_no_support_and_positive_pilot_are_distinct() -> None:
    rows = [_row(name, ars=0.2) for name in REQUIRED_TARGET_CONDITIONS]
    assert evaluate_transfer_matrix(rows)["status"] == "NO_LIVE_SUPPORT"
    rows = [
        _row(name, ars=0.8 if name in {"authentic_game_source", "renamed_game_source"} else 0.2,
             support=1 if name in {"authentic_game_source", "renamed_game_source"} else 0)
        for name in REQUIRED_TARGET_CONDITIONS
    ]
    result = evaluate_transfer_matrix(rows)
    assert result["status"] == "STRUCTURE_TRANSFER_PILOT"
    assert result["estimands"]["authentic_minus_best_destructive_control_ars"] == pytest.approx(0.6)
