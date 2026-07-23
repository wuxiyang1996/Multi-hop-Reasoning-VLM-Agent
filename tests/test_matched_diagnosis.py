from __future__ import annotations

from motif_transfer.matched_diagnosis import diagnose_matched_pair


def _row(**updates):
    row = {
        "initial_state_hash": "s",
        "resolved_game_file": "g",
        "decision_backend": {"fixed": True},
        "decision_call_receipts": [{"usage": {"cache_hit": True}}],
        "actions": ["a"],
        "bindings": [],
        "binding_evidence": [],
        "source_fallback_step": 0,
        "source_failures": [],
        "metrics": {"official_score": 0.0},
    }
    row.update(updates)
    return row


def test_rejected_binding_is_safe_fallback() -> None:
    result = diagnose_matched_pair(_row(), _row())
    assert result["matched"] is True
    assert result["status"] == "BINDING_REJECTED_SAFE_FALLBACK"


def test_admitted_but_failed_review_is_not_transfer() -> None:
    treatment = _row(bindings=[{"id": "b"}], source_failures=["REVIEW:invalid"])
    result = diagnose_matched_pair(_row(), treatment)
    assert result["status"] == "NO_VALID_SOURCE_INTERVENTION"


def test_unmatched_pair_is_invalid_before_effect_size() -> None:
    result = diagnose_matched_pair(_row(), _row(initial_state_hash="different"))
    assert result["status"] == "UNMATCHED_INVALID"
