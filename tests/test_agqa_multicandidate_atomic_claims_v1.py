from scripts.collect_agqa_multicandidate_atomic_claims_v1 import (
    _canonicalize_candidate_evidence,
    _root_window,
    _typed_claim,
)


def test_typed_claim_preserves_explicit_patient_role() -> None:
    claim = _typed_claim(
        "holding", "patient", {"canonical_label": "phone"},
    )
    assert claim == "P0 visibly holding this candidate (phone)"


def test_typed_claim_preserves_relation_orientation() -> None:
    claim = _typed_claim(
        "behind", "relation_object", {"canonical_label": "table"},
    )
    assert "P0" in claim
    assert "behind" in claim
    assert "table" in claim


def test_candidate_evidence_fails_closed_without_track_corroboration() -> None:
    rows = [{
        "candidate_id": "C0",
        "status": "SUPPORTED",
        "confidence": 0.9,
        "evidence_frame_ids": [2, 4],
    }]
    candidates = [{"candidate_id": "C0", "all_frame_ids": [2, 8]}]
    assert _canonicalize_candidate_evidence(rows, candidates) == [{
        "candidate_id": "C0",
        "status": "SUPPORTED",
        "confidence": 0.9,
        "evidence_frame_ids": [2],
    }]


def test_candidate_evidence_abstains_when_all_citations_are_uncorroborated() -> None:
    rows = [{
        "candidate_id": "C0",
        "status": "SUPPORTED",
        "confidence": 0.9,
        "evidence_frame_ids": [4],
    }]
    candidates = [{"candidate_id": "C0", "all_frame_ids": [2, 8]}]
    assert _canonicalize_candidate_evidence(rows, candidates) == [{
        "candidate_id": "C0",
        "status": "UNKNOWN",
        "confidence": 0.0,
        "evidence_frame_ids": [],
    }]


def test_root_window_executes_frozen_anchor_interval() -> None:
    anchors = [{
        "anchor_id": "A0", "status": "SUPPORTED", "confidence": 0.9,
        "evidence_frame_ids": [10, 14],
    }]
    assert _root_window(
        "WHILE", anchors, frame_count=64, uncertainty=4,
    ) == (6, 18)
    assert _root_window(
        "BEFORE", anchors, frame_count=64, uncertainty=4,
    ) == (0, 9)


def test_root_window_fails_closed_without_required_anchor() -> None:
    assert _root_window(
        "AFTER", [], frame_count=64, uncertainty=4,
    ) is None
