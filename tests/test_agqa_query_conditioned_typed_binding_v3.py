from types import SimpleNamespace

import pytest

from motif_transfer.agqa_query_grounder_v2 import EntityTrack
from scripts.collect_agqa_query_conditioned_typed_binding_v3 import (
    _label_inventory,
    _scope,
    _uniform_frame_ids,
    _validate,
)


def test_scope_executes_each_temporal_operator_without_answers():
    assert _scope({"temporal_operator": "VIDEO"}, frame_count=64, uncertainty=4) == (0, 63)
    assert _scope(
        {"temporal_operator": "BEFORE", "anchor_intervals": [[10, 12]]},
        frame_count=64, uncertainty=4,
    ) == (0, 9)
    assert _scope(
        {"temporal_operator": "AFTER", "anchor_intervals": [[10, 12]]},
        frame_count=64, uncertainty=4,
    ) == (13, 63)
    assert _scope(
        {"temporal_operator": "WHILE", "anchor_intervals": [[10, 12]]},
        frame_count=64, uncertainty=4,
    ) == (6, 16)
    assert _scope(
        {
            "temporal_operator": "BETWEEN",
            "anchor_intervals": [[8, 10], [20, 22]],
        }, frame_count=64, uncertainty=2,
    ) == (9, 21)
    assert _scope(
        {"temporal_operator": "WHILE", "anchor_intervals": []},
        frame_count=64, uncertainty=4,
    ) is None


def test_uniform_frames_cover_both_scope_endpoints():
    assert _uniform_frame_ids(4, 7, 8) == [4, 5, 6, 7]
    sampled = _uniform_frame_ids(3, 63, 12)
    assert len(sampled) == 12
    assert sampled[0] == 3
    assert sampled[-1] == 63
    assert sampled == sorted(set(sampled))


def test_inventory_merges_fragmented_tracks_by_public_label():
    stable = SimpleNamespace(tracks=(
        EntityTrack("T0", "person", (), (0, 1, 2), 0.99),
        EntityTrack("T1", "phone", (), (0, 1), 0.7),
        EntityTrack("T2", "phone", (), (2, 3), 0.9),
        EntityTrack("T3", "chair", (), (8,), 0.8),
    ))
    assert _label_inventory(stable, lower=0, upper=3) == [
        {
            "candidate_id": "O0",
            "canonical_label": "phone",
            "track_ids": ["T1", "T2"],
            "visible_frame_ids": [0, 1, 2, 3],
            "detector_confidence": 0.9,
        }
    ]


def test_validator_fails_closed_when_cited_object_track_is_not_visible():
    payload = {
        "status": "SUPPORTED", "selected_candidate_id": "O0",
        "confidence": 0.9, "evidence_frame_ids": [2],
    }
    assert _validate(
        payload, candidate_ids={"O0"}, frame_ids={1, 2},
        supportable_by_candidate={"O0": {1}},
    ) == {
        "status": "UNKNOWN", "selected_candidate_id": "UNKNOWN",
        "confidence": 0.0, "evidence_frame_ids": [],
    }


def test_validator_rejects_malformed_supported_decision():
    with pytest.raises(ValueError, match="needs one inventory ID"):
        _validate(
            {
                "status": "SUPPORTED", "selected_candidate_id": "UNKNOWN",
                "confidence": 0.9, "evidence_frame_ids": [],
            }, candidate_ids={"O0"}, frame_ids={1},
            supportable_by_candidate={"O0": {1}},
        )


def test_forced_proposal_is_typed_but_not_a_safety_authorization():
    assert _validate(
        {
            "status": "PROPOSED", "selected_candidate_id": "O0",
            "confidence": 0.7, "evidence_frame_ids": [1],
        }, candidate_ids={"O0"}, frame_ids={1, 2},
        supportable_by_candidate={"O0": {1}}, force_proposal=True,
    ) == {
        "status": "PROPOSED", "selected_candidate_id": "O0",
        "confidence": 0.7, "evidence_frame_ids": [1],
    }


def test_forced_proposal_is_preserved_for_independent_verification():
    assert _validate(
        {
            "status": "PROPOSED", "selected_candidate_id": "O0",
            "confidence": 0.8, "evidence_frame_ids": [2],
        }, candidate_ids={"O0"}, frame_ids={1, 2},
        supportable_by_candidate={"O0": {1}}, force_proposal=True,
    ) == {
        "status": "PROPOSED", "selected_candidate_id": "O0",
        "confidence": 0.8, "evidence_frame_ids": [2],
    }
