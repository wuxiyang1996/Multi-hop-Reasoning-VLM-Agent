from scripts.collect_agqa_evidence_graph_query_binding_v1 import (
    _candidate_inventory,
    _event_rows,
    _validate,
)


def _video():
    return {
        "stable_tracks": [
            {"track_id": "T0", "canonical_label": "person", "evidence_frames": [0, 1], "confidence": .9},
            {"track_id": "T1", "canonical_label": "phone", "evidence_frames": [0], "confidence": .7},
            {"track_id": "T2", "canonical_label": "phone", "evidence_frames": [1], "confidence": .8},
        ],
        "events": [{
            "event_id": "V0", "predicate": "holding",
            "object_track_id": "T2", "object_role": "patient",
            "start_frame": 1, "end_frame": 1, "evidence_frames": [1],
            "confidence": .9,
        }],
    }


def test_inventory_is_label_unique_and_events_bind_to_candidate_ids():
    candidates, labels = _candidate_inventory(_video())
    assert candidates == [{
        "candidate_id": "O0", "canonical_label": "phone",
        "track_ids": ["T1", "T2"], "visible_frame_ids": [0, 1],
        "detector_confidence": .8,
    }]
    assert _event_rows(_video(), candidates, labels)[0]["candidate_id"] == "O0"


def test_graph_proposal_must_cite_an_event_for_selected_candidate():
    events = [
        {"event_id": "V0", "candidate_id": "O0"},
        {"event_id": "V1", "candidate_id": "O1"},
    ]
    assert _validate(
        {
            "status": "PROPOSED", "selected_candidate_id": "O0",
            "confidence": .7, "supporting_event_ids": ["V0"],
        }, candidates={"O0", "O1"}, events=events,
    )["selected_candidate_id"] == "O0"


def test_graph_proposal_rejects_unrelated_event_citation():
    import pytest

    with pytest.raises(ValueError, match="no event involving"):
        _validate(
            {
                "status": "PROPOSED", "selected_candidate_id": "O0",
                "confidence": .7, "supporting_event_ids": ["V1"],
            }, candidates={"O0", "O1"},
            events=[{"event_id": "V1", "candidate_id": "O1"}],
        )
