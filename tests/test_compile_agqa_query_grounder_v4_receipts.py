from motif_transfer.agqa_query_grounder_v2 import (
    EntityTrack, QueryCandidateEvidence, QueryGroundingV2Receipt, TypedRoleEvent,
)
from scripts.compile_agqa_query_grounder_v4_receipts import adjudicated_receipt


SHA = "a" * 64


def _base():
    return QueryGroundingV2Receipt.create(
        task_id="q", video_sha256=SHA, semantic_slots_sha256=SHA,
        selected_frame_indices=(0, 10, 20, 30),
        selected_frame_sha256s=(SHA, SHA, SHA, SHA),
        tracks=(
            EntityTrack("T0", "person", (), (0, 1, 2, 3), 1.0),
            EntityTrack("T1", "cup", (), (1,), 0.8),
            EntityTrack("T2", "book", (), (2,), 0.9),
        ),
        events=(
            TypedRoleEvent("R0", "holding", (("agent", "T0"), ("patient", "T1")),
                           1, 1, (1,), 0.8, ("S1",)),
            TypedRoleEvent("R1", "opening", (("agent", "T0"), ("patient", "T1")),
                           0, 1, (0,), 0.7, ("S9",)),
        ),
        candidates=(QueryCandidateEvidence("T1", "patient", "SUPPORTED", 0.8, (1,)),),
        public_ontology_sha256=SHA, grounder_backend_sha256=SHA, provider_calls=0,
    )


def test_adjudication_replaces_only_root_binding_and_maps_shared_frames():
    result = adjudicated_receipt(
        base=_base(),
        base_row={"root_semantic_slot_ids": ["S1"], "requested_role": "patient",
                  "root_predicate": "holding"},
        adjudicated={
            "selected_candidate": {"track_id": "T2"}, "confidence": 0.95,
            "evidence_frame_ids": [2], "usage": {"finish_reason": "stop"},
        },
        raw_video={"sampled_original_frame_indices": [0, 10, 20, 30]},
        support_threshold=0.9, backend_sha256=SHA,
    )
    assert [(row.track_id, row.status) for row in result.candidates] == [("T2", "SUPPORTED")]
    roots = [event for event in result.events if "S1" in event.semantic_slot_ids]
    assert len(roots) == 1
    assert roots[0].role_map["patient"] == "T2"
    assert roots[0].evidence_frames == (2,)
    assert any("S9" in event.semantic_slot_ids for event in result.events)


def test_below_threshold_binding_is_unknown_and_cannot_commit():
    result = adjudicated_receipt(
        base=_base(),
        base_row={"root_semantic_slot_ids": ["S1"], "requested_role": "patient",
                  "root_predicate": "holding"},
        adjudicated={
            "selected_candidate": {"track_id": "T2"}, "confidence": 0.7,
            "evidence_frame_ids": [2], "usage": {"finish_reason": "stop"},
        },
        raw_video={"sampled_original_frame_indices": [0, 10, 20, 30]},
        support_threshold=0.9, backend_sha256=SHA,
    )
    assert result.candidates[0].status == "UNKNOWN"
    assert result.candidates[0].evidence_frames == ()
