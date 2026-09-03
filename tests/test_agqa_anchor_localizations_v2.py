from scripts.collect_agqa_anchor_localizations_v2 import (
    _anchor_intervals,
    _anchor_frame_ids_v2,
    _anchor_prompt,
    _anchor_specs_v2,
    _validate_anchor_payload_v2,
    _restrict_anchor_frames_to_named_objects,
    _anchor_visible_track_ids,
    _artifact_status,
)
from motif_transfer.agqa_query_grounder_v2 import EntityTrack


def test_anchor_intervals_are_compiled_only_from_supported_pixel_evidence() -> None:
    rows = [
        {"anchor_id": "A0", "status": "SUPPORTED", "confidence": 0.9,
         "evidence_frame_ids": [8, 3, 8, 6]},
        {"anchor_id": "A1", "status": "UNKNOWN", "confidence": 0.0,
         "evidence_frame_ids": []},
    ]
    assert _anchor_intervals(rows) == [[3, 8]]


def test_development_anchor_status_cannot_claim_transfer_evidence() -> None:
    assert _artifact_status(True) == (
        "CONSUMED_DEVELOPMENT_ANCHOR_PILOT_NOT_TRANSFER_EVIDENCE"
    )
    assert _artifact_status(False).endswith("FROZEN_BEFORE_TARGET_OUTCOME")


def test_anchor_prompt_distinguishes_transition_from_persistent_result() -> None:
    prompt = _anchor_prompt(
        [{"anchor_id": "A0", "phrase": "opening a refrigerator"}],
        [4, 8, 12],
    )
    assert "boundary frames" in prompt
    assert "resulting state persists" in prompt
    assert "answer" in prompt


def test_anchor_track_selection_uses_only_public_phrase_objects_and_person() -> None:
    class Stable:
        tracks = (
            EntityTrack("T0", "person", (), (0,), 1.0),
            EntityTrack("T1", "refrigerator", (), (0, 1), 0.9),
            EntityTrack("T2", "cup", ("cup/glass/bottle",), (1,), 0.8),
        )

    assert _anchor_visible_track_ids(
        Stable(), [{"anchor_id": "A0", "phrase": "opening a refrigerator"}],
    ) == frozenset({"T0", "T1"})


def test_named_object_anchor_cannot_present_frames_without_that_object() -> None:
    detections = {
        0: [("T0", {})],
        4: [("T0", {}), ("T1", {})],
        8: [("T0", {}), ("T1", {})],
    }
    assert _restrict_anchor_frames_to_named_objects(
        [0, 4, 8], detections, frozenset({"T0", "T1"}),
    ) == [4, 8]
    assert _restrict_anchor_frames_to_named_objects(
        [0, 4, 8], detections, frozenset({"T0"}),
    ) == [0, 4, 8]


def test_parser_only_anchor_without_action_score_view_uses_uniform_frames() -> None:
    anchors = _anchor_specs_v2({
        "action_obligations": [{"phrase": "holding a cup"}],
    })
    assert anchors == [{
        "anchor_id": "A0", "phrase": "holding a cup",
        "native_frame_index_view": [],
    }]
    raw = {"sampled_original_frame_indices": list(range(64))}
    assert _anchor_frame_ids_v2(raw, anchors, 20) == [0, 9, 18, 27, 36, 45, 54, 63]


def test_unknown_anchor_canonicalization_discards_non_evidence() -> None:
    payload = {"anchors": [{
        "anchor_id": "A0", "status": "UNKNOWN", "confidence": 0.0,
        "evidence_frame_ids": [4, 8],
    }]}
    assert _validate_anchor_payload_v2(payload, ["A0"], [4, 8]) == [{
        "anchor_id": "A0", "status": "UNKNOWN", "confidence": 0.0,
        "evidence_frame_ids": [],
    }]
