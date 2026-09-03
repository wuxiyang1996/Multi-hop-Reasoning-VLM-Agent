from __future__ import annotations

import pytest

from motif_transfer.agqa_question_blind_event_grounder import (
    QuestionBlindTypedEvent,
    bind_event_to_semantic_slots,
    canonical_event_predicate,
    deduplicate_question_blind_events,
    object_role_for_predicate,
    parse_question_blind_event_payload,
    parse_question_blind_event_payload_with_rejections,
    query_event_candidates,
    query_temporal_event_candidates,
)


def _event(**overrides) -> QuestionBlindTypedEvent:
    values = {
        "event_id": "V0", "predicate": "holding",
        "subject_track_id": "T0", "object_track_id": "T1",
        "object_role": "patient", "start_frame": 2, "end_frame": 6,
        "evidence_frames": (2, 4, 6), "confidence": 0.8,
        "source_clip_ids": ("C0",),
    }
    values.update(overrides)
    return QuestionBlindTypedEvent(**values)


def test_public_predicates_assign_roles_without_target_outcomes() -> None:
    assert canonical_event_predicate("in_front_of") == "in front of"
    assert canonical_event_predicate("below") == "beneath"
    assert object_role_for_predicate("holding") == "patient"
    assert object_role_for_predicate("watching") == "relation_object"


def test_payload_is_strictly_question_and_answer_blind() -> None:
    payload = {"events": [{
        "event_id": "provider-id", "predicate": "holding",
        "subject_track_id": "T0", "object_track_id": "T1",
        "start_frame_id": 2, "end_frame_id": 6,
        "evidence_frame_ids": [2, 6], "confidence": 0.9,
    }], "uncertainty": ""}
    parsed = parse_question_blind_event_payload(
        payload, clip_id="C0", visible_track_ids=("T0", "T1"),
        person_track_ids=("T0",), presented_frame_ids=(0, 2, 4, 6),
    )
    assert parsed == (_event(evidence_frames=(2, 6), confidence=0.9),)
    with pytest.raises(ValueError, match="authority"):
        parse_question_blind_event_payload(
            {**payload, "answer": "cup"}, clip_id="C0",
            visible_track_ids=("T0", "T1"), person_track_ids=("T0",),
            presented_frame_ids=(0, 2, 4, 6),
        )


def test_payload_rejects_unpresented_frames_and_unknown_tracks() -> None:
    payload = {"events": [{
        "event_id": "x", "predicate": "holding",
        "subject_track_id": "T0", "object_track_id": "T9",
        "start_frame_id": 2, "end_frame_id": 7,
        "evidence_frame_ids": [2, 7], "confidence": 0.9,
    }]}
    with pytest.raises(ValueError):
        parse_question_blind_event_payload(
            payload, clip_id="C0", visible_track_ids=("T0", "T1"),
            person_track_ids=("T0",), presented_frame_ids=(0, 2, 4, 6),
        )


def test_payload_rejects_track_ids_without_same_frame_detector_evidence() -> None:
    payload = {"events": [{
        "event_id": "x", "predicate": "opening",
        "subject_track_id": "T0", "object_track_id": "T1",
        "start_frame_id": 2, "end_frame_id": 4,
        "evidence_frame_ids": [2, 4], "confidence": 0.9,
    }]}
    with pytest.raises(ValueError, match="without detector evidence"):
        parse_question_blind_event_payload(
            payload, clip_id="C0", visible_track_ids=("T0", "T1"),
            person_track_ids=("T0",), presented_frame_ids=(0, 2, 4, 6),
            track_visible_frames={
                "T0": frozenset({0, 2, 4, 6}),
                "T1": frozenset({4, 6}),
            },
        )


def test_invalid_event_does_not_erase_independent_valid_event() -> None:
    valid = {
        "predicate": "holding", "subject_track_id": "T0",
        "object_track_id": "T1", "start_frame_id": 2, "end_frame_id": 4,
        "evidence_frame_ids": [2, 4], "confidence": 0.9,
    }
    invalid = {**valid, "object_track_id": "T2", "evidence_frame_ids": [2]}
    events, rejected = parse_question_blind_event_payload_with_rejections(
        {"events": [invalid, valid]}, clip_id="C0",
        visible_track_ids=("T0", "T1", "T2"), person_track_ids=("T0",),
        presented_frame_ids=(0, 2, 4, 6),
        track_visible_frames={
            "T0": frozenset({0, 2, 4, 6}),
            "T1": frozenset({2, 4}),
            "T2": frozenset({4}),
        },
    )
    assert len(events) == 1
    assert events[0].object_track_id == "T1"
    assert len(rejected) == 1
    assert "without detector evidence" in rejected[0]["reason"]


def test_cross_clip_dedup_requires_typed_track_and_temporal_agreement() -> None:
    merged = deduplicate_question_blind_events((
        _event(event_id="V0", start_frame=2, end_frame=6, evidence_frames=(2, 4, 6)),
        _event(
            event_id="V1", start_frame=4, end_frame=7,
            evidence_frames=(4, 7), confidence=0.9, source_clip_ids=("C1",),
        ),
        _event(
            event_id="V2", object_track_id="T2", start_frame=4, end_frame=7,
            evidence_frames=(4, 7), source_clip_ids=("C1",),
        ),
    ), minimum_interval_iou=0.5)
    assert len(merged) == 2
    assert merged[0].evidence_frames == (2, 4, 6, 7)
    assert merged[0].confidence == 0.9
    assert merged[0].source_clip_ids == ("C0", "C1")
    assert merged[1].object_track_id == "T2"


def test_query_projection_uses_frozen_predicate_role_and_scope() -> None:
    events = (
        _event(event_id="V0", object_track_id="T1", confidence=0.7),
        _event(event_id="V1", object_track_id="T2", confidence=0.9),
        _event(
            event_id="V2", predicate="behind", object_track_id="T3",
            object_role="relation_object", confidence=0.95,
        ),
    )
    ranked = query_event_candidates(
        events, predicate="holding", requested_role="patient",
        lower_frame=0, upper_frame=10,
    )
    assert [row["track_id"] for row in ranked] == ["T2", "T1"]
    assert query_event_candidates(
        events, predicate="holding", requested_role="patient",
        lower_frame=7, upper_frame=10,
    ) == ()


def test_downstream_slot_binding_preserves_typed_roles() -> None:
    contact = bind_event_to_semantic_slots(
        _event(), event_id="R0", semantic_slot_ids=("S1", "S2"),
    )
    assert contact.roles == (("agent", "T0"), ("patient", "T1"))
    spatial = bind_event_to_semantic_slots(
        _event(predicate="behind", object_role="relation_object"),
        event_id="R1", semantic_slot_ids=("S3",),
    )
    assert spatial.roles == (
        ("relation_subject", "T0"), ("relation_object", "T1"),
    )


def test_temporal_query_prefers_nearest_event_before_independent_anchor() -> None:
    events = (
        _event(event_id="V0", object_track_id="T1", start_frame=32, end_frame=39,
               evidence_frames=(32, 36, 39), confidence=0.97),
        _event(event_id="V1", object_track_id="T2", start_frame=42, end_frame=43,
               evidence_frames=(42, 43), confidence=0.91),
        _event(event_id="V2", object_track_id="T3", start_frame=50, end_frame=51,
               evidence_frames=(50, 51), confidence=0.99),
    )
    ranked = query_temporal_event_candidates(
        events, predicate="holding", requested_role="patient",
        temporal_operator="BEFORE", anchor_intervals=((48, 55),),
        temporal_uncertainty_frames=4,
    )
    assert [row["track_id"] for row in ranked] == ["T2", "T1"]


def test_temporal_query_dilates_while_by_acquisition_uncertainty() -> None:
    events = (
        _event(event_id="V0", predicate="behind", object_role="relation_object",
               object_track_id="T4", start_frame=16, end_frame=23,
               evidence_frames=(16, 18, 20, 22), confidence=0.72),
        _event(event_id="V1", predicate="behind", object_role="relation_object",
               object_track_id="T31", start_frame=56, end_frame=63,
               evidence_frames=(56, 58, 60, 62), confidence=0.85),
    )
    ranked = query_temporal_event_candidates(
        events, predicate="behind", requested_role="relation_object",
        temporal_operator="WHILE", anchor_intervals=((27, 48),),
        temporal_uncertainty_frames=4,
    )
    assert [row["track_id"] for row in ranked] == ["T4"]


def test_temporal_query_fails_closed_without_required_anchor() -> None:
    assert query_temporal_event_candidates(
        (_event(),), predicate="holding", requested_role="patient",
        temporal_operator="AFTER", anchor_intervals=(),
    ) == ()
