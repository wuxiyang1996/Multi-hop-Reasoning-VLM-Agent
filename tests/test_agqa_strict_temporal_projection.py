from dataclasses import dataclass

import pytest

from motif_transfer.agqa_strict_temporal_projection import (
    action_localization_native_view,
    project_native_window_strict,
    recenter_action_anchor_events,
    recenter_degenerate_boundary_action_events,
    rebind_nested_action_patients,
    strict_track_for_label,
)
from motif_transfer.agqa_query_grounder_v2 import TypedRoleEvent


def test_action_localization_native_view_prefers_separate_dense_field():
    assert action_localization_native_view({
        "native_frame_index_view": [0, 50],
        "localization_native_frame_index_view": [20, 21],
    }) == (20, 21)
    assert action_localization_native_view({
        "native_frame_index_view": [0, 50],
    }) == (0, 50)


def test_recenter_action_anchor_events_changes_only_patient_action():
    patient = TypedRoleEvent(
        "R0", "holding", (("agent", "T0"), ("patient", "T1")),
        1, 3, (2,), .8, ("S2",),
    )
    relation = TypedRoleEvent(
        "R1", "near", (("agent", "T0"), ("relation_object", "T1")),
        1, 3, (2,), .8, ("S2",),
    )
    rows, count = recenter_action_anchor_events(
        (patient, relation),
        {"action_obligations": [{
            "slot_id": "S2", "native_frame_index_view": [20, 40],
        }]},
        [0, 10, 20, 30, 40, 50],
    )
    assert count == 1
    assert (rows[0].start_frame, rows[0].end_frame, rows[0].evidence_frames) == (3, 3, (3,))
    assert rows[1] == relation


def test_boundary_recenter_only_repairs_temporally_degenerate_clip():
    event = TypedRoleEvent(
        "R0", "holding", (("agent", "T0"), ("patient", "T1")),
        0, 3, (2,), .8, ("S2",),
    )
    before = {
        "temporal_operator": "BEFORE", "source_frame_count": 100,
        "action_obligations": [{
            "slot_id": "S2", "native_frame_index_view": [0, 50],
        }],
    }
    rows, count = recenter_degenerate_boundary_action_events(
        (event,), before, [0, 10, 20, 30, 40, 50],
    )
    assert count == 1
    assert (rows[0].start_frame, rows[0].end_frame) == (2, 2)

    nondegenerate = {
        **before,
        "action_obligations": [{
            "slot_id": "S2", "native_frame_index_view": [20, 50],
        }],
    }
    rows, count = recenter_degenerate_boundary_action_events(
        (event,), nondegenerate, [0, 10, 20, 30, 40, 50],
    )
    assert count == 0
    assert rows == (event,)


@dataclass(frozen=True)
class Track:
    track_id: str
    canonical_label: str
    evidence_frames: tuple[int, ...]
    confidence: float


@dataclass(frozen=True)
class Event:
    event_id: str
    roles: tuple[tuple[str, str], ...]
    start_frame: int
    end_frame: int
    semantic_slot_ids: tuple[str, ...]
    confidence: float

    @property
    def role_map(self):
        return dict(self.roles)


@dataclass(frozen=True)
class Slot:
    slot_id: str
    kind: str
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class Semantic:
    slots: tuple[Slot, ...]


def test_strict_projection_never_rounds_across_after_or_before_boundary():
    sampled = [0, 10, 20, 30]
    assert project_native_window_strict(sampled, 11, 30) == (2, 3)
    assert project_native_window_strict(sampled, 0, 19) == (0, 1)


def test_strict_projection_marks_an_unsampled_interval_empty():
    assert project_native_window_strict([0, 10, 20, 30], 11, 19) == (1, 0)


def test_strict_projection_rejects_an_invalid_sampling_lattice():
    with pytest.raises(ValueError):
        project_native_window_strict([0, 20, 10], 0, 20)


def test_track_selection_requires_in_window_pixel_evidence():
    tracks = (
        Track("T1", "cup", (1, 2), 0.99),
        Track("T2", "cup", (8, 9), 0.80),
    )
    canonicalize = lambda value: value.casefold()
    assert strict_track_for_label(
        tracks, "CUP", 7, 10, canonicalize=canonicalize,
    ) == ("T2", 8)
    assert strict_track_for_label(
        tracks, "cup", 3, 6, canonicalize=canonicalize,
    ) is None


def test_nested_action_patient_follows_typed_reference_edge():
    tracks = (
        Track("T1", "cup", (4,), 0.8),
        Track("T2", "book", (4,), 0.9),
        Track("T3", "cup", (5,), 0.95),
    )
    events = (
        Event("R0", (("agent", "T2"), ("patient", "T2")), 3, 6, ("S0",), 0.7),
        Event("R1", (("agent", "T2"), ("relation_object", "T1")), 1, 1, ("S3",), 0.9),
    )
    semantic = Semantic((
        Slot("S0", "LITERAL"),
        Slot("S1", "ACTION", ("S0", "S2")),
        Slot("S2", "QUERY_GOAL", ("S3",)),
        Slot("S3", "RELATION"),
    ))
    rebound, count = rebind_nested_action_patients(events, tracks, semantic)
    assert count == 1
    assert rebound[0].role_map["patient"] == "T3"
