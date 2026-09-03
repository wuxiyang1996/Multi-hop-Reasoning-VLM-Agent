from dataclasses import asdict, replace

import pytest

from motif_transfer.agqa_action_genome_grounder import (
    build_stable_tracks, reciprocal_rank_fusion, relation_track_candidates,
)
from motif_transfer.agqa_query_grounder_v2 import (
    EntityTrack, QueryCandidateEvidence, QueryGroundingV2Receipt, TypedRoleEvent,
    adapt_query_grounding_v2, deduplicate_typed_events, requested_query_predicates,
    query_grounding_v2_from_dict, requested_query_role, requested_query_slot_ids,
)
from motif_transfer.agqa_semantic_slots import parse_compact_semantic_target
from motif_transfer.contracts import stable_hash


def _track(track_id: str, label: str) -> EntityTrack:
    return EntityTrack(track_id, label, (), (1, 3), .9)


def _event(event_id: str, start: int, end: int, patient: str = "T1") -> TypedRoleEvent:
    return TypedRoleEvent(
        event_id, "put", (("agent", "T0"), ("patient", patient), ("destination", "T2")),
        start, end, (start, end), .8, ("S2",),
    )


def test_receipt_preserves_patient_and_destination_without_answer_authority() -> None:
    tracks = (_track("T0", "person"), _track("T1", "book"), _track("T2", "table"))
    receipt = QueryGroundingV2Receipt.create(
        task_id="q0", video_sha256=stable_hash("video"),
        semantic_slots_sha256=stable_hash("slots"), selected_frame_indices=tuple(range(8)),
        selected_frame_sha256s=tuple(stable_hash(("f", i)) for i in range(8)),
        tracks=tracks, events=(_event("R0", 1, 3),),
        candidates=(QueryCandidateEvidence("T1", "patient", "SUPPORTED", .9, (2,)),),
        public_ontology_sha256=stable_hash("ontology"),
        grounder_backend_sha256=stable_hash("backend"), provider_calls=2,
    )
    assert receipt.events[0].role_map["patient"] == "T1"
    assert receipt.events[0].role_map["destination"] == "T2"
    assert not any((receipt.answer_read, receipt.official_scene_graph_read,
                    receipt.functional_program_read, receipt.source_controller_read,
                    receipt.target_outcome_read))


def test_dedup_merges_only_same_typed_roles_with_overlapping_intervals() -> None:
    merged = deduplicate_typed_events((
        _event("R0", 1, 5), _event("R1", 2, 5), _event("R2", 2, 5, patient="T3"),
    ))
    assert len(merged) == 2
    assert merged[0].event_id == "R0"
    assert merged[0].evidence_frames == (1, 2, 5)
    assert merged[1].role_map["patient"] == "T3"


def test_receipt_rejects_unknown_track_and_authority_tampering() -> None:
    tracks = (_track("T0", "person"), _track("T1", "book"), _track("T2", "table"))
    with pytest.raises(ValueError, match="unknown entity track"):
        _event("R0", 1, 3, patient="T9").validate(8, frozenset(x.track_id for x in tracks))
    valid = QueryGroundingV2Receipt.create(
        task_id="q0", video_sha256=stable_hash("video"),
        semantic_slots_sha256=stable_hash("slots"), selected_frame_indices=tuple(range(8)),
        selected_frame_sha256s=tuple(stable_hash(("f", i)) for i in range(8)),
        tracks=tracks, events=(_event("R0", 1, 3),), candidates=(),
        public_ontology_sha256=stable_hash("ontology"),
        grounder_backend_sha256=stable_hash("backend"), provider_calls=1,
    )
    body = asdict(valid); body["answer_read"] = True
    tampered = replace(valid, answer_read=True)
    with pytest.raises(ValueError, match="authority boundary"):
        tampered.validate()


def test_requested_role_comes_from_operator_free_semantics() -> None:
    common = dict(task_id="q", question_sha256=stable_hash("q"),
                  parser_sha256=stable_hash("parser"), parser_training_authority="target-dev")
    relation = parse_compact_semantic_target(
        "goal(object, single_reference(observations(relation_description("
        "objects, semantic_tuple(relation, in_front_of)))))", **common,
    )
    action = parse_compact_semantic_target(
        "goal(object, single_reference(observations(relation_description("
        "objects, semantic_tuple(actions, holding)))))", **common,
    )
    assert requested_query_role(relation) == "relation_object"
    assert requested_query_predicates(relation) == ("in front of",)
    assert requested_query_role(action) == "patient"


def test_legacy_relations_category_uses_typed_predicate_for_action_role() -> None:
    semantic = parse_compact_semantic_target(
        "goal(class, single_reference(observations(video, relation_description("
        "frame, semantic_tuple(relations, taking, objects)))))",
        task_id="q", question_sha256=stable_hash("q"), parser_sha256=stable_hash("parser"),
        parser_training_authority="target-dev",
    )
    assert requested_query_predicates(semantic) == ("taking",)
    assert requested_query_role(semantic) == "patient"


def test_requested_predicate_excludes_nested_temporal_anchor() -> None:
    semantic = parse_compact_semantic_target(
        "goal(class, single_reference(observations(time_window(after, "
        "action_reference(putting down, goal(class, single_reference(observations(video, "
        "relation_description(frame, semantic_tuple(relations, opening, objects))))))), "
        "relation_description(frame, semantic_tuple(relations, on the side of, objects)))))",
        task_id="q", question_sha256=stable_hash("q"), parser_sha256=stable_hash("parser"),
        parser_training_authority="target-dev",
    )
    assert requested_query_predicates(semantic) == ("on the side of",)
    assert requested_query_role(semantic) == "relation_object"
    by_id = {row.slot_id: row for row in semantic.slots}
    assert {by_id[value].surface for value in requested_query_slot_ids(semantic)} >= {
        "on the side of", "ordered semantic tuple", "match a typed relation description",
    }
    assert "putting down" not in {
        by_id[value].surface for value in requested_query_slot_ids(semantic)
    }


def test_between_duration_anchor_keeps_final_answer_bearing_relation() -> None:
    semantic = parse_compact_semantic_target(
        "goal(class, single_reference(observations(time_window(between, "
        "semantic_tuple(standing up, duration_extremum(max, "
        "relation_description(video, semantic_tuple(actions)), "
        "relative_duration(goal(end, action), goal(start, action)))), "
        "relation_description(frame, semantic_tuple(relations, touching, objects))))))",
        task_id="q", question_sha256=stable_hash("q"), parser_sha256=stable_hash("parser"),
        parser_training_authority="target-dev",
    )
    assert requested_query_predicates(semantic) == ("touching",)
    assert requested_query_role(semantic) == "patient"
    by_id = {row.slot_id: row for row in semantic.slots}
    surfaces = {by_id[value].surface for value in requested_query_slot_ids(semantic)}
    assert "touching" in surfaces
    assert "standing up" not in surfaces
    assert "max" not in surfaces


def test_endpoint_query_prefers_detailed_relation_over_presence_and_anchor() -> None:
    semantic = parse_compact_semantic_target(
        "goal(class, single_reference(ordered_endpoint(forward, "
        "time_window(after, action_reference(putting down, goal(class, "
        "single_reference(observations(video, relation_description(frame, "
        "semantic_tuple(relations, opening, objects))))))), "
        "presence_question(holding, relation_description(frame, semantic_tuple(relations))), "
        "relation_description(frame, semantic_tuple(relations, holding, objects)))))",
        task_id="q", question_sha256=stable_hash("q"), parser_sha256=stable_hash("parser"),
        parser_training_authority="target-dev",
    )
    assert requested_query_predicates(semantic) == ("holding",)
    assert requested_query_role(semantic) == "patient"
    by_id = {row.slot_id: row for row in semantic.slots}
    surfaces = {by_id[value].surface for value in requested_query_slot_ids(semantic)}
    assert "holding" in surfaces
    assert "opening" not in surfaces


def test_adapter_projects_only_verified_requested_role_without_changing_harness() -> None:
    semantic = parse_compact_semantic_target(
        "goal(object, single_reference(observations(relation_description("
        "objects, semantic_tuple(actions, putting)))))",
        task_id="q0", question_sha256=stable_hash("q0"), parser_sha256=stable_hash("parser"),
        parser_training_authority="target-dev",
    )
    tracks = (_track("T0", "person"), _track("T1", "book"), _track("T2", "table"))
    receipt = QueryGroundingV2Receipt.create(
        task_id="q0", video_sha256=stable_hash("video"),
        semantic_slots_sha256=semantic.receipt_sha256, selected_frame_indices=tuple(range(8)),
        selected_frame_sha256s=tuple(stable_hash(("f", i)) for i in range(8)),
        tracks=tracks, events=(_event("R0", 1, 3),),
        candidates=(
            QueryCandidateEvidence("T1", "patient", "SUPPORTED", .9, (2,)),
            QueryCandidateEvidence("T2", "patient", "REFUTED", .9, (2,)),
        ), public_ontology_sha256=stable_hash("ontology"),
        grounder_backend_sha256=stable_hash("backend"), provider_calls=3,
    )
    projected = adapt_query_grounding_v2(receipt, semantic, minimum_candidate_confidence=.8)
    assert [row.object for row in projected.events] == ["book"]
    assert projected.answer_read is False and projected.source_controller_read is False
    assert query_grounding_v2_from_dict(asdict(receipt)) == receipt


def test_adapter_preserves_anchor_events_while_gating_outer_query() -> None:
    semantic = parse_compact_semantic_target(
        "goal(class, single_reference(observations(time_window(before, opening a book), "
        "relation_description(frame, semantic_tuple(relations, beneath, objects)))))",
        task_id="q0", question_sha256=stable_hash("q0"), parser_sha256=stable_hash("parser"),
        parser_training_authority="target-dev",
    )
    by_surface = {row.surface: row.slot_id for row in semantic.slots}
    tracks = (_track("T0", "person"), _track("T1", "table"), _track("T2", "book"))
    query_event = TypedRoleEvent(
        "R0", "beneath", (("agent", "T0"), ("relation_object", "T1")),
        1, 1, (1,), .9, (by_surface["beneath"],),
    )
    anchor_event = TypedRoleEvent(
        "R1", "opening", (("agent", "T0"), ("patient", "T2")),
        3, 3, (3,), .8, (by_surface["opening a book"],),
    )
    receipt = QueryGroundingV2Receipt.create(
        task_id="q0", video_sha256=stable_hash("video"),
        semantic_slots_sha256=semantic.receipt_sha256, selected_frame_indices=tuple(range(8)),
        selected_frame_sha256s=tuple(stable_hash(("f", i)) for i in range(8)),
        tracks=tracks, events=(query_event, anchor_event),
        candidates=(QueryCandidateEvidence("T1", "relation_object", "SUPPORTED", .9, (1,)),),
        public_ontology_sha256=stable_hash("ontology"),
        grounder_backend_sha256=stable_hash("backend"), provider_calls=0,
    )
    projected = adapt_query_grounding_v2(receipt, semantic, minimum_candidate_confidence=.7)
    assert [(row.predicate, row.object) for row in projected.events] == [
        ("beneath", "table"), ("opening", "book"),
    ]


def test_prediction_only_sgdet_boxes_form_stable_tracks_and_typed_candidates() -> None:
    def obj(index, frame, label, score, box):
        return {"detection_index": index, "sampled_frame_index": frame,
                "label": label, "score": score, "bbox_xyxy": box}

    video = {
        "model_visible_frame_count": 3,
        "objects": [
            obj(0, 0, "person", .99, [0, 0, 20, 40]),
            obj(1, 0, "paper/notebook", .90, [30, 10, 45, 30]),
            obj(2, 0, "paper/notebook", .40, [31, 10, 46, 30]),  # NMS duplicate
            obj(3, 1, "person", .99, [1, 0, 21, 40]),
            obj(4, 1, "paper/notebook", .92, [31, 10, 46, 30]),
        ],
        "relations": [{
            "sampled_frame_index": 1, "original_frame_index": 10,
            "object_detection_index": 4, "object_label": "paper/notebook",
            "spatial_object_to_person": {"above": .8},
        }],
    }
    compiled = build_stable_tracks(video)
    paper = [row for row in compiled.tracks if row.canonical_label == "paper"]
    assert len(paper) == 1 and paper[0].evidence_frames == (0, 1)
    assert compiled.detection_to_track[2] == compiled.detection_to_track[1]
    candidates = relation_track_candidates(
        video, compiled, predicate="beneath", lower_frame=0, upper_frame=2,
    )
    assert candidates[0]["candidate_label"] == "paper"
    assert candidates[0]["track_id"] == paper[0].track_id


def test_rank_fusion_is_deterministic_and_answer_blind() -> None:
    primary = (
        {"candidate_label": "book", "track_id": "T1", "sampled_frame_index": 1, "score": .8},
        {"candidate_label": "table", "track_id": "T2", "sampled_frame_index": 2, "score": .7},
    )
    secondary = (
        {"candidate_label": "table", "track_id": "T2", "sampled_frame_index": 2, "score": .9},
        {"candidate_label": "book", "track_id": "T1", "sampled_frame_index": 1, "score": .4},
    )
    fused = reciprocal_rank_fusion(primary, secondary, primary_weight=.6)
    assert [row["candidate_label"] for row in fused] == ["book", "table"]
    assert fused[0]["sources"] == ["sgdet", "slowfast"]
