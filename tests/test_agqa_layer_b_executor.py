from motif_transfer.agqa_layer_b_contracts import GroundedEvent, RawVideoEventGraphReceipt
from motif_transfer.agqa_layer_b_executor import (
    event_graph_to_pseudo_stsg, execute_layer_b_semantics,
    semantic_program_from_compact,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.agqa_semantic_slots import (
    parse_compact_semantic_target, serialize_compact_semantic_target,
)


def test_compact_semantics_compile_to_existing_dsl_without_official_program() -> None:
    compact = "alternatives(door,clothes,goal(class,single_reference(observations(video,relation_description(frame,semantic_tuple(relations,behind,objects))))))"
    program = semantic_program_from_compact(compact)
    assert program.startswith("Choose(door, clothes, Query(")
    assert "Filter(frame, [relations, behind, objects])" in program


def test_pixel_events_adapt_to_shared_typed_vm_graph() -> None:
    receipt = RawVideoEventGraphReceipt.create(
        task_id="A0", video_sha256=stable_hash("video"),
        semantic_slots_sha256=stable_hash("slots"),
        selected_frame_indices=(0, 1, 2, 3),
        selected_frame_sha256s=tuple(stable_hash(["frame", i]) for i in range(4)),
        events=(GroundedEvent(
            "E0", "person", "holding", "broom", 1, 2, (1, 2), .9, ("S1",),
        ),), grounder_backend_sha256=stable_hash("grounder"),
        frame_budget=4, provider_calls=0,
    )
    graph, labels, graph_sha = event_graph_to_pseudo_stsg(receipt)
    assert graph["a0"]["phrase"] == "holding broom"
    assert labels[graph["a0"]["verb_id"]] == "holding"
    assert len(graph_sha) == 64


def test_pixel_event_executes_with_the_existing_typed_vm() -> None:
    receipt = RawVideoEventGraphReceipt.create(
        task_id="A1", video_sha256=stable_hash("video-1"),
        semantic_slots_sha256=stable_hash("slots-1"),
        selected_frame_indices=(0, 1, 2, 3),
        selected_frame_sha256s=tuple(stable_hash(["frame-1", i]) for i in range(4)),
        events=(GroundedEvent(
            "E0", "person", "behind", "door", 1, 2, (1, 2), .9, ("S1",),
        ),), grounder_backend_sha256=stable_hash("grounder"),
        frame_budget=4, provider_calls=0,
    )
    compact = "goal(class,single_reference(observations(video,relation_description(frame,semantic_tuple(relations,behind,objects)))))"
    execution = execute_layer_b_semantics(
        compact_semantics=compact, grounding=receipt,
        authorized_operators=("FILTER_EQ", "UNIQUE", "PROJECT"),
        authorized_compositions=None,
    )
    assert execution.receipt.status == "COMMITTED"
    assert execution.receipt.prediction == "door"


def test_strict_source_abstention_and_generic_eager_policy_share_executor() -> None:
    receipt = RawVideoEventGraphReceipt.create(
        task_id="A2", video_sha256=stable_hash("video-2"),
        semantic_slots_sha256=stable_hash("slots-2"),
        selected_frame_indices=(0, 1, 2, 3),
        selected_frame_sha256s=tuple(stable_hash(["frame-2", i]) for i in range(4)),
        events=(
            GroundedEvent("E0", "person", "behind", "door", 1, 1, (1,), .8, ("S1",)),
            GroundedEvent("E1", "person", "behind", "clothes", 2, 2, (2,), .8, ("S1",)),
        ), grounder_backend_sha256=stable_hash("grounder"), frame_budget=4, provider_calls=0,
    )
    compact = "alternatives(door,clothes,goal(class,single_reference(observations(video,relation_description(frame,semantic_tuple(relations,behind,objects))))))"
    strict = execute_layer_b_semantics(
        compact_semantics=compact, grounding=receipt,
        authorized_operators=("FILTER_EQ", "UNIQUE", "PROJECT", "CHOOSE"),
        ambiguity_policy="STRICT",
    )
    eager = execute_layer_b_semantics(
        compact_semantics=compact, grounding=receipt,
        authorized_operators=("FILTER_EQ", "UNIQUE", "PROJECT", "CHOOSE"),
        ambiguity_policy="EAGER",
    )
    assert strict.receipt.status == "ABSTAINED"
    assert eager.receipt.status == "COMMITTED"
    assert eager.receipt.prediction in {"door", "clothes"}


def test_segmented_same_action_merges_but_relation_events_remain_segmented() -> None:
    receipt = RawVideoEventGraphReceipt.create(
        task_id="A3", video_sha256=stable_hash("video-3"), semantic_slots_sha256=stable_hash("slots-3"),
        selected_frame_indices=tuple(range(6)), selected_frame_sha256s=tuple(stable_hash(["f3", i]) for i in range(6)),
        events=(
            GroundedEvent("E0", "person", "smiling at", "something", 0, 2, (0, 2), .8, ("S1",)),
            GroundedEvent("E1", "person", "smiling at", "something", 4, 5, (4, 5), .9, ("S1",)),
        ), grounder_backend_sha256=stable_hash("g3"), frame_budget=6, provider_calls=0,
    )
    graph, _, _ = event_graph_to_pseudo_stsg(receipt)
    actions = [row for row in graph.values() if row.get("type") == "action"]
    relations = [row for row in graph.values() if row.get("type") == "relation"]
    assert len(actions) == 1 and actions[0]["all_f"] == [0, 1, 2, 4, 5]
    assert len(relations) == 5


def test_grounder_inflection_binds_to_exact_semantic_action_phrase() -> None:
    semantic = parse_compact_semantic_target(
        serialize_compact_semantic_target(
            "Compare([before, after], Exists(Localize(temporal tag, reaching for and grabbing a picture), Filter(frame, [relations])))"
        ),
        task_id="A4", question_sha256=stable_hash("q4"),
        parser_sha256=stable_hash("p4"),
        parser_training_authority="OPERATOR_FREE_TEST",
    )
    action_slot = next(
        slot.slot_id for slot in semantic.slots
        if slot.surface == "reaching for and grabbing a picture"
    )
    receipt = RawVideoEventGraphReceipt.create(
        task_id="A4", video_sha256=stable_hash("video-4"),
        semantic_slots_sha256=semantic.receipt_sha256,
        selected_frame_indices=tuple(range(4)),
        selected_frame_sha256s=tuple(stable_hash(["f4", i]) for i in range(4)),
        events=(GroundedEvent(
            "E0", "person", "reach for and grab", "picture", 1, 2, (1,), .95,
            (action_slot,),
        ),), grounder_backend_sha256=stable_hash("g4"), frame_budget=4,
        provider_calls=0,
    )
    graph, _, _ = event_graph_to_pseudo_stsg(receipt, semantic)
    action = next(row for row in graph.values() if row.get("type") == "action")
    assert action["phrase"] == "reaching for and grabbing a picture"


def test_relation_predicate_uses_inflected_bound_literal_without_specificity_leak() -> None:
    semantic = parse_compact_semantic_target(
        serialize_compact_semantic_target(
            "XOR(Exists(paper, Iterate(video, Filter(frame, [relation, holding, objects]))), "
            "Exists(paper, Iterate(Localize(while, holding a picture), "
            "Filter(frame, [relation, holding, objects]))))"
        ),
        task_id="A5", question_sha256=stable_hash("q5"),
        parser_sha256=stable_hash("p5"), parser_training_authority="OPERATOR_FREE_TEST",
    )
    holding_slot = next(slot.slot_id for slot in semantic.slots if slot.surface == "holding")
    picture_slot = next(slot.slot_id for slot in semantic.slots if slot.surface == "holding a picture")
    receipt = RawVideoEventGraphReceipt.create(
        task_id="A5", video_sha256=stable_hash("video-5"),
        semantic_slots_sha256=semantic.receipt_sha256,
        selected_frame_indices=tuple(range(3)),
        selected_frame_sha256s=tuple(stable_hash(["f5", i]) for i in range(3)),
        events=(GroundedEvent(
            "E0", "person", "hold", "blanket", 0, 2, (1,), .9,
            (holding_slot, picture_slot),
        ),), grounder_backend_sha256=stable_hash("g5"), frame_budget=3,
        provider_calls=0,
    )
    graph, labels, _ = event_graph_to_pseudo_stsg(receipt, semantic)
    relation = next(row for row in graph.values() if row.get("type") == "relation")
    assert labels[relation["class"]] == "holding"
    action = next(row for row in graph.values() if row.get("type") == "action")
    assert action["phrase"] == "hold blanket"
