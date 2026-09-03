from motif_transfer.agqa_layer_b_contracts import GroundedEvent, RawVideoEventGraphReceipt
from motif_transfer.agqa_layer_b_executor_v2 import event_graph_to_pseudo_stsg_v2
from motif_transfer.agqa_layer_b_executor_v3 import (
    _typed_full_action_slots,
    event_graph_to_pseudo_stsg_v3,
)
from motif_transfer.agqa_semantic_slots import parse_compact_semantic_target
from motif_transfer.contracts import stable_hash


def _duration_semantic():
    return parse_compact_semantic_target(
        "duration_extremum(max, semantic_tuple(relation_description(video, "
        "semantic_tuple(actions, holding a bag)), relation_description(video, "
        "semantic_tuple(actions, holding a cup of something))), "
        "relative_duration(goal(end, action), goal(start, action)))",
        task_id="D0", question_sha256=stable_hash("duration-question"),
        parser_sha256=stable_hash("duration-parser"),
        parser_training_authority="OPERATOR_FREE_TEST",
    )


def test_description_action_slots_are_discovered_from_typed_tree():
    semantic = _duration_semantic()
    by_surface = {slot.surface: slot.slot_id for slot in semantic.slots}
    assert _typed_full_action_slots(semantic) == frozenset({
        by_surface["holding a bag"], by_surface["holding a cup of something"],
    })


def test_v3_disambiguates_same_verb_using_grounded_object():
    semantic = _duration_semantic()
    by_surface = {slot.surface: slot.slot_id for slot in semantic.slots}
    smeared = (
        by_surface["holding a bag"], by_surface["holding a cup of something"],
    )
    receipt = RawVideoEventGraphReceipt.create(
        task_id="D0", video_sha256=stable_hash("duration-video"),
        semantic_slots_sha256=semantic.receipt_sha256,
        selected_frame_indices=tuple(range(8)),
        selected_frame_sha256s=tuple(stable_hash(["duration-frame", i]) for i in range(8)),
        events=(
            GroundedEvent("E0", "person", "hold", "bag", 0, 2, (0, 2), .9, smeared),
            GroundedEvent("E1", "person", "hold", "cup of something", 3, 7, (3, 7), .9, smeared),
        ),
        grounder_backend_sha256=stable_hash("duration-grounder"),
        frame_budget=8, provider_calls=0,
    )
    old, _, _ = event_graph_to_pseudo_stsg_v2(receipt, semantic)
    new, _, _ = event_graph_to_pseudo_stsg_v3(receipt, semantic)
    old_actions = [row["phrase"] for row in old.values() if row.get("type") == "action"]
    new_actions = sorted(row["phrase"] for row in new.values() if row.get("type") == "action")
    assert old_actions == ["hold"]
    assert new_actions == ["holding a bag", "holding a cup of something"]
