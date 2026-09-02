from __future__ import annotations

from dataclasses import replace

import pytest

from motif_transfer.agqa_layer_b_contracts import (
    AGQASemanticSlotReceipt, GroundedEvent, LayerBTaskStateReceipt,
    RawVideoEventGraphReceipt, SemanticSlotNode,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.video_transfer_measurement import (
    GroundingMode, GroundingToolBudget, SharedVideoGroundingReceipt,
    VideoTransferDecision, evaluate_layer_b_matched_transfer,
)


def semantic() -> AGQASemanticSlotReceipt:
    return AGQASemanticSlotReceipt.create(
        task_id="A0", question_sha256=stable_hash("question"), answer_kind="ENTITY",
        root_slot_id="S1", parser_sha256=stable_hash("semantic-parser-v1"),
        parser_training_authority="AGQA_TRAIN_DEVELOPMENT_QUESTIONS_TO_SHALLOW_SLOTS_ONLY",
        slots=(
            SemanticSlotNode("S0", "ACTION", "opening a refrigerator"),
            SemanticSlotNode("S1", "QUERY_GOAL", "object held before action", ("S0",),
                             (("temporal_relation", "before"),)),
        ),
    )


def event_graph(sem: AGQASemanticSlotReceipt) -> RawVideoEventGraphReceipt:
    return RawVideoEventGraphReceipt.create(
        task_id="A0", video_sha256=stable_hash("raw-video"),
        semantic_slots_sha256=sem.receipt_sha256,
        selected_frame_indices=(0, 10, 20, 30),
        selected_frame_sha256s=tuple(stable_hash(["frame", i]) for i in range(4)),
        events=(GroundedEvent("E0", "person", "open", "refrigerator", 1, 2, (1, 2), .9, ("S0",)),),
        grounder_backend_sha256=stable_hash("frozen-vlm-prompt-and-weights"),
        frame_budget=4, provider_calls=1,
    )


def shared_grounding(state: LayerBTaskStateReceipt) -> SharedVideoGroundingReceipt:
    state.validate()
    return SharedVideoGroundingReceipt.create(
        benchmark="agqa2", task_id="A0", split="development",
        mode=GroundingMode.MODEL_TOOL_EVENT_GRAPH,
        state={"layer_b_task_state_receipt_sha256": state.receipt_sha256},
        evidence_source_sha256=stable_hash("raw-video"),
        tool_backend_sha256=stable_hash("frozen-vlm-prompt-and-weights"),
        allowed_tools=("sample_frames", "vlm_event_graph"),
        tool_budget=GroundingToolBudget(2, 4, 1),
    )


def test_semantic_slots_and_pixel_event_graph_bind_without_program_or_answer() -> None:
    sem = semantic(); graph = event_graph(sem)
    state = LayerBTaskStateReceipt.create(sem, graph); state.validate()
    assert not sem.functional_program_read_at_runtime
    assert not sem.operator_sequence_emitted
    assert not graph.official_scene_graph_read
    assert graph.semantic_slots_sha256 == sem.receipt_sha256


def test_semantic_slots_reject_vm_operator_smuggling() -> None:
    with pytest.raises(ValueError, match="operator"):
        AGQASemanticSlotReceipt.create(
            task_id="A0", question_sha256=stable_hash("q"), answer_kind="BOOLEAN",
            root_slot_id="S0", parser_sha256=stable_hash("parser"),
            parser_training_authority="DEV_ONLY",
            slots=(SemanticSlotNode("S0", "QUERY_GOAL", "PROJECT relation"),),
        )


def test_raw_event_graph_fails_closed_on_oracle_scene_graph_flag() -> None:
    sem = semantic(); graph = event_graph(sem)
    with pytest.raises(ValueError, match="oracle/outcome"):
        replace(graph, official_scene_graph_read=True).validate()


def test_raw_event_graph_requires_valid_semantic_slot_bindings() -> None:
    sem = semantic()
    with pytest.raises(ValueError, match="at least one semantic slot"):
        event_graph(sem).events[0].__class__(
            "E0", "person", "open", "refrigerator", 1, 2, (1, 2), .9,
        ).validate(4)
    graph = replace(
        event_graph(sem),
        events=(GroundedEvent("E0", "person", "open", "refrigerator", 1, 2, (1, 2), .9, ("S99",)),),
    )
    graph = replace(graph, receipt_sha256=stable_hash({
        key: value for key, value in __import__("dataclasses").asdict(graph).items()
        if key != "receipt_sha256"
    }))
    with pytest.raises(ValueError, match="unknown semantic slots"):
        LayerBTaskStateReceipt.create(sem, graph)


def test_layer_b_evaluator_requires_five_arms_and_model_grounding() -> None:
    sem = semantic(); state = LayerBTaskStateReceipt.create(sem, event_graph(sem))
    grounding = shared_grounding(state)
    arms = {
        "neural_only": "table", "generic_scaffold": "table",
        "source_permuted": "table", "source_induced": "cup",
        "target_written_isomorphic": "cup",
    }
    decisions = [VideoTransferDecision.create(
        grounding=grounding, arm=arm, prediction=prediction,
        controller_sha256=stable_hash(["controller", arm]),
        source_program_sha256=(stable_hash("source") if arm in {"source_induced", "source_permuted"} else None),
        tool_calls=2, frames_observed=4, provider_calls=1,
    ) for arm, prediction in arms.items()]
    result = evaluate_layer_b_matched_transfer(
        groundings=(grounding,), decisions=decisions, gold_answers={"A0": "cup"},
    )
    assert result["raw_video_end_to_end_only"]
    assert result["summaries"][0]["arm_correct"]["target_written_isomorphic"] == 1
    with pytest.raises(ValueError, match="exactly the five"):
        evaluate_layer_b_matched_transfer(
            groundings=(grounding,), decisions=decisions[:-1], gold_answers={"A0": "cup"},
        )
