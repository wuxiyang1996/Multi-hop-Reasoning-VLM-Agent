from dataclasses import replace

import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.video_transfer_measurement import (
    GroundingMode,
    GroundingToolCall,
    GroundingToolBudget,
    SharedVideoGroundingReceipt,
    VideoTransferClaim,
    VideoTransferDecision,
    acquire_shared_model_grounding,
    evaluate_matched_transfer,
)


ZERO = GroundingToolBudget(0, 0, 0)
TOOLS = GroundingToolBudget(6, 24, 4)


def _oracle(benchmark: str, task_id: str):
    return SharedVideoGroundingReceipt.create(
        benchmark=benchmark,
        task_id=task_id,
        split="qualification",
        mode=GroundingMode.ORACLE_EVENT_GRAPH,
        state={
            "entities": ["person", "object"],
            "events": [{"predicate": "contact", "start": 2, "end": 4}],
        },
        evidence_source_sha256=stable_hash(f"{benchmark}-official-scene-graphs"),
        tool_budget=ZERO,
        official_scene_graph_read=True,
    )


def _model(benchmark: str, task_id: str):
    return SharedVideoGroundingReceipt.create(
        benchmark=benchmark,
        task_id=task_id,
        split="qualification",
        mode=GroundingMode.MODEL_TOOL_EVENT_GRAPH,
        state={"entities": ["person"], "events": [{"predicate": "unknown"}]},
        evidence_source_sha256=stable_hash(f"{benchmark}-video"),
        tool_backend_sha256=stable_hash("wrapper-plus-vlm"),
        allowed_tools=("sample_frames", "detect_scene_changes", "compare_frames"),
        tool_budget=TOOLS,
    )


def _decisions(grounding, predictions):
    source_program = stable_hash("source-program")
    rows = []
    for arm, prediction in predictions.items():
        rows.append(VideoTransferDecision.create(
            grounding=grounding,
            arm=arm,
            prediction=prediction,
            controller_sha256=stable_hash([grounding.benchmark, arm]),
            source_program_sha256=(
                source_program if arm in {"source_induced", "source_permuted"}
                else None
            ),
            tool_calls=0 if grounding.mode == GroundingMode.ORACLE_EVENT_GRAPH else 3,
            frames_observed=0 if grounding.mode == GroundingMode.ORACLE_EVENT_GRAPH else 12,
            provider_calls=0 if grounding.mode == GroundingMode.ORACLE_EVENT_GRAPH else 2,
        ))
    return rows


def test_oracle_track_is_conditional_and_answer_program_blind_for_both_benchmarks():
    for benchmark in ("clevrer", "agqa2"):
        receipt = _oracle(benchmark, f"{benchmark}-0")
        receipt.validate()
        assert receipt.claim == VideoTransferClaim.CONDITIONAL_SKILL_TRANSFER
        assert receipt.official_scene_graph_read is True
        assert receipt.functional_program_read is False
        assert receipt.gold_answer_read is False


def test_oracle_state_rejects_answer_or_functional_program_leakage():
    for leaked in ({"answer": "yes"}, {"nested": {"functional_program": []}}):
        with pytest.raises(ValueError, match="answer/program"):
            SharedVideoGroundingReceipt.create(
                benchmark="agqa2", task_id="bad", split="test",
                mode=GroundingMode.ORACLE_EVENT_GRAPH,
                state=leaked, evidence_source_sha256=stable_hash("official"),
                tool_budget=ZERO, official_scene_graph_read=True,
            )


def test_model_tool_track_is_end_to_end_and_cannot_read_scene_graph():
    receipt = _model("agqa2", "A0")
    assert receipt.claim == VideoTransferClaim.END_TO_END_VIDEO_TRANSFER
    with pytest.raises(ValueError, match="cannot read an official scene graph"):
        SharedVideoGroundingReceipt.create(
            benchmark="agqa2", task_id="bad", split="test",
            mode=GroundingMode.MODEL_TOOL_EVENT_GRAPH,
            state={"events": []}, evidence_source_sha256=stable_hash("video"),
            tool_backend_sha256=stable_hash("backend"),
            allowed_tools=("sample_frames",), tool_budget=TOOLS,
            official_scene_graph_read=True,
        )


def test_tool_budget_and_outcome_boundary_fail_closed():
    grounding = _model("clevrer", "C0")
    with pytest.raises(ValueError, match="frame budget"):
        VideoTransferDecision.create(
            grounding=grounding, arm="neural_only", prediction="01",
            controller_sha256=stable_hash("controller"), frames_observed=25,
        )
    with pytest.raises(ValueError, match="before outcome"):
        VideoTransferDecision.create(
            grounding=grounding, arm="neural_only", prediction="01",
            controller_sha256=stable_hash("controller"), gold_answer_read=True,
        )


def test_matched_evaluator_keeps_oracle_and_model_tracks_separate():
    clevrer = _oracle("clevrer", "C0")
    agqa = _model("agqa2", "A0")
    decisions = []
    decisions += _decisions(clevrer, {
        "neural_only": "00", "source_induced": "11",
        "source_permuted": "00", "generic_scaffold": "00",
    })
    decisions += _decisions(agqa, {
        "neural_only": "table", "source_induced": "cup",
        "source_permuted": "table", "generic_scaffold": "table",
    })
    result = evaluate_matched_transfer(
        groundings=(clevrer, agqa), decisions=decisions,
        gold_answers={"C0": "11", "A0": "cup"},
    )
    assert result["grounding_modes_combined"] is False
    assert len(result["summaries"]) == 2
    assert all(row["source_vs_neural_wins"] == 1 for row in result["summaries"])
    assert {row["claim"] for row in result["summaries"]} == {
        "CONDITIONAL_SKILL_TRANSFER", "END_TO_END_VIDEO_TRANSFER",
    }


def test_evaluator_rejects_one_arm_receiving_different_grounding():
    first = _oracle("agqa2", "A0")
    second = _oracle("agqa2", "A1")
    rows = _decisions(first, {
        "neural_only": "x", "source_induced": "y",
        "source_permuted": "x", "generic_scaffold": "x",
    })
    rows[1] = replace(rows[1], grounding_receipt_sha256=second.receipt_sha256)
    with pytest.raises(ValueError, match="identical grounding"):
        evaluate_matched_transfer(
            groundings=(first,), decisions=rows, gold_answers={"A0": "y"},
        )


@pytest.mark.parametrize("benchmark", ["clevrer", "agqa2"])
def test_shared_grounding_tools_run_once_before_matched_arms(benchmark):
    seen = []

    def dispatch(tool, arguments):
        seen.append((tool, arguments))
        return {
            "frame_indices": [0, 4, 8],
            "transition_candidates": [{"before": 4, "after": 8}],
            "_usage": {"frames_observed": 3, "provider_calls": 0},
        }

    acquisition = acquire_shared_model_grounding(
        benchmark=benchmark, task_id=f"{benchmark}-tool-0",
        split="development", public_state={"question_sha256": stable_hash("q")},
        evidence_source_sha256=stable_hash("video"),
        tool_backend_sha256=stable_hash("wrapper"), tool_budget=TOOLS,
        plan=(GroundingToolCall(
            "detect_transitions", {"pair_count": 1}, "find event boundaries",
        ),), dispatch=dispatch,
    )
    assert len(seen) == 1
    assert acquisition.tool_calls == 1
    assert acquisition.frames_observed == 3
    assert acquisition.source_controller_read is False
    # All arms bind the same post-tool target state, not separate VLM calls.
    decisions = _decisions(acquisition.grounding, {
        "neural_only": "x", "source_induced": "y",
        "source_permuted": "x", "generic_scaffold": "x",
    })
    assert len({row.grounding_receipt_sha256 for row in decisions}) == 1


def test_shared_grounding_tool_rejects_answer_leakage():
    with pytest.raises(ValueError, match="answer/program"):
        acquire_shared_model_grounding(
            benchmark="agqa2", task_id="bad-tool", split="development",
            public_state={"question_sha256": stable_hash("q")},
            evidence_source_sha256=stable_hash("video"),
            tool_backend_sha256=stable_hash("wrapper"), tool_budget=TOOLS,
            plan=(GroundingToolCall("bad", {}, "bad probe"),),
            dispatch=lambda *_: {"answer": "yes"},
        )
