from pathlib import Path

from PIL import Image

from motif_transfer.visual_wrapper_bridge import (
    WrapperSharedVideoGroundingDispatcher,
    build_tir_registry,
    build_video_registry,
    build_video_transition_grounding_registry,
    execute_tir_intervention,
    execute_transition_grounding,
    execute_video_intervention,
    route_question,
    tir_tool_schemas,
    transition_grounding_tool_schemas,
    video_tool_schemas,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.video_transfer_measurement import (
    GroundingToolBudget,
    GroundingToolCall,
    acquire_shared_model_grounding,
)


WRAPPER = Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent")


def test_wrapper_tir_contract_executes_real_zoom_receipt():
    image = Image.new("RGB", (100, 80), "white")
    routing = route_question(
        "What proportion of the image is occupied by the red object?",
        modality="image", wrapper_root=WRAPPER,
    )
    assert "ratio" in routing.classes
    registry = build_tir_registry(image, wrapper_root=WRAPPER)
    names = {row["function"]["name"] for row in tir_tool_schemas(registry)}
    assert {"zoom_region", "read_text_region", "describe_region"} <= names
    crop, receipt = execute_tir_intervention(
        registry, image, tool="zoom_region",
        arguments={"x": 10, "y": 5, "w": 40, "h": 20, "zoom": 2},
    )
    assert crop.size == (80, 40)
    assert receipt["arguments"]["x"] == 10
    assert "_reobserve_image_b64" not in receipt["result"]


def test_wrapper_video_contract_samples_parameterized_window():
    frames = [Image.new("RGB", (32, 24), (index, 0, 0)) for index in range(12)]
    registry, proxy_fps = build_video_registry(
        frames,
        duration_seconds=11.0,
        wrapper_root=WRAPPER,
        audio_analyzer=lambda **window: {
            "description": "bell, then footsteps",
            "analyzed_window": window,
        },
    )
    assert proxy_fps == 1.0
    names = {row["function"]["name"] for row in video_tool_schemas(registry)}
    assert names == {"inspect_multimodal_window"}
    selected, receipt = execute_video_intervention(
        registry, frames, tool="inspect_multimodal_window",
        arguments={"n": 3, "start_sec": 2, "end_sec": 8},
    )
    assert len(selected) == 3
    assert receipt["proxy_frame_indices"][0] >= 2
    assert receipt["result"]["audio"]["available"] is True
    assert "footsteps" in receipt["result"]["audio"]["description"]


def test_wrapper_visual_only_video_contract_needs_no_audio():
    frames = [Image.new("RGB", (32, 24), (index, 0, 0)) for index in range(12)]
    registry, _ = build_video_registry(
        frames,
        duration_seconds=11.0,
        wrapper_root=WRAPPER,
        required_tools=("sample_frames",),
    )
    names = {
        row["function"]["name"]
        for row in video_tool_schemas(registry, allowed_tools=("sample_frames",))
    }
    assert names == {"sample_frames"}
    selected, receipt = execute_video_intervention(
        registry, frames, tool="sample_frames",
        arguments={"n": 3, "start_sec": 2, "end_sec": 8},
    )
    assert len(selected) == 3
    assert receipt["result"]["audio"]["not_applicable"] is True
    assert receipt["proxy_frame_indices"][0] >= 2


def test_wrapper_transition_grounding_executes_detect_and_compare_receipts():
    frames = []
    for index in range(8):
        color = (0, 0, 0) if index < 4 else (255, 255, 255)
        frames.append(Image.new("RGB", (32, 24), color))
    registry, proxy_fps = build_video_transition_grounding_registry(
        frames,
        duration_seconds=7.0,
        wrapper_root=WRAPPER,
    )
    assert proxy_fps == 1.0
    names = {
        row["function"]["name"]
        for row in transition_grounding_tool_schemas(registry)
    }
    assert names == {"detect_scene_changes", "compare_frames"}

    selected, receipt = execute_transition_grounding(
        registry,
        frames,
        pair_count=3,
        uniform_anchor_count=2,
        threshold=0.0,
    )
    assert len(selected) == 6
    assert receipt["protocol"] == "DETECT_CHANGE_THEN_COMPARE_BEFORE_AFTER_V1"
    assert 4 in receipt["selected_transition_indices"]
    assert len(receipt["comparisons"]) == 3
    assert receipt["tool_sequence"][0] == "detect_scene_changes"
    assert receipt["comparisons"][1]["comparison"]["changed_pixel_pct"] >= 0.0


def test_real_wrapper_produces_shared_grounding_for_both_video_benchmarks():
    frames = [
        Image.new("RGB", (32, 24), "black" if index < 6 else "white")
        for index in range(12)
    ]
    dispatcher = WrapperSharedVideoGroundingDispatcher(
        frames, duration_seconds=11.0, wrapper_root=WRAPPER,
    )
    plan = (
        GroundingToolCall(
            "sample_frames", {"n": 4, "start_sec": 0, "end_sec": 11},
            "coarse chronological evidence",
        ),
        GroundingToolCall(
            "detect_transitions",
            {"pair_count": 2, "uniform_anchor_count": 1, "threshold": 0.0},
            "candidate event boundaries",
        ),
    )
    for benchmark in ("clevrer", "agqa2"):
        acquisition = acquire_shared_model_grounding(
            benchmark=benchmark, task_id=f"{benchmark}-wrapper-smoke",
            split="development", public_state={"question_sha256": stable_hash("q")},
            evidence_source_sha256=stable_hash([benchmark, "synthetic-video"]),
            tool_backend_sha256=stable_hash("visual-wrapper-shared-v1"),
            tool_budget=GroundingToolBudget(2, 8, 0), plan=plan,
            dispatch=dispatcher,
        )
        assert acquisition.tool_calls == 2
        assert acquisition.frames_observed == 8
        assert acquisition.answer_or_program_read is False
