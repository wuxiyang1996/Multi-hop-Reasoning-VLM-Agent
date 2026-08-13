from pathlib import Path

from PIL import Image

from motif_transfer.visual_wrapper_bridge import (
    build_tir_registry,
    build_video_registry,
    execute_tir_intervention,
    execute_video_intervention,
    route_question,
    tir_tool_schemas,
    video_tool_schemas,
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
