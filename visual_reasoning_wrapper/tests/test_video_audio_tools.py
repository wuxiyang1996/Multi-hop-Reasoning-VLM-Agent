from PIL import Image

from visual_reasoning_wrapper.tools_video import build_video_registry


def test_aligned_multimodal_window_uses_caller_audio_analyzer():
    frames = [Image.new("RGB", (24, 16), (index, 0, 0)) for index in range(12)]
    calls = []

    def analyze(**window):
        calls.append(window)
        return {"description": "bell followed by footsteps"}

    registry = build_video_registry(
        frames=frames, fps=1.0, audio_analyzer=analyze,
    )
    result = registry.dispatch("inspect_multimodal_window", {
        "n": 3, "start_sec": 2.0, "end_sec": 8.0,
    })
    assert result.error is None
    assert result.result["visual"]["count"] == 3
    assert result.result["audio"]["available"] is True
    assert result.result["audio"]["description"] == "bell followed by footsteps"
    assert calls == [{"start_sec": 2.0, "end_sec": 8.0}]


def test_audio_window_fails_closed_without_analyzer():
    frames = [Image.new("RGB", (24, 16), "black") for _ in range(4)]
    registry = build_video_registry(frames=frames, fps=1.0)
    result = registry.dispatch("inspect_audio_window", {
        "start_sec": 0.0, "end_sec": 2.0,
    })
    assert result.error is None
    assert result.result["available"] is False
    assert "No audio analyzer" in result.result["error"]
