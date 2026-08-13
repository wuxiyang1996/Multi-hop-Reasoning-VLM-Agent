"""Fail-closed bridge to the repository's visual-reasoning wrapper.

The wrapper owns the target-native tool ontology and execution contract.  This
module deliberately does not copy its tool schemas.  It imports the configured
wrapper checkout, verifies the requested tools exist in its registries, executes
them, and returns compact receipts suitable for matched intervention forks.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import io
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

from PIL import Image


TIR_INTERVENTION_TOOLS = (
    "zoom_region",
    "read_text_region",
    "describe_region",
)
VIDEO_INTERVENTION_TOOLS = ("inspect_multimodal_window",)


@dataclass(frozen=True)
class WrapperRoutingReceipt:
    classes: tuple[str, ...]
    required_tools: tuple[str, ...]
    derivation_kinds: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "classes": list(self.classes),
            "required_tools": list(self.required_tools),
            "derivation_kinds": list(self.derivation_kinds),
        }


def _activate_wrapper(wrapper_root: str | Path) -> Path:
    root = Path(wrapper_root).resolve()
    expected = root / "visual_reasoning_wrapper" / "tools_visual.py"
    if not expected.is_file():
        raise FileNotFoundError(f"visual reasoning wrapper is missing: {expected}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def route_question(
    question: str,
    *,
    modality: str,
    wrapper_root: str | Path,
) -> WrapperRoutingReceipt:
    _activate_wrapper(wrapper_root)
    from visual_reasoning_wrapper.question_router import classify_question

    routing = classify_question(question, modality=modality)
    return WrapperRoutingReceipt(
        classes=tuple(map(str, routing.classes)),
        required_tools=tuple(map(str, routing.required_tools)),
        derivation_kinds=tuple(map(str, routing.derivation_kinds)),
    )


def build_tir_registry(
    image: Image.Image,
    *,
    wrapper_root: str | Path,
):
    _activate_wrapper(wrapper_root)
    from visual_reasoning_wrapper.tools_visual import build_visual_registry

    registry = build_visual_registry(
        image, prefer_gdino=True, include_reasoning=True,
    )
    missing = sorted(set(TIR_INTERVENTION_TOOLS) - set(registry.tool_names()))
    if missing:
        raise RuntimeError(f"wrapper TIR tool contract is missing: {missing}")
    return registry


def tir_tool_schemas(registry) -> list[dict[str, Any]]:
    allowed = set(TIR_INTERVENTION_TOOLS)
    return [
        definition for definition in registry.definitions()
        if definition.get("function", {}).get("name") in allowed
    ]


def _bounded_box(
    arguments: Mapping[str, Any], image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x = max(0, min(int(arguments.get("x", 0)), image_width - 1))
    y = max(0, min(int(arguments.get("y", 0)), image_height - 1))
    width = max(1, min(int(arguments.get("w", image_width)), image_width - x))
    height = max(1, min(int(arguments.get("h", image_height)), image_height - y))
    return x, y, width, height


def execute_tir_intervention(
    registry,
    image: Image.Image,
    *,
    tool: str,
    arguments: Mapping[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    if tool not in TIR_INTERVENTION_TOOLS:
        raise ValueError(f"unsupported wrapper TIR intervention: {tool}")
    x, y, width, height = _bounded_box(arguments, image.size)
    canonical = {"x": x, "y": y, "w": width, "h": height}
    if tool == "zoom_region":
        canonical["zoom"] = float(arguments.get("zoom", 2.0))
        canonical["reason"] = str(arguments.get("reason") or "")
    result = registry.dispatch(tool, canonical)
    if result.error:
        raise RuntimeError(f"wrapper {tool} failed: {result.error}")
    payload = dict(result.result or {})
    encoded = payload.pop("_reobserve_image_b64", None)
    if encoded:
        crop = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    else:
        crop = image.crop((x, y, x + width, y + height)).convert("RGB")
    return crop, {
        "tool": tool,
        "arguments": canonical,
        "result": payload,
        "crop_size": list(crop.size),
    }


def build_video_registry(
    frames: Sequence[Image.Image],
    *,
    duration_seconds: float,
    wrapper_root: str | Path,
    audio_analyzer: Callable[..., Mapping[str, Any] | str] | None = None,
):
    if not frames:
        raise ValueError("video wrapper needs at least one sampled frame")
    _activate_wrapper(wrapper_root)
    from visual_reasoning_wrapper.tools_video import build_video_registry

    proxy_fps = (len(frames) - 1) / max(float(duration_seconds), 1e-6)
    proxy_fps = max(proxy_fps, 1e-6)
    registry = build_video_registry(
        frames=list(frames), fps=proxy_fps, current_index=0,
        audio_analyzer=audio_analyzer,
    )
    missing = sorted(set(VIDEO_INTERVENTION_TOOLS) - set(registry.tool_names()))
    if missing:
        raise RuntimeError(f"wrapper video tool contract is missing: {missing}")
    return registry, proxy_fps


def video_tool_schemas(registry) -> list[dict[str, Any]]:
    allowed = set(VIDEO_INTERVENTION_TOOLS)
    return [
        definition for definition in registry.definitions()
        if definition.get("function", {}).get("name") in allowed
    ]


def execute_video_intervention(
    registry,
    frames: Sequence[Image.Image],
    *,
    tool: str,
    arguments: Mapping[str, Any],
) -> tuple[list[Image.Image], dict[str, Any]]:
    if tool not in VIDEO_INTERVENTION_TOOLS:
        raise ValueError(f"unsupported wrapper video intervention: {tool}")
    canonical = {
        "n": max(1, int(arguments.get("n", 8))),
        "start_sec": max(0.0, float(arguments.get("start_sec", 0.0))),
        "end_sec": max(0.0, float(arguments.get("end_sec", 0.0))),
    }
    if canonical["end_sec"] <= canonical["start_sec"]:
        raise ValueError("video intervention end_sec must exceed start_sec")
    result = registry.dispatch(tool, canonical)
    if result.error:
        raise RuntimeError(f"wrapper {tool} failed: {result.error}")
    payload = dict(result.result or {})
    visual = dict(payload.get("visual") or {})
    audio = dict(payload.get("audio") or {})
    if not audio.get("available"):
        raise RuntimeError(
            "wrapper multimodal intervention has no audio evidence: "
            + str(audio.get("error") or "unknown error")
        )
    indices = [int(row["index"]) for row in visual.get("sampled", ())]
    if not indices or any(index < 0 or index >= len(frames) for index in indices):
        raise RuntimeError("wrapper returned invalid sampled frame indices")
    return [frames[index].convert("RGB") for index in indices], {
        "tool": tool,
        "arguments": canonical,
        "result": payload,
        "proxy_frame_indices": indices,
    }


__all__ = [
    "TIR_INTERVENTION_TOOLS",
    "VIDEO_INTERVENTION_TOOLS",
    "WrapperRoutingReceipt",
    "build_tir_registry",
    "build_video_registry",
    "execute_tir_intervention",
    "execute_video_intervention",
    "route_question",
    "tir_tool_schemas",
    "video_tool_schemas",
]
