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
VIDEO_VISUAL_INTERVENTION_TOOLS = ("sample_frames",)
VIDEO_MULTIMODAL_INTERVENTION_TOOLS = ("inspect_multimodal_window",)
VIDEO_INTERVENTION_TOOLS = (
    *VIDEO_VISUAL_INTERVENTION_TOOLS,
    *VIDEO_MULTIMODAL_INTERVENTION_TOOLS,
)
VIDEO_TRANSITION_GROUNDING_TOOLS = (
    "detect_scene_changes",
    "compare_frames",
)
SHARED_VIDEO_GROUNDING_TOOLS = ("sample_frames", "detect_transitions")


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
    required_tools: Sequence[str] = VIDEO_MULTIMODAL_INTERVENTION_TOOLS,
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
    requested = tuple(map(str, required_tools))
    unsupported = sorted(set(requested) - set(VIDEO_INTERVENTION_TOOLS))
    if unsupported:
        raise ValueError(f"unsupported video intervention tools: {unsupported}")
    missing = sorted(set(requested) - set(registry.tool_names()))
    if missing:
        raise RuntimeError(f"wrapper video tool contract is missing: {missing}")
    return registry, proxy_fps


def build_video_transition_grounding_registry(
    frames: Sequence[Image.Image],
    *,
    duration_seconds: float,
    wrapper_root: str | Path,
):
    """Bind the wrapper's CPU-safe temporal grounding tools.

    This registry is intentionally separate from the intervention registry:
    ``detect_scene_changes`` and ``compare_frames`` consume frame indices, not
    ``(start_sec, end_sec, n)`` windows.  Keeping the two contracts separate
    prevents an action from being silently canonicalised into the wrong tool
    signature.
    """

    if len(frames) < 2:
        raise ValueError("transition grounding needs at least two sampled frames")
    _activate_wrapper(wrapper_root)
    from visual_reasoning_wrapper.tools_video import build_video_registry

    proxy_fps = (len(frames) - 1) / max(float(duration_seconds), 1e-6)
    proxy_fps = max(proxy_fps, 1e-6)
    registry = build_video_registry(
        frames=list(frames), fps=proxy_fps, current_index=0,
    )
    missing = sorted(
        set(VIDEO_TRANSITION_GROUNDING_TOOLS) - set(registry.tool_names())
    )
    if missing:
        raise RuntimeError(
            f"wrapper transition-grounding contract is missing: {missing}"
        )
    return registry, proxy_fps


def transition_grounding_tool_schemas(registry) -> list[dict[str, Any]]:
    allowed = set(VIDEO_TRANSITION_GROUNDING_TOOLS)
    return [
        definition for definition in registry.definitions()
        if definition.get("function", {}).get("name") in allowed
    ]


def execute_transition_grounding(
    registry,
    frames: Sequence[Image.Image],
    *,
    pair_count: int,
    uniform_anchor_count: int,
    threshold: float = 0.0,
) -> tuple[list[Image.Image], dict[str, Any]]:
    """Select matched before/after evidence pairs with real wrapper tools.

    The detector ranks adjacent proxy-frame transitions by pixel change.  A
    predeclared number of uniformly spaced anchors is always retained so a
    low-motion but semantically important event is not excluded merely because
    it has a small pixel delta.  Selection is independent of question labels and
    target outcomes.  Every retained edge is then executed through
    ``compare_frames`` and recorded in the returned grounding receipt.
    """

    total = len(frames)
    if total < 2:
        raise ValueError("transition grounding needs at least two frames")
    pair_count = int(pair_count)
    uniform_anchor_count = int(uniform_anchor_count)
    if not 1 <= pair_count <= total - 1:
        raise ValueError("pair_count must be in [1, len(frames)-1]")
    if not 0 <= uniform_anchor_count <= pair_count:
        raise ValueError("uniform_anchor_count must be in [0, pair_count]")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    detected = registry.dispatch(
        "detect_scene_changes",
        {"start_idx": 0, "end_idx": total - 1, "threshold": threshold},
    )
    if detected.error:
        raise RuntimeError(
            f"wrapper detect_scene_changes failed: {detected.error}"
        )
    detection_payload = dict(detected.result or {})
    scores = {
        int(row["frame_index"]): float(row["diff_score"])
        for row in detection_payload.get("changes", ())
        if 1 <= int(row.get("frame_index", -1)) < total
    }

    anchors: list[int] = []
    if uniform_anchor_count == 1:
        anchors = [max(1, round((total - 1) / 2))]
    elif uniform_anchor_count > 1:
        anchors = [
            1 + round(slot * (total - 2) / (uniform_anchor_count - 1))
            for slot in range(uniform_anchor_count)
        ]
    selected = list(dict.fromkeys(anchors))
    ranked = sorted(
        range(1, total),
        key=lambda index: (-scores.get(index, -1.0), index),
    )
    for index in ranked:
        if index not in selected:
            selected.append(index)
        if len(selected) == pair_count:
            break
    selected = sorted(selected[:pair_count])
    if len(selected) != pair_count:
        raise RuntimeError("transition selection did not produce pair_count edges")

    comparisons = []
    grounded_frames: list[Image.Image] = []
    for pair_id, after_index in enumerate(selected):
        before_index = after_index - 1
        compared = registry.dispatch(
            "compare_frames",
            {"frame_a": before_index, "frame_b": after_index},
        )
        if compared.error:
            raise RuntimeError(
                f"wrapper compare_frames failed for edge {before_index}->{after_index}: "
                f"{compared.error}"
            )
        payload = dict(compared.result or {})
        if "mean_difference" not in payload or "changed_pixel_pct" not in payload:
            raise RuntimeError("wrapper compare_frames returned an incomplete receipt")
        comparisons.append({
            "pair_id": f"T{pair_id}",
            "before_index": before_index,
            "after_index": after_index,
            "detector_diff_score": scores.get(after_index),
            "comparison": payload,
        })
        grounded_frames.extend((
            frames[before_index].convert("RGB"),
            frames[after_index].convert("RGB"),
        ))

    return grounded_frames, {
        "protocol": "DETECT_CHANGE_THEN_COMPARE_BEFORE_AFTER_V1",
        "tool_sequence": [
            "detect_scene_changes",
            *("compare_frames" for _ in comparisons),
        ],
        "threshold": threshold,
        "pair_count": pair_count,
        "uniform_anchor_count": uniform_anchor_count,
        "selected_transition_indices": selected,
        "detector_receipt": detection_payload,
        "comparisons": comparisons,
    }


def video_tool_schemas(
    registry,
    *,
    allowed_tools: Sequence[str] = VIDEO_MULTIMODAL_INTERVENTION_TOOLS,
) -> list[dict[str, Any]]:
    allowed = set(map(str, allowed_tools))
    unsupported = sorted(allowed - set(VIDEO_INTERVENTION_TOOLS))
    if unsupported:
        raise ValueError(f"unsupported video intervention tools: {unsupported}")
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
    raw_payload = dict(result.result or {})
    if tool in VIDEO_VISUAL_INTERVENTION_TOOLS:
        visual = raw_payload
        audio = {
            "available": False,
            "not_applicable": True,
            "reason": "visual_only_benchmark_intervention",
        }
        payload = {
            "visual": visual,
            "audio": audio,
            "aligned_window": {
                "start_sec": canonical["start_sec"],
                "end_sec": canonical["end_sec"],
            },
        }
    else:
        payload = raw_payload
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


class WrapperSharedVideoGroundingDispatcher:
    """Evidence-only adapter used before matched video controller arms run.

    The adapter deliberately exposes no answer model.  It converts the
    repository wrapper's frame sampler and transition comparator into compact
    JSON receipts consumable by ``acquire_shared_model_grounding``.
    """

    def __init__(
        self, frames: Sequence[Image.Image], *, duration_seconds: float,
        wrapper_root: str | Path,
    ):
        if len(frames) < 2:
            raise ValueError("shared video grounding needs at least two frames")
        self.frames = tuple(frame.convert("RGB") for frame in frames)
        self.duration_seconds = float(duration_seconds)
        self.visual_registry, _ = build_video_registry(
            self.frames, duration_seconds=self.duration_seconds,
            wrapper_root=wrapper_root, required_tools=("sample_frames",),
        )
        self.transition_registry, _ = build_video_transition_grounding_registry(
            self.frames, duration_seconds=self.duration_seconds,
            wrapper_root=wrapper_root,
        )

    def __call__(self, tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if tool == "sample_frames":
            _, receipt = execute_video_intervention(
                self.visual_registry, self.frames, tool="sample_frames",
                arguments=arguments,
            )
            indices = list(receipt["proxy_frame_indices"])
            return {
                "tool": tool,
                "frame_indices": indices,
                "window": dict(receipt["arguments"]),
                "wrapper_receipt": receipt["result"],
                "_usage": {"frames_observed": len(indices), "provider_calls": 0},
            }
        if tool == "detect_transitions":
            selected, receipt = execute_transition_grounding(
                self.transition_registry, self.frames,
                pair_count=int(arguments.get("pair_count", 4)),
                uniform_anchor_count=int(arguments.get("uniform_anchor_count", 2)),
                threshold=float(arguments.get("threshold", 0.0)),
            )
            return {
                "tool": tool,
                "transition_indices": list(receipt["selected_transition_indices"]),
                "transition_receipt": receipt,
                "_usage": {
                    "frames_observed": len(selected), "provider_calls": 0,
                },
            }
        raise ValueError(f"unsupported shared video grounding tool: {tool}")


__all__ = [
    "TIR_INTERVENTION_TOOLS",
    "SHARED_VIDEO_GROUNDING_TOOLS",
    "VIDEO_INTERVENTION_TOOLS",
    "VIDEO_MULTIMODAL_INTERVENTION_TOOLS",
    "VIDEO_TRANSITION_GROUNDING_TOOLS",
    "VIDEO_VISUAL_INTERVENTION_TOOLS",
    "WrapperRoutingReceipt",
    "WrapperSharedVideoGroundingDispatcher",
    "build_tir_registry",
    "build_video_registry",
    "build_video_transition_grounding_registry",
    "execute_tir_intervention",
    "execute_video_intervention",
    "execute_transition_grounding",
    "route_question",
    "tir_tool_schemas",
    "transition_grounding_tool_schemas",
    "video_tool_schemas",
]
