"""Outcome-blind temporal windows for video neural-symbolic interventions."""

from __future__ import annotations

from typing import Any, Mapping


def normalize_temporal_window(
    start: Any,
    end: Any,
    *,
    minimum_width: float,
    maximum_width: float,
    requires_full_context: bool = False,
) -> tuple[float, float]:
    """Validate and deterministically clamp a normalized temporal window.

    The model supplies only a rough question-conditioned interval.  Clamping is
    symmetric, outcome-blind, and preserves the proposed centre when possible.
    """

    minimum_width = float(minimum_width)
    maximum_width = float(maximum_width)
    if not 0.0 < minimum_width <= maximum_width <= 1.0:
        raise ValueError("temporal width bounds must satisfy 0 < min <= max <= 1")
    if requires_full_context:
        return 0.0, 1.0
    left, right = float(start), float(end)
    if not 0.0 <= left < right <= 1.0:
        raise ValueError("temporal window must satisfy 0 <= start < end <= 1")
    centre = (left + right) / 2.0
    width = min(max(right - left, minimum_width), maximum_width)
    left = max(0.0, min(centre - width / 2.0, 1.0 - width))
    return left, left + width


def parse_temporal_localization(
    payload: Mapping[str, Any], *, minimum_width: float, maximum_width: float,
) -> dict[str, Any]:
    """Parse the schema returned by the question-only temporal localizer."""

    raw = payload.get("window_fraction")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("window_fraction must be a normalized pair")
    full = payload.get("requires_full_context")
    if not isinstance(full, bool):
        raise ValueError("requires_full_context must be boolean")
    anchor = str(payload.get("anchor_description") or "").strip()
    if not anchor:
        raise ValueError("anchor_description must be nonempty")
    reliability = float(payload.get("sensor_reliability", -1.0))
    if not 0.5 <= reliability <= 1.0:
        raise ValueError("sensor_reliability must be in [0.5,1]")
    start, end = normalize_temporal_window(
        raw[0], raw[1], minimum_width=minimum_width,
        maximum_width=maximum_width, requires_full_context=full,
    )
    return {
        "window_fraction": [start, end],
        "requires_full_context": full,
        "anchor_description": anchor,
        "sensor_reliability": reliability,
    }


def absolute_temporal_window(
    clip_start_seconds: float,
    clip_end_seconds: float,
    window_fraction: tuple[float, float] | list[float],
) -> tuple[float, float]:
    """Map a normalized localization back to absolute video seconds."""

    clip_start = float(clip_start_seconds)
    clip_end = float(clip_end_seconds)
    if clip_end <= clip_start:
        raise ValueError("clip end must be after clip start")
    left, right = map(float, window_fraction)
    if not 0.0 <= left < right <= 1.0:
        raise ValueError("temporal window must be normalized")
    duration = clip_end - clip_start
    return clip_start + left * duration, clip_start + right * duration


__all__ = [
    "absolute_temporal_window", "normalize_temporal_window",
    "parse_temporal_localization",
]
