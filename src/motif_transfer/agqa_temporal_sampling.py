"""Coordinate-safe temporal view helpers for AGQA raw-video grounders.

Model tensors index into a sampled proxy clip, whereas persisted grounding
receipts must use coordinates from the original video.  Keeping this mapping
in a small dependency-free module makes that boundary independently testable.
"""

from __future__ import annotations

from collections.abc import Sequence


def native_index_views(
    sampled_native_indices: Sequence[int],
    proxy_position_views: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Map sampled-clip positions to original-video frame indices.

    Repeated native indices are permitted for videos shorter than the requested
    proxy sample count.  The sampled sequence must nevertheless be monotonic,
    and every proxy position must be in range.
    """

    native = tuple(int(value) for value in sampled_native_indices)
    if not native:
        raise ValueError("sampled native frame indices must be non-empty")
    if any(value < 0 for value in native):
        raise ValueError("native frame indices must be non-negative")
    if tuple(sorted(native)) != native:
        raise ValueError("sampled native frame indices must be chronological")

    mapped: list[tuple[int, ...]] = []
    for raw_view in proxy_position_views:
        view = tuple(int(value) for value in raw_view)
        if not view:
            raise ValueError("temporal views must be non-empty")
        if any(position < 0 or position >= len(native) for position in view):
            raise ValueError("temporal view references an unknown proxy position")
        mapped.append(tuple(native[position] for position in view))
    if not mapped:
        raise ValueError("at least one temporal view is required")
    return tuple(mapped)
