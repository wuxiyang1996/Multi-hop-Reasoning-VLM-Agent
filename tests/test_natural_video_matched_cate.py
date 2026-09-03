from __future__ import annotations

import pytest

from motif_transfer.natural_video_matched_cate import cross_video_binding_rotation


def test_cross_video_rotation_is_exact_and_cell_preserving() -> None:
    rows = []
    for cell in ("a", "b"):
        for video, count in (("v1", 2), ("v2", 2), ("v3", 1)):
            for index in range(count):
                rows.append({
                    "batch": cell,
                    "benchmark": "star",
                    "family": "Sequence",
                    "video_id": f"{cell}:{video}",
                    "sample_id": f"{cell}:{video}:{index}",
                })
    mapping = cross_video_binding_rotation(
        rows, cell_fields=("batch", "benchmark", "family"),
    )
    for index, target in enumerate(mapping):
        assert rows[index]["batch"] == rows[target]["batch"]
        assert rows[index]["family"] == rows[target]["family"]
        assert rows[index]["video_id"] != rows[target]["video_id"]


def test_cross_video_rotation_rejects_impossible_cell() -> None:
    rows = [
        {"batch": "a", "video_id": "v1", "sample_id": f"q{i}"}
        for i in range(3)
    ] + [{"batch": "a", "video_id": "v2", "sample_id": "q3"}]
    with pytest.raises(ValueError, match="impossible"):
        cross_video_binding_rotation(rows, cell_fields=("batch",))
