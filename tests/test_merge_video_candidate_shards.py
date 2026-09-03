import pytest

from scripts.merge_video_candidate_shards import merge_shards


def _row(sample_id):
    return {
        "sample_id": sample_id,
        "complete": True,
        "source_gate_sha256": "source",
        "collector_sha256": "collector",
        "config_sha256": "config",
    }


def test_merge_video_shards_restores_frozen_order():
    assert [row["sample_id"] for row in merge_shards(
        expected_ids=["a", "b"], shards=[[_row("b")], [_row("a")]],
    )] == ["a", "b"]


def test_merge_video_shards_rejects_duplicate_or_incomplete():
    with pytest.raises(ValueError):
        merge_shards(expected_ids=["a"], shards=[[_row("a")], [_row("a")]])
    row = _row("a")
    row["complete"] = False
    with pytest.raises(ValueError):
        merge_shards(expected_ids=["a"], shards=[[row]])
