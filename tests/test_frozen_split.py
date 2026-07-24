from __future__ import annotations

from motif_transfer.frozen_split import freeze_one_shot_split, id_digest


def test_selection_depends_only_on_ids_and_namespace() -> None:
    ids = ["c", "a", "b"]
    result = freeze_one_shot_split(ids, ids, namespace="bench:v1")
    expected = min(ids, key=lambda value: (id_digest(value, namespace="bench:v1"), value))
    assert result["adaptation_id"] == expected
    assert expected not in result["test_ids"]
    assert result["content_or_outcome_used_for_selection"] is False


def test_distinct_official_pools_are_preserved() -> None:
    result = freeze_one_shot_split(["train-1", "train-2"], ["test-1", "test-2"], namespace="x")
    assert set(result["test_ids"]) == {"test-1", "test-2"}
