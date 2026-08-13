from __future__ import annotations

import copy

import pytest

from motif_transfer.real_game_multitarget_manifest import (
    audit_partition,
    freeze_partition,
    freeze_round_robin_partition,
)


def test_freeze_partition_is_deterministic_and_disjoint() -> None:
    ids = [f"item-{index}" for index in range(20)]
    first = freeze_partition(
        ids,
        namespace="unit-test",
        role_counts={"adaptation": 3, "qualification": 4, "held_out": 5},
        excluded_ids=("item-2",),
    )
    second = freeze_partition(
        list(reversed(ids)),
        namespace="unit-test",
        role_counts={"adaptation": 3, "qualification": 4, "held_out": 5},
        excluded_ids=("item-2",),
    )
    assert first == second
    roles = first["roles"]
    flattened = [item_id for values in roles.values() for item_id in values]
    assert len(flattened) == len(set(flattened)) == 19
    assert "item-2" not in flattened
    assert len(roles["adaptation"]) == 3
    assert len(roles["qualification"]) == 4
    assert len(roles["held_out"]) == 5


def test_audit_rejects_role_tampering() -> None:
    ids = [f"item-{index}" for index in range(10)]
    partition = freeze_partition(
        ids,
        namespace="unit-test-tamper",
        role_counts={"adaptation": 2, "held_out": 4},
    )
    tampered = copy.deepcopy(partition)
    tampered["roles"]["adaptation"][0] = tampered["roles"]["held_out"][0]
    with pytest.raises(ValueError, match="selection rule"):
        audit_partition(tampered, candidate_ids=ids)


def test_freeze_partition_rejects_duplicate_candidates() -> None:
    with pytest.raises(ValueError, match="unique"):
        freeze_partition(
            ["same", "same"],
            namespace="unit-test-duplicates",
            role_counts={"held_out": 1},
        )


def test_round_robin_partition_covers_groups_and_is_order_invariant() -> None:
    groups = {
        "b": [f"b-{index}" for index in range(6)],
        "a": [f"a-{index}" for index in range(6)],
        "c": [f"c-{index}" for index in range(6)],
    }
    first = freeze_round_robin_partition(
        groups,
        namespace="unit-test-round-robin",
        role_counts={"adaptation": 4, "qualification": 4, "held_out": 6},
        excluded_ids=("a-0",),
    )
    second = freeze_round_robin_partition(
        {group: list(reversed(ids)) for group, ids in reversed(groups.items())},
        namespace="unit-test-round-robin",
        role_counts={"adaptation": 4, "qualification": 4, "held_out": 6},
        excluded_ids=("a-0",),
    )
    assert first == second
    assert {item_id[0] for item_id in first["roles"]["adaptation"]} == {"a", "b", "c"}
    assert {item_id[0] for item_id in first["roles"]["qualification"]} == {"a", "b", "c"}
