from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


def stable_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(namespace: str, item_id: str) -> str:
    return hashlib.sha256(f"{namespace}\0{item_id}".encode("utf-8")).hexdigest()


def freeze_partition(
    item_ids: Sequence[str],
    *,
    namespace: str,
    role_counts: Mapping[str, int],
    excluded_ids: Sequence[str] = (),
) -> dict:
    if not namespace:
        raise ValueError("partition namespace must be non-empty")
    if not role_counts or any(not role or count < 0 for role, count in role_counts.items()):
        raise ValueError("role_counts must contain non-negative named roles")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("candidate item IDs must be unique")

    excluded = set(map(str, excluded_ids))
    available = [str(item_id) for item_id in item_ids if str(item_id) not in excluded]
    requested = sum(role_counts.values())
    if requested > len(available):
        raise ValueError(
            f"requested {requested} items from a pool containing {len(available)}"
        )
    ordered = sorted(available, key=lambda item_id: (_rank(namespace, item_id), item_id))
    roles: dict[str, list[str]] = {}
    offset = 0
    for role, count in role_counts.items():
        roles[role] = ordered[offset : offset + count]
        offset += count
    roles["reserve"] = ordered[offset:]
    partition = {
        "selection_rule": "ascending_sha256(namespace\\0item_id)",
        "namespace": namespace,
        "candidate_count_before_exclusion": len(item_ids),
        "excluded_ids": sorted(excluded),
        "available_count": len(available),
        "role_counts": dict(role_counts),
        "roles": roles,
    }
    partition["partition_sha256"] = stable_hash(partition)
    audit_partition(partition, candidate_ids=item_ids)
    return partition


def freeze_round_robin_partition(
    grouped_item_ids: Mapping[str, Sequence[str]],
    *,
    namespace: str,
    role_counts: Mapping[str, int],
    excluded_ids: Sequence[str] = (),
) -> dict:
    if not namespace or not grouped_item_ids:
        raise ValueError("round-robin partition needs a namespace and groups")
    flattened = [str(item_id) for ids in grouped_item_ids.values() for item_id in ids]
    if len(flattened) != len(set(flattened)):
        raise ValueError("grouped candidate item IDs must be globally unique")
    excluded = set(map(str, excluded_ids))
    ranked_groups = {
        str(group): sorted(
            (str(item_id) for item_id in ids if str(item_id) not in excluded),
            key=lambda item_id: (_rank(f"{namespace}:{group}", item_id), item_id),
        )
        for group, ids in grouped_item_ids.items()
    }
    ordered = []
    offset = 0
    while True:
        added = False
        for group in sorted(ranked_groups):
            items = ranked_groups[group]
            if offset < len(items):
                ordered.append(items[offset])
                added = True
        if not added:
            break
        offset += 1
    requested = sum(role_counts.values())
    if requested > len(ordered):
        raise ValueError(
            f"requested {requested} items from a pool containing {len(ordered)}"
        )
    roles: dict[str, list[str]] = {}
    offset = 0
    for role, count in role_counts.items():
        roles[str(role)] = ordered[offset : offset + int(count)]
        offset += int(count)
    roles["reserve"] = ordered[offset:]
    result = {
        "selection_rule": (
            "ascending_sha256(namespace:group\\0item_id)_within_group_then_group_round_robin"
        ),
        "namespace": namespace,
        "candidate_count_before_exclusion": len(flattened),
        "excluded_ids": sorted(excluded),
        "available_count": len(ordered),
        "groups": {group: len(items) for group, items in sorted(ranked_groups.items())},
        "role_counts": dict(role_counts),
        "roles": roles,
    }
    assigned = [item_id for ids in roles.values() for item_id in ids]
    if len(assigned) != len(set(assigned)) or set(assigned) & excluded:
        raise AssertionError("round-robin partition is not disjoint")
    result["partition_sha256"] = stable_hash(result)
    return result


def audit_partition(partition: Mapping[str, object], *, candidate_ids: Sequence[str]) -> None:
    if partition.get("selection_rule") != "ascending_sha256(namespace\\0item_id)":
        raise ValueError("unknown partition selection rule")
    namespace = str(partition.get("namespace") or "")
    role_counts = partition.get("role_counts")
    roles = partition.get("roles")
    excluded_ids = partition.get("excluded_ids")
    if not isinstance(role_counts, Mapping) or not isinstance(roles, Mapping):
        raise ValueError("partition roles are malformed")
    if not isinstance(excluded_ids, list):
        raise ValueError("partition excluded_ids are malformed")

    recomputed = freeze_partition_unchecked(
        candidate_ids,
        namespace=namespace,
        role_counts={str(role): int(count) for role, count in role_counts.items()},
        excluded_ids=[str(item_id) for item_id in excluded_ids],
    )
    expected_roles = recomputed["roles"]
    normalized_roles = {
        str(role): [str(item_id) for item_id in item_ids]
        for role, item_ids in roles.items()
        if isinstance(item_ids, list)
    }
    if normalized_roles != expected_roles:
        raise ValueError("partition roles do not match the frozen hash selection rule")

    flattened = [item_id for item_ids in normalized_roles.values() for item_id in item_ids]
    if len(flattened) != len(set(flattened)):
        raise ValueError("an item appears in multiple partition roles")
    if set(flattened) & set(map(str, excluded_ids)):
        raise ValueError("an excluded item appears in a partition role")

    unsigned = dict(partition)
    claimed_hash = unsigned.pop("partition_sha256", None)
    if claimed_hash is not None and claimed_hash != stable_hash(unsigned):
        raise ValueError("partition_sha256 mismatch")


def freeze_partition_unchecked(
    item_ids: Sequence[str],
    *,
    namespace: str,
    role_counts: Mapping[str, int],
    excluded_ids: Sequence[str],
) -> dict:
    excluded = set(map(str, excluded_ids))
    available = [str(item_id) for item_id in item_ids if str(item_id) not in excluded]
    ordered = sorted(available, key=lambda item_id: (_rank(namespace, item_id), item_id))
    roles: dict[str, list[str]] = {}
    offset = 0
    for role, count in role_counts.items():
        roles[str(role)] = ordered[offset : offset + int(count)]
        offset += int(count)
    roles["reserve"] = ordered[offset:]
    return {"roles": roles}
