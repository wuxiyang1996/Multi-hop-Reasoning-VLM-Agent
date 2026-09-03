#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
NAMESPACE = "webshop-v8-clean-confirmation"
DIAGNOSTIC_TASKS = (
    "webshop.1",
    "webshop.6",
    "webshop.13",
    "webshop.24",
    "webshop.28",
    "webshop.31",
    "webshop.34",
    "webshop.49",
)


def _task_index(task_id: str) -> int:
    prefix, value = task_id.split(".", 1)
    if prefix != "webshop":
        raise ValueError(f"not a WebShop task: {task_id}")
    return int(value)


def _rank(group_id: str) -> str:
    return hashlib.sha256(f"{NAMESPACE}\0{group_id}".encode()).hexdigest()


def build_grouped_manifest(frozen: dict) -> dict:
    consumed = {
        task_id
        for task_ids in frozen["roles"].values()
        for task_id in task_ids
    }
    if consumed != set(frozen["task_ids"]):
        raise ValueError("frozen goal roles do not cover exactly the consumed tasks")

    tasks_by_group: dict[str, list[str]] = defaultdict(list)
    for task_id in frozen["task_ids"]:
        goal = frozen["goals"][task_id]
        group_id = str(goal["asin"])
        tasks_by_group[group_id].append(task_id)
    for task_ids in tasks_by_group.values():
        task_ids.sort(key=_task_index)

    diagnostic_groups = {
        str(frozen["goals"][task_id]["asin"])
        for task_id in DIAGNOSTIC_TASKS
    }
    remaining = sorted(
        set(tasks_by_group) - diagnostic_groups,
        key=lambda group_id: (_rank(group_id), group_id),
    )
    if len(remaining) != 4:
        raise ValueError(f"expected 4 non-diagnostic groups, got {len(remaining)}")
    groups_by_role = {
        "adaptation": remaining[:2],
        "calibration": remaining[2:3],
        "confirmation": remaining[3:],
        "diagnostic": sorted(diagnostic_groups, key=lambda value: (_rank(value), value)),
    }
    representatives = {
        role: [tasks_by_group[group_id][0] for group_id in groups]
        for role, groups in groups_by_role.items()
    }
    assigned = [group for groups in groups_by_role.values() for group in groups]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(tasks_by_group):
        raise AssertionError("semantic groups are not a disjoint exhaustive partition")

    artifact = {
        "schema_version": 1,
        "artifact_role": "WEBSHOP_CONSUMED_SEMANTIC_GROUP_SPLIT_V8",
        "claim_limit": "Consumed WebShop goals only; held-out remains unread.",
        "group_key": "goal.asin",
        "selection_rule": (
            "V7 diagnosed ASIN groups are diagnostic; remaining ASIN groups are "
            "sha256-ranked into 2 adaptation, 1 calibration, and 1 confirmation group"
        ),
        "namespace": NAMESPACE,
        "diagnostic_task_ids": list(DIAGNOSTIC_TASKS),
        "groups_by_role": groups_by_role,
        "representative_tasks_by_role": representatives,
        "tasks_by_group": dict(sorted(tasks_by_group.items())),
        "goal_manifest_artifact_sha256": frozen["artifact_sha256"],
        "held_out_read_or_run": False,
    }
    canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
    artifact["artifact_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--goal-manifest",
        type=Path,
        default=REPO / "configs/webshop_consumed_goals_v7.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "configs/webshop_grouped_development_v8.json",
    )
    args = parser.parse_args()
    frozen = json.loads(args.goal_manifest.read_text())
    artifact = build_grouped_manifest(frozen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": "GROUPED",
        "unique_groups": len(artifact["tasks_by_group"]),
        "role_counts": {
            role: len(groups) for role, groups in artifact["groups_by_role"].items()
        },
        "artifact_sha256": artifact["artifact_sha256"],
    }))


if __name__ == "__main__":
    main()
