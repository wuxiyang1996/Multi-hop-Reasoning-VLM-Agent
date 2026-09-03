#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib import request


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


CONSUMED_ROLES = ("adaptation", "qualification", "reserve")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_index(task_id: str) -> int:
    prefix, raw_index = task_id.split(".", 1)
    if prefix != "webshop":
        raise ValueError(f"not a WebShop task id: {task_id}")
    return int(raw_index)


def _consumed_tasks(manifest: dict) -> dict[str, list[str]]:
    roles = manifest["targets"]["webshop"]["partition"]["roles"]
    return {role: list(roles[role]) for role in CONSUMED_ROLES}


def _fetch_goal(base_url: str, task_id: str, namespace: str) -> dict:
    session_id = f"{namespace}_fixed_{_task_index(task_id)}"
    url = f"{base_url.rstrip('/')}/__bridge/session/{session_id}"
    with request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    goal = payload.get("goal")
    if not isinstance(goal, dict) or not goal.get("instruction_text"):
        raise ValueError(f"server returned no goal for {task_id}")
    return goal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition-manifest",
        type=Path,
        default=REPO / "configs/real_game_multitarget_v5_manifest.json",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--goal-seed", type=int, default=233)
    parser.add_argument("--namespace", default="goal-freeze-v7")
    parser.add_argument(
        "--server-app",
        type=Path,
        default=Path(
            "/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/vendor/"
            "WebShop/web_agent_site/app.py"
        ),
    )
    parser.add_argument(
        "--goal-module",
        type=Path,
        default=Path(
            "/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/vendor/"
            "WebShop/web_agent_site/engine/goal.py"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "configs/webshop_consumed_goals_v7.json",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    partition_manifest = json.loads(args.partition_manifest.read_text())
    roles = _consumed_tasks(partition_manifest)
    task_ids = sorted(
        {task for role_tasks in roles.values() for task in role_tasks},
        key=_task_index,
    )
    goals = {
        task_id: _fetch_goal(args.base_url, task_id, args.namespace)
        for task_id in task_ids
    }
    goal_hashes = {task_id: stable_hash(goal) for task_id, goal in goals.items()}
    artifact = {
        "schema_version": 1,
        "artifact_role": "consumed_webshop_goal_freeze_v7",
        "claim_limit": "Contains consumed roles only; WebShop held-out remains unread.",
        "goal_seed": args.goal_seed,
        "roles": roles,
        "task_ids": task_ids,
        "goals": goals,
        "goal_hashes": goal_hashes,
        "goal_set_sha256": stable_hash(goal_hashes),
        "runtime_hashes": {
            "partition_manifest": _file_sha256(args.partition_manifest),
            "server_app": _file_sha256(args.server_app),
            "goal_module": _file_sha256(args.goal_module),
            "freezer": _file_sha256(Path(__file__)),
        },
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    if args.verify:
        frozen = json.loads(args.output.read_text())
        if artifact != frozen:
            changed = [
                task_id for task_id in task_ids
                if artifact["goal_hashes"].get(task_id) != frozen["goal_hashes"].get(task_id)
            ]
            raise SystemExit(f"goal freeze mismatch: {changed}")
        print(json.dumps({
            "status": "MATCH",
            "tasks": len(task_ids),
            "artifact_sha256": artifact["artifact_sha256"],
        }))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": "FROZEN",
        "tasks": len(task_ids),
        "artifact_sha256": artifact["artifact_sha256"],
    }))


if __name__ == "__main__":
    main()
