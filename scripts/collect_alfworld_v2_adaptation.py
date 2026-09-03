#!/usr/bin/env python3
"""Collect frozen ALFWorld train receipts for the V2 target-native grounder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _family(task_id: str) -> str:
    return task_id.split("-", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = config["target"]
    manifest_path = (REPO / target["pool_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_BEFORE_COLLECTION":
        raise SystemExit("adaptation pool was not frozen before collection")
    split_rows = [
        (partition, str(task_id))
        for partition in ("adaptation_train", "adaptation_validation")
        for task_id in manifest["splits"][partition]
    ]
    task_ids = tuple(task_id for _, task_id in split_rows)
    partition_by_task_id = {task_id: partition for partition, task_id in split_rows}
    output = (REPO / target["adaptation_receipts"]).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    episodes = []
    environment = ALFWorldTextBatchEnvironment(
        config_path=str(target["alfworld_config"]),
        data_path=str(target["alfworld_data"]),
        split=str(target["adaptation_split"]),
        seed=int(target["seed"]),
        game_ids=task_ids,
        max_steps=int(target["adaptation_max_steps"]),
    )
    try:
        train_root = Path(manifest["train_root"]).resolve()
        for task_index in range(len(task_ids)):
            transitions = []
            before = environment.reset()
            actual_path = Path(environment.resolved_game_file).resolve()
            task_id = actual_path.relative_to(train_root).as_posix()
            if task_id not in partition_by_task_id:
                raise RuntimeError(f"environment returned an unfrozen game: {task_id}")
            partition = partition_by_task_id[task_id]
            for step in range(int(target["adaptation_max_steps"])):
                expert_action = environment.expert_action()
                after, reward = environment.step(expert_action)
                transitions.append({
                    "step": step,
                    "goal": str(before.state.get("task_goal", "")),
                    "before_observation": str(before.state.get("observation", "")),
                    "native_actions": list(before.native_actions),
                    "expert_action": expert_action,
                    "after_observation": str(after.state.get("observation", "")),
                    "after_native_actions": list(after.native_actions),
                    "reward": float(reward),
                    "official_success_after": bool(after.official_success),
                    "receipt_sha256": stable_hash({
                        "task_id": task_id,
                        "partition": partition,
                        "step": step,
                        "before": dict(before.state),
                        "native": before.native_actions,
                        "expert_action": expert_action,
                        "after": dict(after.state),
                        "reward": reward,
                        "official_success": after.official_success,
                    }),
                })
                before = after
                if after.terminal or after.official_success:
                    break
            success = bool(transitions and transitions[-1]["official_success_after"])
            episodes.append({
                "task_index": task_index,
                "task_id": task_id,
                "task_family": _family(task_id),
                "partition": partition,
                "seed": int(target["seed"]),
                "resolved_game_file": environment.resolved_game_file,
                "official_success": success,
                "transitions": transitions,
            })
            print(json.dumps({
                "task_index": task_index,
                "partition": partition,
                "task_family": _family(task_id),
                "steps": len(transitions),
                "official_success": success,
            }), flush=True)
    finally:
        environment.close()

    payload = {
        "schema_version": "alfworld-target-native-effect-adaptation-v2",
        "authority": "FROZEN_TARGET_TRAIN_OFFICIAL_EXPERT_RECEIPTS",
        "pool_manifest_path": str(manifest_path),
        "pool_manifest_sha256": _sha256(manifest_path),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "selection_used_target_outcomes": False,
        "qualification_or_heldout_read": False,
        "episodes": episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "episodes": len(episodes),
        "successful": sum(row["official_success"] for row in episodes),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
