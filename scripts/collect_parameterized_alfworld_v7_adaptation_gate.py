#!/usr/bin/env python3
"""Collect expert receipts for the frozen V7 target adaptation gate only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment
from motif_transfer.contracts import stable_hash


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=97401)
    parser.add_argument("--max-steps", type=int, default=180)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite adaptation gate: {args.output}")
    manifest = _read(args.manifest)
    body = dict(manifest)
    claimed_hash = str(body.pop("manifest_sha256", ""))
    if stable_hash(body) != claimed_hash:
        raise SystemExit("adaptation-gate manifest hash mismatch")
    if manifest.get("status") != "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET":
        raise SystemExit("adaptation gate was not frozen before reset")
    if manifest.get("selection_used_target_rollout_outcomes"):
        raise SystemExit("adaptation gate selection used target outcomes")
    if set(manifest.get("splits", {})) != {"adaptation_gate"}:
        raise SystemExit("collector is restricted to the adaptation_gate split")
    task_ids = tuple(map(str, manifest["splits"]["adaptation_gate"]))
    train_root = Path(str(manifest["train_root"])).resolve()
    episodes = []
    seen: set[str] = set()
    environment = ALFWorldTextBatchEnvironment(
        config_path=str(args.alfworld_config.resolve()),
        data_path=str(args.alfworld_data.resolve()),
        split="train",
        seed=args.seed,
        game_ids=task_ids,
        max_steps=args.max_steps,
    )
    try:
        for task_index in range(len(task_ids)):
            observation = environment.reset()
            task_id = (
                Path(environment.resolved_game_file).resolve()
                .relative_to(train_root).as_posix()
            )
            if task_id not in task_ids or task_id in seen:
                raise RuntimeError(f"adaptation identity violation: {task_id}")
            seen.add(task_id)
            transitions = []
            for step in range(args.max_steps):
                expert_action = environment.expert_action()
                after, reward = environment.step(expert_action)
                receipt_body = {
                    "task_id": task_id,
                    "partition": "adaptation_gate",
                    "step": step,
                    "before": dict(observation.state),
                    "native_actions": list(observation.native_actions),
                    "expert_action": expert_action,
                    "after": dict(after.state),
                    "after_native_actions": list(after.native_actions),
                    "reward": float(reward),
                    "official_success_after": bool(after.official_success),
                }
                transitions.append({
                    "step": step,
                    "goal": str(observation.state.get("task_goal", "")),
                    "before_observation": str(
                        observation.state.get("observation", "")
                    ),
                    "native_actions": list(observation.native_actions),
                    "expert_action": expert_action,
                    "after_observation": str(after.state.get("observation", "")),
                    "after_native_actions": list(after.native_actions),
                    "reward": float(reward),
                    "official_success_after": bool(after.official_success),
                    "receipt_sha256": stable_hash(receipt_body),
                })
                observation = after
                if after.terminal or after.official_success:
                    break
            success = bool(transitions and transitions[-1]["official_success_after"])
            episodes.append({
                "task_index": task_index,
                "task_id": task_id,
                "task_family": task_id.split("-", 1)[0],
                "partition": "adaptation_gate",
                "seed": args.seed,
                "official_success": success,
                "transitions": transitions,
            })
            print(json.dumps({
                "task_index": task_index,
                "task_id": task_id,
                "success": success,
                "steps": len(transitions),
            }), flush=True)
    finally:
        environment.close()
    if seen != set(task_ids):
        raise RuntimeError("adaptation gate did not cover its frozen task set")
    payload = {
        "schema_version": "parameterized-alfworld-adaptation-gate-receipts-v7",
        "authority": "FROZEN_TARGET_ADAPTATION_EXPERT_RECEIPTS",
        "manifest_path": str(args.manifest.resolve()),
        "manifest_file_sha256": _sha256(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "selection_used_target_outcomes": False,
        "confirmation_or_heldout_read": False,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "episodes": len(episodes),
        "successful": sum(row["official_success"] for row in episodes),
        "confirmation_or_heldout_read": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
