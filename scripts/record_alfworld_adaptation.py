#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_env import ALFWorldTextEnvironment
from motif_transfer.contracts import stable_hash


def _task_family(game_id: str) -> str:
    return game_id.split("/", 1)[0].split("-", 1)[0]


def _observation_hash_payload(observation) -> dict[str, object]:
    return {
        "state": dict(observation.state),
        "native_actions": list(observation.native_actions),
        "terminal": bool(observation.terminal),
        "official_success": bool(observation.official_success),
        "official_score": float(observation.score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one frozen, official-expert ALFWorld adaptation example.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    env = ALFWorldTextEnvironment(
        config_path=str(args.config), data_path=str(args.data), split=args.split,
        seed=args.seed, game_id=args.game_id, max_steps=args.max_steps,
    )
    actions = []
    try:
        before = env.reset()
        for index in range(args.max_steps):
            action = env.expert_action()
            after, reward = env.step(action)
            actions.append({
                "transition_index": index,
                "action": action,
                "before_admissible_actions": list(before.native_actions),
                "after_admissible_actions": list(after.native_actions),
                "admissible_actions_sha256": stable_hash(before.native_actions),
                "next_admissible_actions_sha256": stable_hash(after.native_actions),
                "state_sha256": stable_hash(_observation_hash_payload(before)),
                "next_state_sha256": stable_hash(_observation_hash_payload(after)),
                "reward": reward,
                "terminated": after.terminal,
                "truncated": bool(index + 1 >= args.max_steps and not after.terminal),
                "official_success_after": after.official_success,
            })
            before = after
            if after.terminal or after.official_success:
                break
    finally:
        env.close()
    if not actions or not actions[-1]["official_success_after"]:
        raise SystemExit("frozen expert adaptation example did not reach official success")
    unsigned = {
        "demo_id": f"alfworld:{args.split}:{stable_hash(args.game_id)[:16]}",
        "target_domain": "alfworld",
        "task_family": _task_family(args.game_id),
        "split": args.split,
        "episode_id": args.game_id,
        "executor_kind": "real_official_expert",
        "evaluator": "alfworld_official_won",
        "official_success": True,
        "official_score": max(float(row["reward"]) for row in actions),
        "held_out": False,
        "native_evidence_version": 1,
        "actions": actions,
        "selection_policy": {
            "rule": "frozen_manifest_lowest_sha256_id",
            "game_id": args.game_id,
            "game_id_sha256": stable_hash(args.game_id),
            "best_of_n": False,
        },
        "resolved_game_index": env.resolved_game_index,
        "resolved_game_file": env.resolved_game_file,
    }
    payload = dict(unsigned)
    payload["source_file_sha256"] = stable_hash(unsigned)
    payload["demo_hash"] = stable_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "game_id": args.game_id,
        "resolved_game_index": env.resolved_game_index,
        "steps": len(actions), "official_success": True,
        "demo_hash": payload["demo_hash"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
