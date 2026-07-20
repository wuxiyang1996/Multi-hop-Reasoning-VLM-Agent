#!/usr/bin/env python3
"""Record exactly one fixed successful ALFWorld train demonstration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env_wrappers.alfworld_nl_wrapper import make_alfworld_env  # noqa: E402
from harness.alfworld_demo_recorder import AlfworldDemoRecorder  # noqa: E402


def _expert_action(info: Dict[str, Any]) -> str:
    plan: Any = info.get("extra.expert_plan")
    while isinstance(plan, (list, tuple)) and len(plan) == 1:
        plan = plan[0]
    if isinstance(plan, (list, tuple)) and plan:
        plan = plan[0]
    action = str(plan or "").strip()
    if not action:
        raise RuntimeError("ALFWorld did not expose an expert action for the fixed train demo")
    admissible = list(info.get("action_names") or [])
    if action not in admissible:
        raise RuntimeError(f"expert action is not admissible: {action!r}")
    return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        default=str(REPO_ROOT / "configs/alfworld_pick_and_place_config.yaml"),
    )
    parser.add_argument("--demo-id", default="alfworld-pick-and-place-train-seed42-v2-shot0")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts/admission_demos/alfworld/pick_and_place/train_seed42_v2_shot0.json",
    )
    args = parser.parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")
    env = make_alfworld_env(
        split="train",
        max_steps=args.max_steps,
        config_path=args.config_path,
        random_seed=42,
    )
    recorder = AlfworldDemoRecorder(
        env,
        demo_id=args.demo_id,
        episode_id=args.demo_id,
        task_family="pick_and_place",
        split="train",
    )
    actions: List[str] = []
    try:
        _, info = recorder.reset()
        for _ in range(args.max_steps):
            action = _expert_action(info)
            actions.append(action)
            _, reward, terminated, truncated, info = recorder.step(action)
            won = info.get("won", False)
            if isinstance(won, (list, tuple)):
                won = won[0] if won else False
            if bool(won) or float(reward) >= 1.0 or terminated or truncated:
                break
    finally:
        env.close()
    receipt = recorder.receipt()
    if not receipt.official_success:
        raise SystemExit(
            "fixed one-shot demonstration failed; refusing to retry/select a better demo"
        )
    receipt.validate_for_admission()
    payload = asdict(receipt)
    payload["demo_hash"] = receipt.content_hash()
    payload["selection_policy"] = {
        "protocol_version": 2,
        "split": "train",
        "task_types": [1],
        "random_seed": 42,
        "shot_index": 0,
        "best_of_n": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, args.output)
    print(json.dumps({
        "demo_id": receipt.demo_id,
        "demo_hash": receipt.content_hash(),
        "official_success": receipt.official_success,
        "official_score": receipt.official_score,
        "n_actions": len(actions),
        "operators": sorted({item.operator for item in receipt.actions}),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
