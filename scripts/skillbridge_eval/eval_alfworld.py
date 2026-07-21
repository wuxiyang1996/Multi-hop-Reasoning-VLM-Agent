"""Held-out ALFWorld evaluation driver for the active SkillBridge suite.

Runs the shared SkillBridge actor against ALFWorld's text environment using
only the environment's current admissible commands. Success is read from the
official ``won``/episode-score signal; no language-model judge is involved.

Run this module inside the isolated ALFWorld environment created by
``install/install_alfworld.sh``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
        default="eval_out_of_distribution",
    )
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--vllm-base-url", default="http://localhost:8000/v1",
    )
    parser.add_argument(
        "--harness-mode",
        choices=["full", "plain-text-skills", "off"],
        default="full",
    )
    parser.add_argument("--actor-bank-cap-k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--label", default="skillbridge")
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def _won(info: Dict[str, Any], reward: float) -> bool:
    value = info.get("won", False)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else False
    return bool(value) or float(reward) >= 1.0


def run_episode(
    *,
    env: Any,
    actor: Any,
    split: str,
    episode_idx: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Run one ALFWorld episode and return its auditable summary."""
    started = time.monotonic()
    observation, info = env.reset()
    total_steps = 0
    best_score = 0.0
    success = _won(info, best_score)
    terminated = False
    truncated = False
    actions: List[str] = []
    error: Optional[str] = None

    try:
        while not (success or terminated or truncated) and total_steps < max_steps:
            admissible = list(info.get("action_names") or [])
            if not admissible:
                error = "no_admissible_commands"
                break
            stats = actor.act(
                game=f"alfworld/{split}",
                obs_nl=str(observation),
                structured_state=dict(info.get("structured_state") or {}),
                action_names=admissible,
                episode_id=f"alfworld-{episode_idx:05d}",
                inner_step=total_steps,
            )
            action = str(stats.action)
            if action not in admissible:
                error = f"actor_returned_non_admissible:{action}"
                break
            actions.append(action)
            observation, score, terminated, truncated, info = env.step(action)
            total_steps += 1
            best_score = max(best_score, float(score))
            success = _won(info, best_score)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ALFWorld episode %d failed", episode_idx)
        error = repr(exc)

    return {
        "episode_idx": episode_idx,
        "success": success,
        "score": best_score,
        "steps": total_steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "actions": actions,
        "error": error,
        "wall_time_s": time.monotonic() - started,
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if row.get("error") is None]
    return {
        "n_tasks": 1,
        "n_episodes": len(rows),
        "n_episodes_total": len(valid),
        "n_episodes_completed": len(valid),
        "n_errors": len(rows) - len(valid),
        "success_rate": (
            sum(bool(row["success"]) for row in valid) / len(valid)
            if valid else 0.0
        ),
        "mean_score": mean([float(row["score"]) for row in valid]) if valid else 0.0,
        "mean_steps": mean([int(row["steps"]) for row in valid]) if valid else 0.0,
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not args.run_dir.exists():
        logger.error("run-dir %s missing", args.run_dir)
        return 1
    if args.episodes < 1 or args.max_steps < 1:
        logger.error("--episodes and --max-steps must be positive")
        return 2

    random.seed(args.seed)
    from env_wrappers.alfworld_nl_wrapper import make_alfworld_env
    from scripts.skillbridge_eval.eval_actor import SkillBridgeActor

    bank_dir = args.bank_dir or (args.run_dir / "skillbank")
    game_key = f"alfworld/{args.split}"
    actor = SkillBridgeActor.from_checkpoint(
        checkpoint_dir=None,
        bank_dir=bank_dir,
        vllm_base_url=args.vllm_base_url,
        model=args.model,
        harness_mode=args.harness_mode,
        actor_bank_cap_k=args.actor_bank_cap_k,
        games_for_harness=[game_key],
        harness_domain="alfworld",
        temperature=args.temperature,
    )
    env = make_alfworld_env(
        split=args.split,
        max_steps=args.max_steps,
        config_path=args.config_path,
    )
    try:
        rows = [
            run_episode(
                env=env,
                actor=actor,
                split=args.split,
                episode_idx=index,
                max_steps=args.max_steps,
            )
            for index in range(args.episodes)
        ]
    finally:
        env.close()

    overall = _aggregate(rows)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "schema_version": 1,
        "domain": "alfworld",
        "label": args.label,
        "run_dir": str(args.run_dir),
        "bank_dir": str(bank_dir),
        "model": args.model,
        "split": args.split,
        "max_steps": args.max_steps,
        "harness_mode": args.harness_mode,
        "overall": overall,
        "rows": rows,
    }
    output = args.output or (
        args.run_dir / "eval" / f"alfworld_result_{timestamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    logger.info("eval_alfworld: wrote %s", output)
    print("\n=== ALFWorld eval summary ===")
    print(f"episodes : {overall['n_episodes_completed']}/{overall['n_episodes']}")
    print(f"success  : {overall['success_rate']:.2%}")
    print(f"out      : {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
