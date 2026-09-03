#!/usr/bin/env python3
"""Run one auditable multi-step WebShop fork from a frozen target receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402
from scripts.run_webshop_intervention_forks_v7 import _run_branch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-receipt", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--wrapper-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument(
        "--goal-manifest",
        type=Path,
        default=REPO / "configs/webshop_consumed_goals_v7.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--maximum-output-tokens", type=int, default=3200)
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--fork-horizon", type=int, default=5)
    parser.add_argument("--run-id", default="targeted-v9")
    args = parser.parse_args()

    receipt = json.loads(args.target_receipt.read_text())
    step = receipt["steps"][args.step]
    if int(step["step"]) != args.step:
        raise SystemExit("receipt step indices are not contiguous")
    if not 0 <= args.candidate_index < len(step["candidates"]):
        raise SystemExit("candidate index out of bounds")
    goals = json.loads(args.goal_manifest.read_text())
    expected_goal = goals["goals"][receipt["task_id"]]["instruction_text"]
    if expected_goal != receipt["goal"]:
        raise SystemExit("receipt goal does not match frozen goal manifest")

    values = runpy.run_path(str(args.keys))
    api_key = values.get("OPENROUTER_API_KEY") or values.get("openrouter_api_key")
    if not api_key:
        raise SystemExit("OpenRouter API key is missing")
    os.environ["V9_OPENROUTER_API_KEY"] = str(api_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            args.base_url,
            {"decision": args.model},
            api_key_env="V9_OPENROUTER_API_KEY",
            json_mode=True,
            temperature=0,
            timeout_seconds=180,
            request_overrides={"max_tokens": args.maximum_output_tokens},
        ),
        cache_path=args.output_dir / "decision_cache.json",
    )
    opportunity = {
        "task_id": receipt["task_id"],
        "step": args.step,
        "before_hash": step["before_hash"],
        "prefix_actions": [row["selected_action"] for row in receipt["steps"][:args.step]],
        "prefix_rewards": [row["reward"] for row in receipt["steps"][:args.step]],
        "candidates": step["candidates"],
        "predicted_effects": step["predicted_effects"],
        "selected_indices": {"targeted_target_native_test": args.candidate_index},
        "selection_receipts": {},
    }
    result = _run_branch(
        opportunity=opportunity,
        branch_action_index=args.candidate_index,
        branch_conditions=["targeted_target_native_test"],
        backend=backend,
        wrapper_root=args.wrapper_root,
        expected_goal=expected_goal,
        run_id=args.run_id,
        fork_horizon=args.fork_horizon,
        candidate_count=args.candidate_count,
        schema_retries=args.schema_retries,
    )
    result["claim_limit"] = "WebShop adaptation group only; not confirmation evidence."
    result["runtime"] = {
        "model": args.model,
        "fork_horizon": args.fork_horizon,
        "candidate_count": args.candidate_count,
        "schema_retries": args.schema_retries,
    }
    result["runtime_hashes"] = {
        "runner": file_sha256(Path(__file__)),
        "fork_core": file_sha256(REPO / "scripts/run_webshop_intervention_forks_v7.py"),
        "target_receipt": file_sha256(args.target_receipt),
        "goal_manifest": file_sha256(args.goal_manifest),
        "decision_cache": file_sha256(args.output_dir / "decision_cache.json"),
    }
    result["receipt_sha256"] = stable_hash(result)
    output = args.output_dir / "fork_receipt.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "state_match": result["state_match"],
        "branch_action": result["branch_action"],
        "horizon_reward": result["horizon_reward"],
        "terminated": result["terminated"],
        "steps": len(result["actions"]),
        "failure": result["failure"],
        "output": str(output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
