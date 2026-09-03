#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402


CONDITIONS = (
    "target_only",
    "selective_minimum_repeat",
    "selective_authentic_source",
    "selective_phase_permuted_source",
    "selective_other_game_source",
)


def _load_episode_receipts(directories: list[Path]) -> dict[tuple[str, str], dict]:
    receipts = {}
    for directory in (directory.resolve() for directory in directories):
        for path in sorted(directory.glob("webshop.*.json")):
            if path.name == "summary.json":
                continue
            row = json.loads(path.read_text())
            key = (row["task_id"], row["condition"])
            if key in receipts:
                raise ValueError(f"duplicate receipt: {key}")
            receipts[key] = row
    return receipts


def _episode_metrics(rows: list[dict]) -> dict:
    steps = sum(row["step_count"] for row in rows)
    interventions = sum(row["changed_from_target_rank_zero_count"] for row in rows)
    return {
        "tasks": len(rows),
        "strict_successes": sum(row["strict_success"] for row in rows),
        "pass_successes": sum(row["pass_success"] for row in rows),
        "mean_official_reward": mean(row["official_reward"] for row in rows),
        "mean_steps": mean(row["step_count"] for row in rows),
        "failures": sum(row["failure"] is not None for row in rows),
        "interventions": interventions,
        "total_steps": steps,
        "intervention_rate": interventions / max(1, steps),
    }


def _fork_metrics(fork_report: dict, condition: str) -> dict:
    branches = []
    for opportunity in fork_report["opportunities"]:
        matches = [
            branch for branch in opportunity["branches"]
            if condition in branch["branch_conditions"]
        ]
        if len(matches) != 1:
            raise ValueError(f"fork condition coverage error: {condition}")
        branches.append(matches[0])
    return {
        "opportunities": len(branches),
        "strict_horizon_successes": sum(row["horizon_reward"] >= 1.0 - 1e-9 for row in branches),
        "pass_horizon_successes": sum(row["horizon_reward"] >= 0.5 for row in branches),
        "mean_horizon_reward": mean(row["horizon_reward"] for row in branches),
        "mean_branch_steps": mean(len(row["actions"]) for row in branches),
        "failures": sum(row["failure"] is not None for row in branches),
        "all_state_matches": all(row["state_match"] for row in branches),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--fork-report", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/webshop_neurosymbolic_causal_v7.json",
    )
    parser.add_argument(
        "--goal-manifest", type=Path,
        default=REPO / "configs/webshop_consumed_goals_v7.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipts = _load_episode_receipts(args.episode_dirs)
    task_ids = sorted({task for task, _ in receipts}, key=lambda value: int(value.split(".")[1]))
    if any((task, condition) not in receipts for task in task_ids for condition in CONDITIONS):
        raise ValueError("episode condition matrix is incomplete")
    runner_hashes = {
        receipts[(task, condition)]["runtime_hashes"]["runner"]
        for task in task_ids for condition in CONDITIONS
    }
    goal_hashes = {
        receipts[(task, condition)]["runtime_hashes"].get("goal_manifest")
        for task in task_ids for condition in CONDITIONS
    }
    episode_metrics = {
        condition: _episode_metrics([receipts[(task, condition)] for task in task_ids])
        for condition in CONDITIONS
    }
    fork_report = json.loads(args.fork_report.read_text())
    fork_metrics = {
        condition: _fork_metrics(fork_report, condition) for condition in CONDITIONS
    }
    target_episode = episode_metrics["target_only"]
    authentic_episode = episode_metrics["selective_authentic_source"]
    minimum_episode = episode_metrics["selective_minimum_repeat"]
    target_fork = fork_metrics["target_only"]
    authentic_fork = fork_metrics["selective_authentic_source"]
    minimum_fork = fork_metrics["selective_minimum_repeat"]

    causal_checks = {
        "all_eight_episode_pairs_complete": all(
            metrics["tasks"] == 8 and metrics["failures"] == 0
            for metrics in episode_metrics.values()
        ),
        "all_fork_states_reconstructed": all(
            metrics["all_state_matches"] and metrics["failures"] == 0
            for metrics in fork_metrics.values()
        ),
        "authentic_preserves_episode_strict_success": (
            authentic_episode["strict_successes"] >= target_episode["strict_successes"]
        ),
        "authentic_beats_minimum_repeat_episode_reward": (
            authentic_episode["mean_official_reward"] > minimum_episode["mean_official_reward"]
        ),
        "authentic_beats_target_fork_reward": (
            authentic_fork["mean_horizon_reward"] > target_fork["mean_horizon_reward"]
        ),
        "authentic_beats_minimum_repeat_fork_reward": (
            authentic_fork["mean_horizon_reward"] > minimum_fork["mean_horizon_reward"]
        ),
        "authentic_beats_minimum_repeat_fork_strict_success": (
            authentic_fork["strict_horizon_successes"]
            > minimum_fork["strict_horizon_successes"]
        ),
    }
    transfer_claim_pass = all(causal_checks.values())
    report = {
        "schema_version": 1,
        "experiment": "webshop_neurosymbolic_causal_v7_development",
        "claim_limit": "Consumed tasks only; held-out remains unread.",
        "task_ids": task_ids,
        "episode_metrics": episode_metrics,
        "fork_metrics": fork_metrics,
        "paired_deltas": {
            "authentic_minus_target_episode_reward": (
                authentic_episode["mean_official_reward"]
                - target_episode["mean_official_reward"]
            ),
            "authentic_minus_minimum_episode_reward": (
                authentic_episode["mean_official_reward"]
                - minimum_episode["mean_official_reward"]
            ),
            "authentic_minus_target_fork_reward": (
                authentic_fork["mean_horizon_reward"] - target_fork["mean_horizon_reward"]
            ),
            "authentic_minus_minimum_fork_reward": (
                authentic_fork["mean_horizon_reward"] - minimum_fork["mean_horizon_reward"]
            ),
        },
        "causal_checks": causal_checks,
        "transfer_claim_status": "PASS" if transfer_claim_pass else "REJECT_CURRENT_GATE",
        "interpretation": (
            "Authentic source values sometimes outperform minimum-repeat, but they reduce strict "
            "success and underperform target rank zero. The current repeat applicability gate is rejected."
        ),
        "runtime_hashes": {
            "episode_runners": sorted(runner_hashes),
            "goal_manifests_in_receipts": sorted(value for value in goal_hashes if value),
            "config": file_sha256(args.config),
            "goal_manifest": file_sha256(args.goal_manifest),
            "fork_report": file_sha256(args.fork_report),
            "summarizer": file_sha256(Path(__file__)),
        },
        "held_out_authorized": False,
        "held_out_read_or_run": False,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
