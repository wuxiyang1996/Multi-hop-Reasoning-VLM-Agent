#!/usr/bin/env python3
"""Evaluate identity/count-aware ALFWorld multiplicity reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_multiplicity_grounder import workflow_status  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def summarize(
    *, report_path: Path, config_path: Path, role: str,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    episodes = report["episodes"]
    required = {
        "target_only",
        "authentic_source_plus_target",
        "shuffled_source_plus_target",
        "source_marginal_plus_target",
        "phase_permuted_source_plus_target",
    }
    if set(episodes) != required:
        raise ValueError("multiplicity condition matrix is incomplete")
    by_condition = {
        condition: {row["task_id"]: row for row in rows}
        for condition, rows in episodes.items()
    }
    task_ids = list(by_condition["target_only"])
    if any(set(rows) != set(task_ids) for rows in by_condition.values()):
        raise ValueError("multiplicity paired task identities differ")
    expected_tasks = 11 if role == "consumed_development" else 6
    if len(task_ids) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} {role} tasks")

    authentic = by_condition["authentic_source_plus_target"]
    target = by_condition["target_only"]
    wins = sum(
        authentic[task]["official_success"] and not target[task]["official_success"]
        for task in task_ids
    )
    losses = sum(
        target[task]["official_success"] and not authentic[task]["official_success"]
        for task in task_ids
    )
    changed_tasks = sum(authentic[task]["changed_options"] > 0 for task in task_ids)
    identity_audit = []
    for task_id in task_ids:
        episode = authentic[task_id]
        history: list[str] = []
        maximum_distinct = 0
        for row in episode["records"]:
            goal = str(row["goal"])
            history.append(str(row["action"]))
            status = workflow_status(goal, history)
            maximum_distinct = max(maximum_distinct, status.placed_count)
            if status.placed_count != len(status.placed_object_ids):
                raise ValueError("distinct object count invariant failed")
        identity_audit.append({
            "task_id": task_id,
            "maximum_distinct_placed": maximum_distinct,
            "final_distinct_placed": workflow_status(
                str(episode["records"][0]["goal"]), history,
            ).placed_count if episode["records"] else 0,
            "official_success": bool(episode["official_success"]),
        })

    summaries = report["summaries"]
    authentic_summary = summaries["authentic_source_plus_target"]
    destructive = (
        "shuffled_source_plus_target",
        "source_marginal_plus_target",
        "phase_permuted_source_plus_target",
    )
    minimum_changed_tasks = 8 if role == "consumed_development" else 4
    minimum_successes = 8 if role == "consumed_development" else 4
    gates = {
        "exact_paired_task_count": len(task_ids) == expected_tasks,
        "identity_set_invariant": all(
            row["maximum_distinct_placed"] <= 2 for row in identity_audit
        ),
        "task_level_binding_intervention": changed_tasks >= minimum_changed_tasks,
        "minimum_authentic_successes": (
            authentic_summary["successes"] >= minimum_successes
        ),
        "authentic_strictly_above_target": (
            authentic_summary["successes"] > summaries["target_only"]["successes"]
        ),
        "paired_wins_above_losses": wins > losses,
        "authentic_strictly_above_destructive_controls": all(
            authentic_summary["successes"] > summaries[name]["successes"]
            for name in destructive
        ),
        "authentic_more_efficient_than_target": (
            authentic_summary["mean_steps"] < summaries["target_only"]["mean_steps"]
        ),
    }
    passed = all(gates.values())
    result: dict[str, Any] = {
        "schema_version": "alfworld-multiplicity-v1-summary",
        "status": (
            "CONSUMED_DEVELOPMENT_MULTIPLICITY_GATE_PASSED"
            if passed and role == "consumed_development"
            else (
                "FRESH_MULTIPLICITY_TRANSFER_VALIDATED"
                if passed else "MULTIPLICITY_GATE_FAILED"
            )
        ),
        "role": role,
        "tasks": len(task_ids),
        "summaries": summaries,
        "paired_authentic_vs_target": {
            "wins": wins,
            "losses": losses,
            "ties": len(task_ids) - wins - losses,
            "exact_two_sided_p": _sign_p(wins, losses),
        },
        "task_level_changed_option_tasks": changed_tasks,
        "minimum_changed_option_tasks": minimum_changed_tasks,
        "identity_audit": identity_audit,
        "gates": gates,
        "raw_runner_status_preserved": report["status"],
        "raw_report_sha256": _sha256(report_path),
        "config_sha256": _sha256(config_path),
        "claim_boundary": config["claim_boundary"],
    }
    result["summary_sha256"] = stable_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--role", choices=("consumed_development", "formal"), required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        report_path=args.report, config_path=args.config, role=args.role,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(result["gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
