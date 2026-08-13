#!/usr/bin/env python3
"""Apply frozen paired gates to V10 WebShop receipts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402


AUTHENTIC = "authentic_source_plus_target"
CONTROLS = (
    "target_only",
    "target_native_myopic",
    "shuffled_source_plus_target",
    "source_marginal_plus_target",
)


def exact_binomial_two_sided(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    tail = min(wins, losses)
    return min(
        1.0,
        2.0 * sum(math.comb(total, index) for index in range(tail + 1)) / 2**total,
    )


def _load(directory: Path) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted(directory.glob("webshop.*.*.json"))
        if path.name != "summary.json"
    ]


def _paired(rows: list[dict], control: str) -> dict:
    authentic = {row["task_id"]: row for row in rows if row["condition"] == AUTHENTIC}
    baseline = {row["task_id"]: row for row in rows if row["condition"] == control}
    tasks = sorted(set(authentic) & set(baseline))
    success_delta = np.asarray([
        float(authentic[task]["strict_success"]) - float(baseline[task]["strict_success"])
        for task in tasks
    ])
    reward_delta = np.asarray([
        float(authentic[task]["official_reward"]) - float(baseline[task]["official_reward"])
        for task in tasks
    ])
    wins = int(np.sum(success_delta > 0))
    losses = int(np.sum(success_delta < 0))
    return {
        "control": control,
        "tasks": len(tasks),
        "strict_wins": wins,
        "strict_losses": losses,
        "strict_ties": len(tasks) - wins - losses,
        "strict_success_delta": float(np.mean(success_delta)) if len(tasks) else None,
        "mean_reward_delta": float(np.mean(reward_delta)) if len(tasks) else None,
        "paired_exact_p_two_sided": exact_binomial_two_sided(wins, losses),
        "action_contrast_tasks": sum(
            [step["selected_action"] for step in authentic[task]["steps"]]
            != [step["selected_action"] for step in baseline[task]["steps"]]
            for task in tasks
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--replication-dir", type=Path)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.frozen_config.read_text())
    qualification = _load(args.qualification_dir)
    expected_qualification = set(config["splits"]["qualification"])
    observed_qualification = {row["task_id"] for row in qualification}
    if observed_qualification != expected_qualification:
        raise SystemExit("qualification task set mismatch")
    qualification_comparisons = [_paired(qualification, control) for control in CONTROLS]
    qualification_passed = bool(
        all(row["failure"] is None for row in qualification)
        and all(
            len({
                row["initial_state_hash"]
                for row in qualification if row["task_id"] == task_id
            }) == 1
            for task_id in expected_qualification
        )
        and sum(
            row["source_decision_count"]
            for row in qualification if row["condition"] == AUTHENTIC
        ) > 0
        and all(
            comparison["strict_wins"] > comparison["strict_losses"]
            and comparison["mean_reward_delta"] > 0
            and comparison["action_contrast_tasks"] > 0
            for comparison in qualification_comparisons
        )
    )

    replication = _load(args.replication_dir) if args.replication_dir else []
    final = None
    if replication:
        expected_replication = set(config["splits"]["replication"])
        observed_replication = {row["task_id"] for row in replication}
        if observed_replication != expected_replication:
            raise SystemExit("replication task set mismatch")
        combined = qualification + replication
        comparisons = [_paired(combined, control) for control in CONTROLS]
        final_passed = bool(
            qualification_passed
            and all(row["failure"] is None for row in replication)
            and all(
                len({
                    row["initial_state_hash"]
                    for row in replication if row["task_id"] == task_id
                }) == 1
                for task_id in expected_replication
            )
            and all(
                comparison["strict_wins"] > comparison["strict_losses"]
                and comparison["mean_reward_delta"] > 0
                and comparison["paired_exact_p_two_sided"] <= 0.05
                and comparison["action_contrast_tasks"] > 0
                for comparison in comparisons
            )
        )
        final = {
            "tasks": len(expected_qualification | expected_replication),
            "comparisons": comparisons,
            "passed": final_passed,
            "status": (
                "REAL_WEBSHOP_NEURAL_SYMBOLIC_TRANSFER_VALIDATED"
                if final_passed else "REAL_WEBSHOP_NEURAL_SYMBOLIC_TRANSFER_NOT_VALIDATED"
            ),
        }
    report = {
        "schema_version": 1,
        "experiment": "webshop_neural_symbolic_transfer_v10",
        "claim_boundary": config["claim_boundary"],
        "qualification": {
            "tasks": len(expected_qualification),
            "comparisons": qualification_comparisons,
            "passed": qualification_passed,
        },
        "final": final,
        "scientific_status": (
            final["status"] if final is not None
            else "QUALIFICATION_PASS" if qualification_passed
            else "QUALIFICATION_FAIL"
        ),
        "runtime_hashes": {
            "summarizer": file_sha256(Path(__file__)),
            "frozen_config": file_sha256(args.frozen_config),
            "qualification_summary": file_sha256(args.qualification_dir / "summary.json"),
            "replication_summary": (
                file_sha256(args.replication_dir / "summary.json")
                if args.replication_dir else None
            ),
        },
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
