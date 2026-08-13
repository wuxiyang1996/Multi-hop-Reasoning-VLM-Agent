#!/usr/bin/env python3
"""Apply the frozen V13 paired gates to real-Sokoban WebShop receipts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402


AUTHENTIC = "authentic_sokoban_effect_plus_target"
COMPARATORS = (
    "target_only",
    "target_native_myopic",
    "commit_availability_control_plus_target",
    "inverted_effect_control_plus_target",
    "position_prior_control_plus_target",
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


def load_receipts(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("webshop.*.*.json"))
    ]


def receipt_hash_valid(receipt: Mapping[str, Any]) -> bool:
    body = dict(receipt)
    claimed = body.pop("receipt_sha256", None)
    return claimed == stable_hash(body)


def paired_metrics(
    rows: Sequence[Mapping[str, Any]], comparator: str,
) -> dict[str, Any]:
    authentic = {row["task_id"]: row for row in rows if row["condition"] == AUTHENTIC}
    control = {row["task_id"]: row for row in rows if row["condition"] == comparator}
    tasks = sorted(set(authentic) & set(control))
    success_deltas = [
        int(authentic[task]["strict_success"]) - int(control[task]["strict_success"])
        for task in tasks
    ]
    reward_deltas = [
        float(authentic[task]["official_reward"])
        - float(control[task]["official_reward"])
        for task in tasks
    ]
    step_deltas = [
        int(authentic[task]["step_count"]) - int(control[task]["step_count"])
        for task in tasks
    ]
    wins = sum(delta > 0 for delta in success_deltas)
    losses = sum(delta < 0 for delta in success_deltas)
    contrast_tasks = [
        task for task in tasks
        if [step["selected_action"] for step in authentic[task]["steps"]]
        != [step["selected_action"] for step in control[task]["steps"]]
    ]
    return {
        "comparator": comparator,
        "tasks": len(tasks),
        "strict_wins": wins,
        "strict_losses": losses,
        "strict_ties": len(tasks) - wins - losses,
        "strict_success_delta": (
            sum(success_deltas) / len(tasks) if tasks else None
        ),
        "mean_reward_delta": sum(reward_deltas) / len(tasks) if tasks else None,
        "mean_step_delta": sum(step_deltas) / len(tasks) if tasks else None,
        "paired_exact_p_two_sided": exact_binomial_two_sided(wins, losses),
        "action_contrast_tasks": len(contrast_tasks),
        "action_contrast_task_ids": contrast_tasks,
    }


def evaluate(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tasks = tuple(config["task_ids"])
    conditions = tuple(config["conditions"])
    expected_cells = {(task, condition) for task in tasks for condition in conditions}
    observed_cells = [(row.get("task_id"), row.get("condition")) for row in rows]
    observed_set = set(observed_cells)
    matrix_complete = bool(
        len(observed_cells) == len(observed_set)
        and observed_set == expected_cells
    )
    zero_failures = bool(
        matrix_complete and all(row.get("failure") is None for row in rows)
    )
    matched = bool(matrix_complete and all(
        len({
            row.get("initial_state_hash") for row in rows
            if row.get("task_id") == task
        }) == 1
        for task in tasks
    ))
    hashes_valid = bool(matrix_complete and all(receipt_hash_valid(row) for row in rows))
    receipt_matrix_sha256 = stable_hash(sorted(
        (
            str(row.get("task_id")),
            str(row.get("condition")),
            str(row.get("receipt_sha256")),
        )
        for row in rows
    ))
    authentic_source_decisions = sum(
        int(row.get("source_decision_count", 0))
        for row in rows if row.get("condition") == AUTHENTIC
    )
    comparisons = [paired_metrics(rows, comparator) for comparator in COMPARATORS]
    comparison_gates = {
        row["comparator"]: {
            "complete": row["tasks"] == len(tasks),
            "strict_wins_exceed_losses": row["strict_wins"] > row["strict_losses"],
            "positive_strict_success_delta": row["strict_success_delta"] is not None
            and row["strict_success_delta"] > 0,
            "positive_mean_reward_delta": row["mean_reward_delta"] is not None
            and row["mean_reward_delta"] > 0,
            "paired_exact_p_at_most_0p05": row["paired_exact_p_two_sided"] <= 0.05,
            "action_contrast": row["action_contrast_tasks"] > 0,
        }
        for row in comparisons
    }
    primary_gates = {
        "receipt_matrix_complete": matrix_complete,
        "receipt_hashes_valid": hashes_valid,
        "zero_final_failures": zero_failures,
        "matched_initial_state_hashes": matched,
        "authentic_source_decisions": authentic_source_decisions > 0,
        "all_comparator_gates": all(
            all(gates.values()) for gates in comparison_gates.values()
        ),
    }
    passed = all(primary_gates.values())
    conditions_summary = {}
    for condition in conditions:
        condition_rows = [row for row in rows if row.get("condition") == condition]
        conditions_summary[condition] = {
            "n": len(condition_rows),
            "strict_successes": sum(
                bool(row.get("strict_success")) for row in condition_rows
            ),
            "pass_successes": sum(
                bool(row.get("pass_success")) for row in condition_rows
            ),
            "mean_reward": (
                sum(float(row["official_reward"]) for row in condition_rows)
                / len(condition_rows) if condition_rows else None
            ),
            "mean_steps": (
                sum(int(row["step_count"]) for row in condition_rows)
                / len(condition_rows) if condition_rows else None
            ),
            "source_decisions": sum(
                int(row.get("source_decision_count", 0)) for row in condition_rows
            ),
            "failures": sum(row.get("failure") is not None for row in condition_rows),
        }
    return {
        "tasks": len(tasks),
        "receipt_count": len(rows),
        "receipt_matrix_sha256": receipt_matrix_sha256,
        "conditions": conditions_summary,
        "comparisons": comparisons,
        "comparison_gates": comparison_gates,
        "primary_gates": primary_gates,
        "passed": passed,
        "scientific_status": (
            "REAL_SOKOBAN_TO_WEBSHOP_NEURAL_SYMBOLIC_TRANSFER_VALIDATED"
            if passed else "REAL_SOKOBAN_TO_WEBSHOP_TRANSFER_V13_NOT_VALIDATED"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    rows = load_receipts(args.receipt_dir)
    evaluation = evaluate(rows, config)
    report = {
        "schema_version": 1,
        "experiment": "webshop_sokoban_effect_transfer_v13",
        "claim_boundary": config["claim_boundary"],
        "source_contract": config["source_contract"],
        "target_contract": config["target_contract"],
        **evaluation,
        "operational_retries": json.loads(
            (args.receipt_dir / "summary.json").read_text(encoding="utf-8")
        ).get("operational_retries", []),
        "runtime_hashes": {
            "summarizer": file_sha256(Path(__file__)),
            "frozen_config": file_sha256(args.frozen_config),
            "run_summary": file_sha256(args.receipt_dir / "summary.json"),
        },
        "v12_reserved_data_used": False,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
