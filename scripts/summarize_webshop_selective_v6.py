#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parents[1]
CONDITIONS = (
    "target_only",
    "selective_authentic_source",
    "selective_phase_permuted_source",
    "selective_other_game_source",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_receipts(directories: list[Path]) -> tuple[dict[tuple[str, str], dict], dict[str, str]]:
    receipts: dict[tuple[str, str], dict] = {}
    hashes: dict[str, str] = {}
    for directory in (directory.resolve() for directory in directories):
        for path in sorted(directory.glob("webshop.*.json")):
            if path.name == "summary.json":
                continue
            row = json.loads(path.read_text())
            key = (row["task_id"], row["condition"])
            if key in receipts:
                raise ValueError(f"duplicate receipt: {key}")
            receipts[key] = row
            hashes[str(path.relative_to(REPO))] = _sha256(path)
    return receipts, hashes


def _metrics(rows: list[dict]) -> dict:
    total_steps = sum(row["step_count"] for row in rows)
    interventions = sum(row["changed_from_target_rank_zero_count"] for row in rows)
    return {
        "tasks": len(rows),
        "strict_successes": sum(row["strict_success"] for row in rows),
        "pass_successes": sum(row["pass_success"] for row in rows),
        "any_reward_tasks": sum(row["any_reward"] for row in rows),
        "mean_official_reward": mean(row["official_reward"] for row in rows),
        "mean_steps": mean(row["step_count"] for row in rows),
        "interventions": interventions,
        "total_steps": total_steps,
        "intervention_rate": interventions / max(1, total_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qualification-dirs",
        type=Path,
        nargs="+",
        default=[
            REPO / "runs/real_game_multitarget_neurosymbolic_v6/webshop_reserve_smoke2",
            REPO / "runs/real_game_multitarget_neurosymbolic_v6/webshop_reserve_remaining6",
        ],
    )
    parser.add_argument(
        "--control-dirs",
        type=Path,
        nargs="+",
        default=[
            REPO / "runs/real_game_multitarget_neurosymbolic_v6/webshop_controls_smoke2_full",
            REPO / "runs/real_game_multitarget_neurosymbolic_v6/webshop_controls_remaining6_full",
        ],
    )
    parser.add_argument(
        "--all-condition-dirs",
        type=Path,
        nargs="+",
        help="Directories that each contain receipts for every condition.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=REPO / "configs/webshop_selective_neurosymbolic_v6.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "runs/real_game_multitarget_neurosymbolic_v6/qualification_report.json",
    )
    args = parser.parse_args()

    if args.all_condition_dirs:
        receipts, receipt_hashes = _load_receipts(args.all_condition_dirs)
    else:
        qualification, qualification_hashes = _load_receipts(args.qualification_dirs)
        controls, control_hashes = _load_receipts(args.control_dirs)
        receipts = qualification | controls
        receipt_hashes = qualification_hashes | control_hashes
    all_tasks = sorted({task for task, _ in receipts}, key=lambda value: int(value.split(".")[1]))
    complete_tasks = [
        task for task in all_tasks
        if all(
            (task, condition) in receipts and receipts[(task, condition)]["failure"] is None
            for condition in CONDITIONS
        )
    ]
    excluded = {
        task: {
            condition: receipts.get((task, condition), {}).get("failure", "missing_receipt")
            for condition in CONDITIONS
        }
        for task in all_tasks if task not in complete_tasks
    }
    condition_metrics = {
        condition: _metrics([receipts[(task, condition)] for task in complete_tasks])
        for condition in CONDITIONS
    }
    matched_initial_states = all(
        len({receipts[(task, condition)]["initial_state_hash"] for condition in CONDITIONS}) == 1
        for task in complete_tasks
    )

    target = condition_metrics["target_only"]
    authentic = condition_metrics["selective_authentic_source"]
    controls_metrics = [
        condition_metrics["selective_phase_permuted_source"],
        condition_metrics["selective_other_game_source"],
    ]
    config = json.loads(args.experiment_config.read_text())
    preregistered_gates = {
        "minimum_intervention_rate": authentic["intervention_rate"]
        >= config["gates"]["minimum_intervention_rate"],
        "no_reduction_vs_target_strict_or_pass": (
            authentic["strict_successes"] >= target["strict_successes"]
            and authentic["pass_successes"] >= target["pass_successes"]
        ),
        "improves_mean_reward_or_steps_vs_target": (
            authentic["mean_official_reward"] > target["mean_official_reward"]
            or authentic["mean_steps"] < target["mean_steps"]
        ),
        "beats_or_matches_every_control_on_all_success_metrics": all(
            authentic[metric] >= control[metric]
            for control in controls_metrics
            for metric in ("strict_successes", "pass_successes", "mean_official_reward")
        ),
    }
    operational_safeguards = {
        "minimum_seven_of_eight_complete_pairs": len(complete_tasks) >= 7,
    }
    held_out_authorized = (
        all(preregistered_gates.values()) and all(operational_safeguards.values())
    )

    task_rows = []
    for task in complete_tasks:
        task_rows.append({
            "task_id": task,
            "conditions": {
                condition: {
                    "official_reward": receipts[(task, condition)]["official_reward"],
                    "strict_success": receipts[(task, condition)]["strict_success"],
                    "pass_success": receipts[(task, condition)]["pass_success"],
                    "steps": receipts[(task, condition)]["step_count"],
                    "interventions": receipts[(task, condition)][
                        "changed_from_target_rank_zero_count"
                    ],
                }
                for condition in CONDITIONS
            },
        })

    report = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "claim": (
            "Fixed-runner qualification has matched success and reward with fewer authentic "
            "steps on complete pairs. Infrastructure coverage is insufficient for held-out."
        ),
        "complete_case_policy": "Exclude a task if any paired condition has an infrastructure failure.",
        "complete_tasks": complete_tasks,
        "excluded_tasks": excluded,
        "matched_initial_state_hashes": matched_initial_states,
        "condition_metrics": condition_metrics,
        "paired_authentic_minus_target": {
            "strict_successes": authentic["strict_successes"] - target["strict_successes"],
            "pass_successes": authentic["pass_successes"] - target["pass_successes"],
            "mean_official_reward": (
                authentic["mean_official_reward"] - target["mean_official_reward"]
            ),
            "mean_steps": authentic["mean_steps"] - target["mean_steps"],
        },
        "preregistered_gates": preregistered_gates,
        "operational_safeguards": operational_safeguards,
        "held_out_authorized": held_out_authorized,
        "gate_interpretation": (
            "Fail closed: 'beat or match controls' requires no lower strict success, pass "
            "success, or mean official reward than either destructive control."
        ),
        "task_rows": task_rows,
        "runtime_hashes": {
            "experiment_config": _sha256(args.experiment_config),
            "summarizer": _sha256(Path(__file__)),
            "receipts": receipt_hashes,
        },
        "held_out_read_or_run": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
