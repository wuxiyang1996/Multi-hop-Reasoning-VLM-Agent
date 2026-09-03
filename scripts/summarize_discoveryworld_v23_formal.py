#!/usr/bin/env python3
"""Apply the predeclared V23 DiscoveryWorld formal gates without adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.discoveryworld_env import stable_hash  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def valid_self_hash(value: Mapping[str, Any], field: str) -> bool:
    body = dict(value)
    body.pop("_path", None)
    claimed = body.pop(field, None)
    return isinstance(claimed, str) and stable_hash(body) == claimed


def official_success(row: Mapping[str, Any], condition: str) -> bool:
    condition_row = row["conditions"][condition]
    return bool(condition_row.get("official_success"))


def summarize(
    *, protocol: Mapping[str, Any], freeze: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]], protocol_file_sha256: str,
    freeze_file_sha256: str,
) -> dict[str, Any]:
    task_ids = list(protocol["task_ids"])
    expected_eligible = {
        str(receipt["task_id"])
        for receipt in freeze["receipts"] if receipt.get("eligible")
    }
    result_ids = set(results)
    conditions = list(protocol["conditions"])
    counts = {
        condition: sum(official_success(row, condition) for row in results.values())
        for condition in conditions
    }
    per_task = []
    negative_transfer = 0
    for task_id in sorted(results):
        row = results[task_id]
        target_success = official_success(row, "target_native_myopic")
        authentic_success = official_success(
            row, "authentic_sokoban_effect_plus_target",
        )
        negative = target_success and not authentic_success
        negative_transfer += int(negative)
        per_task.append({
            "task_id": task_id,
            "scenario": str(row["task"]["scenario"]),
            "target_native_myopic": target_success,
            "authentic_sokoban_effect_plus_target": authentic_success,
            "commit_availability_control_plus_target": official_success(
                row, "commit_availability_control_plus_target",
            ),
            "inverted_effect_control_plus_target": official_success(
                row, "inverted_effect_control_plus_target",
            ),
            "position_prior_control_plus_target": official_success(
                row, "position_prior_control_plus_target",
            ),
            "authentic_negative_transfer": negative,
            "result_sha256": str(row["result_sha256"]),
        })

    expected_themes = {"Space Sick", "Proteomics"}
    observed_themes = {str(row["task"]["scenario"]) for row in results.values()}
    runtime_and_receipts = all(
        row.get("status") == "FORMAL_MECHANISM_COMPLETE"
        and row.get("all_selection_receipts_valid") is True
        and all(
            condition_row.get("runtime_error") is None
            for name, condition_row in row["conditions"].items()
            if name != "target_only_recorded"
        )
        and all(
            step["target_native_realization"]["receipt_sha256"]
            == stable_hash({
                key: value
                for key, value in step["target_native_realization"].items()
                if key != "receipt_sha256"
            })
            for name, condition_row in row["conditions"].items()
            if name != "target_only_recorded"
            for step in condition_row["recovery"]
        )
        for row in results.values()
    )
    gates = {
        "minimum_eligible_forks": len(expected_eligible)
        >= int(protocol["formal_gates"]["minimum_eligible_forks"]),
        "both_target_themes_represented": observed_themes == expected_themes,
        "zero_runtime_or_receipt_failures": runtime_and_receipts,
        "zero_authentic_negative_transfer_vs_target_native": negative_transfer == 0,
        "minimum_authentic_success_gain_vs_target_native": (
            counts["authentic_sokoban_effect_plus_target"]
            - counts["target_native_myopic"]
            >= int(protocol["formal_gates"][
                "minimum_authentic_success_gain_vs_target_native"
            ])
        ),
        "authentic_successes_strictly_greater_than_commit_availability_control": (
            counts["authentic_sokoban_effect_plus_target"]
            > counts["commit_availability_control_plus_target"]
        ),
        "authentic_successes_strictly_greater_than_position_prior_control": (
            counts["authentic_sokoban_effect_plus_target"]
            > counts["position_prior_control_plus_target"]
        ),
        "all_policy_and_audit_forks_matched": all(
            row.get("all_matched_forks") is True for row in results.values()
        ),
        "zero_policy_oracle_scorecard_use": all(
            row.get("policy_runtime_saw_oracle_scorecard") is False
            for row in results.values()
        ),
        "frozen_task_and_result_coverage_exact": (
            set(task_ids) >= expected_eligible == result_ids
        ),
        "all_result_self_hashes_valid": all(
            valid_self_hash(row, "result_sha256") for row in results.values()
        ),
        "fork_freeze_self_hash_valid": valid_self_hash(freeze, "summary_sha256"),
        "fork_selection_was_outcome_blind": (
            freeze.get("outcome_fields_read_for_eligibility") is False
        ),
    }
    passed = all(gates.values())
    report = {
        "schema_version": "discoveryworld-sokoban-fresh-easy-formal-summary-v23",
        "status": (
            "FRESH_FORMAL_TRANSFER_VALIDATED" if passed
            else "FRESH_FORMAL_TRANSFER_FAILED"
        ),
        "claim_boundary": protocol["claim_boundary"],
        "source_program_sha256": protocol["source_contract"][
            "source_program_sha256"
        ],
        "source_confirmation_sha256": protocol["source_contract"][
            "source_confirmation_sha256"
        ],
        "frozen_tasks": len(task_ids),
        "eligible_forks": len(expected_eligible),
        "eligible_task_ids": sorted(expected_eligible),
        "success_counts": counts,
        "authentic_gain_vs_target_native_myopic": (
            counts["authentic_sokoban_effect_plus_target"]
            - counts["target_native_myopic"]
        ),
        "authentic_negative_transfer_count": negative_transfer,
        "per_task": per_task,
        "gates": gates,
        "all_predeclared_gates_passed": passed,
        "integrity": {
            "protocol_file_sha256": protocol_file_sha256,
            "fork_freeze_file_sha256": freeze_file_sha256,
            "fork_freeze_summary_sha256": freeze.get("summary_sha256"),
            "result_file_sha256": {
                task_id: file_sha256(Path(str(row["_path"])))
                for task_id, row in results.items()
            },
        },
        "operational_disclosure": (
            "space_sick.easy.seed5 target-only collection was interrupted after "
            "30 steps and resumed by replaying the same cached completions and "
            "deterministic environment prefix; an independently restarted cache "
            "was discarded and is not included in this report."
        ),
    }
    for row in results.values():
        row.pop("_path", None)
    report["report_sha256"] = stable_hash(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--fork-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    freeze_path = args.fork_dir / "fork_freeze_receipt.json"
    freeze = read(freeze_path)
    results = {}
    for generated in freeze["generated_configs"]:
        task_id = Path(str(generated)).stem
        path = args.result_dir / f"{task_id}.json"
        row = read(path)
        row["_path"] = str(path)
        results[task_id] = row
    report = summarize(
        protocol=protocol,
        freeze=freeze,
        results=results,
        protocol_file_sha256=file_sha256(args.protocol),
        freeze_file_sha256=file_sha256(freeze_path),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "eligible_forks": report["eligible_forks"],
        "success_counts": report["success_counts"],
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
