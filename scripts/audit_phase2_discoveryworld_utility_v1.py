#!/usr/bin/env python3
"""Independent raw-receipt audit for DiscoveryWorld Phase-2 utility V1."""

from __future__ import annotations

import argparse
from math import comb
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import SOURCE_GAMES  # noqa: E402


CONDITIONS = (
    "target_native_myopic",
    "authentic_sokoban_effect_plus_target",
    "commit_availability_control_plus_target",
    "inverted_effect_control_plus_target",
    "position_prior_control_plus_target",
)


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def self_hash(value: dict, field: str) -> bool:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    return bool(claimed and claimed == stable_hash(body))


def sign_p(wins: int, losses: int) -> float:
    total = wins + losses
    if not total:
        return 1.0
    tail = min(wins, losses)
    return min(1.0, 2 * sum(comb(total, k) for k in range(tail + 1)) / (2 ** total))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase2_discoveryworld_utility_v1/manifest.json",
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=REPO / "runs/phase2_discoveryworld_utility_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/phase2_discoveryworld_utility_v1_audit.json",
    )
    parser.add_argument(
        "--compact-output", type=Path,
        default=REPO / "docs/results/phase2_discoveryworld_utility_v1_compact.json",
    )
    args = parser.parse_args()
    manifest = read(args.manifest)
    report = read(args.run_root / "report.json")
    preparation = read(args.run_root / "preparation_receipt.json")
    fork_freeze = read(args.run_root / "frozen_forks/fork_freeze_receipt.json")
    tasks = list(manifest.get("tasks") or ())
    cells = []
    raw_results = []
    starts = []
    target_episodes = []
    for task in tasks:
        task_id = str(task["task_id"])
        cell_dir = args.run_root / "cells" / task_id
        cells.append(read(cell_dir / "cell.json"))
        raw_results.append(read(cell_dir / "matched_result.json"))
        starts.append(read(cell_dir / "started.json"))
        target_episodes.append(read(args.run_root / "target_only" / f"{task_id}.json"))

    manifest_body = dict(manifest)
    manifest_claimed = manifest_body.pop("manifest_sha256", None)
    counts = {condition: 0 for condition in CONDITIONS}
    wins = losses = ties = 0
    route_count = 0
    route_hashes_valid = True
    source_bound = True
    raw_result_hashes_valid = True
    raw_result_file_hashes_match = True
    cell_hashes_valid = True
    starts_valid = True
    matched_forks = True
    selection_receipts = True
    oracle_safe = True
    compact_rows = []
    for task, cell, raw, start in zip(tasks, cells, raw_results, starts):
        cell_hashes_valid &= self_hash(cell, "cell_sha256")
        starts_valid &= self_hash(start, "start_sha256")
        raw_result_hashes_valid &= self_hash(raw, "result_sha256")
        raw_path = args.run_root / "cells" / str(task["task_id"]) / "matched_result.json"
        raw_result_file_hashes_match &= (
            cell.get("matched_result_file_sha256") == file_hash(raw_path)
        )
        matched_forks &= raw.get("all_matched_forks") is True
        selection_receipts &= raw.get("all_selection_receipts_valid") is True
        oracle_safe &= raw.get("policy_runtime_saw_oracle_scorecard") is False
        outcomes = dict(cell.get("outcomes") or {})
        if tuple(outcomes) != CONDITIONS:
            raise ValueError(f"condition coverage/order changed: {task['task_id']}")
        for condition in CONDITIONS:
            counts[condition] += int(bool(outcomes[condition]))
        if outcomes[CONDITIONS[1]] and not outcomes[CONDITIONS[0]]:
            wins += 1
        elif outcomes[CONDITIONS[0]] and not outcomes[CONDITIONS[1]]:
            losses += 1
        else:
            ties += 1
        for route in cell.get("authentic_source_routes") or ():
            route_count += 1
            route_body = dict(route)
            route_claimed = route_body.pop("receipt_sha256", None)
            route_hashes_valid &= route_claimed == stable_hash(route_body)
            source_bound &= bool(
                route.get("admitted") is True
                and route.get("source_artifact_sha256") == task["source_artifact_sha256"]
            )
        compact_rows.append({
            "task_id": task["task_id"],
            "source_game": task["source_game"],
            "source_artifact_sha256": task["source_artifact_sha256"],
            "outcomes": outcomes,
            "recovery_steps": cell["recovery_steps"],
            "cell_sha256": cell["cell_sha256"],
            "matched_result_file_sha256": cell["matched_result_file_sha256"],
        })

    p_value = sign_p(wins, losses)
    negative_rate = losses / (wins + losses) if wins + losses else 0.0
    source_counts = {
        game: sum(task["source_game"] == game for task in tasks) for game in SOURCE_GAMES
    }
    preparation_files_match = all(
        row["episode_file_sha256"] == file_hash(
            args.run_root / "target_only" / f"{row['task_id']}.json"
        )
        for row in preparation.get("tasks") or ()
    )
    target_episode_hashes_valid = all(self_hash(row, "episode_sha256") for row in target_episodes)
    gates = {
        "manifest_self_hash_valid": manifest_claimed == stable_hash(manifest_body),
        "frozen_runtime_hashes_match": all(
            file_hash(REPO / relative) == expected
            for relative, expected in manifest["runtime_file_sha256"].items()
        ),
        "exact_36_balanced_fresh_tasks": (
            len(tasks) == 36 and set(source_counts.values()) == {6}
            and all(row["selected_target_previously_executed"] is False for row in tasks)
        ),
        "preparation_self_hash_valid": self_hash(preparation, "preparation_receipt_sha256"),
        "exact_one_target_process_per_task": (
            len(preparation.get("tasks") or ()) == 36
            and all(row["target_process_count"] == 1 for row in preparation["tasks"])
        ),
        "target_episode_files_and_self_hashes_valid": preparation_files_match and target_episode_hashes_valid,
        "fork_freeze_self_hash_and_outcome_blind": (
            self_hash(fork_freeze, "summary_sha256")
            and fork_freeze.get("outcome_fields_read_for_eligibility") is False
            and len(fork_freeze.get("generated_configs") or ()) == 36
        ),
        "exact_36_cell_and_start_receipts_valid": (
            len(cells) == len(starts) == 36 and cell_hashes_valid and starts_valid
        ),
        "raw_matched_results_hash_valid": raw_result_hashes_valid and raw_result_file_hashes_match,
        "all_policy_and_audit_forks_matched": matched_forks,
        "all_selection_receipts_valid": selection_receipts,
        "zero_policy_oracle_scorecard_use": oracle_safe,
        "authentic_routes_valid_source_bound_and_nonempty": (
            route_count > 0 and route_hashes_valid and source_bound
        ),
        "independent_counts_match_report": (
            counts == report.get("condition_successes")
            and {"wins": wins, "losses": losses, "ties": ties}
            == {key: report["authentic_vs_raw"][key] for key in ("wins", "losses", "ties")}
        ),
        "independent_statistics_match_report": (
            p_value == report["authentic_vs_raw"]["exact_two_sided_sign_test_p"]
            and negative_rate == report["authentic_vs_raw"]["discordant_negative_transfer_rate"]
        ),
        "primary_effect_gate_passed": (
            counts[CONDITIONS[1]] > counts[CONDITIONS[0]] and p_value <= 0.05
            and negative_rate <= 0.25
        ),
        "three_symbolic_controls_strictly_worse": all(
            counts[CONDITIONS[1]] > counts[condition] for condition in CONDITIONS[2:]
        ),
        "report_self_hash_and_status_valid": (
            self_hash(report, "report_sha256")
            and report.get("status") == "PHASE2_DISCOVERYWORLD_CAUSAL_UTILITY_VALIDATED"
            and all((report.get("gates") or {}).values())
        ),
    }
    receipt_digest = hashlib.sha256(
        "".join(sorted(cell["cell_sha256"] for cell in cells)).encode()
    ).hexdigest()
    compact_body = {
        "schema_version": "phase2-discoveryworld-causal-utility-compact-v1",
        "status": report["status"],
        "claim_boundary": manifest["claim_boundary"],
        "manifest_sha256": manifest["manifest_sha256"],
        "preparation_receipt_sha256": preparation["preparation_receipt_sha256"],
        "fork_freeze_summary_sha256": fork_freeze["summary_sha256"],
        "report_sha256": report["report_sha256"],
        "condition_successes": counts,
        "authentic_vs_raw": {
            "wins": wins, "losses": losses, "ties": ties,
            "exact_two_sided_sign_test_p": p_value,
            "discordant_negative_transfer_rate": negative_rate,
        },
        "historical_pilot_outcomes_included": False,
        "cell_receipt_aggregate_sha256": receipt_digest,
        "per_task": compact_rows,
    }
    compact = compact_body | {"compact_sha256": stable_hash(compact_body)}
    body = {
        "schema_version": "phase2-discoveryworld-causal-utility-independent-audit-v1",
        "status": (
            "PHASE2_DISCOVERYWORLD_V1_INDEPENDENT_AUDIT_PASSED"
            if all(gates.values()) else "PHASE2_DISCOVERYWORLD_V1_INDEPENDENT_AUDIT_FAILED"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "report_sha256": report["report_sha256"],
        "compact_sha256": compact["compact_sha256"],
        "cell_receipt_aggregate_sha256": receipt_digest,
        "independent_condition_successes": counts,
        "independent_authentic_vs_raw": {
            "wins": wins, "losses": losses, "ties": ties,
            "exact_two_sided_sign_test_p": p_value,
            "discordant_negative_transfer_rate": negative_rate,
        },
        "gates": gates,
        "passed_gates": sum(bool(value) for value in gates.values()),
        "required_gates": len(gates),
    }
    audit = body | {"audit_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.compact_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.compact_output.write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if audit["status"].endswith("_PASSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
