#!/usr/bin/env python3
"""Independent audit of DiscoveryWorld selective causal utility V2."""

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

CONDITIONS = (
    "target_native_myopic", "authentic_sokoban_effect_plus_target",
    "commit_availability_control_plus_target", "inverted_effect_control_plus_target",
    "position_prior_control_plus_target",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def self_hash(value: dict, field: str) -> bool:
    body = dict(value); claimed = body.pop(field, None)
    return bool(claimed and claimed == stable_hash(body))


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if not n: return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "configs/phase2_discoveryworld_utility_v2/manifest.json")
    parser.add_argument("--run-root", type=Path, default=REPO / "runs/phase2_discoveryworld_utility_v2")
    parser.add_argument("--v1-root", type=Path, default=REPO / "runs/phase2_discoveryworld_utility_v1")
    parser.add_argument("--output", type=Path, default=REPO / "docs/results/phase2_discoveryworld_utility_v2_audit.json")
    parser.add_argument("--compact-output", type=Path, default=REPO / "docs/results/phase2_discoveryworld_utility_v2_compact.json")
    parser.add_argument("--v1-output", type=Path, default=REPO / "docs/results/phase2_discoveryworld_utility_v1_coverage_failure.json")
    args = parser.parse_args()
    manifest = read(args.manifest)
    report = read(args.run_root / "report.json")
    v1_forks = read(args.v1_root / "frozen_forks/fork_freeze_receipt.json")
    tasks = manifest["tasks"]
    cells = [read(args.run_root / "cells" / row["task_id"] / "cell.json") for row in tasks]
    starts = [read(args.run_root / "cells" / row["task_id"] / "started.json") for row in tasks]
    by_task = {row["task_id"]: row for row in cells}
    counts = {condition: 0 for condition in CONDITIONS}
    wins = losses = ties = 0
    recorded_successes = recorded_wins = recorded_losses = recorded_ties = 0
    cell_valid = start_valid = raw_valid = raw_file_match = True
    matched_valid = selection_valid = oracle_safe = True
    route_valid = True; route_count = 0
    abstention_valid = True
    compact_rows = []
    for task, cell, start in zip(tasks, cells, starts):
        cell_valid &= self_hash(cell, "cell_sha256") and cell["manifest_sha256"] == manifest["manifest_sha256"]
        start_valid &= self_hash(start, "start_sha256") and start["manifest_sha256"] == manifest["manifest_sha256"]
        outcomes = cell["outcomes"]
        target_episode_path = REPO / task["target_episode"]
        if file_hash(target_episode_path) != task["target_episode_file_sha256"]:
            raise ValueError(f"target episode changed: {task['task_id']}")
        recorded_outcome = bool(read(target_episode_path)["evaluation"]["official_success"])
        recorded_successes += int(recorded_outcome)
        if tuple(outcomes) != CONDITIONS: raise ValueError("condition order changed")
        for condition in CONDITIONS: counts[condition] += int(bool(outcomes[condition]))
        if outcomes[CONDITIONS[1]] and not outcomes[CONDITIONS[0]]: wins += 1
        elif outcomes[CONDITIONS[0]] and not outcomes[CONDITIONS[1]]: losses += 1
        else: ties += 1
        if outcomes[CONDITIONS[1]] and not recorded_outcome: recorded_wins += 1
        elif recorded_outcome and not outcomes[CONDITIONS[1]]: recorded_losses += 1
        else: recorded_ties += 1
        if task["applicable"]:
            raw_path = args.run_root / "cells" / task["task_id"] / "matched_result.json"
            raw = read(raw_path)
            raw_valid &= self_hash(raw, "result_sha256")
            raw_file_match &= file_hash(raw_path) == cell["matched_result_file_sha256"]
            matched_valid &= raw.get("all_matched_forks") is True
            selection_valid &= raw.get("all_selection_receipts_valid") is True
            oracle_safe &= raw.get("policy_runtime_saw_oracle_scorecard") is False
            routes = cell["authentic_source_routes"]
            route_valid &= bool(routes)
            for route in routes:
                route_count += 1; body = dict(route); claimed = body.pop("receipt_sha256", None)
                route_valid &= bool(claimed == stable_hash(body) and route.get("admitted") is True and route.get("source_artifact_sha256") == task["source_artifact_sha256"])
        else:
            abstention_valid &= bool(
                cell["matched_result_file_sha256"] is None and not cell["authentic_source_routes"]
                and len(set(outcomes.values())) == 1
                and cell["abstention_rule"] == "INHERIT_RECORDED_TARGET_ONLY_OUTCOME_FOR_ALL_ARMS"
                and not (args.run_root / "cells" / task["task_id"] / "matched_result.json").exists()
            )
        compact_rows.append({
            "task_id": task["task_id"], "source_game": task["source_game"],
            "applicable": task["applicable"], "outcomes": outcomes,
            "recovery_steps": cell["recovery_steps"], "cell_sha256": cell["cell_sha256"],
            "matched_result_file_sha256": cell["matched_result_file_sha256"],
            "recorded_target_only_outcome": recorded_outcome,
        })
    p = sign_p(wins, losses); neg = losses / (wins + losses) if wins + losses else 0.0
    recorded_p = sign_p(recorded_wins, recorded_losses)
    eligible = [row for row in v1_forks["receipts"] if row["eligible"]]
    ineligible = [row for row in v1_forks["receipts"] if not row["eligible"]]
    gates = {
        "manifest_self_hash_valid": self_hash(manifest, "manifest_sha256"),
        "frozen_runtime_hashes_match": all(file_hash(REPO / path) == value for path, value in manifest["runtime_file_sha256"].items()),
        "v1_coverage_failure_preserved_35_plus_1": len(eligible) == 35 and len(ineligible) == 1 and ineligible[0]["task_id"] == "proteomics.easy.seed70",
        "eligibility_outcome_blind_and_matched_unopened_at_v2_freeze": v1_forks["outcome_fields_read_for_eligibility"] is False and manifest["matched_outcomes_visible_at_freeze"] is False and manifest["v1_matched_arms_executed_before_v2_freeze"] == 0,
        "exact_36_cell_and_start_hashes_valid": len(cells) == len(starts) == 36 and cell_valid and start_valid,
        "exact_35_raw_matched_results_valid": sum(row["applicable"] for row in tasks) == 35 and raw_valid and raw_file_match,
        "all_matched_forks_and_selection_receipts_valid": matched_valid and selection_valid,
        "zero_policy_oracle_scorecard_use": oracle_safe,
        "all_authentic_routes_source_bound_and_admitted": route_valid and route_count == report["source_route_count"],
        "single_fail_closed_abstention_exact": abstention_valid and sum(not row["applicable"] for row in tasks) == 1,
        "independent_counts_match_report": counts == report["condition_successes"],
        "independent_paired_statistics_match_report": (
            {"wins": wins, "losses": losses, "ties": ties} == {k: report["authentic_vs_raw"][k] for k in ("wins", "losses", "ties")}
            and p == report["authentic_vs_raw"]["exact_two_sided_sign_test_p"]
            and neg == report["authentic_vs_raw"]["discordant_negative_transfer_rate"]
        ),
        "primary_causal_effect_gate_passed": counts[CONDITIONS[1]] > counts[CONDITIONS[0]] and p <= .05 and neg <= .25,
        "authentic_strictly_beats_all_symbolic_controls": all(counts[CONDITIONS[1]] > counts[c] for c in CONDITIONS[2:]),
        "report_self_hash_status_and_gates_valid": self_hash(report, "report_sha256") and report["status"] == "PHASE2_DISCOVERYWORLD_SELECTIVE_CAUSAL_UTILITY_VALIDATED" and all(report["gates"].values()),
    }
    digest = hashlib.sha256("".join(sorted(row["cell_sha256"] for row in cells)).encode()).hexdigest()
    compact_body = {
        "schema_version": "phase2-discoveryworld-selective-utility-compact-v2",
        "status": report["status"], "claim_boundary": manifest["claim_boundary"],
        "manifest_sha256": manifest["manifest_sha256"], "report_sha256": report["report_sha256"],
        "v1_fork_freeze_summary_sha256": v1_forks["summary_sha256"],
        "condition_successes": counts,
        "authentic_vs_raw": {"wins": wins, "losses": losses, "ties": ties, "exact_two_sided_sign_test_p": p, "discordant_negative_transfer_rate": neg},
        "secondary_authentic_vs_recorded_target_only": {
            "authentic_successes": counts[CONDITIONS[1]],
            "recorded_target_only_successes": recorded_successes,
            "wins": recorded_wins, "losses": recorded_losses,
            "ties": recorded_ties, "exact_two_sided_sign_test_p": recorded_p,
            "preregistered_primary": False,
        },
        "cell_receipt_aggregate_sha256": digest, "per_task": compact_rows,
    }
    compact = compact_body | {"compact_sha256": stable_hash(compact_body)}
    audit_body = {
        "schema_version": "phase2-discoveryworld-selective-utility-independent-audit-v2",
        "status": "PHASE2_DISCOVERYWORLD_V2_INDEPENDENT_AUDIT_PASSED" if all(gates.values()) else "PHASE2_DISCOVERYWORLD_V2_INDEPENDENT_AUDIT_FAILED",
        "manifest_sha256": manifest["manifest_sha256"], "report_sha256": report["report_sha256"],
        "compact_sha256": compact["compact_sha256"], "cell_receipt_aggregate_sha256": digest,
        "independent_condition_successes": counts,
        "independent_authentic_vs_raw": {"wins": wins, "losses": losses, "ties": ties, "exact_two_sided_sign_test_p": p, "discordant_negative_transfer_rate": neg},
        "secondary_authentic_vs_recorded_target_only": compact["secondary_authentic_vs_recorded_target_only"],
        "gates": gates, "passed_gates": sum(gates.values()), "required_gates": len(gates),
    }
    audit = audit_body | {"audit_sha256": stable_hash(audit_body)}
    v1_body = {
        "schema_version": "phase2-discoveryworld-v1-coverage-failure-v1",
        "status": "PHASE2_DISCOVERYWORLD_V1_COVERAGE_FAILED_NO_MATCHED_ARMS_RUN",
        "manifest_sha256": manifest["parent_v1_manifest_sha256"],
        "target_tasks_completed": 36, "eligible_forks": 35, "ineligible_forks": 1,
        "ineligible_task_id": ineligible[0]["task_id"], "ineligible_reason": ineligible[0]["reason"],
        "eligibility_read_target_outcome": False, "matched_arms_run": 0,
        "fork_freeze_summary_sha256": v1_forks["summary_sha256"],
    }
    v1 = v1_body | {"failure_receipt_sha256": stable_hash(v1_body)}
    for path, value in ((args.output, audit), (args.compact_output, compact), (args.v1_output, v1)):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if audit["status"].endswith("_PASSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
