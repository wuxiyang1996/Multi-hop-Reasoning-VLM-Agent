#!/usr/bin/env python3
"""Promote the existing frozen TIR maze replication through an independent audit.

No new TIR sample is opened.  The input was prospectively frozen and already
executed as an independent 48-task replication; this script checks raw receipts
and makes the causal utility evidence portable.
"""

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


RAW = "raw_target_only"
AUTHENTIC = "authentic_sokoban_topology_plus_target"
ALPHA = "alpha_renamed_authentic"
PERMUTED = "direction_permuted_source_control"
ENDPOINT = "endpoint_only_target_control"
MARGINAL = "path_length_marginal_control"
CONDITIONS = (RAW, AUTHENTIC, ALPHA, PERMUTED, ENDPOINT, MARGINAL)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def self_hash(value: dict, field: str) -> bool:
    body = dict(value)
    claimed = body.pop(field, None)
    return bool(claimed and claimed == stable_hash(body))


def sign_p(wins: int, losses: int) -> float:
    total = wins + losses
    if not total:
        return 1.0
    return min(
        1.0,
        2 * sum(comb(total, k) for k in range(min(wins, losses) + 1)) / 2 ** total,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/tir_maze_topology_replication_v1_frozen.json",
    )
    parser.add_argument(
        "--report", type=Path,
        default=REPO / "runs/tir_maze_topology_replication_v1/heldout_report.json",
    )
    parser.add_argument(
        "--receipts", type=Path,
        default=REPO / "runs/tir_maze_topology_replication_v1/heldout_receipts.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/phase2_tirbench_utility_v1_audit.json",
    )
    parser.add_argument(
        "--compact-output", type=Path,
        default=REPO / "docs/results/phase2_tirbench_utility_v1_compact.json",
    )
    args = parser.parse_args()
    config = read(args.config)
    report = read(args.report)
    receipts = read(args.receipts)
    if not isinstance(receipts, list):
        raise ValueError("TIR receipt matrix must be a list")

    condition_successes = {condition: 0 for condition in CONDITIONS}
    traces = list(report.get("traces") or ())
    trace_by_key = {
        (str(row["sample_id"]), str(row["condition"])): row for row in traces
    }
    receipt_hashes_valid = True
    native_hashes_valid = True
    trace_hashes_valid = True
    alpha_invariant = True
    binding_valid = True
    task_rows = []
    wins = losses = ties = 0
    for receipt in receipts:
        receipt_hashes_valid &= self_hash(receipt, "receipt_sha256")
        binding_valid &= receipt.get("neural_binding_valid") is True
        sample_id = str(receipt["sample_id"])
        outcomes = {}
        answers = {}
        for condition in CONDITIONS:
            trace = trace_by_key.get((sample_id, condition))
            if trace is None:
                raise ValueError(f"missing trace: {sample_id}/{condition}")
            trace_hashes_valid &= self_hash(trace, "trace_sha256")
            correct = bool(trace["correct_evaluator_only"])
            condition_successes[condition] += int(correct)
            outcomes[condition] = correct
            answers[condition] = trace["committed_answer"]
            if condition != RAW:
                native = receipt["conditions"][condition]
                native_hashes_valid &= self_hash(native, "receipt_sha256")
                native_hashes_valid &= (
                    trace["native_receipt_sha256"] == native["receipt_sha256"]
                )
        alpha_invariant &= (
            outcomes[AUTHENTIC] == outcomes[ALPHA]
            and answers[AUTHENTIC] == answers[ALPHA]
        )
        if outcomes[AUTHENTIC] and not outcomes[RAW]:
            wins += 1
        elif outcomes[RAW] and not outcomes[AUTHENTIC]:
            losses += 1
        else:
            ties += 1
        task_rows.append({
            "sample_id": sample_id,
            "neural_binding_sha256": stable_hash(receipt["neural_binding"]),
            "outcomes": outcomes,
            "committed_answers": answers,
            "receipt_sha256": receipt["receipt_sha256"],
        })
    p_value = sign_p(wins, losses)
    report_summaries_match = all(
        report["summaries"][condition]["successes"] == successes
        for condition, successes in condition_successes.items()
    )
    report_paired = report["paired"][RAW]
    report_paired_match = (
        report_paired["wins"] == wins
        and report_paired["losses"] == losses
        and report_paired["ties"] == ties
        and report_paired["exact_two_sided_p"] == p_value
    )
    gates = {
        "prospective_config_and_48_fresh_ids": (
            config.get("status") == "FROZEN_BEFORE_FRESH_QUALIFICATION"
            and config["replication"]["role"] == "INDEPENDENT_FRESH_REPLICATION"
            and config["replication"]["expected_tasks"] == 48
            and len(config["splits"]["heldout"]) == 48
            and len(set(config["splits"]["heldout"])) == 48
        ),
        "config_hash_matches_frozen_report": (
            file_hash(args.config) == report["integrity"]["config_file_sha256"]
        ),
        "raw_receipts_file_hash_matches_report": (
            file_hash(args.receipts) == report["integrity"]["receipts_file_sha256"]
        ),
        "source_artifact_and_confirmation_match": (
            report["source_artifact_sha256"] == config["source"]["artifact_sha256"]
            and report["source_confirmation_sha256"] == config["source"]["confirmation_sha256"]
            and report["integrity"]["source_artifact_file_sha256"]
            == config["source"]["artifact_file_sha256"]
            and report["integrity"]["source_confirmation_file_sha256"]
            == config["source"]["confirmation_file_sha256"]
        ),
        "exact_48_receipts_and_288_traces": len(receipts) == 48 and len(traces) == 288,
        "receipt_trace_and_native_self_hashes_valid": (
            receipt_hashes_valid and native_hashes_valid and trace_hashes_valid
        ),
        "all_target_neural_bindings_valid": binding_valid,
        "alpha_rename_invariance": alpha_invariant,
        "independent_success_counts_match_report": report_summaries_match,
        "independent_paired_statistics_match_report": report_paired_match,
        "authentic_significantly_beats_raw": (
            condition_successes[AUTHENTIC] > condition_successes[RAW]
            and wins > losses and p_value <= 0.05
        ),
        "zero_strict_negative_transfer_vs_raw": losses == 0,
        "authentic_strictly_beats_all_nonisomorphic_controls": all(
            condition_successes[AUTHENTIC] > condition_successes[condition]
            for condition in (PERMUTED, ENDPOINT, MARGINAL)
        ),
        "report_self_hash_status_and_original_gates_valid": (
            self_hash(report, "report_sha256")
            and report["status"] == "FRESH_FORMAL_TRANSFER_VALIDATED"
            and all(report["gates"].values())
        ),
    }
    receipt_digest = hashlib.sha256(
        "".join(sorted(row["receipt_sha256"] for row in receipts)).encode()
    ).hexdigest()
    compact_body = {
        "schema_version": "phase2-tirbench-causal-utility-compact-v1",
        "status": "PHASE2_TIRBENCH_CAUSAL_UTILITY_VALIDATED",
        "claim_boundary": report["claim_boundary"],
        "config_file_sha256": file_hash(args.config),
        "source_artifact_sha256": report["source_artifact_sha256"],
        "source_confirmation_sha256": report["source_confirmation_sha256"],
        "original_report_sha256": report["report_sha256"],
        "raw_receipts_file_sha256": file_hash(args.receipts),
        "receipt_aggregate_sha256": receipt_digest,
        "condition_successes": condition_successes,
        "authentic_vs_raw": {
            "wins": wins, "losses": losses, "ties": ties,
            "exact_two_sided_sign_test_p": p_value,
        },
        "per_task": task_rows,
    }
    compact = compact_body | {"compact_sha256": stable_hash(compact_body)}
    body = {
        "schema_version": "phase2-tirbench-causal-utility-independent-audit-v1",
        "status": (
            "PHASE2_TIRBENCH_V1_INDEPENDENT_AUDIT_PASSED"
            if all(gates.values()) else "PHASE2_TIRBENCH_V1_INDEPENDENT_AUDIT_FAILED"
        ),
        "compact_sha256": compact["compact_sha256"],
        "original_report_sha256": report["report_sha256"],
        "independent_condition_successes": condition_successes,
        "independent_authentic_vs_raw": {
            "wins": wins, "losses": losses, "ties": ties,
            "exact_two_sided_sign_test_p": p_value,
        },
        "gates": gates,
        "passed_gates": sum(bool(value) for value in gates.values()),
        "required_gates": len(gates),
    }
    audit = body | {"audit_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.compact_output.write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if audit["status"].endswith("_PASSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
