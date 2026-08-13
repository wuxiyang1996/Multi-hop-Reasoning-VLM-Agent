#!/usr/bin/env python3
"""Evaluate a frozen family applicability policy on qualification receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.candidate_claim_video_ir import (  # noqa: E402
    CANDIDATE_CLAIM_CONDITIONS, evaluate_candidate_claim_program,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _baseline(source):
    return str(max(
        source["world_model"]["particles"],
        key=lambda row: (float(row["prior_weight"]), str(row["native_answer"])),
    )["native_answer"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--forks-file", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--qualification-config", type=Path, required=True)
    args = parser.parse_args()
    if _sha256(args.policy) != args.policy_sha256:
        raise ValueError("frozen policy hash mismatch")
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if policy["status"] != "FROZEN_BEFORE_QUALIFICATION_COLLECTION":
        raise ValueError("policy was not frozen before qualification")
    qualification = json.loads(args.qualification_config.read_text(encoding="utf-8"))
    if qualification.get("status") != "FROZEN_PROSPECTIVE_QUALIFICATION":
        raise ValueError("qualification config is not prospectively frozen")
    if qualification["frozen_policy"] != {
        "path": str(args.policy.resolve()), "sha256": args.policy_sha256,
    }:
        raise ValueError("qualification config does not bind the supplied policy")
    manifest_path = Path(qualification["split_manifest"])
    if _sha256(manifest_path) != qualification["split_manifest_sha256"]:
        raise ValueError("qualification split manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = list(manifest["benchmarks"][policy["benchmark"]]["splits"]["qualification"])
    source_rows = json.loads((args.run_dir / "receipts.json").read_text(encoding="utf-8"))
    if [str(row["sample_id"]) for row in source_rows] != expected_ids:
        raise ValueError("base receipts do not exactly match frozen qualification order")
    if not all(
        row.get("split") == "qualification"
        and row.get("benchmark") == policy["benchmark"]
        and not bool(row.get("runtime_oracle_inputs"))
        for row in source_rows
    ):
        raise ValueError("base receipt qualification/oracle boundary mismatch")
    sources = {str(row["sample_id"]): row for row in source_rows}
    forks = json.loads((args.run_dir / args.forks_file).read_text(encoding="utf-8"))
    if not forks or not all(bool(row.get("complete")) for row in forks):
        raise ValueError("qualification forks are incomplete")
    if [str(row["sample_id"]) for row in forks] != expected_ids:
        raise ValueError("forks do not exactly match frozen qualification order")
    rows = []
    for fork in forks:
        source = sources[str(fork["sample_id"])]
        raw = evaluate_candidate_claim_program(
            sample_id=str(fork["sample_id"]), gold_answer=str(source["gold_answer"]),
            baseline_answer=_baseline(source), fork=fork,
        )
        family = str(source["family"])
        conditions = {}
        for condition in CANDIDATE_CLAIM_CONDITIONS:
            frozen = policy["conditions"][condition][family]
            use = bool(frozen["use_intervention"])
            conditions[condition] = {
                "use_intervention": use,
                "committed_answer": (
                    raw["conditions"][condition]["committed_answer"]
                    if use else raw["baseline_answer"]
                ),
                "correct": (
                    bool(raw["conditions"][condition]["correct"])
                    if use else bool(raw["baseline_correct"])
                ),
            }
        rows.append({
            "sample_id": raw["sample_id"], "family": family,
            "gold_answer": raw["gold_answer"],
            "baseline_correct": raw["baseline_correct"],
            "conditions": conditions,
        })
    count = len(rows)
    baseline = sum(bool(row["baseline_correct"]) for row in rows)
    metrics = {
        condition: {
            "correct": sum(bool(row["conditions"][condition]["correct"]) for row in rows),
            "accuracy": sum(bool(row["conditions"][condition]["correct"]) for row in rows) / count,
            "interventions": sum(bool(row["conditions"][condition]["use_intervention"]) for row in rows),
        }
        for condition in CANDIDATE_CLAIM_CONDITIONS
    }
    authentic = metrics["authentic_bound_claim_program"]["correct"]
    controls = tuple(
        condition for condition in CANDIDATE_CLAIM_CONDITIONS
        if condition != "authentic_bound_claim_program"
    )
    gates = {
        "policy_hash_match": True,
        "qualification_complete": count == len(expected_ids),
        "authentic_intervenes": metrics["authentic_bound_claim_program"]["interventions"] > 0,
        "authentic_above_baseline": authentic > baseline,
        "authentic_above_target": authentic > metrics["target_unbound_claim_verification"]["correct"],
        "authentic_above_all_edge_controls": all(
            authentic > metrics[condition]["correct"] for condition in controls[1:]
        ),
    }
    report = {
        "schema_version": 1, "benchmark": policy["benchmark"],
        "split": "qualification",
        "status": "QUALIFICATION_TRANSFER_PASS" if all(gates.values()) else "QUALIFICATION_TRANSFER_FAIL",
        "samples": count,
        "baseline": {"correct": baseline, "accuracy": baseline / count},
        "conditions": metrics, "gates": gates, "rows": rows,
        "frozen_policy_sha256": args.policy_sha256,
        "qualification_config_sha256": _sha256(args.qualification_config),
        "split_manifest_sha256": qualification["split_manifest_sha256"],
        "heldout_touched": False,
    }
    path = args.run_dir / "candidate_claim_v6_qualification_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "baseline": report["baseline"],
        "conditions": metrics, "gates": gates, "report": str(path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
