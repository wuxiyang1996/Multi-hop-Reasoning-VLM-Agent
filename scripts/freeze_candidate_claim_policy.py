#!/usr/bin/env python3
"""Freeze family applicability from a completed adaptation report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptation-report", type=Path, required=True)
    parser.add_argument("--selective-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adaptation = json.loads(args.adaptation_report.read_text(encoding="utf-8"))
    selective = json.loads(args.selective_report.read_text(encoding="utf-8"))
    if adaptation["status"] != "CANDIDATE_CLAIM_ADAPTATION_FAIL":
        # The raw policy can fail because it lacks selective fallback; this exact
        # state is expected before freezing the utility evaluator.
        raise ValueError("unexpected raw adaptation status")
    if selective["status"] != "SELECTIVE_TRANSFER_ADAPTATION_PASS":
        raise ValueError("selective adaptation gate did not pass")
    rows = adaptation["rows"]
    receipts = {
        str(row["sample_id"]): str(row["family"])
        for row in json.loads((args.adaptation_report.parent / "receipts.json").read_text(encoding="utf-8"))
    }
    conditions = tuple(adaptation["conditions"])
    policy = {}
    for condition in conditions:
        family_rows = {}
        for row in rows:
            family_rows.setdefault(receipts[row["sample_id"]], []).append(row)
        policy[condition] = {}
        for family, subset in sorted(family_rows.items()):
            delta = sum(
                float(row["conditions"][condition]["correct"])
                - float(row["baseline_correct"])
                for row in subset
            ) / len(subset)
            policy[condition][family] = {
                "adaptation_samples": len(subset),
                "mean_success_delta": delta,
                "use_intervention": delta > 0.0,
            }
    output = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_QUALIFICATION_COLLECTION",
        "benchmark": adaptation["benchmark"],
        "rule": "USE_CONDITION_IFF_ALL_ADAPTATION_ROWS_IN_SAME_FAMILY_HAVE_MEAN_SUCCESS_DELTA_STRICTLY_ABOVE_ZERO_ELSE_BASELINE",
        "threshold": 0.0,
        "conditions": policy,
        "lineage": {
            "adaptation_report": str(args.adaptation_report.resolve()),
            "adaptation_report_sha256": _sha256(args.adaptation_report),
            "selective_report": str(args.selective_report.resolve()),
            "selective_report_sha256": _sha256(args.selective_report),
        },
        "qualification_outcomes_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": output["status"], "benchmark": output["benchmark"],
        "conditions": policy, "output": str(args.output.resolve()),
        "sha256": _sha256(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
