#!/usr/bin/env python3
"""LOVO family-conditional transfer utility for candidate video programs.

The sole target-native calibration statistic is the mean success delta over
other adaptation videos in the same official coarse question family.  A
condition is used iff that strictly held-out estimate is positive; otherwise
the controller fails closed to the original baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITIONS = (
    "target_unbound_claim_verification",
    "authentic_bound_claim_program",
    "reversed_claim_then_bind",
    "wrong_guard_bound_claim",
    "node_only_bind",
    "source_marginal_bind",
    "shuffled_bind_correspondence",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.report.read_text(encoding="utf-8"))
    rows = list(source["rows"])
    if not rows:
        raise ValueError("adaptation report has no rows")
    # The compact report does not duplicate family in each row. Recover it from
    # the ordered full fork/source lineage saved alongside the report.
    run_dir = args.report.parent
    receipts = {
        str(row["sample_id"]): str(row["family"])
        for row in json.loads((run_dir / "receipts.json").read_text(encoding="utf-8"))
    }
    family_by_id = {row["sample_id"]: receipts[row["sample_id"]] for row in rows}
    evaluated = []
    for held in rows:
        family = family_by_id[held["sample_id"]]
        peers = [
            row for row in rows
            if row["sample_id"] != held["sample_id"]
            and family_by_id[row["sample_id"]] == family
        ]
        if not peers:
            raise ValueError(f"family {family} lacks LOVO calibration peers")
        decisions = {}
        for condition in CONDITIONS:
            estimated_delta = sum(
                float(row["conditions"][condition]["correct"])
                - float(row["baseline_correct"])
                for row in peers
            ) / len(peers)
            use_intervention = estimated_delta > 0.0
            chosen_correct = bool(
                held["conditions"][condition]["correct"]
                if use_intervention else held["baseline_correct"]
            )
            chosen_answer = str(
                held["conditions"][condition]["committed_answer"]
                if use_intervention else held["baseline_answer"]
            )
            decisions[condition] = {
                "same_family_training_video_ids": [row["sample_id"] for row in peers],
                "estimated_success_delta": estimated_delta,
                "use_intervention": use_intervention,
                "chosen_answer": chosen_answer,
                "correct": chosen_correct,
            }
        evaluated.append({
            "sample_id": held["sample_id"], "family": family,
            "gold_answer": held["gold_answer"],
            "baseline_correct": bool(held["baseline_correct"]),
            "conditions": decisions,
        })
    count = len(evaluated)
    baseline = sum(row["baseline_correct"] for row in evaluated)
    metrics = {
        condition: {
            "correct": sum(row["conditions"][condition]["correct"] for row in evaluated),
            "accuracy": sum(row["conditions"][condition]["correct"] for row in evaluated) / count,
            "interventions": sum(row["conditions"][condition]["use_intervention"] for row in evaluated),
        }
        for condition in CONDITIONS
    }
    authentic = metrics["authentic_bound_claim_program"]["correct"]
    controls = tuple(value for value in CONDITIONS if value != "authentic_bound_claim_program")
    gates = {
        "leave_one_video_out_complete": count == source["samples"],
        "same_family_calibration_excludes_heldout": all(
            row["sample_id"] not in decision["same_family_training_video_ids"]
            for row in evaluated for decision in row["conditions"].values()
        ),
        "authentic_intervenes": metrics["authentic_bound_claim_program"]["interventions"] > 0,
        "authentic_above_baseline": authentic > baseline,
        "authentic_above_target_selector": authentic > metrics[
            "target_unbound_claim_verification"
        ]["correct"],
        "authentic_above_all_edge_control_selectors": all(
            authentic > metrics[condition]["correct"] for condition in controls[1:]
        ),
    }
    output = {
        "schema_version": 1, "benchmark": source["benchmark"],
        "status": "SELECTIVE_TRANSFER_ADAPTATION_PASS" if all(gates.values()) else "SELECTIVE_TRANSFER_ADAPTATION_FAIL",
        "protocol": "LOVO_SAME_FAMILY_MEAN_SUCCESS_DELTA_STRICTLY_POSITIVE_ELSE_BASELINE",
        "samples": count,
        "baseline": {"correct": baseline, "accuracy": baseline / count},
        "conditions": metrics, "gates": gates, "rows": evaluated,
        "input_report": str(args.report.resolve()),
        "claim_boundary": "Adaptation-only transfer-utility calibration; qualification remains sealed.",
    }
    path = args.report.with_name(args.report.stem + "_selective_report.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "benchmark": output["benchmark"], "status": output["status"],
        "baseline": output["baseline"], "conditions": metrics,
        "gates": gates, "report": str(path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
