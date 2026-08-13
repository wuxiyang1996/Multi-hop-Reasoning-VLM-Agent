#!/usr/bin/env python3
"""Strict adaptation report for candidate BIND->MUTATE video programs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.candidate_mutation_video_ir import (  # noqa: E402
    CANDIDATE_MUTATION_CONDITIONS, evaluate_candidate_mutation_program,
)


def _baseline(source):
    return str(max(source["world_model"]["particles"], key=lambda row: (float(row["prior_weight"]), str(row["native_answer"])))["native_answer"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--forks-file", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--guard-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-action-delta", type=float, default=0.05)
    args = parser.parse_args()
    sources = {str(row["sample_id"]): row for row in json.loads((args.run_dir / "receipts.json").read_text(encoding="utf-8"))}
    forks = json.loads((args.run_dir / args.forks_file).read_text(encoding="utf-8"))
    if not forks or not all(bool(row.get("complete")) for row in forks):
        raise ValueError("mutation forks must be nonempty and complete")
    rows = []
    for fork in forks:
        source = sources[str(fork["sample_id"])]
        rows.append(evaluate_candidate_mutation_program(
            sample_id=str(fork["sample_id"]), gold_answer=str(source["gold_answer"]),
            baseline_answer=_baseline(source), fork=fork,
            threshold=args.threshold, guard_threshold=args.guard_threshold,
            minimum_action_delta=args.minimum_action_delta,
        ))
    count = len(rows)
    metrics = {name: {"correct": sum(bool(row["conditions"][name]["correct"]) for row in rows), "accuracy": sum(bool(row["conditions"][name]["correct"]) for row in rows) / count} for name in CANDIDATE_MUTATION_CONDITIONS}
    baseline = sum(bool(row["baseline_correct"]) for row in rows)
    oracle = sum(bool(row["oracle_correct"]) for row in rows)
    authentic_name = "authentic_bound_mutation_program"
    target_name = "target_unbound_mutation_verification"
    controls = [name for name in CANDIDATE_MUTATION_CONDITIONS if name not in {authentic_name, target_name}]
    authentic = metrics[authentic_name]["correct"]
    source_hashes = {str(row.get("source_gate_sha256") or "") for row in forks}
    distinct_wrong = sum(
        int(row["distinct_wrong_control_candidates"]) for row in rows
    )
    candidate_count = sum(len(row["slots"]) for row in rows)
    distinct_shuffled = sum(
        int(row["distinct_shuffled_control_candidates"]) for row in rows
    )
    gates = {
        "source_gate_receipts_match": len(source_hashes) == 1 and "" not in source_hashes,
        "runtime_gold_and_official_program_sealed": all(not row["compiler_saw_gold_or_official_program"] and not row["mutation_grounders_saw_full_question_option_set_or_gold"] for row in forks),
        "matched_forks_complete": len(rows) == len(forks),
        "oracle_headroom_over_strongest_nonoracle": oracle > max(baseline, *(value["correct"] for value in metrics.values())),
        "bound_changes_mutation_measurement": sum(row["bound_unbound_changed_candidates"] for row in rows) > 0,
        "authentic_action_contrast": any(row["authentic_action_contrast"] for row in rows),
        "wrong_control_executes_distinct_observation": (
            all(row["wrong_control_action_contrast"] for row in rows)
            and distinct_wrong / candidate_count >= 0.9
        ),
        "shuffled_control_executes_distinct_observation": (
            all(row["shuffled_control_action_contrast"] for row in rows)
            and distinct_shuffled / candidate_count >= 0.9
        ),
        "authentic_above_baseline": authentic > baseline,
        "authentic_above_target_unbound": authentic > metrics[target_name]["correct"],
        "authentic_above_all_edge_controls": all(authentic > metrics[name]["correct"] for name in controls),
    }
    family_metrics = {}
    for family in sorted({str(sources[row["sample_id"]]["family"]) for row in rows}):
        subset = [row for row in rows if str(sources[row["sample_id"]]["family"]) == family]
        family_metrics[family] = {"samples": len(subset), "baseline_correct": sum(bool(row["baseline_correct"]) for row in subset), **{name: sum(bool(row["conditions"][name]["correct"]) for row in subset) for name in CANDIDATE_MUTATION_CONDITIONS}}
    report = {
        "schema_version": 1, "benchmark": str(forks[0]["benchmark"]),
        "split": "adaptation",
        "status": "CANDIDATE_MUTATION_ADAPTATION_PASS" if all(gates.values()) else "CANDIDATE_MUTATION_ADAPTATION_FAIL",
        "samples": count, "baseline": {"correct": baseline, "accuracy": baseline / count},
        "oracle": {"correct": oracle, "accuracy": oracle / count},
        "conditions": metrics, "family_metrics": family_metrics,
        "distinct_wrong_control_candidates": distinct_wrong,
        "candidate_count": candidate_count,
        "distinct_wrong_control_candidate_fraction": distinct_wrong / candidate_count,
        "distinct_shuffled_control_candidates": distinct_shuffled,
        "distinct_shuffled_control_candidate_fraction": (
            distinct_shuffled / candidate_count
        ),
        "gates": gates, "rows": rows,
        "claim_boundary": "Adaptation-only source-validated BIND->MUTATE experiment; qualification remains sealed.",
    }
    output = args.run_dir / f"{Path(args.forks_file).stem}_adaptation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "baseline": report["baseline"], "oracle": report["oracle"], "conditions": metrics, "family_metrics": family_metrics, "gates": gates, "report": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
