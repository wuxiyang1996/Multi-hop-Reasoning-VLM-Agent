#!/usr/bin/env python3
"""Strict adaptation report for candidate-factorized video programs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.candidate_claim_video_ir import (  # noqa: E402
    CANDIDATE_CLAIM_CONDITIONS, evaluate_candidate_claim_program,
)


def _baseline(source):
    particles = source["world_model"]["particles"]
    return max(
        particles,
        key=lambda row: (float(row["prior_weight"]), str(row["native_answer"])),
    )["native_answer"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--guard-threshold", type=float, default=0.5)
    parser.add_argument("--forks-file", default="candidate_claim_forks.json")
    args = parser.parse_args()
    sources = {
        str(row["sample_id"]): row
        for row in json.loads((args.run_dir / "receipts.json").read_text(encoding="utf-8"))
    }
    forks = json.loads((args.run_dir / args.forks_file).read_text(encoding="utf-8"))
    if not forks or not all(bool(row.get("complete")) for row in forks):
        raise ValueError("candidate claim forks must be nonempty and complete")
    rows = []
    for fork in forks:
        source = sources[str(fork["sample_id"])]
        rows.append(evaluate_candidate_claim_program(
            sample_id=str(fork["sample_id"]), gold_answer=str(source["gold_answer"]),
            baseline_answer=str(_baseline(source)), fork=fork,
            threshold=args.threshold, guard_threshold=args.guard_threshold,
        ))
    count = len(rows)
    conditions = {
        name: {
            "correct": sum(bool(row["conditions"][name]["correct"]) for row in rows),
            "accuracy": sum(bool(row["conditions"][name]["correct"]) for row in rows) / count,
        }
        for name in CANDIDATE_CLAIM_CONDITIONS
    }
    baseline = sum(bool(row["baseline_correct"]) for row in rows)
    oracle = sum(bool(row["oracle_correct"]) for row in rows)
    authentic = conditions["authentic_bound_claim_program"]["correct"]
    controls = (
        "target_unbound_claim_verification", "reversed_claim_then_bind",
        "wrong_guard_bound_claim", "node_only_bind", "source_marginal_bind",
        "shuffled_bind_correspondence",
    )
    source_hashes = {str(row.get("source_gate_sha256") or "") for row in forks}
    action_contrasts = sum(bool(row["authentic_action_contrast"]) for row in rows)
    rescues = sum(
        bool(row["conditions"]["authentic_bound_claim_program"]["correct"])
        and not bool(row["conditions"]["target_unbound_claim_verification"]["correct"])
        for row in rows
    )
    harms = sum(
        not bool(row["conditions"]["authentic_bound_claim_program"]["correct"])
        and bool(row["conditions"]["target_unbound_claim_verification"]["correct"])
        for row in rows
    )
    gates = {
        "source_gate_receipts_match": len(source_hashes) == 1 and "" not in source_hashes,
        "runtime_gold_and_official_program_sealed": all(
            not row["compiler_saw_gold_or_official_program"]
            and not row["visual_grounders_saw_full_question_option_set_or_gold"]
            for row in forks
        ),
        "matched_forks_complete": len(forks) == count,
        "oracle_headroom_over_strongest_nonoracle": oracle > max(
            baseline, *(conditions[name]["correct"] for name in controls)
        ),
        "bound_changes_relation_measurement": sum(
            row["bound_unbound_changed_candidates"] for row in rows
        ) > 0,
        "authentic_action_contrast": action_contrasts > 0,
        "authentic_above_baseline": authentic > baseline,
        "authentic_above_target_unbound": authentic > conditions[
            "target_unbound_claim_verification"
        ]["correct"],
        "authentic_above_all_edge_controls": all(
            authentic > conditions[name]["correct"] for name in controls[1:]
        ),
    }
    family_metrics = {}
    for family in sorted({str(sources[row["sample_id"]]["family"]) for row in rows}):
        subset = [
            row for row in rows
            if str(sources[row["sample_id"]]["family"]) == family
        ]
        family_metrics[family] = {
            "samples": len(subset),
            "baseline_correct": sum(bool(row["baseline_correct"]) for row in subset),
            **{
                name: sum(bool(row["conditions"][name]["correct"]) for row in subset)
                for name in CANDIDATE_CLAIM_CONDITIONS
            },
        }
    report = {
        "schema_version": 1, "benchmark": str(forks[0]["benchmark"]),
        "status": "CANDIDATE_CLAIM_ADAPTATION_PASS" if all(gates.values()) else "CANDIDATE_CLAIM_ADAPTATION_FAIL",
        "samples": count, "threshold": args.threshold,
        "guard_threshold": args.guard_threshold,
        "baseline": {"correct": baseline, "accuracy": baseline / count},
        "oracle": {"correct": oracle, "accuracy": oracle / count},
        "source_gate_sha256": next(iter(source_hashes)),
        "authentic_action_contrasts": action_contrasts,
        "authentic_rescues_over_target": rescues,
        "authentic_harms_vs_target": harms,
        "conditions": conditions, "family_metrics": family_metrics,
        "gates": gates, "rows": rows,
        "claim_boundary": "Adaptation-only matched executable program; never qualification evidence.",
    }
    output = args.run_dir / f"{Path(args.forks_file).stem}_adaptation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "benchmark": report["benchmark"], "status": report["status"],
        "baseline": report["baseline"], "oracle": report["oracle"],
        "conditions": conditions, "gates": gates, "report": str(output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
