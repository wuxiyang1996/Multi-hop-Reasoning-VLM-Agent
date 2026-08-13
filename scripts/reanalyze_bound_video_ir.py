#!/usr/bin/env python3
"""Analyze executable BIND-handle -> RELATE matched video forks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_structured_video_transfer as runner  # noqa: E402
from motif_transfer.bound_video_ir import (  # noqa: E402
    BOUND_VIDEO_CONDITIONS, evaluate_bound_bind_relate_transfer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--forks-file", default="bound_relation_forks.json")
    args = parser.parse_args()
    raw = json.loads((args.run_dir / "receipts.json").read_text(encoding="utf-8"))
    source_by_id = {str(row["sample_id"]): row for row in raw}
    forks = json.loads((args.run_dir / args.forks_file).read_text(encoding="utf-8"))
    if not forks:
        raise ValueError("fork file is empty")
    evaluated = []
    for fork in forks:
        source = source_by_id[str(fork["sample_id"])]
        world_model, global_receipts = runner._rehydrate(source)
        row = evaluate_bound_bind_relate_transfer(
            sample_id=str(source["sample_id"]),
            gold_answer=str(source["gold_answer"]),
            world_model=world_model,
            global_receipts=global_receipts,
            fork_receipt=fork,
        )
        row["family"] = str(source["family"])
        evaluated.append(row)
    count = len(evaluated)
    conditions = {
        condition: {
            "correct": sum(bool(row["conditions"][condition]["correct"]) for row in evaluated),
            "accuracy": sum(bool(row["conditions"][condition]["correct"]) for row in evaluated) / count,
        }
        for condition in BOUND_VIDEO_CONDITIONS
    }
    authentic = conditions["authentic_bound_bind_relate_ir"]["correct"]
    baseline = sum(bool(row["baseline_correct"]) for row in evaluated)
    oracle = sum(bool(row["oracle_correct"]) for row in evaluated)
    contrasts = sum(bool(row["authentic_action_contrast"]) for row in evaluated)
    changed = sum(bool(row["conditions"]["authentic_bound_bind_relate_ir"][
        "handle_changed_relation_observation"
    ]) for row in evaluated)
    gates = {
        "matched_forks_complete": count > 0,
        "complete_native_answer_coverage": all(
            row["gold_answer"] in row["answer_space"] for row in evaluated
        ),
        "oracle_headroom": oracle > baseline,
        "authentic_guard_obeyed": all(
            bool(row["authentic_guard_obeyed"]) for row in evaluated
        ),
        "authentic_action_contrast": contrasts > 0,
        "handle_changes_observation": changed > 0,
        "authentic_above_target_exact_dp": authentic > conditions[
            "target_native_exact_dp"
        ]["correct"],
        "authentic_above_unbound_ablation": authentic > conditions[
            "authentic_unbound_relation_ablation"
        ]["correct"],
        "authentic_above_reversed": authentic > conditions[
            "reversed_relate_bind_ir"
        ]["correct"],
        "authentic_above_wrong_guard": authentic > conditions[
            "wrong_guard_bound_ir"
        ]["correct"],
        "authentic_above_node_only": authentic > conditions[
            "node_only_bind_bind_ir"
        ]["correct"],
    }
    report = {
        "schema_version": 1,
        "benchmark": str(forks[0]["benchmark"]),
        "status": "BOUND_EDGE_SMOKE_PASS" if all(gates.values()) else "BOUND_EDGE_SMOKE_FAIL",
        "samples": count,
        "baseline": {"correct": baseline, "accuracy": baseline / count},
        "oracle": {"correct": oracle, "accuracy": oracle / count},
        "authentic_action_contrasts": contrasts,
        "handle_changed_observations": changed,
        "conditions": conditions,
        "gates": gates,
        "rows": evaluated,
        "claim_boundary": "Small adaptation-only executable-handle smoke; never confirmatory.",
    }
    stem = Path(args.forks_file).stem
    output = args.run_dir / f"{stem}_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "benchmark": report["benchmark"], "status": report["status"],
        "baseline": report["baseline"], "oracle": report["oracle"],
        "conditions": conditions, "gates": gates, "report": str(output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
