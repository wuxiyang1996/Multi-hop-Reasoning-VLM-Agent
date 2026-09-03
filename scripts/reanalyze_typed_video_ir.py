#!/usr/bin/env python3
"""Adaptation-only BIND->RELATE transfer analysis on matched video receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_structured_video_transfer as runner  # noqa: E402
from motif_transfer.typed_video_ir import (  # noqa: E402
    TYPED_VIDEO_CONDITIONS, evaluate_typed_bind_relate_transfer,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--source-summary", type=Path,
        default=REPO / "docs/results/typed_multisource_v4_summary.json",
    )
    parser.add_argument("--minimum-contrasts", type=int, default=2)
    parser.add_argument("--minimum-edge-traversals", type=int, default=2)
    args = parser.parse_args()
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    if source.get("status") != "SOURCE_TYPED_GATE_PASSED":
        raise SystemExit("typed source gate is not passed")
    if source["edge_replication_gate"]["status"] != "EDGE_REPLICATION_GATE_PASSED":
        raise SystemExit("BIND->RELATE edge replication gate is not passed")
    if source["effect_value_gate"]["status"] != "EFFECT_VALUE_GATE_PASSED":
        raise SystemExit("RELATE effect value gate is not passed")

    receipt_path = args.run_dir / "receipts.json"
    raw_rows = json.loads(receipt_path.read_text(encoding="utf-8"))
    evaluated = []
    for row in raw_rows:
        world_model, receipts = runner._rehydrate(row)
        result = evaluate_typed_bind_relate_transfer(
            sample_id=str(row["sample_id"]),
            gold_answer=str(row["gold_answer"]),
            world_model=world_model,
            probe_receipts=receipts,
        )
        result["family"] = str(row["family"])
        evaluated.append(result)
    count = len(evaluated)
    conditions = {
        condition: {
            "correct": sum(bool(row["conditions"][condition]["correct"]) for row in evaluated),
            "accuracy": sum(bool(row["conditions"][condition]["correct"]) for row in evaluated) / count,
        }
        for condition in TYPED_VIDEO_CONDITIONS
    }
    baseline = sum(bool(row["baseline_correct"]) for row in evaluated)
    oracle = sum(bool(row["oracle_correct"]) for row in evaluated)
    contrast = sum(bool(row["authentic_action_contrast"]) for row in evaluated)
    guards = sum(bool(row["authentic_guard_obeyed"]) for row in evaluated)
    traversals = sum(bool(row["authentic_edge_traversed"]) for row in evaluated)
    authentic = conditions["authentic_bind_relate_ir"]["correct"]
    gates = {
        "source_edge_replication_passed": True,
        "source_relate_value_passed": True,
        "all_receipts_complete": count > 0,
        "complete_native_answer_coverage": all(
            row["gold_answer"] in row["answer_space"] for row in evaluated
        ),
        "oracle_probe_headroom": oracle > baseline,
        "authentic_guard_obeyed_all_samples": guards == count,
        "authentic_edge_traversal_nontrivial": traversals >= args.minimum_edge_traversals,
        "authentic_action_contrast": contrast >= args.minimum_contrasts,
        "authentic_above_target_only": authentic > conditions[
            "target_native_expected_accuracy"
        ]["correct"],
        "authentic_above_reversed_edge": authentic > conditions[
            "reversed_relate_bind_ir"
        ]["correct"],
        "authentic_above_wrong_guard": authentic > conditions[
            "wrong_guard_bind_relate_ir"
        ]["correct"],
        "authentic_above_node_only": authentic > conditions[
            "node_only_bind_bind_ir"
        ]["correct"],
    }
    report = {
        "schema_version": 1,
        "benchmark": str(raw_rows[0]["benchmark"]),
        "status": "ADAPTATION_TYPED_IR_PASS" if all(gates.values()) else "ADAPTATION_TYPED_IR_FAIL",
        "source": {
            "transferred_edge": "BIND --[CARRIER_BOUND]--> RELATE",
            "summary_path": str(args.source_summary.resolve()),
            "summary_sha256": _sha256(args.source_summary),
            "ir_sha256": source["effect_ir"]["ir_sha256"],
            "supporting_tasks": source["edge_replication_gate"]["supporting_source_tasks"],
            "supporting_simulator_families": source["edge_replication_gate"]["supporting_simulator_families"],
        },
        "target_compilation": {
            "BIND": ["OBJECT_PRESENT", "OBJECT_ATTRIBUTE", "OBJECT_TRACK"],
            "RELATE": ["OBJECT_MOTION", "COLLISION", "ENTRY", "EXIT", "EVENT_ORDER", "CAUSAL_ANCESTOR"],
            "guard": "EXACT_TARGET_NATIVE_ENTITY_REF_INTERSECTION",
            "within_node_ranking": "TARGET_NATIVE_EXPECTED_MAP_CONFIDENCE_GAIN",
            "raw_source_or_target_tokens_shared": False,
        },
        "samples": count,
        "baseline": {"correct": baseline, "accuracy": baseline / count},
        "oracle": {"correct": oracle, "accuracy": oracle / count},
        "authentic_action_contrasts": contrast,
        "authentic_guard_obeyed": guards,
        "authentic_edge_traversals": traversals,
        "conditions": conditions,
        "gates": gates,
        "rows": evaluated,
        "claim_boundary": "Adaptation-only typed source-edge preflight; qualification and held-out outcomes remain unread.",
        "receipts": {"path": str(receipt_path.resolve()), "sha256": _sha256(receipt_path)},
    }
    output_path = args.run_dir / "adaptation_typed_bind_relate_report.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "benchmark": report["benchmark"], "status": report["status"],
        "baseline": report["baseline"], "oracle": report["oracle"],
        "conditions": report["conditions"], "gates": report["gates"],
        "report": str(output_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
