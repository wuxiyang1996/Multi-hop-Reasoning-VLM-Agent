#!/usr/bin/env python3
"""Five-arm Layer-B evaluation with shared three-valued visual evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from motif_transfer.agqa_layer_b_epistemic import (
    AtomicVisualClaim, AtomicVisualClaimDecision, AtomicVisualClaimReceipt,
    source_open_world_commit,
)
from motif_transfer.agqa_layer_b_executor import execute_layer_b_semantics
from motif_transfer.agqa_layer_b_harness import ARMS, plan_harness_arm
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import (
    _gold_rows, _grounding, _matches, _mcnemar, _semantic,
)


def _claim_receipt(raw: dict) -> AtomicVisualClaimReceipt:
    value = AtomicVisualClaimReceipt(
        **{
            **raw,
            "claims": tuple(AtomicVisualClaim(**row) for row in raw["claims"]),
            "decisions": tuple(AtomicVisualClaimDecision(
                **{
                    **row,
                    "evidence_frame_indices": tuple(row["evidence_frame_indices"]),
                    "evidence_frame_sha256s": tuple(row["evidence_frame_sha256s"]),
                }
            ) for row in raw["decisions"]),
        }
    )
    value.validate()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/test_balanced.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("consumed_diagnostic", "qualification", "formal"), required=True,
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("epistemic five-arm evaluation is immutable")
    cohort = json.loads(args.cohort.read_text())
    grounding_report = json.loads(args.grounding.read_text())
    claim_report = json.loads(args.claims.read_text())
    fallback_report = json.loads(args.fallback.read_text())
    source = json.loads(args.source_capabilities.read_text())
    if grounding_report["status"] != "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("grounding was not frozen")
    if claim_report["status"] != "ATOMIC_VISUAL_CLAIMS_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("atomic claims were not frozen")
    if fallback_report["status"] != "SHARED_FALLBACK_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("fallback was not frozen")
    if len({cohort["cohort_sha256"], grounding_report["cohort_sha256"],
            claim_report["cohort_sha256"], fallback_report["cohort_sha256"]}) != 1:
        raise ValueError("epistemic five-arm inputs refer to different cohorts")
    if claim_report["base_grounding_report_sha256"] != grounding_report["report_sha256"]:
        raise ValueError("claims do not bind the frozen shared grounding")
    if fallback_report["grounding_report_sha256"] != grounding_report["report_sha256"]:
        raise ValueError("fallback does not bind the frozen shared grounding")
    if not all(report["all_harness_arms_share_exact_receipts"] for report in (
        grounding_report, claim_report,
    )) or not fallback_report["shared_by_all_five_arms"]:
        raise ValueError("matched-arm invariant was not frozen")
    forbidden = ("answer_read", "official_scene_graph_read", "functional_program_read",
                 "source_controller_read")
    if any(report.get(key) for report in (grounding_report, claim_report, fallback_report)
           for key in forbidden):
        raise ValueError("runtime artifact crossed an authority boundary")

    grounding_by_task = {str(row["task_id"]): row for row in grounding_report["rows"]}
    claims_by_task = {
        str(row["task_id"]): _claim_receipt(row["claim_receipt"])
        for row in claim_report["rows"]
    }
    wanted = {str(row["task_id"]) for row in cohort["rows"]}
    if set(grounding_by_task) != wanted or set(claims_by_task) != wanted:
        raise ValueError("grounding/claim reports must cover the entire cohort")
    fallback = {str(row["task_id"]): str(row["prediction"]) for row in fallback_report["rows"]}
    semantic_runtime = json.loads((args.cohort.parent / "semantic_runtime.json").read_text())
    compact_by_task = {
        str(row["task_id"]): str(row["predicted_semantics"])
        for row in semantic_runtime["rows"]
    }
    evaluator = _gold_rows(args.archive, args.entry, wanted)
    all_ops = tuple(str(value) for value in source["authorized_operators"])
    source_edges = tuple(tuple(edge) for edge in source["authorized_compositions"])
    rows = []
    for public_row in cohort["rows"]:
        task_id = str(public_row["task_id"]); raw = grounding_by_task[task_id]
        semantic = _semantic(raw["semantic_receipt"]); grounding = _grounding(raw["grounding_receipt"])
        evidence = claims_by_task[task_id]
        if evidence.semantic_receipt_sha256 != semantic.receipt_sha256 or (
            evidence.raw_event_graph_receipt_sha256 != grounding.receipt_sha256
        ):
            raise ValueError(f"{task_id}: atomic evidence receipt binding mismatch")
        plans = {arm: plan_harness_arm(
            semantic, arm=arm, source_capabilities=source, all_vm_operators=all_ops,
        ) for arm in ARMS}
        strict = execute_layer_b_semantics(
            compact_semantics=compact_by_task[task_id], grounding=grounding, semantic=semantic,
            authorized_operators=all_ops, authorized_compositions=source_edges,
            ambiguity_policy="STRICT",
        )
        eager = execute_layer_b_semantics(
            compact_semantics=compact_by_task[task_id], grounding=grounding, semantic=semantic,
            authorized_operators=all_ops, authorized_compositions=None,
            ambiguity_policy="EAGER",
        )
        source_safe, source_reason = source_open_world_commit(
            required_operators=plans["source_induced"].required_operators,
            symbolic_status=strict.receipt.status,
            symbolic_prediction=strict.receipt.prediction,
            evidence=evidence,
        )
        predictions = {
            "neural_only": fallback[task_id],
            "source_permuted": fallback[task_id],
            "generic_scaffold": (
                str(eager.receipt.prediction)
                if plans["generic_scaffold"].status == "PLANNED" and eager.receipt.status == "COMMITTED"
                else fallback[task_id]
            ),
            "source_induced": str(strict.receipt.prediction) if source_safe else fallback[task_id],
            "target_written_isomorphic": str(strict.receipt.prediction) if source_safe else fallback[task_id],
        }
        gold = str(evaluator[task_id]["answer"])
        rows.append({
            "task_id": task_id, "video_id": raw["video_id"],
            "gold_answer_evaluator_only": gold, "fallback_prediction": fallback[task_id],
            "strict_symbolic_execution": asdict(strict.receipt),
            "generic_eager_execution": asdict(eager.receipt),
            "atomic_claim_receipt_sha256": evidence.receipt_sha256,
            "atomic_claim_statuses": [row.status for row in evidence.decisions],
            "source_open_world_commit": source_safe,
            "source_open_world_reason": source_reason,
            "plans": {arm: asdict(plan) for arm, plan in plans.items()},
            "predictions": predictions,
            "correct": {arm: _matches(value, gold) for arm, value in predictions.items()},
        })
    correct = {arm: [row["correct"][arm] for row in rows] for arm in ARMS}
    n = len(rows)
    summaries = {arm: {
        "correct": sum(correct[arm]), "total": n, "accuracy": sum(correct[arm]) / n,
        "symbolic_commits": sum(
            row["source_open_world_commit"] if arm in {"source_induced", "target_written_isomorphic"}
            else arm == "generic_scaffold" and row["generic_eager_execution"]["status"] == "COMMITTED"
            for row in rows
        ),
    } for arm in ARMS}
    comparisons = {baseline: _mcnemar(correct["source_induced"], correct[baseline])
                   for baseline in ("neural_only", "generic_scaffold", "source_permuted")}
    gates = {
        "source_beats_neural": summaries["source_induced"]["correct"] > summaries["neural_only"]["correct"],
        "source_beats_generic": summaries["source_induced"]["correct"] > summaries["generic_scaffold"]["correct"],
        "source_vs_neural_significant": comparisons["neural_only"]["exact_two_sided_p"] < .05,
        "negative_transfer_losses_at_most_five_percent": comparisons["neural_only"]["losses"] <= math.floor(.05*n),
        "source_permuted_not_better_than_source": summaries["source_permuted"]["correct"] <= summaries["source_induced"]["correct"],
        "target_written_isomorphic_action_equivalence": all(
            row["predictions"]["source_induced"] == row["predictions"]["target_written_isomorphic"]
            for row in rows
        ),
    }
    body = {
        "schema_version": "agqa-layer-b-epistemic-five-arm-evaluation-v2",
        "stage": args.stage,
        "status": "LAYER_B_GATES_PASSED" if all(gates.values()) else "LAYER_B_GATES_FAILED",
        "cohort_sha256": cohort["cohort_sha256"],
        "grounding_report_sha256": grounding_report["report_sha256"],
        "claim_report_sha256": claim_report["report_sha256"],
        "fallback_report_sha256": fallback_report["report_sha256"],
        "source_capability_sha256": source["artifact_sha256"],
        "rows": rows, "summaries": summaries, "comparisons": comparisons, "gates": gates,
        "frames_grounder_parser_executor_fallback_shared": True,
        "only_symbolic_harness_differs": True,
        "raw_video_end_to_end_only": True,
        "official_scene_graph_used_at_runtime": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": body["status"], "summaries": summaries,
                      "comparisons": comparisons, "gates": gates,
                      "report_sha256": body["report_sha256"]}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
