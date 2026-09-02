#!/usr/bin/env python3
"""Preregistered Layer-B risk-constrained five-arm replication evaluator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from motif_transfer.agqa_layer_b_epistemic import source_root_open_world_commit
from motif_transfer.agqa_layer_b_executor import execute_layer_b_semantics
from motif_transfer.agqa_layer_b_harness import (
    ARMS, plan_harness_arm, source_permuted_compositions,
)
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_epistemic_five_arm import _claim_receipt
from scripts.evaluate_agqa_layer_b_five_arm import (
    _gold_rows, _grounding, _matches, _mcnemar, _semantic,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--pre-outcome-receipt", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("risk-constrained five-arm evaluation is immutable")

    prereg = _load(args.preregistration); pre = _load(args.pre_outcome_receipt)
    cohort = _load(args.cohort); runtime = _load(args.semantic_runtime)
    grounding_report = _load(args.grounding); claim_report = _load(args.claims)
    fallback_report = _load(args.fallback); source = _load(args.source_capabilities)
    if prereg["status"] != "FROZEN_BEFORE_ANY_REPLICATION_RUNTIME_OR_OUTCOME":
        raise ValueError("invalid replication preregistration status")
    if pre["status"] != "ALL_RUNTIME_ARTIFACTS_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("pre-outcome gate did not authorize evaluation")
    if prereg["cohort"]["public_cohort_sha256"] != cohort["cohort_sha256"]:
        raise ValueError("preregistered cohort mismatch")
    if pre["cohort_sha256"] != cohort["cohort_sha256"]:
        raise ValueError("pre-outcome cohort mismatch")
    reports = (runtime, grounding_report, claim_report, fallback_report)
    if len({row["cohort_sha256"] for row in reports} | {cohort["cohort_sha256"]}) != 1:
        raise ValueError("runtime artifacts refer to different cohorts")
    if grounding_report["status"] != "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("grounding was not frozen")
    if claim_report["status"] != "ATOMIC_VISUAL_CLAIMS_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("claims were not frozen")
    if fallback_report["status"] != "SHARED_FALLBACK_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("fallback was not frozen")
    if claim_report["base_grounding_report_sha256"] != grounding_report["report_sha256"]:
        raise ValueError("claims do not bind grounding")
    if fallback_report["grounding_report_sha256"] != grounding_report["report_sha256"]:
        raise ValueError("fallback does not bind grounding")
    if not grounding_report["all_harness_arms_share_exact_receipts"]:
        raise ValueError("grounding is not shared")
    if not claim_report["all_harness_arms_share_exact_receipts"]:
        raise ValueError("claims are not shared")
    if not fallback_report["shared_by_all_five_arms"]:
        raise ValueError("fallback is not shared")

    wanted = {str(row["task_id"]) for row in cohort["rows"]}
    semantics = {str(row["task_id"]): str(row["predicted_semantics"])
                 for row in runtime["rows"]}
    grounding_by_task = {str(row["task_id"]): row for row in grounding_report["rows"]}
    claims_by_task = {str(row["task_id"]): _claim_receipt(row["claim_receipt"])
                      for row in claim_report["rows"]}
    fallback = {str(row["task_id"]): str(row["prediction"])
                for row in fallback_report["rows"]}
    if any(set(value) != wanted for value in (
        semantics, grounding_by_task, claims_by_task, fallback,
    )):
        raise ValueError("every shared runtime artifact must cover exactly the cohort")

    evaluator = _gold_rows(args.archive, args.entry, wanted)
    source_ops = tuple(str(value) for value in source["authorized_operators"])
    source_edges = tuple(tuple(edge) for edge in source["authorized_compositions"])
    permuted_edges = source_permuted_compositions(source_ops, source_edges)
    matched_permutation = (
        len(set(permuted_edges)) == len(set(source_edges))
        and set(permuted_edges) != set(source_edges)
        and {value for edge in permuted_edges for value in edge} <= set(source_ops)
    )

    rows = []
    for public in cohort["rows"]:
        task_id = str(public["task_id"]); raw = grounding_by_task[task_id]
        semantic = _semantic(raw["semantic_receipt"])
        graph = _grounding(raw["grounding_receipt"])
        evidence = claims_by_task[task_id]
        if evidence.semantic_receipt_sha256 != semantic.receipt_sha256:
            raise ValueError(f"{task_id}: claim/semantic binding mismatch")
        if evidence.raw_event_graph_receipt_sha256 != graph.receipt_sha256:
            raise ValueError(f"{task_id}: claim/grounding binding mismatch")
        plans = {arm: plan_harness_arm(
            semantic, arm=arm, source_capabilities=source,
            all_vm_operators=source_ops,
        ) for arm in ARMS}
        strict = execute_layer_b_semantics(
            compact_semantics=semantics[task_id], grounding=graph, semantic=semantic,
            authorized_operators=source_ops, authorized_compositions=source_edges,
            ambiguity_policy="STRICT",
        )
        permuted = execute_layer_b_semantics(
            compact_semantics=semantics[task_id], grounding=graph, semantic=semantic,
            authorized_operators=source_ops, authorized_compositions=permuted_edges,
            ambiguity_policy="STRICT",
        )
        eager = execute_layer_b_semantics(
            compact_semantics=semantics[task_id], grounding=graph, semantic=semantic,
            authorized_operators=source_ops, authorized_compositions=None,
            ambiguity_policy="EAGER",
        )
        source_guard, source_reason = source_root_open_world_commit(
            semantic=semantic, symbolic_status=strict.receipt.status,
            symbolic_prediction=strict.receipt.prediction, evidence=evidence,
        )
        permuted_guard, permuted_reason = source_root_open_world_commit(
            semantic=semantic, symbolic_status=permuted.receipt.status,
            symbolic_prediction=permuted.receipt.prediction, evidence=evidence,
        )
        source_commit = plans["source_induced"].status == "PLANNED" and source_guard
        permuted_commit = plans["source_permuted"].status == "PLANNED" and permuted_guard
        generic_commit = plans["generic_scaffold"].status == "PLANNED" and eager.receipt.status == "COMMITTED"
        predictions = {
            "neural_only": fallback[task_id],
            "generic_scaffold": str(eager.receipt.prediction) if generic_commit else fallback[task_id],
            "source_permuted": str(permuted.receipt.prediction) if permuted_commit else fallback[task_id],
            "source_induced": str(strict.receipt.prediction) if source_commit else fallback[task_id],
            "target_written_isomorphic": str(strict.receipt.prediction) if source_commit else fallback[task_id],
        }
        gold = str(evaluator[task_id]["answer"])
        rows.append({
            "task_id": task_id, "video_id": str(public["video_id"]),
            "gold_answer_evaluator_only": gold,
            "plans": {arm: asdict(plan) for arm, plan in plans.items()},
            "source_execution": asdict(strict.receipt),
            "permuted_execution": asdict(permuted.receipt),
            "generic_execution": asdict(eager.receipt),
            "source_commit": source_commit, "source_commit_reason": source_reason,
            "source_permuted_commit": permuted_commit,
            "source_permuted_commit_reason": permuted_reason,
            "generic_commit": generic_commit,
            "predictions": predictions,
            "correct": {arm: _matches(prediction, gold)
                        for arm, prediction in predictions.items()},
        })

    correct = {arm: [row["correct"][arm] for row in rows] for arm in ARMS}
    n = len(rows)
    summaries = {arm: {
        "correct": sum(correct[arm]), "total": n,
        "accuracy": sum(correct[arm]) / n,
        "symbolic_commits": sum(
            row["source_commit"] if arm in {"source_induced", "target_written_isomorphic"}
            else row["source_permuted_commit"] if arm == "source_permuted"
            else row["generic_commit"] if arm == "generic_scaffold" else 0
            for row in rows
        ),
    } for arm in ARMS}
    comparisons = {
        baseline: _mcnemar(correct["source_induced"], correct[baseline])
        for baseline in ("neural_only", "generic_scaffold", "source_permuted")
    }
    versus_neural = {
        arm: _mcnemar(correct[arm], correct["neural_only"])
        for arm in ("generic_scaffold", "source_permuted", "source_induced")
    }
    max_losses = math.floor(float(prereg["gates"]["negative_transfer_fraction_at_most"]) * n)
    feasible_symbolic = [
        arm for arm in ("generic_scaffold", "source_permuted", "source_induced")
        if versus_neural[arm]["losses"] <= max_losses
    ]
    source_best_feasible = all(
        summaries["source_induced"]["correct"] >= summaries[arm]["correct"]
        for arm in feasible_symbolic
    )
    gates = {
        "source_beats_neural": summaries["source_induced"]["correct"] > summaries["neural_only"]["correct"],
        "source_vs_neural_significant": comparisons["neural_only"]["exact_two_sided_p"] < .05,
        "source_negative_transfer_bounded": comparisons["neural_only"]["losses"] <= max_losses,
        "source_beats_matched_permuted": summaries["source_induced"]["correct"] > summaries["source_permuted"]["correct"],
        "source_vs_matched_permuted_significant": comparisons["source_permuted"]["exact_two_sided_p"] < .05,
        "source_is_best_feasible_symbolic_arm": source_best_feasible,
        "permuted_preserves_inventory_and_edge_count_but_changes_lineage": matched_permutation,
        "target_written_isomorphic_action_equivalence": all(
            row["predictions"]["source_induced"] == row["predictions"]["target_written_isomorphic"]
            for row in rows
        ),
        "pre_outcome_coverage_gate_passed": bool(pre["coverage_gate_passed"]),
    }
    body = {
        "schema_version": "agqa-layer-b-risk-constrained-five-arm-replication-v1",
        "status": "RISK_CONSTRAINED_LAYER_B_GATES_PASSED" if all(gates.values())
                  else "RISK_CONSTRAINED_LAYER_B_GATES_FAILED",
        "claim_scope": "RISK_CONSTRAINED_RAW_VIDEO_GAME_TO_VIDEO_TRANSFER",
        "cohort_sha256": cohort["cohort_sha256"],
        "pre_outcome_receipt_sha256": pre["receipt_sha256"],
        "grounding_report_sha256": grounding_report["report_sha256"],
        "claim_report_sha256": claim_report["report_sha256"],
        "fallback_report_sha256": fallback_report["report_sha256"],
        "source_capability_sha256": source["artifact_sha256"],
        "source_permutation": {
            "operators": sorted(source_ops), "authentic_edge_count": len(set(source_edges)),
            "permuted_edge_count": len(set(permuted_edges)),
            "permuted_edges": [list(edge) for edge in permuted_edges],
        },
        "negative_transfer_max_losses": max_losses,
        "feasible_symbolic_arms": feasible_symbolic,
        "rows": rows, "summaries": summaries,
        "comparisons": comparisons, "versus_neural": versus_neural,
        "gates": gates,
        "frames_grounder_parser_executor_fallback_shared": True,
        "only_symbolic_harness_differs": True,
        "raw_video_end_to_end_only": True,
        "official_scene_graph_used_at_runtime": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "summaries": summaries,
        "comparisons": comparisons, "versus_neural": versus_neural,
        "feasible_symbolic_arms": feasible_symbolic, "gates": gates,
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
