#!/usr/bin/env python3
"""Freeze Layer-B risk-replication artifacts before reading any outcome."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_epistemic import source_root_open_world_commit
from motif_transfer.agqa_layer_b_executor import execute_layer_b_semantics
from motif_transfer.agqa_layer_b_harness import (
    plan_harness_arm, source_permuted_compositions,
)
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_epistemic_five_arm import _claim_receipt
from scripts.evaluate_agqa_layer_b_five_arm import _grounding, _semantic


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("pre-outcome receipt is immutable")
    prereg = json.loads(args.preregistration.read_text())
    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    grounding = json.loads(args.grounding.read_text())
    claims = json.loads(args.claims.read_text())
    fallback = json.loads(args.fallback.read_text())
    source = json.loads(args.source_capabilities.read_text())
    if prereg["status"] != "FROZEN_BEFORE_ANY_REPLICATION_RUNTIME_OR_OUTCOME":
        raise ValueError("invalid replication preregistration status")
    if prereg["cohort"]["public_cohort_sha256"] != cohort["cohort_sha256"]:
        raise ValueError("preregistered cohort mismatch")
    if runtime["valid"] != len(cohort["rows"]) or runtime["invalid"]:
        raise ValueError("semantic runtime is incomplete")
    if len({cohort["cohort_sha256"], runtime["cohort_sha256"], grounding["cohort_sha256"],
            claims["cohort_sha256"], fallback["cohort_sha256"]}) != 1:
        raise ValueError("runtime artifacts refer to different cohorts")
    if claims["base_grounding_report_sha256"] != grounding["report_sha256"]:
        raise ValueError("claims do not bind grounding")
    if fallback["grounding_report_sha256"] != grounding["report_sha256"]:
        raise ValueError("fallback does not bind grounding")
    if not grounding["all_harness_arms_share_exact_receipts"]:
        raise ValueError("grounding is not shared")
    if not claims["all_harness_arms_share_exact_receipts"]:
        raise ValueError("claims are not shared")
    if not fallback["shared_by_all_five_arms"]:
        raise ValueError("fallback is not shared")

    compact = {str(row["task_id"]): str(row["predicted_semantics"])
               for row in runtime["rows"]}
    evidence = {str(row["task_id"]): _claim_receipt(row["claim_receipt"])
                for row in claims["rows"]}
    source_ops = tuple(str(value) for value in source["authorized_operators"])
    source_edges = tuple(tuple(edge) for edge in source["authorized_compositions"])
    permuted_edges = source_permuted_compositions(source_ops, source_edges)
    rows = []; source_commits = 0; permuted_commits = 0
    for raw in grounding["rows"]:
        task_id = str(raw["task_id"]); semantic = _semantic(raw["semantic_receipt"])
        graph = _grounding(raw["grounding_receipt"]); claim = evidence[task_id]
        source_plan = plan_harness_arm(
            semantic, arm="source_induced", source_capabilities=source,
            all_vm_operators=source_ops,
        )
        permuted_plan = plan_harness_arm(
            semantic, arm="source_permuted", source_capabilities=source,
            all_vm_operators=source_ops,
        )
        strict = execute_layer_b_semantics(
            compact_semantics=compact[task_id], grounding=graph, semantic=semantic,
            authorized_operators=source_ops, authorized_compositions=source_edges,
            ambiguity_policy="STRICT",
        )
        permuted = execute_layer_b_semantics(
            compact_semantics=compact[task_id], grounding=graph, semantic=semantic,
            authorized_operators=source_ops, authorized_compositions=permuted_edges,
            ambiguity_policy="STRICT",
        )
        source_safe, source_reason = source_root_open_world_commit(
            semantic=semantic, symbolic_status=strict.receipt.status,
            symbolic_prediction=strict.receipt.prediction, evidence=claim,
        )
        permuted_safe, permuted_reason = source_root_open_world_commit(
            semantic=semantic, symbolic_status=permuted.receipt.status,
            symbolic_prediction=permuted.receipt.prediction, evidence=claim,
        )
        source_commit = source_plan.status == "PLANNED" and source_safe
        permuted_commit = permuted_plan.status == "PLANNED" and permuted_safe
        source_commits += int(source_commit); permuted_commits += int(permuted_commit)
        rows.append({
            "task_id": task_id,
            "source_plan_status": source_plan.status,
            "source_executor_status": strict.receipt.status,
            "source_commit": source_commit, "source_reason": source_reason,
            "permuted_plan_status": permuted_plan.status,
            "permuted_executor_status": permuted.receipt.status,
            "permuted_commit": permuted_commit, "permuted_reason": permuted_reason,
            "claim_receipt_sha256": claim.receipt_sha256,
        })
    coverage = source_commits / len(rows)
    threshold = float(prereg["gates"]["outcome_blind_source_execution_coverage_at_least"])
    matched = (
        len(set(permuted_edges)) == len(set(source_edges))
        and set(permuted_edges) != set(source_edges)
        and {value for edge in permuted_edges for value in edge} <= set(source_ops)
    )
    passed = coverage >= threshold and matched
    body = {
        "schema_version": "agqa-layer-b-risk-pre-outcome-freeze-v1",
        "status": "ALL_RUNTIME_ARTIFACTS_FROZEN_BEFORE_OUTCOMES" if passed
                  else "PRE_OUTCOME_GATE_FAILED",
        "preregistration_file_sha256": _file_sha(args.preregistration),
        "cohort_sha256": cohort["cohort_sha256"],
        "semantic_runtime_sha256": runtime["runtime_sha256"],
        "grounding_report_sha256": grounding["report_sha256"],
        "claim_report_sha256": claims["report_sha256"],
        "fallback_report_sha256": fallback["report_sha256"],
        "source_capability_sha256": source["artifact_sha256"],
        "source_commits": source_commits, "source_permuted_commits": permuted_commits,
        "tasks": len(rows), "source_execution_coverage": coverage,
        "coverage_threshold": threshold, "coverage_gate_passed": coverage >= threshold,
        "matched_permutation_gate_passed": matched,
        "authentic_edge_count": len(set(source_edges)),
        "permuted_edge_count": len(set(permuted_edges)),
        "rows": rows,
        "answers_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read_by_grounder": False,
        "next_and_only_outcome_operation": "RISK_CONSTRAINED_FIVE_ARM_EVALUATOR_ONCE",
    }
    body["receipt_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "source_commits": source_commits,
        "source_permuted_commits": permuted_commits,
        "source_execution_coverage": coverage,
        "coverage_gate_passed": body["coverage_gate_passed"],
        "matched_permutation_gate_passed": matched,
        "receipt_sha256": body["receipt_sha256"],
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
