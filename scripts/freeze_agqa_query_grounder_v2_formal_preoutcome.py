#!/usr/bin/env python3
"""Freeze all five-arm decisions before opening fresh formal outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_harness import ARMS, plan_harness_arm, source_permuted_compositions
from motif_transfer.agqa_layer_b_executor import execute_layer_b_semantics
from motif_transfer.anonymous_video_harness import route_grounded_candidate
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _grounding, _semantic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--query-grounding", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--anonymous-controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("formal pre-outcome receipt is immutable")

    protocol = json.loads(args.protocol.read_text())
    manifest = json.loads(args.manifest.read_text())
    qualification = json.loads(args.qualification.read_text())
    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    query = json.loads(args.query_grounding.read_text())
    grounding = json.loads(args.grounding.read_text())
    fallback = json.loads(args.fallback.read_text())
    source = json.loads(args.source_capabilities.read_text())
    controller = json.loads(args.anonymous_controller.read_text())
    if manifest.get("status") != "AGQA_QUERY_GROUNDER_V2_FRESH_FORMAL_FROZEN" or not all(
        manifest.get("gates", {}).values()
    ):
        raise ValueError("fresh formal reserve is not eligible")
    if manifest.get("protocol_file_sha256") != _sha256(args.protocol):
        raise ValueError("formal protocol changed after reserve freeze")
    if qualification.get("status") != "QUERY_GROUNDER_V2_POWERED_QUALIFIED" or not all(
        qualification.get("gates", {}).values()
    ):
        raise ValueError("grounder qualification is not valid")
    frozen = protocol["qualified_grounder"]
    if frozen["qualification_file_sha256"] != _sha256(args.qualification):
        raise ValueError("qualification file differs from protocol")
    if controller.get("status") != "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED":
        raise ValueError("anonymous game controller is not qualified")
    if source.get("artifact_sha256") != protocol["source_harness"]["source_capability_sha256"]:
        raise ValueError("source capability artifact changed")
    if controller.get("artifact_sha256") != protocol["source_harness"]["anonymous_controller_sha256"]:
        raise ValueError("anonymous source controller changed")
    cohort_sha = cohort["cohort_sha256"]
    if len({
        cohort_sha, manifest["cohort_sha256"], runtime["cohort_sha256"],
        query["cohort_sha256"], grounding["cohort_sha256"], fallback["cohort_sha256"],
    }) != 1:
        raise ValueError("formal runtime artifacts refer to different cohorts")
    n = int(protocol["formal_cohort"]["query_object_tasks"])
    if len(cohort["rows"]) != n or runtime["valid"] != n or runtime["invalid"]:
        raise ValueError("formal parser coverage is incomplete")
    if query.get("status") != "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME":
        raise ValueError("typed query grounding is not frozen")
    if grounding.get("status") != "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("Layer-B grounding adapter is not frozen")
    if fallback.get("status") != "SHARED_FALLBACK_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("shared fallback is not frozen")
    if grounding["query_grounding_report_sha256"] != query["report_sha256"]:
        raise ValueError("Layer-B adapter does not bind typed grounding")
    if fallback["grounding_report_sha256"] != grounding["report_sha256"]:
        raise ValueError("fallback does not bind shared grounding")
    threshold = float(frozen["candidate_support_threshold"])
    if float(grounding["minimum_candidate_confidence"]) != threshold:
        raise ValueError("formal adapter threshold differs from qualification")
    budgets = query.get("component_frame_budgets", {})
    if (
        query.get("public_ontology_sha256") != frozen["public_ontology_sha256"]
        or int(budgets.get("sgdet_unique_and_model_presentations", -1))
        != int(frozen["frame_budget_sgdet"])
        or int(budgets.get("slowfast_unique_sampled_frames", -1))
        != int(frozen["frame_budget_slowfast"])
    ):
        raise ValueError("formal grounder differs from qualified operating point")
    forbidden = (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )
    if any(report.get(key) for report in (query, grounding, fallback) for key in forbidden):
        raise ValueError("a formal runtime artifact crossed its authority boundary")
    if not query.get("all_harness_arms_share_exact_receipts") or not grounding.get(
        "all_harness_arms_share_exact_receipts"
    ) or not fallback.get("shared_by_all_five_arms"):
        raise ValueError("five arms do not share the frozen target runtime")

    compact = {str(row["task_id"]): str(row["predicted_semantics"]) for row in runtime["rows"]}
    fallback_by_task = {str(row["task_id"]): str(row["prediction"]) for row in fallback["rows"]}
    grounding_by_task = {str(row["task_id"]): row for row in grounding["rows"]}
    query_by_task = {str(row["task_id"]): row for row in query["rows"]}
    wanted = [str(row["task_id"]) for row in cohort["rows"]]
    if any(set(value) != set(wanted) for value in (
        compact, fallback_by_task, grounding_by_task, query_by_task,
    )):
        raise ValueError("formal artifacts do not exactly cover the cohort")
    operators = tuple(str(value) for value in source["authorized_operators"])
    source_edges = tuple(tuple(str(x) for x in edge) for edge in source["authorized_compositions"])
    permuted_edges = source_permuted_compositions(operators, source_edges)
    outputs = []
    for task_id in wanted:
        raw = grounding_by_task[task_id]
        semantic = _semantic(raw["semantic_receipt"])
        event_graph = _grounding(raw["grounding_receipt"])
        plans = {arm: plan_harness_arm(
            semantic, arm=arm, source_capabilities=source, all_vm_operators=operators,
        ) for arm in ARMS}
        source_execution = execute_layer_b_semantics(
            compact_semantics=compact[task_id], grounding=event_graph, semantic=semantic,
            authorized_operators=operators, authorized_compositions=source_edges,
            ambiguity_policy="STRICT",
        )
        permuted_execution = execute_layer_b_semantics(
            compact_semantics=compact[task_id], grounding=event_graph, semantic=semantic,
            authorized_operators=operators, authorized_compositions=permuted_edges,
            ambiguity_policy="STRICT",
        )
        generic_execution = execute_layer_b_semantics(
            compact_semantics=compact[task_id], grounding=event_graph, semantic=semantic,
            authorized_operators=operators, authorized_compositions=None,
            ambiguity_policy="EAGER",
        )
        source_candidate = (
            plans["source_induced"].status == "PLANNED"
            and source_execution.receipt.status == "COMMITTED"
        )
        permuted_candidate = (
            plans["source_permuted"].status == "PLANNED"
            and permuted_execution.receipt.status == "COMMITTED"
        )
        source_route = route_grounded_candidate(controller, candidate_qualified=source_candidate)
        permuted_route = route_grounded_candidate(controller, candidate_qualified=permuted_candidate)
        source_commit = source_route[-1] == "COMMIT"
        permuted_commit = permuted_route[-1] == "COMMIT"
        generic_commit = (
            plans["generic_scaffold"].status == "PLANNED"
            and generic_execution.receipt.status == "COMMITTED"
        )
        source_prediction = (
            str(source_execution.receipt.prediction) if source_commit else fallback_by_task[task_id]
        )
        permuted_prediction = (
            str(permuted_execution.receipt.prediction)
            if permuted_commit else fallback_by_task[task_id]
        )
        generic_prediction = (
            str(generic_execution.receipt.prediction)
            if generic_commit else fallback_by_task[task_id]
        )
        outputs.append({
            "task_id": task_id,
            "video_id": str(query_by_task[task_id]["video_id"]),
            "root_predicate": str(query_by_task[task_id].get("root_predicate") or "unknown"),
            "requested_role": str(query_by_task[task_id].get("requested_role") or "unknown"),
            "candidate_confidence": float(query_by_task[task_id]["candidate_confidence"]),
            "candidate_supported_at_fixed_threshold": (
                bool(query_by_task[task_id]["receipt"]["candidates"])
                and float(query_by_task[task_id]["candidate_confidence"]) >= threshold
            ),
            "plans": {arm: asdict(plan) for arm, plan in plans.items()},
            "source_execution": asdict(source_execution.receipt),
            "source_permuted_execution": asdict(permuted_execution.receipt),
            "generic_execution": asdict(generic_execution.receipt),
            "source_route": list(source_route),
            "source_permuted_route": list(permuted_route),
            "predictions": {
                "neural_only": fallback_by_task[task_id],
                "generic_scaffold": generic_prediction,
                "source_permuted": permuted_prediction,
                "source_induced": source_prediction,
                "target_written_isomorphic": source_prediction,
            },
            "commits": {
                "neural_only": False,
                "generic_scaffold": generic_commit,
                "source_permuted": permuted_commit,
                "source_induced": source_commit,
                "target_written_isomorphic": source_commit,
            },
            "query_grounding_v2_receipt_sha256": raw["query_grounding_v2_receipt_sha256"],
            "grounding_receipt_sha256": raw["grounding_receipt"]["receipt_sha256"],
        })
    source_commits = sum(row["commits"]["source_induced"] for row in outputs)
    coverage = source_commits / n
    minimum = float(protocol["formal_gates"]["minimum_source_symbolic_commit_fraction"])
    invariants = {
        "source_commit_coverage": coverage >= minimum,
        "target_written_isomorphic_preoutcome_equivalence": all(
            row["predictions"]["source_induced"]
            == row["predictions"]["target_written_isomorphic"] for row in outputs
        ),
        "all_rows_have_content_bound_shared_receipts": all(
            row["query_grounding_v2_receipt_sha256"] and row["grounding_receipt_sha256"]
            for row in outputs
        ),
        "full_cohort_frozen": len(outputs) == n,
    }
    body = {
        "schema_version": "agqa-query-grounder-v2-formal-preoutcome-v1",
        "status": (
            "ALL_FIVE_ARM_DECISIONS_FROZEN_BEFORE_FORMAL_OUTCOMES"
            if all(invariants.values()) else "PREOUTCOME_GATE_FAILED"
        ),
        "protocol_file_sha256": _sha256(args.protocol),
        "manifest_file_sha256": _sha256(args.manifest),
        "qualification_file_sha256": _sha256(args.qualification),
        "cohort_file_sha256": _sha256(args.cohort),
        "semantic_runtime_file_sha256": _sha256(args.semantic_runtime),
        "query_grounding_file_sha256": _sha256(args.query_grounding),
        "grounding_file_sha256": _sha256(args.grounding),
        "fallback_file_sha256": _sha256(args.fallback),
        "source_capability_file_sha256": _sha256(args.source_capabilities),
        "anonymous_controller_file_sha256": _sha256(args.anonymous_controller),
        "cohort_sha256": cohort_sha,
        "query_grounding_report_sha256": query["report_sha256"],
        "grounding_report_sha256": grounding["report_sha256"],
        "fallback_report_sha256": fallback["report_sha256"],
        "source_capability_sha256": source["artifact_sha256"],
        "anonymous_controller_sha256": controller["artifact_sha256"],
        "tasks": n,
        "source_symbolic_commits": source_commits,
        "source_symbolic_commit_fraction": coverage,
        "invariants": invariants,
        "rows": outputs,
        "answers_read": False,
        "official_scene_graph_read": False,
        "functional_program_read": False,
        "target_outcome_read": False,
        "only_next_outcome_operation": "FRESH_FORMAL_FIVE_ARM_EVALUATOR_ONCE",
    }
    body["receipt_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "tasks": n,
        "source_symbolic_commits": source_commits,
        "source_symbolic_commit_fraction": coverage,
        "invariants": invariants, "receipt_sha256": body["receipt_sha256"],
    }, indent=2))
    return 0 if all(invariants.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
