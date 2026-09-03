#!/usr/bin/env python3
"""Freeze V15 compositional development decisions before reading answers.

This is deliberately a development qualification, not a formal transfer
result.  It checks that a question-blind multi-event graph leaves genuine
reasoning work for the source-induced Harness, unlike the answer-like V14
single-candidate grounding.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_executor_v2 import execute_layer_b_semantics_v2
from motif_transfer.agqa_layer_b_harness import ARMS, plan_harness_arm, source_permuted_compositions
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
    for name in (
        "cohort", "semantic-runtime", "query-grounding", "grounding", "fallback",
        "source-capabilities", "anonymous-controller", "action-grounding",
        "slowfast-bindings", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--minimum-source-commit-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-permuted-commit-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-disagreement-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-two-event-fraction", type=float, default=0.80)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V15 development preoutcome receipt is immutable")

    paths = {
        name.replace("_", "-"): getattr(args, name)
        for name in (
            "cohort", "semantic_runtime", "query_grounding", "grounding", "fallback",
            "source_capabilities", "anonymous_controller", "action_grounding",
            "slowfast_bindings",
        )
    }
    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    query = json.loads(args.query_grounding.read_text())
    grounding = json.loads(args.grounding.read_text())
    fallback = json.loads(args.fallback.read_text())
    source = json.loads(args.source_capabilities.read_text())
    controller = json.loads(args.anonymous_controller.read_text())
    action = json.loads(args.action_grounding.read_text())
    bindings = json.loads(args.slowfast_bindings.read_text())

    if cohort.get("status") != "CONSUMED_V13_VIDEO_DEVELOPMENT_COHORT_FROZEN_BEFORE_NEW_TASK_OUTCOMES":
        raise ValueError("V15 cohort is not the declared consumed-video development cohort")
    cohort_sha = cohort["cohort_sha256"]
    if len({
        cohort_sha, runtime.get("cohort_sha256"), query.get("cohort_sha256"),
        grounding.get("cohort_sha256"), fallback.get("cohort_sha256"),
    }) != 1:
        raise ValueError("V15 runtime artifacts refer to different cohorts")
    if runtime.get("valid") != len(cohort["rows"]) or runtime.get("invalid"):
        raise ValueError("semantic runtime is incomplete")
    if query.get("status") != "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME":
        raise ValueError("query grounding is not frozen")
    if grounding.get("status") != "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("Layer-B grounding is not frozen")
    if fallback.get("status") != "SHARED_FALLBACK_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("shared fallback is not frozen")
    if grounding.get("query_grounding_report_sha256") != query.get("report_sha256"):
        raise ValueError("Layer-B adapter does not bind query grounding")
    if fallback.get("grounding_report_sha256") != grounding.get("report_sha256"):
        raise ValueError("fallback does not bind shared grounding")
    if bindings.get("action_grounding_file_sha256") != _sha256(args.action_grounding):
        raise ValueError("SlowFast bindings do not bind action grounding")
    if query.get("inputs", {}).get("action_grounding_sha256") != _sha256(args.action_grounding):
        raise ValueError("compositional grounding does not bind action grounding")
    if controller.get("status") != "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED":
        raise ValueError("anonymous source controller is not qualified")
    if not grounding.get("all_harness_arms_share_exact_receipts") or not fallback.get("shared_by_all_five_arms"):
        raise ValueError("five arms do not share grounding/fallback")
    forbidden = (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )
    if any(report.get(key) for report in (query, grounding, fallback, bindings) for key in forbidden):
        raise ValueError("runtime artifact crossed the authority boundary")
    if any(action.get(key) for key in ("answers_read", "official_program_read", "official_scene_graph_read")):
        raise ValueError("action grounder crossed the authority boundary")

    wanted = [str(row["task_id"]) for row in cohort["rows"]]
    compact = {str(row["task_id"]): str(row["predicted_semantics"]) for row in runtime["rows"]}
    query_by_task = {str(row["task_id"]): row for row in query["rows"]}
    grounding_by_task = {str(row["task_id"]): row for row in grounding["rows"]}
    fallback_by_task = {str(row["task_id"]): str(row["prediction"]) for row in fallback["rows"]}
    if any(set(value) != set(wanted) for value in (compact, query_by_task, grounding_by_task, fallback_by_task)):
        raise ValueError("runtime artifacts do not exactly cover V15")

    operators = tuple(str(value) for value in source["authorized_operators"])
    source_edges = tuple(tuple(str(x) for x in edge) for edge in source["authorized_compositions"])
    permuted_edges = source_permuted_compositions(operators, source_edges)
    outputs = []
    for public in cohort["rows"]:
        task_id = str(public["task_id"])
        raw = grounding_by_task[task_id]
        semantic = _semantic(raw["semantic_receipt"])
        graph = _grounding(raw["grounding_receipt"])
        plans = {
            arm: plan_harness_arm(
                semantic, arm=arm, source_capabilities=source,
                all_vm_operators=operators,
            ) for arm in ARMS
        }
        source_execution = execute_layer_b_semantics_v2(
            compact_semantics=compact[task_id], grounding=graph, semantic=semantic,
            authorized_operators=operators, authorized_compositions=source_edges,
            ambiguity_policy="STRICT",
        )
        permuted_execution = execute_layer_b_semantics_v2(
            compact_semantics=compact[task_id], grounding=graph, semantic=semantic,
            authorized_operators=operators, authorized_compositions=permuted_edges,
            ambiguity_policy="STRICT",
        )
        generic_execution = execute_layer_b_semantics_v2(
            compact_semantics=compact[task_id], grounding=graph, semantic=semantic,
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
        generic_commit = (
            plans["generic_scaffold"].status == "PLANNED"
            and generic_execution.receipt.status == "COMMITTED"
        )
        source_route = route_grounded_candidate(controller, candidate_qualified=source_candidate)
        permuted_route = route_grounded_candidate(controller, candidate_qualified=permuted_candidate)
        source_commit = source_route[-1] == "COMMIT"
        permuted_commit = permuted_route[-1] == "COMMIT"
        neural = fallback_by_task[task_id]
        source_prediction = str(source_execution.receipt.prediction) if source_commit else neural
        permuted_prediction = str(permuted_execution.receipt.prediction) if permuted_commit else neural
        generic_prediction = str(generic_execution.receipt.prediction) if generic_commit else neural
        event_count = len(raw["grounding_receipt"]["events"])
        outputs.append({
            "task_id": task_id, "video_id": str(public["video_id"]),
            "semantic_root": compact[task_id].split("(", 1)[0].strip(),
            "event_count": event_count,
            "plans": {arm: asdict(plan) for arm, plan in plans.items()},
            "source_execution": asdict(source_execution.receipt),
            "source_permuted_execution": asdict(permuted_execution.receipt),
            "generic_execution": asdict(generic_execution.receipt),
            "source_route": list(source_route), "source_permuted_route": list(permuted_route),
            "predictions": {
                "neural_only": neural, "generic_scaffold": generic_prediction,
                "source_permuted": permuted_prediction, "source_induced": source_prediction,
                "target_written_isomorphic": source_prediction,
            },
            "commits": {
                "neural_only": False, "generic_scaffold": generic_commit,
                "source_permuted": permuted_commit, "source_induced": source_commit,
                "target_written_isomorphic": source_commit,
            },
            "query_grounding_v2_receipt_sha256": raw["query_grounding_v2_receipt_sha256"],
            "grounding_receipt_sha256": raw["grounding_receipt"]["receipt_sha256"],
        })

    n = len(outputs)
    source_commits = sum(row["commits"]["source_induced"] for row in outputs)
    permuted_commits = sum(row["commits"]["source_permuted"] for row in outputs)
    disagreements = sum(
        row["predictions"]["source_induced"] != row["predictions"]["neural_only"]
        for row in outputs
    )
    multi_event = sum(row["event_count"] >= 2 for row in outputs)
    metrics = {
        "source_symbolic_commits": source_commits,
        "source_symbolic_commit_fraction": source_commits / n,
        "source_permuted_commits": permuted_commits,
        "source_permuted_commit_fraction": permuted_commits / n,
        "source_neural_prediction_disagreements": disagreements,
        "source_neural_prediction_disagreement_fraction": disagreements / n,
        "two_or_more_event_rows": multi_event,
        "two_or_more_event_fraction": multi_event / n,
    }
    gates = {
        "source_commit_coverage": metrics["source_symbolic_commit_fraction"] >= args.minimum_source_commit_fraction,
        "source_permuted_commit_coverage": metrics["source_permuted_commit_fraction"] <= args.maximum_permuted_commit_fraction,
        "nontrivial_paired_prediction_opportunity": metrics["source_neural_prediction_disagreement_fraction"] >= args.minimum_disagreement_fraction,
        "question_blind_multi_event_graph": metrics["two_or_more_event_fraction"] >= args.minimum_two_event_fraction,
        "target_written_isomorphic_preoutcome_equivalence": all(
            row["predictions"]["source_induced"] == row["predictions"]["target_written_isomorphic"]
            for row in outputs
        ),
        "all_rows_have_content_bound_shared_receipts": all(
            row["query_grounding_v2_receipt_sha256"] and row["grounding_receipt_sha256"]
            for row in outputs
        ),
        "full_cohort_frozen": n == len(cohort["rows"]),
    }
    body = {
        "schema_version": "agqa-offtheshelf-compositional-development-preoutcome-v15",
        "status": "V15_COMPOSITIONAL_DEVELOPMENT_DECISIONS_FROZEN" if all(gates.values()) else "V15_PREOUTCOME_GATE_FAILED",
        **{f"{name.replace('-', '_')}_file_sha256": _sha256(path) for name, path in paths.items()},
        "cohort_sha256": cohort_sha, "query_grounding_report_sha256": query["report_sha256"],
        "grounding_report_sha256": grounding["report_sha256"],
        "fallback_report_sha256": fallback["report_sha256"],
        "source_capability_sha256": source["artifact_sha256"],
        "anonymous_controller_sha256": controller["artifact_sha256"],
        "tasks": n, "thresholds": {
            "minimum_source_commit_fraction": args.minimum_source_commit_fraction,
            "maximum_permuted_commit_fraction": args.maximum_permuted_commit_fraction,
            "minimum_disagreement_fraction": args.minimum_disagreement_fraction,
            "minimum_two_event_fraction": args.minimum_two_event_fraction,
        },
        "metrics": metrics, "gates": gates, "rows": outputs,
        "development_only": True, "answers_read": False,
        "official_scene_graph_read": False, "functional_program_read": False,
        "target_outcome_read": False,
        "only_next_outcome_operation": "V15_DEVELOPMENT_EVALUATOR_ONCE",
    }
    body["receipt_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "tasks": n, "metrics": metrics,
        "gates": gates, "receipt_sha256": body["receipt_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
