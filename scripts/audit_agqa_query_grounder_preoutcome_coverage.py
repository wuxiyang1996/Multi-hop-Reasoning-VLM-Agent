#!/usr/bin/env python3
"""Audit source-Harness commit coverage without reading any target outcome."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_executor_v2 import execute_layer_b_semantics_v2
from motif_transfer.agqa_layer_b_harness import (
    plan_harness_arm,
    source_permuted_compositions,
)
from motif_transfer.agqa_query_grounder_v2 import (
    adapt_query_grounding_v2,
    query_grounding_v2_from_dict,
)
from motif_transfer.anonymous_video_harness import route_grounded_candidate
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _semantic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--query-grounding", type=Path, required=True)
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--anonymous-controller", type=Path, required=True)
    parser.add_argument("--minimum-candidate-confidence", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("pre-outcome coverage audit is immutable")
    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    query = json.loads(args.query_grounding.read_text())
    source = json.loads(args.source_capabilities.read_text())
    controller = json.loads(args.anonymous_controller.read_text())
    if len({cohort["cohort_sha256"], runtime["cohort_sha256"], query["cohort_sha256"]}) != 1:
        raise ValueError("coverage audit inputs refer to different cohorts")
    if any(query.get(key) for key in (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )):
        raise ValueError("query grounding crossed its authority boundary")
    if controller.get("status") != "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED":
        raise ValueError("anonymous source controller is not qualified")
    semantics = {
        str(row["task_id"]): (
            str(row["predicted_semantics"]), _semantic(row["receipt"])
        ) for row in runtime["rows"]
    }
    operators = tuple(str(value) for value in source["authorized_operators"])
    source_edges = tuple(tuple(str(x) for x in edge) for edge in source["authorized_compositions"])
    permuted_edges = source_permuted_compositions(operators, source_edges)
    rows = []
    for raw in query["rows"]:
        task_id = str(raw["task_id"])
        compact, semantic = semantics[task_id]
        typed = query_grounding_v2_from_dict(raw["receipt"])
        candidate_supported = bool(typed.candidates) and (
            float(raw["candidate_confidence"])
            >= args.minimum_candidate_confidence
        )
        grounding = adapt_query_grounding_v2(
            typed, semantic,
            minimum_candidate_confidence=args.minimum_candidate_confidence,
        )
        source_plan = plan_harness_arm(
            semantic, arm="source_induced", source_capabilities=source,
            all_vm_operators=operators,
        )
        permuted_plan = plan_harness_arm(
            semantic, arm="source_permuted", source_capabilities=source,
            all_vm_operators=operators,
        )
        source_execution = execute_layer_b_semantics_v2(
            compact_semantics=compact, grounding=grounding, semantic=semantic,
            authorized_operators=operators, authorized_compositions=source_edges,
            ambiguity_policy="STRICT",
        )
        permuted_execution = execute_layer_b_semantics_v2(
            compact_semantics=compact, grounding=grounding, semantic=semantic,
            authorized_operators=operators, authorized_compositions=permuted_edges,
            ambiguity_policy="STRICT",
        )
        source_candidate = (
            candidate_supported
            and source_plan.status == "PLANNED"
            and source_execution.receipt.status == "COMMITTED"
        )
        permuted_candidate = (
            candidate_supported
            and permuted_plan.status == "PLANNED"
            and permuted_execution.receipt.status == "COMMITTED"
        )
        source_route = route_grounded_candidate(
            controller, candidate_qualified=source_candidate,
        )
        permuted_route = route_grounded_candidate(
            controller, candidate_qualified=permuted_candidate,
        )
        rows.append({
            "task_id": task_id,
            "video_id": str(raw["video_id"]),
            "candidate_confidence": float(raw["candidate_confidence"]),
            "candidate_supported": candidate_supported,
            "source_plan_status": source_plan.status,
            "source_execution": asdict(source_execution.receipt),
            "source_route": list(source_route),
            "source_commit": source_route[-1] == "COMMIT",
            "permuted_plan_status": permuted_plan.status,
            "permuted_execution": asdict(permuted_execution.receipt),
            "permuted_route": list(permuted_route),
            "permuted_commit": permuted_route[-1] == "COMMIT",
            "grounding_receipt_sha256": grounding.receipt_sha256,
        })
    n = len(rows)
    source_commits = sum(bool(row["source_commit"]) for row in rows)
    permuted_commits = sum(bool(row["permuted_commit"]) for row in rows)
    candidate_support = sum(bool(row["candidate_supported"]) for row in rows)
    reasons = Counter(
        str(row["source_execution"]["reason"]) for row in rows
        if not row["source_commit"]
    )
    body = {
        "schema_version": "agqa-query-grounder-preoutcome-coverage-audit-v1",
        "status": "OUTCOME_BLIND_COVERAGE_AUDIT_COMPLETE",
        "cohort_sha256": cohort["cohort_sha256"],
        "query_grounding_report_sha256": query["report_sha256"],
        "source_capability_sha256": source["artifact_sha256"],
        "anonymous_controller_sha256": controller["artifact_sha256"],
        "minimum_candidate_confidence": args.minimum_candidate_confidence,
        "tasks": n,
        "candidate_supported": candidate_support,
        "candidate_supported_fraction": candidate_support / n if n else 0.0,
        "source_commits": source_commits,
        "source_commit_fraction": source_commits / n if n else 0.0,
        "permuted_commits": permuted_commits,
        "permuted_commit_fraction": permuted_commits / n if n else 0.0,
        "source_abstention_reasons": dict(sorted(reasons.items())),
        "rows": rows,
        "answer_read": False,
        "official_scene_graph_read": False,
        "functional_program_read": False,
        "target_outcome_read": False,
        "inputs": {
            "cohort_file_sha256": _sha256(args.cohort),
            "semantic_runtime_file_sha256": _sha256(args.semantic_runtime),
            "query_grounding_file_sha256": _sha256(args.query_grounding),
            "source_capability_file_sha256": _sha256(args.source_capabilities),
            "anonymous_controller_file_sha256": _sha256(args.anonymous_controller),
        },
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: body[key] for key in (
            "status", "tasks", "candidate_supported",
            "candidate_supported_fraction", "source_commits",
            "source_commit_fraction", "permuted_commits",
            "permuted_commit_fraction", "source_abstention_reasons",
            "report_sha256",
        )
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
