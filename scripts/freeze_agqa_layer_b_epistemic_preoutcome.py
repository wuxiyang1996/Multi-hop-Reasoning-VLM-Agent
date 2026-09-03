#!/usr/bin/env python3
"""Freeze all Layer-B V4 artifacts and open-world coverage before outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_epistemic import (
    AtomicVisualClaim, AtomicVisualClaimDecision, AtomicVisualClaimReceipt,
    source_open_world_commit,
)
from motif_transfer.agqa_layer_b_executor import execute_layer_b_semantics
from motif_transfer.agqa_layer_b_harness import plan_harness_arm
from motif_transfer.contracts import stable_hash
from motif_transfer.anonymous_video_harness import route_grounded_candidate
from scripts.evaluate_agqa_layer_b_five_arm import _grounding, _semantic


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _claims(raw: dict) -> AtomicVisualClaimReceipt:
    value = AtomicVisualClaimReceipt(**{
        **raw,
        "claims": tuple(AtomicVisualClaim(**row) for row in raw["claims"]),
        "decisions": tuple(AtomicVisualClaimDecision(**{
            **row,
            "evidence_frame_indices": tuple(row["evidence_frame_indices"]),
            "evidence_frame_sha256s": tuple(row["evidence_frame_sha256s"]),
        }) for row in raw["decisions"]),
    })
    value.validate(); return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--anonymous-controller", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("pre-outcome receipt is immutable")
    prereg = json.loads(args.preregistration.read_text())
    cohort = json.loads(args.cohort.read_text()); runtime = json.loads(args.semantic_runtime.read_text())
    grounding = json.loads(args.grounding.read_text()); claim_report = json.loads(args.claims.read_text())
    fallback = json.loads(args.fallback.read_text()); source = json.loads(args.source_capabilities.read_text())
    controller = (
        json.loads(args.anonymous_controller.read_text())
        if args.anonymous_controller is not None else None
    )
    if controller is not None and controller.get("status") != "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED":
        raise ValueError("anonymous source controller is not qualified")
    new_broad_protocol = prereg.get("schema_version") in {
        "agqa-full-train-broad-layer-b-preregistration-v1",
        "agqa-full-train-broad-layer-b-preregistration-v2",
    }
    expected_status = (
        "FROZEN_AFTER_QUESTION_ONLY_PARSER_QUALIFICATION_AND_BEFORE_RAW_VIDEO_OR_OUTCOME"
        if prereg.get("schema_version") == "agqa-full-train-broad-layer-b-preregistration-v2"
        else "FROZEN_BEFORE_PARSER_GROUNDER_CLAIMS_FALLBACK_OR_OUTCOME"
    )
    if prereg["status"] != expected_status:
        raise ValueError("invalid Layer-B preregistration status")
    prereg_cohort_sha = (
        prereg["cohort"]["cohort_sha256"] if new_broad_protocol
        else prereg["cohort"]["public_cohort_sha256"]
    )
    if prereg_cohort_sha != cohort["cohort_sha256"]:
        raise ValueError("preregistered cohort mismatch")
    if runtime["valid"] != len(cohort["rows"]) or runtime["invalid"]:
        raise ValueError("semantic runtime is incomplete")
    if len({cohort["cohort_sha256"], runtime["cohort_sha256"], grounding["cohort_sha256"],
            claim_report["cohort_sha256"], fallback["cohort_sha256"]}) != 1:
        raise ValueError("runtime artifacts refer to different cohorts")
    if claim_report["base_grounding_report_sha256"] != grounding["report_sha256"]:
        raise ValueError("atomic claims do not bind grounding")
    if fallback["grounding_report_sha256"] != grounding["report_sha256"]:
        raise ValueError("fallback does not bind grounding")
    if not claim_report["all_harness_arms_share_exact_receipts"] or not fallback["shared_by_all_five_arms"]:
        raise ValueError("artifacts are not shared across arms")
    compact = {str(row["task_id"]): str(row["predicted_semantics"]) for row in runtime["rows"]}
    claim_by_task = {str(row["task_id"]): _claims(row["claim_receipt"])
                     for row in claim_report["rows"]}
    ops = tuple(source["authorized_operators"])
    edges = tuple(tuple(edge) for edge in source["authorized_compositions"])
    rows = []; commits = 0
    for raw in grounding["rows"]:
        task_id = str(raw["task_id"]); semantic = _semantic(raw["semantic_receipt"])
        event_graph = _grounding(raw["grounding_receipt"]); evidence = claim_by_task[task_id]
        plan = plan_harness_arm(
            semantic, arm="source_induced", source_capabilities=source,
            all_vm_operators=ops,
        )
        execution = execute_layer_b_semantics(
            compact_semantics=compact[task_id], grounding=event_graph, semantic=semantic,
            authorized_operators=ops, authorized_compositions=edges,
            ambiguity_policy="STRICT",
        )
        safe, reason = source_open_world_commit(
            required_operators=plan.required_operators,
            symbolic_status=execution.receipt.status,
            symbolic_prediction=execution.receipt.prediction,
            evidence=evidence,
        )
        if controller is not None:
            route = route_grounded_candidate(controller, candidate_qualified=safe)
            safe = route[-1] == "COMMIT"
            reason = f"ANONYMOUS_ROUTE:{'/'.join(route)};TARGET_NATIVE:{reason}"
        commits += int(safe)
        rows.append({
            "task_id": task_id, "planned": plan.status == "PLANNED",
            "executor_status": execution.receipt.status, "open_world_commit": safe,
            "open_world_reason": reason, "claim_receipt_sha256": evidence.receipt_sha256,
        })
    coverage = commits / len(rows)
    threshold = float(
        prereg["formal_gates"]["minimum_source_symbolic_commit_fraction"]
        if new_broad_protocol else
        prereg["gates"]["outcome_blind_source_execution_coverage_at_least"]
    )
    body = {
        "schema_version": "agqa-layer-b-epistemic-pre-outcome-freeze-v1",
        "status": "ALL_RUNTIME_ARTIFACTS_FROZEN_BEFORE_OUTCOMES" if coverage >= threshold
                  else "PRE_OUTCOME_COVERAGE_GATE_FAILED",
        "preregistration_file_sha256": _file_sha(args.preregistration),
        "cohort_sha256": cohort["cohort_sha256"],
        "semantic_runtime_sha256": runtime["runtime_sha256"],
        "grounding_report_sha256": grounding["report_sha256"],
        "claim_report_sha256": claim_report["report_sha256"],
        "fallback_report_sha256": fallback["report_sha256"],
        "source_capability_sha256": source["artifact_sha256"],
        "anonymous_controller_sha256": (
            controller["artifact_sha256"] if controller is not None else None
        ),
        "source_open_world_commits": commits, "tasks": len(rows),
        "source_open_world_execution_coverage": coverage,
        "coverage_threshold": threshold, "coverage_gate_passed": coverage >= threshold,
        "rows": rows, "answers_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read_by_grounder": False,
        "next_and_only_outcome_operation": "EPISODIC_FIVE_ARM_QUALIFICATION_EVALUATOR_ONCE",
    }
    body["receipt_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: body[key] for key in (
        "status", "source_open_world_commits", "tasks",
        "source_open_world_execution_coverage", "coverage_gate_passed", "receipt_sha256",
    )}, indent=2))
    return 0 if coverage >= threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
