#!/usr/bin/env python3
"""Freeze strict-boundary five-arm decisions before formal AGQA outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_harness import (
    ARMS,
    plan_harness_arm,
    source_permuted_compositions,
)
from motif_transfer.agqa_layer_b_executor_v2 import execute_layer_b_semantics_v2
from motif_transfer.anonymous_video_harness import route_grounded_candidate
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _grounding, _semantic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_disagreement_opportunity(
    rows: list[dict], *, minimum_fraction: float,
) -> dict:
    """Audit whether the Harness can produce a non-trivial paired comparison.

    This check is deliberately outcome-blind.  Symbolic commits are not enough:
    if the shared neural actor already emits the same answer, the Harness cannot
    contribute paired wins (or losses), regardless of grounding precision.
    """

    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("minimum prediction-disagreement fraction must be in [0,1]")
    disagreements = sum(
        row["predictions"]["source_induced"]
        != row["predictions"]["neural_only"]
        for row in rows
    )
    fraction = disagreements / len(rows) if rows else 0.0
    return {
        "source_neural_prediction_disagreements": disagreements,
        "source_neural_prediction_disagreement_fraction": fraction,
        "minimum_source_neural_prediction_disagreement_fraction": minimum_fraction,
        "passes": fraction >= minimum_fraction,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol", "manifest", "qualification", "cohort", "semantic-runtime",
        "query-grounding", "grounding", "fallback", "source-capabilities",
        "anonymous-controller", "action-grounding", "slowfast-bindings", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--parent-grounding", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("strict-boundary preoutcome receipt is immutable")

    paths = {
        name.replace("_", "-"): getattr(args, name)
        for name in (
            "protocol", "manifest", "qualification", "cohort", "semantic_runtime",
            "query_grounding", "grounding", "fallback", "source_capabilities",
            "anonymous_controller", "action_grounding", "slowfast_bindings",
        )
    }
    if args.parent_grounding is not None:
        paths["parent-grounding"] = args.parent_grounding
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
    action_grounding = json.loads(args.action_grounding.read_text())
    slowfast_bindings = json.loads(args.slowfast_bindings.read_text())
    parent_grounding = (
        json.loads(args.parent_grounding.read_text())
        if args.parent_grounding is not None else None
    )

    if manifest.get("status") != "AGQA_QUERY_GROUNDER_V2_STRICT_BOUNDARY_FRESH_FORMAL_FROZEN" or not all(
        manifest.get("gates", {}).values()
    ):
        raise ValueError("strict-boundary formal reserve is not eligible")
    if manifest.get("protocol_file_sha256") != _sha256(args.protocol):
        raise ValueError("formal protocol changed after reserve freeze")
    if qualification.get("status") != "QUERY_GROUNDER_V2_STRICT_BOUNDARY_QUALIFIED" or not all(
        qualification.get("gates", {}).values()
    ):
        raise ValueError("strict-boundary grounder qualification is invalid")
    frozen = protocol["qualified_grounder"]
    if frozen["qualification_file_sha256"] != _sha256(args.qualification):
        raise ValueError("qualification differs from formal protocol")
    if frozen["qualification_report_sha256"] != qualification["report_sha256"]:
        raise ValueError("qualification report hash differs")
    if source.get("artifact_sha256") != protocol["source_harness"]["source_capability_sha256"]:
        raise ValueError("source capability artifact changed")
    if controller.get("artifact_sha256") != protocol["source_harness"]["anonymous_controller_sha256"]:
        raise ValueError("anonymous controller changed")
    if controller.get("status") != "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED":
        raise ValueError("anonymous source controller is not qualified")

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
    if not all((
        query.get("public_ontology_sha256") == frozen["public_ontology_sha256"],
        int(budgets.get("sgdet_unique_and_model_presentations", -1))
        == int(frozen["frame_budget_sgdet"]),
        int(budgets.get("slowfast_unique_sampled_frames", -1))
        == int(frozen["frame_budget_slowfast_unique"]),
        int(action_grounding.get("frame_presentation_budget", -1))
        == int(frozen["frame_budget_slowfast_presentations"]),
        action_grounding.get("sampling") == frozen["slowfast_sampling"],
        action_grounding.get("checkpoint_sha256") == frozen["slowfast_checkpoint_sha256"],
        slowfast_bindings.get("action_grounding_file_sha256") == _sha256(args.action_grounding),
        query.get("inputs", {}).get("slowfast_bindings_sha256")
        == _sha256(args.slowfast_bindings),
        query.get("strict_temporal_projection") is True,
        query.get("action_event_temporal_representation")
        == frozen["action_event_temporal_representation"],
    )):
        raise ValueError("formal grounder differs from qualified operating point")
    verifier = frozen.get("candidate_verifier")
    if verifier is not None:
        if parent_grounding is None or args.parent_grounding is None:
            raise ValueError("track-verified formal requires its strict parent grounding")
        metadata = query.get("candidate_verification", {})
        if not all((
            query.get("schema_version")
            == verifier.get(
                "grounding_schema_version",
                "agqa-query-grounder-v2-stable-track-verified-v1",
            ),
            metadata.get("formula") == verifier["formula"],
            metadata.get("fitted_weights") is False,
            metadata.get("source_controller_read") is False,
            int(metadata.get("sgdet_frame_budget", -1))
            == int(verifier["sgdet_frame_budget"]),
            query.get("parent_grounding_file_sha256")
            == _sha256(args.parent_grounding),
            query.get("parent_grounding_report_sha256")
            == parent_grounding.get("report_sha256"),
            query.get("parent_grounder_backend_sha256")
            == parent_grounding.get("grounder_backend_sha256"),
        )):
            raise ValueError("formal stable-track verifier differs from qualification")
        parent_rows = {
            str(row["task_id"]): row for row in parent_grounding.get("rows", ())
        }
        if len(parent_rows) != len(query.get("rows", ())) or not all(
            row.get("parent_query_grounding_receipt_sha256")
            == parent_rows.get(str(row["task_id"]), {}).get("receipt", {}).get(
                "receipt_sha256"
            )
            for row in query.get("rows", ())
        ):
            raise ValueError("formal stable-track verifier row hash chain is incomplete")
        if not all(
            _sha256(Path(path)) == verifier["component_sha256s"][name]
            for name, path in verifier["component_paths"].items()
        ):
            raise ValueError("formal stable-track verifier implementation changed")
    forbidden = (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )
    if any(report.get(key) for report in (query, grounding, fallback) for key in forbidden):
        raise ValueError("a formal runtime artifact crossed its authority boundary")
    if any(action_grounding.get(key) for key in (
        "answers_read", "official_program_read", "official_scene_graph_read",
    )) or any(slowfast_bindings.get(key) for key in forbidden):
        raise ValueError("SlowFast artifact crossed its authority boundary")
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
        query_row = query_by_task[task_id]
        candidate_supported = bool(query_row["receipt"]["candidates"]) and (
            float(query_row["candidate_confidence"]) >= threshold
        )
        plans = {
            arm: plan_harness_arm(
                semantic, arm=arm, source_capabilities=source,
                all_vm_operators=operators,
            ) for arm in ARMS
        }
        source_execution = execute_layer_b_semantics_v2(
            compact_semantics=compact[task_id], grounding=event_graph, semantic=semantic,
            authorized_operators=operators, authorized_compositions=source_edges,
            ambiguity_policy="STRICT",
        )
        permuted_execution = execute_layer_b_semantics_v2(
            compact_semantics=compact[task_id], grounding=event_graph, semantic=semantic,
            authorized_operators=operators, authorized_compositions=permuted_edges,
            ambiguity_policy="STRICT",
        )
        generic_execution = execute_layer_b_semantics_v2(
            compact_semantics=compact[task_id], grounding=event_graph, semantic=semantic,
            authorized_operators=operators, authorized_compositions=None,
            ambiguity_policy="EAGER",
        )
        source_candidate = candidate_supported and (
            plans["source_induced"].status == "PLANNED"
            and source_execution.receipt.status == "COMMITTED"
        )
        permuted_candidate = candidate_supported and (
            plans["source_permuted"].status == "PLANNED"
            and permuted_execution.receipt.status == "COMMITTED"
        )
        generic_commit = candidate_supported and (
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
        outputs.append({
            "task_id": task_id, "video_id": str(query_row["video_id"]),
            "root_predicate": str(query_row.get("root_predicate") or "unknown"),
            "requested_role": str(query_row.get("requested_role") or "unknown"),
            "candidate_confidence": float(query_row["candidate_confidence"]),
            "candidate_supported_at_fixed_threshold": candidate_supported,
            "plans": {arm: asdict(plan) for arm, plan in plans.items()},
            "source_execution": asdict(source_execution.receipt),
            "source_permuted_execution": asdict(permuted_execution.receipt),
            "generic_execution": asdict(generic_execution.receipt),
            "source_route": list(source_route),
            "source_permuted_route": list(permuted_route),
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
    source_commits = sum(row["commits"]["source_induced"] for row in outputs)
    permuted_commits = sum(row["commits"]["source_permuted"] for row in outputs)
    coverage = source_commits / n
    permuted_coverage = permuted_commits / n
    formal_gates = protocol["formal_gates"]
    disagreement = _prediction_disagreement_opportunity(
        outputs,
        minimum_fraction=float(formal_gates.get(
            "minimum_source_neural_prediction_disagreement_fraction_preoutcome",
            0.0,
        )),
    )
    invariants = {
        "source_commit_coverage": coverage >= float(
            formal_gates["minimum_source_symbolic_commit_fraction"]
        ),
        "source_permuted_commit_coverage": permuted_coverage <= float(
            formal_gates["maximum_source_permuted_commit_fraction"]
        ),
        "every_symbolic_commit_has_candidate_support": all(
            not any(row["commits"][arm] for arm in (
                "generic_scaffold", "source_permuted", "source_induced",
            )) or row["candidate_supported_at_fixed_threshold"]
            for row in outputs
        ),
        "target_written_isomorphic_preoutcome_equivalence": all(
            row["predictions"]["source_induced"]
            == row["predictions"]["target_written_isomorphic"] for row in outputs
        ),
        "all_rows_have_content_bound_shared_receipts": all(
            row["query_grounding_v2_receipt_sha256"] and row["grounding_receipt_sha256"]
            for row in outputs
        ),
        "full_cohort_frozen": len(outputs) == n,
        "nontrivial_paired_prediction_opportunity": disagreement["passes"],
    }
    body = {
        "schema_version": "agqa-query-grounder-v2-strict-boundary-formal-preoutcome-v1",
        "status": (
            "ALL_FIVE_ARM_STRICT_BOUNDARY_DECISIONS_FROZEN_BEFORE_FORMAL_OUTCOMES"
            if all(invariants.values()) else "PREOUTCOME_GATE_FAILED"
        ),
        **{f"{name.replace('-', '_')}_file_sha256": _sha256(path)
           for name, path in paths.items()},
        "cohort_sha256": cohort_sha, "query_grounding_report_sha256": query["report_sha256"],
        "grounding_report_sha256": grounding["report_sha256"],
        "fallback_report_sha256": fallback["report_sha256"],
        "source_capability_sha256": source["artifact_sha256"],
        "anonymous_controller_sha256": controller["artifact_sha256"],
        "tasks": n, "source_symbolic_commits": source_commits,
        "source_symbolic_commit_fraction": coverage,
        "source_permuted_commits": permuted_commits,
        "source_permuted_commit_fraction": permuted_coverage,
        **{key: value for key, value in disagreement.items() if key != "passes"},
        "invariants": invariants, "rows": outputs,
        "answers_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "target_outcome_read": False,
        "only_next_outcome_operation": "FRESH_FORMAL_FIVE_ARM_EVALUATOR_ONCE",
    }
    body["receipt_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "tasks": n,
        "source_symbolic_commits": source_commits,
        "source_symbolic_commit_fraction": coverage,
        "source_permuted_commits": permuted_commits,
        "source_neural_prediction_disagreements": disagreement[
            "source_neural_prediction_disagreements"
        ],
        "invariants": invariants, "receipt_sha256": body["receipt_sha256"],
    }, indent=2))
    return 0 if all(invariants.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
