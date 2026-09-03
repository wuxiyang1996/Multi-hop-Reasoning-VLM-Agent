#!/usr/bin/env python3
"""Outcome-blind five-arm decisions for consumed V15 Qwen32 development."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_executor_v2 import execute_layer_b_semantics_v2
from motif_transfer.agqa_layer_b_executor_v3 import execute_layer_b_semantics_v3
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
        "cohort", "semantic-runtime", "routed-grounding", "grounding-view",
        "fallback", "source-capabilities", "anonymous-controller", "output",
    ):
        kwargs = {"action": "append"} if name == "grounding-view" else {}
        parser.add_argument(f"--{name}", type=Path, required=True, **kwargs)
    parser.add_argument("--minimum-source-commit-fraction", type=float, default=0.20)
    parser.add_argument("--maximum-permuted-commit-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-disagreement-fraction", type=float, default=0.05)
    parser.add_argument("--executor-version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--formal-protocol", type=Path)
    parser.add_argument("--formal-manifest", type=Path)
    parser.add_argument("--download-receipt", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V16 development preoutcome receipt is immutable")
    if len(args.grounding_view) != 2:
        raise ValueError("exactly two frozen grounding views are required")
    execute_semantics = (
        execute_layer_b_semantics_v3
        if args.executor_version == "v3" else execute_layer_b_semantics_v2
    )

    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    routed = json.loads(args.routed_grounding.read_text())
    views = [json.loads(path.read_text()) for path in args.grounding_view]
    fallback = json.loads(args.fallback.read_text())
    source = json.loads(args.source_capabilities.read_text())
    controller = json.loads(args.anonymous_controller.read_text())
    formal_protocol = json.loads(args.formal_protocol.read_text()) if args.formal_protocol else None
    formal_manifest = json.loads(args.formal_manifest.read_text()) if args.formal_manifest else None
    formal = formal_protocol is not None or formal_manifest is not None
    if formal_protocol is None and formal_manifest is not None or formal_protocol is not None and formal_manifest is None:
        raise ValueError("formal protocol and manifest must be supplied together")
    if formal != (args.download_receipt is not None):
        raise ValueError("a content-verified download receipt is required exactly for formal evaluation")
    cohort_sha = cohort["cohort_sha256"]
    expected_cohort_status = (
        "FROZEN_QWEN32_COMPOSITIONAL_FORMAL_V17_BEFORE_RUNTIME_OR_OUTCOME"
        if formal else
        "CONSUMED_V13_VIDEO_DEVELOPMENT_COHORT_FROZEN_BEFORE_NEW_TASK_OUTCOMES"
    )
    if cohort.get("status") != expected_cohort_status:
        raise ValueError("Qwen32 preoutcome cohort phase/status mismatch")
    if len({cohort_sha, runtime.get("cohort_sha256"), routed.get("cohort_sha256"),
            fallback.get("cohort_sha256"), *(view.get("cohort_sha256") for view in views)}) != 1:
        raise ValueError("V16 artifact cohort mismatch")
    if any(report.get("status") != "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES" for report in (routed, *views)):
        raise ValueError("grounding is not frozen")
    if fallback.get("status") != "SHARED_FALLBACK_FROZEN_BEFORE_OUTCOMES" or fallback.get("grounding_report_sha256") != routed.get("report_sha256"):
        raise ValueError("fallback is not bound to routed grounding")
    if controller.get("status") != "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED":
        raise ValueError("anonymous source controller is invalid")
    if formal:
        download = json.loads(args.download_receipt.read_text())
        if formal_protocol.get("status") != "QWEN32_COMPOSITIONAL_FORMAL_PROTOCOL_FROZEN_AFTER_DEVELOPMENT":
            raise ValueError("formal protocol is invalid")
        if formal_manifest.get("status") != "AGQA_QWEN32_COMPOSITIONAL_FRESH_FORMAL_V17_FROZEN":
            raise ValueError("formal manifest is invalid")
        if formal_manifest.get("protocol_file_sha256") != _sha256(args.formal_protocol):
            raise ValueError("formal manifest does not bind protocol")
        if formal_manifest.get("public_cohort_sha256") != cohort_sha:
            raise ValueError("formal manifest does not bind cohort")
        if download.get("status") != "COMPLETE":
            raise ValueError("formal raw-video download is incomplete")
        if download.get("selection_manifest_sha256") != formal_manifest.get("download_selection_sha256"):
            raise ValueError("formal download receipt does not bind frozen selection")
        frozen = formal_protocol["runtime"]
        if [view.get("grounder_backend_sha256") for view in views] != frozen["qwen32_grounder_backend_sha256s"]:
            raise ValueError("formal Qwen32 backend differs from development-qualified backend")
        if [int(view.get("frame_budget", -1)) for view in views] != frozen["frame_views"]:
            raise ValueError("formal frame views differ from protocol")
        if source.get("artifact_sha256") != formal_protocol["source_harness"]["source_capability_sha256"]:
            raise ValueError("formal source capability changed")
        if controller.get("artifact_sha256") != formal_protocol["source_harness"]["anonymous_controller_sha256"]:
            raise ValueError("formal anonymous controller changed")
        if _sha256(Path(__file__)) != formal_protocol["components"]["preoutcome_sha256"]:
            raise ValueError("formal preoutcome implementation changed")
        component_paths = {
            "semantic_parser_sha256": Path(__file__).with_name("run_agqa_layer_b_semantic_parser.py"),
            "qwen32_collector_sha256": Path(__file__).with_name("collect_agqa_layer_b_qwen235_grounding.py"),
            "grounding_merger_sha256": Path(__file__).with_name("merge_agqa_layer_b_grounding_shards.py"),
            "grounding_router_sha256": Path(__file__).with_name("route_agqa_layer_b_shared_grounding.py"),
            "fallback_sha256": Path(__file__).with_name("collect_agqa_layer_b_shared_fallback.py"),
            "downloader_sha256": Path(__file__).with_name("download_agqa2_active_grounding_v4_reserve.py"),
        }
        if any(_sha256(path) != formal_protocol["components"][key]
               for key, path in component_paths.items()):
            raise ValueError("formal runtime component changed after protocol freeze")
        if runtime.get("parser_sha256") != frozen["semantic_parser_sha256"]:
            raise ValueError("formal semantic parser differs from development-qualified parser")
        if runtime.get("qualification_sha256") != frozen["semantic_parser_qualification_sha256"]:
            raise ValueError("formal semantic parser qualification differs from protocol")
        if any(view.get("model") != frozen["grounder_model"] for view in views):
            raise ValueError("formal grounder model differs from protocol")
        if routed.get("candidate_report_sha256s") != [view["report_sha256"] for view in views]:
            raise ValueError("formal router does not bind ordered grounding views")
        if fallback.get("model") != frozen["fallback_model"]:
            raise ValueError("formal fallback model differs from protocol")
        if args.executor_version != frozen["executor_version"]:
            raise ValueError("formal executor adapter differs from protocol")
        if _sha256(Path(__file__).parents[1] / "src/motif_transfer/agqa_layer_b_executor_v3.py") != formal_protocol["components"]["executor_adapter_sha256"]:
            raise ValueError("formal executor adapter changed after protocol freeze")
        gates = formal_protocol["preoutcome_gates"]
        if any((
            args.minimum_source_commit_fraction != float(gates["minimum_source_commit_fraction"]),
            args.maximum_permuted_commit_fraction != float(gates["maximum_permuted_commit_fraction"]),
            args.minimum_disagreement_fraction != float(gates["minimum_disagreement_fraction"]),
        )):
            raise ValueError("formal preoutcome thresholds differ from protocol")
    forbidden = ("answer_read", "official_scene_graph_read", "functional_program_read", "source_controller_read", "target_outcome_read")
    if any(report.get(key) for report in (routed, *views, fallback) for key in forbidden):
        raise ValueError("V16 runtime crossed the authority boundary")
    if not all(report.get("all_harness_arms_share_exact_receipts") for report in (routed, *views)) or not fallback.get("shared_by_all_five_arms"):
        raise ValueError("five arms do not share exact runtime receipts")

    wanted = [str(row["task_id"]) for row in cohort["rows"]]
    compact = {str(row["task_id"]): str(row["predicted_semantics"]) for row in runtime["rows"]}
    routed_rows = {str(row["task_id"]): row for row in routed["rows"]}
    view_rows = [{str(row["task_id"]): row for row in view["rows"]} for view in views]
    fallback_rows = {str(row["task_id"]): str(row["prediction"]) for row in fallback["rows"]}
    if any(set(rows) != set(wanted) for rows in (compact, routed_rows, fallback_rows, *view_rows)):
        raise ValueError("V16 artifacts do not exactly cover the cohort")
    if formal:
        downloads = {str(row["video_id"]): row for row in download["videos"]}
        cohort_videos = {str(row["video_id"]) for row in cohort["rows"]}
        if set(downloads) != cohort_videos:
            raise ValueError("formal download receipt does not exactly cover cohort videos")
        for task_id in wanted:
            video_id = str(routed_rows[task_id]["video_id"])
            expected_video_sha = downloads[video_id]["sha256"]
            if any(rows[task_id]["grounding_receipt"]["video_sha256"] != expected_video_sha
                   for rows in (routed_rows, *view_rows)):
                raise ValueError("formal grounding is not content-bound to downloaded video")

    operators = tuple(str(value) for value in source["authorized_operators"])
    source_edges = tuple(tuple(str(x) for x in edge) for edge in source["authorized_compositions"])
    permuted_edges = source_permuted_compositions(operators, source_edges)
    outputs = []
    for task_id in wanted:
        routed_row = routed_rows[task_id]
        semantic = _semantic(routed_row["semantic_receipt"])
        plans = {arm: plan_harness_arm(
            semantic, arm=arm, source_capabilities=source,
            all_vm_operators=operators,
        ) for arm in ARMS}
        source_executions = []
        permuted_executions = []
        for rows in view_rows:
            graph = _grounding(rows[task_id]["grounding_receipt"])
            source_executions.append(execute_semantics(
                compact_semantics=compact[task_id], grounding=graph, semantic=semantic,
                authorized_operators=operators, authorized_compositions=source_edges,
                ambiguity_policy="STRICT",
            ))
            permuted_executions.append(execute_semantics(
                compact_semantics=compact[task_id], grounding=graph, semantic=semantic,
                authorized_operators=operators, authorized_compositions=permuted_edges,
                ambiguity_policy="STRICT",
            ))
        source_candidate = (
            plans["source_induced"].status == "PLANNED"
            and all(item.receipt.status == "COMMITTED" for item in source_executions)
            and len({str(item.receipt.prediction) for item in source_executions}) == 1
        )
        permuted_candidate = (
            plans["source_permuted"].status == "PLANNED"
            and all(item.receipt.status == "COMMITTED" for item in permuted_executions)
            and len({str(item.receipt.prediction) for item in permuted_executions}) == 1
        )
        source_route = route_grounded_candidate(controller, candidate_qualified=source_candidate)
        permuted_route = route_grounded_candidate(controller, candidate_qualified=permuted_candidate)
        source_commit = source_route[-1] == "COMMIT"
        permuted_commit = permuted_route[-1] == "COMMIT"
        routed_graph = _grounding(routed_row["grounding_receipt"])
        generic_execution = execute_semantics(
            compact_semantics=compact[task_id], grounding=routed_graph, semantic=semantic,
            authorized_operators=operators, authorized_compositions=None,
            ambiguity_policy="EAGER",
        )
        generic_commit = plans["generic_scaffold"].status == "PLANNED" and generic_execution.receipt.status == "COMMITTED"
        neural = fallback_rows[task_id]
        source_prediction = str(source_executions[0].receipt.prediction) if source_commit else neural
        permuted_prediction = str(permuted_executions[0].receipt.prediction) if permuted_commit else neural
        generic_prediction = str(generic_execution.receipt.prediction) if generic_commit else neural
        outputs.append({
            "task_id": task_id, "video_id": str(routed_row["video_id"]),
            "semantic_root": compact[task_id].split("(", 1)[0].strip(),
            "event_count": len(routed_graph.events),
            "plans": {arm: asdict(plan) for arm, plan in plans.items()},
            "source_view_executions": [asdict(item.receipt) for item in source_executions],
            "source_permuted_view_executions": [asdict(item.receipt) for item in permuted_executions],
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
            "routed_grounding_receipt_sha256": routed_row["grounding_receipt"]["receipt_sha256"],
            "view_grounding_receipt_sha256s": [rows[task_id]["grounding_receipt"]["receipt_sha256"] for rows in view_rows],
        })

    n = len(outputs)
    source_commits = sum(row["commits"]["source_induced"] for row in outputs)
    permuted_commits = sum(row["commits"]["source_permuted"] for row in outputs)
    disagreements = sum(row["predictions"]["source_induced"] != row["predictions"]["neural_only"] for row in outputs)
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
        "question_blind_multi_event_graph": metrics["two_or_more_event_fraction"] >= (
            float(formal_protocol["preoutcome_gates"]["minimum_two_event_fraction"])
            if formal else 0.50
        ),
        "target_written_isomorphic_preoutcome_equivalence": all(row["predictions"]["source_induced"] == row["predictions"]["target_written_isomorphic"] for row in outputs),
        "all_rows_have_content_bound_shared_receipts": all(row["routed_grounding_receipt_sha256"] and all(row["view_grounding_receipt_sha256s"]) for row in outputs),
        "full_cohort_frozen": n == len(cohort["rows"]),
    }
    if formal:
        status = (
            "V17_FORMAL_FIVE_ARM_DECISIONS_FROZEN_BEFORE_OUTCOMES"
            if all(gates.values()) else "V17_FORMAL_PREOUTCOME_GATE_FAILED"
        )
    elif args.executor_version == "v3":
        status = (
            "V18_COMPOSITIONAL_DEVELOPMENT_DECISIONS_FROZEN"
            if all(gates.values()) else "V18_PREOUTCOME_GATE_FAILED"
        )
    else:
        status = (
            "V15_COMPOSITIONAL_DEVELOPMENT_DECISIONS_FROZEN"
            if all(gates.values()) else "V16_PREOUTCOME_GATE_FAILED"
        )
    body = {
        "schema_version": (
            "agqa-offtheshelf-qwen32-multiview-preoutcome-v18"
            if args.executor_version == "v3" else
            "agqa-offtheshelf-qwen32-multiview-development-preoutcome-v16"
        ),
        "status": status,
        "cohort_file_sha256": _sha256(args.cohort),
        "semantic_runtime_file_sha256": _sha256(args.semantic_runtime),
        "routed_grounding_file_sha256": _sha256(args.routed_grounding),
        "grounding_view_file_sha256s": [_sha256(path) for path in args.grounding_view],
        "fallback_file_sha256": _sha256(args.fallback),
        "source_capabilities_file_sha256": _sha256(args.source_capabilities),
        "anonymous_controller_file_sha256": _sha256(args.anonymous_controller),
        "cohort_sha256": cohort_sha, "routed_grounding_report_sha256": routed["report_sha256"],
        "grounding_view_report_sha256s": [view["report_sha256"] for view in views],
        "fallback_report_sha256": fallback["report_sha256"],
        "source_capability_sha256": source["artifact_sha256"],
        "anonymous_controller_sha256": controller["artifact_sha256"],
        "tasks": n, "metrics": metrics, "gates": gates, "rows": outputs,
        "executor_version": args.executor_version,
        "formal_protocol_file_sha256": _sha256(args.formal_protocol) if formal else None,
        "formal_manifest_file_sha256": _sha256(args.formal_manifest) if formal else None,
        "download_receipt_file_sha256": _sha256(args.download_receipt) if formal else None,
        "development_only": not formal,
        "development_outcomes_previously_consumed": not formal,
        "formal_outcomes_unread": formal,
        "answer_loaded_by_this_process": False, "official_scene_graph_read": False,
        "functional_program_read": False, "target_outcome_read": False,
    }
    body["receipt_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": body["status"], "metrics": metrics, "gates": gates, "receipt_sha256": body["receipt_sha256"]}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
