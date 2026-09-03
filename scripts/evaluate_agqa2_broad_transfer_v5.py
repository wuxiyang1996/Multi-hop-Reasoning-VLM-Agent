#!/usr/bin/env python3
"""Evaluate frozen broad-stack/source transfer on untouched AGQA V5."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from motif_transfer.agqa_active_frame_grounder import parse_public_question_plan  # noqa: E402
from motif_transfer.agqa_broad_oracle_executor import execute_broad_public_plan  # noqa: E402
from motif_transfer.agqa_oracle_query_mdp import (  # noqa: E402
    AGQAOracleQueryBackend, AGQAOracleToolBudget,
    compose_localized_with_generic, execute_temporal_object_query,
    load_agqa_id_to_text,
)
from motif_transfer.agqa_temporal_localized_query import parse_temporal_localized_object_question  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.official_video_event_graph import load_builtin_only_pickle, sha256_file  # noqa: E402
from scripts.evaluate_agqa2_oracle_query_transfer_v3 import _metric, _normalize, _paired, _read  # noqa: E402

RUN = REPO / "runs/agqa2_broad_transfer_v5"
PREREG = REPO / "configs/agqa2_broad_transfer_v5_preregistration.json"
AUTHORIZATION = REPO / "configs/agqa2_broad_oracle_authorization_v1.json"
STSG = Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/AGQA_train_stsgs.pkl")
ONTOLOGY = STSG.parent / "ENG.txt"


def main() -> None:
    prereg = json.loads(PREREG.read_text())
    authorization = _read(AUTHORIZATION)
    if authorization["artifact_sha256"] != prereg["frozen_dependencies"]["broad_authorization_artifact_sha256"]:
        raise ValueError("broad authorization differs from preregistration")
    allowed = set(authorization["authorized_comparisons"])
    public = _read(RUN / "public_cohort.json")
    ontology = load_agqa_id_to_text(ONTOLOGY)
    corpus = load_builtin_only_pickle(STSG)
    stsg_sha = sha256_file(STSG)
    permutation = {"BEFORE": "AFTER", "AFTER": "BEFORE", "WHILE": "BEFORE", "BETWEEN": "WHILE"}
    runtime_rows = []
    for row in public["rows"]:
        temporal_plan = parse_temporal_localized_object_question(row["question"])
        broad_plan = parse_public_question_plan(row["question"])
        video_id = str(row["video_id"]); graph_hash = stable_hash([stsg_sha, video_id])
        localized = permuted = unlocalized = broad = None
        local_calls = permuted_calls = generic_calls = broad_calls = 0
        if broad_plan is not None and broad_plan.comparison in allowed:
            backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=graph_hash, budget=AGQAOracleToolBudget(3),
            )
            result = execute_broad_public_plan(broad_plan, backend)
            broad = result.prediction; broad_calls = len(result.receipts)
        if temporal_plan is not None:
            backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=graph_hash, budget=AGQAOracleToolBudget(3),
            )
            result = execute_temporal_object_query(temporal_plan, backend)
            localized = result.prediction; local_calls = len(result.receipts)
            backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=graph_hash, budget=AGQAOracleToolBudget(3),
            )
            backend.locate_action(temporal_plan.anchor_a)
            result = backend.query_relation(temporal_plan.relation, frames=backend.all_frame_numbers())
            generic_calls = len(backend.receipts)
            if len(result["objects"]) == 1:
                unlocalized = str(result["objects"][0])
            permuted_plan = replace(
                temporal_plan, temporal_operator=permutation[temporal_plan.temporal_operator],
                anchor_b="" if temporal_plan.temporal_operator == "BETWEEN" else temporal_plan.anchor_b,
            )
            backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=graph_hash, budget=AGQAOracleToolBudget(3),
            )
            result = execute_temporal_object_query(permuted_plan, backend)
            permuted = result.prediction; permuted_calls = len(result.receipts)
        target = unlocalized if temporal_plan is not None else broad
        target_calls = generic_calls if temporal_plan is not None else broad_calls
        source = compose_localized_with_generic(localized, target)
        shuffled = compose_localized_with_generic(permuted, target)
        generic = compose_localized_with_generic(None, target)
        predictions = {
            "source_induced": source.prediction,
            "source_permuted": shuffled.prediction,
            "generic_scaffold": generic.prediction,
            "isomorphic_target_controller": source.prediction,
        }
        runtime_rows.append({
            "task_id": row["task_id"], "video_id": video_id,
            "question_sha256": row["question_sha256"], "predictions": predictions,
            "tool_calls": {
                "source_induced": local_calls + (target_calls if localized is None else 0),
                "source_permuted": permuted_calls + (target_calls if permuted is None else 0),
                "generic_scaffold": target_calls,
                "isomorphic_target_controller": local_calls + (target_calls if localized is None else 0),
            },
            "runtime_answer_read": False, "runtime_functional_program_read": False,
            "runtime_sg_grounding_read": False,
        })
    runtime_body = {
        "schema_version": "agqa2-broad-transfer-runtime-v5",
        "role": "PREDICTIONS_FROZEN_BEFORE_EVALUATOR",
        "public_artifact_sha256": public["artifact_sha256"],
        "broad_authorization_sha256": authorization["artifact_sha256"],
        "official_stsg_sha256": stsg_sha, "rows": runtime_rows,
    }
    runtime = runtime_body | {"artifact_sha256": stable_hash(runtime_body)}
    (RUN / "runtime_predictions.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    evaluator = _read(RUN / "evaluator_only.json")
    gold = {row["task_id"]: row["answer"] for row in evaluator["rows"]}
    evaluated = []
    for row in runtime_rows:
        evaluated.append(row | {"correct": {
            arm: prediction is not None and _normalize(prediction) == _normalize(gold[row["task_id"]])
            for arm, prediction in row["predictions"].items()
        }})
    metrics = {arm: _metric(evaluated, arm) for arm in runtime_rows[0]["predictions"]}
    paired = {arm: _paired(evaluated, "source_induced", arm) for arm in (
        "generic_scaffold", "source_permuted", "isomorphic_target_controller",
    )}
    freeze = json.loads((RUN / "freeze_receipt.json").read_text())
    gates = {
        "all_videos_unseen_in_v62_v2_v3": freeze["all_videos_unseen_in_v62_v2_v3"],
        "zero_runtime_answer_program_grounding_read": all(
            not row["runtime_answer_read"] and not row["runtime_functional_program_read"]
            and not row["runtime_sg_grounding_read"] for row in runtime_rows
        ),
        "source_correct_strictly_above_generic": metrics["source_induced"]["correct"] > metrics["generic_scaffold"]["correct"],
        "source_correct_strictly_above_temporal_permutation": metrics["source_induced"]["correct"] > metrics["source_permuted"]["correct"],
        "source_losses_vs_generic": paired["generic_scaffold"]["losses"] == 0,
        "isomorphic_target_equivalence_disclosed": metrics["source_induced"] == metrics["isomorphic_target_controller"],
        "matched_tool_budget": all(max(row["tool_calls"].values()) <= 5 for row in runtime_rows),
    }
    body = {
        "schema_version": "agqa2-broad-transfer-qualification-v5",
        "status": "PASSED" if all(gates.values()) else "FAILED",
        "claim_boundary": prereg["claim_boundary"], "tasks": len(evaluated),
        "unique_videos": len({row["video_id"] for row in evaluated}),
        "metrics": metrics, "source_paired": paired, "gates": gates,
        "lineage": {
            "preregistration_sha256": stable_hash(prereg),
            "public_artifact_sha256": public["artifact_sha256"],
            "runtime_artifact_sha256": runtime["artifact_sha256"],
            "evaluator_artifact_sha256": evaluator["artifact_sha256"],
            "broad_authorization_sha256": authorization["artifact_sha256"],
            "official_stsg_sha256": stsg_sha,
        },
    }
    report = body | {"report_sha256": stable_hash(body)}
    output = REPO / "docs/results/agqa2_broad_transfer_v5.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
