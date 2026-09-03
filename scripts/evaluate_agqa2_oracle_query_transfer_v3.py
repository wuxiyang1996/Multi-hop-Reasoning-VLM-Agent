#!/usr/bin/env python3
"""Evaluate preregistered AGQA2 source/generic/permuted query policies."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from motif_transfer.agqa_oracle_query_mdp import (  # noqa: E402
    AGQAOracleQueryBackend, AGQAOracleToolBudget,
    compose_localized_with_generic, execute_temporal_object_query,
    load_agqa_id_to_text,
)
from motif_transfer.agqa_temporal_localized_query import (  # noqa: E402
    parse_temporal_localized_object_question,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.official_video_event_graph import (  # noqa: E402
    load_builtin_only_pickle, sha256_file,
)
from motif_transfer.source_goal_acquisition_induction import (  # noqa: E402
    validate_goal_acquisition_program,
)

RUN = REPO / "runs/agqa2_oracle_query_mdp_v3_transfer"
STSG = Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/AGQA_train_stsgs.pkl")
ONTOLOGY = STSG.parent / "ENG.txt"
PREREG = REPO / "configs/agqa2_oracle_query_mdp_v3_transfer_preregistration.json"
SOURCE = REPO / "runs/sokoban_goal_acquisition_v1/artifact.json"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text()); body = dict(value)
    claimed = body.pop("artifact_sha256")
    if stable_hash(body) != claimed:
        raise ValueError(f"artifact hash mismatch: {path}")
    return value


def _normalize(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    return re.sub(r"^(?:a|an|the)\s+", "", text)


def _metric(rows: list[dict], arm: str) -> dict:
    correct = sum(row["correct"][arm] for row in rows)
    committed = sum(row["predictions"][arm] is not None for row in rows)
    return {
        "correct": correct, "accuracy": correct / len(rows),
        "committed": committed, "coverage": committed / len(rows),
        "conditional_accuracy": correct / committed if committed else 0.0,
    }


def _paired(rows: list[dict], first: str, second: str) -> dict:
    wins = sum(r["correct"][first] and not r["correct"][second] for r in rows)
    losses = sum(not r["correct"][first] and r["correct"][second] for r in rows)
    return {"wins": wins, "losses": losses, "ties": len(rows) - wins - losses,
            "net_wins": wins - losses}


def main() -> None:
    prereg = json.loads(PREREG.read_text())
    public = _read(RUN / "public_cohort.json")
    source = json.loads(SOURCE.read_text())
    validate_goal_acquisition_program(source)
    cardinality = int(source["program"]["relation_binding_cardinality"]["value"])
    if cardinality != 1:
        raise ValueError("source artifact does not induce unique binding")
    ontology = load_agqa_id_to_text(ONTOLOGY)
    corpus = load_builtin_only_pickle(STSG)
    stsg_sha = sha256_file(STSG)
    runtime_rows = []
    permutation = {"BEFORE": "AFTER", "AFTER": "BEFORE", "WHILE": "BEFORE", "BETWEEN": "WHILE"}
    for row in public["rows"]:
        plan = parse_temporal_localized_object_question(row["question"])
        localized = permuted = generic = None
        localized_calls = permuted_calls = generic_calls = 0
        localized_execution = None
        if plan is not None:
            video_id = str(row["video_id"])
            graph_hash = stable_hash([stsg_sha, video_id])
            localized_backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=graph_hash, budget=AGQAOracleToolBudget(3),
            )
            localized_execution = execute_temporal_object_query(plan, localized_backend)
            localized = localized_execution.prediction
            localized_calls = len(localized_execution.receipts)

            generic_backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=graph_hash, budget=AGQAOracleToolBudget(3),
            )
            generic_backend.locate_action(plan.anchor_a)
            generic_result = generic_backend.query_relation(
                plan.relation, frames=generic_backend.all_frame_numbers(),
            )
            generic_calls = len(generic_backend.receipts)
            if len(generic_result["objects"]) == cardinality:
                generic = str(generic_result["objects"][0])

            permuted_plan = replace(
                plan, temporal_operator=permutation[plan.temporal_operator],
                anchor_b="" if plan.temporal_operator == "BETWEEN" else plan.anchor_b,
            )
            permuted_backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=graph_hash, budget=AGQAOracleToolBudget(3),
            )
            permuted_execution = execute_temporal_object_query(permuted_plan, permuted_backend)
            permuted = permuted_execution.prediction
            permuted_calls = len(permuted_execution.receipts)

        source_composed = compose_localized_with_generic(localized, generic)
        permuted_composed = compose_localized_with_generic(permuted, generic)
        predictions = {
            "source_induced": source_composed.prediction,
            "source_permuted": permuted_composed.prediction,
            "generic_scaffold": generic,
            "isomorphic_target_controller": source_composed.prediction,
        }
        source_calls = localized_calls + (generic_calls if localized is None else 0)
        source_permuted_calls = permuted_calls + (generic_calls if permuted is None else 0)
        runtime_rows.append({
            "task_id": row["task_id"], "video_id": row["video_id"],
            "question_sha256": row["question_sha256"],
            "parse_applicable": plan is not None,
            "localized_status": localized_execution.status if localized_execution else "PARSE_ABSTAINED",
            "localized_prediction": localized,
            "source_route": source_composed.route,
            "source_permuted_route": permuted_composed.route,
            "predictions": predictions,
            "tool_calls": {
                "source_induced": source_calls,
                "source_permuted": source_permuted_calls,
                "generic_scaffold": generic_calls,
                "isomorphic_target_controller": source_calls,
            },
            "matched_max_tool_calls": 5,
            "runtime_answer_read": False,
            "runtime_functional_program_read": False,
            "runtime_sg_grounding_read": False,
        })
    if any(max(row["tool_calls"].values()) > 5 for row in runtime_rows):
        raise ValueError("matched tool budget exceeded")
    runtime_body = {
        "schema_version": "agqa2-oracle-query-runtime-v3-transfer",
        "role": "MATCHED_PREDICTIONS_FROZEN_BEFORE_EVALUATOR",
        "public_artifact_sha256": public["artifact_sha256"],
        "source_artifact_sha256": source["artifact_sha256"],
        "official_stsg_sha256": stsg_sha, "rows": runtime_rows,
    }
    runtime = runtime_body | {"artifact_sha256": stable_hash(runtime_body)}
    (RUN / "runtime_predictions.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")

    # Evaluator authority begins only after all four arm predictions are durable.
    evaluator = _read(RUN / "evaluator_only.json")
    gold = {row["task_id"]: row["answer"] for row in evaluator["rows"]}
    evaluated = []
    for row in runtime_rows:
        correct = {
            arm: prediction is not None and _normalize(prediction) == _normalize(gold[row["task_id"]])
            for arm, prediction in row["predictions"].items()
        }
        evaluated.append(row | {"correct": correct})
    metrics = {arm: _metric(evaluated, arm) for arm in runtime_rows[0]["predictions"]}
    committed_localized = [row for row in evaluated if row["localized_prediction"] is not None]
    localized_correct = sum(
        _normalize(row["localized_prediction"]) == _normalize(gold[row["task_id"]])
        for row in committed_localized
    )
    coverage = sum(row["parse_applicable"] for row in evaluated) / len(evaluated)
    freeze = json.loads((RUN / "freeze_receipt.json").read_text())
    gates = {
        "all_videos_unseen_in_v62_and_v2": freeze["all_videos_unseen_in_v62_and_v2"],
        "zero_runtime_answer_program_grounding_read": all(
            not row["runtime_answer_read"] and not row["runtime_functional_program_read"]
            and not row["runtime_sg_grounding_read"] for row in evaluated
        ),
        "minimum_public_compiler_coverage": coverage >= 0.40,
        "minimum_localized_committed_accuracy": bool(committed_localized)
        and localized_correct / len(committed_localized) >= 0.95,
        "source_correct_strictly_above_generic": metrics["source_induced"]["correct"]
        > metrics["generic_scaffold"]["correct"],
        "source_correct_strictly_above_temporal_permutation": metrics["source_induced"]["correct"]
        > metrics["source_permuted"]["correct"],
        "isomorphic_target_equivalence_disclosed": metrics["source_induced"]
        == metrics["isomorphic_target_controller"],
        "matched_maximum_tool_budget_respected": all(
            max(row["tool_calls"].values()) <= row["matched_max_tool_calls"] for row in evaluated
        ),
    }
    body = {
        "schema_version": "agqa2-oracle-query-transfer-qualification-v3",
        "status": "PASSED" if all(gates.values()) else "FAILED",
        "claim_boundary": prereg["claim_boundary"],
        "tasks": len(evaluated), "unique_videos": len({row["video_id"] for row in evaluated}),
        "public_compiler_coverage": coverage,
        "localized_committed": len(committed_localized),
        "localized_committed_correct": localized_correct,
        "localized_committed_accuracy": localized_correct / len(committed_localized),
        "metrics": metrics,
        "source_paired": {
            arm: _paired(evaluated, "source_induced", arm)
            for arm in ("generic_scaffold", "source_permuted", "isomorphic_target_controller")
        },
        "gates": gates,
        "source_specific_transfer_validated": all(gates.values()),
        "lineage": {
            "preregistration_sha256": stable_hash(prereg),
            "public_artifact_sha256": public["artifact_sha256"],
            "runtime_artifact_sha256": runtime["artifact_sha256"],
            "evaluator_artifact_sha256": evaluator["artifact_sha256"],
            "source_artifact_sha256": source["artifact_sha256"],
            "official_stsg_sha256": stsg_sha,
            "official_ontology_sha256": sha256_file(ONTOLOGY),
        },
    }
    report = body | {"report_sha256": stable_hash(body)}
    output = REPO / "docs/results/agqa2_oracle_query_transfer_v3.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
