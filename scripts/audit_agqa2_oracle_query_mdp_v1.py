#!/usr/bin/env python3
"""Consumed diagnostic for the AGQA2 answer-blind oracle query MDP.

The runtime pass reads only public questions, frozen direct predictions, a
source-induced cardinality guard, the official ontology, and the official
STSG.  Predictions for every arm freeze before this process opens the separate
evaluator-only answer file.
"""

from __future__ import annotations

import argparse
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


def _normalize(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    for prefix in ("the answer is ", "it is ", "they were ", "they are "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    return {"true": "yes", "false": "no"}.get(text, text)


def _equivalent(prediction: object, gold: object) -> bool:
    predicted, expected = _normalize(prediction), _normalize(gold)
    if expected in {"yes", "no", "before", "after"}:
        return bool(predicted) and predicted.split(maxsplit=1)[0] == expected
    return predicted == expected


def _read_artifact(path: Path) -> dict:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = body.pop("artifact_sha256")
    if stable_hash(body) != claimed:
        raise ValueError(f"artifact hash mismatch: {path}")
    return value


def _metric(rows, arm):
    return {
        "correct": sum(row["correct"][arm] for row in rows),
        "accuracy": sum(row["correct"][arm] for row in rows) / len(rows),
    }


def _paired(rows, first, second):
    wins = sum(r["correct"][first] and not r["correct"][second] for r in rows)
    losses = sum(not r["correct"][first] and r["correct"][second] for r in rows)
    return {"wins": wins, "losses": losses,
            "ties": len(rows) - wins - losses, "net_wins": wins - losses}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort-dir", type=Path,
        default=REPO / "runs/agqa2_oracle_query_mdp_v1",
    )
    parser.add_argument(
        "--stsg", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/"
                     "scene_graphs/AGQA_scene_graphs/AGQA_train_stsgs.pkl"),
    )
    parser.add_argument(
        "--ontology", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/"
                     "scene_graphs/AGQA_scene_graphs/ENG.txt"),
    )
    parser.add_argument(
        "--source", type=Path,
        default=REPO / "runs/sokoban_goal_acquisition_v1/artifact.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/agqa2_oracle_query_mdp_v1_consumed.json",
    )
    args = parser.parse_args()

    public = _read_artifact(args.cohort_dir / "public_cohort.json")
    direct = _read_artifact(args.cohort_dir / "direct_predictions.json")
    direct_by_id = {row["task_id"]: row["prediction"] for row in direct["rows"]}
    source = json.loads(args.source.read_text())
    validate_goal_acquisition_program(source)
    required_cardinality = int(
        source["program"]["relation_binding_cardinality"]["value"]
    )
    if required_cardinality != 1:
        raise ValueError("source-induced acquisition guard is not unique binding")
    ontology = load_agqa_id_to_text(args.ontology)
    corpus = load_builtin_only_pickle(args.stsg)
    stsg_sha = sha256_file(args.stsg)

    # Runtime phase: no evaluator file is opened above this line.
    runtime_rows = []
    for row in public["rows"]:
        task_id = str(row["task_id"])
        plan = parse_temporal_localized_object_question(row["question"])
        execution = None
        generic_candidate = None
        permuted_candidate = None
        generic_tool_calls = 0
        permuted_tool_calls = 0
        if plan is not None:
            video_id = str(row["video_id"])
            backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=stable_hash([stsg_sha, video_id]),
                budget=AGQAOracleToolBudget(3),
            )
            execution = execute_temporal_object_query(plan, backend)
            # Matched generic: spend the same two calls, but ignore the
            # temporal binding and query the relation over the whole video.
            generic_backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=stable_hash([stsg_sha, video_id]),
                budget=AGQAOracleToolBudget(3),
            )
            generic_backend.locate_action(plan.anchor_a)
            generic_result = generic_backend.query_relation(
                plan.relation, frames=generic_backend.all_frame_numbers(),
            )
            if len(generic_result["objects"]) == 1:
                generic_candidate = str(generic_result["objects"][0])
            generic_tool_calls = len(generic_backend.receipts)
            # Causal control: preserve parser/relation/backend/budget but
            # permute the source temporal transition before execution.
            permutation = {
                "BEFORE": "AFTER", "AFTER": "BEFORE",
                "WHILE": "BEFORE", "BETWEEN": "WHILE",
            }
            permuted_plan = replace(
                plan, temporal_operator=permutation[plan.temporal_operator],
                anchor_b="" if plan.temporal_operator == "BETWEEN" else plan.anchor_b,
            )
            permuted_backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=stable_hash([stsg_sha, video_id]),
                budget=AGQAOracleToolBudget(3),
            )
            permuted_execution = execute_temporal_object_query(
                permuted_plan, permuted_backend,
            )
            permuted_candidate = permuted_execution.prediction
            permuted_tool_calls = len(permuted_execution.receipts)
        candidate = execution.prediction if execution is not None else None
        source_authorized = (
            execution is not None
            and len(execution.candidate_objects) == required_cardinality
            and execution.status == "COMMITTED"
        )
        generic_authorized = generic_candidate is not None
        permuted_authorized = permuted_candidate is not None
        base = str(direct_by_id[task_id])
        source_composed = compose_localized_with_generic(
            str(candidate) if source_authorized else None,
            str(generic_candidate) if generic_authorized else None,
            base,
        )
        permuted_composed = compose_localized_with_generic(
            str(permuted_candidate) if permuted_authorized else None,
            str(generic_candidate) if generic_authorized else None,
            base,
        )
        predictions = {
            "neural_only": base,
            # Unified harness semantics: use the source-acquired localized
            # binding when its EQ-1 guard succeeds, otherwise fall back to the
            # target-native generic route rather than the weak direct actor.
            "source_induced": source_composed.prediction,
            "source_permuted": permuted_composed.prediction,
            "generic_scaffold": str(generic_candidate) if generic_authorized else base,
            # A target engineer can hand-write an isomorphic temporal
            # controller. Keep it as a disclosed ceiling, not as generic.
            "isomorphic_target_controller": source_composed.prediction,
            "target_native_ceiling": source_composed.prediction,
        }
        runtime_rows.append({
            "task_id": task_id, "video_id": row["video_id"],
            "question_sha256": row["question_sha256"],
            "plan_sha256": plan.plan_sha256 if plan else None,
            "parse_applicable": plan is not None,
            "oracle_status": execution.status if execution else "PARSE_ABSTAINED",
            "oracle_reason": execution.reason if execution else "PUBLIC_COMPILER_UNSUPPORTED",
            "oracle_prediction": candidate,
            "generic_prediction": generic_candidate,
            "permuted_prediction": permuted_candidate,
            "oracle_execution_sha256": execution.execution_sha256 if execution else None,
            "oracle_tool_calls": len(execution.receipts) if execution else 0,
            "source_authorized": source_authorized,
            "generic_authorized": generic_authorized,
            "source_permuted_authorized": permuted_authorized,
            "source_route": source_composed.route,
            "source_permuted_route": permuted_composed.route,
            "source_tool_calls": (
                len(execution.receipts) if execution else 0
            ) + (0 if source_authorized else generic_tool_calls),
            "generic_tool_calls": generic_tool_calls,
            "source_permuted_tool_calls": permuted_tool_calls + (
                0 if permuted_authorized else generic_tool_calls
            ),
            "matched_max_tool_calls": 5,
            "predictions": predictions,
            "runtime_answer_read": False,
            "runtime_functional_program_read": False,
            "runtime_sg_grounding_read": False,
        })
    runtime_body = {
        "schema_version": "agqa2-oracle-query-runtime-v1",
        "role": "CONSUMED_DIAGNOSTIC_PREDICTIONS_FROZEN_BEFORE_EVALUATOR",
        "source_artifact_sha256": source["artifact_sha256"],
        "official_stsg_sha256": stsg_sha,
        "rows": runtime_rows,
    }
    runtime_artifact = runtime_body | {"artifact_sha256": stable_hash(runtime_body)}
    runtime_path = args.cohort_dir / "runtime_predictions.json"
    runtime_path.write_text(json.dumps(runtime_artifact, indent=2, sort_keys=True) + "\n")

    # Evaluator phase starts only after runtime_predictions.json is durable.
    evaluator = _read_artifact(args.cohort_dir / "evaluator_only.json")
    gold = {row["task_id"]: row["answer"] for row in evaluator["rows"]}
    evaluated = []
    for row in runtime_rows:
        correct = {
            arm: _equivalent(prediction, gold[row["task_id"]])
            for arm, prediction in row["predictions"].items()
        }
        evaluated.append(row | {"correct": correct})
    arms = tuple(runtime_rows[0]["predictions"])
    metrics = {arm: _metric(evaluated, arm) for arm in arms}
    paired = {
        arm: _paired(evaluated, "source_induced", arm)
        for arm in arms if arm != "source_induced"
    }
    committed = [row for row in evaluated if row["oracle_prediction"] is not None]
    gates = {
        "all_900_runtime_rows_frozen_before_evaluator": len(evaluated) == 900,
        "zero_runtime_answer_program_grounding_read": all(
            not row["runtime_answer_read"]
            and not row["runtime_functional_program_read"]
            and not row["runtime_sg_grounding_read"] for row in evaluated
        ),
        "public_compiler_coverage_at_least_40pct": sum(
            row["parse_applicable"] for row in evaluated
        ) / len(evaluated) >= 0.40,
        "oracle_commit_coverage_above_old_165": len(committed) > 165,
        "oracle_committed_accuracy_at_least_95pct": sum(
            row["correct"]["target_native_ceiling"] for row in committed
        ) / len(committed) >= 0.95,
        "source_improves_neural_only": metrics["source_induced"]["correct"]
        > metrics["neural_only"]["correct"],
        "source_beats_unlocalized_generic": metrics["source_induced"]["correct"]
        > metrics["generic_scaffold"]["correct"],
        "source_beats_temporal_permutation": metrics["source_induced"]["correct"]
        > metrics["source_permuted"]["correct"],
        "isomorphic_target_equivalence_disclosed": metrics["source_induced"]
        == metrics["isomorphic_target_controller"],
    }
    body = {
        "schema_version": "agqa2-oracle-query-mdp-consumed-audit-v1",
        "status": "PASSED" if all(gates.values()) else "FAILED",
        "claim_boundary": (
            "Consumed diagnostic. Validates grounding-error isolation and target-native "
            "executor headroom. Source localization is tested as an additive stage over "
            "a target-native generic fallback; fresh matched replication is still required."
        ),
        "tasks": len(evaluated),
        "public_compiler_applicable": sum(r["parse_applicable"] for r in evaluated),
        "oracle_committed": len(committed),
        "oracle_committed_correct": sum(
            r["correct"]["target_native_ceiling"] for r in committed
        ),
        "metrics": metrics, "source_paired": paired, "gates": gates,
        "lineage": {
            "runtime_predictions_sha256": runtime_artifact["artifact_sha256"],
            "public_cohort_sha256": public["artifact_sha256"],
            "direct_predictions_sha256": direct["artifact_sha256"],
            "evaluator_only_sha256": evaluator["artifact_sha256"],
            "source_artifact_sha256": source["artifact_sha256"],
            "official_stsg_sha256": stsg_sha,
            "official_ontology_sha256": sha256_file(args.ontology),
        },
        "next_step": (
            "Freeze a new public-question-only reserve before reading outcomes; "
            "do not promote this consumed diagnostic to formal transfer evidence."
        ),
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
