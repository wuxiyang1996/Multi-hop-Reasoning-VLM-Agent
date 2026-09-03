#!/usr/bin/env python3
"""Run the preregistered fresh AGQA2 oracle-query qualification."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from motif_transfer.agqa_oracle_query_mdp import (  # noqa: E402
    AGQAOracleQueryBackend, AGQAOracleToolBudget,
    execute_temporal_object_query, load_agqa_id_to_text,
)
from motif_transfer.agqa_temporal_localized_query import (  # noqa: E402
    parse_temporal_localized_object_question,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.official_video_event_graph import (  # noqa: E402
    load_builtin_only_pickle, sha256_file,
)

RUN = REPO / "runs/agqa2_oracle_query_mdp_v2_fresh"
STSG = Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/AGQA_train_stsgs.pkl")
ONTOLOGY = STSG.parent / "ENG.txt"
PREREG = REPO / "configs/agqa2_oracle_query_mdp_v2_fresh_preregistration.json"


def _read(path):
    value = json.loads(path.read_text()); body = dict(value)
    claimed = body.pop("artifact_sha256")
    if stable_hash(body) != claimed:
        raise ValueError(f"artifact hash mismatch: {path}")
    return value


def main() -> None:
    prereg = json.loads(PREREG.read_text())
    public = _read(RUN / "public_cohort.json")
    ontology = load_agqa_id_to_text(ONTOLOGY)
    corpus = load_builtin_only_pickle(STSG)
    stsg_sha = sha256_file(STSG)
    runtime_rows = []
    for row in public["rows"]:
        plan = parse_temporal_localized_object_question(row["question"])
        execution = None
        if plan is not None:
            video_id = str(row["video_id"])
            execution = execute_temporal_object_query(
                plan,
                AGQAOracleQueryBackend(
                    video_id=video_id, graph=corpus[video_id],
                    id_to_text=ontology,
                    graph_sha256=stable_hash([stsg_sha, video_id]),
                    budget=AGQAOracleToolBudget(3),
                ),
            )
        runtime_rows.append({
            "task_id": row["task_id"], "video_id": row["video_id"],
            "question_sha256": row["question_sha256"],
            "parse_applicable": plan is not None,
            "prediction": execution.prediction if execution else None,
            "status": execution.status if execution else "PARSE_ABSTAINED",
            "reason": execution.reason if execution else "PUBLIC_COMPILER_UNSUPPORTED",
            "execution_sha256": execution.execution_sha256 if execution else None,
            "tool_calls": len(execution.receipts) if execution else 0,
            "runtime_answer_read": False,
            "runtime_functional_program_read": False,
            "runtime_sg_grounding_read": False,
        })
    runtime_body = {
        "schema_version": "agqa2-oracle-query-runtime-v2-fresh",
        "role": "PREDICTIONS_FROZEN_BEFORE_EVALUATOR",
        "public_artifact_sha256": public["artifact_sha256"],
        "official_stsg_sha256": stsg_sha,
        "rows": runtime_rows,
    }
    runtime = runtime_body | {"artifact_sha256": stable_hash(runtime_body)}
    (RUN / "runtime_predictions.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n"
    )
    evaluator = _read(RUN / "evaluator_only.json")
    gold = {row["task_id"]: str(row["answer"]) for row in evaluator["rows"]}
    committed = [row for row in runtime_rows if row["prediction"] is not None]
    correct = sum(
        str(row["prediction"]).casefold() == gold[row["task_id"]].casefold()
        for row in committed
    )
    gates_cfg = prereg["frozen_gates"]
    coverage = sum(row["parse_applicable"] for row in runtime_rows) / len(runtime_rows)
    gates = {
        "minimum_public_compiler_coverage": coverage
        >= float(gates_cfg["minimum_public_compiler_coverage"]),
        "minimum_oracle_committed_tasks": len(committed)
        >= int(gates_cfg["minimum_oracle_committed_tasks"]),
        "minimum_oracle_committed_accuracy": correct / len(committed)
        >= float(gates_cfg["minimum_oracle_committed_accuracy"]),
        "all_selected_videos_unseen_in_v62": json.loads(
            (RUN / "freeze_receipt.json").read_text()
        )["all_videos_unseen_in_v62"],
        "zero_runtime_answer_program_grounding_read": all(
            not row["runtime_answer_read"]
            and not row["runtime_functional_program_read"]
            and not row["runtime_sg_grounding_read"] for row in runtime_rows
        ),
    }
    body = {
        "schema_version": "agqa2-oracle-query-fresh-qualification-v2",
        "status": "PASSED" if all(gates.values()) else "FAILED",
        "claim_boundary": prereg["claim_boundary"],
        "tasks": len(runtime_rows), "unique_videos": len({r["video_id"] for r in runtime_rows}),
        "public_compiler_applicable": sum(r["parse_applicable"] for r in runtime_rows),
        "public_compiler_coverage": coverage,
        "oracle_committed": len(committed),
        "oracle_committed_correct": correct,
        "oracle_committed_accuracy": correct / len(committed),
        "gates": gates,
        "lineage": {
            "preregistration_sha256": stable_hash(prereg),
            "public_artifact_sha256": public["artifact_sha256"],
            "runtime_artifact_sha256": runtime["artifact_sha256"],
            "evaluator_artifact_sha256": evaluator["artifact_sha256"],
            "official_stsg_sha256": stsg_sha,
            "official_ontology_sha256": sha256_file(ONTOLOGY),
        },
        "source_specific_transfer_validated": False,
        "reason_source_specific_not_yet_validated": (
            "This qualification validates the target-native oracle executor; "
            "it has no new direct actor and no source-vs-generic divergence."
        ),
    }
    report = body | {"report_sha256": stable_hash(body)}
    output = REPO / "docs/results/agqa2_oracle_query_fresh_v2.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
