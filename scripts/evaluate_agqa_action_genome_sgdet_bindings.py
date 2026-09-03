#!/usr/bin/env python3
"""Evaluate an immutable SGDET binding receipt on consumed development labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_query_grounder_v2_qualification import _answers, _sha


ALIASES = {
    "paper/notebook": "paper",
    "phone/camera": "phone",
    "sofa/couch": "sofa",
    "closet/cabinet": "closet",
    "cup/glass/bottle": "cup",
}


def normalized(value):
    if value is None:
        return None
    return ALIASES.get(str(value), str(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("SGDET binding evaluation is immutable")
    report = json.loads(args.bindings.read_text())
    if not str(report["status"]).endswith("BINDINGS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME"):
        raise ValueError("bindings were not frozen before opening labels")
    if any(report[key] for key in (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read")):
        raise ValueError("binding report violates isolation contract")
    task_ids = {str(row["task_id"]) for row in report["rows"]}
    answers = _answers(args.archive, args.entry, task_ids)
    rows = []
    for row in report["rows"]:
        task_id = str(row["task_id"])
        gold = normalized(answers[task_id])
        predicted = normalized(row["top_candidate"])
        rows.append({
            "task_id": task_id,
            "gold_entity_evaluator_only": gold,
            "predicted_entity": predicted,
            "correct": predicted == gold,
        })
    body = {
        "schema_version": "agqa-action-genome-sgdet-binding-evaluation-v1",
        "status": "CONSUMED_DEVELOPMENT_EVALUATED",
        "bindings_file_sha256": _sha(args.bindings),
        "n_tasks": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "top1_accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else 0.0,
        "outcome_opened_only_after_bindings_froze": True,
        "rows": rows,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: body[key] for key in ("n_tasks", "correct", "top1_accuracy")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
