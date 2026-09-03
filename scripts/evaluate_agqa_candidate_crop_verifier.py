#!/usr/bin/env python3
"""Open consumed development labels after candidate receipts freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_query_grounder_v2_qualification import _answers, _sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("candidate-crop evaluation is immutable")
    report = json.loads(args.grounding.read_text())
    if report["status"] != "CANDIDATE_CROP_RECEIPTS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME":
        raise ValueError("candidate receipts were not frozen")
    task_ids = {str(value["task_id"]) for value in report["rows"]}
    answers = _answers(args.archive, args.entry, task_ids)
    rows = []; supported_total = 0; supported_correct = 0
    for row in report["rows"]:
        gold = answers[str(row["task_id"])]
        supported = [value for value in row["candidates"] if value["status"] == "SUPPORTED"]
        supported.sort(key=lambda value: (
            -(float(value["confidence"]) * float(value["detection_max_confidence"])),
            -float(value["confidence"]), -float(value["detection_max_confidence"]),
            str(value["candidate_label"])))
        top = supported[0]["candidate_label"] if supported else None
        supported_total += len(supported)
        supported_correct += sum(value["candidate_label"] == gold for value in supported)
        rows.append({"task_id": row["task_id"], "gold_entity_evaluator_only": gold,
                     "supported_labels": [value["candidate_label"] for value in supported],
                     "gold_supported": any(value["candidate_label"] == gold for value in supported),
                     "top_supported": top, "top_correct": top == gold})
    body = {"schema_version": "agqa-candidate-crop-evaluation-v1",
            "status": "CANDIDATE_CROP_PILOT_EVALUATED",
            "grounding_file_sha256": _sha(args.grounding), "n_tasks": len(rows),
            "top1_rule": "max(verifier_confidence * detector_max_confidence)",
            "supported_precision": supported_correct / supported_total if supported_total else 0.0,
            "gold_supported_recall": sum(x["gold_supported"] for x in rows) / len(rows),
            "top1_accuracy": sum(x["top_correct"] for x in rows) / len(rows),
            "rows": rows, "outcome_opened_only_after_grounding_freeze": True}
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: body[key] for key in (
        "n_tasks", "supported_precision", "gold_supported_recall", "top1_accuracy")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
