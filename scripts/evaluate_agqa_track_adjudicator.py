#!/usr/bin/env python3
"""Evaluate frozen anonymous track selections on consumed development labels."""

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
        raise FileExistsError("track adjudication evaluation is immutable")
    report = json.loads(args.grounding.read_text())
    if report["status"] != "TRACK_ADJUDICATION_FROZEN_BEFORE_DEVELOPMENT_OUTCOME":
        raise ValueError("track receipts were not frozen")
    task_ids = {str(row["task_id"]) for row in report["rows"]}
    answers = _answers(args.archive, args.entry, task_ids)
    rows = []
    for row in report["rows"]:
        gold = answers[str(row["task_id"])]
        prediction = row["selected_label_for_executor"]
        rows.append({"task_id": row["task_id"], "prediction": prediction,
                     "gold_entity_evaluator_only": gold, "correct": prediction == gold,
                     "confidence": row["confidence"]})
    decisive = [row for row in rows if row["prediction"] is not None]
    body = {"schema_version": "agqa-track-adjudicator-evaluation-v1",
            "status": "TRACK_ADJUDICATOR_PILOT_EVALUATED",
            "grounding_file_sha256": _sha(args.grounding), "n_tasks": len(rows),
            "n_decisive": len(decisive),
            "precision": sum(row["correct"] for row in decisive) / len(decisive) if decisive else 0.0,
            "coverage": len(decisive) / len(rows), "rows": rows,
            "outcome_opened_only_after_grounding_freeze": True}
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: body[key] for key in ("n_tasks", "n_decisive", "precision", "coverage")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
