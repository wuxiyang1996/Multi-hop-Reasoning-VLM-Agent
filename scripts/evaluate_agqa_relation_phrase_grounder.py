#!/usr/bin/env python3
"""Open development answers after relation-phrase receipts are frozen."""

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
        raise FileExistsError("relation phrase evaluation output is immutable")
    grounding = json.loads(args.grounding.read_text())
    if grounding["status"] != "RELATION_PHRASE_RECEIPTS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME":
        raise ValueError("relation phrase receipts were not frozen")
    task_ids = {str(row["task_id"]) for row in grounding["rows"]}
    answers = _answers(args.archive, args.entry, task_ids)
    rows = []
    for row in grounding["rows"]:
        binding = row["binding"]
        prediction = binding["label"] if binding else None
        gold = answers[str(row["task_id"])]
        rows.append({"task_id": row["task_id"], "prediction": prediction,
                     "gold_entity_evaluator_only": gold, "correct": prediction == gold,
                     "score": binding["score"] if binding else None,
                     "margin": (binding["score"] - binding["runner_up_score"] if binding else None)})
    body = {
        "schema_version": "agqa-relation-phrase-grounder-evaluation-v1",
        "status": "RELATION_PHRASE_PILOT_EVALUATED",
        "grounding_file_sha256": _sha(args.grounding), "n_tasks": len(rows),
        "accuracy_when_bound": (sum(x["correct"] for x in rows) /
                                sum(x["prediction"] is not None for x in rows)
                                if any(x["prediction"] is not None for x in rows) else 0.0),
        "coverage": sum(x["prediction"] is not None for x in rows) / len(rows),
        "rows": rows, "outcome_opened_only_after_grounding_freeze": True,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: body[key] for key in ("n_tasks", "accuracy_when_bound", "coverage")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
