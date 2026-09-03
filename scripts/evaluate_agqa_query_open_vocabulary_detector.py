#!/usr/bin/env python3
"""Open consumed development labels only after detector receipts are frozen."""

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
        raise FileExistsError("detector evaluation output is immutable")
    grounding = json.loads(args.grounding.read_text())
    if grounding["status"] != "DETECTOR_RECEIPTS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME":
        raise ValueError("detector receipts were not frozen")
    task_ids = {str(row["task_id"]) for row in grounding["rows"]}
    answers = _answers(args.archive, args.entry, task_ids)
    rows = []
    for row in grounding["rows"]:
        labels = sorted({track["canonical_label"] for track in row["tracks"]})
        gold = answers[str(row["task_id"])]
        rows.append({"task_id": row["task_id"], "gold_entity_evaluator_only": gold,
                     "detected_labels": labels, "gold_present": gold in labels})
    recall = sum(row["gold_present"] for row in rows) / len(rows)
    body = {"schema_version": "agqa-query-open-vocabulary-detector-evaluation-v1",
            "status": "DETECTOR_PILOT_EVALUATED", "grounding_file_sha256": _sha(args.grounding),
            "rows": rows, "entity_recall": recall,
            "outcome_opened_only_after_grounding_freeze": True}
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"entity_recall": recall, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
