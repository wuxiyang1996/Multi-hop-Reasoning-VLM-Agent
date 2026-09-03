#!/usr/bin/env python3
"""Open consumed development labels after local role receipts are frozen."""

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
    parser.add_argument("--confidence-threshold", type=float, default=.70)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("role-verifier evaluation output is immutable")
    grounding = json.loads(args.grounding.read_text())
    if grounding["status"] != "LOCAL_ROLE_RECEIPTS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME":
        raise ValueError("role receipts were not frozen before outcome access")
    task_ids = {str(row["task_id"]) for row in grounding["rows"]}
    answers = _answers(args.archive, args.entry, task_ids)
    rows = []
    true_unique = 0
    decisive = 0
    duplicates = 0
    for row in grounding["rows"]:
        supported = [candidate for candidate in row["candidates"]
                     if candidate["status"] == "SUPPORTED"
                     and float(candidate["confidence"]) >= args.confidence_threshold]
        labels = sorted({candidate["label"] for candidate in supported})
        duplicates += max(0, len(supported) - len(labels))
        gold = answers[str(row["task_id"])]
        is_unique = len(labels) == 1
        decisive += int(is_unique)
        true_unique += int(is_unique and labels[0] == gold)
        rows.append({"task_id": row["task_id"], "supported_labels": labels,
                     "gold_entity_evaluator_only": gold,
                     "unique_supported": is_unique,
                     "unique_correct": is_unique and labels[0] == gold})
    precision = true_unique / decisive if decisive else 0.0
    body = {
        "schema_version": "agqa-local-role-verifier-evaluation-v1",
        "status": "LOCAL_ROLE_PILOT_EVALUATED",
        "grounding_file_sha256": _sha(args.grounding),
        "confidence_threshold": args.confidence_threshold,
        "n_tasks": len(rows), "n_decisive": decisive,
        "unique_role_precision": precision,
        "unique_role_coverage": decisive / len(rows) if rows else 0.0,
        "duplicate_supported_track_count": duplicates,
        "rows": rows,
        "outcome_opened_only_after_grounding_freeze": True,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: body[key] for key in (
        "n_tasks", "n_decisive", "unique_role_precision",
        "unique_role_coverage", "duplicate_supported_track_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
