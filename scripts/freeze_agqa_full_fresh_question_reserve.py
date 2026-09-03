#!/usr/bin/env python3
"""Freeze fresh AGQA questions while leaving their programs/answers unread."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from motif_transfer.contracts import stable_hash
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object


TASK_ID = re.compile(r"^[A-Z0-9]{5}-[0-9]+$")


def collect_task_ids(value, output: set[str]) -> None:
    if isinstance(value, dict):
        for child in value.values(): collect_task_ids(child, output)
    elif isinstance(value, list):
        for child in value: collect_task_ids(child, output)
    elif isinstance(value, str) and TASK_ID.fullmatch(value):
        output.add(value)


def rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/test_balanced.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=2400)
    parser.add_argument("--per-video", type=int, default=4)
    parser.add_argument("--seed", default="AGQA_FULL_FRESH_QUESTION_RESERVE_V1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("fresh reserve is immutable")
    consumed: set[str] = set()
    inventories = sorted(Path("configs").glob("agqa*.json")) + sorted(
        Path("runs").glob("agqa*/public_cohort*.json")
    )
    inventory_receipts = []
    for path in inventories:
        try:
            raw = path.read_bytes()
            collect_task_ids(json.loads(raw), consumed)
            inventory_receipts.append({
                "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            })
        except (OSError, json.JSONDecodeError):
            continue
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    total_rows = 0
    with zipfile.ZipFile(args.archive) as bundle, bundle.open(args.entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            total_rows += 1
            if task_id in consumed:
                continue
            # Do not access answer, program, sg_grounding, or evaluator labels.
            video = str(row["video_id"])
            question = str(row["question"])
            by_video[video].append({
                "task_id": str(task_id), "video_id": video, "question": question,
            })
    for video, rows in by_video.items():
        rows.sort(key=lambda row: rank(args.seed, row["task_id"]))
        by_video[video] = rows[:args.per_video]
    selected = []
    for video in sorted(by_video, key=lambda value: rank(args.seed, value)):
        selected.extend(by_video[video])
        if len(selected) >= args.questions:
            break
    selected = selected[:args.questions]
    if len(selected) != args.questions:
        raise ValueError(f"only {len(selected)} fresh questions available")
    args.output_dir.mkdir(parents=True)
    cohort_body = {
        "schema_version": "agqa-full-fresh-question-cohort-v1",
        "rows": selected,
        "answer_read": False, "functional_program_read": False,
        "sg_grounding_read": False,
    }
    cohort_body["cohort_sha256"] = stable_hash(cohort_body)
    (args.output_dir / "public_cohort.json").write_text(
        json.dumps(cohort_body, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    manifest_body = {
        "schema_version": "agqa-full-fresh-question-reserve-manifest-v1",
        "status": "FROZEN_BEFORE_OUTCOME_OR_PROGRAM_ACCESS",
        "claim_boundary": "UNSEEN_QUESTION_AND_OUTCOME;VIDEOS_AND_STSG_NOT_UNSEEN",
        "questions": len(selected),
        "videos": len({row["video_id"] for row in selected}),
        "historically_consumed_task_ids_excluded": len(consumed),
        "test_rows_considered_without_outcome_access": total_rows,
        "answer_read": False, "functional_program_read": False,
        "sg_grounding_read": False,
        "cohort_sha256": cohort_body["cohort_sha256"],
        "inventory_receipts": inventory_receipts,
        "selection_seed": args.seed, "per_video": args.per_video,
    }
    manifest_body["manifest_sha256"] = stable_hash(manifest_body)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest_body, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({key: manifest_body[key] for key in (
        "status", "claim_boundary", "questions", "videos",
        "historically_consumed_task_ids_excluded", "manifest_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
