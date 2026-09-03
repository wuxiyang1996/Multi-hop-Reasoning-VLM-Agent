#!/usr/bin/env python3
"""Freeze every prior AGQA cohort video before selecting the final reserve."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("AGQA prior-video exclusion ledger is immutable")
    files = []
    excluded_videos = set()
    excluded_tasks = set()
    for path in sorted(args.runs_root.rglob("*cohort*.json")):
        if "agqa" not in str(path).casefold() or path.resolve() == args.output.resolve():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            continue
        videos = sorted({
            str(row["video_id"]) for row in rows
            if isinstance(row, dict) and row.get("video_id") is not None
        })
        tasks = sorted({
            str(row["task_id"]) for row in rows
            if isinstance(row, dict) and row.get("task_id") is not None
        })
        if not videos:
            continue
        excluded_videos.update(videos); excluded_tasks.update(tasks)
        files.append({
            "path": str(path.resolve()), "file_sha256": _sha256(path),
            "status": payload.get("status"), "cohort_sha256": payload.get("cohort_sha256"),
            "videos": len(videos), "tasks": len(tasks),
            "video_ids_sha256": stable_hash(videos), "task_ids_sha256": stable_hash(tasks),
        })
    body = {
        "schema_version": "agqa-prior-video-exclusion-ledger-v17",
        "status": "ALL_EXISTING_AGQA_COHORT_VIDEOS_FROZEN_AS_EXCLUDED",
        "runs_root": str(args.runs_root.resolve()), "cohort_files": files,
        "cohort_file_count": len(files),
        "excluded_video_ids": sorted(excluded_videos),
        "excluded_task_ids": sorted(excluded_tasks),
        "excluded_videos": len(excluded_videos), "excluded_tasks": len(excluded_tasks),
        "answers_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "target_outcome_read": False,
    }
    body["ledger_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "cohort_file_count": len(files),
        "excluded_videos": len(excluded_videos), "excluded_tasks": len(excluded_tasks),
        "ledger_sha256": body["ledger_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
