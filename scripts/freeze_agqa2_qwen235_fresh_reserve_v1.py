#!/usr/bin/env python3
"""Freeze answer-blind fresh-video AGQA EXISTS tasks for Qwen235 evaluation."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_active_frame_grounder import parse_public_question_plan  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402


ARCHIVE = Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip")
ENTRY = "AGQA_balanced/test_balanced.txt"
VIDEO_ROOT = Path("/fs/gamma-projects/vlm-robot/datasets/STAR-official/videos/charades")
OUTPUT = REPO_ROOT / "configs/agqa2_qwen235_fresh_reserve_v1_selection.json"
COUNT = 30
NONCE = "agqa2-qwen235-selective-authorizer-v1-fresh-video-reserve"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prior_exposure() -> tuple[set[str], set[str]]:
    videos: set[str] = set()
    tasks: set[str] = set()
    paths = list(REPO_ROOT.glob("runs/agqa2*/runtime_receipts/*.json"))
    paths += list((REPO_ROOT / "docs/results").glob("agqa2*.json"))
    for path in paths:
        try:
            root = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        stack = [root]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if isinstance(value.get("video_id"), str):
                    videos.add(value["video_id"])
                if isinstance(value.get("task_id"), str):
                    tasks.add(value["task_id"])
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    return videos, tasks


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"selection is immutable once written: {OUTPUT}")
    prior_videos, prior_tasks = _prior_exposure()
    candidates = []
    with zipfile.ZipFile(ARCHIVE) as bundle, bundle.open(ENTRY) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8")
        for task_id, row in _iter_top_level_object(text):
            video_id = str(row["video_id"])
            if video_id in prior_videos or task_id in prior_tasks:
                continue
            plan = parse_public_question_plan(str(row["question"]))
            if (
                plan is None
                or plan.obligation_kind != "RELATION_RECURRENT"
                or plan.comparison != "EXISTS"
            ):
                continue
            rank = stable_hash({"nonce": NONCE, "task_id": task_id})
            candidates.append({
                "task_id": task_id,
                "video_id": video_id,
                "question_sha256": stable_hash(str(row["question"])),
                "public_parser_plan_sha256": stable_hash(plan.as_dict()),
                "rank_sha256": rank,
                "video_path": str(VIDEO_ROOT / f"{video_id}.mp4"),
            })
    # At most one task per video prevents pseudoreplication.
    chosen = []
    used_videos = set()
    for row in sorted(candidates, key=lambda item: item["rank_sha256"]):
        if row["video_id"] in used_videos:
            continue
        used_videos.add(row["video_id"])
        chosen.append(row)
        if len(chosen) == COUNT:
            break
    if len(chosen) != COUNT:
        raise ValueError(f"only {len(chosen)} fresh unique-video EXISTS tasks available")

    # Selection is now fixed. Hashing programs is evaluator-integrity metadata;
    # it cannot alter membership or order and the answers remain unread.
    selected_ids = {row["task_id"] for row in chosen}
    programs = {}
    with zipfile.ZipFile(ARCHIVE) as bundle, bundle.open(ENTRY) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8")
        for task_id, row in _iter_top_level_object(text):
            if task_id in selected_ids:
                programs[task_id] = stable_hash(str(row["program"]))
                if len(programs) == len(selected_ids):
                    break
    samples = [row | {"program_sha256": programs[row["task_id"]]} for row in chosen]
    body = {
        "schema_version": "agqa2-qwen235-fresh-reserve-selection-v1",
        "status": "FROZEN_V65_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V65_CALLS",
        "split": "official_test_fresh_video_relation_exists",
        "selection_nonce": NONCE,
        "selection_rule": "HASH_RANK_ONE_RELATION_RECURRENT_EXISTS_QUESTION_PER_PREVIOUSLY_UNEXPOSED_VIDEO",
        "sample_count": len(samples),
        "unique_video_count": len(used_videos),
        "archive_path": str(ARCHIVE),
        "archive_sha256": _sha256(ARCHIVE),
        "entry": ENTRY,
        "raw_video_archive": {
            "url": "https://archive.org/download/charades/Charades_v1_480.zip",
            "archive_prefix": "Charades_v1_480/"
        },
        "prior_runtime_video_count": len(prior_videos),
        "prior_runtime_task_count": len(prior_tasks),
        "prior_runtime_exposure_sha256": stable_hash({
            "video_ids": sorted(prior_videos), "task_ids": sorted(prior_tasks)
        }),
        "answer_read_during_selection_or_freeze": False,
        "program_read_only_after_membership_froze": True,
        "scene_graph_read_during_selection_or_freeze": False,
        "source_identity_read_during_selection_or_freeze": False,
        "samples": samples,
    }
    output = body | {"manifest_sha256": stable_hash(body)}
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": output["status"], "sample_count": len(samples),
        "unique_video_count": len(used_videos),
        "already_local": sum(Path(row["video_path"]).is_file() for row in samples),
        "manifest_sha256": output["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
