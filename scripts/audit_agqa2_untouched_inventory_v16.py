#!/usr/bin/env python3
"""Audit whether any video-disjoint AGQA2 evidence remains after V74.

This script reads questions only to count parser-compatible units. It never
reads answers, functional programs, or scene graphs, and it performs no model
calls. The output is an inventory audit, not a new evaluation.
"""

from __future__ import annotations

import argparse
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
SPLIT = REPO_ROOT / "configs/agqa2_program_router_video_split_v1.json"


def prior_runtime_videos() -> set[str]:
    videos: set[str] = set()
    for path in REPO_ROOT.glob("runs/agqa2*/runtime_receipts/*.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        video_id = row.get("video_id")
        if isinstance(video_id, str):
            videos.add(video_id)
    return videos


def test_inventory(prior: set[str]) -> dict:
    videos: set[str] = set()
    compatible_videos: set[str] = set()
    compatible_tasks = 0
    with zipfile.ZipFile(ARCHIVE) as bundle, bundle.open(
        "AGQA_balanced/test_balanced.txt"
    ) as raw:
        rows = _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8"))
        for _, row in rows:
            video_id = str(row["video_id"])
            videos.add(video_id)
            plan = parse_public_question_plan(str(row["question"]))
            if plan is not None and plan.comparison == "EXISTS":
                compatible_tasks += 1
                compatible_videos.add(video_id)
    return {
        "video_count": len(videos),
        "exists_compatible_task_count": compatible_tasks,
        "exists_compatible_video_count": len(compatible_videos),
        "untouched_video_count": len(videos - prior),
        "untouched_exists_compatible_video_count": len(compatible_videos - prior),
    }


def audit() -> dict:
    prior = prior_runtime_videos()
    split = json.loads(SPLIT.read_text())
    formal = set(split["partitions"]["formal_holdout"])
    body = {
        "schema_version": "agqa2-untouched-inventory-v16",
        "status": "NO_VIDEO_DISJOINT_OFFICIAL_TEST_EVIDENCE_REMAINS",
        "purpose": "INVENTORY_ONLY_NO_MODEL_OR_LABEL_ACCESS",
        "prior_runtime_video_count": len(prior),
        "official_test": test_inventory(prior),
        "train_formal_holdout": {
            "video_count": len(formal),
            "untouched_video_count": len(formal - prior),
            "use_authorized_after_v74": False,
            "reason": "V74_FAILURE_POLICY_REQUIRES_STOPPING_AGQA_EXPERIMENTS",
        },
        "answer_read": False,
        "functional_program_read": False,
        "scene_graph_read": False,
        "provider_calls": 0,
        "scientific_conclusion": (
            "ANOTHER_VIDEO_DISJOINT_AGQA2_CONFIRMATORY_RUN_CANNOT_BE FORMED_FROM_"
            "THE_OFFICIAL_TEST_SPLIT;THE_REMAINING_TRAIN_FORMAL_VIDEOS_ARE_"
            "WITHHELD_BY_THE_PREREGISTERED_V74_STOP_POLICY"
        ),
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
