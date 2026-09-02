#!/usr/bin/env python3
"""Freeze a powered, downloadable AGQA replication before video access."""

from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import zipfile

from motif_transfer.contracts import stable_hash
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object
from scripts.freeze_agqa_full_train_broad_v1 import (
    STRATA, file_sha256, parser_supervision_tasks, raw_runtime_exposed_videos,
)


REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--parser-supervision", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=180)
    parser.add_argument("--tasks-per-stratum-video", type=int, default=2)
    parser.add_argument("--salt", default="agqa-full-train-broad-powered-v4-20260902")
    parser.add_argument("--exclude-video", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("powered AGQA selection is immutable")
    exposed, exposure = raw_runtime_exposed_videos(REPO)
    supervised = parser_supervision_tasks(args.parser_supervision)
    excluded = set(args.exclude_video)
    candidates: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with zipfile.ZipFile(args.archive) as bundle, bundle.open(args.entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            task_id = str(task_id); video_id = str(row["video_id"])
            structural = str(row.get("structural") or "").casefold()
            if (structural not in STRATA or video_id in exposed or video_id in excluded
                    or task_id in supervised):
                continue
            candidates[video_id][structural].append({
                "task_id": task_id, "video_id": video_id,
                "question_sha256": stable_hash(str(row["question"])),
                "structural": structural,
            })
    k = args.tasks_per_stratum_video
    eligible = [video_id for video_id, by_stratum in candidates.items()
                if all(len(by_stratum[stratum]) >= k for stratum in STRATA)]
    ranked = sorted(eligible, key=lambda value: stable_hash({"salt": args.salt, "video_id": value}))
    if len(ranked) < args.videos:
        raise RuntimeError(f"only {len(ranked)} eligible untouched videos")
    selected = ranked[:args.videos]
    samples = [{
        "video_id": video_id,
        "video_path": str(args.video_root / f"{video_id}.mp4"),
        "rank_sha256": stable_hash({"salt": args.salt, "video_id": video_id}),
        "selected_task_ids": [
            row["task_id"] for stratum in STRATA for row in sorted(
                candidates[video_id][stratum],
                key=lambda item: stable_hash({"salt": args.salt, "task_id": item["task_id"]}),
            )[:k]
        ],
    } for video_id in selected]
    body = {
        "schema_version": "agqa-full-train-broad-powered-download-selection-v4",
        "status": "FROZEN_V4_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V4_CALLS",
        "claim": "powered replication of the unchanged anonymous game-to-AGQA Layer-B harness",
        "power_basis": {
            "consumed_diagnostic_tasks": 240,
            "source_vs_neural_wins": 6,
            "source_vs_neural_losses": 3,
            "observed_accuracy_gain_pp": 1.25,
            "method_changed_after_diagnostic": False,
        },
        "selection_rule": "HASH_RANK_UNTOUCHED_VIDEOS_WITH_TWO_TASKS_IN_EACH_OF_FIVE_PUBLIC_STRUCTURAL_STRATA",
        "selection_salt": args.salt,
        "sample_count": len(samples),
        "unique_video_count": len(samples),
        "tasks_per_video": len(STRATA) * k,
        "projected_task_count": len(samples) * len(STRATA) * k,
        "archive_path": str(args.archive),
        "archive_sha256": file_sha256(args.archive),
        "entry": args.entry,
        "raw_video_archive": {
            "url": "https://archive.org/download/charades/Charades_v1_480.zip",
            "archive_prefix": "Charades_v1_480/"
        },
        "prior_raw_runtime_exposure": exposure,
        "prior_raw_runtime_video_ids_sha256": stable_hash(sorted(exposed)),
        "eligible_video_count": len(eligible),
        "explicit_pre_video_excluded_videos": sorted(excluded),
        "answer_read_during_selection_or_freeze": False,
        "functional_program_read_during_selection_or_freeze": False,
        "scene_graph_read_during_selection_or_freeze": False,
        "source_controller_read_during_selection_or_freeze": False,
        "samples": samples,
    }
    output = {**body, "manifest_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": output["status"], "videos": len(samples),
        "projected_tasks": output["projected_task_count"],
        "already_local": sum(Path(row["video_path"]).is_file() for row in samples),
        "eligible_video_count": len(eligible),
        "manifest_sha256": output["manifest_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
