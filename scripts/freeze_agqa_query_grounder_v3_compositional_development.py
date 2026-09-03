#!/usr/bin/env python3
"""Freeze AG-train compositional object-query rows for global calibration."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import zipfile
import io


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402
from scripts.freeze_agqa_query_grounder_v2_task_disjoint_qualification import (  # noqa: E402
    _flag, _prior_task_ids, _sha256,
)


SALT = "agqa-query-grounder-v3-agtrain-compositional-calibration-v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--videos-per-slice", type=int, default=60)
    parser.add_argument("--tasks-per-video", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("V3 development freeze is immutable")

    prior_tasks, prior_audit = _prior_task_ids()
    local = {path.stem: path for path in args.video_root.glob("*.mp4")}
    candidates = {
        "more_steps": defaultdict(list),
        "novel_composition": defaultdict(list),
    }
    with zipfile.ZipFile(args.archive) as bundle, bundle.open(args.entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            task_id = str(task_id); video_id = str(row["video_id"])
            if (
                task_id in prior_tasks or video_id not in local
                or str(row.get("structural", "")).casefold() != "query"
                or str(row.get("semantic", "")).casefold() != "object"
            ):
                continue
            slices = []
            if _flag(row, "more_steps"): slices.append("more_steps")
            if _flag(row, "novel_comp"): slices.append("novel_composition")
            for slice_name in slices:
                question = str(row["question"])
                candidates[slice_name][video_id].append({
                    "task_id": task_id,
                    "video_id": video_id,
                    "question": question,
                    "question_sha256": stable_hash(question),
                    "structural": "query",
                    "semantic": "object",
                    "answer_type": str(row.get("ans_type") or "unknown").casefold(),
                    "official_evaluation_slice": slice_name,
                    "video_path": str(local[video_id]),
                })

    selected_rows = []
    selected_videos: set[str] = set()
    slice_summary = {}
    for slice_name in ("more_steps", "novel_composition"):
        eligible = [
            video_id for video_id, rows in candidates[slice_name].items()
            if video_id not in selected_videos and len(rows) >= args.tasks_per_video
        ]
        eligible.sort(key=lambda video_id: stable_hash({
            "salt": SALT, "slice": slice_name, "video_id": video_id,
        }))
        chosen = eligible[:args.videos_per_slice]
        if len(chosen) != args.videos_per_slice:
            raise RuntimeError(f"only {len(chosen)} development videos for {slice_name}")
        for video_id in chosen:
            rows = sorted(candidates[slice_name][video_id], key=lambda row: stable_hash({
                "salt": SALT, "slice": slice_name, "task_id": row["task_id"],
            }))
            selected_rows.extend(rows[:args.tasks_per_video])
        selected_videos.update(chosen)
        slice_summary[slice_name] = {
            "eligible_videos": len(eligible),
            "selected_videos": len(chosen),
            "selected_tasks": len(chosen) * args.tasks_per_video,
        }

    video_receipts = [{
        "video_id": video_id,
        "video_path": str(local[video_id]),
        "video_sha256": _sha256(local[video_id]),
    } for video_id in sorted(selected_videos)]
    public = {
        "schema_version": "agqa-query-grounder-v3-compositional-development-public-v1",
        "status": "DEVELOPMENT_FROZEN_BEFORE_PARSER_GROUNDER_OR_OUTCOMES",
        "source_split": "official_balanced_train_action_genome_train",
        "rows": selected_rows,
        "video_receipts": video_receipts,
        "answers_projected": False,
        "functional_programs_projected": False,
        "scene_graph_grounding_projected": False,
        "source_controller_read": False,
    }
    public["cohort_sha256"] = stable_hash(public)
    expected_videos = 2 * args.videos_per_slice
    expected_tasks = expected_videos * args.tasks_per_video
    gates = {
        "video_count": len(selected_videos) == expected_videos,
        "task_count": len(selected_rows) == expected_tasks,
        "tasks_unique": len({row["task_id"] for row in selected_rows}) == expected_tasks,
        "tasks_prior_artifact_disjoint": not ({row["task_id"] for row in selected_rows} & prior_tasks),
        "videos_disjoint_between_slices": len(selected_videos) == expected_videos,
        "videos_content_addressed": all(row["video_sha256"] for row in video_receipts),
    }
    manifest = {
        "schema_version": "agqa-query-grounder-v3-compositional-development-freeze-v1",
        "status": "V3_COMPOSITIONAL_DEVELOPMENT_FROZEN" if all(gates.values()) else "FREEZE_FAILED",
        "cohort_sha256": public["cohort_sha256"],
        "archive_sha256": _sha256(args.archive),
        "archive_entry": args.entry,
        "selection_salt": SALT,
        "selection": {"videos": len(selected_videos), "tasks": len(selected_rows), "slices": slice_summary},
        "prior_task_audit": prior_audit,
        "gates": gates,
        "answers_used": False,
        "functional_programs_used": False,
        "scene_graph_grounding_used": False,
        "claim_eligible": False,
        "reason_not_claim_eligible": "Grounder checkpoints were trained on Action Genome train.",
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "public_cohort.json").write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": manifest["status"], "videos": len(selected_videos),
        "tasks": len(selected_rows), "slices": slice_summary,
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
