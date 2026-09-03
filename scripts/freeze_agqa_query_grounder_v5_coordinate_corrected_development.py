#!/usr/bin/env python3
"""Freeze fresh AGQA-train videos for coordinate-corrected grounder development.

The selector may inspect public question metadata and official generalization
metadata, but never projects answers, functional programs, scene graphs, or
target outcomes into the public cohort.  Videos with any prior raw-frame
exposure are excluded.  Selection deliberately precedes video acquisition:
an absent local file is transport state, not a scientific eligibility gate.
Historical config allocations are reported separately because they are not
evidence that a visual model observed the video.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import sys
import zipfile


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402
from scripts.freeze_agqa_query_grounder_v2_qualification import (  # noqa: E402
    _allocated_config_videos,
    _exact_raw_runtime_exposed_videos,
    _sha256,
)


SALT = "agqa-query-grounder-v5-coordinate-corrected-development-v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--archive-url",
        default=(
            "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/"
            "charades/Charades_v1_480.zip"
        ),
    )
    parser.add_argument("--archive-prefix", default="Charades_v1_480/")
    parser.add_argument("--videos", type=int, default=120)
    parser.add_argument("--tasks-per-video", type=int, default=2)
    parser.add_argument("--salt", default=SALT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("V5 development freeze is immutable")
    if args.videos < 1 or args.tasks_per_video < 1:
        raise ValueError("cohort sizes must be positive")

    raw_exposed, raw_exposure_audit = _exact_raw_runtime_exposed_videos(REPO)
    allocated = _allocated_config_videos()
    # Prior allocation is reported for audit, but is not an exposure event.
    # This cohort is development-only, so excluding every historical frozen or
    # aborted config would incorrectly collapse thousands of never-observed
    # AGQA-train videos.  Actual raw-frame exposure remains a hard exclusion.
    excluded_videos = raw_exposed
    local = {path.stem: path for path in args.video_root.glob("*.mp4")}
    candidates: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(args.archive) as bundle, bundle.open(args.entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            task_id = str(task_id)
            video_id = str(row["video_id"])
            if (
                video_id in excluded_videos
                or str(row.get("structural", "")).casefold() != "query"
                or str(row.get("semantic", "")).casefold() != "object"
            ):
                continue
            question = str(row["question"])
            candidates[video_id].append({
                "task_id": task_id,
                "video_id": video_id,
                "question": question,
                "question_sha256": stable_hash(question),
                "structural": "query",
                "semantic": "object",
                "answer_type": str(row.get("ans_type") or "unknown").casefold(),
                "video_path": str(args.video_root / f"{video_id}.mp4"),
            })

    eligible = [
        video_id for video_id, rows in candidates.items()
        if len(rows) >= args.tasks_per_video
    ]
    eligible.sort(key=lambda video_id: stable_hash({
        "salt": args.salt,
        "video_id": video_id,
    }))
    chosen = eligible[:args.videos]
    if len(chosen) != args.videos:
        raise RuntimeError(f"only {len(chosen)} fresh broad object-query videos")
    selected_rows: list[dict] = []
    for video_id in chosen:
        rows = sorted(candidates[video_id], key=lambda row: stable_hash({
            "salt": args.salt,
            "task_id": row["task_id"],
        }))
        selected_rows.extend(rows[:args.tasks_per_video])
    selected_videos = set(chosen)

    video_selections = [{
        "video_id": video_id,
        "video_path": str(args.video_root / f"{video_id}.mp4"),
        "video_present_at_selection": video_id in local,
    } for video_id in sorted(selected_videos)]
    public = {
        "schema_version": "agqa-query-grounder-v5-development-public-v1",
        "status": "DEVELOPMENT_SELECTION_FROZEN_BEFORE_VIDEO_ACQUISITION_PARSER_GROUNDER_OR_OUTCOMES",
        "source_split": "official_balanced_train_action_genome_train",
        "rows": selected_rows,
        "video_selections": video_selections,
        "answers_projected": False,
        "functional_programs_projected": False,
        "scene_graph_grounding_projected": False,
        "source_controller_read": False,
        "target_outcome_read": False,
    }
    public["cohort_sha256"] = stable_hash(public)
    expected_videos = args.videos
    expected_tasks = expected_videos * args.tasks_per_video
    chosen_video_ids = {row["video_id"] for row in video_selections}
    chosen_task_ids = {row["task_id"] for row in selected_rows}
    gates = {
        "video_count": len(chosen_video_ids) == expected_videos,
        "task_count": len(selected_rows) == expected_tasks,
        "tasks_unique": len(chosen_task_ids) == expected_tasks,
        "videos_prior_raw_runtime_disjoint": not (chosen_video_ids & raw_exposed),
        "videos_unique": len(chosen_video_ids) == expected_videos,
        "video_paths_frozen_before_acquisition": all(
            row["video_path"] for row in video_selections
        ),
    }
    samples = []
    rows_by_video: dict[str, list[dict]] = defaultdict(list)
    for row in selected_rows:
        rows_by_video[row["video_id"]].append(row)
    for video in video_selections:
        samples.append({
            **video,
            "selected_task_ids": sorted(
                row["task_id"] for row in rows_by_video[video["video_id"]]
            ),
        })
    download_selection = {
        "schema_version": "agqa-query-grounder-v5-development-download-selection-v1",
        "status": "FROZEN_V5_DEVELOPMENT_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_OUTCOME_ACCESS",
        "claim_boundary": "grounder development only; never transfer evidence",
        "archive_path": str(args.archive),
        "archive_sha256": _sha256(args.archive),
        "entry": args.entry,
        "raw_video_archive": {
            "archive_prefix": args.archive_prefix,
            "url": args.archive_url,
        },
        "sample_count": len(samples),
        "samples": samples,
        "public_cohort_sha256": public["cohort_sha256"],
        "prior_raw_video_exposure_audit_sha256": raw_exposure_audit.get(
            "report_sha256"
        ),
        "answer_read_during_selection_or_freeze": False,
        "functional_program_read_during_selection_or_freeze": False,
        "official_scene_graph_read_during_selection_or_freeze": False,
        "target_outcome_read_during_selection_or_freeze": False,
    }
    download_selection["manifest_sha256"] = stable_hash(download_selection)
    manifest = {
        "schema_version": "agqa-query-grounder-v5-development-freeze-v1",
        "status": "V5_COORDINATE_CORRECTED_DEVELOPMENT_SELECTION_FROZEN" if all(gates.values()) else "FREEZE_FAILED",
        "cohort_sha256": public["cohort_sha256"],
        "archive_sha256": _sha256(args.archive),
        "archive_entry": args.entry,
        "selection_salt": args.salt,
        "selection": {
            "videos": len(chosen_video_ids),
            "tasks": len(selected_rows),
            "eligible_videos": len(eligible),
            "sampling_frame": "broad structural=query semantic=object",
        },
        "historical_task_id_mentions_used_as_exposure_gate": False,
        "prior_raw_video_exposure_audit": raw_exposure_audit,
        "prior_allocated_video_count": len(allocated),
        "selected_prior_allocation_overlap_count": len(chosen_video_ids & allocated),
        "selected_prior_allocation_overlap_sha256": stable_hash(
            sorted(chosen_video_ids & allocated)
        ),
        "excluded_video_ids_sha256": stable_hash(sorted(excluded_videos)),
        "local_video_availability_is_eligibility_gate": False,
        "selected_present_at_selection_count": sum(
            bool(row["video_present_at_selection"]) for row in video_selections
        ),
        "selected_missing_at_selection_count": sum(
            not bool(row["video_present_at_selection"]) for row in video_selections
        ),
        "download_selection_manifest_sha256": download_selection["manifest_sha256"],
        "gates": gates,
        "authority": {
            "answers_used": False,
            "functional_programs_used": False,
            "official_scene_graph_used": False,
            "source_controller_used": False,
            "target_outcomes_used": False,
            "selection_features": [
                "task_id", "video_id", "question", "structural", "semantic",
                "ans_type",
            ],
        },
        "claim_eligible": False,
        "reason_not_claim_eligible": (
            "Grounder development on AGQA train; Action Genome checkpoints may overlap."
        ),
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "public_cohort.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "download_selection.json").write_text(
        json.dumps(download_selection, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": manifest["status"],
        "videos": len(chosen_video_ids),
        "tasks": len(selected_rows),
        "eligible_videos": len(eligible),
        "raw_exposed_excluded": len(raw_exposed),
        "allocated_reported": len(allocated),
        "selected_present_at_selection": manifest["selected_present_at_selection_count"],
        "selected_missing_at_selection": manifest["selected_missing_at_selection_count"],
        "download_selection_manifest_sha256": download_selection["manifest_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
