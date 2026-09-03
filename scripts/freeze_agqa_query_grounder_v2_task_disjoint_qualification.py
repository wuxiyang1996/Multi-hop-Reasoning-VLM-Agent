#!/usr/bin/env python3
"""Freeze never-mentioned AGQA compositional tasks on previously viewed videos."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402


TASK_ID = re.compile(r'"task_id"\s*:\s*"([^"]+)"')
SALT = "agqa-query-grounder-v2-task-disjoint-compositional-qualification-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prior_task_ids() -> tuple[set[str], dict]:
    paths = sorted(set(
        REPO.glob("configs/agqa2*.json")
    ) | set(
        (REPO / "docs/results").glob("agqa2*.json")
    ) | set(
        REPO.glob("runs/agqa2*/**/*.json")
    ))
    task_ids: set[str] = set()
    scanned = 0
    for path in paths:
        try:
            with path.open(encoding="utf-8", errors="ignore") as stream:
                for line in stream:
                    task_ids.update(TASK_ID.findall(line))
        except OSError:
            continue
        scanned += 1
    return task_ids, {
        "scanned_json_file_count": scanned,
        "prior_mentioned_task_id_count": len(task_ids),
        "prior_mentioned_task_ids_sha256": stable_hash(sorted(task_ids)),
    }


def _flag(row: dict, key: str) -> bool:
    try:
        return int(row.get(key, 0) or 0) == 1
    except (TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/test_balanced.txt")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--exposure-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--videos-per-slice", type=int, default=20)
    parser.add_argument("--tasks-per-video", type=int, default=2)
    parser.add_argument("--semantic-kind")
    parser.add_argument("--salt", default=SALT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("task-disjoint qualification freeze is immutable")

    exposure = json.loads(args.exposure_audit.read_text())
    if exposure.get("status") != "NO_RAW_VIDEO_FRESH_OFFICIAL_SPLIT_REMAINS":
        raise ValueError("task-disjoint fallback is authorized only after exact exposure exhaustion")
    exposed_videos = set(exposure["exposed_official_video_ids"])
    prior_tasks, prior_task_audit = _prior_task_ids()
    local = {path.stem: path for path in args.video_root.glob("*.mp4")}
    candidates: dict[str, dict[str, list[dict]]] = {
        "more_steps": defaultdict(list),
        "novel_composition": defaultdict(list),
    }
    with zipfile.ZipFile(args.archive) as bundle, bundle.open(args.entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            task_id = str(task_id)
            video_id = str(row["video_id"])
            if (
                task_id in prior_tasks
                or video_id not in local
                or str(row.get("structural", "")).casefold() != "query"
                or (
                    args.semantic_kind is not None
                    and str(row.get("semantic", "")).casefold()
                    != args.semantic_kind.casefold()
                )
            ):
                continue
            slices = []
            if _flag(row, "more_steps"):
                slices.append("more_steps")
            if _flag(row, "novel_comp"):
                slices.append("novel_composition")
            for slice_name in slices:
                question = str(row["question"])
                candidates[slice_name][video_id].append({
                    "task_id": task_id,
                    "video_id": video_id,
                    "question": question,
                    "question_sha256": stable_hash(question),
                    "structural": "query",
                    "semantic": str(row.get("semantic") or "unknown").casefold(),
                    "answer_type": str(row.get("ans_type") or "unknown").casefold(),
                    "official_evaluation_slice": slice_name,
                    "video_path": str(local[video_id]),
                })

    selected_rows = []
    selected_videos: set[str] = set()
    per_slice = {}
    # Allocate the rarer more-steps slice first, then enforce video disjointness
    # between slices.  Ranking is independent of answers and model outcomes.
    for slice_name in ("more_steps", "novel_composition"):
        eligible = [
            video_id for video_id, rows in candidates[slice_name].items()
            if video_id not in selected_videos and len(rows) >= args.tasks_per_video
        ]
        eligible.sort(key=lambda video_id: stable_hash({
            "salt": args.salt, "slice": slice_name, "video_id": video_id,
        }))
        chosen = eligible[:args.videos_per_slice]
        if len(chosen) != args.videos_per_slice:
            raise RuntimeError(f"only {len(chosen)} eligible videos for {slice_name}")
        for video_id in chosen:
            ranked = sorted(candidates[slice_name][video_id], key=lambda row: stable_hash({
                "salt": args.salt, "slice": slice_name, "task_id": row["task_id"],
            }))
            selected_rows.extend(ranked[:args.tasks_per_video])
        selected_videos.update(chosen)
        per_slice[slice_name] = {
            "video_count": len(chosen),
            "task_count": len(chosen) * args.tasks_per_video,
            "eligible_video_count_before_selection": len(eligible),
        }

    video_receipts = []
    for video_id in sorted(selected_videos):
        path = local[video_id]
        video_receipts.append({
            "video_id": video_id,
            "video_path": str(path),
            "video_sha256": _sha256(path),
            "prior_raw_video_exposure": video_id in exposed_videos,
        })
    public = {
        "schema_version": "agqa-query-grounder-v2-task-disjoint-public-v1",
        "status": "FROZEN_BEFORE_PARSER_GROUNDER_OR_SELECTED_TASK_OUTCOMES",
        "source_split": "official_balanced_test_action_genome_test",
        "freshness": "TASK_DISJOINT_NOT_VIDEO_DISJOINT",
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
        "requested_video_count": len(selected_videos) == expected_videos,
        "requested_task_count": len(selected_rows) == expected_tasks,
        "tasks_unique": len({row["task_id"] for row in selected_rows}) == expected_tasks,
        "task_ids_absent_from_all_prior_artifacts": not (
            {row["task_id"] for row in selected_rows} & prior_tasks
        ),
        "videos_disjoint_between_compositional_slices": len(selected_videos) == expected_videos,
        "all_videos_action_genome_test": all(video_id in exposed_videos for video_id in selected_videos),
        "all_videos_locally_available_and_content_addressed": all(
            row["video_sha256"] for row in video_receipts
        ),
        "raw_video_freshness_claimed": False,
    }
    manifest = {
        "schema_version": "agqa-query-grounder-v2-task-disjoint-freeze-v1",
        "status": (
            "AGQA_QUERY_GROUNDER_V2_TASK_DISJOINT_QUALIFICATION_FROZEN"
            if all(value for key, value in gates.items() if key != "raw_video_freshness_claimed")
            else "FREEZE_FAILED"
        ),
        "cohort_sha256": public["cohort_sha256"],
        "archive_sha256": _sha256(args.archive),
        "archive_entry": args.entry,
        "selection_salt": args.salt,
        "selection": {
            "videos": len(selected_videos), "tasks": len(selected_rows),
            "semantic_kind": args.semantic_kind, "slices": per_slice,
        },
        "prior_task_audit": prior_task_audit,
        "raw_video_exposure_audit_sha256": exposure["report_sha256"],
        "gates": gates,
        "claim_boundary": (
            "Never-mentioned official compositional tasks with frozen raw-video inputs; "
            "videos are historically exposed and this is not an untouched-video result."
        ),
        "outcome_authority": {
            "hidden_fields_present_in_archive": True,
            "hidden_fields_used_for_selection": False,
            "answers_used": False,
            "functional_programs_used": False,
            "scene_graph_grounding_used": False,
            "selection_features": [
                "task_id", "video_id", "question", "structural", "semantic",
                "ans_type", "more_steps", "novel_comp",
            ],
        },
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "public_cohort.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": manifest["status"],
        "videos": len(selected_videos),
        "tasks": len(selected_rows),
        "prior_task_count": len(prior_tasks),
        "slices": per_slice,
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))
    return 0 if manifest["status"].endswith("_FROZEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
