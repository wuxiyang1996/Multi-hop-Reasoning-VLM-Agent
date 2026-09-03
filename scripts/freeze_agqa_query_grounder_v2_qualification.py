#!/usr/bin/env python3
"""Freeze an AG-test, prior-exposure-disjoint Query Grounder V2 cohort."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402


FRAME_EVIDENCE_KEYS = {
    "selected_frame_sha256s",
    "sampled_frame_sha256s",
    "presented_frame_receipts",
}


def _content_addressed_video_ids(value, inherited_video_id: str | None = None) -> set[str]:
    """Associate frame evidence only with the enclosing row's video.

    Older exposure code marked *every* video id in a JSON file whenever that
    file contained one frame-hash field.  Mixed cohort/report bundles therefore
    over-excluded videos that never reached a visual model.  This traversal is
    conservative at the row level while avoiding that file-level expansion.
    """
    found: set[str] = set()
    if isinstance(value, dict):
        current = value.get("video_id", inherited_video_id)
        current = str(current) if current is not None else None
        if current is not None and FRAME_EVIDENCE_KEYS.intersection(value):
            found.add(current)
        for child in value.values():
            found.update(_content_addressed_video_ids(child, current))
    elif isinstance(value, list):
        for child in value:
            found.update(_content_addressed_video_ids(child, inherited_video_id))
    return found


def _exact_raw_runtime_exposed_videos(root: Path) -> tuple[set[str], dict[str, int]]:
    runtime: set[str] = set()
    for path in root.glob("runs/agqa2*/runtime_receipts/*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value.get("video_id"), str):
            runtime.add(str(value["video_id"]))

    grounded: set[str] = set()
    candidate_files = 0
    parsed_files = 0
    needles = tuple(f'"{key}"' for key in FRAME_EVIDENCE_KEYS)
    for path in root.glob("runs/**/*.json"):
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not any(needle in raw for needle in needles):
            continue
        candidate_files += 1
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        parsed_files += 1
        grounded.update(_content_addressed_video_ids(value))
    union = runtime | grounded
    return union, {
        "runtime_receipt_video_count": len(runtime),
        "row_associated_frame_evidence_video_count": len(grounded),
        "candidate_frame_evidence_files": candidate_files,
        "parsed_frame_evidence_files": parsed_files,
        "union_video_count": len(union),
        "file_level_video_id_expansion_used": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _allocated_config_videos() -> set[str]:
    """Protect previously frozen cohorts even when they have not run yet."""
    videos: set[str] = set()
    for path in (REPO / "configs").glob("agqa2*.json"):
        try:
            root = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        status = str(root.get("status", ""))
        if "FROZEN" not in status and "SELECTION" not in status and "MANIFEST" not in path.name.upper():
            continue
        stack = [root]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if isinstance(value.get("video_id"), str):
                    videos.add(value["video_id"])
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    return videos


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/test_balanced.txt")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=40)
    parser.add_argument("--tasks-per-video", type=int, default=2)
    parser.add_argument("--salt", default="agqa-query-grounder-v2-agtest-qualification-v1")
    parser.add_argument("--exclude-video", action="append", default=[])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("qualification freeze is immutable")

    raw_exposed, exposure = _exact_raw_runtime_exposed_videos(REPO)
    allocated = _allocated_config_videos()
    excluded = raw_exposed | allocated | set(args.exclude_video)
    local = {path.stem: path for path in args.video_root.glob("*.mp4")}
    candidates: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(args.archive) as bundle, bundle.open(args.entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            video_id = str(row["video_id"])
            if (
                str(row.get("structural", "")).casefold() != "query"
                or video_id in excluded or video_id not in local
            ):
                continue
            question = str(row["question"])
            candidates[video_id].append({
                "task_id": str(task_id), "video_id": video_id,
                "question": question, "question_sha256": stable_hash(question),
                "structural": "query",
                "semantic": str(row.get("semantic") or "unknown").casefold(),
                "answer_type": str(row.get("ans_type") or "unknown").casefold(),
                "video_path": str(local[video_id]),
            })
    eligible = sorted(
        (video for video, rows in candidates.items() if len(rows) >= args.tasks_per_video),
        key=lambda video: stable_hash({"salt": args.salt, "video_id": video}),
    )
    if len(eligible) < args.videos:
        raise RuntimeError(f"only {len(eligible)} eligible fresh AG-test videos")
    selected = eligible[:args.videos]
    rows = []
    video_receipts = []
    for video_id in selected:
        path = local[video_id]
        video_receipts.append({
            "video_id": video_id, "video_path": str(path), "video_sha256": _sha256(path),
        })
        ranked = sorted(
            candidates[video_id],
            key=lambda row: stable_hash({"salt": args.salt, "task_id": row["task_id"]}),
        )
        rows.extend(ranked[:args.tasks_per_video])
    public = {
        "schema_version": "agqa-query-grounder-v2-agtest-qualification-public-v1",
        "status": "FROZEN_BEFORE_PARSER_GROUNDER_OR_OUTCOME",
        "source_split": "official_balanced_test_action_genome_test",
        "rows": rows, "video_receipts": video_receipts,
        "answers_projected": False, "functional_programs_projected": False,
        "scene_graph_grounding_projected": False, "source_controller_read": False,
    }
    public["cohort_sha256"] = stable_hash(public)
    gates = {
        "requested_video_count": len(video_receipts) == args.videos,
        "requested_query_task_count": len(rows) == args.videos * args.tasks_per_video,
        "videos_unique": len({row["video_id"] for row in video_receipts}) == args.videos,
        "tasks_unique": len({row["task_id"] for row in rows}) == len(rows),
        "raw_runtime_disjoint": not ({row["video_id"] for row in rows} & raw_exposed),
        "previously_allocated_reserve_disjoint": not ({row["video_id"] for row in rows} & allocated),
        "all_videos_locally_available_and_content_addressed": all(
            row["video_sha256"] for row in video_receipts
        ),
    }
    manifest = {
        "schema_version": "agqa-query-grounder-v2-agtest-qualification-freeze-v1",
        "status": "AGQA_QUERY_GROUNDER_V2_QUALIFICATION_FROZEN" if all(gates.values()) else "FREEZE_FAILED",
        "cohort_sha256": public["cohort_sha256"],
        "archive_sha256": _sha256(args.archive), "archive_entry": args.entry,
        "video_root": str(args.video_root), "rank_salt": args.salt,
        "selection": {"videos": args.videos, "tasks_per_video": args.tasks_per_video,
                      "query_tasks": len(rows)},
        "prior_raw_runtime_exposure": exposure,
        "prior_allocated_config_video_count": len(allocated),
        "excluded_video_ids_sha256": stable_hash(sorted(excluded)),
        "eligible_video_count_before_selection": len(eligible),
        "gates": gates,
        "outcome_authority": {
            "archive_records_contained_hidden_evaluator_fields": True,
            "hidden_fields_used_for_selection": False,
            "answers_used": False, "functional_programs_used": False,
            "scene_graph_grounding_used": False,
            "selection_features": ["task_id", "video_id", "question", "structural", "semantic", "ans_type"],
        },
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "public_cohort.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": manifest["status"], "videos": len(video_receipts), "tasks": len(rows),
        "eligible": len(eligible), "raw_excluded": len(raw_exposed),
        "allocated_excluded": len(allocated), "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
