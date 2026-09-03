#!/usr/bin/env python3
"""Freeze a task- and raw-video-runtime-disjoint AGQA train broad reserve.

The official balanced train records contain evaluator-only fields.  This
selector parses the records but only projects task id, video id, question,
public structural/semantic strata, and answer type.  Answer, program, steps,
and scene-graph grounding are never indexed and cannot affect ranking.

Freshness is defined against actual prior visual runtime evidence, not mere
appearance of a video id in question-only parser supervision:

* any video in an AGQA runtime receipt is excluded;
* any video in an artifact containing sampled-frame hashes is excluded; and
* every selected task id is absent from parser train/validation supervision.
"""

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


STRATA = ("choose", "compare", "logic", "query", "verify")
VIDEO_ID = re.compile(r'"video_id"\s*:\s*"([A-Z0-9]{5})"')


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_runtime_exposed_videos(root: Path) -> tuple[set[str], dict[str, int]]:
    runtime: set[str] = set()
    for path in root.glob("runs/agqa2*/runtime_receipts/*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value.get("video_id"), str):
            runtime.add(value["video_id"])

    grounded: set[str] = set()
    scanned_files = 0
    for path in root.glob("runs/**/*.json"):
        try:
            with path.open(encoding="utf-8", errors="ignore") as stream:
                contains_frames = False
                ids: set[str] = set()
                for line in stream:
                    if '"selected_frame_sha256s"' in line:
                        contains_frames = True
                    ids.update(VIDEO_ID.findall(line))
            if contains_frames:
                scanned_files += 1
                grounded.update(ids)
        except OSError:
            continue
    return runtime | grounded, {
        "runtime_receipt_video_count": len(runtime),
        "sampled_frame_artifact_video_count": len(grounded),
        "sampled_frame_artifact_files": scanned_files,
        "union_video_count": len(runtime | grounded),
    }


def parser_supervision_tasks(paths: list[Path]) -> set[str]:
    task_ids: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                task_ids.add(str(json.loads(line)["task_id"]))
    return task_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--parser-supervision", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=27)
    parser.add_argument("--tasks-per-stratum-video", type=int, default=3)
    parser.add_argument("--salt", default="agqa-full-train-broad-v1-20260902")
    parser.add_argument("--exclude-video", action="append", default=[])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("AGQA broad reserve freeze is immutable")

    exposed, exposure = raw_runtime_exposed_videos(REPO)
    supervised = parser_supervision_tasks(args.parser_supervision)
    local = {path.stem: path for path in args.video_root.glob("*.mp4")}
    candidates: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with zipfile.ZipFile(args.archive) as bundle, bundle.open(args.entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            task_id = str(task_id)
            video_id = str(row["video_id"])
            structural = str(row.get("structural") or "").casefold()
            if (
                structural not in STRATA or video_id in exposed
                or video_id in set(args.exclude_video) or video_id not in local
                or task_id in supervised
            ):
                continue
            question = str(row["question"])
            public = {
                "task_id": task_id,
                "video_id": video_id,
                "question": question,
                "question_sha256": stable_hash(question),
                "structural": structural,
                "semantic": str(row.get("semantic") or "unknown").casefold(),
                "answer_type": str(row.get("ans_type") or "unknown").casefold(),
            }
            candidates[video_id][structural].append(public)

    k = args.tasks_per_stratum_video
    eligible = [
        video_id for video_id, by_stratum in candidates.items()
        if all(len(by_stratum[stratum]) >= k for stratum in STRATA)
    ]
    ranked_videos = sorted(
        eligible, key=lambda value: stable_hash({"salt": args.salt, "video_id": value}),
    )
    if len(ranked_videos) < args.videos:
        raise RuntimeError(f"only {len(ranked_videos)} eligible videos for {args.videos} requested")
    selected_videos = ranked_videos[: args.videos]
    rows = []
    video_receipts = []
    for video_id in selected_videos:
        video_path = local[video_id]
        video_receipts.append({
            "video_id": video_id,
            "video_path": str(video_path),
            "video_sha256": file_sha256(video_path),
        })
        for stratum in STRATA:
            ranked = sorted(
                candidates[video_id][stratum],
                key=lambda row: stable_hash({"salt": args.salt, "task_id": row["task_id"]}),
            )
            for row in ranked[:k]:
                rows.append({**row, "video_path": str(video_path)})

    public = {
        "schema_version": "agqa-full-train-broad-public-v1",
        "status": "FROZEN_BEFORE_PARSER_GROUNDER_OR_OUTCOME",
        "source_split": "official_balanced_train",
        "rows": rows,
        "video_receipts": video_receipts,
        "answers_projected": False,
        "functional_programs_projected": False,
        "scene_graph_grounding_projected": False,
        "source_controller_read": False,
    }
    public["cohort_sha256"] = stable_hash(public)
    gates = {
        "requested_video_count": len(video_receipts) == args.videos,
        "requested_task_count": len(rows) == args.videos * len(STRATA) * k,
        "selected_videos_unique": len({row["video_id"] for row in video_receipts}) == args.videos,
        "tasks_unique": len({row["task_id"] for row in rows}) == len(rows),
        "all_five_public_structural_strata_balanced": all(
            sum(row["structural"] == stratum for row in rows) == args.videos * k
            for stratum in STRATA
        ),
        "raw_video_runtime_disjoint": not ({row["video_id"] for row in rows} & exposed),
        "parser_supervision_task_disjoint": not ({row["task_id"] for row in rows} & supervised),
        "all_raw_mp4_content_bound": all(row["video_sha256"] for row in video_receipts),
    }
    manifest = {
        "schema_version": "agqa-full-train-broad-freeze-v1",
        "status": "AGQA_FRESH_BROAD_RESERVE_FROZEN" if all(gates.values()) else "AGQA_FRESH_BROAD_RESERVE_FAILED",
        "cohort_sha256": public["cohort_sha256"],
        "archive_sha256": file_sha256(args.archive),
        "archive_entry": args.entry,
        "video_root": str(args.video_root),
        "parser_supervision_file_sha256s": {
            str(path): file_sha256(path) for path in args.parser_supervision
        },
        "prior_raw_runtime_exposure": exposure,
        "eligible_video_count_before_selection": len(eligible),
        "rank_salt": args.salt,
        "explicit_pre_video_excluded_videos": sorted(set(args.exclude_video)),
        "selection": {"videos": args.videos, "tasks_per_stratum_video": k, "tasks": len(rows)},
        "gates": gates,
        "outcome_authority": {
            "archive_records_contained_hidden_evaluator_fields": True,
            "hidden_evaluator_fields_indexed_or_used": False,
            "answers_used": False,
            "functional_programs_used": False,
            "scene_graph_grounding_used": False,
            "selection_features": ["task_id", "video_id", "question", "structural", "semantic", "ans_type"],
        },
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "public_cohort.json").write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
