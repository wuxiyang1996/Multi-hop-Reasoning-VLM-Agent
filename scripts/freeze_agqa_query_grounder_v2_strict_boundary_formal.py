#!/usr/bin/env python3
"""Freeze a fresh AGQA reserve after strict-boundary qualification."""

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
from scripts.freeze_agqa_query_grounder_v2_powered_qualification import _select  # noqa: E402
from scripts.freeze_agqa_query_grounder_v2_qualification import (  # noqa: E402
    _exact_raw_runtime_exposed_videos,
    _sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--archive-url",
        default="https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip",
    )
    parser.add_argument("--archive-prefix", default="Charades_v1_480/")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("strict-boundary formal reserve freeze is immutable")

    protocol = json.loads(args.protocol.read_text())
    qualification = json.loads(args.qualification.read_text())
    if protocol.get("status") != (
        "FROZEN_BEFORE_FORMAL_SELECTION_VIDEO_ACQUISITION_GROUNDING_"
        "FALLBACK_OR_OUTCOMES"
    ):
        raise ValueError("formal protocol was not frozen at the required stage")
    if qualification.get("status") != "QUERY_GROUNDER_V2_STRICT_BOUNDARY_QUALIFIED" or not all(
        qualification.get("gates", {}).values()
    ):
        raise ValueError("strict-boundary grounder did not pass qualification")
    qualified = protocol["qualified_grounder"]
    if qualified["qualification_file_sha256"] != _sha256(args.qualification):
        raise ValueError("formal protocol is not bound to this qualification")
    if qualified["qualification_report_sha256"] != qualification["report_sha256"]:
        raise ValueError("qualification report hash changed")

    spec = protocol["formal_cohort"]
    videos = int(spec["videos"])
    tasks_per_video = int(spec["tasks_per_video"])
    expected_tasks = int(spec["query_object_tasks"])
    salt = str(spec["selection_salt"])
    if videos * tasks_per_video != expected_tasks:
        raise ValueError("formal cohort cardinalities are inconsistent")

    raw_exposed, exposure_audit = _exact_raw_runtime_exposed_videos(REPO)
    candidates: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(args.archive) as bundle, bundle.open(args.entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            video_id = str(row["video_id"])
            if (
                video_id in raw_exposed
                or str(row.get("structural", "")).casefold() != "query"
                or str(row.get("semantic", "")).casefold() != "object"
            ):
                continue
            question = str(row["question"])
            candidates[video_id].append({
                "task_id": str(task_id), "video_id": video_id,
                "question": question, "question_sha256": stable_hash(question),
                "structural": "query", "semantic": "object",
                "answer_type": str(row.get("ans_type") or "unknown").casefold(),
                "video_path": str(args.video_root / f"{video_id}.mp4"),
            })
    selected_videos, selected_rows = _select(
        candidates, videos=videos, tasks_per_video=tasks_per_video, salt=salt,
    )
    selected_video_set = set(selected_videos)
    selected_task_set = {row["task_id"] for row in selected_rows}
    public = {
        "schema_version": "agqa-query-grounder-v2-strict-boundary-formal-public-v1",
        "status": "FROZEN_BEFORE_VIDEO_ACQUISITION_GROUNDING_FALLBACK_OR_OUTCOMES",
        "source_split": str(spec["source_split"]),
        "rows": selected_rows,
        "video_selections": [{
            "video_id": video_id,
            "video_path": str(args.video_root / f"{video_id}.mp4"),
        } for video_id in sorted(selected_videos)],
        "answers_projected": False,
        "functional_programs_projected": False,
        "scene_graph_grounding_projected": False,
        "source_controller_read": False,
        "target_outcome_read": False,
        "transfer_evidence": True,
        "claim_scope": "CONTROLLED_BALANCED_TRAIN_QUERY_OBJECT_NOT_OFFICIAL_TEST",
    }
    public["cohort_sha256"] = stable_hash(public)
    samples = [{
        "video_id": video_id,
        "video_path": str(args.video_root / f"{video_id}.mp4"),
        "selected_task_ids": sorted(
            row["task_id"] for row in selected_rows if row["video_id"] == video_id
        ),
    } for video_id in sorted(selected_videos)]
    download = {
        "schema_version": "agqa-query-grounder-v2-strict-boundary-formal-download-v1",
        "status": "FROZEN_BEFORE_VIDEO_DOWNLOAD_GROUNDING_OR_OUTCOMES",
        "archive_path": str(args.archive), "archive_sha256": _sha256(args.archive),
        "entry": args.entry,
        "raw_video_archive": {"archive_prefix": args.archive_prefix, "url": args.archive_url},
        "sample_count": len(samples), "samples": samples,
        "public_cohort_sha256": public["cohort_sha256"],
        "answer_read_during_selection_or_freeze": False,
        "functional_program_read_during_selection_or_freeze": False,
        "official_scene_graph_read_during_selection_or_freeze": False,
        "source_controller_read_during_selection_or_freeze": False,
        "target_outcome_read_during_selection_or_freeze": False,
    }
    download["manifest_sha256"] = stable_hash(download)
    gates = {
        "qualification_passed_before_selection": True,
        "requested_video_count": len(selected_video_set) == videos,
        "requested_task_count": len(selected_rows) == expected_tasks,
        "videos_unique": len(selected_video_set) == videos,
        "tasks_unique": len(selected_task_set) == expected_tasks,
        "videos_disjoint_from_all_prior_raw_runtime": not (selected_video_set & raw_exposed),
        "outcome_fields_not_projected": not any(public[key] for key in (
            "answers_projected", "functional_programs_projected",
            "scene_graph_grounding_projected", "source_controller_read", "target_outcome_read",
        )),
    }
    body = {
        "schema_version": "agqa-query-grounder-v2-strict-boundary-formal-freeze-v1",
        "status": (
            "AGQA_QUERY_GROUNDER_V2_STRICT_BOUNDARY_FRESH_FORMAL_FROZEN"
            if all(gates.values()) else "FREEZE_FAILED"
        ),
        "protocol_file_sha256": _sha256(args.protocol),
        "qualification_file_sha256": _sha256(args.qualification),
        "archive_file_sha256": _sha256(args.archive), "archive_entry": args.entry,
        "selection_salt": salt,
        "selection": {"videos": len(selected_video_set), "tasks": len(selected_rows),
                      "eligible_videos": len(candidates)},
        "cohort_sha256": public["cohort_sha256"],
        "download_selection_manifest_sha256": download["manifest_sha256"],
        "prior_raw_video_exposure": exposure_audit,
        "prior_raw_video_ids_sha256": stable_hash(sorted(raw_exposed)),
        "gates": gates, "formal_eligible": all(gates.values()),
        "transfer_evidence": True, "official_test_claim": False,
    }
    body["manifest_sha256"] = stable_hash(body)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "public_cohort.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "download_selection.json").write_text(
        json.dumps(download, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": body["status"], "videos": len(selected_video_set),
        "tasks": len(selected_rows), "eligible_videos": len(candidates),
        "already_local": sum(Path(row["video_path"]).is_file() for row in samples),
        "manifest_sha256": body["manifest_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
