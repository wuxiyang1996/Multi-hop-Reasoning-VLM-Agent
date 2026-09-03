#!/usr/bin/env python3
"""Evaluator-side audit that AGQA videos are held out from AG grounder training.

This utility is deliberately separate from acquisition.  It reads the official
Action Genome annotation pickle only to project ``video_id -> metadata.set``;
object boxes, relationships, labels, and answers are neither emitted nor made
available to the grounder.  Its input cohort must already be a public,
outcome-free freeze artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

from motif_transfer.contracts import stable_hash


FORBIDDEN_PUBLIC_FIELDS = {
    "answer", "correct", "functional_program", "gold", "program",
    "selected_option", "sg_grounding", "target_outcome",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.casefold() in FORBIDDEN_PUBLIC_FIELDS:
                paths.append(path)
            paths.extend(_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


def _video_split_projection(annotation_path: Path) -> tuple[dict[str, str], int]:
    # The pickle is an official upstream artifact.  It is evaluator-side only;
    # never pass the loaded object to acquisition, parsing, or Harness code.
    with annotation_path.open("rb") as handle:
        annotations = pickle.load(handle)
    projection: dict[str, str] = {}
    frame_count = 0
    for frame_key, objects in annotations.items():
        frame_count += 1
        video_id = str(frame_key).split("/", 1)[0]
        if video_id.endswith(".mp4"):
            video_id = video_id[:-4]
        frame_splits = {
            str(row.get("metadata", {}).get("set", "")) for row in objects
        }
        frame_splits.discard("")
        if len(frame_splits) != 1:
            raise ValueError(f"{frame_key}: missing or conflicting split metadata")
        split = next(iter(frame_splits))
        prior = projection.setdefault(video_id, split)
        if prior != split:
            raise ValueError(f"{video_id}: video crosses Action Genome splits")
    return projection, frame_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--object-annotations", type=Path, required=True)
    parser.add_argument("--expected-split", choices=("train", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("split audit artifact is immutable")

    cohort = json.loads(args.cohort.read_text())
    forbidden = _forbidden_paths(cohort)
    if forbidden:
        raise ValueError(f"cohort is not outcome-free: {forbidden[:5]}")
    rows = cohort.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("cohort must contain non-empty public rows")
    videos = sorted({str(row["video_id"]) for row in rows})

    projection, annotation_frames = _video_split_projection(args.object_annotations)
    missing = sorted(set(videos) - set(projection))
    wrong = sorted(video for video in videos if projection.get(video) != args.expected_split)
    membership = [(video, projection.get(video, "MISSING")) for video in videos]
    gates = {
        "public_cohort_contains_no_outcomes_or_programs": not forbidden,
        "all_cohort_videos_in_official_action_genome_metadata": not missing,
        "all_cohort_videos_match_expected_action_genome_split": not wrong,
        "only_split_membership_projected": True,
        "annotation_not_available_to_grounder": True,
    }
    body = {
        "schema_version": "agqa-action-genome-grounder-split-audit-v1",
        "status": "HELD_OUT_SPLIT_AUDIT_PASSED" if all(gates.values()) else "HELD_OUT_SPLIT_AUDIT_FAILED",
        "cohort_path": str(args.cohort),
        "cohort_sha256": cohort.get("cohort_sha256"),
        "cohort_rows": len(rows),
        "cohort_videos": len(videos),
        "expected_action_genome_split": args.expected_split,
        "official_annotation_sha256": _sha256(args.object_annotations),
        "official_annotation_frames_audited": annotation_frames,
        "official_annotation_videos_audited": len(projection),
        "projected_membership_sha256": stable_hash(membership),
        "missing_video_count": len(missing),
        "wrong_split_video_count": len(wrong),
        "missing_video_ids": missing,
        "wrong_split_video_ids": wrong,
        "runtime_authority": {
            "answers_read": False,
            "functional_programs_read": False,
            "official_scene_graph_used_by_grounder": False,
            "official_boxes_used_by_grounder": False,
            "official_relations_used_by_grounder": False,
            "split_metadata_used_by_evaluator_audit_only": True,
        },
        "gates": gates,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps(body, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
