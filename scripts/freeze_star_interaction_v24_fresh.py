#!/usr/bin/env python3
"""Freeze fresh STAR Interaction clusters for prospective source transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_ids(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            output.update(_collect_ids(child))
    elif isinstance(value, list):
        for child in value:
            output.update(_collect_ids(child))
    elif isinstance(value, str) and (".Q" in value or "_T" in value):
        output.add(value)
    return output


def _rank(salt: str, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{salt}|{namespace}|{value}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v15-manifest", required=True, type=Path)
    parser.add_argument("--star-annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--questions-per-cluster", type=int, default=2)
    parser.add_argument("--salt", default="star-interaction-v24-fresh-20260813")
    args = parser.parse_args()
    v15 = json.loads(args.v15_manifest.read_text(encoding="utf-8"))
    annotations = json.loads(args.star_annotations.read_text(encoding="utf-8"))
    id_to_video = {
        str(row["question_id"]): str(row["video_id"]) for row in annotations
    }
    historical_ids: set[str] = set()
    historical_paths = {}
    for raw_path, expected_hash in v15["excluded_manifest_sha256"].items():
        path = Path(raw_path)
        if _sha256(path) != expected_hash:
            raise ValueError(f"historical exclusion drift: {path}")
        historical_ids.update(_collect_ids(json.loads(path.read_text(encoding="utf-8"))))
        historical_paths[str(path.resolve())] = expected_hash
    historical_videos = {
        id_to_video[sample_id] for sample_id in historical_ids if sample_id in id_to_video
    }
    v15_videos = {
        str(video_id)
        for role_videos in v15["benchmarks"]["star"]["role_video_ids"].values()
        for video_id in role_videos
    }
    excluded_videos = historical_videos | v15_videos
    by_video: dict[str, list[str]] = {}
    for row in annotations:
        sample_id = str(row["question_id"])
        video_id = str(row["video_id"])
        if (
            sample_id.startswith("Interaction_")
            and sample_id not in historical_ids
            and video_id not in excluded_videos
        ):
            by_video.setdefault(video_id, []).append(sample_id)
    eligible = {
        video_id: ids for video_id, ids in by_video.items()
        if len(ids) >= args.questions_per_cluster
    }
    selected_videos = sorted(
        eligible,
        key=lambda video_id: _rank(args.salt, "video", video_id),
    )[: args.clusters]
    if len(selected_videos) != args.clusters:
        raise ValueError("insufficient fresh STAR Interaction video clusters")
    selected = []
    for video_id in selected_videos:
        ids = sorted(
            eligible[video_id],
            key=lambda sample_id: _rank(args.salt, "question", sample_id),
        )[: args.questions_per_cluster]
        selected.extend({
            "sample_id": sample_id,
            "video_id": video_id,
            "family": "Interaction",
        } for sample_id in ids)
    payload = {
        "schema_version": 24,
        "status": "FROZEN_BEFORE_V24_VIDEO_DOWNLOAD_OR_RUNTIME_OUTCOMES",
        "selection_rule": (
            "Exclude all videos recoverable from historical consumed sample IDs and "
            "all V15 development/formal/reserve videos. Among remaining STAR val "
            "videos with at least two Interaction questions, select the 64 minimum "
            "sha256(salt|video|video_id) clusters and the two minimum "
            "sha256(salt|question|sample_id) questions per cluster."
        ),
        "selection_fields": ["question_id", "video_id", "question_family"],
        "forbidden_selection_fields": [
            "answer", "choices", "question", "question_program", "choice_programs",
            "situations", "model_outcome",
        ],
        "outcomes_or_answers_read_by_selector": False,
        "adaptation_source": (
            "V19/V21 consumed development showed source guard vs matched direct 4W/0L "
            "on STAR Interaction; this family/operator choice is adaptation, not a "
            "pre-existing confirmatory hypothesis."
        ),
        "fresh_video_clusters_vs_historical_and_v15": True,
        "questions_per_video_cluster": args.questions_per_cluster,
        "primary_independence_unit": "video_id",
        "salt_sha256": hashlib.sha256(args.salt.encode()).hexdigest(),
        "v15_manifest": str(args.v15_manifest.resolve()),
        "v15_manifest_sha256": _sha256(args.v15_manifest),
        "star_annotations": str(args.star_annotations.resolve()),
        "star_annotations_sha256": _sha256(args.star_annotations),
        "historical_exclusion_sha256": historical_paths,
        "historical_consumed_video_count": len(historical_videos),
        "v15_video_count": len(v15_videos),
        "excluded_video_count": len(excluded_videos),
        "eligible_fresh_video_count": len(eligible),
        "raw_video_archive": {
            "url": "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip",
            "content_length": 16339546533,
            "etag": "d37c91565b08ce1f432a46e11351751e-1948",
            "archive_prefix": "Charades_v1_480/"
        },
        "samples": selected,
    }
    if len(selected) != args.clusters * args.questions_per_cluster:
        raise AssertionError("V24 selected sample count drift")
    if len({row["video_id"] for row in selected}) != args.clusters:
        raise AssertionError("V24 selected video count drift")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "samples": len(selected),
        "video_clusters": len(selected_videos),
        "eligible_fresh_video_count": len(eligible),
        "excluded_video_count": len(excluded_videos),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
