#!/usr/bin/env python3
"""Freeze expanded outcome-blind formal questions on V15 formal video clusters."""

from __future__ import annotations

import argparse
import csv
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


def _rank(salt: str, benchmark: str, sample_id: str) -> str:
    return hashlib.sha256(
        f"{salt}|{benchmark}|{sample_id}".encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v15-manifest", required=True, type=Path)
    parser.add_argument("--star-root", required=True, type=Path)
    parser.add_argument("--nextqa-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--salt", default="natural-video-v19-expanded-formal-20260813")
    args = parser.parse_args()
    v15 = json.loads(args.v15_manifest.read_text(encoding="utf-8"))
    excluded: set[str] = set()
    exclusions = {}
    for raw_path, expected_hash in v15["excluded_manifest_sha256"].items():
        path = Path(raw_path)
        if _sha256(path) != expected_hash:
            raise ValueError(f"historical exclusion drift: {path}")
        excluded.update(_collect_ids(json.loads(path.read_text(encoding="utf-8"))))
        exclusions[str(path.resolve())] = expected_hash

    star_videos = set(v15["benchmarks"]["star"]["role_video_ids"]["formal"])
    nextqa_videos = set(v15["benchmarks"]["nextqa"]["role_video_ids"]["formal"])
    star_rows = json.loads(
        (args.star_root / "annotations/STAR_val.json").read_text(encoding="utf-8")
    )
    star = []
    for row in star_rows:
        sample_id = str(row["question_id"])
        video_id = str(row["video_id"])
        family = sample_id.split("_", 1)[0]
        if (
            video_id in star_videos
            and family in {"Interaction", "Sequence", "Prediction", "Feasibility"}
            and sample_id not in excluded
        ):
            star.append({
                "sample_id": sample_id,
                "video_id": video_id,
                "family": family,
            })

    with (args.nextqa_root / "dataset/nextqa/val.csv").open(
        encoding="utf-8", newline="",
    ) as handle:
        nextqa_rows = list(csv.DictReader(handle))
    nextqa = []
    family_map = {"C": "Causal", "T": "Temporal", "D": "Descriptive"}
    for row in nextqa_rows:
        sample_id = f"{row['video']}.Q{row['qid']}"
        video_id = str(row["video"])
        family = family_map[str(row["type"])[0]]
        if (
            video_id in nextqa_videos
            and family in {"Causal", "Temporal", "Descriptive"}
            and sample_id not in excluded
        ):
            nextqa.append({
                "sample_id": sample_id,
                "video_id": video_id,
                "family": family,
            })

    selected = {
        "star": sorted(
            star, key=lambda row: _rank(args.salt, "star", row["sample_id"])
        ),
        "nextqa": sorted(
            nextqa, key=lambda row: _rank(args.salt, "nextqa", row["sample_id"])
        ),
    }
    if any(not rows for rows in selected.values()):
        raise ValueError("expanded formal selection is empty")
    for benchmark, rows in selected.items():
        expected_videos = star_videos if benchmark == "star" else nextqa_videos
        if {row["video_id"] for row in rows} != expected_videos:
            raise ValueError(f"expanded formal lost a {benchmark} video cluster")
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"expanded formal contains duplicate {benchmark} IDs")

    payload = {
        "schema_version": 19,
        "status": "FROZEN_BEFORE_V19_EXPANDED_FORMAL_OUTCOMES",
        "selection_rule": (
            "Use every previously unconsumed question on the V15 formal video "
            "clusters. Order by salted sample-ID hash; source applicability is "
            "evaluated at runtime and never used to exclude a question."
        ),
        "source_compatible_families": {
            "star": ["Interaction", "Sequence"],
            "nextqa": ["Causal", "Temporal"],
        },
        "selection_fields": ["sample_id", "video_id", "question_type_or_family"],
        "forbidden_selection_fields": [
            "answer", "options", "question", "functional_program",
            "situation_graph", "relation_annotation", "model_outcome",
        ],
        "outcomes_or_answers_read_by_selector": False,
        "formal_videos_frozen_before_v15_development_outcomes": True,
        "questions_unconsumed_before_v19": True,
        "historical_raw_video_reuse_disclosed": True,
        "video_cluster_is_primary_independence_unit": True,
        "salt_sha256": hashlib.sha256(args.salt.encode("utf-8")).hexdigest(),
        "v15_manifest": str(args.v15_manifest.resolve()),
        "v15_manifest_sha256": _sha256(args.v15_manifest),
        "historical_exclusion_sha256": exclusions,
        "excluded_id_count": len(excluded),
        "benchmarks": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "counts": {key: len(value) for key, value in selected.items()},
        "video_counts": {
            key: len({row["video_id"] for row in value})
            for key, value in selected.items()
        },
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
