#!/usr/bin/env python3
"""Freeze every eligible question on untouched V15 reserve video clusters."""

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
    return hashlib.sha256(f"{salt}|{benchmark}|{sample_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v15-manifest", required=True, type=Path)
    parser.add_argument("--star-root", required=True, type=Path)
    parser.add_argument("--nextqa-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--salt", default="natural-video-v22-expanded-reserve-20260813")
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

    star_videos = set(v15["benchmarks"]["star"]["role_video_ids"]["reserve"])
    nextqa_videos = set(v15["benchmarks"]["nextqa"]["role_video_ids"]["reserve"])
    star = []
    star_path = args.star_root / "annotations/STAR_val.json"
    for row in json.loads(star_path.read_text(encoding="utf-8")):
        sample_id = str(row["question_id"])
        video_id = str(row["video_id"])
        family = sample_id.split("_", 1)[0]
        if video_id in star_videos and sample_id not in excluded:
            star.append({"sample_id": sample_id, "video_id": video_id, "family": family})

    nextqa = []
    nextqa_path = args.nextqa_root / "dataset/nextqa/val.csv"
    family_map = {"C": "Causal", "T": "Temporal", "D": "Descriptive"}
    with nextqa_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = f"{row['video']}.Q{row['qid']}"
            video_id = str(row["video"])
            family = family_map[str(row["type"])[0]]
            if video_id in nextqa_videos and sample_id not in excluded:
                nextqa.append({
                    "sample_id": sample_id, "video_id": video_id, "family": family,
                })

    selected = {
        "star": sorted(star, key=lambda row: _rank(args.salt, "star", row["sample_id"])),
        "nextqa": sorted(
            nextqa, key=lambda row: _rank(args.salt, "nextqa", row["sample_id"])
        ),
    }
    expected = {"star": star_videos, "nextqa": nextqa_videos}
    for benchmark, rows in selected.items():
        if not rows or {row["video_id"] for row in rows} != expected[benchmark]:
            raise ValueError(f"V22 reserve lost a {benchmark} video cluster")
        identities = [row["sample_id"] for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate {benchmark} reserve identity")

    payload = {
        "schema_version": 22,
        "status": "FROZEN_BEFORE_V22_RESERVE_RUNTIME_OR_OUTCOMES",
        "selection_rule": (
            "Use every historically unconsumed question on the V15 reserve video "
            "clusters and order by salted sample-ID hash. No answer, question text, "
            "option text, official structure, or model outcome participates."
        ),
        "selection_fields": ["sample_id", "video_id", "question_type_or_family"],
        "forbidden_selection_fields": [
            "answer", "options", "question", "functional_program", "situation_graph",
            "relation_annotation", "model_outcome",
        ],
        "outcomes_or_answers_read_by_selector": False,
        "reserve_videos_frozen_before_v15_development_outcomes": True,
        "reserve_video_disjoint_from_v15_development_and_formal": True,
        "questions_unqueried_before_v22": True,
        "historical_raw_video_reuse_disclosed": True,
        "video_cluster_is_primary_independence_unit": True,
        "salt_sha256": hashlib.sha256(args.salt.encode()).hexdigest(),
        "v15_manifest": str(args.v15_manifest.resolve()),
        "v15_manifest_sha256": _sha256(args.v15_manifest),
        "historical_exclusion_sha256": exclusions,
        "excluded_id_count": len(excluded),
        "annotation_sha256": {
            "star_val": _sha256(star_path),
            "nextqa_val": _sha256(nextqa_path),
        },
        "benchmarks": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "counts": {key: len(value) for key, value in selected.items()},
        "video_counts": {
            key: len({row["video_id"] for row in value}) for key, value in selected.items()
        },
        "family_counts": {
            key: {
                family: sum(row["family"] == family for row in value)
                for family in sorted({row["family"] for row in value})
            }
            for key, value in selected.items()
        },
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
