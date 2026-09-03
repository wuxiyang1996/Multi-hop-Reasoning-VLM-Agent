#!/usr/bin/env python3
"""Freeze one outcome-blind V19 question per consumed video cluster."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(salt: str, benchmark: str, video_id: str, sample_id: str) -> str:
    return hashlib.sha256(
        f"{salt}|{benchmark}|{video_id}|{sample_id}".encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v19-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--salt", default="natural-video-independent-v23-pilot-20260813")
    args = parser.parse_args()
    source = json.loads(args.v19_manifest.read_text(encoding="utf-8"))
    selected = {}
    for benchmark in ("star", "nextqa"):
        by_video = {}
        for row in source["benchmarks"][benchmark]:
            by_video.setdefault(str(row["video_id"]), []).append(row)
        selected[benchmark] = [
            min(
                rows,
                key=lambda row: _rank(
                    args.salt, benchmark, video_id, str(row["sample_id"])
                ),
            )
            for video_id, rows in sorted(by_video.items())
        ]
        selected[benchmark].sort(
            key=lambda row: _rank(
                args.salt, benchmark, str(row["video_id"]), str(row["sample_id"])
            )
        )
    payload = {
        "schema_version": 23,
        "status": "FROZEN_BEFORE_V23_INDEPENDENT_CANDIDATE_PILOT",
        "selection_rule": (
            "For every V19 benchmark/video cluster select the one manifest row "
            "with minimum sha256(salt|benchmark|video_id|sample_id)."
        ),
        "selection_fields": ["benchmark", "video_id", "sample_id", "family"],
        "forbidden_selection_fields": [
            "answer", "question", "options", "model_outcome", "correctness",
            "proof", "generic_direct",
        ],
        "outcomes_or_answers_read_by_selector": False,
        "development_only_consumed_videos": True,
        "salt_sha256": hashlib.sha256(args.salt.encode()).hexdigest(),
        "v19_manifest": str(args.v19_manifest.resolve()),
        "v19_manifest_sha256": _sha256(args.v19_manifest),
        "benchmarks": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "counts": {key: len(value) for key, value in selected.items()},
        "families": {
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
