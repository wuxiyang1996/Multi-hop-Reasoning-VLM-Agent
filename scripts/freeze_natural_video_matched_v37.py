#!/usr/bin/env python3
"""Freeze all unseen questions on videos disjoint from V36 adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_sample_ids(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("sample_id"), str):
            output.add(str(value["sample_id"]))
        for child in value.values():
            output.update(_collect_sample_ids(child))
    elif isinstance(value, list):
        for child in value:
            output.update(_collect_sample_ids(child))
    elif isinstance(value, str) and (".Q" in value or re.search(r"_T\d+_", value)):
        output.add(value)
    return output


def _rank(salt: str, benchmark: str, sample_id: str) -> str:
    return hashlib.sha256(f"{salt}|{benchmark}|{sample_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptation", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--wrapper-root", required=True, type=Path)
    parser.add_argument("--star-root", required=True, type=Path)
    parser.add_argument("--nextqa-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--salt", default="natural-video-matched-v37-formal-20260813")
    args = parser.parse_args()
    adaptation = json.loads(args.adaptation.read_text())
    if adaptation.get("status") != "V36_MATCHED_MODEL_ADAPTATION_COMPILED":
        raise ValueError("V37 requires completed V36 adaptation receipts")
    adaptation_videos = {
        (str(row["benchmark"]), str(row["video_id"])) for row in adaptation["rows"]
    }
    seen_ids: set[str] = set()
    index_rows = []
    output_path = args.output.resolve()
    for path in sorted(args.runs_root.rglob("*.json")):
        if path.resolve() == output_path:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        seen_ids.update(_collect_sample_ids(payload))
        index_rows.append(f"{path.resolve()}:{sha256(path)}")
    historical_index_sha256 = hashlib.sha256("\n".join(index_rows).encode()).hexdigest()

    if str(args.wrapper_root) not in sys.path:
        sys.path.insert(0, str(args.wrapper_root))
    from visual_reasoning_wrapper.benchmarks.star import iter_star_samples
    from visual_reasoning_wrapper.benchmarks.nextqa import iter_nextqa_samples
    samples = {
        "star": list(iter_star_samples(
            "val", star_root=args.star_root, require_video=True,
        )),
        "nextqa": list(iter_nextqa_samples(
            "val", nextqa_root=args.nextqa_root, require_video=True,
        )),
    }
    selected: dict[str, list[dict[str, str]]] = {}
    for benchmark in ("star", "nextqa"):
        rows = []
        for sample in samples[benchmark]:
            sample_id = str(sample.sample_id)
            video_id = str(sample.video_id)
            if (benchmark, video_id) in adaptation_videos or sample_id in seen_ids:
                continue
            family = str(
                getattr(sample, "question_family", None)
                or getattr(sample, "question_type", "")
            )
            rows.append({
                "sample_id": sample_id, "video_id": video_id, "family": family,
            })
        selected[benchmark] = sorted(
            rows, key=lambda row: _rank(args.salt, benchmark, row["sample_id"]),
        )
        if not rows:
            raise ValueError(f"V37 has no eligible {benchmark} question")
    identities = [
        (benchmark, row["sample_id"])
        for benchmark in ("star", "nextqa") for row in selected[benchmark]
    ]
    formal_videos = {
        (benchmark, row["video_id"])
        for benchmark in ("star", "nextqa") for row in selected[benchmark]
    }
    if len(identities) != len(set(identities)) or formal_videos & adaptation_videos:
        raise ValueError("V37 identity or video-disjointness invariant failed")
    payload = {
        "schema_version": 37,
        "status": "FROZEN_BEFORE_V37_MATCHED_FORMAL_CALLS_OR_OUTCOMES",
        "selection_rule": (
            "Use every locally available STAR/NExT-QA question whose sample_id has "
            "never appeared in a pre-V37 run JSON and whose (benchmark,video_id) is "
            "disjoint from all 106 V36 adaptation video groups. Order only by salted "
            "sample-ID hash; do not inspect question, options, answer, official graph, "
            "model confidence, or family when selecting."
        ),
        "selection_fields": ["sample_id", "video_id"],
        "forbidden_selection_fields": [
            "question", "options", "answer", "official_structure", "model_outcome",
            "confidence", "family",
        ],
        "outcomes_or_answers_used_for_selection": False,
        "all_eligible_questions_selected": True,
        "sample_ids_never_in_historical_runs": True,
        "video_disjoint_from_v36_adaptation": True,
        "historical_raw_video_reuse_outside_v36_disclosed": True,
        "video_cluster_is_primary_independence_unit": True,
        "salt_sha256": hashlib.sha256(args.salt.encode()).hexdigest(),
        "adaptation_receipts": str(args.adaptation.resolve()),
        "adaptation_receipts_sha256": sha256(args.adaptation),
        "cate_artifact": str(args.artifact.resolve()),
        "cate_artifact_sha256": sha256(args.artifact),
        "historical_runs_root": str(args.runs_root.resolve()),
        "historical_json_file_count": len(index_rows),
        "historical_run_index_sha256": historical_index_sha256,
        "excluded_historical_sample_id_count": len(seen_ids),
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
        "output_sha256": sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
