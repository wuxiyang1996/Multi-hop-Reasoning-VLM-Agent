#!/usr/bin/env python3
"""Freeze an outcome-blind, already-consumed three-video grounding pilot."""

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


def _rank(seed: str, benchmark: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}|{benchmark}|{sample_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("configs/three_video_benchmark_splits_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-benchmark", type=int, default=4)
    parser.add_argument("--seed", default="grounding-qualification-v28-20260813")
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if source.get("status") != "FROZEN_BEFORE_STRUCTURED_VIDEO_COLLECTION":
        raise ValueError("source split manifest is not frozen")
    selected: dict[str, list[dict[str, Any]]] = {}
    for benchmark in ("clevrer", "star", "nextqa"):
        candidates = list(source["benchmarks"][benchmark]["splits"]["adaptation"])
        if benchmark == "star":
            candidates = [value for value in candidates if value.startswith("Interaction_")]
        ordered = sorted(candidates, key=lambda value: _rank(args.seed, benchmark, value))
        if len(ordered) < args.per_benchmark:
            raise ValueError(f"insufficient consumed {benchmark} candidates")
        selected[benchmark] = [
            {
                "sample_id": sample_id,
                "selection_hash": _rank(args.seed, benchmark, sample_id),
                "source_role": "adaptation_already_consumed",
            }
            for sample_id in ordered[: args.per_benchmark]
        ]
    payload = {
        "schema_version": 28,
        "status": "FROZEN_BEFORE_V28_GROUNDING_QUALIFICATION_CALLS",
        "purpose": "source-free semantic grounding qualification only",
        "selection_rule": (
            "Within each benchmark's already-consumed V1 adaptation IDs, sort "
            "sha256(seed|benchmark|sample_id); STAR is restricted to Interaction "
            "so situation-action timestamps provide an intrinsic localization score."
        ),
        "selection_fields": ["benchmark", "sample_id", "prior_role"],
        "forbidden_selection_fields": [
            "answer", "model_outcome", "correctness", "functional_program",
            "situation_graph", "relation_annotation",
        ],
        "outcomes_or_answers_read_by_selector": False,
        "fresh_confirmation_data_opened": False,
        "source_transfer_enabled": False,
        "seed_sha256": hashlib.sha256(args.seed.encode()).hexdigest(),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": _sha256(args.source_manifest),
        "benchmarks": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "counts": {key: len(value) for key, value in selected.items()},
        "output": str(args.output.resolve()),
        "sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
