#!/usr/bin/env python3
"""Freeze video-disjoint structured-video benchmark splits deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _rank(seed: int, family: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}|{family}|{sample_id}".encode()).hexdigest()


def _freeze(
    samples: Sequence[Any],
    *,
    seed: int,
    families: Sequence[str],
    family_of: Callable[[Any], str],
    counts: dict[str, int],
) -> dict[str, list[str]]:
    output = {split: [] for split in counts}
    used_videos: set[str] = set()
    for family in families:
        rows = sorted(
            (sample for sample in samples if family_of(sample) == family),
            key=lambda sample: _rank(seed, family, sample.sample_id),
        )
        cursor = 0
        for split, count in counts.items():
            selected = 0
            while cursor < len(rows) and selected < count:
                sample = rows[cursor]
                cursor += 1
                if str(sample.video_id) in used_videos:
                    continue
                used_videos.add(str(sample.video_id))
                output[split].append(str(sample.sample_id))
                selected += 1
            if selected != count:
                raise RuntimeError(
                    f"insufficient video-disjoint {family} rows for {split}: "
                    f"needed {count}, found {selected}"
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    sys.path.insert(0, str(args.wrapper_root))
    from visual_reasoning_wrapper.benchmarks.clevrer import (
        iter_clevrer_question_samples,
    )
    from visual_reasoning_wrapper.benchmarks.nextqa import iter_nextqa_samples
    from visual_reasoning_wrapper.benchmarks.star import iter_star_samples

    counts = {"adaptation": 4, "qualification": 6, "held_out": 10}
    clevrer = [
        sample for sample in iter_clevrer_question_samples("validation")
        if sample.answer_length >= 2
    ]
    star = list(iter_star_samples("val"))
    nextqa = list(iter_nextqa_samples("val"))
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_STRUCTURED_VIDEO_COLLECTION",
        "seed": args.seed,
        "video_disjoint_within_each_benchmark": True,
        "counts_per_family": counts,
        "benchmarks": {
            "clevrer": {
                "families": ["explanatory", "predictive", "counterfactual"],
                "splits": _freeze(
                    clevrer,
                    seed=args.seed,
                    families=("explanatory", "predictive", "counterfactual"),
                    family_of=lambda sample: sample.question_type,
                    counts=counts,
                ),
            },
            "star": {
                "families": ["Interaction", "Sequence", "Prediction", "Feasibility"],
                "splits": _freeze(
                    star,
                    seed=args.seed + 1,
                    families=("Interaction", "Sequence", "Prediction", "Feasibility"),
                    family_of=lambda sample: sample.question_type,
                    counts=counts,
                ),
            },
            "nextqa": {
                "families": ["Causal", "Temporal", "Descriptive"],
                "splits": _freeze(
                    nextqa,
                    seed=args.seed + 2,
                    families=("Causal", "Temporal", "Descriptive"),
                    family_of=lambda sample: sample.question_family,
                    counts=counts,
                ),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        benchmark: {
            split: len(ids) for split, ids in value["splits"].items()
        }
        for benchmark, value in payload["benchmarks"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
