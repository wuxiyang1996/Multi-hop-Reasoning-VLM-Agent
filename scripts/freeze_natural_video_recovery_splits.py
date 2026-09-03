#!/usr/bin/env python3
"""Freeze outcome-blind, video-disjoint STAR/NExT-QA recovery splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence


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


def _rank(salt: str, family: str, sample_id: str) -> str:
    return hashlib.sha256(f"{salt}|{family}|{sample_id}".encode("utf-8")).hexdigest()


def _freeze(
    samples: Sequence[Any],
    *,
    families: Sequence[str],
    family_of: Callable[[Any], str],
    excluded_ids: set[str],
    excluded_videos: set[str],
    counts: dict[str, int],
    question_counts: dict[str, int],
    salt: str,
) -> dict[str, Any]:
    anchor_family_roles: dict[str, dict[str, list[str]]] = {}
    role_videos = {role: set() for role in counts}
    used_videos = set(excluded_videos)
    family_order = sorted(
        families,
        key=lambda family: len({
            str(sample.video_id) for sample in samples
            if family_of(sample) == family
            and str(sample.sample_id) not in excluded_ids
            and str(sample.video_id) not in excluded_videos
            and sample.video_path is not None
            and Path(sample.video_path).is_file()
        }),
    )
    for family in family_order:
        ordered = sorted(
            (
                sample for sample in samples
                if family_of(sample) == family
                and str(sample.sample_id) not in excluded_ids
                and str(sample.video_id) not in excluded_videos
                and sample.video_path is not None
                and Path(sample.video_path).is_file()
            ),
            key=lambda sample: _rank(salt, family, str(sample.sample_id)),
        )
        cursor = 0
        anchor_family_roles[family] = {}
        for role, count in counts.items():
            selected = []
            while cursor < len(ordered) and len(selected) < count:
                sample = ordered[cursor]
                cursor += 1
                video_id = str(sample.video_id)
                if video_id in used_videos:
                    continue
                used_videos.add(video_id)
                selected.append(str(sample.sample_id))
            if len(selected) != count:
                raise ValueError(
                    f"insufficient outcome-blind video-disjoint {family}/{role}: "
                    f"wanted {count}, got {len(selected)}"
                )
            anchor_family_roles[family][role] = selected
            role_videos[role].update(
                str(sample.video_id) for sample in ordered
                if str(sample.sample_id) in set(selected)
            )
    roles = {role: [] for role in counts}
    family_roles: dict[str, dict[str, list[str]]] = {}
    for family in families:
        family_roles[family] = {}
        for role, count in question_counts.items():
            candidates = sorted(
                (
                    sample for sample in samples
                    if family_of(sample) == family
                    and str(sample.video_id) in role_videos[role]
                    and str(sample.sample_id) not in excluded_ids
                    and sample.video_path is not None
                    and Path(sample.video_path).is_file()
                ),
                key=lambda sample: _rank(
                    salt + "|questions|" + role, family, str(sample.sample_id)
                ),
            )
            selected = [str(sample.sample_id) for sample in candidates[:count]]
            if len(selected) != count:
                raise ValueError(
                    f"insufficient questions on frozen {role} videos for "
                    f"{family}: wanted {count}, got {len(selected)}"
                )
            family_roles[family][role] = selected
            roles[role].extend(selected)
    if len(set().union(*(set(values) for values in roles.values()))) != sum(
        len(values) for values in roles.values()
    ):
        raise AssertionError("natural-video recovery roles overlap")
    return {
        "families": list(families),
        "anchor_family_roles": anchor_family_roles,
        "role_video_ids": {
            role: sorted(values) for role, values in role_videos.items()
        },
        "family_roles": family_roles,
        "splits": roles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper-root", required=True, type=Path)
    parser.add_argument("--star-root", required=True, type=Path)
    parser.add_argument("--nextqa-root", required=True, type=Path)
    parser.add_argument("--exclude-manifest", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--salt", default="natural-video-recovery-v15-20260813")
    parser.add_argument("--allow-historically-consumed-videos", action="store_true")
    parser.add_argument("--star-development-per-family", type=int, default=12)
    parser.add_argument("--star-formal-per-family", type=int, default=24)
    parser.add_argument("--star-reserve-per-family", type=int, default=12)
    parser.add_argument("--nextqa-development-per-family", type=int, default=12)
    parser.add_argument("--nextqa-formal-per-family", type=int, default=24)
    parser.add_argument("--nextqa-reserve-per-family", type=int, default=12)
    parser.add_argument("--star-development-questions-per-family", type=int, default=12)
    parser.add_argument("--star-formal-questions-per-family", type=int, default=12)
    parser.add_argument("--star-reserve-questions-per-family", type=int, default=6)
    parser.add_argument("--nextqa-development-questions-per-family", type=int, default=12)
    parser.add_argument("--nextqa-formal-questions-per-family", type=int, default=12)
    parser.add_argument("--nextqa-reserve-questions-per-family", type=int, default=6)
    args = parser.parse_args()
    sys.path.insert(0, str(args.wrapper_root))
    from visual_reasoning_wrapper.benchmarks.nextqa import iter_nextqa_samples
    from visual_reasoning_wrapper.benchmarks.star import iter_star_samples

    star = list(iter_star_samples("val", star_root=args.star_root))
    nextqa = list(iter_nextqa_samples("val", nextqa_root=args.nextqa_root))
    lookup = {
        str(sample.sample_id): str(sample.video_id) for sample in star + nextqa
    }
    excluded_ids: set[str] = set()
    exclusion_hashes = {}
    for path in args.exclude_manifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        excluded_ids.update(_collect_ids(payload))
        exclusion_hashes[str(path.resolve())] = _sha256(path)
    historically_consumed_videos = {lookup[value] for value in excluded_ids if value in lookup}
    excluded_videos = (
        set() if args.allow_historically_consumed_videos
        else historically_consumed_videos
    )
    star_counts = {
        "development": args.star_development_per_family,
        "formal": args.star_formal_per_family,
        "reserve": args.star_reserve_per_family,
    }
    nextqa_counts = {
        "development": args.nextqa_development_per_family,
        "formal": args.nextqa_formal_per_family,
        "reserve": args.nextqa_reserve_per_family,
    }
    payload = {
        "schema_version": 15,
        "status": "FROZEN_BEFORE_NATURAL_VIDEO_RECOVERY_DEVELOPMENT_OUTCOMES",
        "selection_rule": (
            "Exclude every prior sample; historical videos may be reused because the local media subset is exhausted. Allocate available videos to mutually disjoint new roles, prioritizing scarce public families, then sort eligible questions by sha256(salt|family|sample_id)."
            if args.allow_historically_consumed_videos else
            "Exclude every prior sample and its entire video; allocate remaining videos to mutually disjoint new roles, prioritizing scarce public families, then sort eligible questions by sha256(salt|family|sample_id)."
        ),
        "selection_fields": ["sample_id", "video_id", "question_type_or_family", "video_path_exists"],
        "forbidden_selection_fields": ["answer", "options", "question", "functional_program", "situation_graph", "relation_annotation"],
        "outcomes_or_answers_read_by_selector": False,
        "video_disjoint_across_prior_and_new_roles": not args.allow_historically_consumed_videos,
        "video_disjoint_across_new_development_formal_reserve_roles": True,
        "historical_video_reuse_disclosed": args.allow_historically_consumed_videos,
        "salt_sha256": hashlib.sha256(args.salt.encode("utf-8")).hexdigest(),
        "excluded_manifest_sha256": exclusion_hashes,
        "excluded_id_count": len(excluded_ids),
        "historically_consumed_video_count": len(historically_consumed_videos),
        "excluded_video_count": len(excluded_videos),
        "benchmarks": {
            "star": _freeze(
                star,
                families=("Interaction", "Sequence", "Prediction", "Feasibility"),
                family_of=lambda sample: str(sample.question_type),
                excluded_ids=excluded_ids,
                excluded_videos=excluded_videos,
                counts=star_counts,
                question_counts={
                    "development": args.star_development_questions_per_family,
                    "formal": args.star_formal_questions_per_family,
                    "reserve": args.star_reserve_questions_per_family,
                },
                salt=args.salt + "|star",
            ),
            "nextqa": _freeze(
                nextqa,
                families=("Causal", "Temporal", "Descriptive"),
                family_of=lambda sample: str(sample.question_family),
                excluded_ids=excluded_ids,
                excluded_videos=excluded_videos,
                counts=nextqa_counts,
                question_counts={
                    "development": args.nextqa_development_questions_per_family,
                    "formal": args.nextqa_formal_questions_per_family,
                    "reserve": args.nextqa_reserve_questions_per_family,
                },
                salt=args.salt + "|nextqa",
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "excluded_id_count": len(excluded_ids),
        "historically_consumed_video_count": len(historically_consumed_videos),
        "excluded_video_count": len(excluded_videos),
        "counts": {
            benchmark: {role: len(ids) for role, ids in value["splits"].items()}
            for benchmark, value in payload["benchmarks"].items()
        },
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
