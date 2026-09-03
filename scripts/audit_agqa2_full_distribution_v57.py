#!/usr/bin/env python3
"""Outcome-blind AGQA 2.0 full-distribution coverage and capacity audit.

Only public question/program structure and dataset taxonomy fields contribute
to the coverage report.  Answers and scene-graph grounding are never indexed,
counted, compared, hashed, or emitted.  The test split is used only to count
video IDs available for a future frozen formal pool.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_active_frame_grounder import (  # noqa: E402
    parse_public_question_plan,
)
from motif_transfer.agqa_program_transfer import (  # noqa: E402
    RELATION_ROUTE,
    TEMPORAL_PAIR_ROUTE,
    profile_program,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import (  # noqa: E402
    _iter_top_level_object,
)
from scripts.freeze_agqa2_active_grounding_v16_reserve import (  # noqa: E402
    _configured_video_ids,
)


DEFAULT_ARCHIVE = Path(
    "/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip"
)
TRAIN_ENTRY = "AGQA_balanced/train_balanced.txt"
TEST_ENTRY = "AGQA_balanced/test_balanced.txt"
DEFAULT_VIDEO_ROOT = Path(
    "/fs/gamma-projects/vlm-robot/datasets/STAR-official/videos/charades"
)
SUPPORTED_COMPARISONS = frozenset({"QUERY_OBJECT", "BEFORE_AFTER"})
FORBIDDEN_OUTCOME_FIELDS = frozenset({"answer", "sg_grounding", "situations"})


def _counter(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _taxonomy_key(value: Any) -> str:
    if isinstance(value, list):
        return "+".join(sorted(str(item) for item in value)) or "<EMPTY>"
    text = str(value or "").strip()
    return text or "<EMPTY>"


def _rows(bundle: zipfile.ZipFile, entry: str) -> Iterable[tuple[str, dict]]:
    with bundle.open(entry, "r") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            yield from _iter_top_level_object(text)


def _train_audit(
    bundle: zipfile.ZipFile, *, limit: int | None,
) -> dict[str, Any]:
    counts: dict[str, Counter] = defaultdict(Counter)
    rows = 0
    videos: set[str] = set()
    parsed_rows = 0
    program_route_matches = 0
    validated_selective_rows = 0
    parse_failures = 0
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    forbidden_field_values_accessed = False
    for task_id, row in _rows(bundle, TRAIN_ENTRY):
        if limit is not None and rows >= limit:
            break
        rows += 1
        # Access only the explicitly enumerated public/taxonomy fields.  The
        # decoded row may contain labels, but their values never enter state.
        question = str(row.get("question", ""))
        program = str(row.get("program", ""))
        video_id = str(row.get("video_id", ""))
        videos.add(video_id)
        profile = profile_program(task_id=task_id, program=program)
        plan = parse_public_question_plan(question)
        comparison = plan.comparison if plan is not None else "UNPARSED"
        counts["program_route"][profile.route_kind] += 1
        counts["program_root"][profile.root_function or "<EMPTY>"] += 1
        counts["program_function_signature"][
            ">".join(profile.functions) or "<EMPTY>"
        ] += 1
        for function in set(profile.functions):
            counts["program_function_presence"][function] += 1
        for field in (
            "global", "semantic", "structural", "ans_type", "steps",
            "novel_comp", "more_steps",
        ):
            counts[field][_taxonomy_key(row.get(field))] += 1
        taxonomy = "|".join(
            _taxonomy_key(row.get(field))
            for field in ("global", "semantic", "structural", "ans_type")
        )
        counts["taxonomy_joint"][taxonomy] += 1
        counts["public_parser_comparison"][comparison] += 1
        counts["route_by_comparison"][
            f"{profile.route_kind}|{comparison}"
        ] += 1
        if plan is None:
            parse_failures += 1
        else:
            parsed_rows += 1
            if plan.obligation_kind == profile.route_kind:
                program_route_matches += 1
            if (
                plan.comparison in SUPPORTED_COMPARISONS
                and plan.obligation_kind == profile.route_kind
                and (
                    (plan.comparison == "QUERY_OBJECT" and profile.route_kind == RELATION_ROUTE)
                    or (
                        plan.comparison == "BEFORE_AFTER"
                        and profile.route_kind == TEMPORAL_PAIR_ROUTE
                    )
                )
            ):
                validated_selective_rows += 1
        missing_family = (
            comparison if comparison != "UNPARSED" else profile.route_kind
        )
        if len(examples[missing_family]) < 3:
            examples[missing_family].append({
                "task_id_sha256": stable_hash(task_id),
                "question_sha256": stable_hash(question),
                "program_sha256": stable_hash(program),
            })
        if rows % 100_000 == 0:
            print(json.dumps({
                "progress_rows": rows,
                "unique_videos": len(videos),
                "validated_selective_rows": validated_selective_rows,
            }), flush=True)

    return {
        "entry": TRAIN_ENTRY,
        "rows": rows,
        "unique_videos": len(videos),
        "current_public_parser_rows": parsed_rows,
        "current_public_parser_fraction": parsed_rows / rows if rows else 0.0,
        "program_route_matches": program_route_matches,
        "program_route_match_fraction": program_route_matches / rows if rows else 0.0,
        "validated_selective_rows": validated_selective_rows,
        "validated_selective_fraction": (
            validated_selective_rows / rows if rows else 0.0
        ),
        "parse_failures": parse_failures,
        "counts": {name: _counter(value) for name, value in sorted(counts.items())},
        "hashed_examples_by_parser_or_program_family": dict(sorted(examples.items())),
        "forbidden_outcome_fields": sorted(FORBIDDEN_OUTCOME_FIELDS),
        "forbidden_field_values_accessed": forbidden_field_values_accessed,
    }


def _test_capacity(
    bundle: zipfile.ZipFile, *, video_root: Path, extra_excluded: set[str],
    limit: int | None,
) -> dict[str, Any]:
    rows = 0
    videos: set[str] = set()
    for _, row in _rows(bundle, TEST_ENTRY):
        if limit is not None and rows >= limit:
            break
        rows += 1
        # Capacity audit deliberately accesses only video_id.
        videos.add(str(row.get("video_id", "")))
        if rows % 100_000 == 0:
            print(json.dumps({
                "test_capacity_progress_rows": rows,
                "test_unique_videos": len(videos),
            }), flush=True)
    configured = _configured_video_ids()
    present = {path.stem for path in video_root.glob("*.mp4")}
    excluded = configured | present | extra_excluded
    available = videos - excluded
    return {
        "entry": TEST_ENTRY,
        "rows": rows,
        "unique_videos": len(videos),
        "configured_video_ids": len(configured),
        "locally_present_video_ids": len(present),
        "extra_excluded_video_ids": sorted(extra_excluded),
        "excluded_union_count": len(videos & excluded),
        "fresh_available_video_count": len(available),
        "fresh_available_video_ids_sha256": stable_hash(sorted(available)),
        "answer_question_program_or_scene_fields_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "docs/results/agqa2_full_distribution_v57_coverage.json",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--extra-excluded-video-id", action="append", default=["YSKX3"],
    )
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as bundle:
        train = _train_audit(bundle, limit=args.limit)
        test = _test_capacity(
            bundle, video_root=args.video_root,
            extra_excluded=set(args.extra_excluded_video_id), limit=args.limit,
        )
    body = {
        "schema_version": "agqa2-full-distribution-v57-coverage-v1",
        "status": "OUTCOME_BLIND_COVERAGE_AUDIT_COMPLETE",
        "archive": str(args.archive),
        "archive_entries": [TRAIN_ENTRY, TEST_ENTRY],
        "train": train,
        "test_capacity": test,
        "current_validated_transfer_comparisons": sorted(SUPPORTED_COMPARISONS),
        "full_distribution_definition": (
            "HASH_SAMPLED_OFFICIAL_AGQA_TEST_ROWS_WITHOUT_OPERATOR_FILTERING;"
            "SOURCE_SKILL_MAY_SELECTIVELY_INTERVENE;ALL_OTHER_ROWS_PRESERVE_"
            "THE_IDENTICAL_TARGET_NATIVE_DIRECT_PREDICTION"
        ),
        "answers_or_scene_graphs_used_for_design": False,
        "limit": args.limit,
    }
    result = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "train_rows": train["rows"],
        "train_unique_videos": train["unique_videos"],
        "validated_selective_rows": train["validated_selective_rows"],
        "validated_selective_fraction": train["validated_selective_fraction"],
        "test_rows": test["rows"],
        "test_unique_videos": test["unique_videos"],
        "fresh_test_videos": test["fresh_available_video_count"],
        "report_sha256": result["report_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
