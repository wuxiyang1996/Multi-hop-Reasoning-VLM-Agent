#!/usr/bin/env python3
"""Freeze a parser-qualified AGQA subset before any video or outcome access.

The exclusion unit is the entire video containing any invalid operator-free
semantic parse.  This prevents cherry-picking individual questions and keeps
the original per-video 5-stratum x 3-task block design intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cohort", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-runtime", type=Path, required=True)
    parser.add_argument("--base-preregistration", type=Path, required=True)
    parser.add_argument("--failed-repair-log", type=Path)
    parser.add_argument("--parser-failure-log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    failure_log = args.parser_failure_log or args.failed_repair_log
    if failure_log is None:
        raise ValueError("a parser failure log is required")
    if args.output_dir.exists():
        raise FileExistsError("parser-qualified reserve is immutable")
    cohort = json.loads(args.base_cohort.read_text())
    manifest = json.loads(args.base_manifest.read_text())
    runtime = json.loads(args.base_runtime.read_text())
    prereg = json.loads(args.base_preregistration.read_text())
    if runtime["cohort_sha256"] != cohort["cohort_sha256"]:
        raise ValueError("base parser runtime/cohort mismatch")
    if any((cohort["answers_projected"], cohort["functional_programs_projected"],
            cohort["scene_graph_grounding_projected"], cohort["source_controller_read"])):
        raise ValueError("base public cohort crossed authority boundary")
    invalid = [row for row in runtime["rows"] if row["status"] != "SEMANTIC_SLOTS_FROZEN"]
    if not invalid:
        raise ValueError("base runtime has no invalid parser rows")
    invalid_videos = sorted({str(row["video_id"]) for row in invalid})
    kept_rows = [row for row in cohort["rows"] if str(row["video_id"]) not in invalid_videos]
    kept_videos = [row for row in cohort["video_receipts"] if str(row["video_id"]) not in invalid_videos]
    kept_runtime = [row for row in runtime["rows"] if str(row["video_id"]) not in invalid_videos]
    if len(kept_rows) != len(kept_runtime) or any(
        row["status"] != "SEMANTIC_SLOTS_FROZEN" for row in kept_runtime
    ):
        raise ValueError("video-block exclusion did not yield a fully valid runtime")
    strata = sorted({str(row["structural"]) for row in kept_rows})
    expected_k = int(manifest["selection"]["tasks_per_stratum_video"])
    video_ids = {str(row["video_id"]) for row in kept_rows}
    counts = {
        video_id: {stratum: sum(
            str(row["video_id"]) == video_id and str(row["structural"]) == stratum
            for row in kept_rows
        ) for stratum in strata}
        for video_id in video_ids
    }
    if strata != ["choose", "compare", "logic", "query", "verify"] or any(
        value != expected_k for by_stratum in counts.values() for value in by_stratum.values()
    ):
        raise ValueError("parser-qualified subset lost the frozen block balance")

    public = {
        **{key: value for key, value in cohort.items()
           if key not in {"rows", "video_receipts", "cohort_sha256", "status"}},
        "schema_version": "agqa-full-train-broad-public-v2",
        "status": "PARSER_QUALIFIED_BEFORE_RAW_VIDEO_OR_OUTCOME",
        "rows": kept_rows,
        "video_receipts": kept_videos,
        "parent_cohort_sha256": cohort["cohort_sha256"],
        "exclusion_rule": "REMOVE_ENTIRE_VIDEO_BLOCK_IF_ANY_QUESTION_ONLY_SEMANTIC_PARSE_IS_INVALID",
        "excluded_video_ids": invalid_videos,
    }
    public["cohort_sha256"] = stable_hash(public)
    semantic = {
        **{key: value for key, value in runtime.items()
           if key not in {"rows", "valid", "invalid", "runtime_sha256", "cohort_sha256"}},
        "cohort_sha256": public["cohort_sha256"],
        "rows": kept_runtime,
        "valid": len(kept_runtime),
        "invalid": 0,
        "base_runtime_sha256": runtime["runtime_sha256"],
        "excluded_video_ids": invalid_videos,
        "qualification_rule": public["exclusion_rule"],
    }
    semantic["runtime_sha256"] = stable_hash(semantic)
    gates = {
        "all_remaining_parses_valid": semantic["invalid"] == 0,
        "whole_video_block_exclusion": not (video_ids & set(invalid_videos)),
        "five_strata_times_three_tasks_per_video": all(
            value == expected_k for by_stratum in counts.values() for value in by_stratum.values()
        ),
        "tasks_unique": len({row["task_id"] for row in kept_rows}) == len(kept_rows),
        "raw_video_runtime_disjoint_inherited": manifest["gates"]["raw_video_runtime_disjoint"],
        "parser_supervision_task_disjoint_inherited": manifest["gates"]["parser_supervision_task_disjoint"],
        "no_outcome_authority": True,
    }
    derived_manifest = {
        "schema_version": "agqa-full-train-broad-parser-qualified-freeze-v2",
        "status": "AGQA_PARSER_QUALIFIED_RESERVE_FROZEN" if all(gates.values()) else "AGQA_PARSER_QUALIFIED_RESERVE_FAILED",
        "cohort_sha256": public["cohort_sha256"],
        "parent_cohort_sha256": cohort["cohort_sha256"],
        "parent_manifest_sha256": manifest["manifest_sha256"],
        "parent_runtime_sha256": runtime["runtime_sha256"],
        "parser_failure_log_sha256": file_sha256(failure_log),
        "selection": {"videos": len(kept_videos), "tasks": len(kept_rows),
                      "tasks_per_stratum_video": expected_k, "excluded_video_ids": invalid_videos},
        "gates": gates,
        "authority": {"raw_video_calls": 0, "answers_read": False,
                      "functional_programs_read": False, "scene_graphs_read": False,
                      "formal_outcomes_read": False},
    }
    derived_manifest["manifest_sha256"] = stable_hash(derived_manifest)
    revised_prereg = {
        **{key: value for key, value in prereg.items()
           if key not in {"cohort", "preregistration_sha256", "status", "schema_version"}},
        "schema_version": "agqa-full-train-broad-layer-b-preregistration-v2",
        "status": "FROZEN_AFTER_QUESTION_ONLY_PARSER_QUALIFICATION_AND_BEFORE_RAW_VIDEO_OR_OUTCOME",
        "parent_preregistration_sha256": prereg["preregistration_sha256"],
        "cohort": {
            **prereg["cohort"], "cohort_sha256": public["cohort_sha256"],
            "manifest_sha256": derived_manifest["manifest_sha256"],
            "videos": len(kept_videos), "tasks": len(kept_rows),
            "parser_validity_selected_before_video_or_outcome": True,
            "excluded_whole_video_blocks": invalid_videos,
        },
        "parser_qualification_amendment": {
            "rule": public["exclusion_rule"], "base_runtime_sha256": runtime["runtime_sha256"],
            "qualified_runtime_sha256": semantic["runtime_sha256"],
            "target_outcome_used": False, "raw_video_used": False,
        },
    }
    revised_prereg["preregistration_sha256"] = stable_hash(revised_prereg)
    args.output_dir.mkdir(parents=True)
    for name, value in (
        ("public_cohort.json", public), ("manifest.json", derived_manifest),
        ("semantic_runtime.json", semantic), ("preregistration.json", revised_prereg),
    ):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": derived_manifest["status"], "videos": len(kept_videos),
        "tasks": len(kept_rows), "excluded_video_ids": invalid_videos,
        "cohort_sha256": public["cohort_sha256"],
        "semantic_runtime_sha256": semantic["runtime_sha256"],
        "preregistration_sha256": revised_prereg["preregistration_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
