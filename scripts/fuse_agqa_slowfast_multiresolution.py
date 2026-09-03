#!/usr/bin/env python3
"""Fuse frozen SlowFast classification and localization receipts.

The uniform48 view remains the only source of action/object candidate scores.
The dense10x32 view contributes only native-frame temporal localization and
pixel receipts.  Both views use the same frozen public Charades checkpoint;
neither side may read AGQA answers, programs, scene graphs, controllers, or
target outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


FORBIDDEN_READ_FLAGS = (
    "answer_read",
    "official_program_read",
    "official_scene_graph_read",
    "functional_program_read",
    "source_controller_read",
    "target_outcome_read",
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_authority_safe(name: str, report: dict) -> None:
    crossed = [key for key in FORBIDDEN_READ_FLAGS if report.get(key)]
    if crossed:
        raise ValueError(f"{name} crossed authority boundary: {crossed}")


def _by_task(report: dict) -> dict[str, dict]:
    rows = {str(row["task_id"]): row for row in report["rows"]}
    if len(rows) != len(report["rows"]):
        raise ValueError("duplicate task_id")
    return rows


def fuse_query_plans(uniform: dict, dense: dict) -> dict:
    _assert_authority_safe("uniform query plans", uniform)
    _assert_authority_safe("dense query plans", dense)
    uniform_rows, dense_rows = _by_task(uniform), _by_task(dense)
    if set(uniform_rows) != set(dense_rows):
        raise ValueError("uniform and dense query-plan task sets differ")

    outputs = []
    localized = 0
    for task_id, coarse in uniform_rows.items():
        fine = dense_rows[task_id]
        for key in ("predicate", "temporal_operator", "source_frame_count", "status"):
            if coarse.get(key) != fine.get(key):
                raise ValueError(f"query-plan {key} differs for {task_id}")
        fine_obligations = {
            (str(row["slot_id"]), str(row["phrase"])): row
            for row in fine.get("action_obligations", ())
        }
        coarse_keys = {
            (str(row["slot_id"]), str(row["phrase"]))
            for row in coarse.get("action_obligations", ())
        }
        if coarse_keys != set(fine_obligations):
            raise ValueError(f"action obligations differ for {task_id}")
        obligations = []
        for original in coarse.get("action_obligations", ()):
            key = (str(original["slot_id"]), str(original["phrase"]))
            localization = fine_obligations[key]
            fused = dict(original)
            view = [int(value) for value in localization.get("native_frame_index_view", ())]
            if view:
                # Keep the already-qualified uniform48 interval for root
                # entity candidate selection.  The dense interval is a
                # separate typed field consumed only by action-anchor event
                # localization in the strict compiler.
                fused["localization_native_frame_index_view"] = view
                fused["localization_sampling"] = "dense10x32"
                fused["localization_argmax_window"] = int(
                    localization.get("argmax_window", 0)
                )
                localized += 1
            obligations.append(fused)
        outputs.append({
            **coarse,
            "inspection_indices": sorted(set(
                int(value) for value in coarse.get("inspection_indices", ())
            ) | set(
                int(value) for value in fine.get("inspection_indices", ())
            )),
            "native_frame_index_views": list(fine.get("native_frame_index_views", ())),
            "action_obligations": obligations,
            "candidate_scoring_sampling": "uniform48",
            "temporal_localization_sampling": "dense10x32",
        })

    report = {
        "schema_version": "agqa-slowfast-multiresolution-query-plans-v1",
        "status": "QUERY_PLANS_FROZEN_WITHOUT_OUTCOMES",
        "classification_source_report_sha256": uniform["report_sha256"],
        "localization_source_report_sha256": dense["report_sha256"],
        "candidate_scoring_sampling": "uniform48",
        "temporal_localization_sampling": "dense10x32",
        "localized_action_obligations": localized,
        "question_read_by_shared_parser": True,
        "answer_read": False,
        "official_scene_graph_read": False,
        "functional_program_read": False,
        "source_controller_read": False,
        "target_outcome_read": False,
        "rows": outputs,
    }
    report["report_sha256"] = stable_hash(report)
    return report


def _union_frame_receipts(*groups) -> tuple[list[dict], int]:
    """Return one canonical receipt per native time coordinate.

    Uniform48 hashes frames after its bounded resize, while dense10x32 hashes
    the decoded native frame.  A shared native index can therefore have two
    legitimate presentation hashes.  Later groups are the canonical temporal
    view; every complete source artifact remains independently content
    addressed in the fused report.
    """
    by_index: dict[int, str] = {}
    collisions = 0
    for group in groups:
        for row in group:
            index, digest = int(row["native_frame_index"]), str(row["frame_sha256"])
            if index in by_index and by_index[index] != digest:
                collisions += 1
            by_index[index] = digest
    return ([
        {"native_frame_index": index, "frame_sha256": by_index[index]}
        for index in sorted(by_index)
    ], collisions)


def fuse_bindings(uniform: dict, dense: dict, *, fused_plan_sha256: str) -> dict:
    _assert_authority_safe("uniform action bindings", uniform)
    _assert_authority_safe("dense action bindings", dense)
    uniform_rows, dense_rows = _by_task(uniform), _by_task(dense)
    if set(uniform_rows) != set(dense_rows):
        raise ValueError("uniform and dense binding task sets differ")

    outputs = []
    unique_budget = classification_presentations = localization_presentations = 0
    preprocessing_hash_collisions = 0
    for task_id, scoring in uniform_rows.items():
        localization = dense_rows[task_id]
        for key in ("predicate", "video_id", "video_sha256", "source_frame_count"):
            if scoring.get(key) != localization.get(key):
                raise ValueError(f"action-binding {key} differs for {task_id}")
        receipts, collisions = _union_frame_receipts(
            scoring.get("presented_frame_receipts", ()),
            localization.get("presented_frame_receipts", ()),
        )
        preprocessing_hash_collisions += collisions
        unique_budget = max(unique_budget, len(receipts))
        classification_presentations = max(
            classification_presentations,
            sum(len(view) for view in scoring.get("native_frame_index_views", ())),
        )
        localization_presentations = max(
            localization_presentations,
            sum(len(view) for view in localization.get("native_frame_index_views", ())),
        )
        outputs.append({
            **scoring,
            "native_frame_index_views": list(
                localization.get("native_frame_index_views", ())
            ),
            "presented_frame_receipts": receipts,
            "candidate_scoring_sampling": "uniform48",
            "temporal_localization_sampling": "dense10x32",
        })

    report = {
        "schema_version": "agqa-slowfast-multiresolution-action-bindings-v1",
        "status": "SLOWFAST_ACTION_BINDINGS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
        "routing_rule": uniform.get("routing_rule"),
        "scoring_rule": uniform.get("scoring_rule"),
        "localization_rule": "argmax native-frame view from dense10x32 for the same typed action obligation",
        "candidate_scoring_sampling": "uniform48",
        "temporal_localization_sampling": "dense10x32",
        "classification_source_report_sha256": uniform["report_sha256"],
        "localization_source_report_sha256": dense["report_sha256"],
        "query_plans_file_sha256": fused_plan_sha256,
        "unique_sampled_frame_budget": unique_budget,
        "classification_frame_presentation_budget": classification_presentations,
        "localization_frame_presentation_budget": localization_presentations,
        "frame_presentation_budget": (
            classification_presentations + localization_presentations
        ),
        "canonical_frame_receipt_policy": "DENSE_NATIVE_PRESENTATION_AT_SHARED_INDEX",
        "preprocessing_hash_collisions": preprocessing_hash_collisions,
        "answer_read": False,
        "official_scene_graph_read": False,
        "functional_program_read": False,
        "source_controller_read": False,
        "target_outcome_read": False,
        "rows": outputs,
    }
    report["action_grounding_file_sha256"] = stable_hash({
        "classification": uniform.get("action_grounding_file_sha256"),
        "localization": dense.get("action_grounding_file_sha256"),
    })
    report["report_sha256"] = stable_hash(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform-query-plans", type=Path, required=True)
    parser.add_argument("--dense-query-plans", type=Path, required=True)
    parser.add_argument("--uniform-bindings", type=Path, required=True)
    parser.add_argument("--dense-bindings", type=Path, required=True)
    parser.add_argument("--output-query-plans", type=Path, required=True)
    parser.add_argument("--output-bindings", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_query_plans, args.output_bindings):
        if output.exists():
            raise FileExistsError("multiresolution output is immutable")

    uniform_plans = json.loads(args.uniform_query_plans.read_text())
    dense_plans = json.loads(args.dense_query_plans.read_text())
    uniform_bindings = json.loads(args.uniform_bindings.read_text())
    dense_bindings = json.loads(args.dense_bindings.read_text())
    plans = fuse_query_plans(uniform_plans, dense_plans)
    args.output_query_plans.parent.mkdir(parents=True, exist_ok=True)
    args.output_query_plans.write_text(json.dumps(plans, indent=2, sort_keys=True) + "\n")
    bindings = fuse_bindings(
        uniform_bindings,
        dense_bindings,
        fused_plan_sha256=_file_hash(args.output_query_plans),
    )
    args.output_bindings.parent.mkdir(parents=True, exist_ok=True)
    args.output_bindings.write_text(json.dumps(bindings, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "MULTIRESOLUTION_SLOWFAST_FUSION_COMPLETE",
        "tasks": len(plans["rows"]),
        "localized_action_obligations": plans["localized_action_obligations"],
        "unique_sampled_frame_budget": bindings["unique_sampled_frame_budget"],
        "frame_presentation_budget": bindings["frame_presentation_budget"],
        "query_plan_report_sha256": plans["report_sha256"],
        "binding_report_sha256": bindings["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
