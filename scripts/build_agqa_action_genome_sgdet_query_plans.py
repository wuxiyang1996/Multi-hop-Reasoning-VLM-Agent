#!/usr/bin/env python3
"""Build answer-blind typed query plans for frozen SGDET video receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.agqa_query_grounder_v2 import requested_query_predicates
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _semantic


def _root_temporal_operator(semantic) -> str:
    by_id = {row.slot_id: row for row in semantic.slots}

    def visit(slot_id: str):
        row = by_id[slot_id]
        if row.kind == "TEMPORAL_CONSTRAINT" and row.children:
            first = by_id[row.children[0]]
            values = [first.surface]
            values.extend(by_id[child].surface for child in first.children)
            for value in values:
                normalized = str(value).strip().upper()
                if normalized in {"BEFORE", "AFTER", "WHILE", "BETWEEN", "VIDEO"}:
                    return normalized
        for child in row.children:
            found = visit(child)
            if found:
                return found
        return None

    return visit(semantic.root_slot_id) or "VIDEO"


def _action_family_matches(phrase: str, class_phrase: str) -> bool:
    phrase = phrase.casefold().strip(); class_phrase = class_phrase.casefold().strip()
    families = {
        "putting down": ("putting ",), "putting": ("putting ",),
        "taking": ("taking ",), "opening": ("opening ",),
        "closing": ("closing ",), "washing": ("washing ", "wash "),
        "holding": ("holding ", "someone is holding "),
        "throwing": ("throwing ",), "tidying": ("tidying ", "tidying up "),
        "sitting": ("sitting ",), "standing up": ("someone is standing up ",),
        "undressing": ("someone is undressing",),
        "playing on": ("playing with ", "working/playing on "),
    }
    return any(class_phrase.startswith(prefix) for prefix in families.get(phrase, ()))


def _augment_obligations(obligations: list[dict], action_row: dict) -> list[dict]:
    output = []
    all_scores = action_row.get("all_class_scores", [])
    for obligation in obligations:
        value = dict(obligation)
        if "argmax_window" not in value and all_scores:
            matched = [row for row in all_scores
                       if _action_family_matches(str(value["phrase"]), str(row["phrase"]))]
            if matched:
                window_scores = [max(float(row["window_scores"][index]) for row in matched)
                                 for index in range(len(matched[0]["window_scores"]))]
                value.update({
                    "mapping_status": "COMPOSED_PUBLIC_ACTION_FAMILY",
                    "matched_public_class_ids": [str(row["class_id"]) for row in matched],
                    "window_scores": window_scores,
                    "max_score": max(window_scores),
                    "argmax_window": max(range(len(window_scores)), key=window_scores.__getitem__),
                })
        if "argmax_window" in value:
            views = action_row.get("native_frame_index_views", ())
            index = int(value["argmax_window"])
            if 0 <= index < len(views) and views[index]:
                value["native_frame_index_view"] = [int(x) for x in views[index]]
        output.append(value)
    return output


def native_temporal_window(plan: dict) -> tuple[int, int] | None:
    """Project frozen action-view evidence into a question-scope interval."""
    source_frames = plan.get("source_frame_count")
    if source_frames is None or int(source_frames) <= 0:
        return None
    last = int(source_frames) - 1
    segments = []
    for obligation in plan.get("action_obligations", ()):
        view = [int(x) for x in obligation.get("native_frame_index_view", ())]
        if view:
            segments.append((min(view), max(view)))
    temporal = str(plan.get("temporal_operator", "VIDEO")).upper()
    if temporal == "VIDEO" or not segments:
        return 0, last
    if temporal == "BEFORE":
        return 0, max(0, segments[0][0] - 1)
    if temporal == "AFTER":
        return min(last, segments[0][1] + 1), last
    if temporal == "WHILE":
        return segments[0]
    if temporal == "BETWEEN" and len(segments) >= 2:
        left, right = sorted(segments[:2])
        return min(last, left[1] + 1), max(0, right[0] - 1)
    return 0, last


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--action-grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("SGDET query plans are immutable")
    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    action = json.loads(args.action_grounding.read_text())
    action_by_task = {str(row["task_id"]): row for row in action["rows"]}
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    semantics = {str(row["task_id"]): _semantic(row["receipt"])
                 for row in runtime["rows"]}
    rows = []
    for source in action["rows"]:
        task_id = str(source["task_id"])
        predicates = requested_query_predicates(semantics[task_id])
        predicate = predicates[0] if predicates else None
        native_views = [[int(x) for x in view]
                        for view in source.get("native_frame_index_views", ())]
        rows.append({
            "task_id": task_id,
            "predicate": predicate,
            "status": "QUERY_PLAN_FROZEN" if predicate else "ABSTAIN_NO_EXPLICIT_PREDICATE",
            "temporal_operator": _root_temporal_operator(semantics[task_id]),
            "inspection_indices": sorted({index for view in native_views for index in view}),
            "source_frame_count": source.get("source_frame_count"),
            "native_frame_index_views": native_views,
            "action_obligations": _augment_obligations(
                list(source["obligations"]), action_by_task[task_id]),
        })
    report = {
        "schema_version": "agqa-action-genome-sgdet-query-plans-v1",
        "status": "QUERY_PLANS_FROZEN_WITHOUT_OUTCOMES",
        "question_read_by_shared_parser": True,
        "answer_read": False,
        "official_scene_graph_read": False,
        "functional_program_read": False,
        "source_controller_read": False,
        "target_outcome_read": False,
        "rows": rows,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
