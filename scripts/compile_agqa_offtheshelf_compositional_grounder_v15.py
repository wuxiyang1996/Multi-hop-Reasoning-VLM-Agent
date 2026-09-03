#!/usr/bin/env python3
"""Compile question-blind detector scores into compositional AGQA events.

The visual backbones score every public action/relation class before outcomes
are read.  Target semantics select which typed event streams are exposed, but
the compiler never selects an answer candidate.  It therefore leaves temporal
composition to the shared actor or symbolic Harness.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from motif_transfer.agqa_action_genome_grounder import build_stable_tracks, relation_track_candidates
from motif_transfer.agqa_query_grounder_v2 import QueryGroundingV2Receipt, TypedRoleEvent, deduplicate_typed_events
from motif_transfer.agqa_query_object_grounder import canonical_object_label
from motif_transfer.agqa_semantic_slots import relation_grounding_obligations
from motif_transfer.agqa_strict_temporal_projection import rebind_nested_action_patients
from motif_transfer.contracts import stable_hash
from scripts.compile_agqa_action_genome_query_grounder_v2 import (
    _object_in_action_phrase, _remap_to_shared_frames, _shared_frame_view,
)
from scripts.evaluate_agqa_layer_b_five_arm import _semantic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _head(value: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", str(value).casefold())
    text = re.sub(r"^someone (?:is )?", "", text).strip()
    return text.split(maxsplit=1)[0] if text else ""


def _family_scores(obligation: dict, all_scores: list[dict]) -> list[dict]:
    """Return public action classes compatible with a parser action phrase."""
    class_id = obligation.get("class_id")
    if class_id:
        return [row for row in all_scores if str(row["class_id"]) == str(class_id)]
    matched_ids = {str(value) for value in obligation.get("matched_public_class_ids", ())}
    if matched_ids:
        return [row for row in all_scores if str(row["class_id"]) in matched_ids]
    head = _head(str(obligation.get("phrase") or ""))
    return [row for row in all_scores if head and _head(str(row["phrase"])) == head]


def _nearest_position(native_indices: list[int], native_frame: int) -> int:
    return min(range(len(native_indices)), key=lambda index: (
        abs(native_indices[index] - native_frame), index,
    ))


def _action_localization(
    obligation: dict, score_rows: list[dict], native_views: list[list[int]],
    relative: float, *, require_object: bool,
):
    object_typed = [row for row in score_rows if _object_in_action_phrase(str(row["phrase"]))]
    if require_object and object_typed:
        score_rows = object_typed
    if not score_rows or not native_views:
        return None
    window_scores = [
        max(float(row["window_scores"][index]) for row in score_rows)
        for index in range(len(native_views))
    ]
    peak = max(window_scores)
    if peak <= 0:
        return None
    argmax = max(range(len(window_scores)), key=window_scores.__getitem__)
    active = [index for index, score in enumerate(window_scores) if score >= relative * peak]
    # Keep only the contiguous active component containing the peak.  This is
    # a generic temporal-localization decoder, fixed before task outcomes.
    component = {argmax}
    cursor = argmax - 1
    while cursor in active:
        component.add(cursor); cursor -= 1
    cursor = argmax + 1
    while cursor in active:
        component.add(cursor); cursor += 1
    centers = [round((min(native_views[index]) + max(native_views[index])) / 2) for index in sorted(component)]
    winning = max(score_rows, key=lambda row: (
        float(row["window_scores"][argmax]), str(row["class_id"]),
    ))
    return {
        "native_lower": min(centers), "native_upper": max(centers),
        "native_center": centers[sorted(component).index(argmax)],
        "confidence": peak, "argmax_window": argmax,
        "active_windows": sorted(component),
        "checkpoint_class_id": str(winning["class_id"]),
        "checkpoint_class_phrase": str(winning["phrase"]),
        "window_scores": window_scores,
    }


def _relation_events(raw: dict, tracks, *, predicate: str, slot_id: str, minimum_score: float):
    observations = []
    for frame in range(int(raw["model_visible_frame_count"])):
        for row in relation_track_candidates(
            raw, tracks, predicate=predicate, lower_frame=frame, upper_frame=frame,
        ):
            if float(row["score"]) >= minimum_score:
                observations.append(row)
    by_track: dict[str, list[dict]] = {}
    for row in observations:
        by_track.setdefault(str(row["track_id"]), []).append(row)
    output = []
    for track_id, values in sorted(by_track.items()):
        values.sort(key=lambda row: int(row["sampled_frame_index"]))
        groups: list[list[dict]] = []
        for row in values:
            frame = int(row["sampled_frame_index"])
            if not groups or frame > int(groups[-1][-1]["sampled_frame_index"]) + 1:
                groups.append([row])
            else:
                groups[-1].append(row)
        for group in groups:
            frames = tuple(sorted({int(row["sampled_frame_index"]) for row in group}))
            output.append(TypedRoleEvent(
                event_id="R0", predicate=predicate,
                roles=(("agent", "T0"), ("relation_object", track_id)),
                start_frame=frames[0], end_frame=frames[-1], evidence_frames=frames,
                confidence=max(float(row["score"]) for row in group),
                semantic_slot_ids=(slot_id,),
            ))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--query-plans", type=Path, required=True)
    parser.add_argument("--action-grounding", type=Path, required=True)
    parser.add_argument("--minimum-object-score", type=float, default=0.05)
    parser.add_argument("--minimum-relation-score", type=float, default=0.25)
    parser.add_argument("--relative-action-window-threshold", type=float, default=0.50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V15 compositional grounding is immutable")
    if not 0 <= args.minimum_object_score <= 1 or not 0 <= args.minimum_relation_score <= 1:
        raise ValueError("grounding thresholds must be in [0,1]")
    if not 0 < args.relative_action_window_threshold <= 1:
        raise ValueError("relative action threshold must be in (0,1]")

    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    sgdet = json.loads(args.sgdet.read_text())
    plans = json.loads(args.query_plans.read_text())
    action = json.loads(args.action_grounding.read_text())
    if cohort["cohort_sha256"] != runtime["cohort_sha256"]:
        raise ValueError("cohort and semantic runtime differ")
    forbidden = (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )
    if any(sgdet.get(key) for key in forbidden) or any(action.get(key) for key in (
        "answers_read", "official_program_read", "official_scene_graph_read",
    )):
        raise ValueError("visual artifact crossed the authority boundary")
    if not action.get("all_class_scores_stored"):
        raise ValueError("question-blind full public action scores are required")

    public = {str(row["task_id"]): row for row in cohort["rows"]}
    semantics = {str(row["task_id"]): _semantic(row["receipt"]) for row in runtime["rows"]}
    plan_by_task = {str(row["task_id"]): row for row in plans["rows"]}
    action_by_task = {str(row["task_id"]): row for row in action["rows"]}
    sgdet_by_video = {str(row["video_id"]): row for row in sgdet["rows"]}
    wanted = set(public)
    if wanted != set(semantics) or wanted != set(plan_by_task) or wanted != set(action_by_task):
        raise ValueError("V15 artifacts do not cover identical task IDs")
    if {str(row["video_id"]) for row in public.values()} - set(sgdet_by_video):
        raise ValueError("SGDET does not cover every V15 video")

    inputs = {
        "cohort_sha256": _sha256(args.cohort),
        "semantic_runtime_sha256": _sha256(args.semantic_runtime),
        "sgdet_sha256": _sha256(args.sgdet),
        "query_plans_sha256": _sha256(args.query_plans),
        "action_grounding_sha256": _sha256(args.action_grounding),
    }
    backend_sha = stable_hash({
        "protocol": "AGQA_OFFTHESHELF_QUESTION_BLIND_COMPOSITIONAL_GROUNDER_V15",
        "inputs": inputs, "minimum_object_score": args.minimum_object_score,
        "minimum_relation_score": args.minimum_relation_score,
        "relative_action_window_threshold": args.relative_action_window_threshold,
        "action_decoder": "PEAK_CONNECTED_COMPONENT_OF_PUBLIC_CLASS_WINDOW_SCORES",
        "relation_decoder": "PUBLIC_PREDICATE_POSTERIOR_PRODUCT_THRESHOLD_AND_CONTIGUOUS_TRACKS",
        "answer_candidate_selected": False,
    })
    positions = {str(row["task_id"]): index for index, row in enumerate(cohort["rows"])}
    outputs = []
    for task_id in sorted(wanted, key=positions.__getitem__):
        semantic = semantics[task_id]
        plan = plan_by_task[task_id]
        action_row = action_by_task[task_id]
        raw = sgdet_by_video[str(public[task_id]["video_id"])]
        tracks = build_stable_tracks(raw, minimum_object_score=args.minimum_object_score)
        native_sgdet = [int(value) for value in raw["sampled_original_frame_indices"]]
        native_views = [[int(value) for value in view] for view in action_row["native_frame_index_views"]]
        events = []
        action_diagnostics = []
        object_bound_action_slots = {
            row.children[0] for row in semantic.slots
            if row.kind == "ACTION" and len(row.children) >= 2
        }
        object_bound_action_slots.update(
            slot_id for _, slot_id in relation_grounding_obligations(semantic)
        )
        for obligation in plan.get("action_obligations", ()):
            compatible = _family_scores(obligation, action_row["all_class_scores"])
            localized = _action_localization(
                obligation, compatible, native_views, args.relative_action_window_threshold,
                require_object=str(obligation["slot_id"]) in object_bound_action_slots,
            )
            if localized is None:
                action_diagnostics.append({"slot_id": str(obligation["slot_id"]), "status": "ABSTAIN_NO_PUBLIC_ACTION_SCORE"})
                continue
            lower = _nearest_position(native_sgdet, localized["native_lower"])
            upper = _nearest_position(native_sgdet, localized["native_upper"])
            lower, upper = min(lower, upper), max(lower, upper)
            center = _nearest_position(native_sgdet, localized["native_center"])
            label = _object_in_action_phrase(localized["checkpoint_class_phrase"])
            matching_tracks = [
                row for row in tracks.tracks
                if label is not None and row.canonical_label == canonical_object_label(label)
            ]
            in_window = [
                (row.confidence, row.track_id, frame)
                for row in matching_tracks for frame in row.evidence_frames
                if lower <= frame <= upper
            ]
            if in_window:
                patient = max(in_window)[1]
                evidence = max(in_window)[2]
            elif matching_tracks:
                # SlowFast provides in-window action evidence.  SGDET supplies
                # the stable video-level entity identity and may miss it at
                # that exact sample; retain the highest-confidence same-label
                # track rather than replacing the typed patient with person.
                patient = max((row.confidence, row.track_id) for row in matching_tracks)[1]
                evidence = center
            else:
                patient = None
                evidence = center
            roles = (("agent", "T0"),) + ((("patient", patient),) if patient else ())
            events.append(TypedRoleEvent(
                event_id="R0", predicate=str(obligation["phrase"]).casefold().strip(),
                roles=roles, start_frame=lower, end_frame=upper,
                evidence_frames=(evidence,), confidence=float(localized["confidence"]),
                semantic_slot_ids=(str(obligation["slot_id"]),),
            ))
            action_diagnostics.append({
                "slot_id": str(obligation["slot_id"]), "phrase": str(obligation["phrase"]),
                "status": "GROUNDED", **localized,
            })
        for predicate, slot_id in relation_grounding_obligations(semantic):
            events.extend(_relation_events(
                raw, tracks, predicate=predicate, slot_id=slot_id,
                minimum_score=args.minimum_relation_score,
            ))
        events, rebound = rebind_nested_action_patients(events, tracks.tracks, semantic)
        events = list(deduplicate_typed_events(events))
        shared_indices, shared_hashes, remap = _shared_frame_view(raw, action_row)
        shared_tracks, shared_events, _ = _remap_to_shared_frames(
            tracks.tracks, tuple(events), (), remap,
        )
        receipt = QueryGroundingV2Receipt.create(
            task_id=task_id, video_sha256=str(raw["video_sha256"]),
            semantic_slots_sha256=semantic.receipt_sha256,
            selected_frame_indices=shared_indices, selected_frame_sha256s=shared_hashes,
            tracks=shared_tracks, events=shared_events, candidates=(),
            public_ontology_sha256=str(sgdet["ontology_sha256"]),
            grounder_backend_sha256=backend_sha, provider_calls=0,
        )
        outputs.append({
            "cohort_position": positions[task_id], "task_id": task_id,
            "video_id": str(public[task_id]["video_id"]), "receipt": asdict(receipt),
            "semantic_root": str(runtime["rows"][positions[task_id]]["predicted_semantics"]).split("(", 1)[0],
            "action_diagnostics": action_diagnostics,
            "rebound_nested_action_events": rebound,
            "action_events": sum(row["status"] == "GROUNDED" for row in action_diagnostics),
            "relation_events": len(shared_events) - sum(row["status"] == "GROUNDED" for row in action_diagnostics),
            "candidate_confidence": 0.0, "answer_candidate_selected": False,
        })

    body = {
        "schema_version": "agqa-offtheshelf-compositional-query-grounder-v15",
        "status": "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME",
        "cohort_sha256": cohort["cohort_sha256"],
        "semantic_runtime_sha256": runtime["runtime_sha256"],
        "public_ontology_sha256": sgdet["ontology_sha256"],
        "grounder_backend_sha256": backend_sha,
        "model": "TEMPURA_SGDET_PLUS_SLOWFAST_R50_CHARADES_FROZEN",
        "rows": outputs, "inputs": {**inputs, "slowfast_bindings_sha256": "NOT_USED_COMPOSITIONAL_NO_ENTITY_CANDIDATE"},
        "component_frame_budgets": {
            "sgdet_unique_and_model_presentations": int(sgdet["maximum_model_visible_frame_budget"]),
            "slowfast_unique_sampled_frames": int(action["unique_sampled_frame_budget"]),
            "slowfast_model_frame_presentations": int(action["frame_presentation_budget"]),
        },
        "all_harness_arms_share_exact_receipts": True,
        "stable_entity_tracking": True, "typed_semantic_roles": True,
        "cross_frame_typed_event_deduplication": True,
        "question_blind_visual_class_scores": True,
        "answer_blind_semantic_slot_binding": True,
        "answer_candidate_selected": False, "provider_calls": 0,
        "answer_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read": False,
        "target_outcome_read": False, "per_video_action_genome_annotation_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "tasks": len(outputs),
        "two_or_more_events": sum(len(row["receipt"]["events"]) >= 2 for row in outputs),
        "two_or_more_action_events": sum(row["action_events"] >= 2 for row in outputs),
        "mean_events": sum(len(row["receipt"]["events"]) for row in outputs) / len(outputs),
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
