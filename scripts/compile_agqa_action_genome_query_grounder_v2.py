#!/usr/bin/env python3
"""Compile frozen SGDET/SlowFast predictions into shared AGQA V2 receipts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import re

from motif_transfer.agqa_action_genome_grounder import (
    build_stable_tracks, reciprocal_rank_fusion, relation_track_candidates,
)
from motif_transfer.agqa_query_grounder_v2 import (
    EntityTrack, QueryCandidateEvidence, QueryGroundingV2Receipt, TypedRoleEvent,
    deduplicate_typed_events, requested_query_role, requested_query_slot_ids,
)
from motif_transfer.agqa_query_object_grounder import (
    AGQA_OBJECT_ONTOLOGY, AGQA_OBJECT_QUERY_TERMS, canonical_object_label,
)
from motif_transfer.agqa_semantic_slots import relation_grounding_obligations
from motif_transfer.contracts import stable_hash
from scripts.compile_agqa_action_genome_sgdet_bindings import dense_window
from scripts.build_agqa_action_genome_sgdet_query_plans import native_temporal_window
from scripts.evaluate_agqa_layer_b_five_arm import _semantic


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_in_action_phrase(phrase: str) -> str | None:
    text = re.sub(r"[^a-z0-9/ ]+", " ", str(phrase).casefold())
    if "undressing" in text:
        return "clothes"
    # Prefer the longest public surface form so "doorway" is not reduced to
    # "door", and canonicalize only after the literal match.
    for term in sorted(set(AGQA_OBJECT_QUERY_TERMS), key=lambda value: (-len(value), value)):
        pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            value = canonical_object_label(term)
            return value if value in AGQA_OBJECT_ONTOLOGY else None
    return None


def _canonical_probability(value: float, *, numerical_tolerance: float = 1e-6) -> float:
    """Canonicalize probability-sized model output at a typed IR boundary.

    Some float32 softmax implementations return values a few ULPs above one.
    Those values are probabilities, not unbounded scores, so normalize only
    that numerical slack and fail closed on materially invalid output.
    """
    probability = float(value)
    if not math.isfinite(probability) or not (
        -numerical_tolerance <= probability <= 1.0 + numerical_tolerance
    ):
        raise ValueError("model probability is materially outside [0,1]")
    return min(1.0, max(0.0, probability))


def _track_for_label(tracks, label: str, lower: int, upper: int) -> tuple[str, int] | None:
    values = [row for row in tracks if row.canonical_label == canonical_object_label(label)]
    if not values:
        return None
    ranked = []
    midpoint = (lower + upper) / 2
    for row in values:
        in_window = [frame for frame in row.evidence_frames if lower <= frame <= upper]
        evidence = min(in_window or list(row.evidence_frames), key=lambda frame: abs(frame - midpoint))
        ranked.append((bool(in_window), row.confidence, -abs(evidence - midpoint), row.track_id, evidence))
    best = max(ranked)
    return str(best[3]), int(best[4])


def _slowfast_candidates(row: dict, tracks, lower: int, upper: int) -> tuple[dict, ...]:
    output = []
    for value in row.get("candidates", ()):  # already target-ontology role-safe
        label = canonical_object_label(str(value["candidate_label"]))
        match = _track_for_label(tracks, label, lower, upper)
        if match is None:
            continue
        track_id, evidence = match
        output.append({
            "candidate_label": label, "track_id": track_id,
            "sampled_frame_index": evidence, "score": float(value["action_score"]),
        })
    return tuple(sorted(output, key=lambda value: (
        -float(value["score"]), str(value["candidate_label"]), str(value["track_id"]),
    )))


def _pairwise_share(label: str, rows: tuple[dict, ...]) -> float | None:
    values: dict[str, float] = {}
    for row in rows:
        key = canonical_object_label(str(row["candidate_label"]))
        values[key] = max(values.get(key, 0.0), float(row["score"]))
    if label not in values:
        return None
    other = max((score for key, score in values.items() if key != label), default=0.0)
    denominator = values[label] + other
    return values[label] / denominator if denominator > 0 else 0.0


def _absolute_score(label: str, rows: tuple[dict, ...]) -> float | None:
    values = [float(row["score"]) for row in rows
              if canonical_object_label(str(row["candidate_label"])) == label]
    return max(values) if values else None


def _support_confidence(label: str, primary: tuple[dict, ...], secondary: tuple[dict, ...],
                        primary_weight: float) -> float:
    p, s = _pairwise_share(label, primary), _pairwise_share(label, secondary)
    pa, sa = _absolute_score(label, primary), _absolute_score(label, secondary)
    if p is None:
        relative = float(s or 0.0)
    elif s is None:
        relative = float(p)
    else:
        relative = primary_weight * p + (1.0 - primary_weight) * s
    # A large rank margin is not positive evidence when every raw score is
    # weak.  Require both a relative winner and at least one independently
    # calibrated neural view with strong absolute support.
    absolute = max(value for value in (pa, sa, 0.0) if value is not None)
    return min(relative, absolute)


def _segment(argmax_window: int, frame_count: int) -> tuple[int, int, int]:
    boundaries = [round(index * (frame_count - 1) / 3) for index in range(4)]
    index = min(2, max(0, int(argmax_window)))
    lower, upper = boundaries[index], boundaries[index + 1]
    return lower, upper, round((lower + upper) / 2)


def _nearest_position(native_indices: list[int], native_frame: int) -> int:
    return min(range(len(native_indices)), key=lambda index: (
        abs(native_indices[index] - native_frame), index,
    ))


def _query_sgdet_window(plan: dict, sampled_native_indices: list[int]) -> tuple[int, int]:
    native = native_temporal_window(plan)
    if native is None:
        return tuple(dense_window(plan))
    lower = _nearest_position(sampled_native_indices, native[0])
    upper = _nearest_position(sampled_native_indices, native[1])
    return (min(lower, upper), max(lower, upper))


def _obligation_sgdet_segment(obligation: dict, sampled_native_indices: list[int],
                              frame_count: int) -> tuple[int, int, int]:
    view = [int(x) for x in obligation.get("native_frame_index_view", ())]
    if view:
        lower = _nearest_position(sampled_native_indices, min(view))
        upper = _nearest_position(sampled_native_indices, max(view))
        lower, upper = min(lower, upper), max(lower, upper)
        return lower, upper, round((lower + upper) / 2)
    return _segment(int(obligation["argmax_window"]), frame_count)


def _shared_frame_view(raw: dict, slowfast_row: dict):
    """Return the union of every pixel observed by either frozen grounder."""
    native_to_hash = {
        int(index): str(digest) for index, digest in zip(
            raw["sampled_original_frame_indices"], raw["selected_frame_sha256s"])
    }
    for row in slowfast_row.get("presented_frame_receipts", ()):
        native_to_hash.setdefault(int(row["native_frame_index"]), str(row["frame_sha256"]))
    native = sorted(native_to_hash)
    union_position = {frame: index for index, frame in enumerate(native)}
    sgdet_to_union = {
        index: union_position[int(frame)]
        for index, frame in enumerate(raw["sampled_original_frame_indices"])
    }
    return tuple(native), tuple(native_to_hash[index] for index in native), sgdet_to_union


def _remap_to_shared_frames(tracks, events, candidates, mapping):
    remapped_tracks = tuple(replace(
        row, evidence_frames=tuple(sorted({mapping[index] for index in row.evidence_frames})),
    ) for row in tracks)
    remapped_events = tuple(replace(
        row, start_frame=mapping[row.start_frame], end_frame=mapping[row.end_frame],
        evidence_frames=tuple(sorted({mapping[index] for index in row.evidence_frames})),
    ) for row in events)
    remapped_candidates = tuple(replace(
        row, evidence_frames=tuple(sorted({mapping[index] for index in row.evidence_frames})),
    ) for row in candidates)
    return remapped_tracks, remapped_events, remapped_candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--query-plans", type=Path, required=True)
    parser.add_argument("--slowfast-bindings", type=Path, required=True)
    parser.add_argument("--minimum-object-score", type=float, default=0.05)
    parser.add_argument("--minimum-candidate-confidence", type=float, default=0.70)
    parser.add_argument("--sgdet-rank-weight", type=float, default=0.60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("compiled V2 grounder artifact is immutable")

    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    sgdet = json.loads(args.sgdet.read_text())
    plans = json.loads(args.query_plans.read_text())
    slowfast = json.loads(args.slowfast_bindings.read_text())
    if cohort["cohort_sha256"] != runtime["cohort_sha256"]:
        raise ValueError("cohort and semantic runtime differ")
    if sgdet.get("mode") != "sgdet" or any(sgdet.get(key) for key in (
        "question_read", "answer_read", "functional_program_read", "official_scene_graph_read",
        "per_video_action_genome_annotation_read", "source_controller_read", "target_outcome_read",
    )):
        raise ValueError("SGDET input violates prediction-only authority")
    if any(slowfast.get(key) for key in (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )):
        raise ValueError("SlowFast binding input violates authority")
    if not 0 <= args.minimum_candidate_confidence <= 1:
        raise ValueError("minimum candidate confidence must be in [0,1]")

    public = {str(row["task_id"]): row for row in cohort["rows"]}
    semantics = {str(row["task_id"]): _semantic(row["receipt"]) for row in runtime["rows"]}
    raw_by_video = {str(row["video_id"]): row for row in sgdet["rows"]}
    plan_by_task = {str(row["task_id"]): row for row in plans["rows"]}
    slow_by_task = {str(row["task_id"]): row for row in slowfast["rows"]}
    task_ids = set(plan_by_task)
    if not task_ids <= set(public) or not task_ids <= set(semantics) or task_ids != set(slow_by_task):
        raise ValueError("query plan, semantic runtime, and SlowFast task sets differ")
    if {str(public[task_id]["video_id"]) for task_id in task_ids} - set(raw_by_video):
        raise ValueError("SGDET report does not cover all query videos")

    tracking = {
        video_id: build_stable_tracks(row, minimum_object_score=args.minimum_object_score)
        for video_id, row in raw_by_video.items()
    }
    backend_sha = stable_hash({
        "protocol": "ACTION_GENOME_SGDET_SLOWFAST_QUERY_GROUNDER_V2",
        "sgdet_report_sha256": sgdet["report_sha256"],
        "slowfast_binding_report_sha256": slowfast["report_sha256"],
        "query_plan_report_sha256": plans["report_sha256"],
        "minimum_object_score": args.minimum_object_score,
        "minimum_candidate_confidence": args.minimum_candidate_confidence,
        "stable_tracking": {"nms_iou": 0.70, "minimum_affinity": 0.15, "maximum_gap": 6},
        "rank_fusion": {"method": "RECIPROCAL_RANK", "sgdet_weight": args.sgdet_rank_weight},
        "candidate_confidence": "MIN_WEIGHTED_PAIRWISE_TOP_SHARE_AND_MAX_ABSOLUTE_NEURAL_SCORE",
        "event_deduplication": {"typed_role_exact": True, "minimum_interval_iou": 0.5},
    })
    outputs = []
    for task_id in sorted(task_ids, key=lambda value: int(plan_by_task[value].get("cohort_position", 0))):
        plan = plan_by_task[task_id]
        semantic = semantics[task_id]
        video_id = str(public[task_id]["video_id"])
        raw = raw_by_video[video_id]
        stable = tracking[video_id]
        frame_count = int(raw["model_visible_frame_count"])
        if len(raw.get("selected_frame_sha256s", ())) != frame_count or not raw.get("video_sha256"):
            raise ValueError("SGDET raw receipt predates content-addressed frame evidence")
        sampled_native = [int(x) for x in raw["sampled_original_frame_indices"]]
        lower, upper = _query_sgdet_window(plan, sampled_native)
        predicate = str(plan.get("predicate") or "").casefold().strip()
        role = requested_query_role(semantic)
        root_slots = requested_query_slot_ids(semantic)
        primary = (relation_track_candidates(
            raw, stable, predicate=predicate, lower_frame=lower, upper_frame=upper,
        ) if predicate else ())
        secondary = (_slowfast_candidates(
            slow_by_task[task_id], stable.tracks, lower, upper,
        ) if predicate else ())
        fused = reciprocal_rank_fusion(
            primary, secondary, primary_weight=args.sgdet_rank_weight,
        )
        fused = tuple(row for row in fused if row.get("track_id") is not None)
        top = fused[0] if fused else None
        confidence = (_support_confidence(
            str(top["candidate_label"]), primary, secondary, args.sgdet_rank_weight,
        ) if top else 0.0)
        evidence = tuple(int(x) for x in (top.get("evidence_frames") or ())) if top else ()
        if top and not evidence:
            match = _track_for_label(stable.tracks, str(top["candidate_label"]), lower, upper)
            evidence = (match[1],) if match else ()

        candidates = []
        if top and evidence:
            candidates.append(QueryCandidateEvidence(
                str(top["track_id"]), role,
                "SUPPORTED" if confidence >= args.minimum_candidate_confidence else "UNKNOWN",
                confidence, evidence if confidence >= args.minimum_candidate_confidence else (),
            ))
        events: list[TypedRoleEvent] = []
        if top and evidence:
            events.append(TypedRoleEvent(
                f"R{len(events)}", predicate,
                (("agent", "T0"), (role, str(top["track_id"]))),
                min(evidence), max(evidence), evidence, confidence, tuple(root_slots),
            ))

        # Preserve nested relation references needed to localize action anchors.
        root_slot_set = set(root_slots)
        for nested_predicate, slot_id in relation_grounding_obligations(semantic):
            if slot_id in root_slot_set:
                continue
            ranked = relation_track_candidates(
                raw, stable, predicate=nested_predicate, lower_frame=0, upper_frame=frame_count - 1,
            )
            if not ranked:
                continue
            nested = ranked[0]
            events.append(TypedRoleEvent(
                f"R{len(events)}", nested_predicate,
                (("agent", "T0"), ("relation_object", str(nested["track_id"]))),
                int(nested["sampled_frame_index"]), int(nested["sampled_frame_index"]),
                (int(nested["sampled_frame_index"]),),
                _canonical_probability(float(nested["score"])), (slot_id,),
            ))

        # Add question-required temporal action anchors from the independent
        # frozen SlowFast view.  Exact object phrases use one stable track;
        # generic action heads use the top SGDET role candidate.
        for obligation in plan.get("action_obligations", ()):
            if "argmax_window" not in obligation:
                continue
            phrase = str(obligation["phrase"]).casefold().strip()
            slot_id = str(obligation["slot_id"])
            if slot_id in root_slot_set:
                continue
            seg_lo, seg_hi, center = _obligation_sgdet_segment(
                obligation, sampled_native, frame_count)
            label = _object_in_action_phrase(str(obligation.get("checkpoint_class_phrase", phrase)))
            matched = _track_for_label(stable.tracks, label, seg_lo, seg_hi) if label else None
            if matched is None:
                possible = relation_track_candidates(
                    raw, stable, predicate=phrase, lower_frame=seg_lo, upper_frame=seg_hi,
                )
                if possible:
                    matched = (str(possible[0]["track_id"]), int(possible[0]["sampled_frame_index"]))
                    label = str(possible[0]["candidate_label"])
            if matched is None:
                continue
            track_id, evidence_frame = matched
            event_predicate = phrase if label is None or canonical_object_label(label) not in phrase else phrase
            events.append(TypedRoleEvent(
                f"R{len(events)}", event_predicate,
                (("agent", "T0"), ("patient", track_id)),
                seg_lo, seg_hi, (evidence_frame if seg_lo <= evidence_frame <= seg_hi else center,),
                float(obligation.get("max_score", 0.0)), (slot_id,),
            ))
        events = list(deduplicate_typed_events(events))
        shared_indices, shared_hashes, sgdet_to_shared = _shared_frame_view(
            raw, slow_by_task[task_id])
        shared_tracks, shared_events, shared_candidates = _remap_to_shared_frames(
            stable.tracks, tuple(events), tuple(candidates), sgdet_to_shared)
        receipt = QueryGroundingV2Receipt.create(
            task_id=task_id, video_sha256=str(raw["video_sha256"]),
            semantic_slots_sha256=semantic.receipt_sha256,
            selected_frame_indices=shared_indices,
            selected_frame_sha256s=shared_hashes,
            tracks=shared_tracks, events=shared_events, candidates=shared_candidates,
            public_ontology_sha256=str(sgdet["ontology_sha256"]),
            grounder_backend_sha256=backend_sha, provider_calls=0,
        )
        outputs.append({
            "cohort_position": next(index for index, row in enumerate(cohort["rows"])
                                    if str(row["task_id"]) == task_id),
            "task_id": task_id, "video_id": video_id, "requested_role": role,
            "receipt": asdict(receipt), "provider_error": None,
            "root_predicate": predicate, "root_semantic_slot_ids": list(root_slots),
            "root_temporal_window": [lower, upper],
            "candidate_ranking": list(fused), "candidate_confidence": confidence,
            "candidate_support_threshold": args.minimum_candidate_confidence,
            "tracking_summary": {
                "stable_tracks": len(stable.tracks),
                "retained_detections": len(stable.retained_detection_indices),
            },
        })
    outputs.sort(key=lambda row: row["cohort_position"])
    body = {
        "schema_version": "agqa-action-genome-query-grounder-v2",
        "status": "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME",
        "cohort_sha256": cohort["cohort_sha256"],
        "semantic_runtime_sha256": runtime["runtime_sha256"],
        "public_ontology_sha256": sgdet["ontology_sha256"],
        "grounder_backend_sha256": backend_sha,
        "model": "TEMPURA_SGDET_PLUS_SLOWFAST_R50_CHARADES_FROZEN",
        "frame_budget": max((len(row["receipt"]["selected_frame_indices"]) for row in outputs), default=0),
        "component_frame_budgets": {
            "sgdet_unique_and_model_presentations": int(
                sgdet["maximum_model_visible_frame_budget"]),
            "slowfast_unique_sampled_frames": int(
                slowfast.get(
                    "unique_sampled_frame_budget",
                    max((len(row.get("presented_frame_receipts", ()))
                         for row in slowfast["rows"]), default=0),
                )
            ),
            "slowfast_model_frame_presentations": int(
                slowfast.get("frame_presentation_budget", 0)),
        },
        "rows": outputs, "reported_receipt_provider_cost_usd": 0.0,
        "all_harness_arms_share_exact_receipts": True,
        "stable_entity_tracking": True, "typed_semantic_roles": True,
        "cross_frame_typed_event_deduplication": True,
        "answer_blind_query_candidate_verification": True,
        "answer_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read": False,
        "target_outcome_read": False, "per_video_action_genome_annotation_read": False,
        "inputs": {
            "cohort_sha256": _file_hash(args.cohort),
            "semantic_runtime_sha256": _file_hash(args.semantic_runtime),
            "sgdet_sha256": _file_hash(args.sgdet),
            "query_plans_sha256": _file_hash(args.query_plans),
            "slowfast_bindings_sha256": _file_hash(args.slowfast_bindings),
        },
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "tasks": len(outputs),
        "supported": sum(bool(row["receipt"]["candidates"])
                         and row["receipt"]["candidates"][0]["status"] == "SUPPORTED"
                         for row in outputs),
        "mean_tracks": sum(row["tracking_summary"]["stable_tracks"] for row in outputs) / len(outputs),
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
