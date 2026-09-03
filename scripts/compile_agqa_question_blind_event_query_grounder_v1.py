#!/usr/bin/env python3
"""Query a frozen per-video event inventory into AGQA Grounding V2 receipts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_action_genome_grounder import build_stable_tracks
from motif_transfer.agqa_query_grounder_v2 import (
    QueryCandidateEvidence, QueryGroundingV2Receipt,
    deduplicate_typed_events, requested_query_role, requested_query_slot_ids,
)
from motif_transfer.agqa_question_blind_event_grounder import (
    QuestionBlindTypedEvent, bind_event_to_semantic_slots, query_event_candidates,
)
from motif_transfer.contracts import stable_hash
from scripts.compile_agqa_action_genome_query_grounder_v2 import _query_sgdet_window
from scripts.evaluate_agqa_layer_b_five_arm import _semantic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _event(value: dict) -> QuestionBlindTypedEvent:
    return QuestionBlindTypedEvent(
        event_id=str(value["event_id"]), predicate=str(value["predicate"]),
        subject_track_id=str(value["subject_track_id"]),
        object_track_id=str(value["object_track_id"]),
        object_role=str(value["object_role"]),
        start_frame=int(value["start_frame"]), end_frame=int(value["end_frame"]),
        evidence_frames=tuple(int(x) for x in value["evidence_frames"]),
        confidence=float(value["confidence"]),
        source_clip_ids=tuple(str(x) for x in value["source_clip_ids"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--query-plans", type=Path, required=True)
    parser.add_argument("--event-inventory", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--minimum-object-score", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("question-blind event query grounding is immutable")
    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    sgdet = json.loads(args.sgdet.read_text())
    plans = json.loads(args.query_plans.read_text())
    inventory = json.loads(args.event_inventory.read_text())
    protocol = json.loads(args.protocol.read_text())
    if inventory.get("status") != "QUESTION_BLIND_EVENT_INVENTORY_FROZEN_BEFORE_TASK_QUERY_OR_OUTCOME":
        raise ValueError("event inventory is not frozen before task query")
    forbidden = (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )
    if any(inventory.get(key) for key in forbidden) or any(sgdet.get(key) for key in forbidden):
        raise ValueError("grounder input crossed its authority boundary")
    expected = protocol["immutable_inputs"]
    actual = {
        "cohort_sha256": cohort["cohort_sha256"],
        "semantic_runtime_file_sha256": _sha256(args.semantic_runtime),
        "sgdet_file_sha256": _sha256(args.sgdet),
        "query_plans_file_sha256": _sha256(args.query_plans),
        "event_inventory_file_sha256": _sha256(args.event_inventory),
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise ValueError(f"immutable input mismatch: {key}")
    if args.minimum_object_score != float(
        protocol["question_blind_event_acquisition"]["minimum_object_score"]
    ):
        raise ValueError("stable tracking threshold differs from frozen protocol")
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    semantics = {str(row["task_id"]): _semantic(row["receipt"]) for row in runtime["rows"]}
    plan_by_task = {str(row["task_id"]): row for row in plans["rows"]}
    raw_by_video = {str(row["video_id"]): row for row in sgdet["rows"]}
    inventory_by_video = {str(row["video_id"]): row for row in inventory["rows"]}
    task_ids = set(plan_by_task)
    if task_ids != set(public) or task_ids != set(semantics):
        raise ValueError("cohort, semantics, and query-plan task sets differ")
    if {str(row["video_id"]) for row in public.values()} - set(inventory_by_video):
        raise ValueError("event inventory does not cover every query video")
    backend_sha = stable_hash({
        "protocol": "AGQA_QUESTION_BLIND_EVENT_QUERY_GROUNDER_V1",
        "event_inventory_report_sha256": inventory["report_sha256"],
        "query_plan_report_sha256": plans["report_sha256"],
        "minimum_object_score": args.minimum_object_score,
        "query_score": "MAX_PROVIDER_CONFIDENCE_EXACT_PUBLIC_PREDICATE_ROLE_IN_SCOPE",
        "event_deduplication": {"typed_track_exact": True, "minimum_interval_iou": 0.5},
    })
    positions = {str(row["task_id"]): index for index, row in enumerate(cohort["rows"])}
    outputs = []
    for task_id in sorted(task_ids, key=positions.__getitem__):
        public_row = public[task_id]
        video_id = str(public_row["video_id"])
        raw = raw_by_video[video_id]
        video_inventory = inventory_by_video[video_id]
        stable = build_stable_tracks(raw, minimum_object_score=args.minimum_object_score)
        tracks = {row.track_id: row for row in stable.tracks}
        events = tuple(_event(row) for row in video_inventory["events"])
        allowed = frozenset(int(row["frame_id"]) for row in video_inventory["presented_frames"])
        known = frozenset(tracks)
        for event in events:
            event.validate(known_track_ids=known, allowed_frame_ids=allowed)
        semantic = semantics[task_id]
        plan = plan_by_task[task_id]
        requested_role = requested_query_role(semantic)
        slot_ids = requested_query_slot_ids(semantic)
        lower, upper = _query_sgdet_window(
            plan, [int(value) for value in raw["sampled_original_frame_indices"]],
        )
        predicate = str(plan.get("predicate") or "").strip()
        # A generic "interact with" query does not identify which public
        # contact/attention predicate is answer-bearing.  Choosing the most
        # confident incidental state (often standing-on-floor or wearing)
        # would be an unjustified semantic guess, so this view fails closed.
        ranking = (() if not predicate else query_event_candidates(
            events, predicate=predicate, requested_role=requested_role,
            lower_frame=lower, upper_frame=upper,
        ))
        top = ranking[0] if ranking else None
        candidates = () if top is None else (QueryCandidateEvidence(
            track_id=str(top["track_id"]), requested_role=requested_role,
            status="SUPPORTED", confidence=float(top["score"]),
            evidence_frames=tuple(int(x) for x in top["evidence_frames"]),
        ),)
        bound = []
        matching_ids = {
            event_id for row in ranking for event_id in row["event_ids"]
        }
        for event in events:
            if event.event_id not in matching_ids:
                continue
            bound.append(bind_event_to_semantic_slots(
                event, event_id=f"R{len(bound)}", semantic_slot_ids=slot_ids,
            ))
        bound = list(deduplicate_typed_events(bound))
        receipt = QueryGroundingV2Receipt.create(
            task_id=task_id, video_sha256=str(raw["video_sha256"]),
            semantic_slots_sha256=semantic.receipt_sha256,
            selected_frame_indices=tuple(int(x) for x in raw["sampled_original_frame_indices"]),
            selected_frame_sha256s=tuple(str(x) for x in raw["selected_frame_sha256s"]),
            tracks=stable.tracks, events=bound, candidates=candidates,
            public_ontology_sha256=str(sgdet["ontology_sha256"]),
            grounder_backend_sha256=backend_sha,
            provider_calls=sum(
                int((clip.get("usage") or {}).get("provider_attempts", 0))
                * int(not clip.get("cache_reused"))
                for clip in video_inventory["clips"]
            ),
        )
        errors = [
            str(clip["provider_error"]) for clip in video_inventory["clips"]
            if clip.get("provider_error") not in {None, "NO_VISIBLE_PERSON_OBJECT_PAIR"}
        ]
        candidate_ranking = [{
            **row,
            "candidate_label": tracks[str(row["track_id"])].canonical_label,
        } for row in ranking]
        outputs.append({
            "cohort_position": positions[task_id], "task_id": task_id,
            "video_id": video_id, "requested_role": requested_role,
            "receipt": asdict(receipt),
            "provider_error": ";".join(errors) if errors else None,
            "root_predicate": predicate,
            "root_semantic_slot_ids": list(slot_ids),
            "root_temporal_window": [lower, upper],
            "candidate_ranking": candidate_ranking,
            "candidate_confidence": float(top["score"]) if top else 0.0,
            "candidate_support_threshold": 0.0,
            "question_blind_event_ids": sorted(matching_ids),
            "tracking_summary": {
                "stable_tracks": len(stable.tracks),
                "retained_detections": len(stable.retained_detection_indices),
            },
        })
    report = {
        "schema_version": "agqa-question-blind-event-query-grounder-v1",
        "status": "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME",
        "cohort_sha256": cohort["cohort_sha256"],
        "semantic_runtime_sha256": runtime["runtime_sha256"],
        "public_ontology_sha256": sgdet["ontology_sha256"],
        "grounder_backend_sha256": backend_sha,
        "model": inventory["model"],
        "frame_budget": int(sgdet["maximum_model_visible_frame_budget"]),
        "question_blind_vlm_unique_frame_budget": int(
            inventory["maximum_unique_vlm_frames_per_video"]
        ),
        "rows": outputs,
        "reported_receipt_provider_cost_usd": float(inventory["reported_cost_usd"]),
        "provider_and_contract_success_fraction": float(
            inventory["clip_contract_success_fraction"]
        ),
        "all_harness_arms_share_exact_receipts": True,
        "stable_entity_tracking": True, "typed_semantic_roles": True,
        "cross_frame_typed_event_deduplication": True,
        "answer_blind_query_candidate_verification": True,
        "question_blind_event_acquisition": True,
        "question_read_during_event_acquisition": False,
        "answer_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read": False,
        "target_outcome_read": False, "per_video_action_genome_annotation_read": False,
        "inputs": actual,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "tasks": len(outputs),
        "supported": sum(bool(row["receipt"]["candidates"]) for row in outputs),
        "mean_tracks": sum(row["tracking_summary"]["stable_tracks"] for row in outputs) / len(outputs),
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
