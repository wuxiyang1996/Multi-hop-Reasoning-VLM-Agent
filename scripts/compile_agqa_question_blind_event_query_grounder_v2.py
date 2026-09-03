#!/usr/bin/env python3
"""Compile answer-blind anchors plus question-blind events into AGQA receipts.

Visual acquisition is target-native and shared by every experimental arm.  It
uses raw frames, public object/action ontology, stable tracks, and parser
anchor phrases, but never answers, official STSGs, functional programs, game
controllers, or target outcomes.  Temporal execution is deterministic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_action_genome_grounder import build_stable_tracks
from motif_transfer.agqa_query_grounder_v2 import (
    QueryCandidateEvidence,
    QueryGroundingV2Receipt,
    deduplicate_typed_events,
    requested_query_role,
    requested_query_slot_ids,
)
from motif_transfer.agqa_question_blind_event_grounder import (
    QuestionBlindTypedEvent,
    bind_event_to_semantic_slots,
    query_temporal_event_candidates,
)
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _semantic


CONSUMED_INVENTORY_STATUS = (
    "CONSUMED_DEVELOPMENT_EVENT_INVENTORY_NOT_TRANSFER_EVIDENCE"
)
FROZEN_INVENTORY_STATUS = (
    "QUESTION_BLIND_EVENT_INVENTORY_FROZEN_BEFORE_TASK_QUERY_OR_OUTCOME"
)
CONSUMED_ANCHOR_STATUS = (
    "CONSUMED_DEVELOPMENT_ANCHOR_PILOT_NOT_TRANSFER_EVIDENCE"
)
FROZEN_ANCHOR_STATUS = (
    "ANSWER_BLIND_ANCHOR_LOCALIZATIONS_FROZEN_BEFORE_TARGET_OUTCOME"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_model_contract(acquisition: dict) -> dict:
    """Canonical endpoint contract shared by collectors and the compiler."""

    return {
        "id": str(acquisition["model"]),
        "omit_temperature": False,
        "seed": int(acquisition.get("seed", 0)),
        "provider": {
            "only": [str(acquisition["provider"])],
            "allow_fallbacks": bool(
                acquisition.get("provider_allow_fallbacks", False)
            ),
        },
    }


def _combined_provider_success(
    inventory_rows: list[dict], anchor_rows: list[dict],
) -> tuple[int, int, float]:
    """Count successful provider opportunities across both visual stages.

    Clips without a visible person/object pair are deterministic abstentions,
    not provider failures.  Tasks with no temporal anchor make no provider call
    and therefore do not enter the denominator.
    """

    event_clips = [clip for row in inventory_rows for clip in row["clips"]]
    anchor_calls = [
        row for row in anchor_rows if row.get("call_receipt") is not None
    ]
    successful = sum(
        clip.get("provider_error") in {None, "NO_VISIBLE_PERSON_OBJECT_PAIR"}
        for clip in event_clips
    ) + sum(row.get("provider_error") is None for row in anchor_calls)
    attempted = len(event_clips) + len(anchor_calls)
    return successful, attempted, successful / attempted if attempted else 1.0


def _event(value: dict) -> QuestionBlindTypedEvent:
    return QuestionBlindTypedEvent(
        event_id=str(value["event_id"]),
        predicate=str(value["predicate"]),
        subject_track_id=str(value["subject_track_id"]),
        object_track_id=str(value["object_track_id"]),
        object_role=str(value["object_role"]),
        start_frame=int(value["start_frame"]),
        end_frame=int(value["end_frame"]),
        evidence_frames=tuple(int(item) for item in value["evidence_frames"]),
        confidence=float(value["confidence"]),
        source_clip_ids=tuple(str(item) for item in value["source_clip_ids"]),
    )


def _expected_statuses(consumed_development_pilot: bool) -> tuple[str, str, str]:
    if consumed_development_pilot:
        return (
            CONSUMED_INVENTORY_STATUS,
            CONSUMED_ANCHOR_STATUS,
            "CONSUMED_DEVELOPMENT_QUERY_GROUNDING_V2_NOT_TRANSFER_EVIDENCE",
        )
    return (
        FROZEN_INVENTORY_STATUS,
        FROZEN_ANCHOR_STATUS,
        "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME",
    )


def _validate_anchor_row(anchor: dict, plan: dict, *, video_id: str) -> None:
    if str(anchor["video_id"]) != video_id:
        raise ValueError("anchor row and query plan reference different videos")
    expected = [
        {"anchor_id": f"A{index}", "phrase": str(item["phrase"])}
        for index, item in enumerate(plan.get("action_obligations", ()))
    ]
    if anchor.get("anchor_specs") != expected:
        raise ValueError("anchor phrases differ from frozen query-plan obligations")
    localized = anchor.get("anchor_localizations")
    if not isinstance(localized, list) or len(localized) != len(expected):
        raise ValueError("anchor localization cardinality differs from obligations")
    if [str(item.get("anchor_id")) for item in localized] != [
        item["anchor_id"] for item in expected
    ]:
        raise ValueError("anchor localization order or IDs differ")
    compiled = []
    for item in localized:
        evidence = sorted(set(int(value) for value in item["evidence_frame_ids"]))
        if item["status"] == "SUPPORTED" and evidence:
            compiled.append([min(evidence), max(evidence)])
        elif item["status"] != "UNKNOWN" or evidence:
            raise ValueError("anchor localization violates fail-closed semantics")
    if compiled != anchor.get("anchor_intervals"):
        raise ValueError("compiled anchor intervals do not match pixel evidence")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--query-plans", type=Path, required=True)
    parser.add_argument("--event-inventory", type=Path, required=True)
    parser.add_argument("--anchor-localizations", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runtime-amendment", type=Path)
    parser.add_argument("--minimum-object-score", type=float, default=0.05)
    parser.add_argument("--consumed-development-pilot", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V2 temporal query grounding is immutable")

    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    sgdet = json.loads(args.sgdet.read_text())
    plans = json.loads(args.query_plans.read_text())
    inventory = json.loads(args.event_inventory.read_text())
    anchors = json.loads(args.anchor_localizations.read_text())
    protocol = json.loads(args.protocol.read_text())
    amendment = (
        json.loads(args.runtime_amendment.read_text())
        if args.runtime_amendment is not None else None
    )
    expected_inventory, expected_anchor, output_status = _expected_statuses(
        args.consumed_development_pilot
    )
    if inventory.get("status") != expected_inventory:
        raise ValueError("event inventory status disagrees with evaluation phase")
    if anchors.get("status") != expected_anchor:
        raise ValueError("anchor-localization status disagrees with evaluation phase")
    if bool(inventory.get("consumed_development_pilot")) != bool(
        args.consumed_development_pilot
    ) or bool(anchors.get("consumed_development_pilot")) != bool(
        args.consumed_development_pilot
    ):
        raise ValueError("consumed-development labels disagree")

    forbidden = (
        "answer_read", "official_scene_graph_read", "official_stsg_read",
        "functional_program_read", "source_controller_read", "target_outcome_read",
        "per_video_action_genome_annotation_read",
    )
    if any(inventory.get(key) for key in forbidden) or any(
        sgdet.get(key) for key in forbidden
    ) or any((anchors.get("authority") or {}).get(key) for key in forbidden):
        raise ValueError("grounder input crossed its authority boundary")
    if (anchors.get("authority") or {}).get("question_text_supplied_to_vlm"):
        raise ValueError("anchor localizer received task question text")
    if (anchors.get("authority") or {}).get("root_query_predicate_supplied_to_vlm"):
        raise ValueError("anchor localizer received root query predicate")
    if (anchors.get("authority") or {}).get("temporal_operator_supplied_to_vlm"):
        raise ValueError("anchor localizer received temporal operator")

    acquisition = protocol["question_blind_event_acquisition"]
    anchor_acquisition = protocol["answer_blind_anchor_localization"]
    if args.minimum_object_score != float(acquisition["minimum_object_score"]):
        raise ValueError("stable tracking threshold differs from protocol")
    if inventory.get("frames_per_clip") != int(acquisition["frames_per_clip"]):
        raise ValueError("event acquisition granularity differs from protocol")
    if inventory.get("model") != _frozen_model_contract(acquisition):
        raise ValueError("event model/provider contract differs from protocol")
    if anchors.get("model") != _frozen_model_contract(anchor_acquisition):
        raise ValueError("anchor model/provider contract differs from protocol")
    if int(anchors.get("maximum_anchor_frames", -1)) != int(
        anchor_acquisition["maximum_anchor_frames"]
    ):
        raise ValueError("anchor frame budget differs from protocol")
    if float(anchors.get("minimum_object_score", -1.0)) != float(
        anchor_acquisition["minimum_object_score"]
    ):
        raise ValueError("anchor tracking threshold differs from protocol")
    if inventory.get("protocol_file_sha256") != _sha256(args.protocol):
        raise ValueError("event inventory is not bound to this protocol")
    if anchors.get("protocol_file_sha256") != _sha256(args.protocol):
        raise ValueError("anchor localizations are not bound to this protocol")
    expected_anchor_inputs = {
        "cohort": _sha256(args.cohort),
        "sgdet": _sha256(args.sgdet),
        "query_plans": _sha256(args.query_plans),
    }
    if anchors.get("input_file_sha256s") != expected_anchor_inputs:
        raise ValueError("anchor input hashes differ from compiler inputs")
    immutable = protocol["immutable_inputs"]
    if inventory.get("collector_file_sha256") != immutable.get(
        "event_collector_sha256"
    ):
        raise ValueError("event inventory collector hash differs from protocol")
    frozen_anchor_collector_sha = immutable.get("anchor_collector_sha256")
    actual_anchor_collector_sha = anchors.get("collector_file_sha256")
    if actual_anchor_collector_sha != frozen_anchor_collector_sha:
        if amendment is None or any((
            amendment.get("replaced_anchor_collector_sha256")
            != frozen_anchor_collector_sha,
            amendment.get("replacement_anchor_collector_sha256")
            != actual_anchor_collector_sha,
            amendment.get("anchor_scope")
            != "CANONICALIZE_UNKNOWN_ANCHOR_TO_EMPTY_EVIDENCE",
        )):
            raise ValueError("anchor collector hash differs without amendment")
        if anchors.get("runtime_amendment_file_sha256") != _sha256(
            args.runtime_amendment
        ):
            raise ValueError("anchor artifact is not bound to its amendment")
    if _sha256(
        Path(__file__).resolve().parents[1]
        / "src/motif_transfer/agqa_question_blind_event_grounder.py"
    ) != immutable.get("event_grounder_module_sha256"):
        raise ValueError("event-grounder module hash differs from protocol")
    current_compiler_sha = _sha256(Path(__file__))
    frozen_compiler_sha = immutable.get("query_compiler_sha256")
    if current_compiler_sha != frozen_compiler_sha:
        if amendment is None:
            raise ValueError("compiler hash differs without a pre-outcome amendment")
        expected_amendment = {
            "parent_acquisition_protocol_file_sha256": _sha256(args.protocol),
            "replaced_query_compiler_sha256": frozen_compiler_sha,
            "replacement_query_compiler_sha256": current_compiler_sha,
            "scope": "COUNT_ANCHOR_AND_EVENT_PROVIDER_SUCCESS_IN_ONE_DENOMINATOR",
            "development_outcomes_opened_before_amendment": False,
            "target_outcomes_read_before_amendment": False,
        }
        if any(amendment.get(key) != value for key, value in expected_amendment.items()):
            raise ValueError("runtime amendment does not authorize this compiler")
    temporal_uncertainty_frames = int(acquisition["frames_per_clip"]) // 2

    public = {str(row["task_id"]): row for row in cohort["rows"]}
    semantics = {
        str(row["task_id"]): _semantic(row["receipt"]) for row in runtime["rows"]
    }
    plan_by_task = {str(row["task_id"]): row for row in plans["rows"]}
    anchor_by_task = {str(row["task_id"]): row for row in anchors["rows"]}
    raw_by_video = {str(row["video_id"]): row for row in sgdet["rows"]}
    inventory_by_video = {
        str(row["video_id"]): row for row in inventory["rows"]
    }
    task_ids = set(public)
    if task_ids != set(semantics) or task_ids != set(plan_by_task):
        raise ValueError("cohort, semantic runtime, and query-plan task sets differ")
    if task_ids != set(anchor_by_task):
        raise ValueError("anchor localizations do not cover every task exactly once")
    if {str(row["video_id"]) for row in public.values()} - set(inventory_by_video):
        raise ValueError("event inventory does not cover every query video")

    inputs = {
        "cohort_file_sha256": _sha256(args.cohort),
        "semantic_runtime_file_sha256": _sha256(args.semantic_runtime),
        "sgdet_file_sha256": _sha256(args.sgdet),
        "query_plans_file_sha256": _sha256(args.query_plans),
        "event_inventory_file_sha256": _sha256(args.event_inventory),
        "anchor_localizations_file_sha256": _sha256(args.anchor_localizations),
        "protocol_file_sha256": _sha256(args.protocol),
    }
    backend_sha = stable_hash({
        "protocol": "AGQA_QUESTION_BLIND_EVENT_QUERY_GROUNDER_V2",
        "inputs": inputs,
        "temporal_uncertainty_frames": temporal_uncertainty_frames,
        "temporal_rule": (
            "strict direction for BEFORE/AFTER; acquisition-radius interval "
            "overlap for WHILE/BETWEEN; nearest valid directional event"
        ),
        "candidate_rule": "top stable track or fail-closed abstention",
    })
    positions = {
        str(row["task_id"]): index for index, row in enumerate(cohort["rows"])
    }
    outputs = []
    for task_id in sorted(task_ids, key=positions.__getitem__):
        public_row = public[task_id]
        video_id = str(public_row["video_id"])
        raw = raw_by_video[video_id]
        video_inventory = inventory_by_video[video_id]
        plan = plan_by_task[task_id]
        anchor = anchor_by_task[task_id]
        _validate_anchor_row(anchor, plan, video_id=video_id)
        stable = build_stable_tracks(
            raw, minimum_object_score=args.minimum_object_score,
        )
        tracks = {row.track_id: row for row in stable.tracks}
        events = tuple(_event(row) for row in video_inventory["events"])
        allowed = frozenset(
            int(row["frame_id"]) for row in video_inventory["presented_frames"]
        )
        known = frozenset(tracks)
        track_visible_frames = {
            track_id: frozenset(track.evidence_frames)
            for track_id, track in tracks.items()
        }
        for event in events:
            event.validate(
                known_track_ids=known,
                allowed_frame_ids=allowed,
                track_visible_frames=track_visible_frames,
            )
        semantic = semantics[task_id]
        requested_role = requested_query_role(semantic)
        slot_ids = requested_query_slot_ids(semantic)
        predicate = str(plan.get("predicate") or "").strip()
        temporal_operator = str(plan.get("temporal_operator") or "").strip().upper()
        anchor_intervals = anchor["anchor_intervals"]
        ranking = (() if not predicate else query_temporal_event_candidates(
            events,
            predicate=predicate,
            requested_role=requested_role,
            temporal_operator=temporal_operator,
            anchor_intervals=anchor_intervals,
            temporal_uncertainty_frames=temporal_uncertainty_frames,
        ))
        top = ranking[0] if ranking else None
        candidates = () if top is None else (QueryCandidateEvidence(
            track_id=str(top["track_id"]),
            requested_role=requested_role,
            status="SUPPORTED",
            confidence=float(top["score"]),
            evidence_frames=tuple(int(item) for item in top["evidence_frames"]),
        ),)
        matching_ids = {
            event_id for row in ranking for event_id in row["event_ids"]
        }
        bound = [
            bind_event_to_semantic_slots(
                event, event_id=f"R{index}", semantic_slot_ids=slot_ids,
            )
            for index, event in enumerate(
                event for event in events if event.event_id in matching_ids
            )
        ]
        bound = list(deduplicate_typed_events(bound))
        event_calls = sum(
            int((clip.get("usage") or {}).get("provider_attempts", 0))
            * int(not clip.get("cache_reused"))
            for clip in video_inventory["clips"]
        )
        anchor_calls = int(
            ((anchor.get("call_receipt") or {}).get("usage") or {}).get(
                "provider_attempts", 0
            )
        ) * int(not (anchor.get("call_receipt") or {}).get("cache_reused"))
        receipt = QueryGroundingV2Receipt.create(
            task_id=task_id,
            video_sha256=str(raw["video_sha256"]),
            semantic_slots_sha256=semantic.receipt_sha256,
            selected_frame_indices=tuple(
                int(item) for item in raw["sampled_original_frame_indices"]
            ),
            selected_frame_sha256s=tuple(
                str(item) for item in raw["selected_frame_sha256s"]
            ),
            tracks=stable.tracks,
            events=bound,
            candidates=candidates,
            public_ontology_sha256=str(sgdet["ontology_sha256"]),
            grounder_backend_sha256=backend_sha,
            provider_calls=event_calls + anchor_calls,
        )
        errors = [
            str(clip["provider_error"]) for clip in video_inventory["clips"]
            if clip.get("provider_error") not in {
                None, "NO_VISIBLE_PERSON_OBJECT_PAIR",
            }
        ]
        if anchor.get("provider_error"):
            errors.append(str(anchor["provider_error"]))
        outputs.append({
            "cohort_position": positions[task_id],
            "task_id": task_id,
            "video_id": video_id,
            "requested_role": requested_role,
            "receipt": asdict(receipt),
            "provider_error": ";".join(errors) if errors else None,
            "root_predicate": predicate,
            "temporal_operator": temporal_operator,
            "anchor_intervals": anchor_intervals,
            "anchor_fail_closed": temporal_operator != "VIDEO" and not ranking,
            "temporal_uncertainty_frames": temporal_uncertainty_frames,
            "root_semantic_slot_ids": list(slot_ids),
            "candidate_ranking": [{
                **row,
                "candidate_label": tracks[str(row["track_id"])].canonical_label,
            } for row in ranking],
            "candidate_confidence": float(top["score"]) if top else 0.0,
            "candidate_support_threshold": 0.0,
            "question_blind_event_ids": sorted(matching_ids),
            "tracking_summary": {
                "stable_tracks": len(stable.tracks),
                "retained_detections": len(stable.retained_detection_indices),
            },
        })

    provider_successes, provider_opportunities, provider_success_fraction = (
        _combined_provider_success(inventory["rows"], anchors["rows"])
    )
    report = {
        "schema_version": "agqa-question-blind-event-query-grounder-v2",
        "status": output_status,
        "consumed_development_pilot": bool(args.consumed_development_pilot),
        "cohort_sha256": cohort["cohort_sha256"],
        "semantic_runtime_sha256": runtime["runtime_sha256"],
        "public_ontology_sha256": sgdet["ontology_sha256"],
        "grounder_backend_sha256": backend_sha,
        "event_model": inventory["model"],
        "anchor_model": anchors["model"],
        "frame_budget": int(sgdet["maximum_model_visible_frame_budget"]),
        "question_blind_vlm_unique_frame_budget": int(
            inventory["maximum_unique_vlm_frames_per_video"]
        ),
        "maximum_anchor_frames_per_task": int(anchors["maximum_anchor_frames"]),
        "temporal_uncertainty_frames": temporal_uncertainty_frames,
        "rows": outputs,
        "reported_receipt_provider_cost_usd": (
            float(inventory["reported_cost_usd"])
            + float(anchors["reported_cost_usd"])
        ),
        "provider_and_contract_success_fraction": provider_success_fraction,
        "provider_and_contract_success_count": provider_successes,
        "provider_opportunity_count": provider_opportunities,
        "event_clip_contract_success_fraction": float(
            inventory["clip_contract_success_fraction"]
        ),
        "anchor_provider_error_task_count": sum(
            row.get("call_receipt") is not None
            and row.get("provider_error") is not None
            for row in anchors["rows"]
        ),
        "all_harness_arms_share_exact_receipts": True,
        "stable_entity_tracking": True,
        "typed_semantic_roles": True,
        "cross_frame_typed_event_deduplication": True,
        "answer_blind_query_candidate_verification": False,
        "candidate_verification_status": "PENDING_INDEPENDENT_SINGLE_CANDIDATE_PIXEL_CHECK",
        "question_blind_event_acquisition": True,
        "question_read_during_event_acquisition": False,
        "question_text_supplied_to_anchor_vlm": False,
        "answer_read": False,
        "official_scene_graph_read": False,
        "functional_program_read": False,
        "source_controller_read": False,
        "target_outcome_read": False,
        "per_video_action_genome_annotation_read": False,
        "inputs": inputs,
        "runtime_amendment_file_sha256": (
            _sha256(args.runtime_amendment)
            if args.runtime_amendment is not None else None
        ),
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "tasks": len(outputs),
        "supported": sum(bool(row["receipt"]["candidates"]) for row in outputs),
        "anchor_fail_closed": sum(row["anchor_fail_closed"] for row in outputs),
        "provider_cost_usd": report["reported_receipt_provider_cost_usd"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
