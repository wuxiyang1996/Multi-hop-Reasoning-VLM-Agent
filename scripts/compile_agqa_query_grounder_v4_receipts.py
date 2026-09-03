#!/usr/bin/env python3
"""Compile answer-blind V4 candidate-ID decisions into typed V2 receipts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_query_grounder_v2 import (
    QueryCandidateEvidence, QueryGroundingV2Receipt, TypedRoleEvent,
    deduplicate_typed_events, query_grounding_v2_from_dict,
)
from motif_transfer.contracts import stable_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _shared_positions(video: dict, receipt: QueryGroundingV2Receipt,
                      sampled_frame_ids: list[int]) -> tuple[int, ...]:
    native = [int(value) for value in video["sampled_original_frame_indices"]]
    shared = {int(value): index for index, value in enumerate(receipt.selected_frame_indices)}
    output = []
    for frame_id in sampled_frame_ids:
        if not 0 <= int(frame_id) < len(native):
            raise ValueError("V4 evidence frame exceeds SGDET sampling")
        native_frame = native[int(frame_id)]
        if native_frame not in shared:
            raise ValueError("V4 presented frame is absent from shared frame union")
        output.append(shared[native_frame])
    return tuple(sorted(set(output)))


def adjudicated_receipt(
    *, base: QueryGroundingV2Receipt, base_row: dict, adjudicated: dict,
    raw_video: dict, support_threshold: float, backend_sha256: str,
) -> QueryGroundingV2Receipt:
    """Replace only the answer-bearing root binding; preserve anchor events."""
    root_slots = frozenset(str(value) for value in base_row["root_semantic_slot_ids"])
    preserved = [
        event for event in base.events
        if not root_slots.intersection(event.semantic_slot_ids)
    ]
    selected = adjudicated.get("selected_candidate")
    candidates = []
    if selected is not None:
        track_id = str(selected["track_id"])
        if track_id not in {track.track_id for track in base.tracks}:
            raise ValueError("V4 selected candidate references an unknown stable track")
        evidence = _shared_positions(
            raw_video, base, [int(value) for value in adjudicated["evidence_frame_ids"]],
        )
        if not evidence:
            raise ValueError("V4 selected candidate lacks pixel evidence")
        confidence = float(adjudicated["confidence"])
        status = "SUPPORTED" if confidence >= support_threshold else "UNKNOWN"
        candidates.append(QueryCandidateEvidence(
            track_id=track_id, requested_role=str(base_row["requested_role"]),
            status=status, confidence=confidence,
            evidence_frames=evidence if status == "SUPPORTED" else (),
        ))
        preserved.append(TypedRoleEvent(
            event_id=f"R{len(preserved)}", predicate=str(base_row["root_predicate"]),
            roles=(("agent", "T0"), (str(base_row["requested_role"]), track_id)),
            start_frame=min(evidence), end_frame=max(evidence),
            evidence_frames=evidence, confidence=confidence,
            semantic_slot_ids=tuple(str(value) for value in base_row["root_semantic_slot_ids"]),
        ))
    events = deduplicate_typed_events(preserved)
    return QueryGroundingV2Receipt.create(
        task_id=base.task_id, video_sha256=base.video_sha256,
        semantic_slots_sha256=base.semantic_slots_sha256,
        selected_frame_indices=base.selected_frame_indices,
        selected_frame_sha256s=base.selected_frame_sha256s,
        tracks=base.tracks, events=events, candidates=candidates,
        public_ontology_sha256=base.public_ontology_sha256,
        grounder_backend_sha256=backend_sha256,
        provider_calls=base.provider_calls + int(adjudicated.get("usage") is not None),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-grounding", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--sgdet-raw", type=Path, required=True)
    parser.add_argument("--support-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("compiled V4 grounding is immutable")
    if not 0 <= args.support_threshold <= 1:
        raise ValueError("support threshold must be in [0,1]")
    base = json.loads(args.base_grounding.read_text())
    adjudication = json.loads(args.adjudication.read_text())
    raw = json.loads(args.sgdet_raw.read_text())
    if base.get("status") != "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME":
        raise ValueError("base candidate grounding is not frozen")
    if adjudication.get("status") != "V4_ADJUDICATION_FROZEN_BEFORE_DEVELOPMENT_OUTCOME":
        raise ValueError("V4 adjudication is not frozen before outcomes")
    if base["report_sha256"] != adjudication["candidate_grounding_report_sha256"]:
        raise ValueError("V4 adjudication and base grounding differ")
    forbidden = (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )
    if any(base.get(key) for key in forbidden) or any(adjudication.get(key) for key in forbidden):
        raise ValueError("V4 compilation input violates authority boundary")
    adjudicated_by_id = {str(row["task_id"]): row for row in adjudication["rows"]}
    raw_by_video = {str(row["video_id"]): row for row in raw["rows"]}
    if set(adjudicated_by_id) != {str(row["task_id"]) for row in base["rows"]}:
        raise ValueError("V4 adjudication task set differs from base grounding")
    backend_sha = stable_hash({
        "protocol": "AGQA_QUERY_GROUNDER_V4_QWEN235_TYPED_RECEIPT_V1",
        "base_grounding_report_sha256": base["report_sha256"],
        "adjudication_report_sha256": adjudication["report_sha256"],
        "support_threshold": args.support_threshold,
    })
    outputs = []
    for row in base["rows"]:
        task_id = str(row["task_id"])
        receipt = adjudicated_receipt(
            base=query_grounding_v2_from_dict(row["receipt"]), base_row=row,
            adjudicated=adjudicated_by_id[task_id],
            raw_video=raw_by_video[str(row["video_id"])],
            support_threshold=args.support_threshold, backend_sha256=backend_sha,
        )
        usage = adjudicated_by_id[task_id].get("usage") or {}
        outputs.append({
            "cohort_position": int(row["cohort_position"]), "task_id": task_id,
            "video_id": str(row["video_id"]), "requested_role": row["requested_role"],
            "root_predicate": row["root_predicate"],
            "root_semantic_slot_ids": row["root_semantic_slot_ids"],
            "root_temporal_window": row["root_temporal_window"],
            "receipt": asdict(receipt),
            "provider_error": (
                "SCHEMA_FAILURE" if usage.get("finish_reason") == "schema_failure" else None
            ),
            "candidate_support_threshold": args.support_threshold,
            "v4_adjudication_status": adjudicated_by_id[task_id]["status"],
            "v4_adjudication_confidence": adjudicated_by_id[task_id]["confidence"],
        })
    outputs.sort(key=lambda row: row["cohort_position"])
    body = {
        "schema_version": "agqa-query-grounder-v4-typed-receipts-v1",
        "status": "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME",
        "cohort_sha256": base["cohort_sha256"],
        "rows": outputs, "candidate_support_threshold": args.support_threshold,
        "base_grounding_report_sha256": base["report_sha256"],
        "adjudication_report_sha256": adjudication["report_sha256"],
        "grounder_backend_sha256": backend_sha,
        "provider_calls": sum(row["receipt"]["provider_calls"] for row in outputs),
        "reported_cost_usd": float(adjudication["reported_cost_usd"]),
        "answer_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read": False,
        "target_outcome_read": False,
        "inputs": {
            "base_grounding_file_sha256": _sha256(args.base_grounding),
            "adjudication_file_sha256": _sha256(args.adjudication),
            "sgdet_raw_file_sha256": _sha256(args.sgdet_raw),
        },
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "tasks": len(outputs),
        "supported": sum(
            any(candidate["status"] == "SUPPORTED" for candidate in row["receipt"]["candidates"])
            for row in outputs
        ),
        "provider_calls": body["provider_calls"],
        "reported_cost_usd": body["reported_cost_usd"],
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
