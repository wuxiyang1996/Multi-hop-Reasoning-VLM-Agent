#!/usr/bin/env python3
"""Adapt all V15 typed events to Layer B without an entity-answer candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_contracts import GroundedEvent, LayerBTaskStateReceipt, RawVideoEventGraphReceipt
from motif_transfer.agqa_query_grounder_v2 import query_grounding_v2_from_dict
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _semantic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--query-grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V15 Layer-B adapter output is immutable")
    cohort = json.loads(args.cohort.read_text())
    runtime = json.loads(args.semantic_runtime.read_text())
    query = json.loads(args.query_grounding.read_text())
    if query.get("status") != "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME" or query.get("answer_candidate_selected"):
        raise ValueError("V15 query grounding is not frozen candidate-free compositional evidence")
    if len({cohort["cohort_sha256"], runtime["cohort_sha256"], query["cohort_sha256"]}) != 1:
        raise ValueError("V15 cohort identity mismatch")
    semantic_by_task = {str(row["task_id"]): _semantic(row["receipt"]) for row in runtime["rows"]}
    public_by_task = {str(row["task_id"]): row for row in cohort["rows"]}
    outputs = []
    for row in query["rows"]:
        task_id = str(row["task_id"])
        semantic = semantic_by_task[task_id]
        receipt = query_grounding_v2_from_dict(row["receipt"])
        tracks = {track.track_id: track for track in receipt.tracks}
        events = []
        for typed in receipt.events:
            roles = typed.role_map
            object_id = next((roles.get(name) for name in (
                "patient", "theme", "relation_object", "destination",
                "instrument", "relation_subject", "agent",
            ) if roles.get(name)), None)
            if object_id not in tracks:
                continue
            agent = tracks.get(roles.get("agent", ""))
            events.append(GroundedEvent(
                event_id=f"E{len(events)}", subject=agent.canonical_label if agent else "person",
                predicate=typed.predicate, object=tracks[object_id].canonical_label,
                start_frame=typed.start_frame, end_frame=typed.end_frame,
                evidence_frames=typed.evidence_frames, confidence=typed.confidence,
                semantic_slot_ids=typed.semantic_slot_ids,
            ))
        grounding = RawVideoEventGraphReceipt.create(
            task_id=task_id, video_sha256=receipt.video_sha256,
            semantic_slots_sha256=receipt.semantic_slots_sha256,
            selected_frame_indices=receipt.selected_frame_indices,
            selected_frame_sha256s=receipt.selected_frame_sha256s,
            events=events, grounder_backend_sha256=stable_hash({
                "adapter": "AGQA_COMPOSITIONAL_ALL_TYPED_EVENTS_TO_LAYER_B_V15",
                "query_receipt_sha256": receipt.receipt_sha256,
            }), frame_budget=len(receipt.selected_frame_indices), provider_calls=0,
        )
        state = LayerBTaskStateReceipt.create(semantic, grounding)
        outputs.append({
            "cohort_position": int(row["cohort_position"]), "task_id": task_id,
            "video_id": str(public_by_task[task_id]["video_id"]),
            "semantic_receipt": asdict(semantic), "grounding_receipt": asdict(grounding),
            "task_state_receipt": asdict(state),
            "query_grounding_v2_receipt_sha256": receipt.receipt_sha256,
        })
    outputs.sort(key=lambda row: row["cohort_position"])
    if [row["task_id"] for row in outputs] != [str(row["task_id"]) for row in cohort["rows"]]:
        raise ValueError("V15 adapter order/coverage mismatch")
    body = {
        "schema_version": "agqa-offtheshelf-compositional-layer-b-adapter-v15",
        "status": "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES",
        "cohort_sha256": cohort["cohort_sha256"],
        "semantic_runtime_sha256": runtime["runtime_sha256"],
        "query_grounding_report_sha256": query["report_sha256"],
        "rows": outputs, "all_harness_arms_share_exact_receipts": True,
        "answer_candidate_required": False, "provider_calls": 0,
        "answer_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read": False,
        "target_outcome_read": False,
        "inputs": {
            "cohort_sha256": _sha256(args.cohort),
            "semantic_runtime_sha256": _sha256(args.semantic_runtime),
            "query_grounding_sha256": _sha256(args.query_grounding),
        },
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "rows": len(outputs),
        "two_or_more_events": sum(len(row["grounding_receipt"]["events"]) >= 2 for row in outputs),
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
