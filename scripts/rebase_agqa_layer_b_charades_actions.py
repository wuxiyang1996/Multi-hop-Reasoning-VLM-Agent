#!/usr/bin/env python3
"""Reuse a frozen SlowFast presentation/probe with a new uniform-48 VLM receipt.

The dense action pixels and scores are independent of the VLM backend.  This
utility avoids decoding the same 320 native frames again: it verifies every
uniform proxy-frame hash, preserves the prior presented-frame timeline, and
replaces only the VLM events.  No question outcome is read.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_contracts import (
    GroundedEvent, LayerBTaskStateReceipt, RawVideoEventGraphReceipt,
)
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _grounding, _semantic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--base-grounding", type=Path, required=True)
    parser.add_argument("--frozen-composite", type=Path, required=True)
    parser.add_argument(
        "--slowfast-policy", choices=("all", "fill_missing"), default="all",
        help="With fill_missing, coarse SlowFast windows cannot overwrite a VLM-bound action interval.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("rebased grounding output is immutable")

    cohort = json.loads(args.cohort.read_text())
    base = json.loads(args.base_grounding.read_text())
    frozen = json.loads(args.frozen_composite.read_text())
    if len({cohort["cohort_sha256"], base["cohort_sha256"], frozen["cohort_sha256"]}) != 1:
        raise ValueError("cohort mismatch")
    if not base.get("rows"):
        raise ValueError("base grounding has no rows")
    proxy_frame_count = len(base["rows"][0]["grounding_receipt"]["selected_frame_sha256s"])
    if proxy_frame_count not in {48, 96}:
        raise ValueError("rebase requires a frozen uniform-48 or uniform-96 proxy protocol")
    if frozen.get("schema_version") != "agqa-layer-b-shared-composite-grounding-v1":
        raise ValueError("reference is not a frozen Layer-B composite")
    action_frame_budget = int(frozen["frame_budget"]) - proxy_frame_count
    if action_frame_budget != 320:
        raise ValueError("reference does not contain the frozen dense10x32 action budget")
    combined_frame_budget = int(base["frame_budget"]) + action_frame_budget
    frozen_by_task = {str(row["task_id"]): row for row in frozen["rows"]}

    rows = []
    backend = stable_hash({
        "merge": "VERIFIED_UNIFORM_REBASE_PLUS_FROZEN_SLOWFAST_DENSE10X32_V1",
        "proxy_frame_count": proxy_frame_count,
        "base_grounder_backend_sha256": base["grounder_backend_sha256"],
        "frozen_composite_sha256": frozen["report_sha256"],
        "action_probe_sha256": frozen["action_probe_sha256"],
        "action_threshold": frozen["action_threshold"],
        "slowfast_policy": args.slowfast_policy,
        "frame_presentation_budget": combined_frame_budget,
    })
    for raw in base["rows"]:
        task_id = str(raw["task_id"])
        old = _grounding(raw["grounding_receipt"])
        semantic = _semantic(raw["semantic_receipt"])
        reference = frozen_by_task[task_id]
        timeline = reference["presented_frame_timeline"]
        proxy_positions = {
            int(item["source_index"]): index
            for index, item in enumerate(timeline)
            if item["source"] == "VLM_UNIFORM48"
        }
        if set(proxy_positions) != set(range(proxy_frame_count)):
            raise ValueError(f"{task_id}: reference proxy timeline does not match base frame count")
        for proxy_index, digest in enumerate(old.selected_frame_sha256s):
            if timeline[proxy_positions[proxy_index]]["sha256"] != digest:
                raise ValueError(f"{task_id}: proxy-frame hash mismatch at {proxy_index}")

        events: list[GroundedEvent] = []
        for event in old.events:
            events.append(GroundedEvent(
                event_id=f"E{len(events)}", subject=event.subject,
                predicate=event.predicate, object=event.object,
                start_frame=proxy_positions[event.start_frame],
                end_frame=proxy_positions[event.end_frame],
                evidence_frames=tuple(proxy_positions[index] for index in event.evidence_frames),
                confidence=event.confidence, semantic_slot_ids=event.semantic_slot_ids,
            ))

        accepted_action_count = sum(
            len(item.get("accepted_windows", ()))
            for item in reference["action_model_receipts"]
        )
        reference_events = _grounding(reference["grounding_receipt"]).events
        action_events = reference_events[-accepted_action_count:] if accepted_action_count else ()
        action_cursor = 0
        action_receipts = []
        for obligation in reference["action_model_receipts"]:
            window_count = len(obligation.get("accepted_windows", ()))
            obligation_events = action_events[action_cursor:action_cursor + window_count]
            action_cursor += window_count
            slot_id = str(obligation.get("slot_id", ""))
            vlm_covered = any(slot_id and slot_id in event.semantic_slot_ids for event in old.events)
            accepted = obligation_events
            status = "ALL_ACCEPTED_WINDOWS_MERGED"
            if args.slowfast_policy == "fill_missing" and vlm_covered:
                accepted = ()
                status = "SKIPPED_COARSE_WINDOWS_VLM_ACTION_ALREADY_BOUND"
            for event in accepted:
                events.append(GroundedEvent(
                    event_id=f"E{len(events)}", subject=event.subject,
                    predicate=event.predicate, object=event.object,
                    start_frame=event.start_frame, end_frame=event.end_frame,
                    evidence_frames=event.evidence_frames, confidence=event.confidence,
                    semantic_slot_ids=event.semantic_slot_ids,
                ))
            action_receipts.append({
                **obligation, "composite_merge_policy": args.slowfast_policy,
                "composite_merge_status": status,
                "merged_accepted_windows": (
                    list(obligation.get("accepted_windows", ())) if accepted else []
                ),
            })
        if action_cursor != len(action_events):
            raise ValueError(f"{task_id}: action receipt/event alignment failed")

        receipt = RawVideoEventGraphReceipt.create(
            task_id=task_id, video_sha256=old.video_sha256,
            semantic_slots_sha256=semantic.receipt_sha256,
            selected_frame_indices=tuple(range(len(timeline))),
            selected_frame_sha256s=tuple(item["sha256"] for item in timeline),
            events=events, grounder_backend_sha256=backend,
            frame_budget=combined_frame_budget, provider_calls=old.provider_calls,
        )
        state = LayerBTaskStateReceipt.create(semantic, receipt)
        row = dict(raw)
        row.update(
            grounding_receipt=asdict(receipt), task_state_receipt=asdict(state),
            action_model_receipts=action_receipts,
            presented_frame_timeline=timeline,
            frozen_action_reuse_receipt_sha256=stable_hash({
                "task_id": task_id,
                "reference_grounding_receipt_sha256": reference["grounding_receipt"]["receipt_sha256"],
                "action_model_receipts": action_receipts,
                "presented_frame_timeline": timeline,
            }),
        )
        rows.append(row)

    body = {key: value for key, value in base.items()
            if key not in {"rows", "report_sha256", "grounder_backend_sha256", "frame_budget"}}
    body.update({
        "schema_version": "agqa-layer-b-shared-composite-grounding-v1",
        "status": "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES",
        "rows": rows, "grounder_backend_sha256": backend,
        "frame_budget": combined_frame_budget,
        "action_threshold": frozen["action_threshold"],
        "slowfast_policy": args.slowfast_policy,
        "action_probe_sha256": frozen["action_probe_sha256"],
        "frozen_action_reference_sha256": frozen["report_sha256"],
        "answers_read": False, "official_program_read": False,
        "official_scene_graph_read": False,
    })
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "report_sha256": body["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
