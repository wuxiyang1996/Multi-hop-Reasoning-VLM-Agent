#!/usr/bin/env python3
"""Merge frozen VLM events and frozen Charades action scores on one timeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image

from motif_transfer.agqa_layer_b_contracts import (
    GroundedEvent, LayerBTaskStateReceipt, RawVideoEventGraphReceipt,
)
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_layer_b_five_arm import _grounding, _semantic


def _frame_hash(frame: Image.Image) -> str:
    return stable_hash({
        "mode": frame.mode,
        "size": frame.size,
        "pixels_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
    })


def _decode_native(path: Path, indices: set[int]) -> dict[int, Image.Image]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    wanted = set(indices)
    output = {}
    # Sequential decode is materially faster and more reproducible on the
    # Charades MP4s than hundreds of random ``CAP_PROP_POS_FRAMES`` seeks.
    # It produces the same RGB bytes and therefore preserves receipt hashes.
    for index in range(max(wanted) + 1 if wanted else 0):
        ok, bgr = capture.read()
        if not ok or bgr is None:
            capture.release()
            raise RuntimeError(f"failed decoding {path} at native frame {index}")
        if index in wanted:
            output[index] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if set(output) != wanted:
        raise RuntimeError(f"failed decoding all requested frames from {path}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--base-grounding", type=Path, required=True)
    parser.add_argument("--action-probe", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("merged grounding output is immutable")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")

    cohort = json.loads(args.cohort.read_text())
    public = {str(row["task_id"]): row for row in cohort["rows"]}
    base = json.loads(args.base_grounding.read_text())
    probe = json.loads(args.action_probe.read_text())
    if probe["sampling"] != "dense10x32" or probe["frame_presentation_budget"] != 320:
        raise ValueError("merge requires the frozen dense10x32 action probe")
    probe_by_id = {str(row["task_id"]): row for row in probe["rows"]}
    combined_frame_budget = int(base["frame_budget"]) + int(probe["frame_presentation_budget"])

    rows = []
    backend = stable_hash({
        "merge": "UNIFORM_VLM_PLUS_FROZEN_SLOWFAST_DENSE10X32_V2",
        "base_grounder_backend_sha256": base["grounder_backend_sha256"],
        "action_probe_sha256": probe["report_sha256"],
        "action_threshold": args.threshold,
        "max_frame_presentation_budget": combined_frame_budget,
    })
    for raw in base["rows"]:
        task_id = str(raw["task_id"])
        old = _grounding(raw["grounding_receipt"])
        semantic = _semantic(raw["semantic_receipt"])
        action_row = probe_by_id[task_id]
        metadata = raw["video_metadata"]
        fps = float(metadata["source_fps"])
        proxy_seconds = [float(value) for value in metadata["proxy_sample_seconds"]]
        native_indices = {
            int(index)
            for view in action_row["native_frame_index_views"]
            for index in view
        }
        decoded = _decode_native(Path(public[task_id]["video_path"]), native_indices)

        # Keep every actually presented image, including a VLM proxy and a
        # SlowFast native frame at nearly the same time.  Stable ordering makes
        # temporal execution auditable without pretending the two inputs were
        # the same pixels.
        presented = []
        for proxy_index, (second, digest) in enumerate(zip(proxy_seconds, old.selected_frame_sha256s)):
            presented.append((second, 0, proxy_index, digest, "VLM_UNIFORM48"))
        for native_index in sorted(native_indices):
            presented.append((native_index / fps, 1, native_index, _frame_hash(decoded[native_index]), "SLOWFAST_NATIVE"))
        presented.sort(key=lambda row: (row[0], row[1], row[2]))
        if len(presented) > combined_frame_budget:
            raise ValueError("combined grounder exceeded the preregistered frame-presentation budget")
        proxy_to_position = {
            original: position for position, (_, kind, original, _, _) in enumerate(presented) if kind == 0
        }
        native_to_position = {
            original: position for position, (_, kind, original, _, _) in enumerate(presented) if kind == 1
        }

        events = []
        for event in old.events:
            events.append(GroundedEvent(
                event_id=f"E{len(events)}", subject=event.subject,
                predicate=event.predicate, object=event.object,
                start_frame=proxy_to_position[event.start_frame],
                end_frame=proxy_to_position[event.end_frame],
                evidence_frames=tuple(proxy_to_position[index] for index in event.evidence_frames),
                confidence=event.confidence, semantic_slot_ids=event.semantic_slot_ids,
            ))

        action_receipts = []
        views = action_row["native_frame_index_views"]
        for obligation in action_row["obligations"]:
            if obligation.get("mapping_status") != "EXACT_PUBLIC_ACTION_CLASS":
                action_receipts.append({**obligation, "accepted_windows": []})
                continue
            accepted = [
                index for index, score in enumerate(obligation["window_scores"])
                if float(score) >= args.threshold
            ]
            for view_index in accepted:
                view = tuple(int(value) for value in views[view_index])
                evidence_native = view[len(view) // 2]
                events.append(GroundedEvent(
                    event_id=f"E{len(events)}", subject="person",
                    predicate=str(obligation["phrase"]), object="",
                    start_frame=native_to_position[min(view)],
                    end_frame=native_to_position[max(view)],
                    evidence_frames=(native_to_position[evidence_native],),
                    confidence=float(obligation["window_scores"][view_index]),
                    semantic_slot_ids=(str(obligation["slot_id"]),),
                ))
            action_receipts.append({**obligation, "accepted_windows": accepted})

        receipt = RawVideoEventGraphReceipt.create(
            task_id=task_id, video_sha256=old.video_sha256,
            semantic_slots_sha256=semantic.receipt_sha256,
            selected_frame_indices=tuple(range(len(presented))),
            selected_frame_sha256s=tuple(row[3] for row in presented),
            events=events, grounder_backend_sha256=backend,
            frame_budget=combined_frame_budget, provider_calls=old.provider_calls,
        )
        state = LayerBTaskStateReceipt.create(semantic, receipt)
        row = dict(raw)
        row.update(
            grounding_receipt=asdict(receipt), task_state_receipt=asdict(state),
            action_model_receipts=action_receipts,
            presented_frame_timeline=[
                {"second": second, "source": source, "source_index": original, "sha256": digest}
                for second, _, original, digest, source in presented
            ],
        )
        rows.append(row)
        print(json.dumps({
            "task_id": task_id, "presented_frames": len(presented),
            "events": len(events),
            "accepted_action_windows": sum(len(row["accepted_windows"]) for row in action_receipts),
        }), flush=True)

    body = {
        key: value for key, value in base.items()
        if key not in {"rows", "report_sha256", "grounder_backend_sha256", "frame_budget"}
    }
    body.update({
        "schema_version": "agqa-layer-b-shared-composite-grounding-v1",
        "status": "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES",
        "rows": rows,
        "grounder_backend_sha256": backend,
        "frame_budget": combined_frame_budget,
        "action_threshold": args.threshold,
        "action_probe_sha256": probe["report_sha256"],
        "answers_read": False,
        "official_program_read": False,
        "official_scene_graph_read": False,
    })
    body["report_sha256"] = stable_hash(body)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "report_sha256": body["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
