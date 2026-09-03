#!/usr/bin/env python3
"""Merge complete question-blind AGQA event-inventory shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_question_blind_event_grounder import QuestionBlindTypedEvent
from motif_transfer.contracts import stable_hash


def _merged_status(consumed_development_pilot: bool) -> str:
    if consumed_development_pilot:
        return "CONSUMED_DEVELOPMENT_EVENT_INVENTORY_NOT_TRANSFER_EVIDENCE"
    return "QUESTION_BLIND_EVENT_INVENTORY_FROZEN_BEFORE_TASK_QUERY_OR_OUTCOME"


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
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("merged question-blind event inventory is immutable")
    protocol_sha = _sha256(args.protocol)
    shards = [json.loads(path.read_text()) for path in args.shard]
    if not shards:
        raise ValueError("no shards")
    count = int(shards[0]["shard_count"])
    indices = {int(row["shard_index"]) for row in shards}
    if len(shards) != count or indices != set(range(count)):
        raise ValueError("question-blind event inventory shards are incomplete")
    invariant_keys = (
        "schema_version", "status", "model", "clip_count", "frames_per_clip",
        "sampling_strategy",
        "maximum_unique_vlm_frames_per_video", "minimum_object_score",
        "cohort_sha256", "sgdet_report_sha256", "public_ontology_sha256",
        "protocol_file_sha256", "question_read", "answer_read",
        "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
        "per_video_action_genome_annotation_read",
        "consumed_development_pilot", "bbox_coordinate_contract",
        "collector_file_sha256",
    )
    expected = {key: shards[0][key] for key in invariant_keys}
    if expected["protocol_file_sha256"] != protocol_sha:
        raise ValueError("protocol hash differs from shard binding")
    if any({key: shard[key] for key in invariant_keys} != expected for shard in shards):
        raise ValueError("question-blind event inventory shard invariants differ")
    if any(expected[key] for key in (
        "question_read", "answer_read", "official_scene_graph_read",
        "functional_program_read", "source_controller_read", "target_outcome_read",
        "per_video_action_genome_annotation_read",
    )):
        raise ValueError("question-blind event inventory crossed its authority boundary")

    rows = []
    seen = set()
    for shard in sorted(shards, key=lambda row: int(row["shard_index"])):
        for row in shard["rows"]:
            video_id = str(row["video_id"])
            if video_id in seen:
                raise ValueError("duplicate video across event-inventory shards")
            seen.add(video_id)
            known = frozenset(str(track["track_id"]) for track in row["stable_tracks"])
            track_visible_frames = {
                str(track["track_id"]): frozenset(
                    int(frame) for frame in track["evidence_frames"]
                )
                for track in row["stable_tracks"]
            }
            allowed = frozenset(int(frame["frame_id"]) for frame in row["presented_frames"])
            for raw_event in row["events"]:
                _event(raw_event).validate(
                    known_track_ids=known, allowed_frame_ids=allowed,
                    track_visible_frames=track_visible_frames,
                )
            rows.append(row)
    rows.sort(key=lambda row: str(row["video_id"]))
    required_videos = int(
        json.loads(args.protocol.read_text())["development_cohort"]["video_count"]
    )
    if len(rows) != required_videos:
        raise ValueError(f"expected {required_videos} videos, received {len(rows)}")
    report = {
        **{key: value for key, value in expected.items() if key not in {
            "schema_version", "status",
        }},
        "schema_version": "agqa-question-blind-typed-event-inventory-v1",
        "status": _merged_status(bool(expected["consumed_development_pilot"])),
        "consumed_development_pilot": bool(expected["consumed_development_pilot"]),
        "rows": rows,
        "provider_calls": sum(int(row["provider_calls"]) for row in shards),
        "reported_cost_usd": sum(float(row["reported_cost_usd"]) for row in shards),
        "shard_file_sha256s": [_sha256(path) for path in args.shard],
        "shard_report_sha256s": [row["report_sha256"] for row in shards],
        "video_count": len(rows),
        "clip_contract_success_fraction": (
            sum(
                clip.get("provider_error") in {None, "NO_VISIBLE_PERSON_OBJECT_PAIR"}
                for row in rows for clip in row["clips"]
            ) / sum(len(row["clips"]) for row in rows)
        ),
        "event_count_before_deduplication": sum(
            int(row["events_before_deduplication"]) for row in rows
        ),
        "event_count_after_deduplication": sum(len(row["events"]) for row in rows),
        "accepted_event_proposals": sum(
            len(clip["events"]) for row in rows for clip in row["clips"]
        ),
        "rejected_event_proposals": sum(
            len(clip.get("rejected_events", ()))
            for row in rows for clip in row["clips"]
        ),
    }
    proposal_count = (
        report["accepted_event_proposals"] + report["rejected_event_proposals"]
    )
    report["same_frame_track_evidence_valid_fraction"] = (
        report["accepted_event_proposals"] / proposal_count
        if proposal_count else 1.0
    )
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "videos": len(rows),
        "events": report["event_count_after_deduplication"],
        "clip_contract_success_fraction": report["clip_contract_success_fraction"],
        "provider_calls": report["provider_calls"],
        "reported_cost_usd": report["reported_cost_usd"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
