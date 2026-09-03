#!/usr/bin/env python3
"""Answer-blind binary verification of one event-graph candidate per AGQA task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

from openai import OpenAI
from PIL import ImageDraw

from motif_transfer.agqa_action_genome_grounder import build_stable_tracks
from motif_transfer.agqa_query_grounder_v2 import query_grounding_v2_from_dict
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import (
    _cached_provider_call, _panel_content, _panels,
)
from scripts.collect_agqa_question_blind_typed_event_inventory_v1 import (
    _provider_call_with_contract_retries,
    _request_cache_contract,
)
from scripts.pilot_agqa_query_grounder_v4_qwen235_adjudicator import (
    _exact_sgdet_frames,
)


SYSTEM = """You are a frozen answer-blind video evidence verifier, not a question-answering agent. You receive one typed person-object event hypothesis from a previously frozen question-blind event inventory. Inspect only the chronological raw frames and labeled stable tracks. Decide whether that exact predicate and typed object binding is visibly SUPPORTED, visibly REFUTED, or UNKNOWN. Never select an alternative object, output an object-name answer, infer correctness, execute a symbolic program, use a source controller, or consult an official annotation. Absence from sampled pixels is UNKNOWN, not REFUTED. Return only the required JSON schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _response_format(frame_ids: list[int]) -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_answer_blind_binary_event_verification_v1", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "status": {
                    "type": "string", "enum": ["SUPPORTED", "REFUTED", "UNKNOWN"],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_frame_ids": {
                    "type": "array", "maxItems": 4,
                    "items": {"type": "integer", "enum": frame_ids},
                },
            },
            "required": ["status", "confidence", "evidence_frame_ids"],
        },
    }}


def _validate(
    payload: dict, frame_ids: list[int],
    supportable_frame_ids: set[int] | frozenset[int] | None = None,
) -> dict:
    if set(payload) != {"status", "confidence", "evidence_frame_ids"}:
        raise ValueError("binary verifier payload contains unexpected fields")
    status = str(payload["status"])
    if status not in {"SUPPORTED", "REFUTED", "UNKNOWN"}:
        raise ValueError("binary verifier status is invalid")
    confidence = float(payload["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError("binary verifier confidence is invalid")
    evidence = sorted(set(int(value) for value in payload["evidence_frame_ids"]))
    if len(evidence) > 4 or any(value not in set(frame_ids) for value in evidence):
        raise ValueError("binary verifier cites an unpresented frame")
    if status == "SUPPORTED" and not evidence:
        raise ValueError("supported binary verification needs pixel evidence")
    if (
        status == "SUPPORTED"
        and supportable_frame_ids is not None
        and any(value not in supportable_frame_ids for value in evidence)
    ):
        status, confidence, evidence = "UNKNOWN", 0.0, []
    if status == "UNKNOWN":
        # Inspected frames are not positive evidence.  Canonicalizing UNKNOWN
        # to an empty evidence list is strictly information-decreasing.
        evidence = []
    return {"status": status, "confidence": confidence, "evidence_frame_ids": evidence}


def _selected_frame_ids(row: dict, receipt, candidate, maximum: int) -> list[int]:
    """Show candidate evidence, anchor boundaries, then a fixed video view."""

    lower, upper = 0, len(receipt.selected_frame_indices) - 1
    matching = [
        event for event in receipt.events
        if event.role_map.get(candidate.requested_role) == candidate.track_id
    ]
    priority = [
        frame for event in matching
        for frame in (event.start_frame, *event.evidence_frames, event.end_frame)
        if lower <= frame <= upper
    ]
    priority.extend(
        frame for frame in candidate.evidence_frames if lower <= frame <= upper
    )
    for interval in row.get("anchor_intervals", ()):
        start, end = (int(value) for value in interval)
        priority.extend((start, end))
    if upper > lower:
        priority.extend(
            round(lower + index * (upper - lower) / 7) for index in range(8)
        )
    else:
        priority.append(lower)
    output = []
    for frame in priority:
        value = max(lower, min(upper, int(frame)))
        if value not in output:
            output.append(value)
        if len(output) == maximum:
            break
    return sorted(output)


def _annotate(images, frame_ids, scales, video, candidate_track_id, stable):
    output = [image.copy() for image in images]
    by_frame = {}
    for detected in video["objects"]:
        index = int(detected["detection_index"])
        track_id = stable.detection_to_track.get(index)
        if track_id not in {"T0", candidate_track_id}:
            continue
        by_frame.setdefault(int(detected["sampled_frame_index"]), []).append(
            (str(track_id), detected)
        )
    for image, frame_id in zip(output, frame_ids):
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), f"S{frame_id}", fill="white", stroke_width=3, stroke_fill="black")
        for track_id, detected in by_frame.get(frame_id, ()):
            scale = float(scales[frame_id])
            box = tuple(float(value) / scale for value in detected["bbox_xyxy"])
            color = "lime" if track_id == "T0" else "magenta"
            label = "P0" if track_id == "T0" else "C0"
            draw.rectangle(box, outline=color, width=5)
            draw.text(
                (box[0] + 2, box[1] + 2), label,
                fill=color, stroke_width=2, stroke_fill="black",
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--candidate-grounding", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--provider", default="parasail")
    parser.add_argument("--maximum-frames", type=int, default=12)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--consumed-development-pilot", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("binary candidate verifier shard is immutable")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    cohort = json.loads(args.cohort.read_text())
    sgdet = json.loads(args.sgdet.read_text())
    grounding = json.loads(args.candidate_grounding.read_text())
    protocol = json.loads(args.protocol.read_text())
    expected_grounding_status = (
        "CONSUMED_DEVELOPMENT_QUERY_GROUNDING_V2_NOT_TRANSFER_EVIDENCE"
        if args.consumed_development_pilot
        else "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME"
    )
    if grounding.get("status") != expected_grounding_status:
        raise ValueError("candidate grounding is not frozen before outcome")
    if any(grounding.get(key) for key in (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )):
        raise ValueError("candidate grounding crossed its authority boundary")
    actual = {
        "cohort_sha256": cohort["cohort_sha256"],
        "sgdet_file_sha256": _sha256(args.sgdet),
        "candidate_grounding_file_sha256": _sha256(args.candidate_grounding),
    }
    if actual != {
        key: protocol["immutable_inputs"][key] for key in actual
    }:
        raise ValueError("binary verifier immutable inputs differ")
    verifier = protocol["binary_candidate_verifier"]
    if (
        args.model != verifier["model"]
        or args.provider != verifier["provider"]
        or args.maximum_frames != int(verifier["maximum_frames"])
    ):
        raise ValueError("binary verifier runtime differs from frozen protocol")

    public_video_paths = {
        str(row["video_id"]): str(row["video_path"])
        for row in cohort["video_selections"]
    }
    raw_by_video = {str(row["video_id"]): row for row in sgdet["rows"]}
    sources = [
        row for index, row in enumerate(grounding["rows"])
        if index % args.shard_count == args.shard_index
    ]
    if args.max_tasks is not None:
        sources = sources[:args.max_tasks]
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key unavailable")
    client = OpenAI(
        api_key=key, base_url="https://openrouter.ai/api/v1",
        timeout=300, max_retries=2,
    )
    model = {
        "id": args.model,
        "omit_temperature": False,
        "seed": int(verifier.get("seed", 0)),
        "provider": {
            "only": [args.provider],
            "allow_fallbacks": bool(
                verifier.get("provider_allow_fallbacks", False)
            ),
        },
    }
    outputs = []
    total_cost = 0.0
    provider_calls = 0
    for row in sources:
        task_id = str(row["task_id"])
        video_id = str(row["video_id"])
        receipt = query_grounding_v2_from_dict(row["receipt"])
        if not receipt.candidates:
            outputs.append({
                "task_id": task_id, "video_id": video_id,
                "status": "ABSTAIN_NO_EVENT_CANDIDATE", "confidence": 0.0,
                "evidence_frame_ids": [], "presented_frame_ids": [],
                "presented_frame_sha256s": [], "panel_sha256s": [],
                "usage": None, "cache_reused": True, "provider_error": None,
            })
            continue
        candidate = receipt.candidates[0]
        raw = raw_by_video[video_id]
        stable = build_stable_tracks(raw, minimum_object_score=0.05)
        tracks = {track.track_id: track for track in stable.tracks}
        candidate_track = tracks[candidate.track_id]
        frame_ids = _selected_frame_ids(
            row, receipt, candidate, args.maximum_frames,
        )
        visible_by_track: dict[str, set[int]] = {}
        for detected in raw["objects"]:
            detection_index = int(detected["detection_index"])
            if detection_index not in stable.retained_detection_indices:
                continue
            track_id = stable.detection_to_track.get(detection_index)
            if track_id is not None:
                visible_by_track.setdefault(str(track_id), set()).add(
                    int(detected["sampled_frame_index"])
                )
        supportable_frame_ids = (
            visible_by_track.get("T0", set())
            & visible_by_track.get(candidate.track_id, set())
            & set(frame_ids)
        )
        images, seconds, scales = _exact_sgdet_frames(
            Path(public_video_paths[video_id]), raw, frame_ids,
        )
        annotated = _annotate(
            images, frame_ids, scales, raw, candidate.track_id, stable,
        )
        panels = _panels(
            annotated, seconds, frames_per_panel=2, frame_width=448, quality=90,
        )
        panel_hashes = [hashlib.sha256(value).hexdigest() for value in panels]
        frame_hashes = [str(raw["selected_frame_sha256s"][index]) for index in frame_ids]
        prompt = (
            f"Typed predicate: {row['root_predicate']}\n"
            f"Requested typed role: {candidate.requested_role}\n"
            f"Person track overlay: P0\n"
            f"Candidate track overlay: C0 ({candidate_track.canonical_label})\n"
            f"Frozen temporal operator: {row.get('temporal_operator', 'VIDEO')}\n"
            f"Frozen anchor intervals: {row.get('anchor_intervals', [])}\n"
            f"Presented exact frame IDs: {frame_ids}\n"
            "Verify only whether P0 visibly has the typed predicate relation/action with C0 "
            "inside this scope. Do not choose or name any alternative object."
        )
        core = {
            "protocol": "AGQA_ANSWER_BLIND_BINARY_EVENT_VERIFIER_V1",
            "protocol_file_sha256": _sha256(args.protocol),
            "task_id": task_id, "candidate_receipt_sha256": receipt.receipt_sha256,
            "candidate_track_id": candidate.track_id,
            "candidate_label": candidate_track.canonical_label,
            "predicate": row["root_predicate"], "requested_role": candidate.requested_role,
            "temporal_operator": row.get("temporal_operator", "VIDEO"),
            "anchor_intervals": row.get("anchor_intervals", []),
            "presented_frame_ids": frame_ids,
            "presented_frame_sha256s": frame_hashes,
            "panel_sha256s": panel_hashes, "model": model,
            "system_sha256": stable_hash(SYSTEM), "prompt_sha256": stable_hash(prompt),
            "request_contract": _request_cache_contract(
                model=model,
                max_tokens=int(verifier["max_tokens"]),
                response_format=_response_format(frame_ids),
                maximum_attempts=int(verifier["maximum_contract_attempts"]),
            ),
        }
        provider_error = None
        try:
            payload, usage, reused = _cached_provider_call(
                cache_dir=args.cache_dir, call_name=f"binary_{task_id}", input_core=core,
                invoke=lambda: _provider_call_with_contract_retries(
                    client, model=model, system=SYSTEM,
                    content=[{"type": "text", "text": prompt}] + _panel_content(panels),
                    max_tokens=int(verifier["max_tokens"]),
                    response_format=_response_format(frame_ids),
                    maximum_attempts=int(verifier["maximum_contract_attempts"]),
                    validator=lambda candidate_payload: _validate(
                        candidate_payload, frame_ids, supportable_frame_ids,
                    ),
                ),
            )
            decision = _validate(payload, frame_ids, supportable_frame_ids)
        except Exception as exc:
            usage = getattr(exc, "usage", {
                "reported_cost_usd": 0.0, "provider_attempts": 0,
                "contract_retry_count": 0, "attempt_receipts": [],
            })
            reused = False
            provider_error = f"{type(exc).__name__}:{exc}"
            decision = {"status": "UNKNOWN", "confidence": 0.0, "evidence_frame_ids": []}
        total_cost += float(usage.get("reported_cost_usd", 0.0))
        provider_calls += int(not reused) * int(usage.get("provider_attempts", 1))
        outputs.append({
            "task_id": task_id, "video_id": video_id,
            **decision, "candidate_track_id": candidate.track_id,
            "candidate_label": candidate_track.canonical_label,
            "candidate_receipt_sha256": receipt.receipt_sha256,
            "supportable_evidence_frame_ids": sorted(supportable_frame_ids),
            "presented_frame_ids": frame_ids,
            "presented_frame_sha256s": frame_hashes,
            "panel_sha256s": panel_hashes, "usage": usage,
            "cache_reused": reused, "provider_error": provider_error,
        })
        print(json.dumps({
            "task_id": task_id, "status": decision["status"],
            "confidence": decision["confidence"],
            "cost_usd_running": total_cost,
        }), flush=True)
    report = {
        "schema_version": "agqa-answer-blind-binary-event-verifier-shard-v1",
        "status": "BINARY_EVENT_VERIFIER_SHARD_ANSWER_BLIND",
        "consumed_development_pilot": bool(args.consumed_development_pilot),
        "model": model, "maximum_frames": args.maximum_frames,
        "shard_count": args.shard_count, "shard_index": args.shard_index,
        "cohort_sha256": cohort["cohort_sha256"],
        "candidate_grounding_report_sha256": grounding["report_sha256"],
        "protocol_file_sha256": _sha256(args.protocol),
        "rows": outputs, "provider_calls": provider_calls,
        "reported_cost_usd": total_cost,
        "question_text_read": False, "answer_read": False,
        "official_scene_graph_read": False, "functional_program_read": False,
        "source_controller_read": False, "target_outcome_read": False,
        "alternative_candidate_selection_allowed": False,
        "candidate_label_emitted_as_answer": False,
        "supported_evidence_requires_same_frame_person_and_candidate_tracks": True,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
