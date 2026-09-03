#!/usr/bin/env python3
"""Consumed-development pilot for answer-blind atomic AGQA video grounding.

The VLM never receives the question, answer, temporal operator, official STSG,
functional program, source controller, or target outcome.  It independently
localizes (1) public question-derived anchor action phrases and (2) one atomic
person-object predicate for fixed detector-derived candidate groups.  A
deterministic executor applies BEFORE/AFTER/WHILE/BETWEEN/VIDEO afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
from typing import Iterable

from openai import OpenAI
from PIL import ImageDraw

from motif_transfer.agqa_action_genome_grounder import build_stable_tracks
from motif_transfer.agqa_query_grounder_v2 import query_grounding_v2_from_dict
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import (
    _cached_provider_call,
    _panel_content,
    _panels,
)
from scripts.collect_agqa_question_blind_typed_event_inventory_v1 import (
    _provider_call_with_contract_retries,
    _request_cache_contract,
)
from scripts.pilot_agqa_query_grounder_v4_qwen235_adjudicator import (
    _exact_sgdet_frames,
)


ANCHOR_SYSTEM = """You are an atomic video action localizer. Locate each supplied action/state phrase independently in the chronological raw frames. Do not answer a question, compare anchors, apply a temporal operator, infer an object answer, or use any information outside the shown pixels. A phrase is SUPPORTED only with directly visible evidence; otherwise use UNKNOWN. Return only the required JSON schema."""


EVENT_SYSTEM = """You are an atomic person-object video relation localizer. For every fixed candidate independently, locate frames where person P0 visibly has exactly the supplied predicate relation/action with that candidate. Do not answer a question, choose a best candidate, compare candidates, apply a temporal operator, or infer from mere co-presence. A candidate is SUPPORTED only with directly visible evidence; absence is UNKNOWN, not REFUTED. Return only the required JSON schema."""


_COLORS = (
    "magenta", "cyan", "yellow", "orange", "lime", "red", "deepskyblue",
    "violet", "gold", "springgreen", "coral", "dodgerblue", "hotpink",
    "aquamarine", "khaki", "salmon",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _uniform(values: Iterable[int], maximum: int) -> list[int]:
    ordered = sorted(set(int(value) for value in values))
    if maximum <= 0 or not ordered:
        return []
    if len(ordered) <= maximum:
        return ordered
    if maximum == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (maximum - 1)) for index in range(maximum)]
    return [ordered[index] for index in dict.fromkeys(indices)]


def _bounded_priority(values: Iterable[int], lower: int, upper: int, maximum: int) -> list[int]:
    output = []
    for raw in values:
        value = max(lower, min(upper, int(raw)))
        if value not in output:
            output.append(value)
        if len(output) == maximum:
            break
    return sorted(output)


def _native_to_sampled(raw_video: dict, native_indices: Iterable[int]) -> list[int]:
    selected = [int(value) for value in raw_video["sampled_original_frame_indices"]]
    return [
        min(range(len(selected)), key=lambda index: abs(selected[index] - int(native)))
        for native in native_indices
    ]


def _anchor_specs(plan: dict) -> list[dict]:
    return [
        {
            "anchor_id": f"A{index}",
            "phrase": str(row["phrase"]),
            "native_frame_index_view": [int(value) for value in row["native_frame_index_view"]],
        }
        for index, row in enumerate(plan.get("action_obligations", ()))
    ]


def _candidate_groups(receipt, root_window: tuple[int, int], top_k: int) -> list[dict]:
    lower, upper = root_window
    grouped: dict[str, list] = {}
    for track in receipt.tracks:
        if track.canonical_label == "person":
            continue
        grouped.setdefault(track.canonical_label, []).append(track)
    ranked = []
    for label, tracks in grouped.items():
        inside = {
            frame for track in tracks for frame in track.evidence_frames
            if lower <= frame <= upper
        }
        all_frames = {frame for track in tracks for frame in track.evidence_frames}
        ranked.append((
            (
                bool(inside), len(inside), max(track.confidence for track in tracks),
                len(all_frames), label,
            ),
            label,
            tracks,
            inside,
            all_frames,
        ))
    ranked.sort(key=lambda row: row[0], reverse=True)
    output = []
    for index, (_, label, tracks, inside, all_frames) in enumerate(ranked[:top_k]):
        member_tracks = sorted(
            tracks,
            key=lambda track: (
                sum(lower <= frame <= upper for frame in track.evidence_frames),
                track.confidence,
                len(track.evidence_frames),
                track.track_id,
            ),
            reverse=True,
        )[:3]
        output.append({
            "candidate_id": f"C{index}",
            "canonical_label": label,
            "member_track_ids": [track.track_id for track in member_tracks],
            "all_member_track_ids": sorted(track.track_id for track in tracks),
            "inside_frame_ids": sorted(inside),
            "all_frame_ids": sorted(all_frames),
            "detector_confidence": max(track.confidence for track in tracks),
        })
    return output


def _anchor_frame_ids(raw_video: dict, anchors: list[dict], maximum: int) -> list[int]:
    upper = len(raw_video["sampled_original_frame_indices"]) - 1
    priority = []
    for anchor in anchors:
        mapped = _native_to_sampled(raw_video, anchor["native_frame_index_view"])
        priority.extend(_uniform(mapped, 12))
    priority.extend(_uniform(range(upper + 1), 8))
    return _bounded_priority(priority, 0, upper, maximum)


def _event_frame_ids(
    raw_video: dict,
    receipt,
    candidates: list[dict],
    root_window: tuple[int, int],
    anchor_evidence: Iterable[int],
    maximum: int,
) -> list[int]:
    upper = len(raw_video["sampled_original_frame_indices"]) - 1
    lower_root, upper_root = root_window
    candidate_tracks = {
        track_id for candidate in candidates for track_id in candidate["member_track_ids"]
    }
    priority = []
    for frame in anchor_evidence:
        priority.extend((int(frame) - 1, int(frame), int(frame) + 1))
    for event in receipt.events:
        if any(track_id in candidate_tracks for _, track_id in event.roles):
            priority.extend(_uniform(event.evidence_frames, 3))
    for candidate in candidates:
        priority.extend(_uniform(candidate["inside_frame_ids"], 3))
        priority.extend(_uniform(candidate["all_frame_ids"], 2))
    priority.extend(_uniform(range(lower_root, upper_root + 1), 6))
    priority.extend(_uniform(range(upper + 1), 4))
    return _bounded_priority(priority, 0, upper, maximum)


def _annotate(
    images,
    frame_ids: list[int],
    scales,
    raw_video: dict,
    detection_to_track: dict[int, str],
    retained_detection_indices: set[int] | frozenset[int],
    candidates: list[dict],
):
    output = [image.copy() for image in images]
    track_to_candidate = {
        track_id: candidate["candidate_id"]
        for candidate in candidates for track_id in candidate["member_track_ids"]
    }
    candidate_order = {
        candidate["candidate_id"]: index for index, candidate in enumerate(candidates)
    }
    by_frame = {}
    for detected in raw_video["objects"]:
        detection_index = int(detected["detection_index"])
        if detection_index not in retained_detection_indices:
            continue
        track_id = detection_to_track.get(detection_index)
        if track_id == "T0":
            label = "P0"
            color = "white"
        elif track_id in track_to_candidate:
            label = track_to_candidate[track_id]
            color = _COLORS[candidate_order[label] % len(_COLORS)]
        else:
            continue
        by_frame.setdefault(int(detected["sampled_frame_index"]), []).append(
            (label, color, detected)
        )
    for image, frame_id in zip(output, frame_ids):
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), f"S{frame_id}", fill="white", stroke_width=3, stroke_fill="black")
        for label, color, detected in by_frame.get(frame_id, ()):
            scale = float(scales[frame_id])
            x1, y1, x2, y2 = (
                float(value) / scale for value in detected["bbox_xyxy"]
            )
            width, height = image.size
            box = (
                max(0.0, min(float(width - 1), x1)),
                max(0.0, min(float(height - 1), y1)),
                max(0.0, min(float(width - 1), x2)),
                max(0.0, min(float(height - 1), y2)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            draw.rectangle(box, outline=color, width=4)
            draw.text(
                (box[0] + 2, box[1] + 2), label,
                fill=color, stroke_width=2, stroke_fill="black",
            )
    return output


def _anchor_response_format(anchor_ids: list[str], frame_ids: list[int]) -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_atomic_anchor_localization_v2", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "anchors": {
                    "type": "array", "minItems": len(anchor_ids), "maxItems": len(anchor_ids),
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "anchor_id": {"type": "string", "enum": anchor_ids},
                            "status": {"type": "string", "enum": ["SUPPORTED", "UNKNOWN"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "evidence_frame_ids": {
                                "type": "array", "maxItems": 8,
                                "items": {"type": "integer", "enum": frame_ids},
                            },
                        },
                        "required": ["anchor_id", "status", "confidence", "evidence_frame_ids"],
                    },
                },
            },
            "required": ["anchors"],
        },
    }}


def _event_response_format(candidate_ids: list[str], frame_ids: list[int]) -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_atomic_candidate_event_localization_v2", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "events": {
                    "type": "array", "minItems": len(candidate_ids), "maxItems": len(candidate_ids),
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "candidate_id": {"type": "string", "enum": candidate_ids},
                            "status": {"type": "string", "enum": ["SUPPORTED", "UNKNOWN"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "evidence_frame_ids": {
                                "type": "array", "maxItems": 8,
                                "items": {"type": "integer", "enum": frame_ids},
                            },
                        },
                        "required": ["candidate_id", "status", "confidence", "evidence_frame_ids"],
                    },
                },
            },
            "required": ["events"],
        },
    }}


def _validate_rows(payload: dict, key: str, id_key: str, ids: list[str], frame_ids: list[int]) -> list[dict]:
    if set(payload) != {key} or not isinstance(payload[key], list):
        raise ValueError(f"atomic grounding payload must contain only {key}")
    rows = payload[key]
    if len(rows) != len(ids) or {str(row.get(id_key)) for row in rows} != set(ids):
        raise ValueError("atomic grounding did not return every fixed identifier exactly once")
    allowed_frames = set(frame_ids)
    output = []
    for row in rows:
        if set(row) != {id_key, "status", "confidence", "evidence_frame_ids"}:
            raise ValueError("atomic grounding row contains unexpected fields")
        status = str(row["status"])
        if status not in {"SUPPORTED", "UNKNOWN"}:
            raise ValueError("atomic grounding status is invalid")
        confidence = float(row["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("atomic grounding confidence is invalid")
        evidence = sorted(set(int(value) for value in row["evidence_frame_ids"]))
        if len(evidence) > 8 or any(value not in allowed_frames for value in evidence):
            raise ValueError("atomic grounding cites an unpresented frame")
        if status == "SUPPORTED" and not evidence:
            raise ValueError("supported atomic grounding requires pixel evidence")
        if status == "UNKNOWN" and evidence:
            raise ValueError("unknown atomic grounding cannot cite positive evidence")
        output.append({
            id_key: str(row[id_key]), "status": status,
            "confidence": confidence, "evidence_frame_ids": evidence,
        })
    return sorted(output, key=lambda row: ids.index(row[id_key]))


def _provider_failure(exc: Exception) -> tuple[dict, str]:
    usage = getattr(exc, "usage", None)
    if not isinstance(usage, dict):
        usage = {
            "model": "unknown", "finish_reason": "provider_exception",
            "prompt_tokens": 0, "completion_tokens": 0,
            "reported_cost_usd": 0.0, "provider_attempts": 0,
            "contract_retry_count": 0, "attempt_receipts": [],
        }
    return usage, f"{type(exc).__name__}:{exc}"


def _execute_temporal(
    temporal_operator: str,
    anchors: list[dict],
    events: list[dict],
    candidates: list[dict],
) -> dict:
    supported_anchors = {
        row["anchor_id"]: row for row in anchors if row["status"] == "SUPPORTED"
    }
    required_anchor_count = 0 if temporal_operator == "VIDEO" else (2 if temporal_operator == "BETWEEN" else 1)
    if len(supported_anchors) < required_anchor_count:
        return {"status": "ABSTAIN_ANCHOR_UNGROUNDED", "selected_candidate_id": None, "ranking": []}
    anchor_intervals = [
        (min(supported_anchors[f"A{index}"]["evidence_frame_ids"]),
         max(supported_anchors[f"A{index}"]["evidence_frame_ids"]))
        for index in range(required_anchor_count)
    ]
    by_candidate = {row["candidate_id"]: row for row in candidates}
    ranking = []
    for row in events:
        if row["status"] != "SUPPORTED":
            continue
        frames = row["evidence_frame_ids"]
        valid = []
        proximity = -10_000
        if temporal_operator == "VIDEO":
            valid = frames
            proximity = len(frames)
        elif temporal_operator == "BEFORE":
            valid = [frame for frame in frames if frame < anchor_intervals[0][0]]
            proximity = max(valid) if valid else -10_000
        elif temporal_operator == "AFTER":
            valid = [frame for frame in frames if frame > anchor_intervals[0][1]]
            proximity = -min(valid) if valid else -10_000
        elif temporal_operator == "WHILE":
            lower, upper = anchor_intervals[0]
            valid = [frame for frame in frames if lower <= frame <= upper]
            proximity = len(valid)
        elif temporal_operator == "BETWEEN":
            lower, upper = anchor_intervals[0][1], anchor_intervals[1][0]
            if lower > upper:
                lower, upper = upper, lower
            valid = [frame for frame in frames if lower < frame < upper]
            proximity = len(valid)
        else:
            raise ValueError(f"unsupported temporal operator {temporal_operator}")
        if not valid:
            continue
        candidate = by_candidate[row["candidate_id"]]
        ranking.append({
            "candidate_id": row["candidate_id"],
            "canonical_label": candidate["canonical_label"],
            "valid_event_frame_ids": valid,
            "temporal_proximity": proximity,
            "event_confidence": row["confidence"],
            "detector_confidence": candidate["detector_confidence"],
        })
    ranking.sort(
        key=lambda row: (
            row["temporal_proximity"], row["event_confidence"],
            len(row["valid_event_frame_ids"]), row["detector_confidence"],
            row["canonical_label"],
        ),
        reverse=True,
    )
    if not ranking:
        return {"status": "ABSTAIN_NO_TEMPORALLY_VALID_EVENT", "selected_candidate_id": None, "ranking": []}
    return {
        "status": "SUPPORTED_BY_ATOMIC_TEMPORAL_EXECUTOR",
        "selected_candidate_id": ranking[0]["candidate_id"],
        "selected_label": ranking[0]["canonical_label"],
        "ranking": ranking,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--candidate-grounding", type=Path, required=True)
    parser.add_argument("--query-plans", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--top-k-candidates", type=int, default=12)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--maximum-anchor-frames", type=int, default=20)
    parser.add_argument("--maximum-event-frames", type=int, default=20)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("atomic temporal grounder pilot output is immutable")
    cohort = json.loads(args.cohort.read_text())
    sgdet = json.loads(args.sgdet.read_text())
    grounding = json.loads(args.candidate_grounding.read_text())
    plans = json.loads(args.query_plans.read_text())
    if any(grounding.get(key) for key in (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )):
        raise ValueError("candidate grounding crossed its authority boundary")
    wanted = set(args.task_id or ())
    rows = [row for row in grounding["rows"] if not wanted or str(row["task_id"]) in wanted]
    if wanted != {str(row["task_id"]) for row in rows} and wanted:
        raise ValueError("requested task ID is absent")
    if args.max_tasks is not None:
        rows = rows[:args.max_tasks]
    public_paths = {str(row["video_id"]): Path(row["video_path"]) for row in cohort["video_selections"]}
    raw_by_video = {str(row["video_id"]): row for row in sgdet["rows"]}
    plan_by_task = {str(row["task_id"]): row for row in plans["rows"]}
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key unavailable")
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=300, max_retries=2)
    # Qwen's OpenRouter endpoints accept structured output, but no currently
    # routable endpoint accepts an explicit reasoning.enabled=false parameter.
    # Omitting the optional parameter preserves the instruct model's normal
    # transport while the prompt still forbids question answering.
    model = {"id": args.model, "omit_temperature": True}
    outputs = []
    total_cost = 0.0
    provider_calls = 0
    for row in rows:
        task_id = str(row["task_id"])
        video_id = str(row["video_id"])
        receipt = query_grounding_v2_from_dict(row["receipt"])
        plan = plan_by_task[task_id]
        predicate = str(plan.get("predicate") or "").strip()
        temporal_operator = str(plan["temporal_operator"])
        root_window = tuple(int(value) for value in row["root_temporal_window"])
        raw_video = raw_by_video[video_id]
        stable = build_stable_tracks(raw_video, minimum_object_score=0.05)
        anchors = _anchor_specs(plan)
        candidates = _candidate_groups(receipt, root_window, args.top_k_candidates)
        anchor_rows = []
        call_receipts = []
        if anchors:
            frame_ids = _anchor_frame_ids(raw_video, anchors, args.maximum_anchor_frames)
            images, seconds, scales = _exact_sgdet_frames(public_paths[video_id], raw_video, frame_ids)
            annotated = _annotate(
                images, frame_ids, scales, raw_video, stable.detection_to_track,
                stable.retained_detection_indices, [],
            )
            panels = _panels(annotated, seconds, frames_per_panel=2, frame_width=448, quality=90)
            anchor_ids = [anchor["anchor_id"] for anchor in anchors]
            prompt = (
                "Fixed atomic anchor phrases:\n"
                + "\n".join(f"{anchor['anchor_id']}: {anchor['phrase']}" for anchor in anchors)
                + f"\nPresented chronological sampled-frame IDs: {frame_ids}\n"
                "Locate each phrase independently. Do not compare their order or answer any question."
            )
            response_format = _anchor_response_format(anchor_ids, frame_ids)
            core = {
                "protocol": "AGQA_ATOMIC_TEMPORAL_GROUNDER_V2_DEV_PILOT_ANCHOR",
                "task_id": task_id, "video_sha256": receipt.video_sha256,
                "anchor_phrases": [{"anchor_id": x["anchor_id"], "phrase": x["phrase"]} for x in anchors],
                "presented_frame_ids": frame_ids,
                "presented_frame_sha256s": [raw_video["selected_frame_sha256s"][index] for index in frame_ids],
                "panel_sha256s": [hashlib.sha256(value).hexdigest() for value in panels],
                "model": model, "system_sha256": stable_hash(ANCHOR_SYSTEM),
                "prompt_sha256": stable_hash(prompt),
                "request_contract": _request_cache_contract(
                    model=model, max_tokens=640,
                    response_format=response_format, maximum_attempts=2,
                ),
            }
            provider_error = None
            try:
                payload, usage, reused = _cached_provider_call(
                    cache_dir=args.cache_dir, call_name=f"atomic_anchor_{task_id}", input_core=core,
                    invoke=lambda: _provider_call_with_contract_retries(
                        client, model=model, system=ANCHOR_SYSTEM,
                        content=[{"type": "text", "text": prompt}] + _panel_content(panels),
                        max_tokens=640, response_format=response_format,
                        maximum_attempts=2,
                        validator=lambda candidate_payload: _validate_rows(
                            candidate_payload, "anchors", "anchor_id", anchor_ids, frame_ids,
                        ),
                    ),
                )
                anchor_rows = _validate_rows(payload, "anchors", "anchor_id", anchor_ids, frame_ids)
            except Exception as exc:
                usage, provider_error = _provider_failure(exc)
                reused = False
                anchor_rows = [{
                    "anchor_id": anchor_id, "status": "UNKNOWN", "confidence": 0.0,
                    "evidence_frame_ids": [],
                } for anchor_id in anchor_ids]
            total_cost += float(usage.get("reported_cost_usd", 0.0))
            provider_calls += int(not reused) * int(usage.get("provider_attempts", 0))
            call_receipts.append({
                "kind": "ANCHOR", "usage": usage, "cache_reused": reused,
                "provider_error": provider_error, **core,
            })
        event_rows = []
        anchor_evidence = [frame for anchor in anchor_rows for frame in anchor["evidence_frame_ids"]]
        if predicate:
            for start in range(0, len(candidates), args.candidate_batch_size):
                batch = candidates[start:start + args.candidate_batch_size]
                frame_ids = _event_frame_ids(
                    raw_video, receipt, batch, root_window, anchor_evidence,
                    args.maximum_event_frames,
                )
                images, seconds, scales = _exact_sgdet_frames(public_paths[video_id], raw_video, frame_ids)
                annotated = _annotate(
                    images, frame_ids, scales, raw_video,
                    stable.detection_to_track, stable.retained_detection_indices,
                    batch,
                )
                panels = _panels(annotated, seconds, frames_per_panel=2, frame_width=448, quality=90)
                candidate_ids = [candidate["candidate_id"] for candidate in batch]
                prompt = (
                    f"Fixed atomic predicate: {predicate}\n"
                    "P0 is the person track. Fixed candidate groups:\n"
                    + "\n".join(
                        f"{candidate['candidate_id']}: {candidate['canonical_label']}"
                        for candidate in batch
                    )
                    + f"\nPresented chronological sampled-frame IDs: {frame_ids}\n"
                    "For every candidate independently, cite only frames where P0 visibly has exactly "
                    "the atomic predicate with that candidate. Do not rank or select candidates."
                )
                response_format = _event_response_format(candidate_ids, frame_ids)
                core = {
                    "protocol": "AGQA_ATOMIC_TEMPORAL_GROUNDER_V2_DEV_PILOT_EVENT",
                    "task_id": task_id, "video_sha256": receipt.video_sha256,
                    "predicate": predicate,
                    "candidate_groups": [{
                        "candidate_id": x["candidate_id"],
                        "canonical_label": x["canonical_label"],
                        "member_track_ids": x["member_track_ids"],
                    } for x in batch],
                    "presented_frame_ids": frame_ids,
                    "presented_frame_sha256s": [raw_video["selected_frame_sha256s"][index] for index in frame_ids],
                    "panel_sha256s": [hashlib.sha256(value).hexdigest() for value in panels],
                    "model": model, "system_sha256": stable_hash(EVENT_SYSTEM),
                    "prompt_sha256": stable_hash(prompt),
                    "request_contract": _request_cache_contract(
                        model=model, max_tokens=800,
                        response_format=response_format, maximum_attempts=2,
                    ),
                }
                provider_error = None
                try:
                    payload, usage, reused = _cached_provider_call(
                        cache_dir=args.cache_dir,
                        call_name=f"atomic_event_{task_id}_{start // args.candidate_batch_size}",
                        input_core=core,
                        invoke=lambda candidate_ids=candidate_ids, frame_ids=frame_ids, panels=panels, prompt=prompt: _provider_call_with_contract_retries(
                            client, model=model, system=EVENT_SYSTEM,
                            content=[{"type": "text", "text": prompt}] + _panel_content(panels),
                            max_tokens=800, response_format=response_format,
                            maximum_attempts=2,
                            validator=lambda candidate_payload: _validate_rows(
                                candidate_payload, "events", "candidate_id", candidate_ids, frame_ids,
                            ),
                        ),
                    )
                    event_rows.extend(_validate_rows(
                        payload, "events", "candidate_id", candidate_ids, frame_ids,
                    ))
                except Exception as exc:
                    usage, provider_error = _provider_failure(exc)
                    reused = False
                    event_rows.extend({
                        "candidate_id": candidate_id, "status": "UNKNOWN",
                        "confidence": 0.0, "evidence_frame_ids": [],
                    } for candidate_id in candidate_ids)
                total_cost += float(usage.get("reported_cost_usd", 0.0))
                provider_calls += int(not reused) * int(usage.get("provider_attempts", 0))
                call_receipts.append({
                    "kind": "EVENT", "usage": usage, "cache_reused": reused,
                    "provider_error": provider_error, **core,
                })
        decision = _execute_temporal(temporal_operator, anchor_rows, event_rows, candidates)
        outputs.append({
            "task_id": task_id, "video_id": video_id,
            "predicate": predicate, "temporal_operator": temporal_operator,
            "root_temporal_window": list(root_window),
            "anchors": [{k: v for k, v in anchor.items() if k != "native_frame_index_view"} for anchor in anchors],
            "anchor_localizations": anchor_rows,
            "candidate_groups": candidates,
            "atomic_event_localizations": event_rows,
            "decision": decision,
            "call_receipts": call_receipts,
        })
        print(json.dumps({
            "task_id": task_id, "status": decision["status"],
            "selected_label": decision.get("selected_label"),
            "calls_running": provider_calls, "cost_usd_running": total_cost,
        }), flush=True)
    report = {
        "schema_version": "agqa-atomic-temporal-grounder-v2-consumed-development-pilot-v1",
        "status": "CONSUMED_DEVELOPMENT_PILOT_NOT_TRANSFER_EVIDENCE",
        "model": model,
        "top_k_candidates": args.top_k_candidates,
        "candidate_batch_size": args.candidate_batch_size,
        "maximum_anchor_frames": args.maximum_anchor_frames,
        "maximum_event_frames": args.maximum_event_frames,
        "rows": outputs,
        "provider_calls": provider_calls,
        "reported_cost_usd": total_cost,
        "authority": {
            "question_text_supplied_to_vlm": False,
            "temporal_operator_supplied_to_vlm": False,
            "answer_read": False,
            "official_stsg_read": False,
            "functional_program_read": False,
            "source_controller_read": False,
            "target_outcome_read": False,
        },
        "immutable_input_file_sha256s": {
            "cohort": _sha256(args.cohort), "sgdet": _sha256(args.sgdet),
            "candidate_grounding": _sha256(args.candidate_grounding),
            "query_plans": _sha256(args.query_plans),
        },
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
