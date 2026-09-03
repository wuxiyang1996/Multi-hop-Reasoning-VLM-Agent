#!/usr/bin/env python3
"""Collect a per-video, question-blind typed event inventory for AGQA.

The provider sees only exact content-addressed raw frames, detector-assigned
stable track IDs, and the public target ontology.  Questions, answers, source
controllers, official STSGs, functional programs, and target outcomes are not
included in any prompt or cache key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
from typing import Any

from openai import OpenAI
from PIL import ImageDraw

from motif_transfer.agqa_action_genome_grounder import build_stable_tracks
from motif_transfer.agqa_question_blind_event_grounder import (
    PUBLIC_EVENT_PREDICATES,
    deduplicate_question_blind_events,
    parse_question_blind_event_payload_with_rejections,
)
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import (
    _cached_provider_call, _decode_json_object, _panel_content, _panels,
)
from scripts.pilot_agqa_query_grounder_v4_qwen235_adjudicator import (
    _exact_sgdet_frames,
)


SYSTEM = """You are a frozen question-blind raw-video event perception tool, not a question-answering agent. Inspect the chronological frames and track overlays. Enumerate visibly supported person-object events and persistent states using the supplied public ontology and exact stable track IDs. A holding, carrying, wearing, sitting, standing, contact, attention, or spatial state directly visible in one frame is valid evidence; observing its onset is not required. Be comprehensive, but never infer an event solely from object co-presence. Never output or infer a task question, answer, correctness, official annotation, functional program, source-domain controller, or target outcome. Return only the required JSON schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _clip_frame_ids(
    frame_count: int, clip_count: int, frames_per_clip: int, *, strategy: str,
) -> list[list[int]]:
    """Return fixed chronological samples without consulting a question."""

    if frame_count < 2 or clip_count < 1 or frames_per_clip < 2:
        raise ValueError("invalid frame/clip budget")
    if strategy == "partition_complete":
        if clip_count * frames_per_clip < frame_count:
            raise ValueError("partition_complete budget cannot cover every frame")
        output = []
        for clip_index in range(clip_count):
            lower = round(clip_index * frame_count / clip_count)
            upper = round((clip_index + 1) * frame_count / clip_count)
            available = list(range(lower, upper))
            if len(available) > frames_per_clip:
                available = [
                    available[round(offset * (len(available) - 1) / (frames_per_clip - 1))]
                    for offset in range(frames_per_clip)
                ]
            output.append(available)
        covered = {frame for clip in output for frame in clip}
        if covered != set(range(frame_count)):
            raise ValueError("partition_complete did not cover the full frozen frame set")
        return output
    if strategy != "overlap_uniform":
        raise ValueError("unknown clip sampling strategy")
    output = []
    for clip_index in range(clip_count):
        core_lower = round(clip_index * (frame_count - 1) / clip_count)
        core_upper = round((clip_index + 1) * (frame_count - 1) / clip_count)
        lower = max(0, core_lower - int(clip_index > 0))
        upper = min(frame_count - 1, core_upper + int(clip_index + 1 < clip_count))
        frames = []
        for offset in range(frames_per_clip):
            value = round(lower + offset * (upper - lower) / (frames_per_clip - 1))
            if value not in frames:
                frames.append(value)
        if len(frames) < 2:
            raise ValueError("clip sampling collapsed")
        output.append(frames)
    return output


def _response_format(
    track_ids: list[str], person_ids: list[str], frame_ids: list[int],
    maximum_events: int,
) -> dict:
    non_person_ids = [value for value in track_ids if value not in set(person_ids)]
    if not person_ids or not non_person_ids:
        raise ValueError("event schema requires visible person and object tracks")
    event = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "predicate": {"type": "string", "enum": list(PUBLIC_EVENT_PREDICATES)},
            "subject_track_id": {"type": "string", "enum": person_ids},
            "object_track_id": {"type": "string", "enum": non_person_ids},
            "start_frame_id": {"type": "integer", "enum": frame_ids},
            "end_frame_id": {"type": "integer", "enum": frame_ids},
            "evidence_frame_ids": {
                "type": "array", "minItems": 1, "maxItems": 4,
                "items": {"type": "integer", "enum": frame_ids},
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "predicate", "subject_track_id", "object_track_id",
            "start_frame_id", "end_frame_id", "evidence_frame_ids", "confidence",
        ],
    }
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_question_blind_typed_events_v1", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "events": {"type": "array", "maxItems": maximum_events, "items": event}
            },
            "required": ["events"],
        },
    }}


def _validate_event_envelope(payload: dict[str, Any], maximum_events: int) -> None:
    if set(payload) != {"events"} or not isinstance(payload["events"], list):
        raise ValueError("event response must contain only an events array")
    if len(payload["events"]) > maximum_events:
        raise ValueError("event response exceeds the frozen event limit")


class ProviderContractError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]):
        super().__init__(message)
        self.usage = usage


def _request_cache_contract(
    *, model: dict[str, Any], max_tokens: int,
    response_format: dict[str, Any], maximum_attempts: int,
) -> dict[str, Any]:
    """Return every request option that can change a cached completion.

    The image, prompt, and system hashes live in the surrounding cache core.
    Keeping transport/schema options here prevents a retry with a larger output
    budget (or a different structured-output schema) from silently reusing an
    older completion.
    """

    return {
        "model": str(model["id"]),
        "max_tokens": int(max_tokens),
        "maximum_contract_attempts": int(maximum_attempts),
        "response_format_sha256": stable_hash(response_format),
        "temperature_mode": "omitted" if model.get("omit_temperature") else "zero",
        "seed": model.get("seed"),
        "provider": {
            "require_parameters": True,
            **dict(model.get("provider") or {}),
        },
        "reasoning": model.get("reasoning"),
    }


def _artifact_status(consumed_development_pilot: bool) -> str:
    if consumed_development_pilot:
        return "CONSUMED_DEVELOPMENT_PILOT_NOT_TRANSFER_EVIDENCE"
    return "QUESTION_BLIND_EVENT_INVENTORY_SHARD_FROZEN_BEFORE_TASK_QUERY_OR_OUTCOME"


def _provider_call_with_contract_retries(
    client: OpenAI, *, model: dict[str, Any], system: str,
    content: list[dict[str, Any]], max_tokens: int, response_format: dict[str, Any],
    maximum_attempts: int, validator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retry identical malformed responses while retaining billed usage."""

    attempts = []
    for attempt_index in range(maximum_attempts):
        extra_body: dict[str, Any] = {
            "provider": {
                "require_parameters": True,
                **dict(model.get("provider") or {}),
            },
        }
        if model.get("reasoning") is not None:
            extra_body["reasoning"] = dict(model["reasoning"])
        request: dict[str, Any] = {
            "model": str(model["id"]), "max_tokens": max_tokens,
            "response_format": response_format,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "extra_body": extra_body,
        }
        if not model.get("omit_temperature"):
            request["temperature"] = 0
        if model.get("seed") is not None:
            request["seed"] = int(model["seed"])
        response = client.chat.completions.create(**request)
        if not getattr(response, "choices", None):
            raise RuntimeError("provider response omitted choices")
        raw = response.choices[0].message.content or ""
        usage = response.usage
        usage_extra = getattr(usage, "model_extra", None) or {}
        row = {
            "attempt_index": attempt_index,
            "model": str(response.model),
            "finish_reason": str(response.choices[0].finish_reason),
            "prompt_tokens": int(usage.prompt_tokens if usage else 0),
            "completion_tokens": int(usage.completion_tokens if usage else 0),
            "reported_cost_usd": float(
                getattr(usage, "cost", 0.0) or usage_extra.get("cost", 0.0) or 0.0
            ),
            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        }
        try:
            payload = _decode_json_object(raw)
            validator(payload)
            row["contract_valid"] = True
            attempts.append(row)
            return payload, {
                "model": str(response.model),
                "finish_reason": str(response.choices[0].finish_reason),
                "prompt_tokens": sum(item["prompt_tokens"] for item in attempts),
                "completion_tokens": sum(item["completion_tokens"] for item in attempts),
                "reported_cost_usd": sum(item["reported_cost_usd"] for item in attempts),
                "response_sha256": stable_hash(payload),
                "provider_attempts": len(attempts),
                "contract_retry_count": len(attempts) - 1,
                "attempt_receipts": attempts,
            }
        except Exception as exc:
            row["contract_valid"] = False
            row["contract_error"] = f"{type(exc).__name__}:{exc}"
            attempts.append(row)
    aggregate = {
        "model": str(model["id"]), "finish_reason": "contract_retries_exhausted",
        "prompt_tokens": sum(item["prompt_tokens"] for item in attempts),
        "completion_tokens": sum(item["completion_tokens"] for item in attempts),
        "reported_cost_usd": sum(item["reported_cost_usd"] for item in attempts),
        "response_sha256": stable_hash(attempts),
        "provider_attempts": len(attempts),
        "contract_retry_count": max(0, len(attempts) - 1),
        "attempt_receipts": attempts,
    }
    raise ProviderContractError(
        "identical provider contract retries exhausted", aggregate,
    )


def _detections_by_frame(video: dict, minimum_object_score: float) -> tuple[Any, dict[int, list[tuple[str, dict]]]]:
    stable = build_stable_tracks(video, minimum_object_score=minimum_object_score)
    by_frame: dict[int, list[tuple[str, dict]]] = {}
    best: dict[tuple[int, str], dict] = {}
    for detected in video["objects"]:
        index = int(detected["detection_index"])
        if index not in stable.retained_detection_indices:
            continue
        track_id = stable.detection_to_track.get(index)
        if track_id is None:
            continue
        key = (int(detected["sampled_frame_index"]), str(track_id))
        if key not in best or float(detected["score"]) > float(best[key]["score"]):
            best[key] = detected
    for (frame, track_id), detected in best.items():
        by_frame.setdefault(frame, []).append((track_id, detected))
    for frame in by_frame:
        by_frame[frame].sort(key=lambda value: (
            value[0] != "T0", -float(value[1]["score"]), value[0],
        ))
    return stable, by_frame


def _annotate_frames(images, frame_ids, scales, detections, track_labels):
    output = [image.copy() for image in images]
    palette = (
        "red", "lime", "cyan", "yellow", "magenta", "orange",
        "deepskyblue", "violet", "springgreen", "gold", "white",
    )
    for image, frame_id in zip(output, frame_ids):
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), f"S{frame_id}", fill="white", stroke_width=3, stroke_fill="black")
        for index, (track_id, detected) in enumerate(detections.get(frame_id, ())):
            scale = float(scales[frame_id])
            box = tuple(float(value) / scale for value in detected["bbox_xyxy"])
            box = (
                max(0.0, min(float(image.width - 1), box[0])),
                max(0.0, min(float(image.height - 1), box[1])),
                max(0.0, min(float(image.width - 1), box[2])),
                max(0.0, min(float(image.height - 1), box[3])),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            color = palette[int(track_id[1:]) % len(palette)]
            draw.rectangle(box, outline=color, width=4)
            draw.text(
                (box[0] + 2, box[1] + 2), f"{track_id}:{track_labels[track_id]}",
                fill=color, stroke_width=2, stroke_fill="black",
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--provider", default="alibaba")
    parser.add_argument("--clip-count", type=int, default=4)
    parser.add_argument("--frames-per-clip", type=int, default=6)
    parser.add_argument(
        "--sampling-strategy", choices=("overlap_uniform", "partition_complete"),
        default="overlap_uniform",
    )
    parser.add_argument("--minimum-object-score", type=float, default=0.05)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument(
        "--consumed-development-pilot", action="store_true",
        help="Label an after-outcome method diagnostic so it cannot be cited as transfer evidence.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("question-blind event inventory shard is immutable")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")

    cohort = json.loads(args.cohort.read_text())
    sgdet = json.loads(args.sgdet.read_text())
    ontology = json.loads(args.ontology.read_text())
    protocol = json.loads(args.protocol.read_text())
    forbidden = (
        "answer_read", "functional_program_read", "official_scene_graph_read",
        "source_controller_read", "target_outcome_read",
    )
    if any(sgdet.get(key) for key in forbidden):
        raise ValueError("SGDET input crosses the authority boundary")
    expected = protocol["immutable_inputs"]
    actual = {
        "cohort_sha256": cohort["cohort_sha256"],
        "sgdet_file_sha256": _sha256(args.sgdet),
        "public_ontology_file_sha256": _sha256(args.ontology),
        "event_grounder_module_sha256": _sha256(
            Path(__file__).resolve().parents[1]
            / "src/motif_transfer/agqa_question_blind_event_grounder.py"
        ),
        "event_collector_sha256": _sha256(Path(__file__)),
    }
    if any(expected.get(key) != value for key, value in actual.items()):
        raise ValueError(f"immutable inputs differ: expected={expected} actual={actual}")
    acquisition = protocol["question_blind_event_acquisition"]
    for key, value in (
        ("model", args.model), ("clip_count", args.clip_count),
        ("frames_per_clip", args.frames_per_clip),
        ("sampling_strategy", args.sampling_strategy),
        ("minimum_object_score", args.minimum_object_score),
    ):
        if acquisition[key] != value:
            raise ValueError(f"runtime {key} differs from frozen protocol")
    # V1d predated endpoint pinning.  Every V2 protocol must bind the concrete
    # OpenRouter provider as well as the public model name: provider-side model
    # variants were empirically non-equivalent on the consumed development
    # videos, and dynamic routing would make the grounding artifact impossible
    # to reproduce or audit.
    if "provider" in acquisition and acquisition["provider"] != args.provider:
        raise ValueError("runtime provider differs from frozen protocol")
    if "seed" in acquisition and int(acquisition["seed"]) != 0:
        raise ValueError("event acquisition currently supports only frozen seed 0")
    if acquisition.get("provider_allow_fallbacks") not in {None, False}:
        raise ValueError("event acquisition must disable provider fallback")
    if ontology.get("object_classes") is None or ontology.get("contacting_relationships") is None:
        raise ValueError("public ontology is incomplete")

    raw_by_video = {str(row["video_id"]): row for row in sgdet["rows"]}
    # Deliberately derive work only from the video allocation.  The public
    # cohort also stores questions, but this acquisition stage never reads its
    # task rows.
    selections = sorted(cohort["video_selections"], key=lambda row: str(row["video_id"]))
    selections = [
        row for index, row in enumerate(selections)
        if index % args.shard_count == args.shard_index
    ]
    if args.max_videos is not None:
        selections = selections[:args.max_videos]
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key unavailable")
    client = OpenAI(
        api_key=key, base_url="https://openrouter.ai/api/v1",
        timeout=300, max_retries=2,
    )
    # The frozen protocol specifies temperature zero and thinking disabled.
    # OpenRouter exposes this instruct model as non-reasoning, so no reasoning
    # parameter is sent; the provider and seed are pinned to prevent routing or
    # sampling drift between shards.
    model = {
        "id": args.model,
        "omit_temperature": False,
        "seed": 0,
        "provider": {"only": [args.provider], "allow_fallbacks": False},
    }

    outputs = []
    total_cost = 0.0
    provider_calls = 0
    for selection in selections:
        video_id = str(selection["video_id"])
        video = raw_by_video[video_id]
        frame_count = int(video["model_visible_frame_count"])
        stable, detections = _detections_by_frame(video, args.minimum_object_score)
        track_labels = {row.track_id: row.canonical_label for row in stable.tracks}
        person_ids = sorted(
            track_id for track_id, label in track_labels.items() if label == "person"
        )
        all_events = []
        clip_rows = []
        for clip_index, frame_ids in enumerate(_clip_frame_ids(
            frame_count, args.clip_count, args.frames_per_clip,
            strategy=args.sampling_strategy,
        )):
            clip_id = f"C{clip_index}"
            images, seconds, scales = _exact_sgdet_frames(
                Path(selection["video_path"]), video, frame_ids,
            )
            visible = sorted({
                track_id for frame_id in frame_ids
                for track_id, _ in detections.get(frame_id, ())
            }, key=lambda value: int(value[1:]))
            visible_persons = [value for value in person_ids if value in visible]
            visible_objects = [value for value in visible if value not in visible_persons]
            track_visible_frames = {
                track_id: frozenset(
                    frame_id for frame_id in frame_ids
                    if any(
                        candidate_id == track_id
                        for candidate_id, _ in detections.get(frame_id, ())
                    )
                )
                for track_id in visible
            }
            frame_receipts = [{
                "frame_id": frame_id,
                "native_frame_index": int(video["sampled_original_frame_indices"][frame_id]),
                "raw_frame_sha256": str(video["selected_frame_sha256s"][frame_id]),
                "second": float(second),
            } for frame_id, second in zip(frame_ids, seconds)]
            if not visible_persons or not visible_objects:
                clip_rows.append({
                    "clip_id": clip_id, "presented_frames": frame_receipts,
                    "visible_track_ids": visible, "events": [],
                    "provider_error": "NO_VISIBLE_PERSON_OBJECT_PAIR",
                    "usage": None, "cache_reused": False,
                })
                continue
            annotated = _annotate_frames(
                images, frame_ids, scales, detections, track_labels,
            )
            panels = _panels(
                annotated, seconds, frames_per_panel=2, frame_width=448, quality=90,
            )
            panel_hashes = [hashlib.sha256(value).hexdigest() for value in panels]
            visible_table = "; ".join(
                f"{track_id}={track_labels[track_id]}" for track_id in visible
            )
            per_frame_visible = "; ".join(
                f"S{frame_id}=" + ",".join(
                    track_id for track_id in visible
                    if frame_id in track_visible_frames[track_id]
                )
                for frame_id in frame_ids
            )
            prompt = (
                "This is one chronological segment from a raw video. No task question is supplied.\n"
                f"Presented exact frame IDs: {frame_ids}\n"
                f"Visible frozen stable tracks: {visible_table}\n"
                f"Tracks visibly detected in each frame: {per_frame_visible}\n"
                "Public event predicates: " + ", ".join(PUBLIC_EVENT_PREDICATES) + "\n"
                "Enumerate every clearly visible person-object event among those tracks. "
                "Use only displayed track IDs and frame IDs. Cite 1-4 evidence frames. "
                "Set interval endpoints to displayed frames that bound the visible evidence. "
                "Persistent states such as holding, carrying, wearing, looking at, sitting on, "
                "standing on, behind, in front of, above, beneath, or beside are valid when "
                "directly visible; do not require their transition onset. Report all such "
                "supported states, even if only one sampled frame shows them. Do not report "
                "mere co-presence as interaction and do not guess events hidden between sampled "
                "frames. Spatial predicates describe person relative to object. Return at most "
                f"{int(acquisition['maximum_events_per_clip'])} highest-confidence distinct events."
            )
            response_format = _response_format(
                visible, visible_persons, frame_ids,
                int(acquisition["maximum_events_per_clip"]),
            )
            core = {
                "protocol": "AGQA_QUESTION_BLIND_TYPED_EVENT_INVENTORY_V1",
                "protocol_file_sha256": _sha256(args.protocol),
                "video_id": video_id, "video_sha256": video["video_sha256"],
                "clip_id": clip_id, "presented_frames": frame_receipts,
                "visible_tracks": {track_id: track_labels[track_id] for track_id in visible},
                "panel_sha256s": panel_hashes, "public_predicates": PUBLIC_EVENT_PREDICATES,
                "model": model, "system_sha256": stable_hash(SYSTEM),
                "prompt_sha256": stable_hash(prompt),
                "request_contract": _request_cache_contract(
                    model=model,
                    max_tokens=int(acquisition["max_tokens"]),
                    response_format=response_format,
                    maximum_attempts=int(acquisition["maximum_contract_attempts"]),
                ),
            }
            call_name = f"qb_event_{video_id}_{clip_id}"
            provider_error = None
            rejected_events = ()
            try:
                payload, usage, reused = _cached_provider_call(
                    cache_dir=args.cache_dir, call_name=call_name, input_core=core,
                    invoke=lambda: _provider_call_with_contract_retries(
                        client, model=model, system=SYSTEM,
                        content=[{"type": "text", "text": prompt}] + _panel_content(panels),
                        max_tokens=int(acquisition["max_tokens"]),
                        response_format=response_format,
                        maximum_attempts=int(acquisition["maximum_contract_attempts"]),
                        validator=lambda candidate: _validate_event_envelope(
                            candidate,
                            int(acquisition["maximum_events_per_clip"]),
                        ),
                    ),
                )
                events, rejected_events = parse_question_blind_event_payload_with_rejections(
                    payload, clip_id=clip_id, visible_track_ids=visible,
                    person_track_ids=visible_persons, presented_frame_ids=frame_ids,
                    first_event_index=len(all_events),
                    track_visible_frames=track_visible_frames,
                )
            except Exception as exc:
                payload = {"events": []}
                usage = getattr(exc, "usage", {
                    "model": args.model, "finish_reason": "fail_closed",
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "reported_cost_usd": 0.0, "response_sha256": stable_hash(payload),
                    "provider_attempts": 0, "contract_retry_count": 0,
                    "attempt_receipts": [],
                })
                reused = False
                events = ()
                rejected_events = ()
                provider_error = f"{type(exc).__name__}:{exc}"
            all_events.extend(events)
            total_cost += float(usage.get("reported_cost_usd", 0.0))
            provider_calls += int(not reused) * int(usage.get("provider_attempts", 1))
            clip_rows.append({
                "clip_id": clip_id, "presented_frames": frame_receipts,
                "visible_track_ids": visible,
                "panel_sha256s": panel_hashes,
                "events": [event.as_dict() for event in events],
                "rejected_events": list(rejected_events),
                "provider_error": provider_error, "usage": usage,
                "cache_reused": reused,
            })
        deduplicated = deduplicate_question_blind_events(all_events)
        presented = {}
        for clip in clip_rows:
            for receipt in clip["presented_frames"]:
                presented[int(receipt["frame_id"])] = receipt
        outputs.append({
            "video_id": video_id, "video_sha256": str(video["video_sha256"]),
            "stable_tracks": [track.__dict__ for track in stable.tracks],
            "presented_frames": [presented[key] for key in sorted(presented)],
            "clips": clip_rows,
            "events_before_deduplication": len(all_events),
            "events": [event.as_dict() for event in deduplicated],
        })
        print(json.dumps({
            "video_id": video_id, "events": len(deduplicated),
            "clips_failed": sum(bool(row["provider_error"]) for row in clip_rows),
            "cost_usd_running": total_cost,
        }), flush=True)

    report = {
        "schema_version": "agqa-question-blind-typed-event-inventory-shard-v1",
        "status": _artifact_status(args.consumed_development_pilot),
        "consumed_development_pilot": bool(args.consumed_development_pilot),
        "collector_file_sha256": _sha256(Path(__file__)),
        "bbox_coordinate_contract": "SGDET_BOXES_ARE_NATIVE_XYXY_RENDER_CLAMP_ONLY",
        "model": model, "clip_count": args.clip_count,
        "frames_per_clip": args.frames_per_clip,
        "sampling_strategy": args.sampling_strategy,
        "maximum_unique_vlm_frames_per_video": min(
            int(sgdet["maximum_model_visible_frame_budget"]),
            args.clip_count * args.frames_per_clip,
        ),
        "minimum_object_score": args.minimum_object_score,
        "shard_count": args.shard_count, "shard_index": args.shard_index,
        "cohort_sha256": cohort["cohort_sha256"],
        "sgdet_report_sha256": sgdet["report_sha256"],
        "public_ontology_sha256": sgdet["ontology_sha256"],
        "protocol_file_sha256": _sha256(args.protocol),
        "rows": outputs, "provider_calls": provider_calls,
        "reported_cost_usd": total_cost,
        "accepted_event_proposals": sum(
            len(clip["events"]) for row in outputs for clip in row["clips"]
        ),
        "rejected_event_proposals": sum(
            len(clip.get("rejected_events", ()))
            for row in outputs for clip in row["clips"]
        ),
        "question_read": False, "answer_read": False,
        "official_scene_graph_read": False, "functional_program_read": False,
        "source_controller_read": False, "target_outcome_read": False,
        "per_video_action_genome_annotation_read": False,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
