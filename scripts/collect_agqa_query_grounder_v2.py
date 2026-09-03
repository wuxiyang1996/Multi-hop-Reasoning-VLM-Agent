#!/usr/bin/env python3
"""Collect shared, outcome-blind typed query grounding from raw AGQA videos."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import runpy

from openai import OpenAI
from PIL import ImageDraw

from motif_transfer.agqa_query_grounder_v2 import (
    EntityTrack, QueryCandidateEvidence, QueryGroundingV2Receipt, TypedRoleEvent,
    deduplicate_typed_events, requested_query_predicates, requested_query_role,
)
from motif_transfer.agqa_query_object_grounder import (
    AGQA_OBJECT_ONTOLOGY, AGQA_OBJECT_QUERY_TERMS, canonical_object_label,
)
from motif_transfer.agqa_open_vocabulary_grounder import detect_ontology_tracks
from motif_transfer.agqa_semantic_slots import action_anchor_obligations
from motif_transfer.contracts import stable_hash
from scripts.collect_agqa2_active_grounding_v3 import (
    _cached_provider_call, _panel_content, _provider_json_call,
    _operand_response_format, _operand_system,
    _panels as _timestamped_panels,
    _sample_video_range,
)
from motif_transfer.agqa_active_frame_grounder import parse_operand_receipt
from scripts.collect_agqa2_frame_grounding_v2 import _panels, _sample_video
from scripts.collect_agqa_layer_b_local_grounding import (
    _contains_forbidden_key, _frame_hash, _sha_file, _slot_prompt,
)
from scripts.evaluate_agqa_layer_b_five_arm import _semantic
from scripts.collect_agqa_layer_b_qwen235_grounding import provider_json_with_retries


INVENTORY_SYSTEM = """You are a frozen, answer-blind video entity tracking tool. Build an exhaustive entity inventory and typed event graph from chronological panels. Use only the supplied public ontology labels. Give the same physical entity a stable track_id across frames. For actions distinguish agent, patient/theme, destination, and instrument. For spatial relations distinguish relation_subject and relation_object: predicate(subject, object) always means relation_subject is in the named relation to relation_object, never the reverse. Include short or partially occluded entities when visibly supported, but never infer an answer. Bind events only to supplied perceptual semantic slot IDs. Do not output an answer, selected candidate, functional program, official scene graph, source game, controller, or correctness."""

CANDIDATE_SYSTEM = """You are a frozen, answer-blind single-candidate video verifier. You receive exactly one tracked entity and one requested semantic role. Decide only whether that entity visibly fills that role at the exact temporal, ordinal, reference, and relation scope described by the question. A red detector box is only a fallible proposal: first verify from pixels that it encloses the named category, otherwise return UNKNOWN. Preserve event semantics from the literal question: go/move beneath requires evidence of the person entering or occupying the candidate-specific beneath relation, touch requires visible contact, carry requires the same object to move with the person, washing requires repeated cleaning contact, and a purely stative wording requires only the named state. Generic relations that happen to hold (for example floor beneath a person) do not satisfy a question asking what the person went beneath. Use SUPPORTED only for direct candidate-specific evidence, never for plausibility or room layout. Directed relation notation subject --predicate--> object is literal and must never be reversed. You never see competing candidates and must not answer the question. SUPPORTED and REFUTED require cited frames; use UNKNOWN when pixels are insufficient. Return JSON only."""

TRACKING_ONTOLOGY = ("person",) + AGQA_OBJECT_ONTOLOGY

ANCHOR_SYSTEM = """You are an answer-blind temporal action localizer. Locate only the supplied action anchor in chronological video panels. Report every directly visible occurrence with tight onset/end intervals. Do not identify the queried object, answer the question, infer an official scene graph, or emit a program. Use UNKNOWN rather than guessing. Return JSON only."""


def _anchor_format(frame_count: int) -> dict:
    occurrence = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "start_frame": {"type": "integer", "minimum": 0, "maximum": frame_count - 1},
            "end_frame": {"type": "integer", "minimum": 0, "maximum": frame_count - 1},
            "evidence_frames": {"type": "array", "minItems": 1, "maxItems": 4,
                                "items": {"type": "integer", "minimum": 0, "maximum": frame_count - 1}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["start_frame", "end_frame", "evidence_frames", "confidence"],
    }
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["OBSERVED", "UNKNOWN"]},
            "occurrences": {"type": "array", "maxItems": 4, "items": occurrence},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["status", "occurrences", "uncertainties"],
    }
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_query_action_anchor_v2", "strict": True, "schema": schema,
    }}


def _temporal_scope(semantic) -> str:
    by_id = {slot.slot_id: slot for slot in semantic.slots}
    for slot in semantic.slots:
        if slot.kind != "TEMPORAL_CONSTRAINT":
            continue
        for child in slot.children:
            value = by_id[child].surface.casefold().strip()
            if value in {"before", "after", "while"}:
                return value
    return "video"


def _localized_window(*, scope: str, anchor_payload: dict, anchor_seconds: list[float],
                      duration: float) -> tuple[float, float, dict]:
    valid = []
    for row in anchor_payload.get("occurrences", ()):
        try:
            start, end = int(row["start_frame"]), int(row["end_frame"])
            evidence = [int(x) for x in row["evidence_frames"]]
            confidence = float(row["confidence"])
            if 0 <= start <= end < len(anchor_seconds) and evidence and confidence >= 0.5:
                valid.append((confidence, start, end, evidence))
        except (KeyError, TypeError, ValueError):
            continue
    if not valid or scope == "video":
        return 0.0, duration, {"status": "FULL_VIDEO_FAIL_CLOSED", "scope": scope}
    confidence, start_i, end_i, evidence = max(valid)
    start, end = anchor_seconds[start_i], anchor_seconds[end_i]
    span = max(6.0, min(15.0, duration * 0.45))
    if scope == "before":
        low, high = max(0.0, start - span), min(duration, start + 0.5)
    elif scope == "after":
        low, high = max(0.0, end - 0.5), min(duration, end + span)
    else:
        low, high = max(0.0, start - 1.5), min(duration, end + 1.5)
    if high <= low:
        low, high = 0.0, duration
    return low, high, {"status": "ANCHOR_LOCALIZED", "scope": scope,
                       "anchor_confidence": confidence, "anchor_frames": evidence,
                       "anchor_start_second": start, "anchor_end_second": end}


def _merge_frame_views(global_frames, global_seconds, local_frames, local_seconds):
    rows = {}
    for frame, second in list(zip(global_frames, global_seconds)) + list(zip(local_frames, local_seconds)):
        rows.setdefault(round(float(second), 3), frame)
    seconds = sorted(rows)
    frames = [rows[x] for x in seconds]
    local_to_merged = tuple(min(range(len(seconds)), key=lambda j: abs(seconds[j] - float(value)))
                            for value in local_seconds)
    return frames, seconds, local_to_merged


def _remap_inventory(tracks, events, mapping, frame_count):
    remapped_tracks = tuple(EntityTrack(
        row.track_id, row.canonical_label, row.aliases,
        tuple(sorted({mapping[x] for x in row.evidence_frames})), row.confidence,
    ) for row in tracks)
    remapped_events = tuple(TypedRoleEvent(
        row.event_id, row.predicate, row.roles, mapping[row.start_frame], mapping[row.end_frame],
        tuple(sorted({mapping[x] for x in row.evidence_frames})), row.confidence,
        row.semantic_slot_ids,
    ) for row in events)
    for row in remapped_tracks:
        row.validate(frame_count)
    return remapped_tracks, deduplicate_typed_events(remapped_events)


def _visual_query(*, question: str, predicates: tuple[str, ...], scope: str,
                  anchors: tuple[tuple[str, str], ...]) -> str:
    predicate = predicates[0] if predicates else "related to"
    lower = question.casefold()
    if f"go {predicate}" in lower or f"went {predicate}" in lower:
        relation = f"goes {predicate}"
    else:
        relation = predicate
    value = f"a person {relation} an unknown object"
    if scope in {"before", "after", "while"} and anchors:
        value += f" {scope} {anchors[0][0]}"
    return value


def _operand_payload(receipt, *, requested_role: str, slot_ids: list[str]) -> dict:
    observations = [row for row in receipt.observations
                    if row.observability in {"OBSERVED", "PARTIAL"} and row.evidence_frames]
    all_evidence = tuple(sorted({x for row in observations for x in row.evidence_frames})) or (0,)
    tracks = [{"track_id": "T0", "canonical_label": "person", "aliases": [],
               "evidence_frames": list(all_evidence[:6]), "confidence": 1.0}]
    events = []
    for row in observations:
        label = canonical_object_label(row.object)
        if label not in TRACKING_ONTOLOGY or label == "person":
            continue
        track_id = f"T{len(tracks)}"
        evidence = tuple(sorted(set(row.evidence_frames)))
        tracks.append({"track_id": track_id, "canonical_label": label,
                       "aliases": [] if label == row.object.casefold().strip() else [row.object],
                       "evidence_frames": list(evidence[:6]), "confidence": row.confidence})
        events.append({"event_id": f"R{len(events)}", "predicate": row.predicate,
                       "roles": [{"role": "agent", "track_id": "T0"},
                                 {"role": requested_role, "track_id": track_id}],
                       "start_frame": row.start_frame if row.start_frame is not None else min(evidence),
                       "end_frame": row.end_frame if row.end_frame is not None else max(evidence),
                       "evidence_frames": list(evidence), "confidence": row.confidence,
                       "semantic_slot_ids": slot_ids})
    return {"tracks": tracks, "events": events,
            "uncertainties": list(receipt.uncertainties)}


def _candidate_overlay_panels(frames, seconds, detections, *, label: str,
                              frames_per_panel: int, frame_width: int):
    marked = [frame.copy() for frame in frames]
    for row in detections:
        if row.label != label:
            continue
        draw = ImageDraw.Draw(marked[row.frame_index])
        x1, y1, x2, y2 = row.bbox_xyxy
        draw.rectangle((x1, y1, x2, y2), outline="red", width=5)
        draw.text((max(0, x1), max(0, y1 - 14)), label, fill="red")
    return _timestamped_panels(marked, seconds, frames_per_panel=frames_per_panel,
                               frame_width=frame_width, quality=82)


def _inventory_format(frame_count: int, slot_ids: list[str]) -> dict:
    track = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "track_id": {"type": "string"},
            "canonical_label": {"type": "string", "enum": list(TRACKING_ONTOLOGY)},
            "aliases": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
            "evidence_frames": {"type": "array", "minItems": 1, "maxItems": 6,
                                "items": {"type": "integer", "minimum": 0, "maximum": frame_count - 1}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["track_id", "canonical_label", "aliases", "evidence_frames", "confidence"],
    }
    role = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "role": {"type": "string", "enum": [
                "agent", "patient", "theme", "destination", "instrument",
                "relation_subject", "relation_object",
            ]},
            "track_id": {"type": "string"},
        }, "required": ["role", "track_id"],
    }
    event = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "event_id": {"type": "string"}, "predicate": {"type": "string"},
            "roles": {"type": "array", "minItems": 1, "maxItems": 7, "items": role},
            "start_frame": {"type": "integer", "minimum": 0, "maximum": frame_count - 1},
            "end_frame": {"type": "integer", "minimum": 0, "maximum": frame_count - 1},
            "evidence_frames": {"type": "array", "minItems": 1, "maxItems": 3,
                                "items": {"type": "integer", "minimum": 0, "maximum": frame_count - 1}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "semantic_slot_ids": {"type": "array", "minItems": 1, "maxItems": 6,
                                  "items": {"type": "string", "enum": slot_ids}},
        },
        "required": ["event_id", "predicate", "roles", "start_frame", "end_frame",
                     "evidence_frames", "confidence", "semantic_slot_ids"],
    }
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "tracks": {"type": "array", "maxItems": 12, "items": track},
            "events": {"type": "array", "maxItems": 24, "items": event},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        }, "required": ["tracks", "events", "uncertainties"],
    }
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_query_typed_grounding_v2", "strict": True, "schema": schema,
    }}


def _split_inventory_formats(frame_count: int, slot_ids: list[str]) -> tuple[dict, dict]:
    tracks = copy.deepcopy(_inventory_format(frame_count, slot_ids))
    events = copy.deepcopy(tracks)
    tracks["json_schema"]["name"] = "agqa_query_entity_tracks_v2"
    tracks["json_schema"]["schema"]["properties"].pop("events")
    tracks["json_schema"]["schema"]["required"].remove("events")
    events["json_schema"]["name"] = "agqa_query_typed_events_v2"
    events["json_schema"]["schema"]["properties"].pop("tracks")
    events["json_schema"]["schema"]["required"].remove("tracks")
    return tracks, events


def _candidate_format(frame_count: int) -> dict:
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["SUPPORTED", "REFUTED", "UNKNOWN"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_frames": {"type": "array", "maxItems": 4,
                                "items": {"type": "integer", "minimum": 0, "maximum": frame_count - 1}},
            "rationale": {"type": "string"},
        }, "required": ["status", "confidence", "evidence_frames", "rationale"],
    }
    return {"type": "json_schema", "json_schema": {
        "name": "agqa_query_single_candidate_v2", "strict": True, "schema": schema,
    }}


def _parse_inventory(payload: dict, *, frame_count: int, slot_ids: frozenset[str]):
    tracks = []
    for index, row in enumerate(payload.get("tracks", ())):
        label = canonical_object_label(row["canonical_label"])
        if label not in TRACKING_ONTOLOGY:
            continue
        aliases = tuple(dict.fromkeys(
            value for value in (canonical_object_label(x) for x in row.get("aliases", ()))
            if value and value != label
        ))
        track = EntityTrack(
            f"T{len(tracks)}", label, aliases,
            tuple(sorted(set(int(x) for x in row["evidence_frames"]))), float(row["confidence"]),
        )
        track.validate(frame_count); tracks.append(track)
    model_to_local = {
        str(row.get("track_id")): track.track_id
        for row, track in zip(payload.get("tracks", ()), tracks)
    }
    known = frozenset(track.track_id for track in tracks); events = []
    for row in payload.get("events", ()):
        roles = tuple((str(x["role"]), model_to_local.get(str(x["track_id"]), ""))
                      for x in row["roles"])
        bindings = tuple(dict.fromkeys(str(x) for x in row["semantic_slot_ids"] if str(x) in slot_ids))
        try:
            event = TypedRoleEvent(
                f"R{len(events)}", str(row["predicate"]), roles,
                int(row["start_frame"]), int(row["end_frame"]),
                tuple(sorted(set(int(x) for x in row["evidence_frames"]))),
                float(row["confidence"]), bindings,
            )
            event.validate(frame_count, known); events.append(event)
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(tracks), deduplicate_typed_events(events)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=Path, required=True); p.add_argument("--semantic-runtime", type=Path, required=True)
    p.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    p.add_argument("--output", type=Path, required=True); p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--model", default="qwen/qwen3-vl-32b-instruct")
    p.add_argument("--frame-count", type=int, default=96); p.add_argument("--max-tokens", type=int, default=2400)
    p.add_argument("--anchor-frame-count", type=int, default=0,
                   help="Global anchor frames; remaining frame budget is used for localized acquisition")
    p.add_argument("--inventory-response-mode", choices=("json_schema", "json_object"),
                   default="json_schema")
    p.add_argument("--localized-frames-per-panel", type=int, default=4)
    p.add_argument("--localized-panel-frame-width", type=int, default=224)
    p.add_argument("--split-event-call", action="store_true")
    p.add_argument("--operand-inventory-call", action="store_true",
                   help="Use the provider-compatible isolated-operand schema to construct query-local tracks/events")
    p.add_argument("--open-vocab-inventory", action="store_true")
    p.add_argument("--detector-inspection-frames", type=int, default=8)
    p.add_argument("--detector-box-threshold", type=float, default=.12)
    p.add_argument("--detector-ontology-chunk-size", type=int, default=6)
    p.add_argument("--positions", default="all")
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError("V2 query grounding output is immutable")
    cohort = json.loads(args.cohort.read_text()); runtime = json.loads(args.semantic_runtime.read_text())
    if cohort["cohort_sha256"] != runtime["cohort_sha256"]: raise ValueError("cohort/runtime mismatch")
    semantics = {str(x["task_id"]): _semantic(x["receipt"]) for x in runtime["rows"]}
    query_rows = [(i, row) for i, row in enumerate(cohort["rows"]) if row.get("structural") == "query"]
    if args.positions != "all":
        selected = {int(x) for x in args.positions.replace(":", ",").split(",") if x.strip()}
        query_rows = [x for j, x in enumerate(query_rows) if j in selected]
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key: raise ValueError("OpenRouter API key unavailable")
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=300, max_retries=2)
    model = {"id": args.model, "omit_temperature": True}
    ontology_sha = stable_hash(list(TRACKING_ONTOLOGY))
    backend = stable_hash({"protocol": "AGQA_QUERY_GROUNDER_V2", "model": model,
                           "inventory_system": INVENTORY_SYSTEM, "candidate_system": CANDIDATE_SYSTEM,
                           "anchor_system": ANCHOR_SYSTEM if args.anchor_frame_count else None,
                           "ontology_sha256": ontology_sha, "frame_count": args.frame_count,
                           "anchor_frame_count": args.anchor_frame_count,
                           "inventory_response_mode": args.inventory_response_mode,
                           "split_event_call": args.split_event_call,
                           "operand_inventory_call": args.operand_inventory_call,
                           "open_vocab_inventory": args.open_vocab_inventory,
                           "detector": ({"model": "IDEA-Research/grounding-dino-base",
                                         "inspection_frames": args.detector_inspection_frames,
                                         "box_threshold": args.detector_box_threshold,
                                         "ontology_chunk_size": args.detector_ontology_chunk_size}
                                        if args.open_vocab_inventory else None),
                           "localized_frames_per_panel": args.localized_frames_per_panel,
                           "localized_panel_frame_width": args.localized_panel_frame_width})
    outputs = []; total_cost = 0.0
    for position, public in query_rows:
        task_id = str(public["task_id"]); semantic = semantics[task_id]
        try: role = requested_query_role(semantic)
        except ValueError: role = None
        predicates = requested_query_predicates(semantic)
        anchor_payload = {}; anchor_usage = {"reported_cost_usd": 0.0}; anchor_reused = False
        anchor_error = None; localization = {"status": "DISABLED", "scope": "video"}
        anchors = action_anchor_obligations(semantic); scope = _temporal_scope(semantic)
        if args.anchor_frame_count:
            if not 0 < args.anchor_frame_count < args.frame_count:
                raise ValueError("anchor-frame-count must be between zero and frame-count")
            global_frames, global_seconds, global_metadata = _sample_video_range(
                Path(public["video_path"]), frame_count=args.anchor_frame_count, max_side=448)
            global_panels = _timestamped_panels(
                global_frames, global_seconds, frames_per_panel=4, frame_width=224, quality=82)
            if anchors:
                anchor_phrase = " AND ".join(value for value, _ in anchors)
                acore = {"protocol": "AGQA_QUERY_ACTION_ANCHOR_V2", "task_id": task_id,
                         "anchor_phrase": anchor_phrase, "model": model,
                         "panels": [hashlib.sha256(x).hexdigest() for x in global_panels]}
                try:
                    anchor_payload, anchor_usage, anchor_reused = _cached_provider_call(
                        cache_dir=args.cache_dir, call_name=f"anchor_{task_id}", input_core=acore,
                        invoke=lambda: provider_json_with_retries(
                            client, model=model, system=ANCHOR_SYSTEM,
                            content=[{"type": "text", "text": f"action_anchor: {anchor_phrase}\nLocate only this anchor."}]
                                    + _panel_content(global_panels),
                            max_tokens=650, response_format=_anchor_format(args.anchor_frame_count), attempts=3))
                    total_cost += float(anchor_usage.get("reported_cost_usd", 0.0))
                except Exception as exc:
                    anchor_error = f"{type(exc).__name__}:{exc}"
                    anchor_payload = {"status": "UNKNOWN", "occurrences": [], "uncertainties": [anchor_error]}
            low, high, localization = _localized_window(
                scope=scope, anchor_payload=anchor_payload, anchor_seconds=global_seconds,
                duration=float(global_metadata["duration_seconds"]))
            local_count = args.frame_count - args.anchor_frame_count
            local_frames, local_seconds, local_metadata = _sample_video_range(
                Path(public["video_path"]), frame_count=local_count, max_side=448,
                start_second=low, end_second=high)
            frames, seconds, local_mapping = _merge_frame_views(
                global_frames, global_seconds, local_frames, local_seconds)
            metadata = {"global": global_metadata, "localized": local_metadata,
                        "localization": localization, "unique_frame_count": len(frames)}
            panels = _timestamped_panels(
                local_frames, local_seconds,
                frames_per_panel=args.localized_frames_per_panel,
                frame_width=args.localized_panel_frame_width, quality=82)
            inventory_frame_count = local_count
        else:
            frames, seconds, metadata = _sample_video(
                Path(public["video_path"]), frame_count=args.frame_count, max_side=448)
            panels = _panels(
                frames, seconds, {"frames_per_panel": 6, "panel_frame_width": 224, "jpeg_quality": 82})
            local_frames, local_seconds = frames, seconds
            local_mapping = tuple(range(len(frames))); inventory_frame_count = len(frames)
        hashes = tuple(_frame_hash(frame) for frame in frames)
        panel_hashes = [hashlib.sha256(x).hexdigest() for x in panels]
        perceptual = [s.slot_id for s in semantic.slots if s.kind in {"LITERAL", "ENTITY", "ACTION", "RELATION"}]
        prompt = (f"Question is perceptual context only; never answer it:\n{public['question']}\n\n"
                  f"Frozen operator-free semantic slots:\n{_slot_prompt(semantic)}\n\n"
                  f"Temporal acquisition scope: {scope}; action anchors: {[x for x, _ in anchors]}; "
                  "the supplied panels are the outcome-blind localized search window. Exhaustively list every "
                  "visible ontology entity that could fill the requested typed role, including small or brief objects.\n"
                  f"Explicit operator-free target predicates: {list(predicates)}. Preserve whether the literal "
                  "question asks for a transition/action (go, move, touch, carry) or a stative relation; emit the "
                  "question-scoped typed event rather than every incidental spatial relation.\n"
                  "Return one JSON object with tracks, events, and uncertainties arrays. Each track has track_id, "
                  "canonical_label, aliases, evidence_frames, confidence. Each event has event_id, predicate, roles "
                  "as role/track_id objects, start_frame, end_frame, evidence_frames, confidence, semantic_slot_ids.\n"
                  f"Public tracking ontology: {', '.join(TRACKING_ONTOLOGY)}")
        core = {"protocol": "AGQA_QUERY_INVENTORY_V2", "task_id": task_id, "model": model,
                "question_sha256": public["question_sha256"], "semantic_sha256": semantic.receipt_sha256,
                "panels": panel_hashes, "ontology_sha256": ontology_sha, "frame_count": args.frame_count}
        event_payload = None; event_error = None; detector_detections = ()
        provider_base_calls = 1
        try:
            split_formats = _split_inventory_formats(inventory_frame_count, perceptual)
            if args.open_vocab_inventory:
                inspection = tuple(dict.fromkeys(round(i * (inventory_frame_count - 1)
                                                          / max(1, args.detector_inspection_frames - 1))
                                                   for i in range(args.detector_inspection_frames)))
                detected_tracks, detector_detections = detect_ontology_tracks(
                    local_frames, frame_indices=inspection, ontology=TRACKING_ONTOLOGY,
                    query_terms=("person",) + AGQA_OBJECT_QUERY_TERMS,
                    box_threshold=args.detector_box_threshold,
                    text_threshold=args.detector_box_threshold, maximum_tracks=12,
                    ontology_chunk_size=args.detector_ontology_chunk_size)
                payload = {"tracks": [asdict(row) for row in detected_tracks],
                           "events": [], "uncertainties": []}
                usage = {"reported_cost_usd": 0.0, "model": "grounding-dino-base-local"}
                reused = False; provider_base_calls = 0
            elif args.operand_inventory_call:
                requested = _visual_query(
                    question=public["question"], predicates=predicates,
                    scope=scope, anchors=anchors)
                operand_core = {**core, "protocol": "AGQA_QUERY_ISOLATED_OPERAND_INVENTORY_V2",
                                "requested_operand": requested}
                raw_operand, usage, reused = _cached_provider_call(
                    cache_dir=args.cache_dir, call_name=f"operand_inventory_{task_id}",
                    input_core=operand_core,
                    invoke=lambda: provider_json_with_retries(
                        client, model=model, system=_operand_system(inventory_frame_count),
                        content=[{"type": "text", "text": (
                            f"operand_role: A\nrequested_operand: {requested}\n"
                            "grounding_mode: QUERY_OBJECT_LOCALIZED\nGround only this operand."
                        )}] + _panel_content(panels),
                        max_tokens=1000, response_format=_operand_response_format(inventory_frame_count),
                        attempts=3))
                parsed_operand = parse_operand_receipt(
                    raw_operand, expected_role="A", expected_operand=requested,
                    frame_count=inventory_frame_count)
                payload = _operand_payload(
                    parsed_operand, requested_role=role or "patient", slot_ids=perceptual)
            else:
                payload, usage, reused = _cached_provider_call(
                    cache_dir=args.cache_dir, call_name=f"inventory_{task_id}", input_core=core,
                    invoke=lambda: provider_json_with_retries(
                        client, model=model, system=INVENTORY_SYSTEM,
                        content=[{"type": "text", "text": prompt}] + _panel_content(panels),
                        max_tokens=args.max_tokens,
                        response_format=(split_formats[0] if args.split_event_call
                                         else _inventory_format(inventory_frame_count, perceptual)
                                         if args.inventory_response_mode == "json_schema"
                                         else {"type": "json_object"}),
                        attempts=3,
                    ),
                )
            if _contains_forbidden_key(payload):
                raise ValueError("inventory payload emitted a forbidden authority field")
            if args.split_event_call and not args.operand_inventory_call:
                event_prompt = (
                    prompt + "\nFrozen entity tracks from the first perceptual pass:\n"
                    + json.dumps(payload.get("tracks", []), sort_keys=True)
                    + "\nEmit only typed question-scoped events whose roles reference those track_id values."
                )
                ecore = {**core, "protocol": "AGQA_QUERY_TYPED_EVENTS_SPLIT_V2",
                         "tracks_sha256": stable_hash(payload.get("tracks", []))}
                try:
                    event_payload, eusage, _ = _cached_provider_call(
                        cache_dir=args.cache_dir, call_name=f"events_{task_id}", input_core=ecore,
                        invoke=lambda: provider_json_with_retries(
                            client, model=model, system=INVENTORY_SYSTEM,
                            content=[{"type": "text", "text": event_prompt}] + _panel_content(panels),
                            max_tokens=1200, response_format=split_formats[1], attempts=3))
                    total_cost += float(eusage.get("reported_cost_usd", 0.0))
                    if _contains_forbidden_key(event_payload):
                        raise ValueError("event payload emitted a forbidden authority field")
                except Exception as exc:
                    event_error = f"{type(exc).__name__}:{exc}"
                    event_payload = {"events": [], "uncertainties": [event_error]}
                payload = {"tracks": payload.get("tracks", []),
                           "events": event_payload.get("events", []),
                           "uncertainties": list(payload.get("uncertainties", ()))
                                            + list(event_payload.get("uncertainties", ())) }
            tracks, events = _parse_inventory(
                payload, frame_count=inventory_frame_count, slot_ids=frozenset(perceptual))
            tracks, events = _remap_inventory(tracks, events, local_mapping, len(frames))
            total_cost += float(usage.get("reported_cost_usd", 0.0)); error = event_error
        except Exception as exc:
            tracks, events, payload, usage, reused = (), (), {}, {"reported_cost_usd": 0.0}, False
            error = f"{type(exc).__name__}:{exc}"
        decisions = []; candidate_payloads = {}
        if role is not None:
            for track in tracks:
                if track.canonical_label == "person":
                    continue
                directed_scope = (
                    f"Required directed relation: person --{predicates[0]}--> {track.canonical_label}. "
                    f"This means the person is {predicates[0]} the candidate, not that the candidate is "
                    f"{predicates[0]} the person."
                    if role == "relation_object" and predicates else
                    f"Required action role: person performs {list(predicates)} with the candidate as {role}."
                )
                candidate_prompt = (f"Question scope (do not answer): {public['question']}\nRequested role: {role}\n"
                                    f"Explicit operator-free target predicates: {list(predicates)}. The literal question "
                                    "wording controls whether this is an action/transition or a stative relation.\n"
                                    f"{directed_scope}\n"
                                    f"Single candidate track: {track.track_id}, label={track.canonical_label}, "
                                    f"aliases={list(track.aliases)}. Verify it in the supplied localized panels.")
                candidate_panels = (_candidate_overlay_panels(
                    local_frames, local_seconds, detector_detections,
                    label=track.canonical_label,
                    frames_per_panel=args.localized_frames_per_panel,
                    frame_width=args.localized_panel_frame_width)
                    if args.open_vocab_inventory else panels)
                candidate_panel_hashes = [hashlib.sha256(x).hexdigest() for x in candidate_panels]
                if args.open_vocab_inventory:
                    candidate_prompt += "\nRed boxes mark only this candidate class; use them as localization evidence."
                ccore = {"protocol": "AGQA_QUERY_SINGLE_CANDIDATE_V2", "task_id": task_id,
                         "track": asdict(track), "role": role,
                         "panels": candidate_panel_hashes, "model": model}
                try:
                    value, cusage, _ = _cached_provider_call(
                        cache_dir=args.cache_dir, call_name=f"candidate_{task_id}_{track.track_id}", input_core=ccore,
                        invoke=lambda: provider_json_with_retries(
                            client, model=model, system=CANDIDATE_SYSTEM,
                            content=[{"type": "text", "text": candidate_prompt}]
                                    + _panel_content(candidate_panels),
                            max_tokens=500, response_format=_candidate_format(inventory_frame_count), attempts=3,
                        ),
                    )
                    frames_out = tuple(sorted({local_mapping[int(x)] for x in value["evidence_frames"]}))
                    decision = QueryCandidateEvidence(track.track_id, role, str(value["status"]),
                                                      float(value["confidence"]), frames_out)
                    if (args.open_vocab_inventory and decision.status == "SUPPORTED"
                            and not set(decision.evidence_frames) & set(track.evidence_frames)):
                        decision = QueryCandidateEvidence(track.track_id, role, "UNKNOWN", 0.0, ())
                    decision.validate(len(frames), frozenset(x.track_id for x in tracks))
                    total_cost += float(cusage.get("reported_cost_usd", 0.0)); candidate_payloads[track.track_id] = value
                except Exception as exc:
                    decision = QueryCandidateEvidence(track.track_id, role, "UNKNOWN", 0.0, ())
                    candidate_payloads[track.track_id] = {"error": f"{type(exc).__name__}:{exc}"}
                decisions.append(decision)
        if args.open_vocab_inventory and role is not None:
            person = next((track for track in tracks if track.canonical_label == "person"), None)
            if person is not None:
                slot_bindings = tuple(perceptual)
                for decision in decisions:
                    if decision.status != "SUPPORTED" or not decision.evidence_frames:
                        continue
                    events += (TypedRoleEvent(
                        f"R{len(events)}", predicates[0] if predicates else "related to",
                        (("agent", person.track_id), (role, decision.track_id)),
                        min(decision.evidence_frames), max(decision.evidence_frames),
                        decision.evidence_frames, decision.confidence, slot_bindings,
                    ),)
                events = deduplicate_typed_events(events)
        receipt = QueryGroundingV2Receipt.create(
            task_id=task_id, video_sha256=_sha_file(Path(public["video_path"])),
            semantic_slots_sha256=semantic.receipt_sha256, selected_frame_indices=tuple(range(len(frames))),
            selected_frame_sha256s=hashes, tracks=tracks, events=events, candidates=decisions,
            public_ontology_sha256=ontology_sha, grounder_backend_sha256=backend,
            provider_calls=provider_base_calls + len(decisions) + int(bool(args.anchor_frame_count and anchors))
                           + int(args.split_event_call and not args.operand_inventory_call),
        )
        outputs.append({"cohort_position": position, "task_id": task_id, "video_id": public["video_id"],
                        "requested_role": role, "receipt": asdict(receipt), "inventory_payload": payload,
                        "candidate_payloads": candidate_payloads, "provider_error": error,
                        "anchor_payload": anchor_payload, "anchor_usage": anchor_usage,
                        "event_payload": event_payload, "event_error": event_error,
                        "detector_detections": [asdict(row) for row in detector_detections],
                        "anchor_cache_reused": anchor_reused, "anchor_error": anchor_error,
                        "localization": localization, "video_metadata": metadata,
                        "panel_sha256s": panel_hashes})
        print(json.dumps({"task_id": task_id, "tracks": len(tracks), "events": len(events),
                          "supported": sum(x.status == 'SUPPORTED' for x in decisions)}), flush=True)
    body = {"schema_version": "agqa-query-grounder-v2-development-v1",
            "status": "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME",
            "cohort_sha256": cohort["cohort_sha256"], "semantic_runtime_sha256": runtime["runtime_sha256"],
            "public_ontology_sha256": ontology_sha, "grounder_backend_sha256": backend,
            "model": args.model, "frame_budget": args.frame_count,
            "anchor_frame_budget": args.anchor_frame_count, "rows": outputs,
            "reported_receipt_provider_cost_usd": total_cost, "all_harness_arms_share_exact_receipts": True,
            "answer_read": False, "official_scene_graph_read": False, "functional_program_read": False,
            "source_controller_read": False, "target_outcome_read": False}
    body["report_sha256"] = stable_hash(body); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(outputs), "cost_usd": total_cost, "report_sha256": body["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
