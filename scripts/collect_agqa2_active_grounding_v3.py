#!/usr/bin/env python3
"""Collect and evaluate two-stage AGQA V3 neural-symbolic receipts."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import sys
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_active_frame_grounder import (  # noqa: E402
    AGQAOperandReceipt,
    AGQAQueryPlan,
    AGQASourceAcquisitionController,
    choose_operand_receipt,
    merge_operand_receipts,
    operand_needs_rescan,
    parse_operand_receipt,
    parse_public_question_plan,
    parse_query_plan,
    reconcile_recurrent_receipts,
    recurrent_rescan_window,
    remap_operand_receipt,
    source_controller_for_plan,
)
from motif_transfer.agqa_frame_grounder import (  # noqa: E402
    execute_grounding_receipt,
    parse_frame_grounding_receipt,
    select_source_for_grounding,
)
from motif_transfer.agqa_local_object_grounder import (  # noqa: E402
    detect_objects,
    inspection_indices,
    refine_query_object_receipt,
)
from motif_transfer.agqa_program_transfer import (  # noqa: E402
    RELATION_ROUTE,
    TEMPORAL_PAIR_ROUTE,
    TEMPORAL_SINGLE_ROUTE,
    profile_program,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_ir_applicability import (  # noqa: E402
    SourceIRContract,
)
from scripts.audit_agqa2_program_transfer_v1 import (  # noqa: E402
    _iter_top_level_object,
    _load_sources,
    _target_written_equivalent,
)


ROUTES = (RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE)
PROMPT_VERSION = "AGQA_OPERAND_ISOLATED_ACTIVE_GROUNDER_V3_0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("provider response omitted a JSON object")
    payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("provider JSON response must be an object")
    return payload


def _provider_json_call(
    client: OpenAI, *, model: Mapping[str, Any], system: str,
    content: list[dict[str, Any]], max_tokens: int,
    response_format: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    extra_body: dict[str, Any] = {"provider": {"require_parameters": True}}
    reasoning = model.get("reasoning")
    if reasoning:
        extra_body["reasoning"] = dict(reasoning)
    response = None
    last_transport_error = ""
    # OpenAI-compatible routers can rarely return a nominally successful
    # envelope with null choices. Retry the identical request; never cache it.
    for _ in range(3):
        try:
            candidate = client.chat.completions.create(
                model=str(model["id"]), temperature=0, max_tokens=max_tokens,
                response_format=dict(response_format),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                extra_body=extra_body,
            )
            if not getattr(candidate, "choices", None):
                raise TypeError("provider response omitted choices")
            choice = candidate.choices[0]
            if getattr(choice, "message", None) is None:
                raise TypeError("provider response omitted assistant message")
            response = candidate
            break
        except (AttributeError, IndexError, TypeError) as exc:
            last_transport_error = f"{type(exc).__name__}: {exc}"
    if response is None:
        raise RuntimeError(
            "provider response transport retries exhausted: "
            + last_transport_error
        )
    raw = response.choices[0].message.content or ""
    payload = _decode_json_object(raw)
    usage = response.usage
    usage_extra = getattr(usage, "model_extra", None) or {}
    return payload, {
        "model": str(response.model),
        "finish_reason": str(response.choices[0].finish_reason),
        "prompt_tokens": int(usage.prompt_tokens if usage else 0),
        "completion_tokens": int(usage.completion_tokens if usage else 0),
        "reported_cost_usd": float(
            getattr(usage, "cost", 0.0) or usage_extra.get("cost", 0.0) or 0.0
        ),
        "response_sha256": stable_hash(payload),
    }


def _cached_provider_call(
    *, cache_dir: Path, call_name: str, input_core: Mapping[str, Any],
    invoke: Any,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    input_sha = stable_hash(input_core)
    path = cache_dir / f"{call_name}.{input_sha[:16]}.json"
    if path.is_file():
        cached = json.loads(path.read_text())
        if cached.get("input_sha256") == input_sha:
            return cached["payload"], cached["usage"], True
    payload, usage = invoke()
    body = {
        "input_sha256": input_sha,
        "payload": payload,
        "usage": usage,
    }
    body["call_receipt_sha256"] = stable_hash(body)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return payload, usage, False


def _query_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agqa_query_plan_v3",
            "strict": True,
            "schema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "obligation_kind": {"type": "string", "enum": list(ROUTES)},
                    "comparison": {"type": "string", "enum": [
                        "EXISTS", "QUERY_OBJECT", "CHOOSE_OBJECT", "BEFORE_AFTER",
                        "SELECT_LONGER", "SELECT_SHORTER", "VERIFY_A_LONGER",
                        "VERIFY_A_SHORTER",
                    ]},
                    "operand_a": {"type": "string"},
                    "operand_b": {"type": "string"},
                    "visual_query_a": {"type": "string"},
                    "visual_query_b": {"type": "string"},
                    "parser_uncertainties": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
                "required": [
                    "obligation_kind", "comparison", "operand_a", "operand_b",
                    "visual_query_a", "visual_query_b", "parser_uncertainties",
                ],
            },
        },
    }


def _query_system() -> str:
    return (
        "You parse a public video question into a typed plan; you never answer it. "
        "Return only the requested JSON. Use RELATION_RECURRENT for questions asking "
        "whether a person-object relation/action exists, which object participates, "
        "or which of two objects participates. Use TEMPORAL_PAIR_RECURRENT only for "
        "before/after event ordering. Use TEMPORAL_SINGLE_NONRECURRENT for every "
        "longer/shorter duration question. comparison is EXISTS, QUERY_OBJECT, or "
        "CHOOSE_OBJECT for relation; BEFORE_AFTER for ordering; SELECT_LONGER or "
        "SELECT_SHORTER when the answer is an event phrase; VERIFY_A_LONGER or "
        "VERIFY_A_SHORTER when the answer is yes/no. For before/after, operand A MUST "
        "be the grammatical event whose ordering the question asks about and operand B "
        "the reference event: 'Was opening a laptop before or after sitting down?' "
        "means A=opening a laptop, B=sitting down. For duration verification, A is the "
        "event explicitly claimed longer/shorter and B its comparator. For "
        "CHOOSE_OBJECT, operand_a/operand_b are candidates in mention order, but "
        "visual_query_a MUST describe one candidate-blind relation with 'an unknown "
        "object' and visual_query_b MUST be empty. For "
        "QUERY_OBJECT, operand_a is the relation phrase, operand_b and visual_query_b "
        "are empty, and visual_query_a explicitly says 'an unknown object'. For every "
        "other type visual_query_a/b are standalone visible event descriptions. Do "
        "not emit an answer, selected operand, program, scene graph, source, or choice."
    )


def _operand_response_format(frame_count: int) -> dict[str, Any]:
    frame_or_null = {"anyOf": [
        {"type": "integer", "minimum": 0, "maximum": frame_count - 1},
        {"type": "null"},
    ]}
    observation = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "occurrence_id": {"type": "string"},
            "label": {"type": "string"},
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
            "observability": {"type": "string", "enum": [
                "OBSERVED", "PARTIAL", "UNOBSERVED",
            ]},
            "start_frame": frame_or_null,
            "end_frame": frame_or_null,
            "evidence_frames": {
                "type": "array", "maxItems": 4,
                "items": {"type": "integer", "minimum": 0,
                          "maximum": frame_count - 1},
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "occurrence_id", "label", "subject", "predicate", "object",
            "observability", "start_frame", "end_frame", "evidence_frames",
            "confidence", "uncertainties",
        ],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agqa_isolated_operand_v3", "strict": True,
            "schema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "operand_role": {"type": "string", "enum": ["A", "B"]},
                    "requested_operand": {"type": "string"},
                    "observations": {
                        "type": "array", "minItems": 1, "maxItems": 4,
                        "items": observation,
                    },
                    "coverage": {"type": "string", "enum": [
                        "SUFFICIENT", "PARTIAL", "INSUFFICIENT",
                    ]},
                    "uncertainties": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
                "required": [
                    "operand_role", "requested_operand", "observations",
                    "coverage", "uncertainties",
                ],
            },
        },
    }


def _operand_system(frame_count: int) -> str:
    return (
        "You are an operand-isolated video grounder. You receive exactly one target "
        "event/relation and chronological frame panels, never the original question, "
        "a competing operand, an answer, a program, a scene graph, or source identity. "
        "Return only the requested JSON. Copy operand_role and requested_operand "
        "exactly. Emit one row per visually supported occurrence, at most four, with "
        "consecutive IDs O0,O1,... . Use a tight interval from first direct onset "
        "evidence through last sustained evidence. For holding/sitting/standing states, "
        "duration includes only frames directly showing the state, not nearby inferred "
        "offscreen time. For QUERY_OBJECT-like phrases containing 'unknown object', "
        "put only the directly related object noun in object; distinguish furniture "
        "such as chair, bed, sofa, table by contact/support, not mere proximity. "
        "For a fixed candidate relation, object must be that candidate only when the "
        "requested relation is visible. OBSERVED requires evidence frames and interval; "
        "UNOBSERVED has null endpoints and no evidence. If no occurrence is visible, "
        "still emit O0 as UNOBSERVED. Evidence indices must be chronological and inside "
        f"the interval, using integers F0..F{frame_count - 1}. Do not answer or compare."
    )


def _direct_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agqa_matched_direct_v3", "strict": True,
            "schema": {
                "type": "object", "additionalProperties": False,
                "properties": {"response": {"type": "string"}},
                "required": ["response"],
            },
        },
    }


def _sample_video_range(
    path: Path, *, frame_count: int, max_side: int,
    start_second: float | None = None, end_second: float | None = None,
) -> tuple[list[Image.Image], list[float], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        capture.release()
        raise RuntimeError(f"invalid video metadata: {path}")
    duration = total / fps
    first = max(0.0, float(start_second or 0.0))
    last = min(
        max(0.0, duration - 1.0 / fps),
        float(end_second) if end_second is not None else duration,
    )
    if last <= first:
        last = min(max(0.0, duration - 1.0 / fps), first + 1.0 / fps)
    seconds = np.linspace(first, last, frame_count)
    frames = []
    for second in seconds:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000)
        ok, bgr = capture.read()
        if not ok or bgr is None:
            capture.release()
            raise RuntimeError(f"failed decoding {path} at {second:.3f}s")
        frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        frame.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        frames.append(frame)
    capture.release()
    rounded = [round(float(second), 3) for second in seconds]
    return frames, rounded, {
        "source_fps": fps,
        "source_frame_count": total,
        "duration_seconds": duration,
        "sample_start_second": first,
        "sample_end_second": last,
        "proxy_sample_seconds": rounded,
    }


def _panels(
    frames: Sequence[Image.Image], seconds: Sequence[float], *,
    frames_per_panel: int, frame_width: int, quality: int,
) -> list[bytes]:
    output = []
    for start in range(0, len(frames), frames_per_panel):
        indices = list(range(start, min(start + frames_per_panel, len(frames))))
        frame_height = round(frame_width * 9 / 16)
        canvas = Image.new("RGB", (len(indices) * frame_width, frame_height + 24), "white")
        draw = ImageDraw.Draw(canvas)
        for column, index in enumerate(indices):
            frame = frames[index].convert("RGB").copy()
            frame.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
            x = column * frame_width
            draw.text((x + 3, 3), f"F{index} {seconds[index]:.1f}s", fill="black")
            canvas.paste(frame, (x, 24))
        data = io.BytesIO()
        canvas.save(data, format="JPEG", quality=quality)
        output.append(data.getvalue())
    return output


def _image_content(data: bytes) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + base64.b64encode(data).decode(),
    }}


def _panel_content(panels: Sequence[bytes]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, panel in enumerate(panels):
        content.extend([
            {"type": "text", "text": f"Chronological panel {index + 1}:"},
            _image_content(panel),
        ])
    return content


def _query_call(
    client: OpenAI, *, question: str, model: Mapping[str, Any], cache_dir: Path,
) -> tuple[AGQAQueryPlan, list[dict[str, Any]]]:
    attempts = []
    last_error = ""
    for attempt in range(int(model["schema_retries"])):
        content = [{"type": "text", "text": f"Public question: {question.strip()}"}]
        if last_error:
            content.append({"type": "text", "text": "Fix this schema error: " + last_error})
        input_core = {
            "stage": "query", "prompt_version": PROMPT_VERSION,
            "model": model, "system": _query_system(), "content": content,
        }
        payload, usage, reused = _cached_provider_call(
            cache_dir=cache_dir, call_name=f"query_{attempt}", input_core=input_core,
            invoke=lambda: _provider_json_call(
                client, model=model, system=_query_system(), content=content,
                max_tokens=int(model["max_query_tokens"]),
                response_format=_query_response_format(),
            ),
        )
        attempts.append({"payload": payload, "usage": usage, "cache_reused": reused})
        try:
            return parse_query_plan(payload), attempts
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("query schema retries exhausted: " + last_error)


def _deterministic_query_call(
    *, question: str,
) -> tuple[AGQAQueryPlan, list[dict[str, Any]]]:
    plan = parse_public_question_plan(question)
    if plan is None:
        raise ValueError(
            "public question is outside the explicit-operand AGQA grammar"
        )
    payload = {
        key: value for key, value in plan.as_dict().items()
        if key in {
            "obligation_kind", "comparison", "operand_a", "operand_b",
            "visual_query_a", "visual_query_b", "parser_uncertainties",
        }
    }
    usage = {
        "model": "deterministic_agqa_explicit_operand_grammar_v1",
        "finish_reason": "local",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reported_cost_usd": 0.0,
        "response_sha256": stable_hash(payload),
        "local_non_provider_call": True,
    }
    return plan, [{"payload": payload, "usage": usage, "cache_reused": False}]


def _operand_call(
    client: OpenAI, *, role: str, requested_operand: str,
    panels: Sequence[bytes], frame_count: int, mode: str,
    model: Mapping[str, Any], cache_dir: Path, call_prefix: str,
) -> tuple[AGQAOperandReceipt, list[dict[str, Any]]]:
    attempts = []
    last_error = ""
    for attempt in range(int(model["schema_retries"])):
        content = [{"type": "text", "text": (
            f"operand_role: {role}\nrequested_operand: {requested_operand}\n"
            f"grounding_mode: {mode}\nGround only this operand."
        )}] + _panel_content(panels)
        if last_error:
            content.append({"type": "text", "text": "Fix this schema error: " + last_error})
        input_core = {
            "stage": "operand", "prompt_version": PROMPT_VERSION,
            "model": model, "system": _operand_system(frame_count),
            "role": role, "requested_operand": requested_operand,
            "mode": mode, "panel_sha256": [hashlib.sha256(x).hexdigest() for x in panels],
            "retry_error": last_error,
        }
        payload, usage, reused = _cached_provider_call(
            cache_dir=cache_dir, call_name=f"{call_prefix}_{attempt}",
            input_core=input_core,
            invoke=lambda: _provider_json_call(
                client, model=model, system=_operand_system(frame_count), content=content,
                max_tokens=int(model["max_operand_tokens"]),
                response_format=_operand_response_format(frame_count),
            ),
        )
        attempts.append({"payload": payload, "usage": usage, "cache_reused": reused})
        try:
            return parse_operand_receipt(
                payload, expected_role=role, expected_operand=requested_operand,
                frame_count=frame_count,
            ), attempts
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("operand schema retries exhausted: " + last_error)


def _direct_call(
    client: OpenAI, *, question: str, panels: Sequence[bytes],
    model: Mapping[str, Any], cache_dir: Path,
) -> tuple[str, dict[str, Any], dict[str, Any], bool]:
    content = [{"type": "text", "text": (
        f"Question: {question.strip()}\n"
        "Answer with the shortest canonical answer phrase supported by the video."
    )}] + _panel_content(panels)
    input_core = {
        "stage": "direct", "prompt_version": PROMPT_VERSION,
        "model": model, "question": question,
        "panel_sha256": [hashlib.sha256(x).hexdigest() for x in panels],
    }
    payload, usage, reused = _cached_provider_call(
        cache_dir=cache_dir, call_name="direct", input_core=input_core,
        invoke=lambda: _provider_json_call(
            client, model=model,
            system="Return JSON only as {\"response\": string}.", content=content,
            max_tokens=int(model["max_direct_tokens"]),
            response_format=_direct_response_format(),
        ),
    )
    response = str(payload.get("response") or "").strip()
    if not response:
        raise ValueError("direct baseline returned an empty response")
    return response, payload, usage, reused


def _mode(plan: AGQAQueryPlan) -> str:
    if plan.obligation_kind == RELATION_ROUTE:
        return (
            "RELATION_OBJECT"
            if plan.comparison in {"QUERY_OBJECT", "CHOOSE_OBJECT"}
            else "RELATION_EXISTS"
        )
    if plan.obligation_kind == TEMPORAL_PAIR_ROUTE:
        return "EVENT_ORDER_INTERVAL"
    return "EVENT_DURATION_ALL_OCCURRENCES"


def _wrong_source(sources: Sequence[SourceIRContract], route: str) -> SourceIRContract:
    mapping = {
        RELATION_ROUTE: sources[1],
        TEMPORAL_PAIR_ROUTE: sources[2],
        TEMPORAL_SINGLE_ROUTE: sources[0],
    }
    return mapping[route]


def _controller_dynamics(controller: AGQASourceAcquisitionController) -> tuple[Any, ...]:
    return (
        controller.obligation_kind, controller.required_operands,
        controller.recurrent, controller.maximum_rescans_per_operand,
    )


def _collect_runtime(
    sample: Mapping[str, Any], *, config: Mapping[str, Any], api_key: str,
    sources: Sequence[SourceIRContract], grounder_sha256: str, cache_root: Path,
) -> dict[str, Any]:
    video_path = Path(sample["video_path"])
    if _sha256(video_path) != sample["video_sha256"]:
        raise ValueError(f"video hash mismatch: {sample['video_id']}")
    model, media = config["model"], config["media"]
    parser_model = config.get("parser_model", model)
    rescan_model = config.get("rescan_model", model)
    nonrecurrent_model = config.get("nonrecurrent_model", model)
    dense_frames, dense_seconds, metadata = _sample_video_range(
        video_path, frame_count=int(media["dense_proxy_frame_count"]),
        max_side=int(media["proxy_frame_max_side"]),
    )
    dense_panels = _panels(
        dense_frames, dense_seconds,
        frames_per_panel=int(media["frames_per_panel"]),
        frame_width=int(media["panel_frame_width"]),
        quality=int(media["jpeg_quality"]),
    )
    task_cache = cache_root / str(sample["task_id"])
    client = OpenAI(
        api_key=api_key, base_url=str(model["base_url"]),
        timeout=float(model["timeout_seconds"]), max_retries=int(model["max_retries"]),
    )
    if config.get("query_parser_mode") == "DETERMINISTIC_EXPLICIT_OPERAND_GRAMMAR_V1":
        plan, query_attempts = _deterministic_query_call(
            question=str(sample["question"]),
        )
    else:
        plan, query_attempts = _query_call(
            client, question=str(sample["question"]), model=parser_model,
            cache_dir=task_cache,
        )
    controller = source_controller_for_plan(plan, sources)
    target_written_source = next(
        source for source in sources
        if source.contract_sha256 == controller.anonymous_source_contract_sha256
    )
    target_written_controller = source_controller_for_plan(
        plan, (_target_written_equivalent(target_written_source),),
    )
    target_written_match = (
        _controller_dynamics(controller) == _controller_dynamics(target_written_controller)
    )
    try:
        source_controller_for_plan(plan, (_wrong_source(sources, plan.obligation_kind),))
        wrong_source_abstained = False
    except ValueError:
        wrong_source_abstained = True

    roles = [("A", plan.visual_query_a)]
    if controller.required_operands == 2:
        roles.append(("B", plan.visual_query_b))
    chosen: dict[str, AGQAOperandReceipt] = {}
    operand_runs = {}
    for role, requested in roles:
        primary_model = nonrecurrent_model if not controller.recurrent else model
        primary, primary_attempts = _operand_call(
            client, role=role, requested_operand=requested, panels=dense_panels,
            frame_count=len(dense_frames), mode=_mode(plan), model=primary_model,
            cache_dir=task_cache, call_prefix=f"operand_{role}_primary",
        )
        rescan = None
        rescan_attempts: list[dict[str, Any]] = []
        rescan_metadata = None
        if operand_needs_rescan(
            primary, controller=controller,
            confidence_threshold=float(config["acquisition"]["rescan_confidence_threshold"]),
            require_specific_object=(plan.comparison == "QUERY_OBJECT"),
        ):
            start, end = recurrent_rescan_window(
                primary, seconds=dense_seconds,
                duration=float(metadata["duration_seconds"]),
                require_specific_object=(plan.comparison == "QUERY_OBJECT"),
            )
            zoom_frames, zoom_seconds, rescan_metadata = _sample_video_range(
                video_path, frame_count=int(media["rescan_frame_count"]),
                max_side=int(media["rescan_frame_max_side"]),
                start_second=start, end_second=end,
            )
            zoom_panels = _panels(
                zoom_frames, zoom_seconds,
                frames_per_panel=int(media["rescan_frames_per_panel"]),
                frame_width=int(media["rescan_panel_frame_width"]),
                quality=int(media["jpeg_quality"]),
            )
            local_rescan, rescan_attempts = _operand_call(
                client, role=role, requested_operand=requested, panels=zoom_panels,
                frame_count=len(zoom_frames), mode=_mode(plan) + "_RECURRENT_RESCAN",
                model=rescan_model, cache_dir=task_cache,
                call_prefix=f"operand_{role}_rescan",
            )
            rescan = remap_operand_receipt(
                local_rescan, local_seconds=zoom_seconds, global_seconds=dense_seconds,
            )
        selected_receipt = (
            reconcile_recurrent_receipts(primary, rescan)
            if controller.recurrent
            else choose_operand_receipt(primary, rescan)
        )
        local_object_receipt = None
        local_object_canonicalizations: tuple[str, ...] = ()
        if plan.comparison == "QUERY_OBJECT":
            detector_config = config["local_object_grounder"]
            indices = inspection_indices(
                selected_receipt,
                maximum=int(detector_config["maximum_interval_frames"]),
            )
            if indices:
                local_object_receipt = detect_objects(
                    dense_frames, frame_indices=indices,
                    model_path=Path(detector_config["model_path"]),
                    expected_model_sha256=str(detector_config["model_sha256"]),
                    confidence_threshold=float(detector_config["confidence_threshold"]),
                    nms_threshold=float(detector_config["nms_threshold"]),
                )
                selected_receipt, local_object_canonicalizations = (
                    refine_query_object_receipt(
                        selected_receipt, local_object_receipt,
                    )
                )
        chosen[role] = selected_receipt
        operand_runs[role] = {
            "requested_operand": requested,
            "primary_receipt": primary.as_dict(),
            "primary_attempts": primary_attempts,
            "rescan_triggered": rescan is not None,
            "rescan_receipt_global_timeline": rescan.as_dict() if rescan else None,
            "rescan_attempts": rescan_attempts,
            "rescan_video_metadata": rescan_metadata,
            "chosen_receipt_sha256": chosen[role].receipt_sha256,
            "local_object_grounding_receipt": (
                local_object_receipt.as_dict() if local_object_receipt else None
            ),
            "local_object_canonicalizations": list(
                local_object_canonicalizations
            ),
        }
    receipt = merge_operand_receipts(
        plan, operand_a=chosen["A"], operand_b=chosen.get("B"),
        frame_count=len(dense_frames),
    )
    execution = execute_grounding_receipt(receipt)
    # Fail closed: source acquisition may shape sensing before qualification,
    # but no source program may authorize a target decision yet.
    prequalification = select_source_for_grounding(
        sources, task_id=str(sample["task_id"]), receipt=receipt,
        target_grounder_sha256=grounder_sha256, grounder_qualified=False,
    )
    direct, direct_payload, direct_usage, direct_reused = _direct_call(
        client, question=str(sample["question"]), panels=dense_panels,
        model=model, cache_dir=task_cache,
    )
    body = {
        "task_id": str(sample["task_id"]), "video_id": str(sample["video_id"]),
        "question_sha256": stable_hash(str(sample["question"])),
        "video_sha256": sample["video_sha256"],
        "dense_panel_sha256": [hashlib.sha256(x).hexdigest() for x in dense_panels],
        "video_metadata": metadata,
        "query_plan": plan.as_dict(), "query_attempts": query_attempts,
        "source_acquisition_controller": controller.as_dict(),
        "operand_runs": operand_runs,
        "grounding_receipt": receipt.as_dict(),
        "target_native_execution": execution,
        "prequalification_source_selection": prequalification,
        "source_permuted_wrong_type_abstained": wrong_source_abstained,
        "target_written_equivalent_dynamics_match": target_written_match,
        "direct_response": direct, "direct_raw_payload": direct_payload,
        "direct_usage": direct_usage, "direct_cache_reused": direct_reused,
        "runtime_visible_fields": [
            "public_question_at_parser_only", "one_operand_at_each_grounder_call",
            "chronological_dense_frames", "frame_timestamps",
        ],
        "runtime_answer_read": False, "runtime_functional_program_read": False,
        "runtime_scene_graph_read": False, "runtime_source_identity_read": False,
        "operand_grounder_question_read": False,
        "operand_grounder_competing_operand_read": False,
        "direct_call_started_after_typed_receipt_froze": True,
        "grounder_sha256": grounder_sha256,
    }
    return body | {"runtime_receipt_sha256": stable_hash(body)}


def _normalize(value: Any) -> str:
    text = str(value).casefold().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    for prefix in ("the answer is ", "it is ", "they were ", "they are "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    return {"true": "yes", "false": "no"}.get(text, text)


def _answer_matches(prediction: Any, gold: Any) -> bool:
    predicted, expected = _normalize(prediction), _normalize(gold)
    if expected in {"yes", "no", "before", "after"}:
        return bool(predicted) and predicted.split(maxsplit=1)[0] == expected
    return predicted == expected


def _load_selected_rows(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    wanted = {str(row["task_id"]) for row in manifest["samples"]}
    output = {}
    with zipfile.ZipFile(manifest["archive_path"]) as bundle:
        with bundle.open(manifest["entry"], "r") as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as text:
                for task_id, row in _iter_top_level_object(text):
                    if task_id in wanted:
                        output[task_id] = row
                        if len(output) == len(wanted):
                            break
    if set(output) != wanted:
        raise ValueError("manifest tasks are missing from AGQA metadata")
    return output


def _usage_rows(runtime: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = [x["usage"] for x in runtime["query_attempts"]]
    for operand in runtime["operand_runs"].values():
        rows.extend(x["usage"] for x in operand["primary_attempts"])
        rows.extend(x["usage"] for x in operand["rescan_attempts"])
    rows.append(runtime["direct_usage"])
    return [row for row in rows if not row.get("local_non_provider_call")]


def _cumulative_cache_usage(cache_root: Path) -> list[Mapping[str, Any]]:
    """Count every accepted V3 provider call, including rejected dev candidates."""

    usage = []
    for path in sorted(cache_root.glob("*/*.json")):
        row = json.loads(path.read_text())
        claimed = row.get("call_receipt_sha256")
        body = dict(row)
        body.pop("call_receipt_sha256", None)
        if claimed != stable_hash(body):
            raise ValueError(f"cached provider-call receipt hash mismatch: {path}")
        usage.append(row["usage"])
    return usage


def _runtime_applicability_score(runtime: Mapping[str, Any]) -> tuple[float, ...]:
    """Score typed evidence without reading direct responses or annotations."""

    receipt = runtime["grounding_receipt"]
    events = receipt["events"]
    decision = runtime["target_native_execution"]["decision"] is not None
    conflict = any(
        "CONFLICT" in marker for marker in receipt.get("canonicalizations", [])
    )
    observed = [
        row for row in events
        if row["observability"] == "OBSERVED"
        and float(row["confidence"]) >= 0.5
    ]
    minimum_confidence = min(
        (float(row["confidence"]) for row in observed), default=0.0,
    )
    structural_margin = 0.0
    comparison = runtime["query_plan"]["comparison"]
    if comparison == "BEFORE_AFTER":
        starts = {
            role: [
                int(row["start_frame"]) for row in observed
                if row["operand_role"] == role and row["start_frame"] is not None
            ]
            for role in ("A", "B")
        }
        if starts["A"] and starts["B"]:
            structural_margin = float(abs(min(starts["A"]) - min(starts["B"])))
    elif comparison.startswith("SELECT_") or comparison.startswith("VERIFY_"):
        durations = {}
        for role in ("A", "B"):
            intervals = sorted(
                (int(row["start_frame"]), int(row["end_frame"]))
                for row in observed
                if row["operand_role"] == role
                and row["start_frame"] is not None
                and row["end_frame"] is not None
            )
            merged = []
            for start, end in intervals:
                if merged and start <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            durations[role] = sum(end - start + 1 for start, end in merged)
        structural_margin = float(abs(durations["A"] - durations["B"]))
    elif comparison == "EXISTS":
        structural_margin = 2.0 if (
            "RECURRENT_DOUBLE_SCAN_CONFIRMED_UNOBSERVED"
            in receipt.get("canonicalizations", [])
        ) else float(bool(observed))
    return (
        float(decision), float(not conflict), structural_margin,
        minimum_confidence,
    )


def _select_runtime_rows(
    frozen_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Mapping[str, Mapping[str, Any]], *, per_route: int,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Freeze top evidence-qualified rows per predicted route before evaluation."""

    grouped: dict[str, list[tuple[tuple[float, ...], str, Mapping[str, Any]]]] = {
        route: [] for route in ROUTES
    }
    scores = {}
    for frozen in frozen_rows:
        task_id = str(frozen["task_id"])
        runtime = runtime_rows[task_id]
        route = str(runtime["query_plan"]["obligation_kind"])
        score = _runtime_applicability_score(runtime)
        scores[task_id] = list(score)
        grouped.setdefault(route, []).append((score, stable_hash(task_id), frozen))
    selected = []
    for route in ROUTES:
        ranked = sorted(
            grouped.get(route, []),
            key=lambda item: tuple(-value for value in item[0]) + (item[1],),
        )
        if len(ranked) < per_route:
            raise ValueError(f"runtime selector lacks candidates for {route}")
        selected.extend(item[2] for item in ranked[:per_route])
    selected.sort(key=lambda row: str(row["task_id"]))
    core = {
        "mode": "OUTCOME_BLIND_TYPED_EVIDENCE_RANK_V1",
        "per_predicted_route": per_route,
        "candidate_count": len(frozen_rows),
        "selected_task_ids": sorted(str(row["task_id"]) for row in selected),
        "scores_by_task_id": scores,
        "direct_response_read": False,
        "official_answer_read": False,
        "official_program_read_by_selector": False,
        "official_scene_graph_read": False,
    }
    return selected, core | {"selection_sha256": stable_hash(core)}


def collect(
    *, config_path: Path, keys_path: Path, output_path: Path,
    workers: int, limit: int | None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    prereg_path = REPO_ROOT / config["preregistration"]
    if _sha256(prereg_path) != config["preregistration_file_sha256"]:
        raise ValueError("AGQA V3 preregistration hash mismatch")
    prereg = json.loads(prereg_path.read_text())
    expected_preregistration_status = config.get(
        "expected_preregistration_status", "FROZEN_BEFORE_ANY_V3_NEURAL_CALL",
    )
    if prereg["status"] != expected_preregistration_status:
        raise ValueError("AGQA V3 preregistration is not frozen")
    manifest_path = REPO_ROOT / config["manifest"]
    if _sha256(manifest_path) != config["manifest_file_sha256"]:
        raise ValueError("AGQA V3 manifest file hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    manifest_body = dict(manifest)
    manifest_hash = manifest_body.pop("manifest_sha256")
    if stable_hash(manifest_body) != manifest_hash:
        raise ValueError("AGQA V3 manifest content hash mismatch")
    if manifest["split"] != config["split"]:
        raise ValueError("AGQA V3 config/manifest split mismatch")
    expected_status = config.get("expected_manifest_status") or {
        "development": "FROZEN_CONSUMED_DEVELOPMENT_BEFORE_V3_NEURAL_CALLS",
        "reserve": "FROZEN_RAW_VIDEO_UNSEEN_RESERVE_BEFORE_V3_NEURAL_CALLS",
    }[config["split"]]
    if manifest["status"] != expected_status:
        raise ValueError("AGQA V3 manifest has the wrong frozen status")
    labels = ["module", "collector"]
    if "executor" in config["grounder"]:
        labels.append("executor")
    for label in labels:
        path = REPO_ROOT / config["grounder"][label]
        if _sha256(path) != config["grounder"][f"{label}_sha256"]:
            raise ValueError(f"AGQA V3 {label} hash mismatch")
    local_spec = config["local_object_grounder"]
    local_module = REPO_ROOT / local_spec["module"]
    if _sha256(local_module) != local_spec["module_sha256"]:
        raise ValueError("AGQA V3 local object-grounder module hash mismatch")
    local_model = Path(local_spec["model_path"])
    if _sha256(local_model) != local_spec["model_sha256"]:
        raise ValueError("AGQA V3 local object-grounder model hash mismatch")
    development_dependency = None
    if config["split"] == "reserve":
        dependency_path = REPO_ROOT / config["development_qualification_report"]
        if _sha256(dependency_path) != config["development_qualification_file_sha256"]:
            raise ValueError("AGQA V3 development qualification hash mismatch")
        development_dependency = json.loads(dependency_path.read_text())
        if not development_dependency.get("grounder_qualified"):
            raise ValueError("reserve cannot run before development qualification")
    key_values = runpy.run_path(str(keys_path))
    api_key = key_values.get(config["model"]["api_key_name"])
    if not api_key:
        raise ValueError("OpenRouter API key is unavailable")
    sources, arcade_report = _load_sources(config)
    semantic_core = {
        "prompt_version": PROMPT_VERSION,
        "query_parser_mode": config.get("query_parser_mode", "NEURAL_JSON_V3"),
        "applicability_mode": config.get("applicability_mode", "ALL_ROUTED_TASKS"),
        "runtime_selection": config.get("runtime_selection"),
        "grounder_module_sha256": config["grounder"]["module_sha256"],
        "grounder_collector_sha256": config["grounder"]["collector_sha256"],
        "grounder_executor_sha256": config["grounder"].get("executor_sha256"),
        "parser_model": config.get("parser_model", config["model"]),
        "rescan_model": config.get("rescan_model", config["model"]),
        "nonrecurrent_model": config.get("nonrecurrent_model", config["model"]),
        "local_object_grounder": config["local_object_grounder"],
        "model": config["model"], "media": config["media"],
        "acquisition": config["acquisition"],
        "visible_fields": config["grounder"]["visible_fields"],
        "forbidden_fields": config["grounder"]["forbidden_fields"],
        "source_contract_sha256": sorted(x.contract_sha256 for x in sources),
    }
    grounder_sha256 = stable_hash(semantic_core)
    if development_dependency is not None and (
        development_dependency["grounder_sha256"] != grounder_sha256
    ):
        raise ValueError("reserve grounder differs from qualified development grounder")

    metadata_rows = _load_selected_rows(manifest)
    frozen_rows = list(manifest["samples"])
    if limit is not None:
        frozen_rows = frozen_rows[:limit]
    runtime_inputs = []
    for frozen in frozen_rows:
        task_id = str(frozen["task_id"])
        question = str(metadata_rows[task_id]["question"])
        if stable_hash(question) != frozen["question_sha256"]:
            raise ValueError(f"question hash mismatch: {task_id}")
        runtime_inputs.append(dict(frozen) | {"question": question})

    cache_root = output_path.parent / "call_cache"
    runtime_dir = output_path.parent / "runtime_receipts"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_rows: dict[str, dict[str, Any]] = {}
    pending = []
    for sample in runtime_inputs:
        path = runtime_dir / f"{sample['task_id']}.json"
        if path.is_file():
            cached = json.loads(path.read_text())
            if (
                cached.get("grounder_sha256") == grounder_sha256
                and cached.get("question_sha256") == stable_hash(sample["question"])
                and cached.get("video_sha256") == sample["video_sha256"]
            ):
                runtime_rows[str(sample["task_id"])] = cached
                print(f"reused {sample['task_id']}", flush=True)
                continue
        pending.append(sample)
    errors = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _collect_runtime, sample, config=config, api_key=api_key,
                sources=sources, grounder_sha256=grounder_sha256,
                cache_root=cache_root,
            ): str(sample["task_id"])
            for sample in pending
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                errors[task_id] = f"{type(exc).__name__}: {exc}"
                print(f"failed {task_id}: {errors[task_id]}", flush=True)
                continue
            runtime_rows[task_id] = row
            (runtime_dir / f"{task_id}.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n"
            )
            print(f"completed {task_id}", flush=True)
    if errors:
        (output_path.parent / "worker_errors.json").write_text(json.dumps({
            "grounder_sha256": grounder_sha256, "errors": errors,
        }, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(
            "AGQA V3 workers failed; accepted calls and complete rows were cached: "
            + repr(errors)
        )

    evaluation_frozen_rows = frozen_rows
    runtime_selection_receipt = None
    if config.get("runtime_selection"):
        evaluation_frozen_rows, runtime_selection_receipt = _select_runtime_rows(
            frozen_rows, runtime_rows,
            per_route=int(config["runtime_selection"]["per_predicted_route"]),
        )

    # Evaluator-only access begins after every runtime receipt and optional
    # outcome-blind applicability selection are immutable.
    evaluated = []
    for frozen in evaluation_frozen_rows:
        task_id = str(frozen["task_id"])
        target = metadata_rows[task_id]
        runtime = runtime_rows[task_id]
        program = str(target["program"])
        if stable_hash(program) != frozen["program_sha256"]:
            raise ValueError(f"functional-program hash mismatch: {task_id}")
        oracle_route = profile_program(task_id=task_id, program=program).route_kind
        gold = str(target["answer"])
        decision = runtime["target_native_execution"]["decision"]
        direct = runtime["direct_response"]
        decisive = decision is not None
        typed_prediction = decision if decisive else direct
        evaluated.append(runtime | {
            "oracle_route_evaluator_only": oracle_route,
            "gold_answer_evaluator_only": gold,
            "predicted_route_correct": (
                runtime["query_plan"]["obligation_kind"] == oracle_route
            ),
            "decisive_execution": decisive,
            "decisive_correct": _answer_matches(decision, gold) if decisive else None,
            "direct_correct": _answer_matches(direct, gold),
            "typed_fallback_prediction": typed_prediction,
            "typed_fallback_correct": _answer_matches(typed_prediction, gold),
            "official_answer_first_read_after_all_runtime_rows_froze": True,
            "official_scene_graph_read_by_evaluator": False,
        })

    valid = len(evaluated)
    route_correct = sum(row["predicted_route_correct"] for row in evaluated)
    decisive_rows = [row for row in evaluated if row["decisive_execution"]]
    decisive_correct = sum(row["decisive_correct"] for row in decisive_rows)
    direct_correct = sum(row["direct_correct"] for row in evaluated)
    typed_correct = sum(row["typed_fallback_correct"] for row in evaluated)
    wins = sum(row["typed_fallback_correct"] and not row["direct_correct"] for row in evaluated)
    losses = sum(row["direct_correct"] and not row["typed_fallback_correct"] for row in evaluated)
    controls = {
        "source_permuted_abstentions": sum(
            row["source_permuted_wrong_type_abstained"] for row in evaluated
        ),
        "target_written_equivalent_matches": sum(
            row["target_written_equivalent_dynamics_match"] for row in evaluated
        ),
    }
    runtime_usage = [
        item for frozen in frozen_rows
        for item in _usage_rows(runtime_rows[str(frozen["task_id"])])
    ]
    cumulative_usage = _cumulative_cache_usage(cache_root)
    runtime_cost = sum(float(row["reported_cost_usd"]) for row in runtime_usage)
    total_cost = sum(float(row["reported_cost_usd"]) for row in cumulative_usage)
    gate_spec = config["qualification_gates"]
    gates = {
        "required_valid_runtime_rows": valid >= gate_spec["required_valid_runtime_rows"],
        "minimum_route_correct": route_correct >= gate_spec["minimum_route_correct"],
        "minimum_decisive_executions": len(decisive_rows) >= gate_spec["minimum_decisive_executions"],
        "minimum_decisive_accuracy": bool(decisive_rows) and (
            decisive_correct / len(decisive_rows) >= gate_spec["minimum_decisive_accuracy"]
        ),
        "no_typed_vs_direct_losses": losses <= gate_spec["maximum_typed_vs_direct_losses"],
        "minimum_typed_vs_direct_wins": wins >= gate_spec["minimum_typed_vs_direct_wins"],
        "source_permuted_wrong_type_abstains": controls["source_permuted_abstentions"] >= gate_spec["required_source_permuted_abstentions"],
        "target_written_equivalent_matches": controls["target_written_equivalent_matches"] >= gate_spec["required_target_written_equivalent_matches"],
        "provider_cost_within_cap": total_cost <= gate_spec["maximum_reported_provider_cost_usd"],
        "runtime_no_answer_program_scene_graph_or_source_identity": all(
            not row[key] for row in evaluated for key in (
                "runtime_answer_read", "runtime_functional_program_read",
                "runtime_scene_graph_read", "runtime_source_identity_read",
                "operand_grounder_question_read",
                "operand_grounder_competing_operand_read",
            )
        ),
        "prequalification_harness_abstained": all(
            row["prequalification_source_selection"]["selected_program_sha256"] is None
            for row in evaluated
        ),
    }
    qualified = all(gates.values())
    for row in evaluated:
        receipt = parse_frame_grounding_receipt(
            row["grounding_receipt"], frame_count=int(config["media"]["dense_proxy_frame_count"]),
        )
        selection = select_source_for_grounding(
            sources, task_id=row["task_id"], receipt=receipt,
            target_grounder_sha256=grounder_sha256, grounder_qualified=qualified,
        )
        row["postqualification_source_selection"] = selection
        authorized = (
            selection["selected_program_sha256"] is not None
            and row["target_native_execution"]["decision"] is not None
        )
        row["unified_harness_executor_authorized"] = authorized
        row["unified_harness_prediction"] = (
            row["target_native_execution"]["decision"] if authorized
            else row["direct_response"]
        )
        row["unified_harness_correct"] = _answer_matches(
            row["unified_harness_prediction"], row["gold_answer_evaluator_only"],
        )
    unified_correct = sum(row["unified_harness_correct"] for row in evaluated)
    unified_authorized = sum(row["unified_harness_executor_authorized"] for row in evaluated)
    report_version = str(config.get("report_version", "V3")).upper()
    status = (
        f"AGQA2_ACTIVE_GROUNDER_{report_version}_{config['split'].upper()}_QUALIFIED"
        if qualified else
        f"AGQA2_ACTIVE_GROUNDER_{report_version}_{config['split'].upper()}_NOT_QUALIFIED"
    )
    result = {
        "schema_version": f"agqa2-active-grounding-report-{report_version.casefold()}",
        "status": status, "split": config["split"],
        "claim_boundary": config["claim_boundary"],
        "config_sha256": _sha256(config_path),
        "preregistration_sha256": _sha256(prereg_path),
        "manifest_sha256": manifest_hash,
        "grounder_sha256": grounder_sha256,
        "model": config["model"]["id"], "sample_count": valid,
        "acquisition_candidate_count": len(frozen_rows),
        "unique_video_count": len({row["video_id"] for row in evaluated}),
        "new_video_downloads": 0,
        "accepted_runtime_provider_calls": len(runtime_usage),
        "accepted_runtime_reported_provider_cost_usd": runtime_cost,
        "provider_calls": len(cumulative_usage),
        "reported_provider_cost_usd": total_cost,
        "provider_cost_accounting": (
            "CUMULATIVE_ALL_HASHED_ACCEPTED_CALLS_IN_SPLIT_CACHE_INCLUDING_"
            "REJECTED_DEVELOPMENT_CANDIDATES"
        ),
        "metrics": {
            "valid_runtime_rows": valid, "route_correct": route_correct,
            "route_accuracy": route_correct / valid if valid else 0,
            "decisive_executions": len(decisive_rows),
            "decisive_coverage": len(decisive_rows) / valid if valid else 0,
            "decisive_correct": decisive_correct,
            "decisive_accuracy": decisive_correct / len(decisive_rows) if decisive_rows else 0,
            "direct_correct": direct_correct,
            "typed_fallback_correct": typed_correct,
            "typed_vs_direct_wins": wins, "typed_vs_direct_losses": losses,
            "unified_harness_executor_authorizations": unified_authorized,
            "unified_harness_correct": unified_correct,
            "unified_harness_vs_direct_delta": unified_correct - direct_correct,
            "rescans_triggered": sum(
                operand["rescan_triggered"] for row in evaluated
                for operand in row["operand_runs"].values()
            ),
        },
        "controls": controls, "qualification_gates": gates,
        "runtime_selection_receipt": runtime_selection_receipt,
        "grounder_qualified": qualified,
        "development_qualification_dependency": (
            development_dependency["report_sha256"] if development_dependency else None
        ),
        "source_portfolio_caveat": {
            "status": arcade_report["status"],
            "authentic_correct": arcade_report["qualified_aggregate"]["authentic_correct"],
            "source_permuted_correct": arcade_report["qualified_aggregate"]["permuted_correct"],
            "source_specific_claim_passed": False,
        },
        "rows": sorted(evaluated, key=lambda row: row["task_id"]),
        "untouched_benchmark_claim": False, "source_provenance_claim": False,
    }
    result["report_sha256"] = stable_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = collect(
        config_path=args.config.resolve(), keys_path=args.keys.resolve(),
        output_path=args.output.resolve(), workers=args.workers, limit=args.limit,
    )
    print(json.dumps({
        "status": result["status"], "metrics": result["metrics"],
        "controls": result["controls"], "qualification_gates": result["qualification_gates"],
        "provider_calls": result["provider_calls"],
        "reported_provider_cost_usd": result["reported_provider_cost_usd"],
        "report_sha256": result["report_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
