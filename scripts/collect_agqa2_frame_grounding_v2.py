#!/usr/bin/env python3
"""Collect a low-cost, answer-blind AGQA frame-grounding pilot."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
from dataclasses import asdict
import hashlib
import io
import json
import math
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

from motif_transfer.agqa_frame_grounder import (  # noqa: E402
    AGQAFrameGroundingReceipt,
    execute_grounding_receipt,
    parse_frame_grounding_receipt,
    select_source_for_grounding,
)
from motif_transfer.agqa_program_transfer import (  # noqa: E402
    RELATION_ROUTE,
    TEMPORAL_PAIR_ROUTE,
    TEMPORAL_SINGLE_ROUTE,
    profile_program,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import (  # noqa: E402
    _iter_top_level_object,
    _load_sources,
)


ROUTES = (RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE)
PROMPT_VERSION = "AGQA_ANSWER_BLIND_TYPED_EVENT_GROUNDER_V2_1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    return stable_hash(value)


def _decode_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("provider response omitted a JSON object")
    payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("provider JSON response must be an object")
    return payload


def _provider_json_call(
    client: OpenAI, *, model: str, system: str,
    content: list[dict[str, Any]], max_tokens: int,
    response_format: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        response_format=dict(response_format),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        extra_body={"provider": {"require_parameters": True}},
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
        "response_sha256": _content_hash(payload),
    }


def _grounding_response_format(frame_count: int) -> dict[str, Any]:
    frame_or_null = {
        "anyOf": [
            {"type": "integer", "minimum": 0, "maximum": frame_count - 1},
            {"type": "null"},
        ]
    }
    event = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "event_id": {"type": "string"},
            "operand_role": {"type": "string", "enum": ["A", "B", "CONTEXT"]},
            "label": {"type": "string"},
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
            "observability": {
                "type": "string", "enum": ["OBSERVED", "PARTIAL", "UNOBSERVED"],
            },
            "start_frame": frame_or_null,
            "end_frame": frame_or_null,
            "evidence_frames": {
                "type": "array", "maxItems": 3,
                "items": {
                    "type": "integer", "minimum": 0, "maximum": frame_count - 1,
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "event_id", "operand_role", "label", "subject", "predicate", "object",
            "observability", "start_frame", "end_frame", "evidence_frames",
            "confidence", "uncertainties",
        ],
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "obligation_kind": {"type": "string", "enum": [
                RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE,
                "NO_EXACT_SOURCE_TYPE",
            ]},
            "comparison": {"type": "string", "enum": [
                "EXISTS", "QUERY_OBJECT", "CHOOSE_OBJECT", "BEFORE_AFTER",
                "SELECT_LONGER", "SELECT_SHORTER", "VERIFY_A_LONGER",
                "VERIFY_A_SHORTER", "UNSUPPORTED",
            ]},
            "operand_a": {"type": "string"},
            "operand_b": {"type": "string"},
            "events": {"type": "array", "maxItems": 6, "items": event},
            "coverage": {
                "type": "string", "enum": ["SUFFICIENT", "PARTIAL", "INSUFFICIENT"],
            },
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "obligation_kind", "comparison", "operand_a", "operand_b", "events",
            "coverage", "uncertainties",
        ],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agqa_typed_event_grounding",
            "strict": True,
            "schema": schema,
        },
    }


def _direct_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agqa_direct_response",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"response": {"type": "string"}},
                "required": ["response"],
            },
        },
    }


def _sample_video(
    path: Path, *, frame_count: int, max_side: int,
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
    seconds = np.linspace(0.0, max(0.0, duration - 1.0 / fps), frame_count)
    frames = []
    for second in seconds:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000.0)
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
        "proxy_sample_seconds": rounded,
    }


def _panel_bytes(
    frames: Sequence[Image.Image], seconds: Sequence[float], *,
    start: int, end: int, frame_width: int, quality: int,
) -> bytes:
    chosen = list(range(start, end))
    columns = len(chosen)
    frame_height = round(frame_width * 9 / 16)
    canvas = Image.new("RGB", (columns * frame_width, frame_height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for column, index in enumerate(chosen):
        frame = frames[index].convert("RGB").copy()
        frame.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
        x = column * frame_width
        draw.text((x + 3, 3), f"F{index} {seconds[index]:.1f}s", fill="black")
        canvas.paste(frame, (x, 24))
    output = io.BytesIO()
    canvas.save(output, format="JPEG", quality=quality)
    return output.getvalue()


def _panels(
    frames: Sequence[Image.Image], seconds: Sequence[float],
    media: Mapping[str, Any],
) -> list[bytes]:
    per_panel = int(media["frames_per_panel"])
    return [
        _panel_bytes(
            frames, seconds, start=start,
            end=min(start + per_panel, len(frames)),
            frame_width=int(media["panel_frame_width"]),
            quality=int(media["jpeg_quality"]),
        )
        for start in range(0, len(frames), per_panel)
    ]


def _image_content(data: bytes) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(data).decode(),
        },
    }


def _grounding_system(frame_count: int) -> str:
    return (
        "You are an answer-blind video event grounder. Return JSON only with keys: "
        "obligation_kind, comparison, operand_a, operand_b, events, coverage, "
        "uncertainties. operand_a and operand_b MUST each be a plain JSON string, "
        "never an object, list, or null. obligation_kind must be RELATION_RECURRENT, "
        "TEMPORAL_PAIR_RECURRENT, TEMPORAL_SINGLE_NONRECURRENT, or "
        "NO_EXACT_SOURCE_TYPE. comparison must be: EXISTS, QUERY_OBJECT, or "
        "CHOOSE_OBJECT for relation questions; BEFORE_AFTER for before/after "
        "questions; SELECT_LONGER or SELECT_SHORTER when the question asks which "
        "event has that duration; VERIFY_A_LONGER or VERIFY_A_SHORTER when it asks "
        "whether operand A is longer/shorter than B; UNSUPPORTED otherwise. "
        "TEMPORAL_PAIR_RECURRENT is ONLY for BEFORE_AFTER event ordering. Every "
        "longer/shorter duration question is TEMPORAL_SINGLE_NONRECURRENT, even "
        "though it contains two event operands, because each duration is scored "
        "independently before comparison. "
        "For VERIFY duration questions, operand A is the event whose longer/shorter "
        "claim is being tested and operand B is its comparator, regardless of mention "
        "order. For QUERY_OBJECT, operand_a describes the queried relation and an "
        "observed A event's object field must contain only the grounded object noun. "
        "For CHOOSE_OBJECT, operand_a and operand_b are the two candidate object nouns "
        "and events bind them by role A/B. For temporal questions, events bind the two "
        "event operands by role A/B. Each event has exactly: event_id, operand_role, "
        "label, subject, predicate, object, observability, start_frame, end_frame, "
        "evidence_frames, confidence, uncertainties. event IDs are consecutive E0...; "
        "operand_role is A, B, or CONTEXT; observability is OBSERVED, PARTIAL, or "
        "UNOBSERVED. OBSERVED events need 1-3 chronological pixel evidence frames and "
        "a best-estimate interval. UNOBSERVED events use null interval and no evidence. "
        f"Frame IDs are integers 0..{frame_count - 1}. coverage is SUFFICIENT, PARTIAL, "
        "or INSUFFICIENT. Do not answer, select an operand, judge correctness, or emit "
        "answer/choice/program/scene-graph/source-game fields. When pixels are "
        "ambiguous, use PARTIAL or UNOBSERVED instead of inventing evidence. "
        "Example query shape for 'Which object were they sitting on?': "
        "{\"obligation_kind\":\"RELATION_RECURRENT\","
        "\"comparison\":\"QUERY_OBJECT\","
        "\"operand_a\":\"person sitting on an object\",\"operand_b\":\"\","
        "\"events\":[{\"event_id\":\"E0\",\"operand_role\":\"A\","
        "\"label\":\"person sitting on chair\",\"subject\":\"person\","
        "\"predicate\":\"sitting on\",\"object\":\"chair\","
        "\"observability\":\"OBSERVED\",\"start_frame\":2,"
        "\"end_frame\":7,\"evidence_frames\":[2,7],\"confidence\":0.8,"
        "\"uncertainties\":[]}],\"coverage\":\"SUFFICIENT\","
        "\"uncertainties\":[]}"
    )


def _ground_call(
    client: OpenAI, *, question: str, panels: Sequence[bytes],
    frame_count: int, model: Mapping[str, Any],
) -> tuple[AGQAFrameGroundingReceipt, dict[str, Any], list[dict[str, Any]]]:
    base_content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"Public question: {question.strip()}\n"
            "Construct typed visual observations only. Do not answer the question."
        ),
    }]
    for index, panel in enumerate(panels):
        base_content.extend([
            {"type": "text", "text": f"Chronological panel {index + 1}:"},
            _image_content(panel),
        ])
    attempts = []
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        content = list(base_content)
        if last_error:
            content.append({
                "type": "text",
                "text": "Your prior JSON violated the schema: " + last_error,
            })
        payload, usage = _provider_json_call(
            client,
            model=str(model["id"]),
            system=_grounding_system(frame_count),
            content=content,
            max_tokens=int(model["max_ground_tokens"]),
            response_format=_grounding_response_format(frame_count),
        )
        attempts.append({"payload": payload, "usage": usage})
        try:
            return (
                parse_frame_grounding_receipt(payload, frame_count=frame_count),
                usage,
                attempts,
            )
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("grounding schema retries exhausted: " + last_error)


def _direct_call(
    client: OpenAI, *, question: str, panels: Sequence[bytes],
    model: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"Question: {question.strip()}\n"
            "Answer with the shortest canonical answer phrase supported by the video."
        ),
    }]
    for index, panel in enumerate(panels):
        content.extend([
            {"type": "text", "text": f"Chronological panel {index + 1}:"},
            _image_content(panel),
        ])
    payload, usage = _provider_json_call(
        client,
        model=str(model["id"]),
        system="Return JSON only as {\"response\": string}.",
        content=content,
        max_tokens=int(model["max_direct_tokens"]),
        response_format=_direct_response_format(),
    )
    response = str(payload.get("response") or "").strip()
    if not response:
        raise ValueError("direct baseline returned an empty response")
    return response, payload, usage


def _collect_runtime(
    sample: Mapping[str, Any], *, config: Mapping[str, Any], api_key: str,
    sources: Sequence[Any], grounder_sha256: str,
) -> dict[str, Any]:
    path = Path(sample["video_path"])
    if _sha256(path) != sample["video_sha256"]:
        raise ValueError(f"video hash mismatch: {sample['video_id']}")
    media = config["media"]
    frames, seconds, metadata = _sample_video(
        path,
        frame_count=int(media["proxy_frame_count"]),
        max_side=int(media["proxy_frame_max_side"]),
    )
    panel_bytes = _panels(frames, seconds, media)
    client = OpenAI(
        api_key=api_key,
        base_url=str(config["model"]["base_url"]),
        timeout=float(config["model"]["timeout_seconds"]),
        max_retries=int(config["model"]["max_retries"]),
    )
    receipt, ground_usage, attempts = _ground_call(
        client,
        question=str(sample["question"]),
        panels=panel_bytes,
        frame_count=len(frames),
        model=config["model"],
    )
    execution = execute_grounding_receipt(receipt)
    prequalification = select_source_for_grounding(
        sources,
        task_id=str(sample["task_id"]),
        receipt=receipt,
        target_grounder_sha256=grounder_sha256,
        grounder_qualified=False,
    )
    # The answer-bearing direct baseline is a separate call made only after the
    # typed receipt, target-native execution, and prequalification abstention freeze.
    direct, direct_payload, direct_usage = _direct_call(
        client,
        question=str(sample["question"]),
        panels=panel_bytes,
        model=config["model"],
    )
    body = {
        "task_id": str(sample["task_id"]),
        "video_id": str(sample["video_id"]),
        "question_sha256": stable_hash(str(sample["question"])),
        "video_sha256": sample["video_sha256"],
        "panel_sha256": [hashlib.sha256(value).hexdigest() for value in panel_bytes],
        "video_metadata": metadata,
        "grounding_receipt": receipt.as_dict(),
        "grounding_raw_attempts": attempts,
        "grounding_usage": ground_usage,
        "target_native_execution": execution,
        "prequalification_source_selection": prequalification,
        "direct_response": direct,
        "direct_raw_payload": direct_payload,
        "direct_usage": direct_usage,
        "grounder_visible_fields": [
            "public_question", "chronological_proxy_frames", "frame_timestamps",
        ],
        "grounder_answer_read": False,
        "grounder_functional_program_read": False,
        "grounder_scene_graph_read": False,
        "grounder_source_identity_read": False,
        "direct_call_started_after_grounding_froze": True,
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
    aliases = {"true": "yes", "false": "no"}
    return aliases.get(text, text)


def _upgrade_compatible_cache(
    cached: Mapping[str, Any], *, sample: Mapping[str, Any],
    config: Mapping[str, Any], sources: Sequence[Any], grounder_sha256: str,
) -> dict[str, Any]:
    """Revalidate raw output when only receipt canonicalization changed."""

    attempts = list(cached.get("grounding_raw_attempts") or ())
    if not attempts:
        raise ValueError("compatible cache omitted raw grounding attempts")
    receipt = parse_frame_grounding_receipt(
        attempts[-1]["payload"],
        frame_count=int(config["media"]["proxy_frame_count"]),
    )
    body = dict(cached)
    body.pop("runtime_receipt_sha256", None)
    body["grounding_receipt"] = receipt.as_dict()
    body["target_native_execution"] = execute_grounding_receipt(receipt)
    body["prequalification_source_selection"] = select_source_for_grounding(
        sources,
        task_id=str(sample["task_id"]),
        receipt=receipt,
        target_grounder_sha256=grounder_sha256,
        grounder_qualified=False,
    )
    body["cache_migration"] = {
        "from_grounder_sha256": str(cached["grounder_sha256"]),
        "reason": (
            "RAW_PROVIDER_OUTPUT_REVALIDATED_WITH_EXPLICIT_INTERVAL_ENDPOINT_"
            "CANONICALIZATION"
        ),
        "new_provider_call": False,
    }
    body["grounder_sha256"] = grounder_sha256
    return body | {"runtime_receipt_sha256": stable_hash(body)}


def _answer_matches(prediction: Any, gold: Any) -> bool:
    predicted = _normalize(prediction)
    expected = _normalize(gold)
    if expected in {"yes", "no", "before", "after"}:
        first_token = predicted.split(maxsplit=1)[0] if predicted else ""
        return first_token == expected
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


def collect(
    *, config_path: Path, keys_path: Path, output_path: Path,
    workers: int, limit: int | None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    manifest_path = REPO_ROOT / config["manifest"]
    if _sha256(manifest_path) != config["manifest_file_sha256"]:
        raise ValueError("AGQA frame-grounding manifest hash mismatch")
    module_path = REPO_ROOT / config["grounder"]["module"]
    if _sha256(module_path) != config["grounder"]["module_sha256"]:
        raise ValueError("AGQA frame-grounder module hash mismatch")
    collector_path = REPO_ROOT / config["grounder"]["collector"]
    if _sha256(collector_path) != config["grounder"]["collector_sha256"]:
        raise ValueError("AGQA frame-grounder collector hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != (
        "FROZEN_CONSUMED_METADATA_DEVELOPMENT_BEFORE_NEURAL_CALLS"
    ):
        raise ValueError("AGQA frame-grounding manifest is not frozen")
    key_values = runpy.run_path(str(keys_path))
    api_key = key_values.get(config["model"]["api_key_name"])
    if not api_key:
        raise ValueError("OpenRouter API key is unavailable")
    rows = _load_selected_rows(manifest)
    sources, arcade_report = _load_sources(config)
    contract_core = {
        "prompt_version": PROMPT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "grounder_module_sha256": config["grounder"]["module_sha256"],
        "model": config["model"],
        "media": config["media"],
        "visible_fields": config["grounder"]["visible_fields"],
        "forbidden_fields": config["grounder"]["forbidden_fields"],
    }
    grounder_sha256 = stable_hash(contract_core)

    manifest_rows = list(manifest["samples"])
    if limit is not None:
        manifest_rows = manifest_rows[:limit]
    runtime_inputs = []
    for frozen in manifest_rows:
        task_id = str(frozen["task_id"])
        row = rows[task_id]
        question = str(row["question"])
        if stable_hash(question) != frozen["question_sha256"]:
            raise ValueError(f"question hash mismatch: {task_id}")
        runtime_inputs.append(dict(frozen) | {"question": question})

    runtime_rows: dict[str, dict[str, Any]] = {}
    cache_dir = output_path.parent / "runtime_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending_inputs = []
    for sample in runtime_inputs:
        cache_path = cache_dir / f"{sample['task_id']}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text())
            cache_hash = cached.get("grounder_sha256")
            compatible_hashes = set(
                config["grounder"].get("compatible_runtime_grounder_sha256") or ()
            )
            if (
                cache_hash in ({grounder_sha256} | compatible_hashes)
                and cached.get("question_sha256") == stable_hash(sample["question"])
                and cached.get("video_sha256") == sample["video_sha256"]
            ):
                if cache_hash != grounder_sha256:
                    cached = _upgrade_compatible_cache(
                        cached,
                        sample=sample,
                        config=config,
                        sources=sources,
                        grounder_sha256=grounder_sha256,
                    )
                    cache_path.write_text(
                        json.dumps(cached, indent=2, sort_keys=True) + "\n"
                    )
                runtime_rows[str(sample["task_id"])] = cached
                print(f"reused {sample['task_id']}", flush=True)
                continue
        pending_inputs.append(sample)
    worker_errors = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _collect_runtime,
                sample,
                config=config,
                api_key=api_key,
                sources=sources,
                grounder_sha256=grounder_sha256,
            ): str(sample["task_id"])
            for sample in pending_inputs
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                runtime = future.result()
            except Exception as exc:  # preserve all other immutable receipts
                worker_errors[task_id] = f"{type(exc).__name__}: {exc}"
                print(f"failed {task_id}: {worker_errors[task_id]}", flush=True)
                continue
            runtime_rows[task_id] = runtime
            cache_path = cache_dir / f"{task_id}.json"
            cache_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
            print(f"completed {task_id}", flush=True)
    if worker_errors:
        error_path = output_path.parent / "worker_errors.json"
        error_path.write_text(json.dumps({
            "grounder_sha256": grounder_sha256,
            "errors": worker_errors,
        }, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(
            "AGQA frame-grounding workers failed; completed runtime receipts were "
            f"cached: {worker_errors}"
        )

    # Evaluator-only access begins here, after every runtime row has frozen.
    evaluated = []
    for frozen in manifest_rows:
        task_id = str(frozen["task_id"])
        row = rows[task_id]
        runtime = runtime_rows[task_id]
        program = str(row["program"])
        if stable_hash(program) != frozen["program_sha256"]:
            raise ValueError(f"functional-program hash mismatch: {task_id}")
        oracle_route = profile_program(
            task_id=task_id, program=program,
        ).route_kind
        gold = str(row["answer"])
        decision = runtime["target_native_execution"]["decision"]
        direct = runtime["direct_response"]
        decisive = decision is not None
        typed_prediction = decision if decisive else direct
        evaluated.append(runtime | {
            "oracle_route_evaluator_only": oracle_route,
            "gold_answer_evaluator_only": gold,
            "predicted_route_correct": (
                runtime["grounding_receipt"]["obligation_kind"] == oracle_route
            ),
            "decisive_execution": decisive,
            "decisive_correct": (
                _answer_matches(decision, gold) if decisive else None
            ),
            "direct_correct": _answer_matches(direct, gold),
            "typed_fallback_prediction": typed_prediction,
            "typed_fallback_correct": _answer_matches(typed_prediction, gold),
            "official_scene_graph_read_by_evaluator": False,
            "official_answer_first_read_after_all_runtime_rows_froze": True,
        })

    valid = len(evaluated)
    route_correct = sum(row["predicted_route_correct"] for row in evaluated)
    decisive_rows = [row for row in evaluated if row["decisive_execution"]]
    decisive_correct = sum(row["decisive_correct"] for row in decisive_rows)
    direct_correct = sum(row["direct_correct"] for row in evaluated)
    typed_correct = sum(row["typed_fallback_correct"] for row in evaluated)
    wins = sum(
        row["typed_fallback_correct"] and not row["direct_correct"]
        for row in evaluated
    )
    losses = sum(
        row["direct_correct"] and not row["typed_fallback_correct"]
        for row in evaluated
    )
    total_cost = sum(
        row["grounding_usage"]["reported_cost_usd"]
        + row["direct_usage"]["reported_cost_usd"]
        + sum(
            attempt["usage"]["reported_cost_usd"]
            for attempt in row["grounding_raw_attempts"][:-1]
        )
        for row in evaluated
    )
    gates_config = config["qualification_gates"]
    gates = {
        "all_provider_receipts_schema_valid": (
            valid == min(manifest["sample_count"], limit or manifest["sample_count"])
        ),
        "required_valid_receipts": valid >= gates_config["required_valid_receipts"],
        "minimum_route_accuracy": route_correct >= gates_config["minimum_route_correct"],
        "minimum_decisive_coverage": (
            len(decisive_rows) >= gates_config["minimum_decisive_executions"]
        ),
        "minimum_decisive_accuracy": (
            bool(decisive_rows)
            and decisive_correct / len(decisive_rows)
            >= gates_config["minimum_decisive_accuracy"]
        ),
        "no_grounded_vs_direct_negative_transfer": (
            losses <= gates_config["maximum_grounded_vs_direct_losses"]
        ),
        "provider_cost_within_cap": (
            total_cost <= gates_config["maximum_reported_provider_cost_usd"]
        ),
        "runtime_grounder_saw_no_answer_program_scene_graph_or_source_identity": all(
            not row[flag]
            for row in evaluated
            for flag in (
                "grounder_answer_read", "grounder_functional_program_read",
                "grounder_scene_graph_read", "grounder_source_identity_read",
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
            row["grounding_receipt"],
            frame_count=int(config["media"]["proxy_frame_count"]),
        )
        row["postqualification_source_selection_diagnostic"] = (
            select_source_for_grounding(
                sources,
                task_id=row["task_id"],
                receipt=receipt,
                target_grounder_sha256=grounder_sha256,
                grounder_qualified=qualified,
            )
        )
        authorized = (
            row["postqualification_source_selection_diagnostic"][
                "selected_program_sha256"
            ] is not None
            and row["target_native_execution"]["decision"] is not None
        )
        row["unified_harness_executor_authorized"] = authorized
        row["unified_harness_prediction"] = (
            row["target_native_execution"]["decision"]
            if authorized else row["direct_response"]
        )
        row["unified_harness_correct"] = _answer_matches(
            row["unified_harness_prediction"],
            row["gold_answer_evaluator_only"],
        )

    unified_authorizations = sum(
        row["unified_harness_executor_authorized"] for row in evaluated
    )
    unified_correct = sum(row["unified_harness_correct"] for row in evaluated)

    status = (
        "AGQA2_FRAME_GROUNDER_DEVELOPMENT_QUALIFIED"
        if qualified else "AGQA2_FRAME_GROUNDER_DEVELOPMENT_NOT_QUALIFIED"
    )
    result = {
        "schema_version": "agqa2-frame-grounding-report-v2",
        "status": status,
        "claim_boundary": config["claim_boundary"],
        "config_sha256": _sha256(config_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "grounder_sha256": grounder_sha256,
        "model": config["model"]["id"],
        "sample_count": valid,
        "unique_video_count": len({row["video_id"] for row in evaluated}),
        "new_video_downloads": 0,
        "provider_calls": sum(
            len(row["grounding_raw_attempts"]) + 1 for row in evaluated
        ),
        "reported_provider_cost_usd": total_cost,
        "metrics": {
            "valid_receipts": valid,
            "route_correct": route_correct,
            "route_accuracy": route_correct / valid if valid else 0.0,
            "decisive_executions": len(decisive_rows),
            "decisive_coverage": len(decisive_rows) / valid if valid else 0.0,
            "decisive_correct": decisive_correct,
            "decisive_accuracy": (
                decisive_correct / len(decisive_rows) if decisive_rows else 0.0
            ),
            "direct_correct": direct_correct,
            "typed_fallback_correct": typed_correct,
            "typed_vs_direct_wins": wins,
            "typed_vs_direct_losses": losses,
            "unified_harness_executor_authorizations": unified_authorizations,
            "unified_harness_correct": unified_correct,
            "unified_harness_vs_direct_delta": unified_correct - direct_correct,
        },
        "qualification_gates": gates,
        "grounder_qualified": qualified,
        "source_portfolio_caveat": {
            "status": arcade_report["status"],
            "authentic_correct": arcade_report["qualified_aggregate"][
                "authentic_correct"
            ],
            "source_permuted_correct": arcade_report["qualified_aggregate"][
                "permuted_correct"
            ],
            "source_specific_claim_passed": False,
        },
        "rows": sorted(evaluated, key=lambda row: row["task_id"]),
        "untouched_formal_claim": False,
        "source_provenance_claim": False,
    }
    body = dict(result)
    result["report_sha256"] = stable_hash(body)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs/agqa2_frame_grounding_v2_development.json",
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs/agqa2_frame_grounding_v2_development/report.json",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = collect(
        config_path=args.config.resolve(),
        keys_path=args.keys.resolve(),
        output_path=args.output.resolve(),
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps({
        "status": result["status"],
        "sample_count": result["sample_count"],
        "metrics": result["metrics"],
        "provider_calls": result["provider_calls"],
        "reported_provider_cost_usd": result["reported_provider_cost_usd"],
        "report_sha256": result["report_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
