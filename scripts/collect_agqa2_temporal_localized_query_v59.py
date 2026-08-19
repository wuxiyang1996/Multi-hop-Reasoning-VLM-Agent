#!/usr/bin/env python3
"""Collect AGQA temporal-window-then-relation neural-symbolic receipts.

Runtime has access to one public question and its video only.  All direct and
typed predictions for the complete batch freeze before the evaluator opens
functional programs or answers.  Unsupported public questions use the exact
same direct baseline and never invoke a source executor.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_active_frame_grounder import (  # noqa: E402
    merge_operand_receipts,
    parse_query_plan,
    remap_operand_receipt,
)
from motif_transfer.agqa_frame_grounder import execute_grounding_receipt  # noqa: E402
from motif_transfer.agqa_operand_normalization import (  # noqa: E402
    parse_normalized_operand_receipt,
)
from motif_transfer.agqa_query_object_grounder import (  # noqa: E402
    parse_object_ontology_receipt,
)
from motif_transfer.agqa_query_object_source_specific import (  # noqa: E402
    exact_one_sided_pvalue,
)
from motif_transfer.agqa_temporal_localized_query import (  # noqa: E402
    calibrate_window_object_consensus,
    consensus_anchor_interval,
    execute_temporal_window,
    parse_temporal_localized_object_question,
    select_composite_source_programs,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_ir_applicability import (  # noqa: E402
    SourceIRContract,
)
from scripts.audit_agqa2_program_transfer_v1 import (  # noqa: E402
    _iter_top_level_object,
    _load_sources,
)
import scripts.collect_agqa2_active_grounding_v3 as active  # noqa: E402
import scripts.collect_agqa2_query_object_v20 as object_v20  # noqa: E402


PROMPT_VERSION = "AGQA_TEMPORAL_LOCALIZED_QUERY_V59_0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relation_plan(relation: str):
    return parse_query_plan({
        "obligation_kind": "RELATION_RECURRENT",
        "comparison": "QUERY_OBJECT",
        "operand_a": relation,
        "operand_b": "",
        "visual_query_a": f"a person {relation} an unknown object",
        "visual_query_b": "",
        "parser_uncertainties": [],
    })


def _ontology_format(frame_count: int) -> dict[str, Any]:
    schema = deepcopy(object_v20._ontology_response_format())
    schema["json_schema"]["schema"]["properties"]["evidence_frames"][
        "items"
    ]["maximum"] = frame_count - 1
    return schema


def _ontology_system(frame_count: int) -> str:
    return object_v20._ontology_system().replace(
        "chronological F0..F47 evidence",
        f"chronological F0..F{frame_count - 1} evidence",
    )


def _ontology_call_range(
    *, requested_relation: str, video_path: Path, model: Mapping[str, Any],
    media: Mapping[str, Any], api_key: str, cache_dir: Path,
    start_second: float, end_second: float, view_name: str,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    frame_count = int(media["window_frame_count"])
    frames, seconds, metadata = active._sample_video_range(
        video_path, frame_count=frame_count,
        max_side=int(media["window_frame_max_side"]),
        start_second=start_second, end_second=end_second,
    )
    panels = active._panels(
        frames, seconds,
        frames_per_panel=int(media["window_frames_per_panel"]),
        frame_width=int(media["window_panel_frame_width"]),
        quality=int(media["jpeg_quality"]),
    )
    client = OpenAI(
        api_key=api_key, base_url=str(model["base_url"]),
        timeout=float(model["timeout_seconds"]),
        max_retries=int(model["max_retries"]),
    )
    attempts = []
    last_error = ""
    for attempt in range(int(model["schema_retries"])):
        content = [{"type": "text", "text": (
            f"requested_relation: {requested_relation}\n"
            "These frames are already restricted to the authorized temporal "
            "window. Ground only this relation and return its related object."
        )}] + active._panel_content(panels)
        if last_error:
            content.append({
                "type": "text", "text": "Fix this schema error: " + last_error,
            })
        system = _ontology_system(frame_count)
        input_core = {
            "stage": "temporal_window_object_ontology",
            "prompt_version": PROMPT_VERSION,
            "view_name": view_name,
            "model": model,
            "requested_relation": requested_relation,
            "system": system,
            "window_seconds": [start_second, end_second],
            "panel_sha256": [hashlib.sha256(value).hexdigest() for value in panels],
            "retry_error": last_error,
        }
        payload, usage, reused = active._cached_provider_call(
            cache_dir=cache_dir, call_name=f"{view_name}_{attempt}",
            input_core=input_core,
            invoke=lambda: active._provider_json_call(
                client, model=model, system=system, content=content,
                max_tokens=int(model["max_ontology_tokens"]),
                response_format=_ontology_format(frame_count),
            ),
        )
        attempts.append({
            "payload": payload, "usage": usage, "cache_reused": reused,
            "view": view_name,
        })
        try:
            return (
                parse_object_ontology_receipt(payload, frame_count=frame_count),
                attempts, metadata,
            )
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("window ontology schema retries exhausted: " + last_error)


def _anchor_view(
    *, client: OpenAI, requested: str, panels, local_frame_count: int,
    model: Mapping[str, Any], cache_dir: Path, name: str,
):
    return active._operand_call(
        client, role="A", requested_operand=requested, panels=panels,
        frame_count=local_frame_count, mode="TEMPORAL_ANCHOR_RECURRENCE",
        model=model, cache_dir=cache_dir, call_prefix=name,
    )


def _ground_anchor(
    *, requested: str, video_path: Path, dense_seconds: Sequence[float],
    dense_duration: float, dense_panels, config: Mapping[str, Any],
    api_key: str, cache_dir: Path,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    primary_model = config["model"]
    secondary_model = config["anchor_secondary_model"]
    client = OpenAI(
        api_key=api_key, base_url=str(primary_model["base_url"]),
        timeout=float(primary_model["timeout_seconds"]),
        max_retries=int(primary_model["max_retries"]),
    )
    primary, primary_attempts = _anchor_view(
        client=client, requested=requested, panels=dense_panels,
        local_frame_count=len(dense_seconds), model=primary_model,
        cache_dir=cache_dir, name="anchor_primary",
    )
    media = config["media"]
    rescan_start, rescan_end = active.recurrent_rescan_window(
        primary, seconds=dense_seconds, duration=dense_duration,
        require_specific_object=False,
    )
    secondary_frames, secondary_seconds, secondary_metadata = (
        active._sample_video_range(
            video_path, frame_count=int(media["anchor_secondary_frame_count"]),
            max_side=int(media["anchor_secondary_frame_max_side"]),
            start_second=rescan_start, end_second=rescan_end,
        )
    )
    secondary_panels = active._panels(
        secondary_frames, secondary_seconds,
        frames_per_panel=int(media["anchor_secondary_frames_per_panel"]),
        frame_width=int(media["anchor_secondary_panel_frame_width"]),
        quality=int(media["jpeg_quality"]),
    )
    secondary_client = OpenAI(
        api_key=api_key, base_url=str(secondary_model["base_url"]),
        timeout=float(secondary_model["timeout_seconds"]),
        max_retries=int(secondary_model["max_retries"]),
    )
    secondary_local, secondary_attempts = _anchor_view(
        client=secondary_client, requested=requested, panels=secondary_panels,
        local_frame_count=len(secondary_frames), model=secondary_model,
        cache_dir=cache_dir, name="anchor_secondary",
    )
    secondary = remap_operand_receipt(
        secondary_local, local_seconds=secondary_seconds,
        global_seconds=dense_seconds,
    )
    consensus = consensus_anchor_interval(
        (primary.as_dict(), secondary.as_dict()),
        minimum_confidence=float(config["calibration"]["anchor_minimum_confidence"]),
        maximum_endpoint_spread=int(
            config["calibration"]["anchor_maximum_endpoint_spread"]
        ),
    )
    attempts = [
        *(dict(row) | {"view": "anchor_primary"} for row in primary_attempts),
        *(dict(row) | {"view": "anchor_secondary"} for row in secondary_attempts),
    ]
    views = [
        {"view": "anchor_primary", "receipt": primary.as_dict()},
        {"view": "anchor_secondary", "receipt": secondary.as_dict(),
         "sampling_metadata": secondary_metadata},
    ]
    if not consensus.authorized:
        tiebreak_model = config["anchor_tiebreak_model"]
        tiebreak_frames, tiebreak_seconds, tiebreak_metadata = (
            active._sample_video_range(
                video_path, frame_count=int(media["tiebreak_frame_count"]),
                max_side=int(media["tiebreak_frame_max_side"]),
                start_second=rescan_start, end_second=rescan_end,
            )
        )
        tiebreak_panels = active._panels(
            tiebreak_frames, tiebreak_seconds,
            frames_per_panel=int(media["tiebreak_frames_per_panel"]),
            frame_width=int(media["tiebreak_panel_frame_width"]),
            quality=int(media["jpeg_quality"]),
        )
        tiebreak_client = OpenAI(
            api_key=api_key, base_url=str(tiebreak_model["base_url"]),
            timeout=float(tiebreak_model["timeout_seconds"]),
            max_retries=int(tiebreak_model["max_retries"]),
        )
        tiebreak_local, tiebreak_attempts = _anchor_view(
            client=tiebreak_client, requested=requested,
            panels=tiebreak_panels, local_frame_count=len(tiebreak_frames),
            model=tiebreak_model, cache_dir=cache_dir,
            name="anchor_tiebreak",
        )
        tiebreak = remap_operand_receipt(
            tiebreak_local, local_seconds=tiebreak_seconds,
            global_seconds=dense_seconds,
        )
        attempts.extend(
            dict(row) | {"view": "anchor_tiebreak"}
            for row in tiebreak_attempts
        )
        views.append({
            "view": "anchor_tiebreak", "receipt": tiebreak.as_dict(),
            "sampling_metadata": tiebreak_metadata,
        })
        consensus = consensus_anchor_interval(
            (primary.as_dict(), secondary.as_dict(), tiebreak.as_dict()),
            minimum_confidence=float(
                config["calibration"]["anchor_minimum_confidence"]
            ),
            maximum_endpoint_spread=int(
                config["calibration"]["anchor_maximum_endpoint_spread"]
            ),
        )
    return consensus, attempts, views


def _relation_view(
    *, plan, client: OpenAI, video_path: Path, config: Mapping[str, Any],
    cache_dir: Path, start_second: float, end_second: float,
):
    media = config["media"]
    frames, seconds, metadata = active._sample_video_range(
        video_path, frame_count=int(media["window_frame_count"]),
        max_side=int(media["window_frame_max_side"]),
        start_second=start_second, end_second=end_second,
    )
    panels = active._panels(
        frames, seconds,
        frames_per_panel=int(media["window_frames_per_panel"]),
        frame_width=int(media["window_panel_frame_width"]),
        quality=int(media["jpeg_quality"]),
    )
    operand, attempts = active._operand_call(
        client, role="A", requested_operand=plan.visual_query_a,
        panels=panels, frame_count=len(frames),
        mode="TEMPORAL_WINDOW_RELATION_OBJECT", model=config["model"],
        cache_dir=cache_dir, call_prefix="window_relation",
    )
    merged = merge_operand_receipts(
        plan, operand_a=operand, operand_b=None, frame_count=len(frames),
    )
    execution = execute_grounding_receipt(merged)
    return operand, attempts, metadata, execution, panels


def _window_question_call(
    *, client: OpenAI, question: str, panels, model: Mapping[str, Any],
    cache_dir: Path, view_name: str,
) -> tuple[str, dict[str, Any], dict[str, Any], bool]:
    content = [{"type": "text", "text": (
        "A symbolic temporal executor has already restricted these frames to "
        "the interval requested by the question. Identify only the object in "
        "the requested person-object relation inside these frames.\n"
        f"Public question: {question.strip()}\n"
        "Return the shortest canonical object name."
    )}] + active._panel_content(panels)
    input_core = {
        "stage": "temporal_window_question",
        "prompt_version": PROMPT_VERSION,
        "view_name": view_name,
        "model": model,
        "question": question,
        "panel_sha256": [hashlib.sha256(value).hexdigest() for value in panels],
    }
    payload, usage, reused = active._cached_provider_call(
        cache_dir=cache_dir, call_name=view_name, input_core=input_core,
        invoke=lambda: active._provider_json_call(
            client, model=model,
            system="Return JSON only as {\"response\": string}.",
            content=content, max_tokens=int(model["max_direct_tokens"]),
            response_format=active._direct_response_format(),
        ),
    )
    response = str(payload.get("response") or "").strip()
    if not response:
        raise ValueError("window question grounder returned an empty response")
    return response, payload, usage, reused


def _collect_runtime(
    sample: Mapping[str, Any], *, config: Mapping[str, Any], api_key: str,
    sources: Sequence[SourceIRContract], grounder_sha256: str,
    cache_root: Path,
) -> dict[str, Any]:
    video_path = Path(sample["video_path"])
    if _sha256(video_path) != sample["video_sha256"]:
        raise ValueError(f"video hash mismatch: {sample['video_id']}")
    question = str(sample["question"])
    public_plan = parse_temporal_localized_object_question(question)
    media = config["media"]
    dense_frames, dense_seconds, dense_metadata = active._sample_video_range(
        video_path, frame_count=int(media["dense_proxy_frame_count"]),
        max_side=int(media["proxy_frame_max_side"]),
    )
    dense_panels = active._panels(
        dense_frames, dense_seconds,
        frames_per_panel=int(media["frames_per_panel"]),
        frame_width=int(media["panel_frame_width"]),
        quality=int(media["jpeg_quality"]),
    )
    task_cache = cache_root / str(sample["task_id"])
    model = config["model"]
    client = OpenAI(
        api_key=api_key, base_url=str(model["base_url"]),
        timeout=float(model["timeout_seconds"]),
        max_retries=int(model["max_retries"]),
    )
    anchor_receipts = []
    anchor_attempts = []
    anchor_views = []
    window = None
    relation_receipt = None
    relation_attempts = []
    relation_metadata = None
    relation_execution = None
    ontology_receipts = []
    ontology_attempts = []
    ontology_metadata = []
    window_question_views = []
    calibrated = None
    if public_plan is not None:
        requested_anchors = [public_plan.visual_anchor_a_query]
        if public_plan.anchor_b:
            requested_anchors.append(public_plan.visual_anchor_b_query)
        for index, requested in enumerate(requested_anchors):
            consensus, attempts, views = _ground_anchor(
                requested=requested, video_path=video_path,
                dense_seconds=dense_seconds,
                dense_duration=float(dense_metadata["duration_seconds"]),
                dense_panels=dense_panels,
                config=config, api_key=api_key,
                cache_dir=task_cache / f"anchor_{index}",
            )
            anchor_receipts.append(consensus)
            anchor_attempts.extend(attempts)
            anchor_views.extend(
                dict(row) | {"anchor_index": index} for row in views
            )
        if all(row.authorized for row in anchor_receipts):
            window = execute_temporal_window(
                temporal_operator=public_plan.temporal_operator,
                frame_count=len(dense_frames),
                anchor_a_interval=anchor_receipts[0].consensus_interval,
                anchor_b_interval=(
                    anchor_receipts[1].consensus_interval
                    if len(anchor_receipts) == 2 else None
                ),
                minimum_window_frames=int(
                    config["calibration"]["minimum_window_frames"]
                ),
            )
        if window is not None and window.authorized:
            assert window.window_start_frame is not None
            assert window.window_end_frame is not None
            start_second = float(dense_seconds[window.window_start_frame])
            end_second = float(dense_seconds[window.window_end_frame])
            relation_plan = _relation_plan(public_plan.relation)
            (
                relation_receipt, relation_attempts, relation_metadata,
                relation_execution, window_panels,
            ) = (
                _relation_view(
                    plan=relation_plan, client=client, video_path=video_path,
                    config=config, cache_dir=task_cache,
                    start_second=start_second, end_second=end_second,
                )
            )
            for view_name, ontology_model in (
                ("window_ontology_primary", config["ontology_models"][0]),
            ):
                receipt, attempts, metadata = _ontology_call_range(
                    requested_relation=public_plan.relation,
                    video_path=video_path, model=ontology_model, media=media,
                    api_key=api_key, cache_dir=task_cache,
                    start_second=start_second, end_second=end_second,
                    view_name=view_name,
                )
                ontology_receipts.append(receipt)
                ontology_attempts.extend(attempts)
                ontology_metadata.append(metadata)
            family_responses = {}
            for family, view_name, view_model in (
                ("qwen3_vl", "window_question_primary", config["model"]),
                (
                    "gemini3_flash", "window_question_secondary",
                    config["anchor_secondary_model"],
                ),
            ):
                view_client = OpenAI(
                    api_key=api_key, base_url=str(view_model["base_url"]),
                    timeout=float(view_model["timeout_seconds"]),
                    max_retries=int(view_model["max_retries"]),
                )
                response, payload, usage, reused = _window_question_call(
                    client=view_client, question=question, panels=window_panels,
                    model=view_model, cache_dir=task_cache, view_name=view_name,
                )
                family_responses[family] = response
                window_question_views.append({
                    "model_family": family, "view": view_name,
                    "response": response, "payload": payload, "usage": usage,
                    "cache_reused": reused,
                })
            calibrated = calibrate_window_object_consensus(
                model_family_responses=family_responses,
                ontology_family_receipts={
                    "gemini2_5_flash_lite": ontology_receipts[0].as_dict(),
                },
                ontology_minimum_confidence=float(
                    config["calibration"]["ontology_minimum_confidences"][0]
                ),
                minimum_model_families=int(
                    config["calibration"]["minimum_neural_votes"]
                ),
            )

    # Direct is intentionally last and never enters a neural consensus vote.
    direct, direct_payload, direct_usage, direct_reused = active._direct_call(
        client, question=question, panels=dense_panels,
        model=model, cache_dir=task_cache,
    )
    candidate = calibrated["decision"] if calibrated else None
    prequalification = select_composite_source_programs(
        sources, grounder_qualified=False,
    )
    core = {
        "task_id": str(sample["task_id"]),
        "video_id": str(sample["video_id"]),
        "video_sha256": str(sample["video_sha256"]),
        "question_sha256": stable_hash(question),
        "public_plan": public_plan.as_dict() if public_plan else None,
        "composite_applicable": public_plan is not None,
        "anchor_consensus_receipts": [row.as_dict() for row in anchor_receipts],
        "anchor_views": anchor_views,
        "anchor_attempts": anchor_attempts,
        "temporal_window_receipt": window.as_dict() if window else None,
        "relation_operand_receipt": (
            relation_receipt.as_dict() if relation_receipt else None
        ),
        "relation_attempts": relation_attempts,
        "relation_video_metadata": relation_metadata,
        "relation_execution": relation_execution,
        "object_ontology_receipts": [row.as_dict() for row in ontology_receipts],
        "object_ontology_attempts": ontology_attempts,
        "object_ontology_video_metadata": ontology_metadata,
        "window_question_views": window_question_views,
        "calibrated_composite_execution": calibrated,
        "candidate_typed_prediction": candidate,
        "direct_response": direct,
        "direct_payload": direct_payload,
        "direct_usage": direct_usage,
        "direct_cache_reused": direct_reused,
        "dense_video_metadata": dense_metadata,
        "prequalification_source_selection": prequalification,
        "grounder_sha256": grounder_sha256,
        "runtime_answer_read": False,
        "runtime_functional_program_read": False,
        "runtime_scene_graph_read": False,
        "runtime_source_identity_read": False,
        "public_parser_program_read": False,
        "anchor_grounder_question_read": False,
        "relation_grounder_question_read": False,
        "ontology_grounder_question_read": False,
        "direct_call_started_after_all_typed_receipts_froze": True,
    }
    return core | {"runtime_receipt_sha256": stable_hash(core)}


def _usage_rows(runtime: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = [runtime["direct_usage"]]
    for key in ("anchor_attempts", "relation_attempts", "object_ontology_attempts"):
        rows.extend(attempt["usage"] for attempt in runtime[key])
    rows.extend(view["usage"] for view in runtime["window_question_views"])
    return rows


def _answer_matches(left: Any, right: Any) -> bool:
    return active._answer_matches(left, right)


def _paired(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    wins = sum(row["source_correct"] and not row["direct_correct"] for row in rows)
    losses = sum(row["direct_correct"] and not row["source_correct"] for row in rows)
    return {
        "source_correct": sum(row["source_correct"] for row in rows),
        "direct_correct": sum(row["direct_correct"] for row in rows),
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "net_gain": wins - losses,
        "exact_one_sided_pvalue": exact_one_sided_pvalue(
            source_wins=wins, source_losses=losses,
        ),
    }


def _validate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / config["manifest"]
    if _sha256(path) != config["manifest_file_sha256"]:
        raise ValueError("V59 manifest file hash mismatch")
    manifest = json.loads(path.read_text())
    body = dict(manifest)
    claimed = body.pop("manifest_sha256")
    if stable_hash(body) != claimed:
        raise ValueError("V59 manifest content hash mismatch")
    return manifest


def _load_metadata(config: Mapping[str, Any], manifest: Mapping[str, Any]):
    archive = Path(config["dataset"]["archive_path"])
    if _sha256(archive) != config["dataset"]["archive_sha256"]:
        raise ValueError("AGQA archive hash mismatch")
    wanted = {str(row["task_id"]) for row in manifest["samples"]}
    selected = {}
    import io
    import zipfile
    with zipfile.ZipFile(archive) as bundle:
        with bundle.open(config["dataset"]["entry"], "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            for task_id, row in _iter_top_level_object(text):
                if task_id in wanted:
                    selected[task_id] = row
                    if len(selected) == len(wanted):
                        break
    if set(selected) != wanted:
        raise ValueError("manifest tasks missing from AGQA metadata")
    return selected


def collect(
    *, config_path: Path, keys_path: Path, output_path: Path,
    workers: int = 3,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    manifest = _validate_manifest(config)
    metadata = _load_metadata(config, manifest)
    keys = runpy.run_path(str(keys_path))
    api_key = keys.get(config["model"]["api_key_name"])
    if not api_key:
        raise ValueError("OpenRouter API key is unavailable")
    sources, arcade_report = _load_sources(config)
    grounder_core = {
        "prompt_version": PROMPT_VERSION,
        "compiler_module_sha256": _sha256(
            REPO_ROOT / "src/motif_transfer/agqa_temporal_localized_query.py"
        ),
        "collector_sha256": _sha256(Path(__file__)),
        "models": {
            "primary": config["model"],
            "anchor_secondary": config["anchor_secondary_model"],
            "anchor_tiebreak": config["anchor_tiebreak_model"],
            "ontology": config["ontology_models"],
        },
        "media": config["media"],
        "calibration": config["calibration"],
        "source_contract_sha256": sorted(row.contract_sha256 for row in sources),
    }
    grounder_sha256 = stable_hash(grounder_core)
    expected = config.get("expected_grounder_sha256")
    if expected not in {None, grounder_sha256}:
        raise ValueError("V59 grounder differs from frozen qualification")
    runtime_inputs = []
    for frozen in manifest["samples"]:
        task_id = str(frozen["task_id"])
        row = metadata[task_id]
        question = str(row.get("question", ""))
        if stable_hash(question) != frozen["question_sha256"]:
            raise ValueError(f"question hash mismatch: {task_id}")
        runtime_inputs.append(dict(frozen) | {"question": question})

    runtime_dir = output_path.parent / "runtime_receipts"
    cache_root = output_path.parent / "call_cache"
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

    original_parser = active.parse_operand_receipt
    active.parse_operand_receipt = parse_normalized_operand_receipt
    errors = {}
    try:
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
                path = runtime_dir / f"{task_id}.json"
                path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
                print(f"completed {task_id}", flush=True)
    finally:
        active.parse_operand_receipt = original_parser
    if errors:
        raise RuntimeError("V59 workers failed: " + json.dumps(errors, sort_keys=True))

    # No outcome is opened until every runtime receipt is immutable.
    gate = config["qualification_gates"]
    preliminary_qualified = (
        len(runtime_rows) >= int(gate["required_valid_rows"])
        and sum(row["candidate_typed_prediction"] is not None for row in runtime_rows.values())
        >= int(gate["minimum_candidate_predictions"])
    )
    selection = select_composite_source_programs(
        sources, grounder_qualified=preliminary_qualified,
    )
    evaluated = []
    for frozen in manifest["samples"]:
        task_id = str(frozen["task_id"])
        target = metadata[task_id]
        runtime = runtime_rows[task_id]
        program = str(target.get("program", ""))
        if stable_hash(program) != frozen["program_sha256"]:
            raise ValueError(f"program hash mismatch: {task_id}")
        gold = str(target.get("answer", ""))
        authorized = (
            selection["status"] == "AUTHORIZED"
            and runtime["candidate_typed_prediction"] is not None
        )
        prediction = (
            runtime["candidate_typed_prediction"]
            if authorized else runtime["direct_response"]
        )
        evaluated.append(runtime | {
            "gold_answer_evaluator_only": gold,
            "program_sha256_evaluator_only": stable_hash(program),
            "source_executor_authorized": authorized,
            "source_prediction": prediction,
            "source_correct": _answer_matches(prediction, gold),
            "direct_correct": _answer_matches(runtime["direct_response"], gold),
            "official_answer_first_read_after_all_runtime_rows_froze": True,
            "official_scene_graph_read_by_evaluator": False,
        })
    paired = _paired(evaluated)
    authorizations = sum(row["source_executor_authorized"] for row in evaluated)
    gates = {
        "required_valid_rows": len(evaluated) >= int(gate["required_valid_rows"]),
        "required_unique_videos": len({row["video_id"] for row in evaluated})
        >= int(gate["required_unique_videos"]),
        "minimum_candidate_predictions": authorizations
        >= int(gate["minimum_candidate_predictions"]),
        "minimum_wins": paired["wins"] >= int(gate["minimum_wins"]),
        "maximum_losses": paired["losses"] <= int(gate["maximum_losses"]),
        "minimum_net_gain": paired["net_gain"] >= int(gate["minimum_net_gain"]),
        "maximum_exact_one_sided_pvalue": paired["exact_one_sided_pvalue"]
        <= float(gate.get("maximum_exact_one_sided_pvalue", 1.0)),
        "all_runtime_blind": all(
            row[key] is False for row in evaluated for key in (
                "runtime_answer_read", "runtime_functional_program_read",
                "runtime_scene_graph_read", "runtime_source_identity_read",
                "public_parser_program_read", "anchor_grounder_question_read",
                "relation_grounder_question_read", "ontology_grounder_question_read",
            )
        ),
        "prequalification_abstained": all(
            row["prequalification_source_selection"]["status"] == "ABSTAINED"
            for row in evaluated
        ),
    }
    qualified = all(gates.values())
    usages = [usage for row in runtime_rows.values() for usage in _usage_rows(row)]
    cost = sum(float(row.get("reported_cost_usd", 0.0)) for row in usages)
    result_body = {
        "schema_version": "agqa2-temporal-localized-query-report-v59",
        "status": (
            "AGQA2_TEMPORAL_LOCALIZED_QUERY_QUALIFIED" if qualified
            else "AGQA2_TEMPORAL_LOCALIZED_QUERY_NOT_QUALIFIED"
        ),
        "split": config["split"],
        "claim_boundary": config["claim_boundary"],
        "grounder_sha256": grounder_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_count": len(evaluated),
        "unique_video_count": len({row["video_id"] for row in evaluated}),
        "applicable_rows": sum(row["composite_applicable"] for row in evaluated),
        "source_executor_authorizations": authorizations,
        "source_vs_direct": paired,
        "qualification_gates": gates,
        "grounder_qualified": qualified,
        "source_selection": selection,
        "arcade_source_report_status": arcade_report["status"],
        "provider_calls": len(usages),
        "reported_provider_cost_usd": cost,
        "rows": sorted(evaluated, key=lambda row: row["task_id"]),
        "target_native_composition_rule": "TEMPORAL_WINDOW_THEN_RELATION_SCAN",
        "source_induced_primitives": [
            "RECURRENT_TEMPORAL_ACQUISITION", "RECURRENT_GOAL_RELATION_SCAN",
        ],
        "source_provenance_claim": False,
        "full_agqa_distribution_claim": False,
    }
    result = result_body | {"report_sha256": stable_hash(result_body)}
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
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = collect(
        config_path=args.config.resolve(), keys_path=args.keys.resolve(),
        output_path=args.output.resolve(), workers=args.workers,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "sample_count", "applicable_rows",
        "source_executor_authorizations", "source_vs_direct",
        "qualification_gates", "provider_calls", "reported_provider_cost_usd",
        "grounder_sha256", "report_sha256",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
