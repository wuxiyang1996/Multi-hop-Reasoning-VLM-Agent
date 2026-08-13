#!/usr/bin/env python3
"""Collect candidate-factorized BIND->RELATE video receipts."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

from openai import OpenAI
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import augment_bound_relation_smoke as bound  # noqa: E402
import collect_overlay_relation_forks as overlay  # noqa: E402
import run_active_video_wrapper_transfer as media_helpers  # noqa: E402
import run_structured_video_transfer as runner  # noqa: E402
from motif_transfer.video_temporal_localization import (  # noqa: E402
    absolute_temporal_window,
    parse_temporal_localization,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_items(sample: Mapping[str, Any]) -> tuple[str, list[tuple[str, str]]]:
    if "candidates" in sample:
        values = list(sample["candidates"])
        return "binary_vector", [(str(index), str(value)) for index, value in enumerate(values)]
    options = dict(sample["options"])
    return "single_choice", [(str(slot), str(value)) for slot, value in options.items()]


def _localize_question_window(
    client: OpenAI,
    *,
    config: Mapping[str, Any],
    question: str,
    scout: bytes,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Find a broad event window without seeing answer candidates or gold."""

    model = config["model"]
    temporal = config["media"]["temporal_localization"]
    prompt = (
        "Locate the broad temporal interval needed to visually investigate this "
        "question. You do not receive answer choices and must not answer the "
        "question. Use the complete clip only when comparing distant events is "
        "essential; otherwise identify the event/action anchor and a broad window "
        "around it. Return normalized fractions of the whole supplied clip. "
        f"Question only: {question}"
    )
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        payload, usage = media_helpers._json_call(
            client,
            model=str(model["id"]),
            system=(
                "Return JSON only: {window_fraction:[number,number],"
                "requires_full_context:bool,anchor_description:string,"
                "sensor_reliability:number}. Do not predict an answer."
            ),
            content=[
                {
                    "type": "text",
                    "text": prompt + (f" Schema error: {last_error}" if last_error else ""),
                },
                {"type": "text", "text": "Low-bandwidth whole-clip scout frames:"},
                media_helpers._image_content(scout),
            ],
            max_tokens=int(model["max_localization_tokens"]),
        )
        try:
            parsed = parse_temporal_localization(
                payload,
                minimum_width=float(temporal["minimum_width_fraction"]),
                maximum_width=float(temporal["maximum_width_fraction"]),
            )
            return parsed, payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError(f"question-only temporal localizer failed: {last_error}")


def _jpeg_bytes(image: Image.Image, *, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _multiframe_bind_track(
    client: OpenAI, *, config: Mapping[str, Any], entity: str,
    frames: Sequence[Image.Image], indices: Sequence[int], seconds: Sequence[float],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Track identity with separately encoded frames, avoiding collage coordinates."""

    model = config["model"]
    prompt = (
        "Target-native BIND intervention. Track this exact entity across the "
        f"separately supplied temporal frames: {entity}. Each image is an entire "
        "video frame. For every I0..I{n}, return visibility and a tight normalized "
        "bbox in that image's own coordinate system. Do not substitute a nearby "
        "actor or a different color, shape, or object class. observed_true means "
        "the same exact identity is visible in at least three frames."
    ).format(n=len(indices) - 1)
    encoded = [_jpeg_bytes(frames[index], quality=int(config["media"]["jpeg_quality"])) for index in indices]
    evidence_hash = hashlib.sha256(b"".join(encoded)).hexdigest()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for slot, (index, payload) in enumerate(zip(indices, encoded)):
        content.extend([
            {"type": "text", "text": f"I{slot}, t={seconds[index]:.2f}s"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")}},
        ])
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        attempt = list(content)
        if last_error:
            attempt.append({"type": "text", "text": f"Schema error: {last_error}"})
        payload, usage = media_helpers._json_call(
            client, model=str(model["id"]),
            system=(
                "Return JSON only: {observed_true:bool,sensor_reliability:number,"
                "tracks:[{panel_index:int,visible:bool,bbox_xyxy_normalized:"
                "[number,number,number,number] or null}],measurement:string}."
            ),
            content=attempt, max_tokens=int(model["max_track_tokens"]),
        )
        try:
            observed, reliability, tracks = bound._parse_track(
                payload, expected_count=len(indices),
            )
            return {
                "observed_true": observed, "sensor_reliability": reliability,
                "tracks": [list(box) if box is not None else None for box in tracks],
                "raw": payload,
            }, usage, evidence_hash
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError(f"separate-frame BIND schema failed: {last_error}")


def _dual_view_panel(
    frames: Sequence[Image.Image], indices: Sequence[int], seconds: Sequence[float],
    *, config: Mapping[str, Any], track_indices: Sequence[int] | None = None,
    tracks: Sequence[Sequence[float] | None] | None = None, prefix: str,
) -> tuple[bytes, int]:
    selected = [frames[index] for index in indices]
    if tracks is None or track_indices is None:
        # Matched target control: same number and layout of pixels, but no
        # source-specified persistent handle. The second view is a generic center zoom.
        globals_ = [frame.convert("RGB") for frame in selected]
        zooms = []
        for frame in globals_:
            width, height = frame.size
            zooms.append(frame.crop((width // 4, height // 4, 3 * width // 4, 3 * height // 4)))
        fallbacks = len(indices)
    else:
        globals_, flags = overlay._overlay_frames(
            frames, indices, track_indices, tracks, entity_id="CARRIER",
        )
        zooms, crop_flags = bound._bound_crops(
            frames, indices, track_indices, tracks,
        )
        fallbacks = sum(a or b for a, b in zip(flags, crop_flags))
    views: list[Image.Image] = []
    labels: list[str] = []
    for slot, (index, global_frame, zoom) in enumerate(zip(indices, globals_, zooms)):
        views.extend([global_frame, zoom])
        labels.extend([
            f"{prefix}G{slot} {seconds[index]:.2f}s",
            f"{prefix}Z{slot} {seconds[index]:.2f}s",
        ])
    return media_helpers._panel_bytes(
        views, labels=labels,
        frame_width=int(config["media"]["evidence_frame_width"]),
        quality=int(config["media"]["jpeg_quality"]),
    ), fallbacks


def _compile_claims(
    client: OpenAI, *, config: Mapping[str, Any], sample: Mapping[str, Any],
    entity_catalog: Sequence[Mapping[str, Any]], candidates: Sequence[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = config["model"]
    distinct_decoy = bool(config.get("controls", {}).get("require_distinct_decoy", False))
    decoy_instruction = (
        " For every carrier also select a decoy_entity_visual_description: a "
        "concrete, visible entity from the same clip that is visually distinct "
        "from the carrier in class, color, clothing, shape, or role. It is used "
        "only for a wrong-correspondence control and must not be the carrier."
        if distinct_decoy else ""
    )
    prompt = (
        "Compile every answer candidate into an independent visually testable "
        "claim for a candidate-factorized neural-symbolic video program. You do "
        "not know the answer and must not rank, eliminate, score, or favor any "
        "candidate. For each slot, combine the question semantics with that one "
        "candidate so claim_supported=true is exactly equivalent to that slot "
        "being correct. Select one concrete visible carrier entity whose identity "
        "must be established before measuring the claim relation. The carrier can "
        "be option-specific. Its visual description must identify something that "
        "could be localized in frames, never an abstract reason. Never substitute "
        "a different color, shape, object class, or person for an entity named in "
        "the candidate. If the exact candidate entity is absent from the catalog, "
        "preserve its exact textual identity anyway so the BIND action can fail. "
        "relation_description "
        "must state what temporal, interaction, causal, or physical relation should "
        "be measured after binding. Use the full clip unless a broad normalized "
        "window is clearly sufficient."
        + decoy_instruction
        + " Return all slots exactly once.\nQuestion: "
        + str(sample["question"])
        + "\nCandidates: " + json.dumps(
            [{"slot": slot, "text": text} for slot, text in candidates],
            ensure_ascii=False,
        )
        + "\nUnlabeled target-native visual entity catalog: "
        + json.dumps(list(entity_catalog), ensure_ascii=False)
    )
    expected = [slot for slot, _ in candidates]
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        payload, usage = media_helpers._json_call(
            client, model=str(model["id"]),
            system=(
                "Return JSON only: {candidate_programs:[{slot:string,claim:string,"
                "bind_entity_visual_description:string,"
                + ("decoy_entity_visual_description:string," if distinct_decoy else "")
                + "relation_description:string,"
                "window_fraction:[number,number]}]}. No correctness estimates."
            ),
            content=[{"type": "text", "text": prompt + (f"\nSchema error: {last_error}" if last_error else "")}],
            max_tokens=int(model["max_compile_tokens"]),
        )
        try:
            rows = list(payload.get("candidate_programs") or ())
            if [str(row.get("slot")) for row in rows] != expected:
                raise ValueError("compiler slots must exactly preserve candidate order")
            parsed = []
            for row in rows:
                claim = str(row.get("claim") or "").strip()
                entity = str(row.get("bind_entity_visual_description") or "").strip()
                decoy = str(row.get("decoy_entity_visual_description") or "").strip()
                relation = str(row.get("relation_description") or "").strip()
                window = tuple(map(float, row.get("window_fraction") or ()))
                if not claim or not entity or not relation:
                    raise ValueError("candidate program strings must be nonempty")
                if distinct_decoy and (
                    not decoy or decoy.casefold() == entity.casefold()
                ):
                    raise ValueError("decoy must be nonempty and distinct from carrier")
                if len(window) != 2 or not 0 <= window[0] < window[1] <= 1:
                    raise ValueError("candidate window must be a valid normalized pair")
                compiled = {
                    "slot": str(row["slot"]), "claim": claim,
                    "bind_entity_visual_description": entity,
                    "relation_description": relation,
                    "window_fraction": list(window),
                }
                if distinct_decoy:
                    compiled["decoy_entity_visual_description"] = decoy
                parsed.append(compiled)
            return parsed, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError(f"candidate compiler schema failed: {last_error}")


def _parse_relation(payload: Mapping[str, Any]) -> dict[str, Any]:
    probability = float(payload.get("support_probability", -1))
    reliability = float(payload.get("sensor_reliability", -1))
    if not 0 <= probability <= 1:
        raise ValueError("support_probability must be in [0,1]")
    if not 0.5 <= reliability <= 1:
        raise ValueError("sensor_reliability must be in [0.5,1]")
    facts = payload.get("observed_facts")
    if not isinstance(facts, list) or not all(isinstance(value, str) for value in facts):
        raise ValueError("observed_facts must be a string list")
    return {
        "support_probability": probability,
        "sensor_reliability": reliability,
        "observed_facts": facts,
        "measurement": str(payload.get("measurement") or "").strip(),
    }


def _verify_identity(
    client: OpenAI, *, config: Mapping[str, Any], entity: str, panel: bytes,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    prompt = (
        "Independently audit a target-native BIND handle. Magenta boxes show the "
        "proposed trajectory. Decide whether the boxed pixels actually depict the "
        "same exact entity in at least three temporal subframes. Exact target: "
        f"{entity}. Do not accept a nearby actor, another object, or a substitute "
        "with a different color, shape, or class. The boxes are proposals, not "
        "evidence of a match. You do not know any question or answer option."
    )
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        payload, usage = media_helpers._json_call(
            client, model=str(model["id"]),
            system=(
                "Return JSON only: {identity_match_probability:number,"
                "sensor_reliability:number,matched_frame_count:int,"
                "observed_facts:[string],measurement:string}."
            ),
            content=[
                {"type": "text", "text": prompt + (f" Schema error: {last_error}" if last_error else "")},
                media_helpers._image_content(panel),
            ],
            max_tokens=int(model["max_relation_tokens"]),
        )
        try:
            probability = float(payload.get("identity_match_probability", -1))
            reliability = float(payload.get("sensor_reliability", -1))
            matched = int(payload.get("matched_frame_count", -1))
            facts = payload.get("observed_facts")
            if not 0 <= probability <= 1:
                raise ValueError("identity probability must be in [0,1]")
            if not 0.5 <= reliability <= 1:
                raise ValueError("identity reliability must be in [0.5,1]")
            if not 0 <= matched <= 16:
                raise ValueError("matched frame count must be in [0,16]")
            if not isinstance(facts, list) or not all(isinstance(value, str) for value in facts):
                raise ValueError("identity facts must be a string list")
            return {
                "identity_match_probability": probability,
                "sensor_reliability": reliability,
                "matched_frame_count": matched,
                "observed_facts": facts,
                "measurement": str(payload.get("measurement") or "").strip(),
            }, payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError(f"identity audit schema failed: {last_error}")


def _ground_relation(
    client: OpenAI, *, config: Mapping[str, Any], program: Mapping[str, Any],
    panel: bytes, observation_kind: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    marker = (
        "A magenta BOUND box is a persistent identity handle from a preceding "
        "BIND action. Use it only to keep identity stable; the box itself is not "
        "evidence that the claim is true. "
        if observation_kind != "UNBOUND_FULL_CONTEXT" else
        "No entity handle is available; infer identity directly from the full frames. "
    )
    prompt = (
        marker
        + "Measure this one candidate claim from the temporal frames without "
        "answering or comparing any multiple-choice question. Claim: "
        + str(program["claim"])
        + " Bound carrier intended by the program: "
        + str(program["bind_entity_visual_description"])
        + " Relation to measure after identity binding: "
        + str(program["relation_description"])
        + " Return calibrated support_probability=P(claim true | only visible "
        "evidence in this panel). Report uncertainty when motion, causality, or "
        "identity is not observable."
    )
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        payload, usage = media_helpers._json_call(
            client, model=str(model["id"]),
            system=(
                "Return JSON only: {support_probability:number,"
                "sensor_reliability:number,observed_facts:[string],measurement:string}."
            ),
            content=[
                {"type": "text", "text": prompt + (f" Schema error: {last_error}" if last_error else "")},
                media_helpers._image_content(panel),
            ],
            max_tokens=int(model["max_relation_tokens"]),
        )
        try:
            return _parse_relation(payload), payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError(f"candidate relation schema failed: {last_error}")


def _panel(
    frames: Sequence[Image.Image], indices: Sequence[int], seconds: Sequence[float],
    *, config: Mapping[str, Any], prefix: str,
) -> bytes:
    return media_helpers._panel_bytes(
        [frames[index] for index in indices],
        labels=[f"{prefix}{slot} {seconds[index]:.2f}s" for slot, index in enumerate(indices)],
        frame_width=int(config["media"]["evidence_frame_width"]),
        quality=int(config["media"]["jpeg_quality"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--output-file", default="candidate_claim_forks.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") == "FROZEN_PROSPECTIVE_QUALIFICATION":
        policy_spec = config.get("frozen_policy") or {}
        policy_path = Path(str(policy_spec.get("path") or ""))
        if not policy_path.is_file() or _sha256(policy_path) != policy_spec.get("sha256"):
            raise ValueError("frozen qualification policy hash mismatch")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if policy.get("status") != "FROZEN_BEFORE_QUALIFICATION_COLLECTION":
            raise ValueError("frozen qualification policy lifecycle mismatch")
    source_path = Path(config["source"]["typed_summary"])
    if _sha256(source_path) != config["source"]["typed_summary_sha256"]:
        raise ValueError("source gate receipt hash mismatch")
    source_gate = json.loads(source_path.read_text(encoding="utf-8"))
    if source_gate.get("status") != "SOURCE_TYPED_GATE_PASSED":
        raise ValueError("source typed gate did not pass")
    receipts = json.loads((args.run_dir / "receipts.json").read_text(encoding="utf-8"))
    if args.sample_offset < 0 or args.sample_count < 1:
        raise ValueError("sample offset/count must select a nonempty forward shard")
    selected = receipts[args.sample_offset:args.sample_offset + args.sample_count]
    if not selected:
        raise ValueError("sample shard is empty")
    output_path = args.run_dir / args.output_file
    existing: dict[str, dict[str, Any]] = {}
    if output_path.is_file():
        existing = {
            str(row["sample_id"]): row
            for row in json.loads(output_path.read_text(encoding="utf-8"))
        }
    keys = runpy.run_path(str(args.keys))
    model = config["model"]
    client = OpenAI(
        api_key=str(keys[model["api_key_name"]]), base_url=str(model["base_url"]),
        timeout=float(model["timeout_seconds"]), max_retries=int(model["max_retries"]),
    )

    def save() -> None:
        ordered = [existing[str(row["sample_id"])] for row in selected if str(row["sample_id"]) in existing]
        output_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for source in selected:
        sample_id = str(source["sample_id"])
        row = existing.get(sample_id)
        if row and row.get("complete"):
            continue
        sample = source["sample"]
        contract, items = _candidate_items(sample)
        full_frames, metadata = runner._sample_clip(
            Path(sample["video_path"]),
            start_sec=float(source["video_metadata"]["clip_start_seconds"]),
            end_sec=float(source["video_metadata"]["clip_end_seconds"]),
            frame_count=int(config["media"]["proxy_frame_count"]),
            max_side=int(config["media"]["proxy_frame_max_side"]),
        )
        frames = full_frames
        seconds = metadata["proxy_sample_seconds"]
        localization = None
        localization_raw = None
        localization_usage = None
        temporal_config = config["media"].get("temporal_localization")
        if temporal_config:
            scout = runner._panel(
                full_frames, seconds,
                count=int(temporal_config["scout_frame_count"]),
                width=int(temporal_config["scout_frame_width"]),
                quality=int(config["media"]["jpeg_quality"]),
            )
            if row is not None and "temporal_localization" in row:
                localization = dict(row["temporal_localization"])
            else:
                localization, localization_raw, localization_usage = (
                    _localize_question_window(
                        client, config=config, question=str(sample["question"]),
                        scout=scout,
                    )
                )
            absolute_start, absolute_end = absolute_temporal_window(
                float(metadata["clip_start_seconds"]),
                float(metadata["clip_end_seconds"]),
                localization["window_fraction"],
            )
            frames, localized_metadata = runner._sample_clip(
                Path(sample["video_path"]), start_sec=absolute_start,
                end_sec=absolute_end,
                frame_count=int(config["media"]["proxy_frame_count"]),
                max_side=int(config["media"]["proxy_frame_max_side"]),
            )
            relative_offset = absolute_start - float(metadata["clip_start_seconds"])
            seconds = [
                relative_offset + float(value)
                for value in localized_metadata["proxy_sample_seconds"]
            ]
            localization = {
                **localization,
                "absolute_start_seconds": absolute_start,
                "absolute_end_seconds": absolute_end,
                "whole_clip_scout_sha256": hashlib.sha256(scout).hexdigest(),
                "question_only": True,
                "answer_candidates_seen": False,
                "gold_or_official_structure_seen": False,
            }
        frame_count = int(config["media"].get("program_frame_count", 16))
        track_indices = [
            round(index * (len(frames) - 1) / (frame_count - 1))
            for index in range(frame_count)
        ]
        track_panel = _panel(frames, track_indices, seconds, config=config, prefix="T")
        mode = str(config["media"].get("binding_observation_mode", "COLLAGE_OVERLAY"))
        matched_control_pixels = bool(
            config.get("controls", {}).get("require_distinct_decoy", False)
        )
        if mode == "SEPARATE_FRAMES_DUAL_VIEW":
            global_panel, _ = _dual_view_panel(
                frames, track_indices, seconds, config=config,
                prefix="E" if matched_control_pixels else "U",
            )
        else:
            global_panel = _panel(
                frames, track_indices, seconds, config=config,
                prefix="E" if matched_control_pixels else "U",
            )
        if row is None:
            programs, compile_usage = _compile_claims(
                client, config=config, sample=sample,
                entity_catalog=source["world_model_raw"]["entity_catalog"],
                candidates=items,
            )
            row = {
                "schema_version": 1, "benchmark": source["benchmark"],
                "sample_id": sample_id, "answer_contract": contract,
                "complete": False, "source_gate_sha256": _sha256(source_path),
                "source_collection_contract_sha256": source["collection_contract_sha256"],
                "video_sha256": source["video_sha256"],
                "collector_sha256": _sha256(Path(__file__).resolve()),
                "config_sha256": _sha256(args.config),
                "compiler_usage": compile_usage, "candidates": programs,
                "track_panel_sha256": hashlib.sha256(track_panel).hexdigest(),
                "unbound_panel_sha256": hashlib.sha256(global_panel).hexdigest(),
                "compiler_saw_question_and_candidates": True,
                "compiler_saw_gold_or_official_program": False,
                "visual_grounders_saw_full_question_option_set_or_gold": False,
            }
            if localization is not None:
                row.update({
                    "temporal_localization": localization,
                    "temporal_localization_raw": localization_raw,
                    "temporal_localization_usage": localization_usage,
                })
            existing[sample_id] = row
            save()
        candidates = row["candidates"]
        for candidate_index, candidate in enumerate(candidates):
            if "track" not in candidate:
                cached = next((
                    other for other in candidates[:candidate_index]
                    if str(other["bind_entity_visual_description"]).casefold()
                    == str(candidate["bind_entity_visual_description"]).casefold()
                    and "track" in other
                ), None)
                if cached is not None:
                    candidate.update({
                        "track": copy.deepcopy(cached["track"]),
                        "track_usage": {"cached_identical_bind": True},
                        "track_indices": list(cached["track_indices"]),
                        "track_evidence_sha256": cached["track_evidence_sha256"],
                    })
                elif mode == "SEPARATE_FRAMES_DUAL_VIEW":
                    track, usage, track_evidence_hash = _multiframe_bind_track(
                        client, config=config,
                        entity=str(candidate["bind_entity_visual_description"]),
                        frames=frames, indices=track_indices, seconds=seconds,
                    )
                else:
                    track, usage = bound._bind_track(
                        client, config=config,
                        entity=str(candidate["bind_entity_visual_description"]),
                        panel=track_panel,
                    )
                    track_evidence_hash = hashlib.sha256(track_panel).hexdigest()
                if cached is None:
                    candidate["track"] = track
                    candidate["track_usage"] = usage
                    candidate["track_indices"] = track_indices
                    candidate["track_evidence_sha256"] = track_evidence_hash
                save()
            if "identity_verification" not in candidate:
                cached = next((
                    other for other in candidates[:candidate_index]
                    if str(other["bind_entity_visual_description"]).casefold()
                    == str(candidate["bind_entity_visual_description"]).casefold()
                    and "identity_verification" in other
                ), None)
                if cached is not None:
                    candidate["identity_verification"] = copy.deepcopy(
                        cached["identity_verification"]
                    )
                    candidate["identity_verification"]["usage"] = {
                        "cached_identical_bind": True,
                    }
                else:
                    overlaid, fallbacks = overlay._overlay_frames(
                        frames, track_indices, candidate["track_indices"],
                        candidate["track"]["tracks"], entity_id="CARRIER",
                    )
                    identity_panel = media_helpers._panel_bytes(
                        overlaid,
                        labels=[f"I{slot} {seconds[frame_index]:.2f}s" for slot, frame_index in enumerate(track_indices)],
                        frame_width=int(config["media"]["evidence_frame_width"]),
                        quality=int(config["media"]["jpeg_quality"]),
                    )
                    parsed, raw, usage = _verify_identity(
                        client, config=config,
                        entity=str(candidate["bind_entity_visual_description"]),
                        panel=identity_panel,
                    )
                    candidate["identity_verification"] = {
                        **parsed, "raw": raw, "usage": usage,
                        "panel_sha256": hashlib.sha256(identity_panel).hexdigest(),
                        "overlay_fallback_count": sum(fallbacks),
                    }
                save()
            if (
                "decoy_entity_visual_description" in candidate
                and "decoy_track" not in candidate
            ):
                decoy_entity = str(candidate["decoy_entity_visual_description"])
                cached = next((
                    other for other in candidates[:candidate_index]
                    if str(other.get("decoy_entity_visual_description") or "").casefold()
                    == decoy_entity.casefold() and "decoy_track" in other
                ), None)
                if cached is not None:
                    candidate.update({
                        "decoy_track": copy.deepcopy(cached["decoy_track"]),
                        "decoy_track_usage": {"cached_identical_bind": True},
                        "decoy_track_indices": list(cached["decoy_track_indices"]),
                        "decoy_track_evidence_sha256": cached["decoy_track_evidence_sha256"],
                    })
                elif mode == "SEPARATE_FRAMES_DUAL_VIEW":
                    track, usage, evidence_hash = _multiframe_bind_track(
                        client, config=config, entity=decoy_entity,
                        frames=frames, indices=track_indices, seconds=seconds,
                    )
                else:
                    track, usage = bound._bind_track(
                        client, config=config, entity=decoy_entity,
                        panel=track_panel,
                    )
                    evidence_hash = hashlib.sha256(track_panel).hexdigest()
                if cached is None:
                    candidate.update({
                        "decoy_track": track,
                        "decoy_track_usage": usage,
                        "decoy_track_indices": track_indices,
                        "decoy_track_evidence_sha256": evidence_hash,
                    })
                save()
            if (
                "decoy_entity_visual_description" in candidate
                and "decoy_identity_verification" not in candidate
            ):
                decoy_entity = str(candidate["decoy_entity_visual_description"])
                cached = next((
                    other for other in candidates[:candidate_index]
                    if str(other.get("decoy_entity_visual_description") or "").casefold()
                    == decoy_entity.casefold()
                    and "decoy_identity_verification" in other
                ), None)
                if cached is not None:
                    candidate["decoy_identity_verification"] = copy.deepcopy(
                        cached["decoy_identity_verification"]
                    )
                    candidate["decoy_identity_verification"]["usage"] = {
                        "cached_identical_bind": True,
                    }
                else:
                    overlaid, fallbacks = overlay._overlay_frames(
                        frames, track_indices, candidate["decoy_track_indices"],
                        candidate["decoy_track"]["tracks"], entity_id="DECOY",
                    )
                    identity_panel = media_helpers._panel_bytes(
                        overlaid,
                        labels=[
                            f"D{slot} {seconds[frame_index]:.2f}s"
                            for slot, frame_index in enumerate(track_indices)
                        ],
                        frame_width=int(config["media"]["evidence_frame_width"]),
                        quality=int(config["media"]["jpeg_quality"]),
                    )
                    parsed, raw, usage = _verify_identity(
                        client, config=config, entity=decoy_entity,
                        panel=identity_panel,
                    )
                    candidate["decoy_identity_verification"] = {
                        **parsed, "raw": raw, "usage": usage,
                        "panel_sha256": hashlib.sha256(identity_panel).hexdigest(),
                        "overlay_fallback_count": sum(fallbacks),
                    }
                save()
        for index, candidate in enumerate(candidates):
            if "unbound_relation" not in candidate:
                parsed, raw, usage = _ground_relation(
                    client, config=config, program=candidate, panel=global_panel,
                    observation_kind="UNBOUND_FULL_CONTEXT",
                )
                candidate["unbound_relation"] = {
                    **parsed, "raw": raw, "usage": usage,
                    "panel_sha256": hashlib.sha256(global_panel).hexdigest(),
                }
                save()
            if "bound_relation" not in candidate:
                if mode == "SEPARATE_FRAMES_DUAL_VIEW":
                    panel, fallback_count = _dual_view_panel(
                        frames, track_indices, seconds, config=config,
                        track_indices=candidate["track_indices"],
                        tracks=candidate["track"]["tracks"],
                        prefix="E" if matched_control_pixels else "B",
                    )
                else:
                    overlaid, fallbacks = overlay._overlay_frames(
                        frames, track_indices, candidate["track_indices"],
                        candidate["track"]["tracks"], entity_id="CARRIER",
                    )
                    panel = media_helpers._panel_bytes(
                        overlaid,
                        labels=[f"B{slot} {seconds[frame_index]:.2f}s" for slot, frame_index in enumerate(track_indices)],
                        frame_width=int(config["media"]["evidence_frame_width"]),
                        quality=int(config["media"]["jpeg_quality"]),
                    )
                    fallback_count = sum(fallbacks)
                parsed, raw, usage = _ground_relation(
                    client, config=config, program=candidate, panel=panel,
                    observation_kind="BOUND_FULL_CONTEXT",
                )
                candidate["bound_relation"] = {
                    **parsed, "raw": raw, "usage": usage,
                    "panel_sha256": hashlib.sha256(panel).hexdigest(),
                    "overlay_fallback_count": fallback_count,
                }
                save()
            if "wrong_guard_relation" not in candidate:
                wrong = candidates[(index + 1) % len(candidates)]
                wrong_track = (
                    candidate["decoy_track"]
                    if "decoy_track" in candidate else wrong["track"]
                )
                wrong_track_indices = (
                    candidate["decoy_track_indices"]
                    if "decoy_track_indices" in candidate else wrong["track_indices"]
                )
                if mode == "SEPARATE_FRAMES_DUAL_VIEW":
                    panel, fallback_count = _dual_view_panel(
                        frames, track_indices, seconds, config=config,
                        track_indices=wrong_track_indices,
                        tracks=wrong_track["tracks"],
                        prefix="E" if matched_control_pixels else "W",
                    )
                else:
                    overlaid, fallbacks = overlay._overlay_frames(
                        frames, track_indices, wrong_track_indices,
                        wrong_track["tracks"], entity_id="DECOY",
                    )
                    panel = media_helpers._panel_bytes(
                        overlaid,
                        labels=[f"W{slot} {seconds[frame_index]:.2f}s" for slot, frame_index in enumerate(track_indices)],
                        frame_width=int(config["media"]["evidence_frame_width"]),
                        quality=int(config["media"]["jpeg_quality"]),
                    )
                    fallback_count = sum(fallbacks)
                parsed, raw, usage = _ground_relation(
                    client, config=config, program=candidate, panel=panel,
                    observation_kind="WRONG_GUARD_FULL_CONTEXT",
                )
                candidate["wrong_guard_relation"] = {
                    **parsed, "raw": raw, "usage": usage,
                    "panel_sha256": hashlib.sha256(panel).hexdigest(),
                    "overlay_fallback_count": fallback_count,
                    "wrong_track_slot": (
                        f"{candidate['slot']}:decoy"
                        if "decoy_track" in candidate else str(wrong["slot"])
                    ),
                }
                save()
        row["complete"] = True
        save()
        print(json.dumps({
            "sample_id": sample_id, "candidate_count": len(candidates),
            "bound_unbound_probability_deltas": [
                round(abs(float(candidate["bound_relation"]["support_probability"]) - float(candidate["unbound_relation"]["support_probability"])), 3)
                for candidate in candidates
            ],
            "track_observed": [bool(candidate["track"]["observed_true"]) for candidate in candidates],
            "identity_match_probabilities": [
                float(candidate["identity_verification"]["identity_match_probability"])
                for candidate in candidates
            ],
        }, ensure_ascii=False), flush=True)
    print(str(output_path.resolve()))


if __name__ == "__main__":
    main()
