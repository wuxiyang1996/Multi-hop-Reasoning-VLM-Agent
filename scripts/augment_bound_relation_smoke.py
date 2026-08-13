#!/usr/bin/env python3
"""Smoke-test executable neural BIND handles and bound RELATE crops."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

from openai import OpenAI
from PIL import Image, ImageDraw


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_active_video_wrapper_transfer as media_helpers  # noqa: E402
import run_structured_video_transfer as runner  # noqa: E402
from motif_transfer.structured_video_transfer import parse_typed_probe_receipt  # noqa: E402


def _track_panel(
    frames: Sequence[Image.Image], seconds: Sequence[float], *, quality: int,
) -> tuple[bytes, list[int]]:
    indices = [round(index * (len(frames) - 1) / 15) for index in range(16)]
    panel = media_helpers._panel_bytes(
        [frames[index] for index in indices],
        labels=[f"T{slot} {seconds[index]:.2f}s" for slot, index in enumerate(indices)],
        frame_width=256,
        quality=quality,
    )
    return panel, indices


def _parse_track(
    payload: Mapping[str, Any], *, expected_count: int,
) -> tuple[bool, float, tuple[tuple[float, float, float, float] | None, ...]]:
    observed = payload.get("observed_true")
    if not isinstance(observed, bool):
        raise ValueError("track observed_true must be boolean")
    reliability = float(payload.get("sensor_reliability", 0.0))
    if not 0.5 <= reliability <= 1:
        raise ValueError("track reliability must be in [0.5,1]")
    raw = list(payload.get("tracks") or ())
    if len(raw) != expected_count:
        raise ValueError(f"track must cover all {expected_count} panel frames")
    parsed: list[tuple[float, float, float, float] | None] = []
    for expected, row in enumerate(raw):
        if int(row.get("panel_index", -1)) != expected:
            raise ValueError("track panel indices must be complete and ordered")
        visible = row.get("visible")
        if not isinstance(visible, bool):
            raise ValueError("track visibility must be boolean")
        if not visible:
            parsed.append(None)
            continue
        box = tuple(map(float, row.get("bbox_xyxy_normalized") or ()))
        if len(box) != 4:
            raise ValueError("visible track requires xyxy bbox")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("track bbox must be normalized and nonempty")
        parsed.append((x0, y0, x1, y1))
    if observed and sum(box is not None for box in parsed) < 3:
        raise ValueError("positive BIND handle needs at least three visible frames")
    return observed, reliability, tuple(parsed)


def _bind_track(
    client: OpenAI,
    *,
    config: Mapping[str, Any],
    entity: str,
    panel: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = config["model"]
    prompt = (
        "Target-native BIND intervention. Track only this canonical visual entity "
        f"through the 16 labeled temporal subframes: {entity}. For every T0..T15 "
        "return visibility and a tight bbox [x0,y0,x1,y1] normalized within that "
        "individual subframe (not the 4x4 panel). Do not answer any benchmark "
        "question and do not infer an answer option. observed_true means the same "
        "entity identity is grounded in at least three frames."
    )
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        payload, usage = media_helpers._json_call(
            client,
            model=str(model["id"]),
            system=(
                "Return JSON only: {observed_true:bool,sensor_reliability:number,"
                "tracks:[{panel_index:int,visible:bool,bbox_xyxy_normalized:"
                "[number,number,number,number] or null}],measurement:string}."
            ),
            content=[
                {"type": "text", "text": prompt + (f" Schema error: {last_error}" if last_error else "")},
                media_helpers._image_content(panel),
            ],
            max_tokens=1800,
        )
        try:
            observed, reliability, tracks = _parse_track(payload, expected_count=16)
            return {
                "observed_true": observed,
                "sensor_reliability": reliability,
                "tracks": [list(box) if box is not None else None for box in tracks],
                "raw": payload,
            }, usage
        except ValueError as exc:
            last_error = str(exc)
    raise ValueError(f"BIND track schema failed: {last_error}")


def _nearest_box(
    target: int,
    track_indices: Sequence[int],
    tracks: Sequence[Sequence[float] | None],
) -> Sequence[float] | None:
    available = [
        (abs(index - target), slot) for slot, index in enumerate(track_indices)
        if tracks[slot] is not None
    ]
    return tracks[min(available)[1]] if available else None


def _bound_crops(
    frames: Sequence[Image.Image],
    relation_indices: Sequence[int],
    track_indices: Sequence[int],
    tracks: Sequence[Sequence[float] | None],
) -> tuple[list[Image.Image], list[bool]]:
    output, fallbacks = [], []
    for index in relation_indices:
        frame = frames[index].convert("RGB")
        box = _nearest_box(index, track_indices, tracks)
        if box is None:
            output.append(frame)
            fallbacks.append(True)
            continue
        width, height = frame.size
        x0, y0, x1, y1 = map(float, box)
        center_x = (x0 + x1) * width / 2
        center_y = (y0 + y1) * height / 2
        box_width = max(0.2 * width, (x1 - x0) * width * 3.0)
        box_height = max(0.2 * height, (y1 - y0) * height * 3.0)
        left = max(0, round(center_x - box_width / 2))
        top = max(0, round(center_y - box_height / 2))
        right = min(width, round(center_x + box_width / 2))
        bottom = min(height, round(center_y + box_height / 2))
        crop = frame.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
        draw = ImageDraw.Draw(crop)
        draw.rectangle((0, 0, crop.width - 1, crop.height - 1), outline="lime", width=3)
        output.append(crop)
        fallbacks.append(False)
    return output, fallbacks


def _choose_pair(world_model) -> tuple[int, int]:
    bind = [i for i, probe in enumerate(world_model.probes) if probe.target_event_role == "BIND"]
    relate = [i for i, probe in enumerate(world_model.probes) if probe.target_event_role == "RELATE"]
    pairs = [
        (left, right)
        for left in bind for right in relate
        if set(world_model.probes[left].entity_refs) & set(world_model.probes[right].entity_refs)
    ]
    if not pairs:
        raise ValueError("world model has no guarded BIND->RELATE pair")
    return max(
        pairs,
        key=lambda pair: (
            max(world_model.probes[pair[1]].latent_true_probability_by_particle)
            - min(world_model.probes[pair[1]].latent_true_probability_by_particle),
            -pair[0], -pair[1],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = json.loads((args.run_dir / "receipts.json").read_text(encoding="utf-8"))
    keys = runpy.run_path(str(args.keys))
    model = config["model"]
    client = OpenAI(
        api_key=str(keys[model["api_key_name"]]),
        base_url=str(model["base_url"]), timeout=float(model["timeout_seconds"]),
        max_retries=int(model["max_retries"]),
    )
    outputs = []
    for row in rows[:args.sample_count]:
        world_model, _ = runner._rehydrate(row)
        bind_index, relate_index = _choose_pair(world_model)
        bind_probe = world_model.probes[bind_index]
        relate_probe = world_model.probes[relate_index]
        shared = sorted(set(bind_probe.entity_refs) & set(relate_probe.entity_refs))
        frames, metadata = runner._sample_clip(
            Path(row["sample"]["video_path"]),
            start_sec=float(row["video_metadata"]["clip_start_seconds"]),
            end_sec=float(row["video_metadata"]["clip_end_seconds"]),
            frame_count=int(config["media"]["proxy_frame_count"]),
            max_side=int(config["media"]["proxy_frame_max_side"]),
        )
        seconds = metadata["proxy_sample_seconds"]
        track_panel, track_indices = _track_panel(
            frames, seconds, quality=int(config["media"]["jpeg_quality"]),
        )
        track, track_usage = _bind_track(
            client, config=config, entity=shared[0], panel=track_panel,
        )
        relation_indices = row["wrapper_receipts"][relate_probe.probe_id][
            "proxy_frame_indices"
        ]
        crops, fallbacks = _bound_crops(
            frames, relation_indices, track_indices, track["tracks"],
        )
        bound_panel = media_helpers._panel_bytes(
            crops,
            labels=[f"B{slot} {seconds[index]:.2f}s" for slot, index in enumerate(relation_indices)],
            frame_width=int(config["media"]["evidence_frame_width"]),
            quality=int(config["media"]["jpeg_quality"]),
        )
        bound_receipt, bound_raw, bound_usage = runner._ground_probe(
            client, config=config, probe=relate_probe, evidence_panel=bound_panel,
        )
        output = {
            "sample_id": row["sample_id"],
            "bind_probe_id": bind_probe.probe_id,
            "relate_probe_id": relate_probe.probe_id,
            "shared_entity": shared[0],
            "track": track,
            "track_usage": track_usage,
            "track_panel_sha256": hashlib.sha256(track_panel).hexdigest(),
            "relation_proxy_indices": relation_indices,
            "crop_fallbacks": fallbacks,
            "bound_panel_sha256": hashlib.sha256(bound_panel).hexdigest(),
            "global_relation_receipt": row["probe_receipts"][relate_probe.probe_id],
            "bound_relation_receipt": {
                "parsed": vars(bound_receipt), "raw": bound_raw, "usage": bound_usage,
            },
            "question_options_or_gold_seen_by_bind_or_relate": False,
        }
        outputs.append(output)
        print(json.dumps({
            "sample_id": row["sample_id"],
            "pair": [bind_probe.probe_id, relate_probe.probe_id],
            "track_visible": sum(value is not None for value in track["tracks"]),
            "crop_fallbacks": sum(fallbacks),
            "global_observed": row["probe_receipts"][relate_probe.probe_id]["observed_true"],
            "bound_observed": bound_receipt.observed_true,
        }, ensure_ascii=False), flush=True)
    output_path = args.run_dir / "bound_relation_smoke.json"
    output_path.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(str(output_path.resolve()))


if __name__ == "__main__":
    main()
