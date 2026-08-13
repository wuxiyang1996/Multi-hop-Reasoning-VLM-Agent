#!/usr/bin/env python3
"""Collect full-context RELATE receipts with an overlaid neural BIND handle."""

from __future__ import annotations

import argparse
import copy
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

import augment_bound_relation_smoke as bound  # noqa: E402
import run_active_video_wrapper_transfer as media_helpers  # noqa: E402
import run_structured_video_transfer as runner  # noqa: E402
from motif_transfer.structured_video_transfer import parse_typed_probe_receipt  # noqa: E402


def _overlay_frames(
    frames: Sequence[Image.Image],
    relation_indices: Sequence[int],
    track_indices: Sequence[int],
    tracks: Sequence[Sequence[float] | None],
    *,
    entity_id: str,
) -> tuple[list[Image.Image], list[bool]]:
    output, fallbacks = [], []
    for index in relation_indices:
        frame = frames[index].convert("RGB").copy()
        box = bound._nearest_box(index, track_indices, tracks)
        if box is None:
            output.append(frame)
            fallbacks.append(True)
            continue
        width, height = frame.size
        x0, y0, x1, y1 = map(float, box)
        rectangle = (
            round(x0 * width), round(y0 * height),
            round(x1 * width), round(y1 * height),
        )
        draw = ImageDraw.Draw(frame)
        line_width = max(2, round(min(width, height) / 100))
        draw.rectangle(rectangle, outline=(255, 0, 255), width=line_width)
        draw.text(
            (rectangle[0] + 3, max(2, rectangle[1] - 14)),
            f"BOUND {entity_id}", fill=(255, 0, 255),
        )
        output.append(frame)
        fallbacks.append(False)
    return output, fallbacks


def _ground_overlay(
    client: OpenAI,
    *,
    config: Mapping[str, Any],
    probe,
    bound_entity: str,
    panel: bytes,
):
    model = config["model"]
    prompt = (
        "Measure only the typed relation predicate in these full-context temporal "
        "frames. A magenta box labeled BOUND marks the persistent entity handle "
        f"created by the preceding BIND intervention: {bound_entity}. The box is "
        "an identity aid, not evidence that the relation is true. Preserve all "
        "surrounding context. "
        f"Predicate kind: {probe.predicate_kind}. Entities: {list(probe.entity_refs)}. "
        f"Relative window: [{probe.start_sec:.3f}, {probe.end_sec:.3f}]. "
        "Do not infer or answer any benchmark question. Return observed_true and "
        "sensor_reliability in [0.5,1]."
    )
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        payload, usage = media_helpers._json_call(
            client, model=str(model["id"]),
            system=(
                "Return JSON only: {observed_true:bool,sensor_reliability:number,"
                "measurement:string}. You never receive or predict the final answer."
            ),
            content=[
                {"type": "text", "text": prompt + (f" Schema error: {last_error}" if last_error else "")},
                media_helpers._image_content(panel),
            ],
            max_tokens=int(model["max_probe_tokens"]),
        )
        try:
            receipt = parse_typed_probe_receipt(
                payload, probe=probe,
                evidence_sha256=(hashlib.sha256(panel).hexdigest(),),
            )
            return receipt, payload, usage
        except ValueError as exc:
            last_error = str(exc)
    raise ValueError(f"overlay RELATE schema failed: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_rows = json.loads((args.run_dir / "receipts.json").read_text(encoding="utf-8"))
    source_by_id = {str(row["sample_id"]): row for row in source_rows}
    crop_forks = json.loads((args.run_dir / "bound_relation_forks.json").read_text(encoding="utf-8"))
    selected = crop_forks[:args.sample_count]
    output_path = args.run_dir / "overlay_relation_forks.json"
    existing = {}
    if output_path.is_file():
        existing = {
            row["sample_id"]: row
            for row in json.loads(output_path.read_text(encoding="utf-8"))
        }
    keys = runpy.run_path(str(args.keys))
    model_config = config["model"]
    client = OpenAI(
        api_key=str(keys[model_config["api_key_name"]]),
        base_url=str(model_config["base_url"]),
        timeout=float(model_config["timeout_seconds"]),
        max_retries=int(model_config["max_retries"]),
    )
    for crop_fork in selected:
        sample_id = str(crop_fork["sample_id"])
        if sample_id in existing:
            continue
        source = source_by_id[sample_id]
        world_model, _ = runner._rehydrate(source)
        frames, metadata = runner._sample_clip(
            Path(source["sample"]["video_path"]),
            start_sec=float(source["video_metadata"]["clip_start_seconds"]),
            end_sec=float(source["video_metadata"]["clip_end_seconds"]),
            frame_count=int(config["media"]["proxy_frame_count"]),
            max_side=int(config["media"]["proxy_frame_max_side"]),
        )
        seconds = metadata["proxy_sample_seconds"]
        output = copy.deepcopy(crop_fork)
        output["bound_relation_receipts"] = {}
        output["matched_matrix"] = "2_BIND_TRACKS_X_3_FULL_CONTEXT_OVERLAY_RELATE_GROUNDINGS"
        for bind_id, track_payload in crop_fork["tracks"].items():
            primary = str(track_payload["primary_entity_ref"])
            entity_id = primary.split(":", 1)[0]
            output["bound_relation_receipts"][bind_id] = {}
            for relate_id, old_cell in crop_fork["bound_relation_receipts"][bind_id].items():
                probe = next(value for value in world_model.probes if value.probe_id == relate_id)
                indices = old_cell["relation_proxy_indices"]
                overlaid, fallbacks = _overlay_frames(
                    frames, indices,
                    track_payload["track_indices"],
                    track_payload["track"]["tracks"],
                    entity_id=entity_id,
                )
                panel = media_helpers._panel_bytes(
                    overlaid,
                    labels=[f"O{slot} {seconds[index]:.2f}s" for slot, index in enumerate(indices)],
                    frame_width=int(config["media"]["evidence_frame_width"]),
                    quality=int(config["media"]["jpeg_quality"]),
                )
                receipt, raw, usage = _ground_overlay(
                    client, config=config, probe=probe,
                    bound_entity=primary, panel=panel,
                )
                output["bound_relation_receipts"][bind_id][relate_id] = {
                    "shared_primary_entity": primary in probe.entity_refs,
                    "crop_fallback_count": sum(fallbacks),
                    "relation_proxy_indices": indices,
                    "bound_panel_sha256": hashlib.sha256(panel).hexdigest(),
                    "receipt": vars(receipt), "raw": raw, "usage": usage,
                    "observation_kind": "FULL_CONTEXT_WITH_BOUND_TRACK_OVERLAY",
                }
        existing[sample_id] = output
        ordered = [existing[row["sample_id"]] for row in selected if row["sample_id"] in existing]
        output_path.write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps({
            "sample_id": sample_id, "progress": f"{len(ordered)}/{len(selected)}",
            "overlay_global_disagreements": sum(
                cell["receipt"]["observed_true"]
                != source["probe_receipts"][relate_id]["observed_true"]
                for rows in output["bound_relation_receipts"].values()
                for relate_id, cell in rows.items()
            ),
        }), flush=True)
    print(str(output_path.resolve()))


if __name__ == "__main__":
    main()
