#!/usr/bin/env python3
"""Collect matched BIND-handle x RELATE observation forks on saved videos."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import runpy
import sys

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import augment_bound_relation_smoke as bound  # noqa: E402
import run_active_video_wrapper_transfer as media_helpers  # noqa: E402
import run_structured_video_transfer as runner  # noqa: E402
from motif_transfer.structured_video_transfer import parse_typed_probe_receipt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_rows = json.loads((args.run_dir / "receipts.json").read_text(encoding="utf-8"))
    selected = source_rows[:args.sample_count]
    output_path = args.run_dir / "bound_relation_forks.json"
    existing = {}
    if output_path.is_file():
        existing = {
            row["sample_id"]: row
            for row in json.loads(output_path.read_text(encoding="utf-8"))
        }
    keys = runpy.run_path(str(args.keys))
    model = config["model"]
    client = OpenAI(
        api_key=str(keys[model["api_key_name"]]),
        base_url=str(model["base_url"]), timeout=float(model["timeout_seconds"]),
        max_retries=int(model["max_retries"]),
    )
    for row in selected:
        sample_id = str(row["sample_id"])
        if sample_id in existing:
            continue
        world_model, _ = runner._rehydrate(row)
        bind_indices = [
            index for index, probe in enumerate(world_model.probes)
            if probe.target_event_role == "BIND"
        ]
        relate_indices = [
            index for index, probe in enumerate(world_model.probes)
            if probe.target_event_role == "RELATE"
        ]
        if len(bind_indices) != 2 or len(relate_indices) != 3:
            raise ValueError("fork collection requires exactly 2 BIND x 3 RELATE")
        frames, metadata = runner._sample_clip(
            Path(row["sample"]["video_path"]),
            start_sec=float(row["video_metadata"]["clip_start_seconds"]),
            end_sec=float(row["video_metadata"]["clip_end_seconds"]),
            frame_count=int(config["media"]["proxy_frame_count"]),
            max_side=int(config["media"]["proxy_frame_max_side"]),
        )
        seconds = metadata["proxy_sample_seconds"]
        track_panel, track_indices = bound._track_panel(
            frames, seconds, quality=int(config["media"]["jpeg_quality"]),
        )
        tracks, bound_receipts = {}, {}
        for bind_index in bind_indices:
            bind_probe = world_model.probes[bind_index]
            entity = bind_probe.entity_refs[0]
            track, usage = bound._bind_track(
                client, config=config, entity=entity, panel=track_panel,
            )
            bind_receipt = parse_typed_probe_receipt(
                {
                    "observed_true": track["observed_true"],
                    "sensor_reliability": track["sensor_reliability"],
                },
                probe=bind_probe,
                evidence_sha256=(hashlib.sha256(track_panel).hexdigest(),),
            )
            tracks[bind_probe.probe_id] = {
                "primary_entity_ref": entity,
                "track_indices": track_indices,
                "track": track,
                "usage": usage,
                "bind_receipt": asdict(bind_receipt),
            }
            bound_receipts[bind_probe.probe_id] = {}
            for relate_index in relate_indices:
                relate_probe = world_model.probes[relate_index]
                relation_indices = row["wrapper_receipts"][relate_probe.probe_id][
                    "proxy_frame_indices"
                ]
                crops, fallbacks = bound._bound_crops(
                    frames, relation_indices, track_indices, track["tracks"],
                )
                panel = media_helpers._panel_bytes(
                    crops,
                    labels=[
                        f"B{slot} {seconds[index]:.2f}s"
                        for slot, index in enumerate(relation_indices)
                    ],
                    frame_width=int(config["media"]["evidence_frame_width"]),
                    quality=int(config["media"]["jpeg_quality"]),
                )
                receipt, raw, relation_usage = runner._ground_probe(
                    client, config=config, probe=relate_probe,
                    evidence_panel=panel,
                )
                bound_receipts[bind_probe.probe_id][relate_probe.probe_id] = {
                    "shared_primary_entity": entity in relate_probe.entity_refs,
                    "crop_fallback_count": sum(fallbacks),
                    "relation_proxy_indices": relation_indices,
                    "bound_panel_sha256": hashlib.sha256(panel).hexdigest(),
                    "receipt": asdict(receipt),
                    "raw": raw,
                    "usage": relation_usage,
                }
        existing[sample_id] = {
            "schema_version": 1,
            "benchmark": row["benchmark"],
            "sample_id": sample_id,
            "source_collection_contract_sha256": row["collection_contract_sha256"],
            "video_sha256": row["video_sha256"],
            "bind_track_panel_sha256": hashlib.sha256(track_panel).hexdigest(),
            "tracks": tracks,
            "bound_relation_receipts": bound_receipts,
            "matched_matrix": "2_BIND_TRACKS_X_3_RELATE_GROUNDINGS",
            "question_options_or_gold_seen_by_bind_or_relate": False,
        }
        ordered = [existing[item["sample_id"]] for item in selected if item["sample_id"] in existing]
        output_path.write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "sample_id": sample_id,
            "progress": f"{len(ordered)}/{len(selected)}",
            "track_visible": {
                key: sum(value is not None for value in payload["track"]["tracks"])
                for key, payload in tracks.items()
            },
            "bound_global_disagreements": sum(
                payload["receipt"]["observed_true"]
                != row["probe_receipts"][relate_id]["observed_true"]
                for bind_id in bound_receipts
                for relate_id, payload in bound_receipts[bind_id].items()
            ),
        }, ensure_ascii=False), flush=True)
    print(str(output_path.resolve()))


if __name__ == "__main__":
    main()
