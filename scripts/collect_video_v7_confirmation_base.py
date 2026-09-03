#!/usr/bin/env python3
"""Collect prospective video baselines with the exact V4 world-model contract.

Unlike the obsolete lightweight qualification baseline, this invokes the same
``_propose_world_model`` function and prompt used for adaptation.  Proposed
probes are retained but deliberately not grounded; the candidate program uses
only the answer priors and unlabeled entity catalogue.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_active_video_wrapper_transfer as media_helpers  # noqa: E402
import run_structured_video_transfer as structured  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect_one(
    sample: Any,
    *,
    benchmark: str,
    config: Mapping[str, Any],
    api_key: str,
    contract_sha256: str,
) -> dict[str, Any]:
    media = config["media"]
    start = float(getattr(sample, "start_sec", 0.0) or 0.0)
    raw_end = getattr(sample, "end_sec", None)
    frames, metadata = structured._sample_clip(
        Path(sample.video_path), start_sec=start,
        end_sec=float(raw_end) if raw_end is not None else None,
        frame_count=int(media["proxy_frame_count"]),
        max_side=int(media["proxy_frame_max_side"]),
    )
    scout = structured._panel(
        frames, metadata["proxy_sample_seconds"],
        count=int(media["scout_frame_count"]),
        width=int(media["scout_frame_width"]),
        quality=int(media["jpeg_quality"]),
    )
    model = config["model"]
    client = OpenAI(
        api_key=api_key, base_url=str(model["base_url"]),
        timeout=float(model["timeout_seconds"]),
        max_retries=int(model["max_retries"]),
    )
    world, raw, usage = structured._propose_world_model(
        client, config=config, benchmark=benchmark, sample=sample, scout=scout,
        duration_seconds=float(metadata["duration_seconds"]),
    )
    family = str(
        getattr(sample, "question_family", None)
        or getattr(sample, "question_type", "")
    )
    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "split": "confirmation",
        "sample_id": str(sample.sample_id),
        "family": family,
        "gold_answer": str(sample.answer),
        "collection_contract_sha256": contract_sha256,
        "sample": sample.to_dict(),
        "video_sha256": media_helpers.file_sha256(Path(sample.video_path)),
        "video_metadata": metadata,
        "scout_sha256": hashlib.sha256(scout).hexdigest(),
        "world_model": {
            "particles": [asdict(value) for value in world.particles],
            "particle_summaries": list(world.particle_summaries),
            "probes": [asdict(value) for value in world.probes],
            "probe_rationales": list(world.probe_rationales),
        },
        "world_model_raw": raw,
        "world_model_usage": usage,
        "probe_receipts": {},
        "probe_grounder_raw": {},
        "probe_usage": {},
        "wrapper_receipts": {},
        "runtime_oracle_inputs": False,
        "baseline_prompt_function": "run_structured_video_transfer._propose_world_model",
        "model_saw_gold_or_official_structure": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("clevrer", "star", "nextqa"), required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_VIDEO_V7_COLLECTION":
        raise ValueError("V7 confirmation manifest is not prospectively frozen")
    ids, samples = structured._load_samples(
        args.benchmark, config, manifest, "confirmation",
    )
    contract_sha256 = hashlib.sha256(json.dumps({
        "baseline_config_sha256": _sha256(args.config),
        "manifest_sha256": _sha256(args.manifest),
        "collector_sha256": _sha256(Path(__file__).resolve()),
        "sample_ids": ids,
        "world_model_function": "run_structured_video_transfer._propose_world_model",
        "ground_probes": False,
    }, sort_keys=True).encode()).hexdigest()
    keys = runpy.run_path(str(args.keys))
    key = keys.get(config["model"]["api_key_name"])
    if not key:
        raise SystemExit("configured API key is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "receipts.json"
    existing: dict[str, dict[str, Any]] = {}
    if path.is_file():
        existing = {
            str(row["sample_id"]): row
            for row in json.loads(path.read_text(encoding="utf-8"))
        }
        if any(
            row.get("collection_contract_sha256") != contract_sha256
            for row in existing.values()
        ):
            raise ValueError("cached confirmation receipt contract mismatch")
    pending = [sample_id for sample_id in ids if sample_id not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_one, samples[sample_id], benchmark=args.benchmark,
                config=config, api_key=str(key), contract_sha256=contract_sha256,
            ): sample_id
            for sample_id in pending
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                existing[sample_id] = future.result()
            except Exception as exc:
                print(json.dumps({
                    "failed": sample_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                continue
            ordered = [existing[value] for value in ids if value in existing]
            path.write_text(
                json.dumps(ordered, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({
                "completed": sample_id, "progress": f"{len(ordered)}/{len(ids)}",
            }), flush=True)
    missing = [sample_id for sample_id in ids if sample_id not in existing]
    if missing:
        raise SystemExit(f"incomplete confirmation baseline; rerun: {missing}")
    print(str(path.resolve()))


if __name__ == "__main__":
    main()
