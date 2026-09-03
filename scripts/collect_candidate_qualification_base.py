#!/usr/bin/env python3
"""Collect outcome-blind baseline priors for frozen video qualification rows.

This is deliberately smaller than ``run_structured_video_transfer.py``: the
candidate-claim evaluator needs only a target-native answer prior and an
unlabelled entity catalogue before it executes its matched candidate programs.
No typed probe, answer label, benchmark program, or annotation is supplied to
the model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_active_video_wrapper_transfer as media_helpers  # noqa: E402
import run_structured_video_transfer as structured  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_base_payload(
    payload: Mapping[str, Any], *, answer_space: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows = list(payload.get("answer_priors") or ())
    expected = list(map(str, answer_space))
    if [str(row.get("native_answer")) for row in rows] != expected:
        raise ValueError("answer_priors must preserve the complete frozen answer order")
    particles: list[dict[str, Any]] = []
    total = 0.0
    for row in rows:
        weight = float(row.get("prior_weight", -1.0))
        summary = str(row.get("event_graph_summary") or "").strip()
        if weight <= 0.0:
            raise ValueError("every prior_weight must be strictly positive")
        if not summary:
            raise ValueError("event_graph_summary must be nonempty")
        particles.append({
            "particle_id": f"w{len(particles)}",
            "native_answer": str(row["native_answer"]),
            "prior_weight": weight,
            "event_graph_summary": summary,
        })
        total += weight
    for row in particles:
        row["prior_weight"] = float(row["prior_weight"]) / total

    raw_entities = list(payload.get("entity_catalog") or ())
    entities: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in raw_entities:
        entity_id = str(row.get("entity_id") or "").strip()
        description = str(row.get("visual_description") or "").strip()
        if not entity_id or not description:
            raise ValueError("entity_catalog rows require entity_id and visual_description")
        if entity_id in seen:
            raise ValueError("entity_catalog ids must be unique")
        seen.add(entity_id)
        entities.append({"entity_id": entity_id, "visual_description": description})
    if not entities:
        raise ValueError("entity_catalog must contain at least one visible entity hypothesis")
    return particles, entities


def _propose_base(
    client: OpenAI, *, config: Mapping[str, Any], sample: Any,
    scout: bytes, duration_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    _, _, answer_space, contract = structured._answer_contract(sample, "nextqa")
    prompt = (
        sample.format_question()
        + f"\nThe clip spans relative 0.0 to {duration_seconds:.3f} seconds. "
        + "Estimate a target-native prior over every answer candidate from only "
        "the low-bandwidth scout frames. Return exactly one answer_priors row for "
        f"each candidate in this exact order: {list(answer_space)}. The native_answer "
        f"must be {contract}. Keep uncertainty; every prior must be positive. For "
        "each row, summarize the answer-conditioned visible temporal/event hypothesis. "
        "Also list concrete visible entities that a later identity tracker could "
        "localize. The entity catalogue is unlabelled: do not state which answer is "
        "correct and do not associate an entity with correctness. You have no access "
        "to answer labels, official programs, relation annotations, or benchmark graphs."
    )
    system = (
        "Return JSON only: {answer_priors:[{native_answer:string,prior_weight:number,"
        "event_graph_summary:string}],entity_catalog:[{entity_id:string,"
        "visual_description:string}]}."
    )
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        content = [
            {"type": "text", "text": prompt + (f"\nSchema error: {last_error}" if last_error else "")},
            {"type": "text", "text": "Low-bandwidth scout frames:"},
            media_helpers._image_content(scout),
        ]
        payload, usage = media_helpers._json_call(
            client, model=str(model["id"]), system=system, content=content,
            max_tokens=int(model["max_base_tokens"]),
        )
        try:
            particles, entities = _parse_base_payload(
                payload, answer_space=answer_space,
            )
            normalized = dict(payload)
            normalized["entity_catalog"] = entities
            return particles, normalized, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError(f"qualification base schema failed: {last_error}")


def _collect_one(
    sample: Any, *, config: Mapping[str, Any], api_key: str,
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
    particles, raw, usage = _propose_base(
        client, config=config, sample=sample, scout=scout,
        duration_seconds=float(metadata["duration_seconds"]),
    )
    family = str(sample.question_family)
    return {
        "schema_version": 1,
        "benchmark": "nextqa",
        "split": "qualification",
        "sample_id": str(sample.sample_id),
        "family": family,
        # Stored for the evaluator, never included in any model request.
        "gold_answer": str(sample.answer),
        "collection_contract_sha256": contract_sha256,
        "sample": sample.to_dict(),
        "video_sha256": media_helpers.file_sha256(Path(sample.video_path)),
        "video_metadata": metadata,
        "scout_sha256": hashlib.sha256(scout).hexdigest(),
        "world_model": {
            "particles": particles,
            "particle_summaries": [row["event_graph_summary"] for row in particles],
            "probes": [],
            "probe_rationales": [],
        },
        "world_model_raw": raw,
        "world_model_usage": usage,
        "probe_receipts": {},
        "probe_grounder_raw": {},
        "probe_usage": {},
        "wrapper_receipts": {},
        "runtime_oracle_inputs": False,
        "model_saw_gold_or_official_structure": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_PROSPECTIVE_QUALIFICATION":
        raise ValueError("qualification config is not frozen")
    policy_path = Path(config["frozen_policy"]["path"])
    if _sha256(policy_path) != config["frozen_policy"]["sha256"]:
        raise ValueError("frozen policy hash mismatch")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN_BEFORE_QUALIFICATION_COLLECTION":
        raise ValueError("policy lifecycle does not permit qualification")
    manifest_path = Path(config["split_manifest"])
    if _sha256(manifest_path) != config["split_manifest_sha256"]:
        raise ValueError("frozen split manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_ids, samples = structured._load_samples(
        "nextqa", config, manifest, "qualification",
    )
    contract_sha256 = hashlib.sha256(json.dumps({
        "config_sha256": _sha256(args.config),
        "collector_sha256": _sha256(Path(__file__).resolve()),
        "policy_sha256": config["frozen_policy"]["sha256"],
        "split_manifest_sha256": config["split_manifest_sha256"],
        "sample_ids": sample_ids,
    }, sort_keys=True).encode()).hexdigest()
    keys = runpy.run_path(str(args.keys))
    api_key = keys.get(config["model"]["api_key_name"])
    if not api_key:
        raise SystemExit("configured API key is missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = args.output_dir / "receipts.json"
    existing: dict[str, dict[str, Any]] = {}
    if receipts_path.is_file():
        existing = {
            str(row["sample_id"]): row
            for row in json.loads(receipts_path.read_text(encoding="utf-8"))
        }
        if any(
            row.get("collection_contract_sha256") != contract_sha256
            for row in existing.values()
        ):
            raise ValueError("cached qualification receipt contract mismatch")

    pending = [sample_id for sample_id in sample_ids if sample_id not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_one, samples[sample_id], config=config,
                api_key=str(api_key), contract_sha256=contract_sha256,
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
            ordered = [existing[value] for value in sample_ids if value in existing]
            receipts_path.write_text(
                json.dumps(ordered, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({
                "completed": sample_id,
                "progress": f"{len(ordered)}/{len(sample_ids)}",
            }), flush=True)
    missing = [sample_id for sample_id in sample_ids if sample_id not in existing]
    if missing:
        raise SystemExit(f"incomplete qualification base; rerun to resume: {missing}")
    report = {
        "schema_version": 1,
        "status": "QUALIFICATION_BASE_COMPLETE",
        "benchmark": "nextqa",
        "split": "qualification",
        "samples": len(sample_ids),
        "collection_contract_sha256": contract_sha256,
        "frozen_policy_sha256": config["frozen_policy"]["sha256"],
        "runtime_oracle_inputs": False,
        "model_saw_gold_or_official_structure": False,
        "heldout_touched": False,
        "receipts_sha256": _sha256(receipts_path),
    }
    report_path = args.output_dir / "qualification_base_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
