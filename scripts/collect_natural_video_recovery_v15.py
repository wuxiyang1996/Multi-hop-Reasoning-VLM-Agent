#!/usr/bin/env python3
"""Collect paired primary/proof receipts for STAR and NExT-QA recovery."""

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
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_active_video_wrapper_transfer as media_helpers  # noqa: E402
import run_structured_video_transfer as structured  # noqa: E402
from motif_transfer.natural_video_recovery import (  # noqa: E402
    PROOF_KINDS,
    build_features,
    parse_primary_receipt,
    parse_proof_receipt,
)
from motif_transfer.sokoban_video_recovery import validate_source_receipt  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _panels(
    sample: Any, config: Mapping[str, Any],
) -> tuple[bytes, list[bytes], dict[str, Any]]:
    media = config["media"]
    frames, metadata = structured._sample_clip(
        Path(sample.video_path),
        start_sec=float(getattr(sample, "start_sec", 0.0) or 0.0),
        end_sec=(
            float(sample.end_sec) if getattr(sample, "end_sec", None) is not None
            else None
        ),
        frame_count=int(media["proof_frame_count"]),
        max_side=int(media["proxy_frame_max_side"]),
    )
    seconds = metadata["proxy_sample_seconds"]
    primary_count = int(media["primary_frame_count"])
    primary_indices = [
        round(index * (len(frames) - 1) / (primary_count - 1))
        for index in range(primary_count)
    ]
    primary = media_helpers._panel_bytes(
        [frames[index] for index in primary_indices],
        labels=[f"P{slot} {seconds[index]:.2f}s" for slot, index in enumerate(primary_indices)],
        frame_width=int(media["primary_frame_width"]),
        quality=int(media["jpeg_quality"]),
    )
    proof_indices = list(range(len(frames)))
    panels = []
    per_panel = int(media["proof_frames_per_panel"])
    for panel_index, start in enumerate(range(0, len(proof_indices), per_panel)):
        indices = proof_indices[start : start + per_panel]
        panels.append(media_helpers._panel_bytes(
            [frames[index] for index in indices],
            labels=[f"E{index} {seconds[index]:.2f}s" for index in indices],
            frame_width=int(media["proof_frame_width"]),
            quality=int(media["jpeg_quality"]),
        ))
    return primary, panels, metadata


def _primary_call(
    client: OpenAI, *, sample: Any, panel: bytes, config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    slots = tuple(sample.answer_slots)
    prompt = (
        "Make a first-pass native answer from the uniformly sampled full-clip "
        "frames. Use only visible temporal evidence. Do not invent annotations, "
        "functional programs, or hidden events. Return probability mass for every "
        "option and list concise observed evidence and unresolved uncertainty.\n"
        + sample.format_question()
    )
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        payload, usage = media_helpers._json_call(
            client,
            model=str(model["id"]),
            system=(
                "Return JSON only: {answer:string, probabilities:{slot:number},"
                "observed_evidence:[string],unresolved_uncertainties:[string],"
                "reason:string}. answer must be the unique probability argmax."
            ),
            content=[
                {"type": "text", "text": prompt + (f"\nSchema error: {last_error}" if last_error else "")},
                {"type": "text", "text": "Uniform low-bandwidth temporal overview:"},
                media_helpers._image_content(panel),
            ],
            max_tokens=int(model["max_primary_tokens"]),
        )
        try:
            return parse_primary_receipt(payload, slots), payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError(f"primary receipt schema failed: {last_error}")


def _proof_call(
    client: OpenAI, *, sample: Any, panels: Sequence[bytes], config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    slots = tuple(sample.answer_slots)
    prompt = (
        "Execute a target-native candidate-factorized visual proof. Independently "
        "evaluate every option; do not compare against or assume any earlier answer. "
        "For each option emit exactly five typed steps in this order: "
        + ", ".join(PROOF_KINDS)
        + ". Mark a step SUPPORTED only when the visible frames support it, REFUTED "
        "when visible evidence contradicts it, and UNKNOWN when the required fact is "
        "not observable. CAUSAL_LINK must be UNKNOWN unless temporal evidence supports "
        "causality rather than mere co-occurrence. Then return calibrated native "
        "answer probabilities. No annotations, official programs, graphs, or gold are available.\n"
        + sample.format_question()
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, panel in enumerate(panels):
        content.extend([
            {"type": "text", "text": f"Dense chronological evidence panel {index + 1}/{len(panels)}:"},
            media_helpers._image_content(panel),
        ])
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        attempt = list(content)
        if last_error:
            attempt.append({"type": "text", "text": f"Schema error: {last_error}"})
        payload, usage = media_helpers._json_call(
            client,
            model=str(model["id"]),
            system=(
                "Return JSON only: {answer:string,probabilities:{slot:number},"
                "candidates:[{slot:string,support_probability:number,"
                "sensor_reliability:number,proof_steps:[{kind:string,status:"
                "SUPPORTED|REFUTED|UNKNOWN,confidence:number,visible_fact:string}]}],"
                "global_uncertainties:[string],reason:string}. Preserve option and "
                "typed-step order exactly; answer is the unique probability argmax."
            ),
            content=attempt,
            max_tokens=int(model["max_proof_tokens"]),
        )
        try:
            return parse_proof_receipt(payload, slots), payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError(f"proof receipt schema failed: {last_error}")


def _collect_one(
    sample: Any,
    *,
    benchmark: str,
    config: Mapping[str, Any],
    api_key: str,
    contract_sha256: str,
) -> dict[str, Any]:
    client = OpenAI(
        api_key=api_key,
        base_url=str(config["model"]["base_url"]),
        timeout=float(config["model"]["timeout_seconds"]),
        max_retries=int(config["model"]["max_retries"]),
    )
    primary_panel, proof_panels, metadata = _panels(sample, config)
    primary, primary_raw, primary_usage = _primary_call(
        client, sample=sample, panel=primary_panel, config=config,
    )
    proof, proof_raw, proof_usage = _proof_call(
        client, sample=sample, panels=proof_panels, config=config,
    )
    family = str(
        getattr(sample, "question_family", None)
        or getattr(sample, "question_type", "")
    )
    features = build_features(
        benchmark=benchmark, family=family, primary=primary, proof=proof,
    )
    # Gold is attached only after both runtime branches and features are frozen.
    gold = str(sample.answer)
    return {
        "schema_version": 15,
        "benchmark": benchmark,
        "split": str(config["target"]["split"]),
        "sample_id": str(sample.sample_id),
        "video_id": str(sample.video_id),
        "family": family,
        "gold_answer": gold,
        "primary": primary,
        "proof": proof,
        "primary_correct": primary["answer"] == gold,
        "proof_correct": proof["answer"] == gold,
        "uplift": int(proof["answer"] == gold) - int(primary["answer"] == gold),
        "features": list(map(float, features)),
        "sample_public": {
            "question": str(sample.question),
            "options": dict(sample.options),
            "answer_slots": list(sample.answer_slots),
            "video_path": str(sample.video_path),
            "clip_start_seconds": float(getattr(sample, "start_sec", 0.0) or 0.0),
            "clip_end_seconds": (
                float(sample.end_sec) if getattr(sample, "end_sec", None) is not None else None
            ),
        },
        "runtime_saw_gold_or_official_structure": False,
        "primary_raw": primary_raw,
        "proof_raw": proof_raw,
        "usage": {"primary": primary_usage, "proof": proof_usage},
        "video_metadata": metadata,
        "video_sha256": _sha256(Path(sample.video_path)),
        "primary_panel_sha256": hashlib.sha256(primary_panel).hexdigest(),
        "proof_panel_sha256": [hashlib.sha256(value).hexdigest() for value in proof_panels],
        "collection_contract_sha256": contract_sha256,
    }


def _load_samples(
    benchmark: str, ids: Sequence[str], config: Mapping[str, Any],
) -> dict[str, Any]:
    wrapper_root = Path(config["wrapper_root"])
    if str(wrapper_root) not in sys.path:
        sys.path.insert(0, str(wrapper_root))
    if benchmark == "star":
        from visual_reasoning_wrapper.benchmarks.star import iter_star_samples
        samples = iter_star_samples(
            "val", star_root=config["benchmarks"]["star"]["root"], sample_ids=ids,
            require_video=True,
        )
    elif benchmark == "nextqa":
        from visual_reasoning_wrapper.benchmarks.nextqa import iter_nextqa_samples
        samples = iter_nextqa_samples(
            "val", nextqa_root=config["benchmarks"]["nextqa"]["root"], sample_ids=ids,
            require_video=True,
        )
    else:
        raise ValueError(f"unsupported natural-video benchmark: {benchmark}")
    output = {str(sample.sample_id): sample for sample in samples}
    if set(output) != set(ids):
        raise ValueError(f"missing {benchmark} samples/videos: {sorted(set(ids) - set(output))}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = Path(config["source_receipt"])
    if _sha256(source_path) != config["frozen_lineage"]["source_receipt_sha256"]:
        raise ValueError("source receipt hash mismatch")
    validate_source_receipt(json.loads(source_path.read_text(encoding="utf-8")))
    manifest_path = Path(config["target"]["split_manifest"])
    if _sha256(manifest_path) != config["frozen_lineage"]["split_manifest_sha256"]:
        raise ValueError("natural-video split manifest hash mismatch")
    code_paths = {
        "collector_sha256": Path(__file__).resolve(),
        "feature_module_sha256": REPO / "src/motif_transfer/natural_video_recovery.py",
    }
    for key, path in code_paths.items():
        if _sha256(path) != config["frozen_lineage"].get(key):
            raise ValueError(f"collector lineage mismatch: {key}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = str(config["target"]["split"])
    if split != "development":
        raise ValueError("this frozen collector config is development-only")
    ordered_pairs = [
        (benchmark, sample_id)
        for benchmark in ("star", "nextqa")
        for sample_id in manifest["benchmarks"][benchmark]["splits"][split]
    ]
    samples = {
        benchmark: _load_samples(
            benchmark,
            [sample_id for name, sample_id in ordered_pairs if name == benchmark],
            config,
        )
        for benchmark in ("star", "nextqa")
    }
    contract_sha256 = _content_hash({
        "config_sha256": _sha256(args.config),
        "manifest_sha256": _sha256(manifest_path),
        "collector_sha256": _sha256(Path(__file__).resolve()),
        "feature_module_sha256": _sha256(REPO / "src/motif_transfer/natural_video_recovery.py"),
        "ordered_pairs": ordered_pairs,
    })
    keys = runpy.run_path(str(args.keys))
    api_key = keys.get(config["model"]["api_key_name"])
    if not api_key:
        raise SystemExit("configured natural-video API key is missing")
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.is_file():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        for row in payload:
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached natural-video receipt contract mismatch")
            existing[(str(row["benchmark"]), str(row["sample_id"]))] = row
    pending = [pair for pair in ordered_pairs if pair not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        ordered = [existing[pair] for pair in ordered_pairs if pair in existing]
        args.output.write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_one,
                samples[benchmark][sample_id],
                benchmark=benchmark,
                config=config,
                api_key=str(api_key),
                contract_sha256=contract_sha256,
            ): (benchmark, sample_id)
            for benchmark, sample_id in pending
        }
        for future in as_completed(futures):
            pair = futures[future]
            try:
                existing[pair] = future.result()
            except Exception as exc:
                print(json.dumps({
                    "failed": list(pair), "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                continue
            save()
            print(json.dumps({
                "completed": list(pair),
                "progress": f"{len(existing)}/{len(ordered_pairs)}",
            }), flush=True)
    missing = [pair for pair in ordered_pairs if pair not in existing]
    if missing:
        raise SystemExit(f"incomplete V15 development collection; rerun: {missing}")
    rows = [existing[pair] for pair in ordered_pairs]
    print(json.dumps({
        "status": "NATURAL_VIDEO_V15_DEVELOPMENT_COLLECTED",
        "samples": len(rows),
        "benchmark_counts": {
            benchmark: sum(row["benchmark"] == benchmark for row in rows)
            for benchmark in ("star", "nextqa")
        },
        "primary_correct": sum(row["primary_correct"] for row in rows),
        "proof_correct": sum(row["proof_correct"] for row in rows),
        "uplift_counts": {
            str(value): sum(row["uplift"] == value for row in rows)
            for value in (-1, 0, 1)
        },
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
