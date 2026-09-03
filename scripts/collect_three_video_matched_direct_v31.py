#!/usr/bin/env python3
"""Collect a same-model, same-frame direct baseline for typed V30 receipts."""

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
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_natural_video_v19_formal as transport  # noqa: E402
import collect_three_video_grounding_qualification_v28 as v28  # noqa: E402
import collect_three_video_typed_claims_v30 as v30  # noqa: E402
import run_structured_video_transfer as structured  # noqa: E402
from motif_transfer.typed_video_claim_grounder import (  # noqa: E402
    parse_typed_claim_receipt,
    rotate_bindings,
)


def _valid_answer(sample: Any, answer: str) -> bool:
    validator = getattr(sample, "validate_answer", None)
    if callable(validator):
        return bool(validator(answer))
    return answer in tuple(map(str, sample.answer_slots))


def _direct_call(
    client: OpenAI,
    *,
    sample: Any,
    panels: Sequence[bytes],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    prompt = (
        "Answer the video question from the complete chronological proxy view. "
        "Inspect transitions carefully and keep predictive or causal uncertainty "
        "separate from visible facts.\n" + sample.format_question()
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, panel in enumerate(panels):
        content.extend([
            {
                "type": "text",
                "text": f"Chronological panel {index + 1}/{len(panels)}:",
            },
            v28.media_helpers._image_content(panel),
        ])
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        attempt = list(content)
        if last_error:
            attempt.append({"type": "text", "text": "Schema error: " + last_error})
        payload, usage = transport._provider_json_call(
            client,
            model=str(model["id"]),
            system=(
                "Return JSON only: {answer:string,confidence:number,"
                "observed_evidence:[string],uncertainties:[string],reason:string}."
            ),
            content=attempt,
            max_tokens=int(model["max_answer_tokens"]),
        )
        try:
            answer = str(payload["answer"]).strip()
            confidence = float(payload["confidence"])
            evidence = payload["observed_evidence"]
            uncertainties = payload["uncertainties"]
            if not _valid_answer(sample, answer):
                raise ValueError(f"answer violates benchmark contract: {answer!r}")
            if not 0 <= confidence <= 1:
                raise ValueError("confidence is outside [0,1]")
            if not isinstance(evidence, list) or not all(
                isinstance(value, str) for value in evidence
            ):
                raise ValueError("observed_evidence must be a string list")
            if not isinstance(uncertainties, list) or not all(
                isinstance(value, str) for value in uncertainties
            ):
                raise ValueError("uncertainties must be a string list")
            parsed = {
                "answer": answer,
                "confidence": confidence,
                "observed_evidence": list(evidence),
                "uncertainties": list(uncertainties),
                "reason": str(payload.get("reason") or "").strip(),
            }
            return parsed, payload, usage
        except (KeyError, TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("matched direct schema retries exhausted: " + last_error)


def _public_typed_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Strip every outcome field before a row enters target inference."""
    return {
        "benchmark": str(row["benchmark"]),
        "sample_id": str(row["sample_id"]),
        "video_sha256": str(row["video_sha256"]),
        "panel_sha256": list(row["panel_sha256"]),
        "candidates": [
            {
                "slot": str(value["slot"]),
                "claim": str(value["claim"]),
                "receipt": value["receipt"],
            }
            for value in row["candidates"]
        ],
    }


def _collect_one(
    benchmark: str,
    sample: Any,
    typed_public: Mapping[str, Any],
    *,
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
    media = config["media"]
    frames, metadata = structured._sample_clip(
        Path(sample.video_path),
        start_sec=float(getattr(sample, "start_sec", 0.0) or 0.0),
        end_sec=(
            float(sample.end_sec)
            if getattr(sample, "end_sec", None) is not None
            else None
        ),
        frame_count=int(media["proxy_frame_count"]),
        max_side=int(media["proxy_frame_max_side"]),
    )
    panels = v28._scout_panels(frames, metadata, config)
    panel_hashes = [hashlib.sha256(panel).hexdigest() for panel in panels]
    if panel_hashes != list(typed_public["panel_sha256"]):
        raise ValueError("V31 direct panels do not exactly match V30 typed panels")
    if v30._sha256(Path(sample.video_path)) != typed_public["video_sha256"]:
        raise ValueError("V31 video bytes do not match V30")
    direct, raw_direct, usage = _direct_call(
        client, sample=sample, panels=panels, config=config,
    )
    baseline = str(direct["answer"])
    family = v30._family(sample)
    bound = list(typed_public["candidates"])
    authentic = v30._execute(benchmark, family, baseline, bound, len(frames))
    typed_for_rotation = [
        {
            "slot": str(row["slot"]),
            "receipt": parse_typed_claim_receipt(
                row["receipt"], frame_count=len(frames),
            ),
        }
        for row in bound
    ]
    rotated = rotate_bindings(typed_for_rotation)
    rotated_bound = [
        {"slot": row["slot"], "receipt": row["receipt"].as_dict()}
        for row in rotated
    ]
    binding_control = v30._execute(
        benchmark, family, baseline, rotated_bound, len(frames),
    )

    # Outcome access starts only after direct inference and both executions freeze.
    gold = str(sample.answer)
    return {
        "schema_version": 31,
        "benchmark": benchmark,
        "split": "consumed_v30_matched_gemini_development",
        "sample_id": str(sample.sample_id),
        "video_id": str(sample.video_id),
        "family": family,
        "gold_answer": gold,
        "direct": direct,
        "direct_correct": baseline == gold,
        "authentic_execution": authentic,
        "authentic_correct": str(authentic["answer"]) == gold,
        "binding_control_execution": binding_control,
        "binding_control_correct": str(binding_control["answer"]) == gold,
        "raw_direct": raw_direct,
        "usage": usage,
        "video_metadata": metadata,
        "video_sha256": typed_public["video_sha256"],
        "panel_sha256": panel_hashes,
        "typed_candidate_receipts_sha256": v30._content_hash(bound),
        "collection_contract_sha256": contract_sha256,
        "same_model_as_typed_grounder": True,
        "same_proxy_frames_as_typed_grounder": True,
        "same_panel_encoding_as_typed_grounder": True,
        "source_skill_or_structure_available_at_runtime": False,
        "gold_available_before_direct_or_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    typed_path = Path(config["typed_v30_receipts"])
    if v30._sha256(typed_path) != config["typed_v30_receipts_sha256"]:
        raise ValueError("V31 typed V30 receipt hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_V28_GROUNDING_QUALIFICATION_CALLS":
        raise ValueError("V31 consumed manifest is not sealed")
    ordered_pairs = [
        (benchmark, str(row["sample_id"]))
        for benchmark in ("clevrer", "star", "nextqa")
        for row in manifest["benchmarks"][benchmark]
    ]
    typed_rows = {
        (str(row["benchmark"]), str(row["sample_id"])): _public_typed_row(row)
        for row in json.loads(typed_path.read_text(encoding="utf-8"))
    }
    if set(ordered_pairs) != set(typed_rows):
        raise ValueError("V31 identities do not exactly match V30")
    samples = {
        benchmark: v28._load_samples(
            benchmark,
            [sample_id for name, sample_id in ordered_pairs if name == benchmark],
            config,
        )
        for benchmark in ("clevrer", "star", "nextqa")
    }
    contract_sha256 = v30._content_hash({
        "config_sha256": v30._sha256(args.config),
        "manifest_sha256": v30._sha256(manifest_path),
        "typed_sha256": v30._sha256(typed_path),
        "collector_sha256": v30._sha256(Path(__file__).resolve()),
        "typed_module_sha256": v30._sha256(
            REPO / "src/motif_transfer/typed_video_claim_grounder.py"
        ),
        "ordered_pairs": ordered_pairs,
    })
    api_key = runpy.run_path(str(args.keys)).get(config["model"]["api_key_name"])
    if not api_key:
        raise SystemExit("configured OpenRouter key is missing")
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached V31 contract mismatch")
            existing[(str(row["benchmark"]), str(row["sample_id"]))] = row
    pending = [pair for pair in ordered_pairs if pair not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(
            [existing[pair] for pair in ordered_pairs if pair in existing],
            ensure_ascii=False,
            indent=2,
        ) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_one,
                benchmark,
                samples[benchmark][sample_id],
                typed_rows[pair],
                config=config,
                api_key=str(api_key),
                contract_sha256=contract_sha256,
            ): pair
            for pair in pending for benchmark, sample_id in [pair]
        }
        for future in as_completed(futures):
            pair = futures[future]
            try:
                existing[pair] = future.result()
            except Exception as exc:
                print(json.dumps({
                    "failed": list(pair),
                    "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                continue
            save()
            print(json.dumps({
                "completed": list(pair),
                "progress": f"{len(existing)}/{len(ordered_pairs)}",
            }), flush=True)
    missing = [pair for pair in ordered_pairs if pair not in existing]
    if missing:
        raise SystemExit(f"incomplete V31 matched direct; rerun: {missing}")
    rows = [existing[pair] for pair in ordered_pairs]
    print(json.dumps({
        "status": "THREE_VIDEO_MATCHED_DIRECT_V31_COLLECTED",
        "rows": len(rows),
        "direct_correct": sum(row["direct_correct"] for row in rows),
        "authentic_correct": sum(row["authentic_correct"] for row in rows),
        "binding_control_correct": sum(
            row["binding_control_correct"] for row in rows
        ),
        "reported_cost_usd": sum(
            float(row["usage"].get("cost", 0.0) or 0.0) for row in rows
        ),
        "output": str(args.output.resolve()),
        "output_sha256": v30._sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
