#!/usr/bin/env python3
"""Collect candidate-local high-resolution adjudications for V30 supports."""

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
from motif_transfer.focused_video_claim_adjudicator import (  # noqa: E402
    focus_indices,
    fuse_supported_receipt,
    parse_focused_adjudication,
)
from motif_transfer.typed_video_claim_grounder import (  # noqa: E402
    parse_typed_claim_receipt,
    rotate_bindings,
)


def _public_typed_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": str(row["benchmark"]),
        "sample_id": str(row["sample_id"]),
        "video_sha256": str(row["video_sha256"]),
        "candidates": [
            {
                "slot": str(value["slot"]),
                "claim": str(value["claim"]),
                "receipt": value["receipt"],
            }
            for value in row["candidates"]
        ],
    }


def _public_direct_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": str(row["benchmark"]),
        "sample_id": str(row["sample_id"]),
        "answer": str(row["direct"]["answer"]),
        "panel_sha256": list(row["panel_sha256"]),
        "video_sha256": str(row["video_sha256"]),
    }


def _focus_panels(
    frames: Sequence[Any],
    seconds: Sequence[float],
    indices: Sequence[int],
    config: Mapping[str, Any],
) -> list[bytes]:
    media = config["media"]
    per_panel = int(media["focus_frames_per_panel"])
    return [
        v28._panel(
            frames,
            seconds,
            indices[start : start + per_panel],
            frame_width=int(media["focus_frame_width"]),
            jpeg_quality=int(media["jpeg_quality"]),
        )
        for start in range(0, len(indices), per_panel)
    ]


def _adjudicate_call(
    client: OpenAI,
    *,
    question: str,
    candidate_claim: str,
    panels: Sequence[bytes],
    visible_indices: Sequence[int],
    frame_count: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    prompt = (
        "Independently adjudicate one candidate answer using high-resolution "
        "chronological frames selected by an answer-free temporal proposal. You do "
        "not know the candidate slot, competing answers, a prior verdict, or gold. "
        "The selected view may omit decisive evidence, so use UNKNOWN rather than "
        "guessing. A visible object after a pose/viewpoint change does not by itself "
        "prove a state transition. For an action, identify the specific required "
        "precondition, postcondition, and direction: e.g. hand-supported to "
        "surface-supported proves put-down; a visible aperture does not prove opening "
        "unless a closed-to-open closure transition is visible. Explicitly record the "
        "strongest alternative explanation. CLAIM_ENTAILMENT may be SUPPORTED only "
        "when entity binding, precondition, postcondition, and transition direction "
        "are all SUPPORTED. Cite at most three frames.\n"
        f"Visible original frame IDs: {list(visible_indices)}\n"
        f"Question: {question.strip()}\n"
        f"Single candidate answer claim: {candidate_claim.strip()}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, panel in enumerate(panels):
        content.extend([
            {
                "type": "text",
                "text": f"Focused chronological panel {index + 1}/{len(panels)}:",
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
                "Return JSON only: {entity_binding:SUPPORTED|REFUTED|UNKNOWN|"
                "NOT_APPLICABLE,precondition:SUPPORTED|REFUTED|UNKNOWN|"
                "NOT_APPLICABLE,postcondition:SUPPORTED|REFUTED|UNKNOWN|"
                "NOT_APPLICABLE,transition_direction:SUPPORTED|REFUTED|UNKNOWN|"
                "NOT_APPLICABLE,claim_entailment:SUPPORTED|REFUTED|UNKNOWN,"
                "evidence_frames:[integer],alternative_explanation:string,"
                "confidence:number,reason:string}. Frame IDs are original F0..F"
                f"{frame_count - 1}; evidence values must be bare integers. Never emit "
                "an answer field, candidate slot, choice id, competitor, or correctness."
            ),
            content=attempt,
            max_tokens=int(model["max_adjudication_tokens"]),
        )
        try:
            parsed = parse_focused_adjudication(payload, frame_count=frame_count)
            if any(index not in visible_indices for index in parsed.evidence_frames):
                raise ValueError("adjudication cited a frame outside its focused view")
            return parsed.as_dict(), payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("focused adjudication retries exhausted: " + last_error)


def _collect_one(
    benchmark: str,
    sample: Any,
    typed_public: Mapping[str, Any],
    direct_public: Mapping[str, Any],
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
    if v30._sha256(Path(sample.video_path)) != typed_public["video_sha256"]:
        raise ValueError("V32 video bytes do not match V30")
    if typed_public["video_sha256"] != direct_public["video_sha256"]:
        raise ValueError("V30/V31 video lineage mismatch")
    seconds = list(map(float, metadata["proxy_sample_seconds"]))
    fused = []
    adjudications = {}
    raw = {}
    usage = {}
    focus_views = {}
    for candidate in typed_public["candidates"]:
        slot = str(candidate["slot"])
        original = parse_typed_claim_receipt(
            candidate["receipt"], frame_count=len(frames),
        )
        final_receipt = original.as_dict()
        if original.claim_status == "SUPPORTED":
            indices = focus_indices(
                original.as_dict(),
                frame_count=len(frames),
                radius=int(media["focus_radius"]),
                limit=int(media["focus_frame_limit"]),
            )
            panels = _focus_panels(frames, seconds, indices, config)
            parsed, raw_payload, candidate_usage = _adjudicate_call(
                client,
                question=str(sample.question),
                candidate_claim=str(candidate["claim"]),
                panels=panels,
                visible_indices=indices,
                frame_count=len(frames),
                config=config,
            )
            adjudication = parse_focused_adjudication(
                parsed, frame_count=len(frames),
            )
            final_receipt = fuse_supported_receipt(
                original.as_dict(), adjudication, frame_count=len(frames),
            )
            adjudications[slot] = parsed
            raw[slot] = raw_payload
            usage[slot] = candidate_usage
            focus_views[slot] = {
                "frame_indices": list(indices),
                "panel_sha256": [hashlib.sha256(panel).hexdigest() for panel in panels],
            }
        fused.append({
            "slot": slot,
            "claim": str(candidate["claim"]),
            "receipt": final_receipt,
        })
    baseline = str(direct_public["answer"])
    family = v30._family(sample)
    authentic = v30._execute(benchmark, family, baseline, fused, len(frames))
    typed_for_rotation = [
        {
            "slot": str(row["slot"]),
            "receipt": parse_typed_claim_receipt(
                row["receipt"], frame_count=len(frames),
            ),
        }
        for row in fused
    ]
    rotated = rotate_bindings(typed_for_rotation)
    control_bound = [
        {"slot": row["slot"], "receipt": row["receipt"].as_dict()}
        for row in rotated
    ]
    binding_control = v30._execute(
        benchmark, family, baseline, control_bound, len(frames),
    )

    # Gold becomes available only after target calls and both executions freeze.
    gold = str(sample.answer)
    return {
        "schema_version": 32,
        "benchmark": benchmark,
        "split": "consumed_v31_focused_adjudication_development",
        "sample_id": str(sample.sample_id),
        "video_id": str(sample.video_id),
        "family": family,
        "gold_answer": gold,
        "baseline_answer": baseline,
        "baseline_correct": baseline == gold,
        "fused_candidates": fused,
        "focused_adjudications": adjudications,
        "authentic_execution": authentic,
        "authentic_correct": str(authentic["answer"]) == gold,
        "binding_control_execution": binding_control,
        "binding_control_correct": str(binding_control["answer"]) == gold,
        "raw_adjudications": raw,
        "usage": usage,
        "focus_views": focus_views,
        "video_metadata": metadata,
        "video_sha256": typed_public["video_sha256"],
        "collection_contract_sha256": contract_sha256,
        "each_call_saw_exactly_one_candidate": True,
        "prior_verdict_visible_to_adjudicator": False,
        "candidate_slot_bound_after_inference": True,
        "source_skill_or_structure_available_at_runtime": False,
        "gold_available_before_calls_or_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--only-sample", action="append", default=[])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    typed_path = Path(config["typed_v30_receipts"])
    direct_path = Path(config["matched_v31_receipts"])
    if v30._sha256(typed_path) != config["typed_v30_receipts_sha256"]:
        raise ValueError("V32 V30 receipt hash mismatch")
    if v30._sha256(direct_path) != config["matched_v31_receipts_sha256"]:
        raise ValueError("V32 V31 receipt hash mismatch")
    typed_rows = {
        (str(row["benchmark"]), str(row["sample_id"])): _public_typed_row(row)
        for row in json.loads(typed_path.read_text(encoding="utf-8"))
        if row["benchmark"] in {"star", "nextqa"}
    }
    direct_rows = {
        (str(row["benchmark"]), str(row["sample_id"])): _public_direct_row(row)
        for row in json.loads(direct_path.read_text(encoding="utf-8"))
        if row["benchmark"] in {"star", "nextqa"}
    }
    if set(typed_rows) != set(direct_rows):
        raise ValueError("V32 V30/V31 natural-video identities differ")
    only = set(map(str, args.only_sample))
    ordered_pairs = [
        pair for pair in typed_rows
        if not only or pair[1] in only
    ]
    ordered_pairs.sort(key=lambda pair: (("star", "nextqa").index(pair[0]), pair[1]))
    if only and {pair[1] for pair in ordered_pairs} != only:
        raise ValueError("an --only-sample identity was not found")
    samples = {
        benchmark: v28._load_samples(
            benchmark,
            [sample_id for name, sample_id in ordered_pairs if name == benchmark],
            config,
        )
        for benchmark in sorted({name for name, _ in ordered_pairs})
    }
    contract_sha256 = v30._content_hash({
        "config_sha256": v30._sha256(args.config),
        "typed_sha256": v30._sha256(typed_path),
        "direct_sha256": v30._sha256(direct_path),
        "collector_sha256": v30._sha256(Path(__file__).resolve()),
        "adjudicator_sha256": v30._sha256(
            REPO / "src/motif_transfer/focused_video_claim_adjudicator.py"
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
                raise ValueError("cached V32 contract mismatch")
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
                direct_rows[pair],
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
        raise SystemExit(f"incomplete V32 focused adjudication; rerun: {missing}")
    rows = [existing[pair] for pair in ordered_pairs]
    print(json.dumps({
        "status": "NATURAL_VIDEO_FOCUSED_ADJUDICATION_V32_COLLECTED",
        "rows": len(rows),
        "adjudication_calls": sum(len(row["usage"]) for row in rows),
        "baseline_correct": sum(row["baseline_correct"] for row in rows),
        "authentic_correct": sum(row["authentic_correct"] for row in rows),
        "binding_control_correct": sum(
            row["binding_control_correct"] for row in rows
        ),
        "reported_cost_usd": sum(
            float(value.get("cost", 0.0) or 0.0)
            for row in rows for value in row["usage"].values()
        ),
        "output": str(args.output.resolve()),
        "output_sha256": v30._sha256(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
