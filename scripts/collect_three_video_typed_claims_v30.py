#!/usr/bin/env python3
"""Collect candidate-isolated typed visual claims on the consumed V29 pilot."""

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
import run_structured_video_transfer as structured  # noqa: E402
from motif_transfer.typed_video_claim_grounder import (  # noqa: E402
    CHECK_KINDS,
    execute_binary_vector_guard,
    execute_mcq_guard,
    parse_typed_claim_receipt,
    rotate_bindings,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _candidate_rows(sample: Any) -> list[dict[str, str]]:
    if hasattr(sample, "candidates"):
        return [
            {"slot": str(index), "claim": str(value)}
            for index, value in enumerate(sample.candidates)
        ]
    return [
        {"slot": str(slot), "claim": str(value)}
        for slot, value in sample.options.items()
    ]


def _family(sample: Any) -> str:
    return str(
        getattr(sample, "question_family", None)
        or getattr(sample, "question_type", "")
    )


def _required_checks(benchmark: str, family: str) -> tuple[str, ...]:
    if benchmark == "clevrer" or family in {
        "Causal", "Temporal", "Sequence", "Interaction", "explanatory",
    }:
        return (
            "ENTITY_BINDING", "PRECONDITION", "POSTCONDITION",
            "DIRECTIONAL_OR_CAUSAL_LINK", "CLAIM_ENTAILMENT",
        )
    return ("ENTITY_BINDING", "CLAIM_ENTAILMENT")


def _claim_call(
    client: OpenAI,
    *,
    question: str,
    candidate_claim: str,
    family: str,
    panels: Sequence[bytes],
    frame_count: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    prompt = (
        "Independently verify exactly one candidate answer claim against chronological "
        "video frames. You cannot see its option slot or any competing candidates. Do "
        "not assume the candidate is true merely because it is mentioned. Search for "
        "both positive evidence and contradiction. PRECONDITION must state the specific "
        "candidate-required state before the event; POSTCONDITION must state its required "
        "state after the event. DIRECTIONAL_OR_CAUSAL_LINK must explicitly distinguish "
        "inverse actions such as pick-up versus put-down, open versus close, and sit "
        "versus lie. The same generic interaction cannot support inverse claims. UNKNOWN "
        "is mandatory when pixels do not distinguish the transition. For predictive or "
        "counterfactual claims, mark rollout-only checks as INFERRED and keep observed "
        "premises separate. Cite at most three sparse frame IDs per check. "
        "NOT_APPLICABLE is allowed except for CLAIM_ENTAILMENT. "
        "The final claim status must be determined by CLAIM_ENTAILMENT.\n"
        f"Question family: {family}\nQuestion: {question.strip()}\n"
        f"Single candidate answer claim: {candidate_claim.strip()}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, panel in enumerate(panels):
        content.extend([
            {"type": "text", "text": f"Chronological panel {index + 1}/{len(panels)}:"},
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
                "Return JSON only: {claim_status:SUPPORTED|REFUTED|UNKNOWN,"
                "confidence:number,checks:[{kind:string,status:SUPPORTED|REFUTED|"
                "UNKNOWN|NOT_APPLICABLE,confidence:number,basis:OBSERVED|INFERRED|"
                "NOT_APPLICABLE,evidence_frames:[integer],"
                "fact:string}],uncertainties:[string],reason:string}. checks must occur "
                "exactly in this order: " + ",".join(CHECK_KINDS) + ". Frame IDs are "
                f"F0..F{frame_count - 1}, but evidence_frames JSON values must be bare "
                "integers such as [0,7,12], never strings such as ['F0']. Never emit a "
                "slot, choice id, answer field, "
                "competing candidate, or gold judgement."
            ),
            content=attempt,
            max_tokens=int(model["max_claim_tokens"]),
        )
        try:
            parsed = parse_typed_claim_receipt(payload, frame_count=frame_count)
            return parsed.as_dict(), payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("typed candidate receipt retries exhausted: " + last_error)


def _execute(
    benchmark: str,
    family: str,
    baseline: str,
    bound: Sequence[Mapping[str, Any]],
    frame_count: int,
) -> dict[str, Any]:
    required = _required_checks(benchmark, family)
    # Reparse serialized receipts so the executor never consumes raw provider JSON.
    typed = [
        {
            "slot": str(row["slot"]),
            "receipt": parse_typed_claim_receipt(row["receipt"], frame_count=frame_count),
        }
        for row in bound
    ]
    if benchmark == "clevrer":
        return execute_binary_vector_guard(
            baseline, typed, required_checks=required,
        )
    return execute_mcq_guard(baseline, typed, required_checks=required)


def _collect_one(
    benchmark: str,
    sample: Any,
    baseline_row: Mapping[str, Any],
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
            float(sample.end_sec) if getattr(sample, "end_sec", None) is not None
            else None
        ),
        frame_count=int(media["proxy_frame_count"]),
        max_side=int(media["proxy_frame_max_side"]),
    )
    panels = v28._scout_panels(frames, metadata, config)
    panel_hashes = [hashlib.sha256(panel).hexdigest() for panel in panels]
    candidates = _candidate_rows(sample)
    family = _family(sample)
    bound = []
    raw = {}
    usage = {}
    for candidate in candidates:
        parsed, raw_payload, candidate_usage = _claim_call(
            client,
            question=str(sample.question),
            candidate_claim=str(candidate["claim"]),
            family=family,
            panels=panels,
            frame_count=len(frames),
            config=config,
        )
        slot = str(candidate["slot"])
        bound.append({"slot": slot, "claim": candidate["claim"], "receipt": parsed})
        raw[slot] = raw_payload
        usage[slot] = candidate_usage
    baseline = str(baseline_row["conditions"]["uniform_direct"]["answer"])
    authentic = _execute(benchmark, family, baseline, bound, len(frames))
    typed_for_rotation = [
        {
            "slot": str(row["slot"]),
            "receipt": parse_typed_claim_receipt(row["receipt"], frame_count=len(frames)),
        }
        for row in bound
    ]
    rotated_typed = rotate_bindings(typed_for_rotation)
    rotated_bound = [
        {"slot": row["slot"], "receipt": row["receipt"].as_dict()}
        for row in rotated_typed
    ]
    binding_control = _execute(
        benchmark, family, baseline, rotated_bound, len(frames),
    )

    # Gold access begins only after every candidate-isolated neural call and
    # both authentic/control symbolic executions freeze.
    gold = str(sample.answer)
    candidate_gold = {
        str(row["slot"]): (
            gold[int(row["slot"])] == "1" if benchmark == "clevrer"
            else str(row["slot"]) == gold
        )
        for row in bound
    }
    return {
        "schema_version": 30,
        "benchmark": benchmark,
        "split": "consumed_v29_candidate_isolated_development",
        "sample_id": str(sample.sample_id),
        "video_id": str(sample.video_id),
        "family": family,
        "gold_answer": gold,
        "baseline_answer": baseline,
        "baseline_correct": baseline == gold,
        "candidates": bound,
        "candidate_gold": candidate_gold,
        "authentic_execution": authentic,
        "authentic_correct": str(authentic["answer"]) == gold,
        "binding_control_execution": binding_control,
        "binding_control_correct": str(binding_control["answer"]) == gold,
        "raw_candidate_receipts": raw,
        "usage": usage,
        "video_metadata": metadata,
        "video_sha256": _sha256(Path(sample.video_path)),
        "panel_sha256": panel_hashes,
        "baseline_v29_row_sha256": _content_hash(baseline_row),
        "collection_contract_sha256": contract_sha256,
        "each_neural_call_saw_exactly_one_candidate": True,
        "candidate_slot_bound_after_inference": True,
        "source_skill_or_structure_available_at_runtime": False,
        "runtime_calls_saw_gold_competitors_or_official_structure": False,
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
    baseline_path = Path(config["baseline_v29_receipts"])
    if _sha256(baseline_path) != config["baseline_v29_receipts_sha256"]:
        raise ValueError("V30 baseline receipt hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_V28_GROUNDING_QUALIFICATION_CALLS":
        raise ValueError("V30 consumed manifest is not sealed")
    ordered_pairs = [
        (benchmark, str(row["sample_id"]))
        for benchmark in ("clevrer", "star", "nextqa")
        for row in manifest["benchmarks"][benchmark]
    ]
    baselines = {
        (str(row["benchmark"]), str(row["sample_id"])): row
        for row in json.loads(baseline_path.read_text(encoding="utf-8"))
    }
    if any(pair not in baselines for pair in ordered_pairs):
        raise ValueError("V30 baseline does not cover all frozen identities")
    samples = {
        benchmark: v28._load_samples(
            benchmark,
            [sample_id for name, sample_id in ordered_pairs if name == benchmark],
            config,
        )
        for benchmark in ("clevrer", "star", "nextqa")
    }
    contract_sha256 = _content_hash({
        "config_sha256": _sha256(args.config),
        "manifest_sha256": _sha256(manifest_path),
        "baseline_sha256": _sha256(baseline_path),
        "collector_sha256": _sha256(Path(__file__).resolve()),
        "typed_module_sha256": _sha256(REPO / "src/motif_transfer/typed_video_claim_grounder.py"),
        "ordered_pairs": ordered_pairs,
    })
    api_key = runpy.run_path(str(args.keys)).get(config["model"]["api_key_name"])
    if not api_key:
        raise SystemExit("configured OpenRouter key is missing")
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached V30 contract mismatch")
            existing[(str(row["benchmark"]), str(row["sample_id"]))] = row
    pending = [pair for pair in ordered_pairs if pair not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(
            [existing[pair] for pair in ordered_pairs if pair in existing],
            ensure_ascii=False, indent=2,
        ) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_one, benchmark, samples[benchmark][sample_id], baselines[pair],
                config=config, api_key=str(api_key), contract_sha256=contract_sha256,
            ): pair
            for pair in pending for benchmark, sample_id in [pair]
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
                "completed": list(pair), "progress": f"{len(existing)}/{len(ordered_pairs)}",
            }), flush=True)
    missing = [pair for pair in ordered_pairs if pair not in existing]
    if missing:
        raise SystemExit(f"incomplete V30 candidate claims; rerun: {missing}")
    rows = [existing[pair] for pair in ordered_pairs]
    print(json.dumps({
        "status": "THREE_VIDEO_TYPED_CLAIMS_V30_COLLECTED",
        "rows": len(rows),
        "candidate_calls": sum(len(row["candidates"]) for row in rows),
        "baseline_correct": sum(row["baseline_correct"] for row in rows),
        "authentic_correct": sum(row["authentic_correct"] for row in rows),
        "binding_control_correct": sum(row["binding_control_correct"] for row in rows),
        "reported_cost_usd": sum(
            float(value.get("cost", 0.0) or 0.0)
            for row in rows for value in row["usage"].values()
        ),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
