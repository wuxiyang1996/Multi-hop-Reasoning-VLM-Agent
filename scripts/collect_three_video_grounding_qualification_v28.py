#!/usr/bin/env python3
"""Collect source-free semantic-grounding factorials on three video benchmarks."""

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
import run_active_video_wrapper_transfer as media_helpers  # noqa: E402
import run_structured_video_transfer as structured  # noqa: E402
from motif_transfer.video_grounding_qualification import (  # noqa: E402
    EventGroundingReceipt,
    EventLedgerReceipt,
    ledger_localized_indices,
    ledger_prompt_text,
    localized_indices,
    parse_event_grounding_receipt,
    parse_event_ledger_receipt,
    receipt_prompt_text,
    shifted_indices,
    uniform_indices,
)
from motif_transfer.visual_wrapper_bridge import (  # noqa: E402
    build_video_transition_grounding_registry,
    execute_transition_grounding,
)


CONDITIONS = (
    "uniform_direct",
    "uniform_receipt",
    "localized_direct",
    "localized_receipt",
    "shifted_receipt",
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


def _load_samples(
    benchmark: str,
    sample_ids: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    wrapper_root = str(config["wrapper_root"])
    if wrapper_root not in sys.path:
        sys.path.insert(0, wrapper_root)
    root = str(config["benchmarks"][benchmark]["root"])
    if benchmark == "clevrer":
        from visual_reasoning_wrapper.benchmarks.clevrer import (
            iter_clevrer_question_samples,
        )
        rows = iter_clevrer_question_samples(
            "validation", clevrer_root=root, sample_ids=sample_ids,
            require_video=True,
        )
    elif benchmark == "star":
        from visual_reasoning_wrapper.benchmarks.star import iter_star_samples
        rows = iter_star_samples(
            "val", star_root=root, sample_ids=sample_ids, require_video=True,
        )
    elif benchmark == "nextqa":
        from visual_reasoning_wrapper.benchmarks.nextqa import iter_nextqa_samples
        rows = iter_nextqa_samples(
            "val", nextqa_root=root, sample_ids=sample_ids, require_video=True,
        )
    else:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    output = {str(sample.sample_id): sample for sample in rows}
    missing = sorted(set(sample_ids) - set(output))
    if missing:
        raise FileNotFoundError(f"missing {benchmark} pilot samples/videos: {missing}")
    return output


def _candidate_concepts(sample: Any) -> list[str]:
    if hasattr(sample, "candidates"):
        return [str(value) for value in sample.candidates]
    return [str(value) for value in sample.options.values()]


def _grounding_public_prompt(sample: Any) -> str:
    concepts = "\n".join(f"- {value}" for value in _candidate_concepts(sample))
    return (
        f"Question: {str(sample.question).strip()}\n"
        "Candidate concepts are shown only to identify which entities and events "
        "need visual grounding; do not select, rank, support, or reject one:\n"
        f"{concepts}"
    )


def _family(sample: Any) -> str:
    return str(
        getattr(sample, "question_family", None)
        or getattr(sample, "question_type", "")
    )


def _panel(
    frames: Sequence[Any],
    seconds: Sequence[float],
    indices: Sequence[int],
    *,
    frame_width: int,
    jpeg_quality: int,
) -> bytes:
    return media_helpers._panel_bytes(
        [frames[index] for index in indices],
        labels=[f"F{index} +{float(seconds[index]):.2f}s" for index in indices],
        frame_width=frame_width,
        quality=jpeg_quality,
    )


def _scout_panels(
    frames: Sequence[Any], metadata: Mapping[str, Any], config: Mapping[str, Any],
) -> list[bytes]:
    media = config["media"]
    seconds = list(metadata["proxy_sample_seconds"])
    per_panel = int(media["scout_frames_per_panel"])
    return [
        _panel(
            frames, seconds, list(range(start, min(start + per_panel, len(frames)))),
            frame_width=int(media["scout_frame_width"]),
            jpeg_quality=int(media["jpeg_quality"]),
        )
        for start in range(0, len(frames), per_panel)
    ]


def _compact_transition_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": str(receipt.get("protocol")),
        "selected_transition_indices": list(receipt["selected_transition_indices"]),
        "comparisons": [
            {
                "frame_a": int(row["before_index"]),
                "frame_b": int(row["after_index"]),
                "mean_difference": float(row["comparison"]["mean_difference"]),
                "changed_pixel_pct": float(row["comparison"]["changed_pixel_pct"]),
            }
            for row in receipt["comparisons"]
        ],
    }


def _ground_call(
    client: OpenAI,
    *,
    sample: Any,
    panels: Sequence[bytes],
    transition_receipt: Mapping[str, Any],
    frame_count: int,
    config: Mapping[str, Any],
) -> tuple[EventGroundingReceipt | EventLedgerReceipt, dict[str, Any], dict[str, Any]]:
    model = config["model"]
    candidate_blind_ledger = (
        config["grounding"]["semantic_tool"]
        == "CANDIDATE_BLIND_MULTI_EVENT_LEDGER_V2"
    )
    prompt = (
        (
            "Construct a compact ledger of one to six salient visible events and "
            "state transitions that may be useful for the question. "
            if candidate_blind_ledger else
            "Ground the single most question-relevant visible event or state transition "
        )
        + "in the chronological proxy frames. The wrapper's pixel-change tool receipt "
        "is only a candidate-time proposal and may be semantically wrong. Inspect the "
        "frames yourself. Do not answer the question, select an option, rank candidates, "
        "or emit any answer/choice field. "
        + (
            "You are not shown answer candidates. Record only events positively visible "
            "in the frames; do not invent a candidate entity. Use PARTIAL for an event "
            "whose transition is only partly visible and report overall coverage. For "
            "each event cite only one to three maximally discriminative evidence frames "
            "(prefer a before/after pair), never every frame in an interval. Avoid "
            "redundant events that restate the same transition. "
            if candidate_blind_ledger else
            "If the required event is predictive, counterfactual, outside the visible "
            "clip, or otherwise not directly visible, use PARTIAL or UNOBSERVED and "
            "state the uncertainty. "
        )
        + "Frame IDs must refer to "
        f"F0..F{frame_count - 1}.\n"
        + (
            f"Question only (no answer candidates): {str(sample.question).strip()}"
            if candidate_blind_ledger else _grounding_public_prompt(sample)
        )
        + "\nWrapper transition proposal: "
        + json.dumps(transition_receipt, ensure_ascii=False, separators=(",", ":"))
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, panel in enumerate(panels):
        content.extend([
            {"type": "text", "text": f"Chronological scout panel {index + 1}/{len(panels)}:"},
            media_helpers._image_content(panel),
        ])
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        attempt = list(content)
        if last_error:
            attempt.append({"type": "text", "text": "Schema error: " + last_error})
        system = (
            "Return JSON only: {events:[{event_id:E0|E1|...,subject:string,"
            "predicate:string,object:string,observability:OBSERVED|PARTIAL,"
            "start_frame:integer,end_frame:integer,evidence_frames:[integer],"
            "before_state:string,after_state:string,confidence:number,"
            "uncertainties:[string],reason:string}],coverage:SUFFICIENT|PARTIAL|"
            "INSUFFICIENT,uncertainties:[string]}. Use consecutive event IDs. "
            "Every event must cite one to three visible frames. Never emit an answer, answer slot, "
            "choice id, option preference, correctness judgement, predicted future, "
            "or counterfactual outcome."
            if candidate_blind_ledger else
            "Return JSON only: {subject:string,predicate:string,object:string,"
            "observability:OBSERVED|PARTIAL|UNOBSERVED,start_frame:integer|null,"
            "end_frame:integer|null,evidence_frames:[integer],before_state:string,"
            "after_state:string,confidence:number,uncertainties:[string],reason:string}. "
            "evidence_frames must be unique, chronological, and within start/end. "
            "UNOBSERVED may use null interval and empty evidence. Never emit an answer, "
            "answer slot, choice id, option preference, or correctness judgement."
        )
        payload, usage = transport._provider_json_call(
            client,
            model=str(model["id"]),
            system=system,
            content=attempt,
            max_tokens=int(model["max_ground_tokens"]),
        )
        try:
            if candidate_blind_ledger:
                return (
                    parse_event_ledger_receipt(payload, frame_count=frame_count),
                    payload,
                    usage,
                )
            return (
                parse_event_grounding_receipt(payload, frame_count=frame_count),
                payload,
                usage,
            )
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("event grounding schema retries exhausted: " + last_error)


def _valid_answer(sample: Any, answer: str) -> bool:
    validator = getattr(sample, "validate_answer", None)
    if callable(validator):
        return bool(validator(answer))
    return answer in tuple(map(str, sample.answer_slots))


def _answer_call(
    client: OpenAI,
    *,
    sample: Any,
    panel: bytes,
    condition: str,
    receipt: EventGroundingReceipt | EventLedgerReceipt | None,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    receipt_instruction = (
        "No semantic event receipt is available; reason only from the matched frame view."
        if receipt is None
        else (
            "A separate answer-free target-native event grounder emitted the following "
            "receipt. Treat it as fallible evidence, verify it against the shown frame IDs, "
            "and do not assume its confidence is calibrated:\n"
            + (
                ledger_prompt_text(receipt)
                if isinstance(receipt, EventLedgerReceipt)
                else receipt_prompt_text(receipt)
            )
        )
    )
    prompt = (
        "Answer this video question using only the shown evidence view and the disclosed "
        "receipt, if present. Missing events must remain uncertain.\n"
        f"Condition: {condition}\n{receipt_instruction}\n{sample.format_question()}"
    )
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        content = [
            {"type": "text", "text": prompt + ("\nSchema error: " + last_error if last_error else "")},
            {"type": "text", "text": "Matched-budget chronological evidence view:"},
            media_helpers._image_content(panel),
        ]
        payload, usage = transport._provider_json_call(
            client,
            model=str(model["id"]),
            system=(
                "Return JSON only: {answer:string,confidence:number,"
                "observed_evidence:[string],uncertainties:[string],reason:string}."
            ),
            content=content,
            max_tokens=int(model["max_answer_tokens"]),
        )
        try:
            answer = str(payload["answer"]).strip()
            confidence = float(payload["confidence"])
            evidence = payload["observed_evidence"]
            uncertainties = payload["uncertainties"]
            if not _valid_answer(sample, answer):
                raise ValueError(f"answer violates the benchmark contract: {answer!r}")
            if not 0 <= confidence <= 1:
                raise ValueError("confidence is outside [0,1]")
            if not isinstance(evidence, list) or not all(isinstance(v, str) for v in evidence):
                raise ValueError("observed_evidence must be a string list")
            if not isinstance(uncertainties, list) or not all(
                isinstance(v, str) for v in uncertainties
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
    raise ValueError("answer schema retries exhausted: " + last_error)


def _collect_one(
    benchmark: str,
    sample: Any,
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
    transition_registry, proxy_fps = build_video_transition_grounding_registry(
        frames,
        duration_seconds=float(metadata["duration_seconds"]),
        wrapper_root=config["wrapper_root"],
    )
    _transition_frames, transition_receipt = execute_transition_grounding(
        transition_registry,
        frames,
        pair_count=int(config["grounding"]["transition_pair_count"]),
        uniform_anchor_count=int(config["grounding"]["uniform_anchor_count"]),
        threshold=float(config["grounding"]["pixel_change_threshold"]),
    )
    compact_transition = _compact_transition_receipt(transition_receipt)
    receipt, raw_receipt, ground_usage = _ground_call(
        client,
        sample=sample,
        panels=_scout_panels(frames, metadata, config),
        transition_receipt=compact_transition,
        frame_count=len(frames),
        config=config,
    )
    budget = int(media["answer_frame_count"])
    uniform = uniform_indices(len(frames), budget)
    localized = (
        ledger_localized_indices(receipt, frame_count=len(frames), budget=budget)
        if isinstance(receipt, EventLedgerReceipt)
        else localized_indices(receipt, frame_count=len(frames), budget=budget)
    )
    shifted = shifted_indices(localized, frame_count=len(frames))
    indices_by_condition = {
        "uniform_direct": uniform,
        "uniform_receipt": uniform,
        "localized_direct": localized,
        "localized_receipt": localized,
        "shifted_receipt": shifted,
    }
    results: dict[str, Any] = {}
    raw_answers: dict[str, Any] = {}
    usage: dict[str, Any] = {"event_grounder": ground_usage}
    panel_hashes: dict[str, str] = {}
    seconds = list(metadata["proxy_sample_seconds"])
    for condition in CONDITIONS:
        panel = _panel(
            frames,
            seconds,
            indices_by_condition[condition],
            frame_width=int(media["answer_frame_width"]),
            jpeg_quality=int(media["jpeg_quality"]),
        )
        parsed, raw, condition_usage = _answer_call(
            client,
            sample=sample,
            panel=panel,
            condition=condition,
            receipt=(None if condition.endswith("direct") else receipt),
            config=config,
        )
        results[condition] = parsed
        raw_answers[condition] = raw
        usage[condition] = condition_usage
        panel_hashes[condition] = hashlib.sha256(panel).hexdigest()

    # Outcome access is deliberately delayed until the event receipt, all five
    # matched answer branches, and the destructive temporal control are frozen.
    gold = str(sample.answer)
    return {
        "schema_version": int(config["schema_version"]),
        "benchmark": benchmark,
        "split": "consumed_adaptation_grounding_qualification_pilot",
        "sample_id": str(sample.sample_id),
        "video_id": str(sample.video_id),
        "family": _family(sample),
        "gold_answer": gold,
        "conditions": results,
        "correct": {
            condition: str(results[condition]["answer"]) == gold
            for condition in CONDITIONS
        },
        "event_grounding_receipt": receipt.as_dict(),
        "raw_event_grounding_receipt": raw_receipt,
        "wrapper_transition_receipt": compact_transition,
        "frame_indices": indices_by_condition,
        "frame_budget": budget,
        "uniform_and_localized_differ": uniform != localized,
        "localized_and_shifted_differ": localized != shifted,
        "raw_answers": raw_answers,
        "usage": usage,
        "video_metadata": metadata,
        "proxy_fps": proxy_fps,
        "video_sha256": _sha256(Path(sample.video_path)),
        "panel_sha256": panel_hashes,
        "collection_contract_sha256": contract_sha256,
        "source_skill_or_structure_available_at_runtime": False,
        "runtime_calls_saw_gold_or_official_programs": False,
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_V28_GROUNDING_QUALIFICATION_CALLS":
        raise ValueError("V28 grounding manifest is not sealed")
    if manifest.get("source_transfer_enabled") is not False:
        raise ValueError("V28 must remain source-free")
    source_manifest_path = Path(manifest["source_manifest"])
    if _sha256(source_manifest_path) != manifest["source_manifest_sha256"]:
        raise ValueError("V28 consumed-source manifest hash mismatch")
    if tuple(config.get("conditions") or ()) != CONDITIONS:
        raise ValueError("V28 matched condition contract drift")
    ordered_pairs = [
        (benchmark, str(row["sample_id"]))
        for benchmark in ("clevrer", "star", "nextqa")
        for row in manifest["benchmarks"][benchmark]
    ]
    samples = {
        benchmark: _load_samples(
            benchmark,
            [sample_id for name, sample_id in ordered_pairs if name == benchmark],
            config,
        )
        for benchmark in ("clevrer", "star", "nextqa")
    }
    contract_sha256 = _content_hash({
        "config_sha256": _sha256(args.config),
        "manifest_sha256": _sha256(manifest_path),
        "collector_sha256": _sha256(Path(__file__).resolve()),
        "receipt_module_sha256": _sha256(
            REPO / "src/motif_transfer/video_grounding_qualification.py"
        ),
        "wrapper_bridge_sha256": _sha256(
            REPO / "src/motif_transfer/visual_wrapper_bridge.py"
        ),
        "ordered_pairs": ordered_pairs,
    })
    key_values = runpy.run_path(str(args.keys))
    api_key = key_values.get(config["model"]["api_key_name"])
    if not api_key:
        raise SystemExit("configured OpenRouter key is missing")
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached V28 contract mismatch")
            existing[(str(row["benchmark"]), str(row["sample_id"]))] = row
    pending = [pair for pair in ordered_pairs if pair not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(
            json.dumps(
                [existing[pair] for pair in ordered_pairs if pair in existing],
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_one,
                benchmark,
                samples[benchmark][sample_id],
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
        raise SystemExit(f"incomplete V28 grounding collection; rerun: {missing}")
    rows = [existing[pair] for pair in ordered_pairs]
    print(json.dumps({
        "status": (
            f"THREE_VIDEO_GROUNDING_QUALIFICATION_V"
            f"{int(config['schema_version'])}_COLLECTED"
        ),
        "rows": len(rows),
        "condition_correct": {
            condition: sum(bool(row["correct"][condition]) for row in rows)
            for condition in CONDITIONS
        },
        "reported_cost_usd": sum(
            float(value.get("cost", 0.0) or 0.0)
            for row in rows for value in row["usage"].values()
        ),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
