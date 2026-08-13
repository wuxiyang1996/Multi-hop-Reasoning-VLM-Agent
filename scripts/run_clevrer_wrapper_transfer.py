#!/usr/bin/env python3
"""CLEVRER visual-window matched forks for neural-symbolic transfer.

The model never sees CLEVRER functional programs, choice labels, or oracle
event histories.  It receives a three-frame low-bandwidth scout and may TEST
exactly one question-conditioned ``sample_frames(start_sec,end_sec,n)``
window before committing to the benchmark's native correct/wrong label.
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

from motif_transfer.active_video_transfer import (  # noqa: E402
    normalized_probabilities,
    stable_hash,
)
from motif_transfer.candidate_transfer_experiment import (  # noqa: E402
    evaluate_candidate_adaptation,
)
from motif_transfer.visual_wrapper_bridge import (  # noqa: E402
    build_video_registry,
    execute_video_intervention,
    route_question,
    video_tool_schemas,
)


ANSWER_SLOTS = ("A", "B")


def _collection_contract(config: Mapping[str, Any]) -> str:
    wrapper_root = Path(config["wrapper"]["root"])
    paths = (
        Path(__file__).resolve(),
        REPO / "src/motif_transfer/active_video_transfer.py",
        REPO / "src/motif_transfer/candidate_transfer_experiment.py",
        REPO / "src/motif_transfer/visual_wrapper_bridge.py",
        wrapper_root / "visual_reasoning_wrapper/tools_video.py",
        wrapper_root / "visual_reasoning_wrapper/benchmarks/clevrer.py",
    )
    return stable_hash({
        "config": config,
        "code_sha256": {
            str(path): media_helpers.file_sha256(path) for path in paths
        },
    })


def _load_samples(
    config: Mapping[str, Any], dataset_root: Path,
) -> dict[str, Any]:
    wrapper_root = Path(config["wrapper"]["root"])
    root_text = str(wrapper_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from visual_reasoning_wrapper.benchmarks.clevrer import (
        iter_clevrer_choice_samples,
    )

    requested = {
        sample_id
        for split in ("adaptation", "qualification", "held_out")
        for sample_id in config["splits"][split]
    }
    samples = iter_clevrer_choice_samples(
        str(config["target"]["split"]),
        clevrer_root=dataset_root,
        sample_ids=requested,
        require_video=True,
    )
    output = {sample.sample_id: sample for sample in samples}
    missing = sorted(requested - set(output))
    if missing:
        raise FileNotFoundError(
            f"frozen CLEVRER samples or videos are missing: {missing[:10]}"
        )
    return output


def _overview_panel(
    frames: Sequence[Any],
    *,
    seconds: Sequence[float],
    count: int,
    frame_width: int,
    quality: int,
) -> bytes:
    if count < 2 or count > len(frames):
        raise ValueError("overview_frame_count must be in [2, proxy_frame_count]")
    indices = [
        round(index * (len(frames) - 1) / (count - 1))
        for index in range(count)
    ]
    return media_helpers._panel_bytes(
        [frames[index] for index in indices],
        labels=[f"P{index} {seconds[index]:.2f}s" for index in indices],
        frame_width=frame_width,
        quality=quality,
    )


def _answer(
    client: OpenAI,
    *,
    model: str,
    sample: Any,
    overview_panel: bytes,
    evidence_panel: bytes | None,
    wrapper_receipt: Mapping[str, Any] | None,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = (
        sample.format_question()
        + "\nUse only visible object attributes, motion, temporal order, "
        "collisions, entries, and exits. For predictive questions, separate "
        "events already seen from events implied after the observed prefix. "
        "For counterfactual questions, infer how removing the named object "
        "changes the event chain. Return calibrated probability mass over A "
        "and B."
    )
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "Low-bandwidth three-frame scout:"},
        media_helpers._image_content(overview_panel),
    ]
    if evidence_panel is not None and wrapper_receipt is not None:
        compact = {
            "tool": wrapper_receipt["tool"],
            "arguments": wrapper_receipt["arguments"],
            "proxy_frame_indices": wrapper_receipt["proxy_frame_indices"],
        }
        content.extend([
            {
                "type": "text",
                "text": (
                    "Target-native visual TEST receipt: "
                    + json.dumps(compact, ensure_ascii=False)
                    + "\nFocused re-observation from exactly that window:"
                ),
            },
            media_helpers._image_content(evidence_panel),
        ])
    payload, usage = media_helpers._json_call(
        client,
        model=model,
        system=(
            "Return JSON only: {\"answer\":\"A or B\","
            "\"probabilities\":{\"A\":number,\"B\":number},"
            "\"observed_events\":[\"brief\"],"
            "\"symbolic_update\":\"brief event-graph update\","
            "\"reason\":\"brief\"}. Do not mention hidden annotations or "
            "functional programs."
        ),
        content=content,
        max_tokens=max_tokens,
    )
    probabilities = normalized_probabilities(
        payload.get("probabilities") or {}, answer_slots=ANSWER_SLOTS,
    )
    answer = str(payload.get("answer") or "").strip().upper()[:1]
    if answer not in ANSWER_SLOTS:
        answer = ANSWER_SLOTS[int(probabilities.argmax())]
    return {
        "answer": answer,
        "probabilities": {
            slot: float(value) for slot, value in zip(ANSWER_SLOTS, probabilities)
        },
        "observed_events": [str(value) for value in payload.get("observed_events", ())],
        "symbolic_update": str(payload.get("symbolic_update") or ""),
        "reason": str(payload.get("reason") or ""),
    }, usage


def _propose_actions(
    client: OpenAI,
    *,
    model: str,
    sample: Any,
    overview_panel: bytes,
    duration_seconds: float,
    tool_schemas: Sequence[Mapping[str, Any]],
    routing: Mapping[str, Any],
    candidate_count: int,
    frames_per_candidate: int,
    maximum_window_fraction: float,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    maximum_span = maximum_window_fraction * duration_seconds
    content = [{
        "type": "text",
        "text": (
            f"Propose exactly {candidate_count} distinct visual TEST windows "
            "for the following CLEVRER candidate. Every action must call "
            "sample_frames. Each window must test a different uncertainty: "
            "object identity/initial motion, a possible collision/causal "
            "precondition, or the late outcome. Use only the visible scout to "
            "choose windows; do not answer A/B. Windows must be within "
            f"0..{duration_seconds:.3f}s, span at most {maximum_span:.3f}s, "
            f"and request n={frames_per_candidate}. "
            f"Routing: {json.dumps(routing, ensure_ascii=False)}. "
            f"Tool schema: {json.dumps(list(tool_schemas), ensure_ascii=False)}. "
            f"Question: {sample.question} Candidate: {sample.candidate}"
        ),
    }, media_helpers._image_content(overview_panel)]
    payload, usage = media_helpers._json_call(
        client,
        model=model,
        system=(
            "Return JSON only: {\"actions\":[{\"candidate_id\":\"C0\","
            "\"tool\":\"sample_frames\",\"arguments\":{\"n\":int,"
            "\"start_sec\":number,\"end_sec\":number},"
            "\"score\":number,\"hypothesis\":\"visible event tested\"}]}."
        ),
        content=content,
        max_tokens=max_tokens,
    )
    actions = list(payload.get("actions") or ())
    if len(actions) != candidate_count:
        raise ValueError("planner did not return the frozen candidate count")
    output: list[dict[str, Any]] = []
    windows: set[tuple[float, float]] = set()
    for index, action in enumerate(actions):
        if str(action.get("tool") or "") != "sample_frames":
            raise ValueError("CLEVRER planner emitted a non-visual tool")
        arguments = dict(action.get("arguments") or {})
        start = max(0.0, float(arguments.get("start_sec", 0.0)))
        end = min(duration_seconds, float(arguments.get("end_sec", 0.0)))
        if end <= start:
            raise ValueError("CLEVRER planner emitted an empty window")
        if end - start > maximum_span:
            center = (start + end) / 2.0
            start = max(0.0, center - maximum_span / 2.0)
            end = min(duration_seconds, start + maximum_span)
            start = max(0.0, end - maximum_span)
        key = (round(start, 3), round(end, 3))
        if key in windows:
            raise ValueError("CLEVRER planner emitted duplicate windows")
        windows.add(key)
        score = float(action.get("score", 0.0))
        if not 0.0 <= score <= 1.0:
            raise ValueError("CLEVRER planner score is outside [0,1]")
        output.append({
            "candidate_id": f"C{index}",
            "proposed_candidate_id": str(action.get("candidate_id") or ""),
            "tool": "sample_frames",
            "arguments": {
                "n": frames_per_candidate,
                "start_sec": start,
                "end_sec": end,
            },
            "planner_score": score,
            "hypothesis": str(action.get("hypothesis") or ""),
        })
    return output, usage


def _collect_sample(
    sample: Any,
    *,
    config: Mapping[str, Any],
    api_key: str,
    contract_sha256: str,
) -> dict[str, Any]:
    path = Path(sample.video_path)
    media = config["media"]
    frames, video_meta = media_helpers._proxy_frames(
        path,
        frame_count=int(media["proxy_frame_count"]),
        max_side=int(media["proxy_frame_max_side"]),
    )
    seconds = video_meta["proxy_sample_seconds"]
    overview = _overview_panel(
        frames,
        seconds=seconds,
        count=int(media["overview_frame_count"]),
        frame_width=int(media["overview_frame_width"]),
        quality=int(media["jpeg_quality"]),
    )
    wrapper_root = Path(config["wrapper"]["root"])
    registry, proxy_fps = build_video_registry(
        frames,
        duration_seconds=float(video_meta["duration_seconds"]),
        wrapper_root=wrapper_root,
        required_tools=("sample_frames",),
    )
    routed_text = f"{sample.question} Candidate: {sample.candidate}"
    routing = route_question(
        routed_text, modality="video", wrapper_root=wrapper_root,
    ).as_dict()
    model_config = config["model"]
    client = OpenAI(
        api_key=api_key,
        base_url=str(model_config["base_url"]),
        timeout=float(model_config["timeout_seconds"]),
        max_retries=int(model_config["max_retries"]),
    )
    model = str(model_config["id"])
    proposals, proposal_usage = _propose_actions(
        client,
        model=model,
        sample=sample,
        overview_panel=overview,
        duration_seconds=float(video_meta["duration_seconds"]),
        tool_schemas=video_tool_schemas(
            registry, allowed_tools=("sample_frames",),
        ),
        routing=routing,
        candidate_count=int(config["interventions"]["candidate_count"]),
        frames_per_candidate=int(media["frames_per_candidate"]),
        maximum_window_fraction=float(config["interventions"][
            "maximum_window_fraction"
        ]),
        max_tokens=int(model_config["max_planner_tokens"]),
    )
    baseline, baseline_usage = _answer(
        client,
        model=model,
        sample=sample,
        overview_panel=overview,
        evidence_panel=None,
        wrapper_receipt=None,
        max_tokens=int(model_config["max_answer_tokens"]),
    )
    candidates = []
    for proposal in proposals:
        selected, wrapper_receipt = execute_video_intervention(
            registry,
            frames,
            tool="sample_frames",
            arguments=proposal["arguments"],
        )
        indices = wrapper_receipt["proxy_frame_indices"]
        evidence_panel = media_helpers._panel_bytes(
            selected,
            labels=[f"P{index} {seconds[index]:.2f}s" for index in indices],
            frame_width=int(media["evidence_frame_width"]),
            quality=int(media["jpeg_quality"]),
        )
        answer, usage = _answer(
            client,
            model=model,
            sample=sample,
            overview_panel=overview,
            evidence_panel=evidence_panel,
            wrapper_receipt=wrapper_receipt,
            max_tokens=int(model_config["max_answer_tokens"]),
        )
        candidates.append({
            **proposal,
            "descriptor": media_helpers._descriptor(
                wrapper_receipt["arguments"],
                duration_seconds=float(video_meta["duration_seconds"]),
                proxy_count=len(frames),
            ),
            "wrapper_receipt": wrapper_receipt,
            "evidence_sha256": hashlib.sha256(evidence_panel).hexdigest(),
            "answer": answer,
            "usage": usage,
        })
    video_meta["wrapper_proxy_fps"] = proxy_fps
    video_meta["overview_panel_sha256"] = hashlib.sha256(overview).hexdigest()
    return {
        "schema_version": 1,
        "benchmark": "CLEVRER",
        "collection_contract_sha256": contract_sha256,
        "sample_id": sample.sample_id,
        "family": sample.question_type,
        "answer_slots": list(ANSWER_SLOTS),
        "gold_answer": sample.answer,
        "sample": sample.to_dict(include_oracle_programs=False),
        "oracle_diagnostics": {
            "question_program": list(sample.question_program),
            "choice_program": list(sample.choice_program),
            "never_supplied_to_model": True,
        },
        "video_path": str(path),
        "video_sha256": media_helpers.file_sha256(path),
        "video_meta": video_meta,
        "wrapper_routing": routing,
        "wrapper_tool_names": registry.tool_names(),
        "proposal_usage": proposal_usage,
        "baseline": {"answer": baseline, "usage": baseline_usage},
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--split", choices=("adaptation",), default="adaptation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    contract_sha256 = _collection_contract(config)
    samples = _load_samples(config, args.dataset_root)
    sample_ids = list(config["splits"][args.split])
    keys = runpy.run_path(str(args.keys))
    key_name = str(config["model"]["api_key_name"])
    api_key = keys.get(key_name)
    if not api_key:
        raise SystemExit(f"{key_name} is missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = args.output_dir / "receipts.json"
    existing: dict[str, Any] = {}
    if receipts_path.is_file():
        existing = {
            str(row["sample_id"]): row
            for row in json.loads(receipts_path.read_text(encoding="utf-8"))
        }
        bad = [
            sample_id for sample_id, row in existing.items()
            if row.get("collection_contract_sha256") != contract_sha256
        ]
        if bad:
            raise SystemExit(f"receipt/config contract mismatch: {bad}")

    pending = [sample_id for sample_id in sample_ids if sample_id not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_sample,
                samples[sample_id],
                config=config,
                api_key=str(api_key),
                contract_sha256=contract_sha256,
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
        raise SystemExit(f"incomplete receipts; rerun to resume: {missing}")
    receipts = [existing[sample_id] for sample_id in sample_ids]

    controlled_path = Path(config["source"]["controlled_v3_config"])
    controlled = json.loads(controlled_path.read_text(encoding="utf-8"))
    if stable_hash(controlled) != config["source"][
        "controlled_v3_config_content_sha256"
    ]:
        raise SystemExit("controlled source config content hash mismatch")
    report, artifact = evaluate_candidate_adaptation(
        receipts, config=config, controlled_config=controlled,
    )
    artifact_path = args.output_dir / "target_grounder_candidate.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["receipts"] = {
        "path": str(receipts_path.resolve()),
        "sha256": media_helpers.file_sha256(receipts_path),
    }
    report["target_grounder_candidate"] = {
        "path": str(artifact_path.resolve()),
        "sha256": media_helpers.file_sha256(artifact_path),
        "content_sha256": artifact["artifact_sha256"],
    }
    report_path = args.output_dir / "adaptation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "baseline_accuracy": report["baseline_accuracy"],
        "cross_fitted_selector_accuracy": report["cross_fitted_selector_accuracy"],
        "oracle_candidate_accuracy": report["oracle_candidate_accuracy"],
        "conditions": report["conditions_cross_fitted"],
        "gates": report["gates"],
        "report": str(report_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
