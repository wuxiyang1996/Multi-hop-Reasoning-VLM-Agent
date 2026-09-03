#!/usr/bin/env python3
"""Run matched typed-probe transfer on CLEVRER, STAR, or NExT-QA."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_active_video_wrapper_transfer as media_helpers  # noqa: E402

from motif_transfer.active_video_transfer import (  # noqa: E402
    build_source_value_models,
    stable_hash,
)
from motif_transfer.structured_video_transfer import (  # noqa: E402
    FIXED_TEST_CONDITIONS,
    ParsedTargetWorldModel,
    evaluate_fixed_test_budget,
    parse_target_world_model,
    parse_typed_probe_receipt,
)
from motif_transfer.visual_wrapper_bridge import (  # noqa: E402
    build_video_registry,
    execute_video_intervention,
)


def _contract(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    paths = (
        Path(__file__).resolve(),
        REPO / "src/motif_transfer/video_dynamics_mdp.py",
        REPO / "src/motif_transfer/structured_video_transfer.py",
        REPO / "src/motif_transfer/visual_wrapper_bridge.py",
    )
    return stable_hash({
        "config": config,
        "manifest": manifest,
        "code_sha256": {
            str(path): media_helpers.file_sha256(path) for path in paths
        },
    })


def _load_samples(
    benchmark: str,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    split: str,
) -> tuple[list[str], dict[str, Any]]:
    wrapper_root = str(config["wrapper"]["root"])
    if wrapper_root not in sys.path:
        sys.path.insert(0, wrapper_root)
    ids = list(manifest["benchmarks"][benchmark]["splits"][split])
    target = config["benchmarks"][benchmark]
    if benchmark == "clevrer":
        from visual_reasoning_wrapper.benchmarks.clevrer import (
            iter_clevrer_question_samples,
        )
        rows = iter_clevrer_question_samples(
            target["annotation_split"], clevrer_root=target["root"],
            sample_ids=ids, require_video=True,
        )
    elif benchmark == "star":
        from visual_reasoning_wrapper.benchmarks.star import iter_star_samples
        rows = iter_star_samples(
            target["annotation_split"], star_root=target["root"],
            sample_ids=ids, require_video=True,
        )
    elif benchmark == "nextqa":
        from visual_reasoning_wrapper.benchmarks.nextqa import iter_nextqa_samples
        rows = iter_nextqa_samples(
            target["annotation_split"], nextqa_root=target["root"],
            sample_ids=ids, require_video=True,
        )
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")
    indexed = {sample.sample_id: sample for sample in rows}
    missing = [sample_id for sample_id in ids if sample_id not in indexed]
    if missing:
        raise FileNotFoundError(
            f"{benchmark} frozen videos/samples are missing: {missing[:12]}"
        )
    return ids, indexed


def _sample_clip(
    path: Path,
    *,
    start_sec: float,
    end_sec: float | None,
    frame_count: int,
    max_side: int,
) -> tuple[list[Image.Image], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        capture.release()
        raise RuntimeError(f"invalid video metadata: {path}")
    full_duration = total_frames / fps
    clip_start = max(0.0, float(start_sec))
    clip_end = min(full_duration, float(end_sec) if end_sec is not None else full_duration)
    if clip_end <= clip_start:
        capture.release()
        raise RuntimeError(f"invalid clip boundaries for {path}")
    relative = np.linspace(0.0, max(0.0, clip_end - clip_start - 1.0 / fps), frame_count)
    frames: list[Image.Image] = []
    for offset in relative:
        capture.set(cv2.CAP_PROP_POS_MSEC, (clip_start + float(offset)) * 1000.0)
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"failed decoding {path} at {clip_start + offset:.3f}s")
        frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max(frame.size) > max_side:
            scale = max_side / max(frame.size)
            frame = frame.resize(
                (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))
            )
        frames.append(frame)
    capture.release()
    return frames, {
        "fps": fps,
        "total_frames": total_frames,
        "full_duration_seconds": full_duration,
        "clip_start_seconds": clip_start,
        "clip_end_seconds": clip_end,
        "duration_seconds": clip_end - clip_start,
        "proxy_sample_seconds": list(map(float, relative)),
    }


def _panel(
    frames: Sequence[Image.Image],
    seconds: Sequence[float],
    *,
    count: int,
    width: int,
    quality: int,
) -> bytes:
    indices = [round(index * (len(frames) - 1) / (count - 1)) for index in range(count)]
    return media_helpers._panel_bytes(
        [frames[index] for index in indices],
        labels=[f"P{index} {seconds[index]:.2f}s" for index in indices],
        frame_width=width,
        quality=quality,
    )


def _answer_contract(
    sample: Any, benchmark: str,
) -> tuple[Sequence[str] | None, int | None, tuple[str, ...], str]:
    if benchmark == "clevrer":
        length = int(sample.answer_length)
        answer_space = tuple(
            "".join(bits) for bits in itertools.product("01", repeat=length)
        )
        return None, length, answer_space, (
            f"one of the complete {length}-bit vectors listed below"
        )
    slots = tuple(sample.answer_slots)
    return slots, None, slots, "one of " + ", ".join(slots)


def _propose_world_model(
    client: OpenAI,
    *,
    config: Mapping[str, Any],
    benchmark: str,
    sample: Any,
    scout: bytes,
    duration_seconds: float,
) -> tuple[ParsedTargetWorldModel, dict[str, Any], dict[str, Any]]:
    model_config = config["model"]
    intervention = config["interventions"]
    valid_answers, binary_length, answer_space, contract = _answer_contract(
        sample, benchmark,
    )
    exhaustive = bool(intervention.get("answer_complete_hypotheses", False))
    if exhaustive:
        particle_count = len(answer_space)
        answer_instruction = (
            "Return exactly one world hypothesis for every candidate in this exact "
            f"order: {list(answer_space)}. Do not omit, duplicate, reorder, or invent "
            "an answer. Assign every candidate a positive prior_weight."
        )
    else:
        particle_count = int(intervention["particle_count"])
        answer_instruction = (
            f"Construct exactly {particle_count} alternative target-native "
            "world/event/dynamics hypotheses. Preserve uncertainty by using at least "
            "two distinct answers."
        )
    prompt = (
        sample.format_question()
        + f"\nThe provided clip runs from relative 0.0 to {duration_seconds:.3f} seconds. "
        + answer_instruction
        + " Each hypothesis must execute to a native answer "
        f"that is {contract}. Keep each event_graph_summary concise and state the "
        "answer-conditioned temporal/causal events rather than merely paraphrasing the answer. "
        f"Then propose exactly {intervention['probe_count']} distinct typed visual probes. "
        "A probe must test a finite predicate about named entities in a bounded observed "
        "normalized window_fraction satisfying 0 <= start < end <= 1; the wrapper maps "
        "this fraction deterministically to clip seconds. It must not ask the "
        "final question. For each probe estimate predicate-true "
        "probability under every world particle, in the same frozen answer order, and "
        "expected sensor reliability. Prefer probes whose outcome distinguishes multiple "
        "answer hypotheses and avoid predicates predicted identically under all hypotheses. Allowed "
        f"predicate kinds: {', '.join(intervention['allowed_predicates'])}. Do not use or "
        "claim access to functional programs, answer labels, situation graphs, or annotations."
        + (
            " Probe-design contract: " + str(intervention["probe_design_contract"])
            if intervention.get("probe_design_contract") else ""
        )
    )
    system = (
        "Return JSON only with keys world_particles, typed_probes, and optional entity_catalog. "
        "world_particles rows: native_answer, prior_weight, event_graph_summary. "
        "typed_probes rows: predicate_kind, entity_refs, window_fraction, "
        "true_probability_by_particle, expected_sensor_reliability, rationale. "
        "When a typed BIND/RELATE contract is requested, entity_catalog rows are "
        "{entity_id,visual_description}, and typed_probes use target_event_role and "
        "entity_ids instead of free-text entity_refs. "
        "Latent predicate probabilities must be in [0,1]; reliability must be in [0.5,1]."
    )
    last_error = ""
    for attempt in range(int(model_config["schema_retries"])):
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt + (f"\nPrevious schema error: {last_error}" if last_error else "")},
            {"type": "text", "text": "Low-bandwidth scout frames:"},
            media_helpers._image_content(scout),
        ]
        payload, usage = media_helpers._json_call(
            client, model=str(model_config["id"]), system=system,
            content=content, max_tokens=int(model_config["max_world_model_tokens"]),
        )
        try:
            parsed = parse_target_world_model(
                payload,
                duration_seconds=duration_seconds,
                particle_count=particle_count,
                probe_count=int(intervention["probe_count"]),
                valid_answers=valid_answers,
                binary_vector_length=binary_length,
                required_answers=answer_space if exhaustive else None,
                maximum_object_present_probes=(
                    int(intervention["maximum_object_present_probes"])
                    if "maximum_object_present_probes" in intervention else None
                ),
                minimum_non_presence_probes=int(
                    intervention.get("minimum_non_presence_probes", 0)
                ),
                minimum_likelihood_span=float(
                    intervention.get("minimum_likelihood_span", 0.0)
                ),
                require_bind_relate_roles=bool(
                    intervention.get("require_bind_relate_roles", False)
                ),
                minimum_bind_probes=int(
                    intervention.get("minimum_bind_probes", 0)
                ),
                minimum_relate_probes=int(
                    intervention.get("minimum_relate_probes", 0)
                ),
                minimum_relate_likelihood_span=float(
                    intervention.get("minimum_relate_likelihood_span", 0.0)
                ),
            )
            return parsed, payload, usage
        except ValueError as exc:
            last_error = str(exc)
    raise ValueError(f"world-model schema failed after retries: {last_error}")


def _ground_probe(
    client: OpenAI,
    *,
    config: Mapping[str, Any],
    probe: Any,
    evidence_panel: bytes,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    model_config = config["model"]
    prompt = (
        "Measure only this typed predicate in the supplied focused frames. "
        f"Predicate kind: {probe.predicate_kind}. Entities: {list(probe.entity_refs)}. "
        f"Relative window: [{probe.start_sec:.3f}, {probe.end_sec:.3f}]. "
        "Do not infer or answer any benchmark question. Return observed_true as a JSON "
        "boolean and sensor_reliability in [0.5,1]. Reliability 0.5 means the visual "
        "measurement is uninformative. Briefly report the target-native measurement."
    )
    last_error = ""
    for _ in range(int(model_config["schema_retries"])):
        payload, usage = media_helpers._json_call(
            client,
            model=str(model_config["id"]),
            system=(
                "Return JSON only: {\"observed_true\":bool,"
                "\"sensor_reliability\":number,\"measurement\":\"brief\"}. "
                "You never receive or predict the final answer."
            ),
            content=[
                {"type": "text", "text": prompt + (f" Schema error: {last_error}" if last_error else "")},
                media_helpers._image_content(evidence_panel),
            ],
            max_tokens=int(model_config["max_probe_tokens"]),
        )
        try:
            receipt = parse_typed_probe_receipt(
                payload,
                probe=probe,
                evidence_sha256=(hashlib.sha256(evidence_panel).hexdigest(),),
            )
            return receipt, payload, usage
        except ValueError as exc:
            last_error = str(exc)
    raise ValueError(f"typed probe schema failed after retries: {last_error}")


def _collect_sample(
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
    frames, metadata = _sample_clip(
        Path(sample.video_path),
        start_sec=start,
        end_sec=float(raw_end) if raw_end is not None else None,
        frame_count=int(media["proxy_frame_count"]),
        max_side=int(media["proxy_frame_max_side"]),
    )
    seconds = metadata["proxy_sample_seconds"]
    scout = _panel(
        frames, seconds,
        count=int(media["scout_frame_count"]),
        width=int(media["scout_frame_width"]),
        quality=int(media["jpeg_quality"]),
    )
    model_config = config["model"]
    client = OpenAI(
        api_key=api_key,
        base_url=str(model_config["base_url"]),
        timeout=float(model_config["timeout_seconds"]),
        max_retries=int(model_config["max_retries"]),
    )
    world_model, world_payload, world_usage = _propose_world_model(
        client,
        config=config,
        benchmark=benchmark,
        sample=sample,
        scout=scout,
        duration_seconds=float(metadata["duration_seconds"]),
    )
    registry, _ = build_video_registry(
        frames,
        duration_seconds=float(metadata["duration_seconds"]),
        wrapper_root=config["wrapper"]["root"],
        required_tools=("sample_frames",),
    )
    receipts = {}
    probe_payloads = {}
    probe_usage = {}
    wrapper_receipts = {}
    for probe in world_model.probes:
        selected, wrapper_receipt = execute_video_intervention(
            registry,
            frames,
            tool="sample_frames",
            arguments={
                "n": int(media["frames_per_probe"]),
                "start_sec": probe.start_sec,
                "end_sec": probe.end_sec,
            },
        )
        indices = wrapper_receipt["proxy_frame_indices"]
        evidence = media_helpers._panel_bytes(
            selected,
            labels=[f"P{index} {seconds[index]:.2f}s" for index in indices],
            frame_width=int(media["evidence_frame_width"]),
            quality=int(media["jpeg_quality"]),
        )
        receipt, payload, usage = _ground_probe(
            client, config=config, probe=probe, evidence_panel=evidence,
        )
        receipts[probe.probe_id] = receipt
        probe_payloads[probe.probe_id] = payload
        probe_usage[probe.probe_id] = usage
        wrapper_receipts[probe.probe_id] = wrapper_receipt
    family = str(
        getattr(sample, "question_family", None)
        or getattr(sample, "question_type", "")
    )
    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "sample_id": sample.sample_id,
        "family": family,
        "gold_answer": sample.answer,
        "collection_contract_sha256": contract_sha256,
        "sample": sample.to_dict(),
        "video_sha256": media_helpers.file_sha256(Path(sample.video_path)),
        "video_metadata": metadata,
        "scout_sha256": hashlib.sha256(scout).hexdigest(),
        "world_model": {
            "particles": [asdict(value) for value in world_model.particles],
            "particle_summaries": list(world_model.particle_summaries),
            "probes": [asdict(value) for value in world_model.probes],
            "probe_rationales": list(world_model.probe_rationales),
        },
        "world_model_raw": world_payload,
        "world_model_usage": world_usage,
        "probe_receipts": {
            key: asdict(value) for key, value in receipts.items()
        },
        "probe_grounder_raw": probe_payloads,
        "probe_usage": probe_usage,
        "wrapper_receipts": wrapper_receipts,
        "runtime_oracle_inputs": False,
    }


def _rehydrate(row: Mapping[str, Any]) -> tuple[ParsedTargetWorldModel, dict[str, Any]]:
    from motif_transfer.video_dynamics_mdp import (
        PredicateProbe, PredicateProbeReceipt, WorldParticle,
    )
    model = row["world_model"]
    parsed = ParsedTargetWorldModel(
        tuple(WorldParticle(**value) for value in model["particles"]),
        tuple(model["particle_summaries"]),
        tuple(PredicateProbe(**value) for value in model["probes"]),
        tuple(model["probe_rationales"]),
    )
    receipts = {
        key: PredicateProbeReceipt(**value)
        for key, value in row["probe_receipts"].items()
    }
    return parsed, receipts


def _report(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    source_models: Mapping[str, Any],
) -> dict[str, Any]:
    evaluated = []
    test_budget = int(config["interventions"]["max_tests"])
    for row in rows:
        model, receipts = _rehydrate(row)
        result = evaluate_fixed_test_budget(
            sample_id=str(row["sample_id"]),
            gold_answer=str(row["gold_answer"]),
            world_model=model,
            probe_receipts=receipts,
            source_models=source_models,
            test_budget=test_budget,
        )
        result["family"] = row["family"]
        evaluated.append(result)
    count = len(evaluated)
    baseline = sum(row["baseline_correct"] for row in evaluated)
    oracle = sum(row["oracle_correct"] for row in evaluated)
    covered = sum(row["gold_answer"] in row["answer_space"] for row in evaluated)
    contrast = sum(row["authentic_action_contrast"] for row in evaluated)
    conditions = {
        condition: {
            "correct": sum(row["conditions"][condition]["correct"] for row in evaluated),
            "accuracy": sum(row["conditions"][condition]["correct"] for row in evaluated) / count,
            "selected_probe_sequences": {
                "->".join(row["conditions"][condition]["selected_probe_ids"]): sum(
                    other["conditions"][condition]["selected_probe_ids"]
                    == row["conditions"][condition]["selected_probe_ids"]
                    for other in evaluated
                )
                for row in evaluated
            },
        }
        for condition in FIXED_TEST_CONDITIONS
    }
    gates_config = config["adaptation_gates"]
    authentic = conditions["authentic_source_plus_target"]["correct"]
    gates = {
        "all_receipts_complete": count > 0,
        "gold_answer_world_coverage": covered / count >= float(
            gates_config["minimum_gold_answer_world_coverage_fraction"]
        ),
        "oracle_probe_headroom": oracle - baseline >= int(
            gates_config["minimum_oracle_headroom_samples"]
        ),
        "authentic_action_contrast": contrast >= int(
            gates_config["minimum_authentic_action_contrasts"]
        ),
        "authentic_above_target_only": authentic > conditions[
            "target_native_information_gain"
        ]["correct"],
        "authentic_above_target_expected_accuracy": authentic > conditions[
            "target_native_expected_accuracy"
        ]["correct"],
        "authentic_above_shuffled": authentic > conditions[
            "shuffled_source_plus_target"
        ]["correct"],
        "authentic_above_marginal": authentic > conditions[
            "source_marginal_plus_target"
        ]["correct"],
    }
    return {
        "schema_version": 1,
        "status": "ADAPTATION_PREFLIGHT_PASS" if all(gates.values()) else "ADAPTATION_PREFLIGHT_FAIL",
        "samples": count,
        "test_budget": test_budget,
        "baseline": {"correct": baseline, "accuracy": baseline / count},
        "oracle_probe": {"correct": oracle, "accuracy": oracle / count},
        "gold_answer_world_coverage": {"samples": covered, "fraction": covered / count},
        "authentic_action_contrasts": contrast,
        "conditions": conditions,
        "gates": gates,
        "rows": evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("clevrer", "star", "nextqa"), required=True)
    parser.add_argument("--split", choices=("adaptation",), default="adaptation")
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--resume-collection-contract",
        help=(
            "Explicit prior collection contract for resuming receipts after an "
            "analysis-only code change. Existing receipts must all match it."
        ),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
    computed_contract_sha256 = _contract(config, manifest)
    contract_sha256 = (
        str(args.resume_collection_contract)
        if args.resume_collection_contract else computed_contract_sha256
    )
    sample_ids, samples = _load_samples(
        args.benchmark, config, manifest, args.split,
    )
    keys = runpy.run_path(str(args.keys))
    api_key = keys.get(config["model"]["api_key_name"])
    if not api_key:
        raise SystemExit("configured API key is missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = args.output_dir / "receipts.json"
    existing = {}
    if receipts_path.is_file():
        existing = {
            row["sample_id"]: row
            for row in json.loads(receipts_path.read_text(encoding="utf-8"))
        }
        if any(
            row.get("collection_contract_sha256") != contract_sha256
            for row in existing.values()
        ):
            raise SystemExit("cached receipt contract mismatch")
        if args.resume_collection_contract is None and (
            contract_sha256 != computed_contract_sha256
        ):
            raise SystemExit("resume contract drift")
    pending = [sample_id for sample_id in sample_ids if sample_id not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_sample,
                samples[sample_id],
                benchmark=args.benchmark,
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

    if "typed_summary" in config["source"]:
        source_path = Path(config["source"]["typed_summary"])
        if media_helpers.file_sha256(source_path) != config["source"][
            "typed_summary_sha256"
        ]:
            raise SystemExit("typed source summary hash mismatch")
        source_summary = json.loads(source_path.read_text(encoding="utf-8"))
        if source_summary.get("status") != "SOURCE_TYPED_GATE_PASSED":
            raise SystemExit("typed source gate is not passed")
        collection_report = {
            "schema_version": 1,
            "status": "ADAPTATION_COLLECTION_COMPLETE_TYPED_IR_ANALYSIS_REQUIRED",
            "benchmark": args.benchmark,
            "split": args.split,
            "samples": len(rows := [existing[sample_id] for sample_id in sample_ids]),
            "collection_contract_sha256": contract_sha256,
            "analysis_code_contract_sha256": computed_contract_sha256,
            "source_summary_sha256": config["source"]["typed_summary_sha256"],
            "receipts": {
                "path": str(receipts_path.resolve()),
                "sha256": media_helpers.file_sha256(receipts_path),
            },
            "runtime_oracle_inputs": False,
        }
        report_path = args.output_dir / "adaptation_collection_report.json"
        report_path.write_text(
            json.dumps(collection_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(collection_report, ensure_ascii=False, indent=2))
        return

    controlled = json.loads(Path(
        config["source"]["controlled_v3_config"]
    ).read_text(encoding="utf-8"))
    if stable_hash(controlled) != config["source"]["controlled_v3_config_content_sha256"]:
        raise SystemExit("controlled source content hash mismatch")
    source_models = build_source_value_models(
        controlled,
        seed=int(config["source"]["model_seed"]),
        objective_test_cost=float(config["source"]["target_objective_test_cost"]),
    )
    rows = [existing[sample_id] for sample_id in sample_ids]
    report = _report(rows, config=config, source_models=source_models)
    report["benchmark"] = args.benchmark
    report["split"] = args.split
    report["collection_contract_sha256"] = contract_sha256
    report["analysis_code_contract_sha256"] = computed_contract_sha256
    report["receipts"] = {
        "path": str(receipts_path.resolve()),
        "sha256": media_helpers.file_sha256(receipts_path),
    }
    report_path = args.output_dir / "adaptation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "baseline": report["baseline"],
        "oracle_probe": report["oracle_probe"],
        "conditions": report["conditions"],
        "gates": report["gates"],
        "report": str(report_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
