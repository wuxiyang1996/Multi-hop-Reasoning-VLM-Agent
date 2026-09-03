#!/usr/bin/env python3
"""Run matched wrapper-grounded STAR Interaction development branches."""

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

import collect_natural_video_recovery_v15 as paired  # noqa: E402
import collect_natural_video_v19_formal as transport  # noqa: E402
import collect_star_interaction_v24_fresh as v24  # noqa: E402
from motif_transfer.natural_video_recovery import (  # noqa: E402
    PROOF_KINDS,
    parse_primary_receipt,
    parse_proof_receipt,
)
from motif_transfer.natural_video_symbolic_controls import execute_recovery  # noqa: E402
from motif_transfer.sokoban_video_recovery import validate_source_receipt  # noqa: E402
from motif_transfer.visual_wrapper_bridge import (  # noqa: E402
    build_video_transition_grounding_registry,
    execute_transition_grounding,
    transition_grounding_tool_schemas,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _active_panels(
    sample: Any,
    *,
    config: Mapping[str, Any],
) -> tuple[list[bytes], dict[str, Any], dict[str, Any]]:
    media = config["media"]
    frames, metadata = paired.structured._sample_clip(
        Path(sample.video_path),
        start_sec=float(getattr(sample, "start_sec", 0.0) or 0.0),
        end_sec=(
            float(sample.end_sec)
            if getattr(sample, "end_sec", None) is not None
            else None
        ),
        frame_count=int(media["proof_frame_count"]),
        max_side=int(media["proxy_frame_max_side"]),
    )
    registry, proxy_fps = build_video_transition_grounding_registry(
        frames,
        duration_seconds=float(metadata["duration_seconds"]),
        wrapper_root=config["wrapper_root"],
    )
    allowed = {
        row["function"]["name"]
        for row in transition_grounding_tool_schemas(registry)
    }
    if allowed != {"detect_scene_changes", "compare_frames"}:
        raise RuntimeError(f"unexpected transition-grounding tool surface: {allowed}")
    grounded, receipt = execute_transition_grounding(
        registry,
        frames,
        pair_count=int(media["transition_pair_count"]),
        uniform_anchor_count=int(media["uniform_anchor_pair_count"]),
        threshold=float(media["scene_change_threshold"]),
    )
    seconds = list(map(float, metadata["proxy_sample_seconds"]))
    labels: list[str] = []
    for comparison in receipt["comparisons"]:
        pair_id = str(comparison["pair_id"])
        before = int(comparison["before_index"])
        after = int(comparison["after_index"])
        comparison["before_seconds"] = seconds[before]
        comparison["after_seconds"] = seconds[after]
        labels.extend((
            f"{pair_id} BEFORE E{before} {seconds[before]:.2f}s",
            f"{pair_id} AFTER E{after} {seconds[after]:.2f}s",
        ))
    per_panel = int(media["proof_frames_per_panel"])
    panels = [
        paired.media_helpers._panel_bytes(
            grounded[start : start + per_panel],
            labels=labels[start : start + per_panel],
            frame_width=int(media["proof_frame_width"]),
            quality=int(media["jpeg_quality"]),
        )
        for start in range(0, len(grounded), per_panel)
    ]
    receipt["proxy_fps"] = proxy_fps
    receipt["panel_layout"] = "chronological adjacent BEFORE/AFTER pairs"
    receipt["panel_count"] = len(panels)
    receipt["panel_frame_count"] = len(grounded)
    return panels, metadata, receipt


def _active_generic_call(
    client: OpenAI,
    *,
    sample: Any,
    panels: Sequence[bytes],
    grounding_receipt: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    slots = tuple(sample.answer_slots)
    prompt = (
        "Answer this video question directly from target-native transition "
        "grounding evidence. Each T pair contains adjacent BEFORE and AFTER "
        "frames selected outcome-blind by the wrapper's scene-change and "
        "uniform-anchor protocol. Use ordinary end-to-end visual reasoning. "
        "Do not execute a candidate-factorized proof or typed source program. "
        "Return probability mass for every option and concise visible evidence. "
        "No annotations, official structures, or gold are available.\n"
        + sample.format_question()
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, panel in enumerate(panels):
        content.extend((
            {
                "type": "text",
                "text": (
                    f"Transition grounding panel {index + 1}/{len(panels)}; "
                    "read each labeled T pair as BEFORE then AFTER:"
                ),
            },
            paired.media_helpers._image_content(panel),
        ))
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        attempt = list(content)
        if last_error:
            attempt.append({"type": "text", "text": "Schema error: " + last_error})
        payload, usage = transport._provider_json_call(
            client,
            model=str(model["id"]),
            system=(
                "Return JSON only: {answer:string,probabilities:{slot:number},"
                "observed_evidence:[string],unresolved_uncertainties:[string],"
                "reason:string}. answer must be the unique probability argmax."
            ),
            content=attempt,
            max_tokens=int(model["max_tokens"]),
        )
        try:
            return parse_primary_receipt(payload, slots), payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("active generic receipt schema retries exhausted: " + last_error)


def _active_proof_call(
    client: OpenAI,
    *,
    sample: Any,
    panels: Sequence[bytes],
    grounding_receipt: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    slots = tuple(sample.answer_slots)
    prompt = (
        "Execute the transferred candidate-factorized verification protocol over "
        "target-native transition grounding evidence. Independently evaluate every "
        "option. Each T pair is an adjacent BEFORE/AFTER observation selected "
        "outcome-blind by real wrapper tools. For each option emit exactly five "
        "typed steps in this order: "
        + ", ".join(PROOF_KINDS)
        + ". ENTITY_GROUNDING must bind visible target entities; EVENT_OCCURRENCE "
        "and TEMPORAL_ORDER must cite labeled T pairs. Mark SUPPORTED only from "
        "visible evidence, REFUTED only from visible contradiction, and otherwise "
        "UNKNOWN. CAUSAL_LINK must remain UNKNOWN unless a BEFORE/AFTER pair supports "
        "the claimed transition. No annotations, official structures, or gold are "
        "available.\n"
        + sample.format_question()
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, panel in enumerate(panels):
        content.extend((
            {
                "type": "text",
                "text": (
                    f"Transition grounding panel {index + 1}/{len(panels)}; "
                    "read each labeled T pair as BEFORE then AFTER:"
                ),
            },
            paired.media_helpers._image_content(panel),
        ))
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        attempt = list(content)
        if last_error:
            attempt.append({"type": "text", "text": "Schema error: " + last_error})
        payload, usage = transport._provider_json_call(
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
    raise ValueError("active proof receipt schema retries exhausted: " + last_error)


def _collect_one(
    input_row: Mapping[str, Any],
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
    _primary_panel, uniform_panels, uniform_metadata = paired._panels(sample, config)
    uniform_hashes = [hashlib.sha256(value).hexdigest() for value in uniform_panels]
    if uniform_hashes != list(input_row["proof_panel_sha256"]):
        raise ValueError("V26 did not reconstruct the exact V25 uniform panels")
    panels, metadata, grounding_receipt = _active_panels(sample, config=config)
    if metadata["proxy_sample_seconds"] != uniform_metadata["proxy_sample_seconds"]:
        raise ValueError("V26 transition evidence did not use the V25 proxy frames")
    panel_hashes = [hashlib.sha256(value).hexdigest() for value in panels]
    direct, direct_raw, direct_usage = _active_generic_call(
        client,
        sample=sample,
        panels=panels,
        grounding_receipt=grounding_receipt,
        config=config,
    )
    proof, proof_raw, proof_usage = _active_proof_call(
        client,
        sample=sample,
        panels=panels,
        grounding_receipt=grounding_receipt,
        config=config,
    )
    direct_answer = str(direct["answer"])
    source_answer = execute_recovery(direct_answer, proof)
    binding_answer = execute_recovery(
        direct_answer, proof, shuffled_binding=True,
    )
    topology_answer = execute_recovery(
        direct_answer, proof, shuffled_topology=True,
    )
    # Gold is attached only after the target-native tool calls, both neural
    # branches, the source executor, and destructive controls are immutable.
    gold = str(input_row["gold_answer"])
    return {
        "schema_version": 26,
        "benchmark": "star",
        "split": "consumed_v24_transition_grounding_development",
        "sample_id": str(input_row["sample_id"]),
        "video_id": str(input_row["video_id"]),
        "family": "Interaction",
        "gold_answer": gold,
        "uniform_v25_direct_correct": bool(input_row["direct_correct"]),
        "uniform_v25_source_correct": bool(input_row["source_authentic_correct"]),
        "active_generic_direct": direct,
        "active_typed_proof": proof,
        "active_source_answer": source_answer,
        "active_binding_control_answer": binding_answer,
        "active_topology_control_answer": topology_answer,
        "active_generic_correct": direct_answer == gold,
        "active_typed_proof_correct": str(proof["answer"]) == gold,
        "active_source_correct": source_answer == gold,
        "active_binding_control_correct": binding_answer == gold,
        "active_topology_control_correct": topology_answer == gold,
        "active_source_recover": source_answer != direct_answer,
        "active_binding_control_recover": binding_answer != direct_answer,
        "active_topology_control_recover": topology_answer != direct_answer,
        "direct_raw": direct_raw,
        "proof_raw": proof_raw,
        "usage": {"active_generic": direct_usage, "active_typed_proof": proof_usage},
        "video_metadata": metadata,
        "video_sha256": str(input_row["video_sha256"]),
        "uniform_panel_sha256": uniform_hashes,
        "transition_panel_sha256": panel_hashes,
        "active_direct_and_proof_panels_identical": True,
        "transition_grounding_receipt": grounding_receipt,
        "input_v25_row_sha256": _content_hash(input_row),
        "collection_contract_sha256": contract_sha256,
        "runtime_saw_gold_or_official_structure": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lineage_paths = {
        "source_receipt_sha256": Path(config["source_receipt"]),
        "input_v25_receipts_sha256": Path(config["input_v25_receipts"]),
        "collector_sha256": Path(__file__).resolve(),
        "v24_collector_sha256": REPO / "scripts/collect_star_interaction_v24_fresh.py",
        "paired_collector_sha256": REPO / "scripts/collect_natural_video_recovery_v15.py",
        "transport_module_sha256": REPO / "scripts/collect_natural_video_v19_formal.py",
        "symbolic_executor_sha256": REPO / "src/motif_transfer/natural_video_symbolic_controls.py",
        "wrapper_bridge_sha256": REPO / "src/motif_transfer/visual_wrapper_bridge.py",
        "wrapper_tools_video_sha256": (
            Path(config["wrapper_root"]) / "visual_reasoning_wrapper/tools_video.py"
        ),
    }
    for key, path in lineage_paths.items():
        if _sha256(path) != config["frozen_lineage"].get(key):
            raise ValueError(f"V26 transition-grounding lineage mismatch: {key}")
    validate_source_receipt(json.loads(
        Path(config["source_receipt"]).read_text(encoding="utf-8")
    ))
    input_rows = json.loads(
        Path(config["input_v25_receipts"]).read_text(encoding="utf-8")
    )
    if len(input_rows) != int(config["expected_rows"]):
        raise ValueError("V26 requires the complete consumed V25 rows")
    ordered_ids = [str(row["sample_id"]) for row in input_rows]
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("V26 input identities are not unique")
    input_by_id = {str(row["sample_id"]): row for row in input_rows}
    samples = v24._load_samples(ordered_ids, config)
    contract_sha256 = _content_hash({
        "config_sha256": _sha256(args.config),
        "collector_sha256": _sha256(Path(__file__).resolve()),
        "ordered_ids": ordered_ids,
    })
    key_values = runpy.run_path(str(args.keys))
    api_key = key_values.get(config["model"]["api_key_name"])
    if not api_key:
        raise SystemExit("V26 OpenRouter key is missing")
    existing = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached V26 contract mismatch")
            existing[str(row["sample_id"])] = row
    pending = [sample_id for sample_id in ordered_ids if sample_id not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(
            [existing[sample_id] for sample_id in ordered_ids if sample_id in existing],
            ensure_ascii=False,
            indent=2,
        ) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_one,
                input_by_id[sample_id],
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
            save()
            print(json.dumps({
                "completed": sample_id,
                "progress": f"{len(existing)}/{len(ordered_ids)}",
            }), flush=True)
    missing = [sample_id for sample_id in ordered_ids if sample_id not in existing]
    if missing:
        raise SystemExit(f"incomplete V26 transition-grounding collection; rerun: {missing}")
    rows = [existing[sample_id] for sample_id in ordered_ids]
    print(json.dumps({
        "status": "STAR_INTERACTION_TRANSITION_GROUNDING_V26_COLLECTED",
        "rows": len(rows),
        "video_clusters": len({row["video_id"] for row in rows}),
        "uniform_v25_direct_correct": sum(row["uniform_v25_direct_correct"] for row in rows),
        "active_generic_correct": sum(row["active_generic_correct"] for row in rows),
        "active_typed_proof_correct": sum(row["active_typed_proof_correct"] for row in rows),
        "active_source_correct": sum(row["active_source_correct"] for row in rows),
        "active_binding_control_correct": sum(row["active_binding_control_correct"] for row in rows),
        "active_topology_control_correct": sum(row["active_topology_control_correct"] for row in rows),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
