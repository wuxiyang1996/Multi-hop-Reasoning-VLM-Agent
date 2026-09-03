#!/usr/bin/env python3
"""Collect a same-model 2x2 uniform/active x direct/source STAR factorial."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_natural_video_generic_control_v21 as generic  # noqa: E402
import collect_natural_video_recovery_v15 as paired  # noqa: E402
import collect_natural_video_v19_formal as transport  # noqa: E402
import collect_star_interaction_transition_grounding_v26 as active  # noqa: E402
import collect_star_interaction_v24_fresh as v24  # noqa: E402
from motif_transfer.natural_video_symbolic_controls import execute_recovery  # noqa: E402
from motif_transfer.sokoban_video_recovery import validate_source_receipt  # noqa: E402


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


def _branches(direct: Mapping[str, Any], proof: Mapping[str, Any]) -> dict[str, Any]:
    commitment = str(direct["answer"])
    source = execute_recovery(commitment, proof)
    binding = execute_recovery(commitment, proof, shuffled_binding=True)
    topology = execute_recovery(commitment, proof, shuffled_topology=True)
    return {
        "source_answer": source,
        "binding_control_answer": binding,
        "topology_control_answer": topology,
        "source_recover": source != commitment,
        "binding_control_recover": binding != commitment,
        "topology_control_recover": topology != commitment,
    }


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
        raise ValueError("V27 did not reconstruct the exact V25 uniform panels")
    active_panels, active_metadata, grounding_receipt = active._active_panels(
        sample, config=config,
    )
    if active_metadata["proxy_sample_seconds"] != uniform_metadata["proxy_sample_seconds"]:
        raise ValueError("V27 active and uniform branches did not use the same proxy frames")
    active_hashes = [hashlib.sha256(value).hexdigest() for value in active_panels]

    uniform_direct, uniform_direct_raw, uniform_direct_usage = generic._generic_call(
        client, sample=sample, panels=uniform_panels, config=config,
    )
    uniform_proof, uniform_proof_raw, uniform_proof_usage = paired._proof_call(
        client, sample=sample, panels=uniform_panels, config=config,
    )
    active_direct, active_direct_raw, active_direct_usage = active._active_generic_call(
        client,
        sample=sample,
        panels=active_panels,
        grounding_receipt=grounding_receipt,
        config=config,
    )
    active_proof, active_proof_raw, active_proof_usage = active._active_proof_call(
        client,
        sample=sample,
        panels=active_panels,
        grounding_receipt=grounding_receipt,
        config=config,
    )
    uniform = _branches(uniform_direct, uniform_proof)
    grounded = _branches(active_direct, active_proof)

    # Outcome access begins only after both evidence views, all four neural
    # calls, both source executions, and all four destructive controls freeze.
    gold = str(input_row["gold_answer"])
    return {
        "schema_version": 27,
        "benchmark": "star",
        "split": "consumed_v24_grounding_factorial_development",
        "sample_id": str(input_row["sample_id"]),
        "video_id": str(input_row["video_id"]),
        "family": "Interaction",
        "gold_answer": gold,
        "uniform_direct": uniform_direct,
        "uniform_typed_proof": uniform_proof,
        "uniform_source_answer": uniform["source_answer"],
        "uniform_binding_control_answer": uniform["binding_control_answer"],
        "uniform_topology_control_answer": uniform["topology_control_answer"],
        "active_direct": active_direct,
        "active_typed_proof": active_proof,
        "active_source_answer": grounded["source_answer"],
        "active_binding_control_answer": grounded["binding_control_answer"],
        "active_topology_control_answer": grounded["topology_control_answer"],
        "uniform_direct_correct": str(uniform_direct["answer"]) == gold,
        "uniform_typed_proof_correct": str(uniform_proof["answer"]) == gold,
        "uniform_source_correct": str(uniform["source_answer"]) == gold,
        "uniform_binding_control_correct": str(uniform["binding_control_answer"]) == gold,
        "uniform_topology_control_correct": str(uniform["topology_control_answer"]) == gold,
        "active_direct_correct": str(active_direct["answer"]) == gold,
        "active_typed_proof_correct": str(active_proof["answer"]) == gold,
        "active_source_correct": str(grounded["source_answer"]) == gold,
        "active_binding_control_correct": str(grounded["binding_control_answer"]) == gold,
        "active_topology_control_correct": str(grounded["topology_control_answer"]) == gold,
        "uniform_source_recover": bool(uniform["source_recover"]),
        "uniform_binding_control_recover": bool(uniform["binding_control_recover"]),
        "uniform_topology_control_recover": bool(uniform["topology_control_recover"]),
        "active_source_recover": bool(grounded["source_recover"]),
        "active_binding_control_recover": bool(grounded["binding_control_recover"]),
        "active_topology_control_recover": bool(grounded["topology_control_recover"]),
        "raw": {
            "uniform_direct": uniform_direct_raw,
            "uniform_typed_proof": uniform_proof_raw,
            "active_direct": active_direct_raw,
            "active_typed_proof": active_proof_raw,
        },
        "usage": {
            "uniform_direct": uniform_direct_usage,
            "uniform_typed_proof": uniform_proof_usage,
            "active_direct": active_direct_usage,
            "active_typed_proof": active_proof_usage,
        },
        "video_metadata": active_metadata,
        "video_sha256": str(input_row["video_sha256"]),
        "uniform_panel_sha256": uniform_hashes,
        "active_panel_sha256": active_hashes,
        "within_view_direct_and_proof_panels_identical": True,
        "uniform_and_active_use_same_proxy_frames": True,
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
        "active_collector_sha256": REPO / "scripts/collect_star_interaction_transition_grounding_v26.py",
        "v24_collector_sha256": REPO / "scripts/collect_star_interaction_v24_fresh.py",
        "paired_collector_sha256": REPO / "scripts/collect_natural_video_recovery_v15.py",
        "generic_collector_sha256": REPO / "scripts/collect_natural_video_generic_control_v21.py",
        "transport_module_sha256": REPO / "scripts/collect_natural_video_v19_formal.py",
        "symbolic_executor_sha256": REPO / "src/motif_transfer/natural_video_symbolic_controls.py",
        "wrapper_bridge_sha256": REPO / "src/motif_transfer/visual_wrapper_bridge.py",
        "wrapper_tools_video_sha256": (
            Path(config["wrapper_root"]) / "visual_reasoning_wrapper/tools_video.py"
        ),
    }
    for key, path in lineage_paths.items():
        if _sha256(path) != config["frozen_lineage"].get(key):
            raise ValueError(f"V27 grounding-factorial lineage mismatch: {key}")
    validate_source_receipt(json.loads(
        Path(config["source_receipt"]).read_text(encoding="utf-8")
    ))
    input_rows = json.loads(
        Path(config["input_v25_receipts"]).read_text(encoding="utf-8")
    )
    if len(input_rows) != int(config["expected_rows"]):
        raise ValueError("V27 requires the complete consumed V25 rows")
    ordered_ids = [str(row["sample_id"]) for row in input_rows]
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("V27 input identities are not unique")
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
        raise SystemExit("V27 OpenRouter key is missing")
    paired.media_helpers._json_call = transport._provider_json_call
    existing = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached V27 contract mismatch")
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
        raise SystemExit(f"incomplete V27 grounding factorial; rerun: {missing}")
    rows = [existing[sample_id] for sample_id in ordered_ids]
    print(json.dumps({
        "status": "STAR_INTERACTION_GROUNDING_FACTORIAL_V27_COLLECTED",
        "rows": len(rows),
        "video_clusters": len({row["video_id"] for row in rows}),
        **{
            field: sum(bool(row[field]) for row in rows)
            for field in (
                "uniform_direct_correct",
                "uniform_typed_proof_correct",
                "uniform_source_correct",
                "active_direct_correct",
                "active_typed_proof_correct",
                "active_source_correct",
                "active_binding_control_correct",
                "active_topology_control_correct",
            )
        },
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
