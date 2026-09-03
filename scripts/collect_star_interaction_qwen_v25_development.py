#!/usr/bin/env python3
"""Qualify same-Qwen direct versus typed source execution on consumed V24 rows."""

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
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


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
    _primary_panel, panels, metadata = paired._panels(sample, config)
    panel_hashes = [hashlib.sha256(value).hexdigest() for value in panels]
    if panel_hashes != list(input_row["proof_panel_sha256"]):
        raise ValueError("V25 did not reconstruct the exact V24 dense panels")
    direct, direct_raw, direct_usage = generic._generic_call(
        client, sample=sample, panels=panels, config=config,
    )
    proof, proof_raw, proof_usage = paired._proof_call(
        client, sample=sample, panels=panels, config=config,
    )
    direct_answer = str(direct["answer"])
    source_answer = execute_recovery(direct_answer, proof)
    binding_answer = execute_recovery(
        direct_answer, proof, shuffled_binding=True,
    )
    topology_answer = execute_recovery(
        direct_answer, proof, shuffled_topology=True,
    )
    gold = str(input_row["gold_answer"])
    return {
        "schema_version": 25,
        "benchmark": "star",
        "split": "consumed_v24_qwen_development",
        "sample_id": str(input_row["sample_id"]),
        "video_id": str(input_row["video_id"]),
        "family": "Interaction",
        "gold_answer": gold,
        "qwen_direct": direct,
        "qwen_typed_proof": proof,
        "source_authentic_answer": source_answer,
        "shuffled_binding_answer": binding_answer,
        "shuffled_topology_answer": topology_answer,
        "direct_correct": direct_answer == gold,
        "typed_proof_correct": str(proof["answer"]) == gold,
        "source_authentic_correct": source_answer == gold,
        "shuffled_binding_correct": binding_answer == gold,
        "shuffled_topology_correct": topology_answer == gold,
        "source_recover": source_answer != direct_answer,
        "binding_control_recover": binding_answer != direct_answer,
        "topology_control_recover": topology_answer != direct_answer,
        "direct_raw": direct_raw,
        "proof_raw": proof_raw,
        "usage": {"direct": direct_usage, "typed_proof": proof_usage},
        "video_metadata": metadata,
        "video_sha256": str(input_row["video_sha256"]),
        "proof_panel_sha256": panel_hashes,
        "direct_and_proof_panels_identical": True,
        "input_v24_row_sha256": _content_hash(input_row),
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
        "input_v24_receipts_sha256": Path(config["input_v24_receipts"]),
        "collector_sha256": Path(__file__).resolve(),
        "v24_collector_sha256": REPO / "scripts/collect_star_interaction_v24_fresh.py",
        "paired_collector_sha256": REPO / "scripts/collect_natural_video_recovery_v15.py",
        "generic_collector_sha256": REPO / "scripts/collect_natural_video_generic_control_v21.py",
        "transport_module_sha256": REPO / "scripts/collect_natural_video_v19_formal.py",
        "symbolic_executor_sha256": REPO / "src/motif_transfer/natural_video_symbolic_controls.py",
    }
    for key, path in lineage_paths.items():
        if _sha256(path) != config["frozen_lineage"].get(key):
            raise ValueError(f"V25 Qwen lineage mismatch: {key}")
    validate_source_receipt(json.loads(
        Path(config["source_receipt"]).read_text(encoding="utf-8")
    ))
    input_rows = json.loads(Path(config["input_v24_receipts"]).read_text(encoding="utf-8"))
    if len(input_rows) != int(config["expected_rows"]):
        raise ValueError("V25 requires the complete consumed V24 rows")
    ordered_ids = [str(row["sample_id"]) for row in input_rows]
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
        raise SystemExit("V25 OpenRouter key is missing")
    paired.media_helpers._json_call = transport._provider_json_call
    existing = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached V25 contract mismatch")
            existing[str(row["sample_id"])] = row
    pending = [sample_id for sample_id in ordered_ids if sample_id not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(
            [existing[sample_id] for sample_id in ordered_ids if sample_id in existing],
            ensure_ascii=False, indent=2,
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
                    "failed": sample_id, "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                continue
            save()
            print(json.dumps({
                "completed": sample_id, "progress": f"{len(existing)}/{len(ordered_ids)}",
            }), flush=True)
    missing = [sample_id for sample_id in ordered_ids if sample_id not in existing]
    if missing:
        raise SystemExit(f"incomplete V25 Qwen collection; rerun: {missing}")
    rows = [existing[sample_id] for sample_id in ordered_ids]
    print(json.dumps({
        "status": "STAR_INTERACTION_QWEN_V25_DEVELOPMENT_COLLECTED",
        "rows": len(rows),
        "video_clusters": len({row["video_id"] for row in rows}),
        "direct_correct": sum(row["direct_correct"] for row in rows),
        "typed_proof_correct": sum(row["typed_proof_correct"] for row in rows),
        "source_authentic_correct": sum(row["source_authentic_correct"] for row in rows),
        "binding_control_correct": sum(row["shuffled_binding_correct"] for row in rows),
        "topology_control_correct": sum(row["shuffled_topology_correct"] for row in rows),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
