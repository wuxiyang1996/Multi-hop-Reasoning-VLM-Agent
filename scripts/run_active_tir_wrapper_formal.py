#!/usr/bin/env python3
"""Collect unopened TIR forks and evaluate a frozen adaptation artifact once."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.active_video_transfer import stable_hash  # noqa: E402
from motif_transfer.candidate_transfer_formal import (  # noqa: E402
    evaluate_frozen_candidate_transfer,
    validate_frozen_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_collector():
    path = REPO / "scripts/run_active_tir_wrapper_transfer.py"
    spec = importlib.util.spec_from_file_location("tir_adaptation_collector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen TIR collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_integrity(config: dict[str, Any]) -> None:
    for relative, expected in config["integrity"]["file_sha256"].items():
        path = (REPO / relative).resolve()
        if _sha256(path) != str(expected):
            raise SystemExit(f"formal frozen dependency changed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_QUALIFICATION_COLLECTION":
        raise SystemExit("TIR formal config is not frozen")
    _validate_integrity(config)
    artifact_path = (REPO / config["artifact"]["path"]).resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if _sha256(artifact_path) != config["artifact"]["file_sha256"]:
        raise SystemExit("TIR artifact file hash mismatch")
    controlled_path = (REPO / config["source"]["controlled_config"]).resolve()
    controlled = json.loads(controlled_path.read_text(encoding="utf-8"))
    validate_frozen_artifact(artifact, controlled_config=controlled)

    collector = _load_collector()
    collection_contract = collector._collection_contract(config)
    rows = json.loads(
        (args.dataset_root / "TIR-Bench.json").read_text(encoding="utf-8")
    )
    index = {str(row["id"]): row for row in rows}
    sample_ids = list(map(str, config["splits"]["qualification"]))
    missing = [sample_id for sample_id in sample_ids if sample_id not in index]
    if missing:
        raise SystemExit(f"frozen qualification IDs are missing: {missing}")
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = args.output_dir / "qualification_receipts.json"
    existing: dict[str, Any] = {}
    if receipts_path.is_file():
        existing = {
            str(row["sample_id"]): row
            for row in json.loads(receipts_path.read_text(encoding="utf-8"))
        }
        invalid = [
            sample_id for sample_id, row in existing.items()
            if row.get("collection_contract_sha256") != collection_contract
        ]
        if invalid:
            raise SystemExit(f"resumed receipt contract mismatch: {invalid}")

    pending = [sample_id for sample_id in sample_ids if sample_id not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                collector._collect_sample,
                sample_id,
                row=index[sample_id],
                dataset_root=args.dataset_root,
                config=config,
                api_key=str(key),
                contract_sha256=collection_contract,
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
        raise SystemExit(f"incomplete qualification receipts; rerun: {missing}")

    receipts = [existing[sample_id] for sample_id in sample_ids]
    report = evaluate_frozen_candidate_transfer(
        receipts,
        config=config,
        artifact=artifact,
        controlled_config=controlled,
    )
    report_body = dict(report)
    report_body.pop("report_sha256", None)
    report_body.update({
        "config_path": str(config_path),
        "config_file_sha256": _sha256(config_path),
        "collection_contract_sha256": collection_contract,
        "qualification_receipts_file_sha256": _sha256(receipts_path),
        "qualification_receipt_matrix_sha256": stable_hash(receipts),
        "formal_held_out_consumed": False,
    })
    report_body["report_sha256"] = stable_hash(report_body)
    report_path = args.output_dir / "qualification_report.json"
    report_path.write_text(
        json.dumps(report_body, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report_body["status"],
        "conditions": report_body["conditions"],
        "paired_comparisons": report_body["paired_comparisons"],
        "gates": report_body["gates"],
        "report": str(report_path.resolve()),
    }, indent=2), flush=True)
    return 0 if report_body["status"] == "FORMAL_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
