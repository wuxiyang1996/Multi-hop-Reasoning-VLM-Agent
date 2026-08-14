#!/usr/bin/env python3
"""Collect frozen TIR receipts and run the corrected Sokoban effect program."""

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

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.tir_sokoban_effect_harness import (  # noqa: E402
    evaluate_tir_effect_transfer,
    validate_source_receipt,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_collector():
    path = REPO / "scripts/run_active_tir_wrapper_transfer.py"
    spec = importlib.util.spec_from_file_location("tir_v7_collector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen TIR collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_self_hash(payload: dict[str, Any], field: str) -> None:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise SystemExit(f"invalid frozen dependency {field}")


def _validate_integrity(config: dict[str, Any]) -> None:
    for relative, expected in config["integrity"]["file_sha256"].items():
        path = (REPO / relative).resolve()
        if _sha256(path) != str(expected):
            raise SystemExit(f"frozen dependency changed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("qualification", "heldout"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_FRESH_QUALIFICATION":
        raise SystemExit("TIR V7 config is not frozen")
    _validate_integrity(config)

    development_path = (REPO / config["development_report"]).resolve()
    development = json.loads(development_path.read_text(encoding="utf-8"))
    _validate_self_hash(development, "report_sha256")
    if development.get("status") != "DEVELOPMENT_GATE_PASSED_FREEZE_QUALIFICATION":
        raise SystemExit("consumed development gate did not authorize V7")
    source_path = (REPO / config["source_receipt"]).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_source_receipt(source)

    if args.split == "heldout":
        qualification_path = (
            REPO / config["qualification_authority"]["report_path"]
        ).resolve()
        if not qualification_path.is_file():
            raise SystemExit("heldout is locked until qualification report exists")
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        _validate_self_hash(qualification, "report_sha256")
        if qualification.get("status") != "QUALIFICATION_GATE_PASSED_FREEZE_HELDOUT":
            raise SystemExit("heldout is locked because qualification failed")
        if _sha256(qualification_path) != config["qualification_authority"].get(
            "report_file_sha256_after_pass", _sha256(qualification_path)
        ):
            raise SystemExit("qualification report hash differs from formal authority")

    collector = _load_collector()
    collection_contract = collector._collection_contract(config)
    dataset_file = args.dataset_root / "TIR-Bench.json"
    if _sha256(dataset_file) != config["dataset"]["file_sha256"]:
        raise SystemExit("TIR dataset file hash mismatch")
    rows = json.loads(dataset_file.read_text(encoding="utf-8"))
    index = {str(row["id"]): row for row in rows}
    sample_ids = list(map(str, config["splits"][args.split]))
    missing = [sample_id for sample_id in sample_ids if sample_id not in index]
    if missing:
        raise SystemExit(f"frozen IDs missing from TIR dataset: {missing}")
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = args.output_dir / f"{args.split}_receipts.json"
    existing: dict[str, Any] = {}
    if receipts_path.is_file():
        existing = {
            str(row["sample_id"]): row
            for row in json.loads(receipts_path.read_text(encoding="utf-8"))
        }
        bad = [
            sample_id for sample_id, row in existing.items()
            if row.get("collection_contract_sha256") != collection_contract
        ]
        if bad:
            raise SystemExit(f"resumed receipt contract mismatch: {bad}")
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
        raise SystemExit(f"incomplete {args.split} receipts; rerun: {missing}")

    evidence_tier = (
        "FRESH_QUALIFICATION"
        if args.split == "qualification"
        else "FRESH_FORMAL_CONFIRMATION"
    )
    report = evaluate_tir_effect_transfer(
        [existing[sample_id] for sample_id in sample_ids],
        source_receipt=source,
        expected_ids=sample_ids,
        claim_boundary=str(config["claim_boundary"][args.split]),
        evidence_tier=evidence_tier,
    )
    body = dict(report)
    body.pop("report_sha256")
    body["integrity"] = {
        "config_file_sha256": _sha256(config_path),
        "collection_contract_sha256": collection_contract,
        "source_receipt_file_sha256": _sha256(source_path),
        "development_report_file_sha256": _sha256(development_path),
        "receipts_file_sha256": _sha256(receipts_path),
    }
    body["formal_heldout_consumed"] = args.split == "heldout"
    body["report_sha256"] = stable_hash(body)
    report_path = args.output_dir / f"{args.split}_report.json"
    report_path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": body["status"],
        "summaries": body["summaries"],
        "paired": body["paired"],
        "gates": body["gates"],
        "report": str(report_path.resolve()),
    }, indent=2), flush=True)
    return 0 if all(body["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
