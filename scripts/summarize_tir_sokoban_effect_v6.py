#!/usr/bin/env python3
"""Evaluate corrected Sokoban POSITION/COMMIT semantics on TIR receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.tir_sokoban_effect_harness import (  # noqa: E402
    evaluate_tir_effect_transfer,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path = (REPO / config["source_receipt"]).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    receipts = []
    receipt_hashes = {}
    for relative in config["target_receipts"]:
        path = (REPO / relative).resolve()
        receipt_hashes[str(relative)] = _sha256(path)
        receipts.extend(json.loads(path.read_text(encoding="utf-8")))
    index = {str(row["sample_id"]): row for row in receipts}
    expected = tuple(map(str, config["expected_ids"]))
    missing = [sample_id for sample_id in expected if sample_id not in index]
    if missing:
        raise SystemExit(f"missing frozen TIR receipts: {missing}")
    ordered = [index[sample_id] for sample_id in expected]
    report = evaluate_tir_effect_transfer(
        ordered,
        source_receipt=source,
        expected_ids=expected,
        claim_boundary=str(config["claim_boundary"]),
        evidence_tier=str(config["evidence_tier"]),
    )
    body = dict(report)
    body.pop("report_sha256")
    body["integrity"] = {
        "config_file_sha256": _sha256(config_path),
        "source_receipt_file_sha256": _sha256(source_path),
        "target_receipt_file_sha256": receipt_hashes,
    }
    from motif_transfer.contracts import stable_hash
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": body["status"],
        "summaries": body["summaries"],
        "paired": body["paired"],
        "gates": body["gates"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0 if all(body["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
