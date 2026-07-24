#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.matched_diagnosis import diagnose_matched_pair


def _row(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row: {path}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO / "runs/target_feasibility_v1")
    parser.add_argument("--output", type=Path, default=REPO / "docs/results/alfworld_target_feasibility_v1.json")
    args = parser.parse_args()
    files = {
        "alfworld_valid_seen": (
            "alfworld_valid_seen_target_only_cache_v2.json",
            "alfworld_valid_seen_authentic_cache_v2.json",
        ),
        "alfworld_valid_unseen": (
            "alfworld_valid_unseen_target_only_cache_v2.json",
            "alfworld_valid_unseen_authentic_cache_v2.json",
        ),
    }
    cells = {}
    for cell, (baseline_name, treatment_name) in files.items():
        baseline_path, treatment_path = args.root / baseline_name, args.root / treatment_name
        cells[cell] = diagnose_matched_pair(_row(baseline_path), _row(treatment_path))
        cells[cell]["baseline_artifact"] = str(baseline_path)
        cells[cell]["treatment_artifact"] = str(treatment_path)
    payload = {
        "schema_version": 1,
        "claim_limit": "ALFWORLD_MATCHED_MECHANISM_DIAGNOSIS_ONLY",
        "source_status": "GENERIC_ONLY_NOT_SOURCE_SUPPORTED",
        "cells": cells,
        "positive_transfer_cells": sum(row["status"] == "POSITIVE_TRANSFER_PILOT" for row in cells.values()),
        "negative_transfer_cells": sum(row["status"] == "NEGATIVE_TRANSFER_PILOT" for row in cells.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
