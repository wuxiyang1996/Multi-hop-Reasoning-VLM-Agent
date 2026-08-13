#!/usr/bin/env python3
"""Merge disjoint candidate-claim checkpoints in frozen receipt order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt_ids = [
        str(row["sample_id"])
        for row in json.loads((args.run_dir / "receipts.json").read_text(encoding="utf-8"))
    ]
    merged = {}
    for name in args.shards:
        rows = json.loads((args.run_dir / name).read_text(encoding="utf-8"))
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in merged and row != merged[sample_id]:
                raise ValueError(f"conflicting duplicate shard row: {sample_id}")
            merged[sample_id] = row
    missing = [sample_id for sample_id in receipt_ids if sample_id not in merged]
    extra = sorted(set(merged) - set(receipt_ids))
    if missing or extra:
        raise ValueError(f"incomplete shard merge; missing={missing}, extra={extra}")
    ordered = [merged[sample_id] for sample_id in receipt_ids]
    if not all(bool(row.get("complete")) for row in ordered):
        raise ValueError("one or more merged fork rows are incomplete")
    path = args.run_dir / args.output
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()
