#!/usr/bin/env python3
"""Merge complete candidate-video shards in frozen base-receipt order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def merge_shards(
    *, expected_ids: Sequence[str], shards: Sequence[Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    lineage: tuple[str, str, str] | None = None
    for shard in shards:
        for row in shard:
            sample_id = str(row["sample_id"])
            if not bool(row.get("complete")):
                raise ValueError(f"incomplete video fork: {sample_id}")
            current = (
                str(row.get("source_gate_sha256") or ""),
                str(row.get("collector_sha256") or ""),
                str(row.get("config_sha256") or ""),
            )
            if not all(current):
                raise ValueError(f"missing lineage hashes: {sample_id}")
            if lineage is None:
                lineage = current
            elif current != lineage:
                raise ValueError("video shard lineage hashes do not match")
            if sample_id in indexed:
                raise ValueError(f"duplicate video fork: {sample_id}")
            indexed[sample_id] = row
    expected = list(map(str, expected_ids))
    missing = [sample_id for sample_id in expected if sample_id not in indexed]
    extra = [sample_id for sample_id in indexed if sample_id not in set(expected)]
    if missing or extra:
        raise ValueError(f"video shard coverage mismatch; missing={missing}, extra={extra}")
    return [indexed[sample_id] for sample_id in expected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-receipts", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = [
        str(row["sample_id"])
        for row in json.loads(args.base_receipts.read_text(encoding="utf-8"))
    ]
    shards = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.shards
    ]
    merged = merge_shards(expected_ids=expected, shards=shards)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"samples": len(merged), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
