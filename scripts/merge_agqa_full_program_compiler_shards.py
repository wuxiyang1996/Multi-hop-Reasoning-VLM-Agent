#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("merged qualification is immutable")
    reports = sorted(
        (json.loads(path.read_text(encoding="utf-8")) for path in args.shards),
        key=lambda row: row["row_range"]["start"],
    )
    cursor = 0; total = reports[0]["row_range"]["validation_total"]
    sums = Counter(); roots: dict[str, Counter[str]] = {}
    for report in reports:
        interval = report["row_range"]
        if interval["start"] != cursor or interval["validation_total"] != total:
            raise ValueError("compiler shard ranges are not contiguous")
        cursor = interval["end"]
        rows = report["metrics"]["rows"]
        sums["rows"] += rows
        sums["syntax"] += round(report["metrics"]["syntax_valid_rate"] * rows)
        sums["admitted"] += round(report["metrics"]["source_admission_rate"] * rows)
        sums["exact"] += round(report["metrics"]["program_exact_rate"] * rows)
        for root, values in report["by_root"].items(): roots.setdefault(root, Counter()).update(values)
    if cursor != total: raise ValueError(f"compiler shards stop at {cursor}, expected {total}")
    metrics = {"rows": total, "syntax_valid": sums["syntax"],
               "source_admitted": sums["admitted"], "program_exact": sums["exact"],
               "syntax_valid_rate": sums["syntax"] / total,
               "source_admission_rate": sums["admitted"] / total,
               "program_exact_rate": sums["exact"] / total}
    passed = metrics["syntax_valid_rate"] >= .995 and metrics["source_admission_rate"] >= .995 and metrics["program_exact_rate"] >= .98
    body = {"schema_version": "agqa-full-program-compiler-heldout-eval-v1-merged",
            "status": "COMPILER_QUALIFIED" if passed else "COMPILER_NOT_QUALIFIED",
            "formal_programs_read": False, "formal_answers_read": False,
            "metrics": metrics, "by_root": {k: dict(v) for k, v in sorted(roots.items())},
            "shard_report_sha256s": [r["report_sha256"] for r in reports]}
    body["report_sha256"] = stable_hash(body)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": body["status"], "metrics": metrics}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__": raise SystemExit(main())
