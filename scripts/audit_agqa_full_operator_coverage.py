#!/usr/bin/env python3
"""Audit source capability coverage on target development programs only."""

from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import zipfile

from motif_transfer.agqa_typed_program import compile_receipt, parse_program
from motif_transfer.contracts import stable_hash
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    capabilities = json.loads(args.capabilities.read_text(encoding="utf-8"))
    validation = set(split["partitions"]["router_validation"])
    formal = set(split["partitions"]["formal_holdout"])
    counts: Counter[str] = Counter()
    roots: Counter[str] = Counter()
    required: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    formal_skipped = 0
    with zipfile.ZipFile(split["archive_path"]) as bundle, bundle.open(split["entry"]) as raw:
        for _, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            video = str(row["video_id"])
            if video in formal:
                formal_skipped += 1
                continue
            if video not in validation:
                continue
            program = str(row["program"])
            receipt = compile_receipt(
                program, capabilities["authorized_operators"],
                capabilities.get("authorized_compositions", ()),
            )
            counts[receipt["status"]] += 1
            ast = parse_program(program)
            roots[getattr(ast, "function", "ATOM")] += 1
            required.update(receipt["required_operators"])
            missing.update(receipt["missing_operators"])
    total = sum(counts.values())
    body = {
        "schema_version": "agqa-full-operator-development-coverage-v1",
        "status": "FULL_DEVELOPMENT_OPERATOR_COVERAGE" if counts["COMPILED"] == total else "OPERATOR_COVERAGE_GAP",
        "authority": "TARGET_DEVELOPMENT_FUNCTIONAL_PROGRAMS_ONLY",
        "formal_rows_skipped_before_program_access": formal_skipped,
        "formal_outcomes_read": False,
        "source_capability_artifact_sha256": capabilities["artifact_sha256"],
        "total_programs": total,
        "counts": dict(sorted(counts.items())),
        "coverage": counts["COMPILED"] / total if total else 0.0,
        "root_functions": dict(sorted(roots.items())),
        "required_operator_frequency": dict(sorted(required.items())),
        "missing_operator_frequency": dict(sorted(missing.items())),
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(body, indent=2, sort_keys=True))
    return 0 if body["status"] == "FULL_DEVELOPMENT_OPERATOR_COVERAGE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
