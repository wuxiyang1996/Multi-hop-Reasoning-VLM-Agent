#!/usr/bin/env python3
"""Freeze V5c source structural programs, reserve seeds, code, and gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_delta_induction import (  # noqa: E402
    validate_structural_program,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-report", type=Path,
        default=REPO / "runs/source_structural_v5b_development/report.json",
    )
    parser.add_argument(
        "--reserve-config", type=Path,
        default=REPO / "configs/source_structural_v5c_fresh.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "configs/source_structural_v5c_frozen",
    )
    parser.add_argument(
        "--expected-summary", type=Path,
        default=REPO / "runs/source_structural_v5c_fresh/summary.json",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output: {args.output_dir}")
    if args.expected_summary.exists():
        raise SystemExit("fresh reserve summary already exists before freeze")

    report = _read(args.development_report)
    _self_hash(report, "report_sha256")
    if report.get("status") != "SOURCE_STRUCTURAL_DEVELOPMENT_PASSED":
        raise SystemExit("development structural induction did not pass")
    if not all(report.get("gates", {}).values()):
        raise SystemExit("development report contains a failed gate")
    config = _read(args.reserve_config)
    if config.get("schema_version") != "source-structural-fresh-config-v5c":
        raise SystemExit("unexpected reserve config schema")

    historical = {21, 22, 23, 24}
    allocated: set[tuple[str, int]] = set()
    for row in config.get("tasks") or ():
        if row.get("required_effects"):
            raise SystemExit("reserve config must not name required effects")
        for seed in row.get("seeds") or ():
            value = int(seed["seed"])
            if value in historical:
                raise SystemExit("fresh reserve seed overlaps consumed development")
            if seed.get("split") != "qualification":
                raise SystemExit("fresh reserve must be qualification-only")
            key = (str(row["task_id"]), value)
            if key in allocated:
                raise SystemExit(f"duplicate reserve seed: {key}")
            allocated.add(key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    program_dir = args.output_dir / "programs"
    program_dir.mkdir(parents=True, exist_ok=True)
    program_receipts = {}
    for lineage in report["lineages"]:
        task = str(lineage["task_id"])
        program = dict(lineage["program"])
        validate_structural_program(program)
        path = program_dir / f"{task}.json"
        path.write_text(
            json.dumps(program, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        program_receipts[task] = {
            "path": str(path.relative_to(REPO)),
            "file_sha256": _sha(path),
            "program_sha256": program["program_sha256"],
        }

    code_paths = (
        "src/motif_transfer/contracts.py",
        "src/motif_transfer/structural_delta_induction.py",
        "src/motif_transfer/typed_source_tasks.py",
        "scripts/collect_source_structural_paths_v5.py",
        "scripts/evaluate_source_structural_v5c.py",
    )
    frozen_code = {relative: _sha(REPO / relative) for relative in code_paths}
    body = {
        "schema_version": "source-structural-reserve-manifest-v5c",
        "development_report_path": str(args.development_report.relative_to(REPO)),
        "development_report_file_sha256": _sha(args.development_report),
        "development_report_sha256": report["report_sha256"],
        "source_programs": program_receipts,
        "frozen_source_program_permutation": report[
            "source_program_permutation"
        ],
        "reserve_config_path": str(args.reserve_config.relative_to(REPO)),
        "reserve_config_file_sha256": _sha(args.reserve_config),
        "reserve_config": config,
        "expected_summary_path": str(args.expected_summary.relative_to(REPO)),
        "expected_summary_file_sha256": "SET_AFTER_COLLECTION_BY_SEAL_SCRIPT",
        "reserve_opened_at_freeze": False,
        "fresh_seed_pairs": [list(row) for row in sorted(allocated)],
        "all_fresh_seeds_disjoint": True,
        "frozen_code_sha256": frozen_code,
        "preregistered_thresholds": {
            "minimum_authentic_sequence_support": 0.90,
            "minimum_source_permutation_gap": 0.30,
            "minimum_correct_program_selection_rate": 0.80,
            "minimum_authentic_binding_accuracy": 0.90,
            "minimum_shuffled_binding_gap": 0.50
        },
        "claim_boundary": "FROZEN_BEFORE_FRESH_SOURCE_COLLECTION;NO_TARGET_DATA_OR_OUTCOMES_READ",
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    path = args.output_dir / "manifest.unsealed.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(path), "manifest_sha256": manifest["manifest_sha256"],
        "fresh_seed_pairs": len(allocated), "reserve_opened_at_freeze": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
