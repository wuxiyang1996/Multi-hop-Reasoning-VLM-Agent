#!/usr/bin/env python3
"""Repair only the null manifest stable-hash receipt in a V23 enumeration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_hash(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise ValueError(f"invalid stable hash: {field}")
    return claimed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite repaired V23 receipt: {args.output}")
    report = _read(args.input)
    original_report_hash = _validate_hash(report, "report_sha256")
    manifest = _read(args.manifest)
    manifest_hash = _validate_hash(manifest, "manifest_sha256")
    if report.get("outcomes_recorded") or report.get("rewards_recorded"):
        raise SystemExit("cannot repair an outcome-bearing enumeration")
    if report["manifest"].get("manifest_sha256") is not None:
        raise SystemExit("V23 enumeration manifest receipt is not the known null bug")
    if report["manifest"]["file_sha256"] != _sha256(args.manifest):
        raise SystemExit("V23 enumeration manifest file receipt mismatch")
    opportunities_hash = stable_hash(report["opportunities"])
    tasks_hash = stable_hash(report["tasks"])
    body = dict(report)
    body.pop("report_sha256")
    body["manifest"] = dict(body["manifest"])
    body["manifest"]["manifest_sha256"] = manifest_hash
    body["receipt_repair"] = {
        "authority": "NULL_RETURN_FROM_VALIDATING_V20_HELPER_ONLY",
        "original_path": str(args.input.resolve()),
        "original_file_sha256": _sha256(args.input),
        "original_report_sha256": original_report_hash,
        "field_changed": "manifest.manifest_sha256",
        "replacement": manifest_hash,
        "tasks_sha256_unchanged": tasks_hash,
        "opportunities_sha256_unchanged": opportunities_hash,
        "actions_scores_admissions_outcomes_changed": False,
    }
    repaired = body | {"report_sha256": stable_hash(body)}
    if stable_hash(repaired["tasks"]) != tasks_hash:
        raise RuntimeError("V23 repair changed tasks")
    if stable_hash(repaired["opportunities"]) != opportunities_hash:
        raise RuntimeError("V23 repair changed opportunities")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(repaired, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "report_sha256": repaired["report_sha256"],
        "manifest_sha256": manifest_hash,
        "opportunities_sha256": opportunities_hash,
        "task_count": repaired["task_count"],
        "opportunity_count": repaired["opportunity_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
