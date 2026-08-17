#!/usr/bin/env python3
"""Seal only the fresh collection receipt into an already frozen V5c manifest."""

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
    parser.add_argument("--unsealed-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    manifest = _read(args.unsealed_manifest)
    _self_hash(manifest, "manifest_sha256")
    if manifest["expected_summary_file_sha256"] != "SET_AFTER_COLLECTION_BY_SEAL_SCRIPT":
        raise SystemExit("input manifest was already sealed")
    summary = _read(args.summary)
    _self_hash(summary, "summary_sha256")
    if summary["config_file_sha256"] != manifest["reserve_config_file_sha256"]:
        raise SystemExit("collection did not use the frozen config")
    expected = {
        (str(row["task_id"]), int(seed["seed"]))
        for row in manifest["reserve_config"]["tasks"]
        for seed in row["seeds"]
    }
    observed = {
        (str(row["task_id"]), int(row["seed"]))
        for row in summary.get("receipts") or ()
    }
    if observed != expected or len(observed) != len(summary.get("receipts") or ()):
        raise SystemExit("fresh summary seed identities do not match the freeze")
    body = dict(manifest)
    body.pop("manifest_sha256")
    body["expected_summary_file_sha256"] = _sha(args.summary)
    body["collection_receipt_sealed_without_reading_outcomes"] = True
    sealed = body | {"manifest_sha256": stable_hash(body)}
    args.output.write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(args.output),
        "manifest_sha256": sealed["manifest_sha256"],
        "summary_file_sha256": sealed["expected_summary_file_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
