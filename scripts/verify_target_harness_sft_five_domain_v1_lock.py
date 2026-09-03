#!/usr/bin/env python3
"""Verify that the frozen five-domain V1 engineering baseline is unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "configs/target_harness_sft_five_domain_v1_baseline_lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    checks = {}
    for name, spec in sorted(lock["artifacts"].items()):
        path = Path(spec["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        actual = _sha256(path) if path.is_file() else None
        checks[name] = {
            "path": str(path.resolve()),
            "expected_sha256": spec["sha256"],
            "actual_sha256": actual,
            "passed": actual == spec["sha256"],
        }
    passed = all(value["passed"] for value in checks.values())
    print(json.dumps({
        "schema_version": "target-harness-sft-baseline-lock-audit-v1",
        "status": "FROZEN_BASELINE_UNCHANGED" if passed else "FROZEN_BASELINE_DRIFTED",
        "baseline_id": lock["baseline_id"],
        "checks": checks,
    }, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
