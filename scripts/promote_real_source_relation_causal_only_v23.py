#!/usr/bin/env python3
"""Promote V23 to sealed confirmation only after its development gate passes."""

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
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V23 promotion: {args.output}")
    candidate = _read(args.candidate)
    candidate_hash = _validate_hash(candidate, "candidate_sha256")
    report = _read(args.development_report)
    report_hash = _validate_hash(report, "report_sha256")
    if candidate.get("status") != "CAUSAL_ONLY_DEVELOPMENT_GATE_AUTHORIZED":
        raise SystemExit("unexpected pre-development V23 candidate")
    if report.get("status") != (
        "V23_DEVELOPMENT_TRANSFER_GATE_PASSED_CONFIRMATION_AUTHORIZED"
    ) or not report.get("all_gates_passed"):
        raise SystemExit("V23 development gate did not pass")
    plan = _read(Path(str(report["plan"]["path"])))
    _validate_hash(plan, "plan_sha256")
    if plan["candidate"]["candidate_sha256"] != candidate_hash:
        raise SystemExit("V23 development report used another candidate")
    body = dict(candidate)
    body.pop("candidate_sha256")
    body.update({
        "status": "V23_DEVELOPMENT_TRANSFER_GATE_PASSED_CONFIRMATION_AUTHORIZED",
        "claim_boundary": (
            "V23_CAUSAL_ONLY_POLICY_PASSED_FROZEN_FULL_DEVELOPMENT_SPLIT; "
            "SEALED_CONFIRMATION_AUTHORIZED_BUT_UNREAD; VALID_UNSEEN_UNREAD"
        ),
        "development_report": {
            "path": str(args.development_report.resolve()),
            "file_sha256": _sha256(args.development_report),
            "report_sha256": report_hash,
            "primary_metrics": report["policy_metrics"]["v23_causal_only"],
            "gates": report["gates"],
        },
        "development_authorized": False,
        "confirmation_authorized": True,
        "development_read_or_run": True,
        "confirmation_read_or_run": False,
    })
    promoted = body | {"candidate_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(promoted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_sha256": promoted["candidate_sha256"],
        "status": promoted["status"],
        "confirmation_authorized": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
