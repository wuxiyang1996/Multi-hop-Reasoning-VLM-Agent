#!/usr/bin/env python3
"""Authorize sealed V20 confirmation only after the frozen development gate."""

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
        raise SystemExit(f"refusing to overwrite V20 promotion: {args.output}")
    candidate = _read(args.candidate)
    candidate_hash = _validate_hash(candidate, "candidate_sha256")
    development = _read(args.development_report)
    development_hash = _validate_hash(development, "report_sha256")
    if candidate.get("status") != "TARGET_CAUSAL_AND_UTILITY_GATE_PASSED":
        raise SystemExit("V20 causal/utility candidate did not pass")
    if development.get("status") != (
        "DEVELOPMENT_TRANSFER_GATE_PASSED_CONFIRMATION_AUTHORIZED"
    ) or not development.get("all_gates_passed"):
        raise SystemExit("V20 frozen development gate did not pass")
    plan_path = Path(str(development["plan"]["path"]))
    plan = _read(plan_path)
    _validate_hash(plan, "plan_sha256")
    if plan["candidate"]["candidate_sha256"] != candidate_hash:
        raise SystemExit("V20 development used another candidate")
    body = {
        "schema_version": "real-source-relation-confirmation-candidate-v20",
        "status": "DEVELOPMENT_TRANSFER_GATE_PASSED_CONFIRMATION_AUTHORIZED",
        "claim_boundary": (
            "TARGET_CAUSAL_AND_UTILITY_ADAPTATION_CALIBRATION_PASSED; "
            "DISJOINT_FROZEN_DEVELOPMENT_TRANSFER_GATE_PASSED; SEALED_"
            "CONFIRMATION_OUTCOMES_UNREAD; EXISTING_VALID_UNSEEN_UNREAD"
        ),
        "base_candidate": {
            "path": str(args.candidate.resolve()),
            "file_sha256": _sha256(args.candidate),
            "candidate_sha256": candidate_hash,
        },
        "development_report": {
            "path": str(args.development_report.resolve()),
            "file_sha256": _sha256(args.development_report),
            "report_sha256": development_hash,
        },
        "parent_candidate": candidate["parent_candidate"],
        "source_summary": candidate["source_summary"],
        "score_contract": candidate["score_contract"],
        "support_contract": candidate["support_contract"],
        "target_causal_effect_head": candidate["target_causal_effect_head"],
        "target_incremental_utility_head": candidate[
            "target_incremental_utility_head"
        ],
        "conformal": candidate["conformal"],
        "development_authorized": True,
        "confirmation_authorized": True,
        "development_read_and_passed": True,
        "sealed_confirmation_read_or_run": False,
        "existing_valid_unseen_read_or_run": False,
    }
    promoted = body | {"candidate_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(promoted, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_sha256": promoted["candidate_sha256"],
        "status": promoted["status"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
