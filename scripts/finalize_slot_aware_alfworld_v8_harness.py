#!/usr/bin/env python3
"""Authorize V8 confirmation only from a passing hash-bound adaptation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from motif_transfer.contracts import stable_hash


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_hash(value: dict[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise SystemExit(f"invalid frozen artifact hash: {field}")


def _validate_file(receipt: Mapping[str, Any]) -> None:
    path = Path(str(receipt["path"]))
    if _sha256(path) != receipt["file_sha256"]:
        raise SystemExit(f"frozen file changed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--adaptation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite final Harness: {args.output}")
    candidate = _read(args.candidate)
    _validate_hash(candidate, "candidate_sha256")
    if candidate.get("status") != "ADAPTATION_GATE_ONLY":
        raise SystemExit("finalizer requires an adaptation-only candidate")
    for receipt in candidate["implementation"].values():
        _validate_file(receipt)
    gate = _read(args.adaptation_report)
    _validate_hash(gate, "report_sha256")
    if gate.get("status") != "ADAPTATION_GATE_PASSED":
        raise SystemExit("V8 adaptation gate did not pass")
    if gate.get("phase") != "adaptation_gate" or not gate.get("passed"):
        raise SystemExit("V8 gate report has invalid phase/status")
    if not all(bool(value) for value in gate.get("gates", {}).values()):
        raise SystemExit("V8 gate report contains a failed requirement")
    if gate.get("existing_valid_unseen_heldout_read"):
        raise SystemExit("V8 gate crossed the reserved heldout boundary")
    if gate.get("official_outcome_used_for_action_selection"):
        raise SystemExit("V8 gate used evaluator outcomes for action selection")
    if (
        gate.get("artifact_hash_field") != "candidate_sha256"
        or gate.get("artifact_sha256") != candidate["candidate_sha256"]
        or gate.get("artifact_file_sha256") != _sha256(args.candidate)
    ):
        raise SystemExit("V8 gate was not produced by this candidate")
    if (
        gate.get("manifest_sha256")
        != candidate["manifest"]["manifest_sha256"]
        or gate.get("manifest_file_sha256")
        != candidate["manifest"]["file_sha256"]
    ):
        raise SystemExit("V8 gate used a different manifest")
    candidate_body = dict(candidate)
    candidate_hash = str(candidate_body.pop("candidate_sha256"))
    candidate_body["schema_version"] = "slot-aware-alfworld-harness-v8"
    candidate_body["status"] = "FRESH_CONFIRMATION_AUTHORIZED"
    candidate_body["claim_boundary"] = (
        "RELATIONAL_BIND_RELATE_TRANSFER_ONLY; HASH_BOUND_ADAPTATION_GATE_"
        "PASSED; FRESH_TRAIN_INSTANCE_CONFIRMATION_AUTHORIZED_ONCE; "
        "EXISTING_VALID_UNSEEN_HELDOUT_FORBIDDEN"
    )
    candidate_body["candidate_parent"] = {
        "path": str(args.candidate.resolve()),
        "file_sha256": _sha256(args.candidate),
        "candidate_sha256": candidate_hash,
    }
    candidate_body["adaptation_gate_report"] = {
        "path": str(args.adaptation_report.resolve()),
        "file_sha256": _sha256(args.adaptation_report),
        "report_sha256": gate["report_sha256"],
        "successes": gate["summaries"]["authentic_slot_ir"]["successes"],
        "target_only_successes": gate["summaries"]["target_only"][
            "successes"
        ],
        "changed_by_effect": gate["summaries"]["authentic_slot_ir"][
            "changed_by_effect"
        ],
    }
    candidate_body["permissions"] = dict(candidate_body["permissions"])
    candidate_body["permissions"]["fresh_confirmation"] = [
        "RUN_EXACT_BOUND_MANIFEST_FRESH_CONFIRMATION_ONCE",
        "REPORT_ALL_CONDITIONS_AND_PAIRED_OUTCOMES",
    ]
    result = candidate_body | {
        "harness_sha256": stable_hash(candidate_body)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": result["status"],
        "harness_sha256": result["harness_sha256"],
        "candidate_sha256": candidate_hash,
        "adaptation_report_sha256": gate["report_sha256"],
        "manifest_sha256": result["manifest"]["manifest_sha256"],
        "transfer_scope": result["transfer_scope"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
