#!/usr/bin/env python3
"""Repair null stable-hash references in an outcome-blind V21 enumeration."""

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


def _validate(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise ValueError(f"invalid stable hash: {field}")
    return claimed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite repaired report: {args.output}")
    original = _read(args.input)
    original_hash = _validate(original, "report_sha256")
    if original.get("status") != "OUTCOME_BLIND_EVAL_OPPORTUNITIES_COMPLETE":
        raise SystemExit("V21 input is not an outcome-blind enumeration")
    if original.get("outcomes_recorded") or original.get("rewards_recorded"):
        raise SystemExit("V21 input unexpectedly contains outcomes")
    manifest_path = Path(str(original["manifest"]["path"]))
    candidate_path = Path(str(original["candidate"]["path"]))
    if _sha256(manifest_path) != original["manifest"]["file_sha256"]:
        raise SystemExit("V21 manifest file receipt changed")
    if _sha256(candidate_path) != original["candidate"]["file_sha256"]:
        raise SystemExit("V21 candidate file receipt changed")
    manifest = _read(manifest_path)
    manifest_hash = _validate(manifest, "manifest_sha256")
    candidate = _read(candidate_path)
    candidate_hash = _validate(candidate, "candidate_sha256")
    if original["manifest"].get("manifest_sha256") is not None:
        raise SystemExit("V21 manifest hash is not the known null-return bug")
    if original["candidate"].get("candidate_sha256") is not None:
        raise SystemExit("V21 candidate hash is not the known null-return bug")
    opportunities_hash = stable_hash(original["opportunities"])
    body = dict(original)
    body.pop("report_sha256")
    body["manifest"] = dict(body["manifest"]) | {
        "manifest_sha256": manifest_hash,
    }
    body["candidate"] = dict(body["candidate"]) | {
        "candidate_sha256": candidate_hash,
    }
    body["receipt_repair"] = {
        "authority": (
            "REFERENCE_HASH_FIELDS_ONLY_AFTER_VALIDATING_FILE_SHA_AND_"
            "STABLE_HASH; NO_TASK_SCORE_ADMISSION_OR_OUTCOME_FIELD_CHANGED"
        ),
        "original_path": str(args.input.resolve()),
        "original_file_sha256": _sha256(args.input),
        "original_report_sha256": original_hash,
        "original_opportunities_sha256": opportunities_hash,
        "repaired_opportunities_sha256": stable_hash(body["opportunities"]),
        "repair_code_file_sha256": _sha256(Path(__file__)),
    }
    if body["receipt_repair"]["original_opportunities_sha256"] != (
        body["receipt_repair"]["repaired_opportunities_sha256"]
    ):
        raise RuntimeError("V21 receipt repair changed opportunities")
    repaired = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(repaired, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "report_sha256": repaired["report_sha256"],
        "original_report_sha256": original_hash,
        "opportunities_sha256": opportunities_hash,
        "manifest_sha256": manifest_hash,
        "candidate_sha256": candidate_hash,
        "outcomes_recorded": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
