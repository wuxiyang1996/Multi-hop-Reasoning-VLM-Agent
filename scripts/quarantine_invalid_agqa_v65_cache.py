#!/usr/bin/env python3
"""Quarantine provider cache rows already rejected by the V65 validator.

V65 cached a provider payload before semantic validation.  A malformed row
therefore became permanently unretryable.  This utility moves only cache rows
that independently fail the same receipt invariants; it never repairs a
payload, opens a target outcome, or changes the next request's input hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from motif_transfer.contracts import stable_hash  # noqa: E402


def _semantic_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        return ["missing observations"]
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            errors.append(f"observation[{index}] is not an object")
            continue
        observed = observation.get("observability") == "OBSERVED"
        evidence = observation.get("evidence_frames")
        start, end = observation.get("start_frame"), observation.get("end_frame")
        if observed and (
            not isinstance(evidence, list) or not evidence
            or not isinstance(start, int) or not isinstance(end, int)
        ):
            errors.append("OBSERVED operand occurrences require pixel evidence")
        if not observed and (
            evidence not in ([], None) or start is not None or end is not None
        ):
            errors.append("UNOBSERVED operand occurrences cannot claim pixel evidence")
    return errors


def quarantine(cache_root: Path, errors_path: Path, output: Path) -> dict[str, Any]:
    errors = json.loads(errors_path.read_text()).get("errors", {})
    candidates = {
        task_id for task_id, error in errors.items()
        if "schema retries exhausted" in str(error)
    }
    rows = []
    quarantine_root = cache_root.parent / "rejected_schema_cache"
    for task_id in sorted(candidates):
        for path in sorted((cache_root / task_id).glob("operand_*.json")):
            cached = json.loads(path.read_text())
            semantic_errors = _semantic_errors(cached.get("payload", {}))
            if not semantic_errors:
                continue
            destination = quarantine_root / task_id / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"quarantine destination exists: {destination}")
            original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            shutil.move(str(path), str(destination))
            rows.append({
                "task_id": task_id,
                "cache_relative_path": str(path.relative_to(cache_root.parent)),
                "quarantine_relative_path": str(
                    destination.relative_to(cache_root.parent)
                ),
                "cache_file_sha256": original_sha,
                "input_sha256": cached.get("input_sha256"),
                "semantic_errors": semantic_errors,
                "payload_changed": False,
                "target_outcome_read": False,
            })
    body = {
        "schema_version": "agqa-v65-invalid-cache-quarantine-v1",
        "status": "QUARANTINED_INVALID_PROVIDER_CACHE",
        "source_worker_errors_file_sha256": hashlib.sha256(
            errors_path.read_bytes()
        ).hexdigest(),
        "quarantined_count": len(rows),
        "request_input_or_prompt_changed": False,
        "model_or_validator_changed": False,
        "target_outcome_read": False,
        "rows": rows,
    }
    result = body | {"receipt_sha256": stable_hash(body)}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--worker-errors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = quarantine(
        args.cache_root.resolve(), args.worker_errors.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["quarantined_count"]:
        raise SystemExit("no semantically invalid cached payloads found")


if __name__ == "__main__":
    main()
