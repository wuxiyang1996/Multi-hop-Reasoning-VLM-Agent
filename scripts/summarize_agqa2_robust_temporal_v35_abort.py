#!/usr/bin/env python3
"""Record the V35 development runtime abort before outcome evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    run = REPO_ROOT / "runs/agqa2_robust_temporal_v35_development"
    errors_path = run / "worker_errors.json"
    if (run / "base_report.json").exists() or (run / "report.json").exists():
        raise RuntimeError("V35 produced an evaluation report; abort is invalid")
    errors = json.loads(errors_path.read_text())
    receipts = sorted((run / "runtime_receipts").glob("*.json"))
    calls = sorted((run / "call_cache").glob("*/*.json"))
    cost = 0.0
    for path in calls:
        value = json.loads(path.read_text())
        body = dict(value)
        claimed = body.pop("call_receipt_sha256")
        if stable_hash(body) != claimed:
            raise ValueError(f"call receipt hash mismatch: {path}")
        cost += float(value["usage"]["reported_cost_usd"])
    config_path = REPO_ROOT / (
        "configs/agqa2_robust_temporal_v35_development.json"
    )
    prereg_path = REPO_ROOT / (
        "configs/agqa2_robust_temporal_v35_development_preregistration.json"
    )
    core = {
        "schema_version": "agqa2-robust-temporal-v35-runtime-abort-v1",
        "status": "AGQA2_ROBUST_TEMPORAL_V35_DEVELOPMENT_RUNTIME_INCOMPLETE",
        "completed_runtime_receipts": len(receipts),
        "failed_runtime_rows": len(errors["errors"]),
        "worker_errors": errors["errors"],
        "provider_call_receipts": len(calls),
        "reported_provider_cost_usd": cost,
        "base_report_created": False,
        "postground_report_created": False,
        "evaluator_loop_entered": False,
        "official_answer_field_accessed": False,
        "config_file_sha256": _sha256(config_path),
        "preregistration_file_sha256": _sha256(prereg_path),
        "worker_errors_file_sha256": _sha256(errors_path),
        "completed_runtime_receipt_set_sha256": stable_hash([
            {"path": path.name, "file_sha256": _sha256(path)}
            for path in receipts
        ]),
        "next_authorized_use": (
            "DEVELOPMENT_ONLY_STABLE_SORT_DEDUP_OF_EXISTING_VALID_EVIDENCE_"
            "FRAME_IDS_BEFORE_INTERVAL_ENVELOPE_NORMALIZATION"
        ),
        "confirmatory_claim_allowed": False,
    }
    result = core | {"result_sha256": stable_hash(core)}
    output = REPO_ROOT / (
        "docs/results/agqa2_robust_temporal_v35_runtime_abort.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
