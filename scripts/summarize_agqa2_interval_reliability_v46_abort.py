#!/usr/bin/env python3
"""Record V46's post-base, pre-calibrated legacy-key abort."""

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
    run = REPO_ROOT / "runs/agqa2_interval_reliability_v46_qualification"
    base_path = run / "base_report.json"
    if not base_path.exists() or (run / "report.json").exists():
        raise RuntimeError("unexpected V46 report state")
    base = json.loads(base_path.read_text())
    body = dict(base)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed or len(base["rows"]) != 150:
        raise ValueError("V46 base report integrity failure")
    config_path = REPO_ROOT / "configs/agqa2_interval_reliability_v46_qualification.json"
    prereg_path = REPO_ROOT / (
        "configs/agqa2_interval_reliability_v46_qualification_preregistration.json"
    )
    core = {
        "schema_version": "agqa2-interval-reliability-v46-runtime-abort-v1",
        "status": "AGQA2_INTERVAL_RELIABILITY_V46_QUALIFICATION_RUNTIME_INCOMPLETE",
        "failure_class": "LEGACY_PREREGISTRATION_ALIAS_MISSING",
        "failure_detail": "qualified_v33_development_report_sha256",
        "completed_runtime_receipts": 150,
        "base_report_created": True,
        "base_report_sha256": claimed,
        "base_report_file_sha256": _sha256(base_path),
        "base_legacy_evaluator_outcome_accessed_after_receipts_froze": True,
        "calibrated_prediction_loop_entered": False,
        "calibrated_metrics_externalized": False,
        "provider_calls": int(base["provider_calls"]),
        "reported_provider_cost_usd": float(base["reported_provider_cost_usd"]),
        "config_file_sha256": _sha256(config_path),
        "preregistration_file_sha256": _sha256(prereg_path),
        "next_authorized_use": (
            "DEVELOPMENT_QUALIFICATION_ONLY;ADD_LEGACY_ALIAS_EQUAL_TO_THE_"
            "ALREADY_FROZEN_V45_ARTIFACT_HASH;NO_RULE_GATE_SAMPLE_OR_RECEIPT_CHANGE"
        ),
        "formal_claim_allowed": False,
    }
    result = core | {"result_sha256": stable_hash(core)}
    output = REPO_ROOT / (
        "docs/results/agqa2_interval_reliability_v46_runtime_abort.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
