#!/usr/bin/env python3
"""Record the V36 evaluator-interface abort without scoring its rows."""

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
    run = REPO_ROOT / "runs/agqa2_robust_temporal_v36_development"
    base_path = run / "base_report.json"
    report_path = run / "report.json"
    if not base_path.exists() or report_path.exists():
        raise RuntimeError("V36 abort requires a base report and no scored report")
    base = json.loads(base_path.read_text())
    body = dict(base)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed or len(base["rows"]) != 100:
        raise ValueError("V36 base report integrity failure")
    config_path = REPO_ROOT / "configs/agqa2_robust_temporal_v36_development.json"
    prereg_path = REPO_ROOT / (
        "configs/agqa2_robust_temporal_v36_development_preregistration.json"
    )
    core = {
        "schema_version": "agqa2-robust-temporal-v36-runtime-abort-v1",
        "status": "AGQA2_ROBUST_TEMPORAL_V36_DEVELOPMENT_RUNTIME_INCOMPLETE",
        "completed_runtime_receipts": 100,
        "base_report_created": True,
        "base_report_sha256": claimed,
        "postground_report_created": False,
        "evaluator_function_entered": True,
        "prediction_freeze_loop_entered": False,
        "official_answer_field_accessed": False,
        "failure_class": "EVALUATOR_PREREGISTRATION_INTERFACE_KEY_MISSING",
        "failure_detail": "qualified_v33_development_report_sha256",
        "grounding_schema_failures_remaining": 0,
        "provider_call_receipts": int(base["provider_calls"]),
        "reported_provider_cost_usd": float(base["reported_provider_cost_usd"]),
        "config_file_sha256": _sha256(config_path),
        "preregistration_file_sha256": _sha256(prereg_path),
        "next_authorized_use": (
            "DEVELOPMENT_ONLY_EVALUATION_OF_THE_UNCHANGED_HASHED_V36_BASE_"
            "REPORT_WITH_THE_V33_EVIDENCE_LINEAGE_COMPATIBILITY_KEY"
        ),
        "confirmatory_claim_allowed": False,
    }
    result = core | {"result_sha256": stable_hash(core)}
    output = REPO_ROOT / (
        "docs/results/agqa2_robust_temporal_v36_runtime_abort.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
