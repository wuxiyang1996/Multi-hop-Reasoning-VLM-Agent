#!/usr/bin/env python3
"""Record V43's post-base, pre-calibrated-evaluator serialization abort."""

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
    run = REPO_ROOT / "runs/agqa2_view_reliability_v43_qualification"
    base_path = run / "base_report.json"
    if not base_path.exists() or (run / "report.json").exists():
        raise RuntimeError("unexpected V43 report state")
    base = json.loads(base_path.read_text())
    body = dict(base)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed or len(base["rows"]) != 150:
        raise ValueError("V43 base report integrity failure")
    config_path = REPO_ROOT / "configs/agqa2_view_reliability_v43_qualification.json"
    prereg_path = REPO_ROOT / (
        "configs/agqa2_view_reliability_v43_qualification_preregistration.json"
    )
    core = {
        "schema_version": "agqa2-view-reliability-v43-runtime-abort-v1",
        "status": "AGQA2_VIEW_RELIABILITY_V43_QUALIFICATION_RUNTIME_INCOMPLETE",
        "failure_class": "JSON_LIST_TO_TUPLE_CANONICALIZATION_MISSING",
        "failure_detail": "allowed_singleton_views",
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
            "DEVELOPMENT_QUALIFICATION_ONLY;CANONICALIZE_THE_EXISTING_JSON_"
            "LIST_TO_THE_IDENTICAL_TUPLE_RULE;NO_RULE_GATE_OR_SAMPLE_CHANGE"
        ),
        "formal_claim_allowed": False,
    }
    result = core | {"result_sha256": stable_hash(core)}
    output = REPO_ROOT / (
        "docs/results/agqa2_view_reliability_v43_runtime_abort.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
