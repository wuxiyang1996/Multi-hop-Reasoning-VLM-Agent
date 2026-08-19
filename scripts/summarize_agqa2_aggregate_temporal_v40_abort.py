#!/usr/bin/env python3
"""Record V40's post-runtime, pre-source-evaluator assembly failure."""

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
    run = REPO_ROOT / "runs/agqa2_aggregate_temporal_v40_formal"
    if (run / "base_report.json").exists() or (run / "report.json").exists():
        raise RuntimeError("V40 already has a report")
    receipts = sorted((run / "runtime_receipts").glob("*.json"))
    calls = sorted((run / "call_cache").glob("*/*.json"))
    if len(receipts) != 100:
        raise ValueError("V40 did not freeze all runtime receipts")
    cost = 0.0
    for path in calls:
        value = json.loads(path.read_text())
        body = dict(value)
        claimed = body.pop("call_receipt_sha256")
        if stable_hash(body) != claimed:
            raise ValueError(f"call receipt hash mismatch: {path}")
        cost += float(value["usage"]["reported_cost_usd"])
    config_path = REPO_ROOT / "configs/agqa2_aggregate_temporal_v40_formal.json"
    prereg_path = REPO_ROOT / (
        "configs/agqa2_aggregate_temporal_v40_formal_preregistration.json"
    )
    core = {
        "schema_version": "agqa2-aggregate-temporal-v40-runtime-abort-v1",
        "status": "AGQA2_AGGREGATE_TEMPORAL_V40_RUNTIME_ASSEMBLY_INCOMPLETE",
        "failure_class": "DEVELOPMENT_DEPENDENCY_REPORT_SHA256_ALIAS_MISSING",
        "completed_runtime_receipts": len(receipts),
        "runtime_receipt_set_sha256": stable_hash([
            {"path": path.name, "file_sha256": _sha256(path)}
            for path in receipts
        ]),
        "provider_call_receipts": len(calls),
        "reported_provider_cost_usd": cost,
        "base_legacy_evaluator_loop_entered": True,
        "official_answer_field_accessed_after_all_runtime_receipts_froze": True,
        "source_prediction_freeze_loop_entered": False,
        "postground_report_created": False,
        "formal_metrics_externalized": False,
        "human_visible_formal_scores_before_repair": False,
        "source_method_adapter_or_gate_changed": False,
        "config_file_sha256": _sha256(config_path),
        "preregistration_file_sha256": _sha256(prereg_path),
        "next_authorized_use": (
            "DETERMINISTIC_COMPLETION_FROM_EXACT_FROZEN_RECEIPTS_WITH_A_"
            "DEPENDENCY_SCHEMA_ALIAS_ONLY;NO_METHOD_GATE_OR_SAMPLE_CHANGE"
        ),
        "confirmatory_claim_pending_completion": True,
    }
    result = core | {"result_sha256": stable_hash(core)}
    output = REPO_ROOT / (
        "docs/results/agqa2_aggregate_temporal_v40_runtime_abort.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
