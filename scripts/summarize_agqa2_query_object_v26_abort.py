#!/usr/bin/env python3
"""Seal the outcome-blind V26 acquisition abort before any repair."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256  # noqa: E402


RETRY_HISTORY = {
    "GL2JW-1948": {
        "initial_error": "JSONDecodeError: truncated JSON object",
        "same_protocol_resume_attempts": 1,
        "eventually_completed": True,
    },
    "HLB3J-14232": {
        "initial_error": "ValueError: provider response omitted a JSON object",
        "same_protocol_resume_attempts": 1,
        "eventually_completed": True,
    },
    "QMIKJ-29239": {
        "initial_error": "ValueError: provider response omitted a JSON object",
        "same_protocol_resume_attempts": 2,
        "eventually_completed": False,
    },
}


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_query_object_v26_reserve"
    report_path = run_root / "report.json"
    if report_path.exists():
        raise RuntimeError("V26 formal report exists; abort summary would be false")
    errors = json.loads((run_root / "worker_errors.json").read_text())
    if set(errors["errors"]) != {"QMIKJ-29239"}:
        raise ValueError("V26 terminal worker error differs from audited ledger")
    receipts = sorted((run_root / "runtime_receipts").glob("*.json"))
    if len(receipts) != 119:
        raise ValueError("V26 abort must contain exactly 119 frozen receipts")
    config_path = REPO_ROOT / "configs/agqa2_query_object_v26_reserve.json"
    config = json.loads(config_path.read_text())
    for path in receipts:
        row = json.loads(path.read_text())
        body = dict(row)
        claimed = body.pop("runtime_receipt_sha256")
        if (
            stable_hash(body) != claimed
            or row["grounder_sha256"] != config["expected_grounder_sha256"]
        ):
            raise ValueError(f"invalid V26 runtime receipt: {path.name}")
    calls = [
        json.loads(path.read_text())
        for path in sorted((run_root / "call_cache").glob("*/*.json"))
    ]
    body = {
        "schema_version": "agqa2-query-object-v26-runtime-abort-v1",
        "status": "V26_RUNTIME_INCOMPLETE_BEFORE_FORMAL_EVALUATION",
        "stage": "TARGET_NATIVE_ONTOLOGY_PROVIDER_JSON_DECODE",
        "completed_runtime_receipts": len(receipts),
        "required_runtime_receipts": 120,
        "terminal_worker_errors": errors["errors"],
        "retry_history": RETRY_HISTORY,
        "accepted_provider_calls": len(calls),
        "accepted_reported_provider_cost_usd": sum(
            float(row["usage"]["reported_cost_usd"]) for row in calls
        ),
        "grounder_sha256": config["expected_grounder_sha256"],
        "config_file_sha256": _sha256(config_path),
        "selection_manifest_sha256": json.loads((
            REPO_ROOT / "configs/agqa2_query_object_v26_reserve_selection.json"
        ).read_text())["manifest_sha256"],
        "formal_report_created": False,
        "formal_gold_evaluation_started": False,
        "official_answers_inspected_for_repair": False,
        "post_abort_gold_blind_diagnostic": {
            "task_id": "QMIKJ-29239",
            "relation": "above",
            "model": "google/gemini-2.5-flash-lite",
            "finish_reason": "length",
            "prompt_tokens": 2348,
            "completion_tokens": 300,
            "configured_max_ontology_tokens": 300,
            "refusal": False,
            "reasoning_tokens_observed": False,
            "reported_cost_usd": 0.00020027,
            "diagnostic_not_used_as_formal_receipt": True,
        },
        "repair_policy": (
            "INCREASE_PRIMARY_ONTOLOGY_MAX_TOKENS_300_TO_500;REQUALIFY_ON_"
            "DEVELOPMENT;REPLAY_THE_EXACT_OUTCOME_BLIND_V26_POOL_WITH_IDENTICAL_"
            "SOURCE_SPECIFIC_GATES;DO_NOT_SELECT_A_NEW_SEED"
        ),
    }
    result = body | {"abort_sha256": stable_hash(body)}
    output = REPO_ROOT / "docs/results/agqa2_query_object_v26_runtime_abort.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output.relative_to(REPO_ROOT)),
        "status": result["status"],
        "completed_runtime_receipts": len(receipts),
        "accepted_provider_calls": len(calls),
        "accepted_reported_provider_cost_usd": result[
            "accepted_reported_provider_cost_usd"
        ],
        "abort_sha256": result["abort_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
