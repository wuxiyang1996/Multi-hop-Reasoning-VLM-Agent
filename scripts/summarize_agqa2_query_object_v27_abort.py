#!/usr/bin/env python3
"""Seal the V27 token-cap replay abort before the bounded-schema repair."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256  # noqa: E402


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_query_object_v27_replay"
    if (run_root / "report.json").exists():
        raise RuntimeError("V27 formal report exists; abort summary would be false")
    errors = json.loads((run_root / "worker_errors.json").read_text())
    if set(errors["errors"]) != {"QMIKJ-29239"}:
        raise ValueError("V27 terminal worker error is not the audited task")
    receipts = sorted((run_root / "runtime_receipts").glob("*.json"))
    if len(receipts) != 119:
        raise ValueError("V27 abort must contain exactly 119 frozen receipts")
    config_path = REPO_ROOT / "configs/agqa2_query_object_v27_replay.json"
    config = json.loads(config_path.read_text())
    for path in receipts:
        row = json.loads(path.read_text())
        body = dict(row)
        claimed = body.pop("runtime_receipt_sha256")
        if stable_hash(body) != claimed:
            raise ValueError(f"invalid V27 runtime receipt: {path.name}")
        if row["grounder_sha256"] != config["expected_grounder_sha256"]:
            raise ValueError(f"wrong V27 grounder receipt: {path.name}")
    calls = [
        json.loads(path.read_text())
        for path in sorted((run_root / "call_cache").glob("*/*.json"))
    ]
    body = {
        "schema_version": "agqa2-query-object-v27-runtime-abort-v1",
        "status": "V27_RUNTIME_INCOMPLETE_BEFORE_FORMAL_EVALUATION",
        "stage": "TARGET_NATIVE_ONTOLOGY_PROVIDER_JSON_DECODE",
        "completed_runtime_receipts": len(receipts),
        "required_runtime_receipts": 120,
        "terminal_worker_errors": errors["errors"],
        "accepted_provider_calls_including_v26_hash_reuse": len(calls),
        "accepted_reported_provider_cost_usd_including_v26_hash_reuse": sum(
            float(row["usage"]["reported_cost_usd"]) for row in calls
        ),
        "grounder_sha256": config["expected_grounder_sha256"],
        "config_file_sha256": _sha256(config_path),
        "formal_report_created": False,
        "formal_gold_evaluation_started": False,
        "official_answers_inspected_for_repair": False,
        "repair_policy": (
            "BOUND_VISUAL_DESCRIPTION_AND_UNCERTAINTY_TO_160_CHARACTERS;KEEP_"
            "DECISION_CONFIDENCE_EVIDENCE_MODEL_AND_FRAMES_UNCHANGED;REQUALIFY_"
            "ON_DEVELOPMENT;USE_A_NEW_V28_POOL_EXCLUDING_V26_VIDEOS"
        ),
    }
    result = body | {"abort_sha256": stable_hash(body)}
    output = REPO_ROOT / "docs/results/agqa2_query_object_v27_runtime_abort.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output.relative_to(REPO_ROOT)),
        "status": result["status"],
        "completed_runtime_receipts": len(receipts),
        "accepted_provider_calls": len(calls),
        "accepted_reported_provider_cost_usd": result[
            "accepted_reported_provider_cost_usd_including_v26_hash_reuse"
        ],
        "abort_sha256": result["abort_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
