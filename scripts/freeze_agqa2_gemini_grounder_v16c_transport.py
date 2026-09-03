#!/usr/bin/env python3
"""Freeze a clean rerun with sufficient Gemini JSON transport budgets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "configs/agqa2_gemini_grounder_v16b_development.json"
OUTPUT = REPO_ROOT / "configs/agqa2_gemini_grounder_v16c_development.json"
ABORT = REPO_ROOT / "docs/results/agqa2_gemini_grounder_v16b_transport_abort.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"frozen artifact already exists: {OUTPUT}")
    config = json.loads(SOURCE.read_text())
    abort = json.loads(ABORT.read_text())
    if abort["status"] != "ABORTED_BEFORE_OUTCOME_ACCESS_OR_GATE_EVALUATION":
        raise ValueError("V16b transport abort receipt is invalid")
    config["status"] = "FROZEN_V75C_CLEAN_TRANSPORT_RERUN_BEFORE_OUTCOME_ACCESS"
    config["report_version"] = "GEMINI31_V75C_DEV"
    config["claim_boundary"] += ";V16B_INCOMPLETE_TRANSPORT_RUN_DISCLOSED"
    config["model"]["max_operand_tokens"] = 1600
    config["model"]["max_direct_tokens"] = 256
    config["transport_amendment"] = {
        "prior_config_file_sha256": sha256(SOURCE),
        "abort_receipt_file_sha256": sha256(ABORT),
        "all_samples_rerun_from_scratch": True,
        "prior_runtime_receipts_reused": False,
        "prompt_changed": False,
        "grounder_ir_changed": False,
        "qualification_gates_changed": False,
        "operand_token_limit": 1600,
        "direct_token_limit": 256,
        "reason": "ALLOW_COMPLETE_SCHEMA_CONFORMING_JSON_AFTER_MINIMAL_PROVIDER_REASONING",
    }
    OUTPUT.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": config["status"],
        "config_file_sha256": sha256(OUTPUT),
        "max_operand_tokens": config["model"]["max_operand_tokens"],
        "max_direct_tokens": config["model"]["max_direct_tokens"],
    }, indent=2))


if __name__ == "__main__":
    main()
