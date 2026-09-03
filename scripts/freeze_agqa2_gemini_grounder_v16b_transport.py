#!/usr/bin/env python3
"""Freeze the transport-only Gemini reasoning-budget repair for V75."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "configs/agqa2_gemini_grounder_v16_development.json"
OUTPUT = REPO_ROOT / "configs/agqa2_gemini_grounder_v16b_development.json"
ABORT = REPO_ROOT / "docs/results/agqa2_gemini_grounder_v16_transport_abort.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"frozen artifact already exists: {OUTPUT}")
    config = json.loads(SOURCE.read_text())
    abort = json.loads(ABORT.read_text())
    if abort["status"] != "ABORTED_BEFORE_VALID_EVALUATION":
        raise ValueError("V16 transport abort receipt is invalid")
    config["status"] = "FROZEN_V75B_TRANSPORT_ONLY_REPAIR_BEFORE_VALID_RUNTIME"
    config["report_version"] = "GEMINI31_V75B_DEV"
    config["claim_boundary"] += ";V16_EMPTY_CONTENT_TRANSPORT_ABORT_DISCLOSED"
    config["model"]["reasoning"] = {"effort": "minimal", "exclude": True}
    config["transport_amendment"] = {
        "prior_config_file_sha256": sha256(SOURCE),
        "abort_receipt_file_sha256": sha256(ABORT),
        "prompt_changed": False,
        "grounder_ir_changed": False,
        "qualification_gates_changed": False,
        "reason": "PREVENT_HIDDEN_REASONING_FROM_EXHAUSTING_JSON_RESPONSE_BUDGET",
    }
    OUTPUT.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": config["status"],
        "config_file_sha256": sha256(OUTPUT),
        "reasoning": config["model"]["reasoning"],
    }, indent=2))


if __name__ == "__main__":
    main()
