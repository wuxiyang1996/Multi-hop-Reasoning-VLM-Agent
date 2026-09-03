#!/usr/bin/env python3
"""File-hash transport wrapper for the frozen V13 evaluator function."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
from scripts.evaluate_agqa2_source_executor_v13 import evaluate  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    config = json.loads(args.config.read_text())
    report = json.loads(args.report.read_text())
    if report.get("config_sha256") != sha256(args.config):
        raise ValueError("report belongs to a different config file")
    if config.get("formal_protocol_file_sha256") != sha256(args.protocol):
        raise ValueError("config belongs to a different protocol file")
    core = REPO_ROOT / "scripts/evaluate_agqa2_source_executor_v13.py"
    if protocol.get("evaluator_file_sha256") != sha256(core):
        raise ValueError("frozen evaluator core differs from protocol")
    result = evaluate(protocol, report)
    result["transport_fix"] = {
        "reason": "CHECK_COLLECTOR_CONFIG_AS_FILE_SHA256",
        "frozen_evaluator_core_sha256": sha256(core),
        "prediction_or_gate_semantics_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
