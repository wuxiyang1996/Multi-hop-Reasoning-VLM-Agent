#!/usr/bin/env python3
"""Transport-only binding fix for the frozen V15 evaluator core.

The collector records a file SHA256 in ``report.config_sha256``.  The frozen
evaluator CLI compared it to a JSON stable hash.  This wrapper performs the
correct file-hash check and delegates all outcome and gate computation to the
unchanged, protocol-hashed ``evaluate`` function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
from scripts.evaluate_agqa2_source_executor_v13 import evaluate  # noqa: E402


def _sha256(path: Path) -> str:
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
    if report.get("config_sha256") != _sha256(args.config):
        raise ValueError("report belongs to a different activated config file")
    if config.get("formal_protocol_file_sha256") != _sha256(args.protocol):
        raise ValueError("activated config belongs to a different formal protocol")
    core = REPO_ROOT / "scripts/evaluate_agqa2_source_executor_v13.py"
    if _sha256(core) != protocol["mechanism"]["evaluator_file_sha256"]:
        raise ValueError("frozen evaluator core differs from protocol")
    result = evaluate(protocol, report)
    result["transport_fix"] = {
        "reason": "COLLECTOR_CONFIG_FILE_SHA_WAS_COMPARED_AS_JSON_STABLE_HASH",
        "frozen_evaluator_core_sha256": _sha256(core),
        "prediction_or_gate_semantics_changed": False,
    }
    # Do not rewrite the core report hash after appending an explicitly
    # non-semantic transport receipt; the core hash remains independently
    # verifiable over the fields emitted by evaluate().
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
