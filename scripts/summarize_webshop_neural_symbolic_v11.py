#!/usr/bin/env python3
"""Evaluate the independently frozen V11 fresh-goal replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402
from scripts.summarize_webshop_neural_symbolic_v10 import (  # noqa: E402
    AUTHENTIC,
    CONTROLS,
    _load,
    _paired,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.frozen_config.read_text())
    rows = _load(args.receipt_dir)
    expected = set(config["task_ids"])
    observed = {row["task_id"] for row in rows}
    if observed != expected:
        raise SystemExit("V11 task set mismatch")
    comparisons = [_paired(rows, control) for control in CONTROLS]
    zero_failures = all(row["failure"] is None for row in rows)
    matched = all(
        len({
            row["initial_state_hash"] for row in rows if row["task_id"] == task_id
        }) == 1
        for task_id in expected
    )
    source_decisions = sum(
        row["source_decision_count"] for row in rows if row["condition"] == AUTHENTIC
    )
    passed = bool(
        zero_failures
        and matched
        and source_decisions > 0
        and all(
            comparison["strict_wins"] > comparison["strict_losses"]
            and comparison["strict_success_delta"] > 0
            and comparison["mean_reward_delta"] > 0
            and comparison["paired_exact_p_two_sided"] <= 0.05
            and comparison["action_contrast_tasks"] > 0
            for comparison in comparisons
        )
    )
    report = {
        "schema_version": 1,
        "experiment": "webshop_neural_symbolic_transfer_v11_independent_replication",
        "claim_boundary": config["claim_boundary"],
        "tasks": len(expected),
        "zero_failures": zero_failures,
        "matched_initial_state_hashes": matched,
        "authentic_source_decisions": source_decisions,
        "comparisons": comparisons,
        "passed": passed,
        "scientific_status": (
            "REAL_WEBSHOP_NEURAL_SYMBOLIC_TRANSFER_INDEPENDENTLY_VALIDATED"
            if passed else "REAL_WEBSHOP_NEURAL_SYMBOLIC_TRANSFER_V11_NOT_VALIDATED"
        ),
        "v10_data_used_in_v11_test": False,
        "runtime_hashes": {
            "summarizer": file_sha256(Path(__file__)),
            "frozen_config": file_sha256(args.frozen_config),
            "run_summary": file_sha256(args.receipt_dir / "summary.json"),
        },
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
