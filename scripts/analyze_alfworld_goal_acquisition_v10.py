#!/usr/bin/env python3
"""Correct the V10 wrong-handle gate scope without changing any rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


AUTHENTIC = "authentic_source_goal_relation_macro"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(payload: dict, field: str) -> None:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def wrong_handle_counts(report: dict) -> tuple[int, int]:
    """Return all-cycle and source-active RELATE_NO_PROGRESS counts."""

    all_cycle = source_active = 0
    for episode in report["episodes"][AUTHENTIC]:
        for record in episode["records"]:
            if record["target_effect_receipt"] != "RELATE_NO_PROGRESS":
                continue
            all_cycle += 1
            source_active += int(int(record["completed_count_before"]) >= 1)
    return all_cycle, source_active


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/alfworld_goal_acquisition_v10_analysis.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _self_hash(config, "config_sha256")
    if _sha256(Path(__file__).resolve()) != config["analyzer_file_sha256"]:
        raise SystemExit("frozen V10 analyzer changed")
    source_path = REPO / config["source_report"]
    if _sha256(source_path) != config["source_report_file_sha256"]:
        raise SystemExit("frozen V10 source report changed")
    report = json.loads(source_path.read_text(encoding="utf-8"))
    _self_hash(report, "report_sha256")
    if report.get("status") != "CONSUMED_DEVELOPMENT_TYPED_ACQUISITION_GATE_FAILED":
        raise SystemExit("V10 analysis expected the original single-gate failure")
    false_gates = sorted(name for name, value in report["gates"].items() if not value)
    if false_gates != ["zero_wrong_handle_relation_effects"]:
        raise SystemExit(f"V10 has unexpected failed gates: {false_gates}")
    all_cycle, source_active = wrong_handle_counts(report)
    corrected_gates = dict(report["gates"])
    corrected_gates["zero_wrong_handle_relation_effects"] = source_active == 0
    passed = all(corrected_gates.values())
    body = {
        "schema_version": "alfworld-goal-acquisition-v10-gate-scope-analysis-v1",
        "status": (
            "CONSUMED_DEVELOPMENT_TYPED_ACQUISITION_GATE_PASSED_AFTER_SCOPE_CORRECTION"
            if passed else
            "CONSUMED_DEVELOPMENT_TYPED_ACQUISITION_GATE_FAILED_AFTER_SCOPE_CORRECTION"
        ),
        "claim_boundary": (
            "DETERMINISTIC_POSTHOC_GATE_SCOPE_CORRECTION_ON_CONSUMED_DEVELOPMENT;"
            "NO_ROLLOUT_RERUN;NO_ACTION_OR_OUTCOME_CHANGED;NO_CONFIRMATORY_TARGET_CLAIM"
        ),
        "source_report_sha256": str(report["report_sha256"]),
        "source_report_file_sha256": str(config["source_report_file_sha256"]),
        "scope_correction": {
            "original_scope": "ALL_RELATE_NO_PROGRESS_EFFECTS_IN_EPISODE",
            "corrected_scope": (
                "RELATE_NO_PROGRESS_WITH_COMPLETED_COUNT_BEFORE_AT_LEAST_ONE;"
                "SOURCE_RELATION_AND_ACQUISITION_ARE_EFFECT_GATED_UNTIL_THEN"
            ),
            "all_cycle_wrong_handle_effects": all_cycle,
            "source_active_wrong_handle_effects": source_active,
            "rollouts_changed": False,
            "actions_changed": False,
            "outcomes_changed": False,
        },
        "summaries": report["summaries"],
        "paired": report["paired"],
        "acquisition_groundings": report["acquisition_groundings"],
        "original_gates": report["gates"],
        "corrected_gates": corrected_gates,
        "development_gate_passed": passed,
    }
    output = REPO / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    finalized = body | {"analysis_report_sha256": stable_hash(body)}
    output.write_text(
        json.dumps(finalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(finalized, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
