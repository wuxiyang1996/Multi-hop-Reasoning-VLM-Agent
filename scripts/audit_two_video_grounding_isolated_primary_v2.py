#!/usr/bin/env python3
"""Consolidate the grounding-isolated primary and raw-video diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_two_video_grounding_isolated_primary_v1 import audit as audit_primary  # noqa: E402

RAW_V1_AUDIT = REPO / "docs/results/agqa2_router_v65_formal_v1_audit.json"
RAW_V2_PROTOCOL = REPO / "configs/agqa2_multiclass_router_v65_formal_v2_protocol.json"
RAW_V2_REPORT = REPO / "runs/agqa2_multiclass_router_v65_formal_v2/base_report.json"
RAW_V2_EVALUATION = REPO / "runs/agqa2_multiclass_router_v65_formal_v2/formal_evaluation.json"
V74_FINAL = REPO / "docs/results/agqa2_qwen235_source_transfer_v65_v74_final_summary.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified(path: Path, key: str) -> dict:
    value = json.loads(path.read_text())
    body = dict(value); claimed = body.pop(key)
    if stable_hash(body) != claimed:
        raise ValueError(f"content hash mismatch: {path}")
    return value


def _one_sided_exact(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    return sum(comb(discordant, k) for k in range(wins, discordant + 1)) / (2 ** discordant)


def audit() -> dict:
    primary = audit_primary()
    raw_v1 = _verified(RAW_V1_AUDIT, "audit_sha256")
    raw_v2_protocol = _verified(RAW_V2_PROTOCOL, "protocol_sha256")
    raw_v2_report = _verified(RAW_V2_REPORT, "report_sha256")
    raw_v2 = _verified(RAW_V2_EVALUATION, "evaluation_sha256")
    v74 = json.loads(V74_FINAL.read_text())
    route_correct = int(raw_v2_report["metrics"]["route_correct"])
    route_total = int(raw_v2_report["metrics"]["valid_runtime_rows"])
    agqa_pair = primary["agqa2"]["source_vs_generic"]
    clevrer_pair = primary["clevrer"]["source_vs_neural"]
    gates = {
        "grounding_isolated_primary_recomputes_and_passes": (
            primary["status"] == "PASSED" and all(primary["gates"].values())
        ),
        "agqa_primary_uses_one_shared_official_stsg_backend": (
            primary["grounding_policy"]["same_grounding_receipt_or_backend_for_all_matched_arms"]
            and primary["agqa2"]["grounding_induced_source_losses"] == 0
        ),
        "clevrer_primary_uses_one_shared_receipt_per_task": (
            primary["grounding_policy"]["same_grounding_receipt_or_backend_for_all_matched_arms"]
            and primary["clevrer"]["grounding_induced_source_losses_vs_neural"] == 0
        ),
        "no_answer_or_functional_program_available_to_primary_runtime": (
            primary["grounding_policy"]["answers_or_official_functional_programs_available_to_runtime"] is False
        ),
        "raw_qwen_v1_retained_as_failed_secondary": (
            raw_v1["gates"]["success_gain_gate_passed"] is False
            and raw_v1["gates"]["negative_transfer_gate_passed"] is False
        ),
        "raw_qwen_v2_retained_as_failed_secondary": (
            raw_v2["status"] == "FAILED"
            and raw_v2["gates"]["selection_runtime_oracle_routes_agree"] is False
            and raw_v2["source_authorizations"] == 0
        ),
        "v2_route_failure_measured_after_receipt_freeze": (
            route_correct == 78 and route_total == 91
            and raw_v2["gates"]["runtime_blindness"] is True
        ),
        "post_v74_runs_cannot_overturn_negative_raw_video_verdict": (
            v74["conclusion"]["failure_policy"]
            == "NO_FURTHER_AGQA_ADAPTATION_ON_CONSUMED_COHORTS"
        ),
        "primary_paired_gains_are_significant": (
            _one_sided_exact(agqa_pair["wins"], agqa_pair["losses"]) <= 0.05
            and _one_sided_exact(clevrer_pair["wins"], clevrer_pair["losses"]) <= 0.05
        ),
    }
    body = {
        "schema_version": "two-video-grounding-isolated-primary-audit-v2",
        "status": "PASSED" if all(gates.values()) else "FAILED",
        "paper_primary_estimand": (
            "PAIRED_CONTROLLER_SKILL_TRANSFER_CONDITIONAL_ON_SHARED_"
            "BENCHMARK_NATIVE_STRUCTURED_GROUNDING"
        ),
        "claim_boundary": (
            "GROUNDING_CANNOT_CAUSE_THE_MATCHED_ARM_DELTA;ABSOLUTE_ACCURACY_"
            "REMAINS_CONDITIONAL_ON_TARGET_NATIVE_STRUCTURED_GROUNDING;"
            "NO_RAW_PIXEL_VIDEO_QA_OR_SOTA_CLAIM"
        ),
        "primary": {
            "agqa2": primary["agqa2"],
            "clevrer": primary["clevrer"],
            "grounding_policy": primary["grounding_policy"],
            "audit_sha256": primary["report_sha256"],
            "paired_significance": {
                "agqa2_source_vs_generic_one_sided_exact_pvalue": _one_sided_exact(
                    agqa_pair["wins"], agqa_pair["losses"]
                ),
                "clevrer_source_vs_neural_one_sided_exact_pvalue": _one_sided_exact(
                    clevrer_pair["wins"], clevrer_pair["losses"]
                ),
            },
        },
        "secondary_raw_video": {
            "agqa2_qwen235_v1": {
                "status": raw_v1["status"],
                "source_vs_neural": raw_v1["formal_result"]["source_vs_neural_only"],
                "note": "POSITIVE_POINT_ESTIMATE_BUT_FAILED_PREREGISTERED_GATES",
            },
            "agqa2_qwen235_multiclass_v2": {
                "status": "POST_V74_EXPLORATORY_FAILED",
                "protocol_status": raw_v2_protocol["status"],
                "route_correct": route_correct,
                "route_total": route_total,
                "route_accuracy": raw_v2_report["metrics"]["route_accuracy"],
                "source_authorizations": raw_v2["source_authorizations"],
                "arm_correct": raw_v2["arm_correct"],
                "provider_cost_usd": raw_v2["reported_provider_cost_usd"],
                "failure": "QUESTION_ONLY_ROUTE_SHIFT_TRIGGERED_GLOBAL_FAIL_CLOSED",
            },
            "role": "ROBUSTNESS_DIAGNOSTIC_ONLY_NOT_PAPER_PRIMARY",
        },
        "gates": gates,
        "lineage": {
            "primary_v1_audit_file_sha256": _sha(
                REPO / "docs/results/two_video_grounding_isolated_primary_v1.json"
            ),
            "raw_v1_audit_file_sha256": _sha(RAW_V1_AUDIT),
            "raw_v2_protocol_file_sha256": _sha(RAW_V2_PROTOCOL),
            "raw_v2_report_file_sha256": _sha(RAW_V2_REPORT),
            "raw_v2_evaluation_file_sha256": _sha(RAW_V2_EVALUATION),
            "v74_final_summary_file_sha256": _sha(V74_FINAL),
        },
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
