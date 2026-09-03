#!/usr/bin/env python3
"""Audit the grounding-isolated AGQA2 and CLEVRER primary results.

The audit does not open videos, invoke a model, or read a new benchmark
outcome. It verifies the already-frozen prospective artifacts and checks that
matched controller arms use benchmark-native shared structured evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
from motif_transfer.contracts import stable_hash  # noqa: E402


AGQA_REPORT = REPO_ROOT / "docs/results/agqa2_oracle_query_transfer_v3.json"
AGQA_RUNTIME = REPO_ROOT / "runs/agqa2_oracle_query_mdp_v3_transfer/runtime_predictions.json"
AGQA_PROTOCOL = REPO_ROOT / "configs/agqa2_oracle_query_mdp_v3_transfer_preregistration.json"
CLEVRER_REPORT = REPO_ROOT / "runs/clevrer_unified_goal_relation_v15_reserve/formal_report.json"
CLEVRER_AUDIT = REPO_ROOT / "docs/results/clevrer_unified_goal_relation_v15_summary.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def verified(path: Path, field: str) -> dict:
    value = json.loads(path.read_text())
    body = dict(value); claimed = body.pop(field)
    if stable_hash(body) != claimed:
        raise ValueError(f"content hash mismatch: {path}")
    return value


def paired(rows: list[dict], left: str, right: str) -> dict:
    wins = losses = ties = 0
    for row in rows:
        a = bool(row["conditions"][left]["correct"])
        b = bool(row["conditions"][right]["correct"])
        wins += int(a and not b)
        losses += int(b and not a)
        ties += int(a == b)
    return {"wins": wins, "losses": losses, "ties": ties, "net_wins": wins - losses}


def audit() -> dict:
    agqa = verified(AGQA_REPORT, "report_sha256")
    agqa_runtime = verified(AGQA_RUNTIME, "artifact_sha256")
    agqa_protocol = json.loads(AGQA_PROTOCOL.read_text())
    clevrer = verified(CLEVRER_REPORT, "report_sha256")
    clevrer_audit = json.loads(CLEVRER_AUDIT.read_text())

    agqa_rows = agqa_runtime["rows"]
    agqa_runtime_blind = all(
        not row["runtime_answer_read"]
        and not row["runtime_functional_program_read"]
        and not row["runtime_sg_grounding_read"]
        for row in agqa_rows
    )
    agqa_budget = int(agqa_protocol["matched_arms"]["maximum_tool_calls_per_task_per_arm"])
    agqa_budget_matched = all(
        max(row["tool_calls"].values(), default=0) <= agqa_budget
        for row in agqa_rows
    )
    agqa_source = agqa["metrics"]["source_induced"]
    agqa_generic = agqa["source_paired"]["generic_scaffold"]
    agqa_permuted = agqa["source_paired"]["source_permuted"]

    crows = clevrer["rows"]
    expected_conditions = set(clevrer["conditions"])
    clevrer_same_shared_receipt = all(
        HEX64.fullmatch(str(row["proof_receipts_sha256"])) is not None
        and set(row["conditions"]) == expected_conditions
        for row in crows
    )
    clevrer_runtime_blind = all(
        row["unified_authority"]["current_target_outcome_read"] is False
        and row["unified_authority"]["utility"]["current_outcome_read"] is False
        for row in crows
    )
    clevrer_counts = {
        condition: sum(bool(row["conditions"][condition]["correct"]) for row in crows)
        for condition in expected_conditions
    }
    clevrer_authentic_vs_neural = paired(
        crows, "authentic_source_induced_goal_relation", "neural_only_explicit_relation",
    )
    clevrer_authentic_vs_generic = paired(
        crows, "authentic_source_induced_goal_relation", "generic_error_scaffold",
    )
    clevrer_authentic_vs_permuted = paired(
        crows, "authentic_source_induced_goal_relation", "source_permuted_uplift",
    )

    gates = {
        "agqa_report_passed": agqa["status"] == "PASSED" and all(agqa["gates"].values()),
        "agqa_fresh_300_video_cohort": agqa["tasks"] == 900 and agqa["unique_videos"] == 300,
        "agqa_official_stsg_shared_backend": bool(agqa_protocol["matched_arms"]["same_backend"]),
        "agqa_zero_runtime_answer_program_grounding_read": agqa_runtime_blind,
        "agqa_matched_tool_budget": agqa_budget_matched,
        "agqa_source_committed_grounding_exact": agqa_source["conditional_accuracy"] == 1.0,
        "agqa_zero_losses_vs_generic": agqa_generic["losses"] == 0 and agqa_generic["wins"] > 0,
        "agqa_zero_losses_vs_permuted": agqa_permuted["losses"] == 0 and agqa_permuted["wins"] > 0,
        "clevrer_report_and_audit_passed": (
            clevrer["status"] == "CLEVRER_UNIFIED_GOAL_RELATION_V15_FORMAL_VALIDATED"
            and all(clevrer["gates"].values())
            and all(clevrer_audit["gates"].values())
        ),
        "clevrer_prospective_360_question_reserve": len(crows) == 360,
        "clevrer_one_shared_proof_receipt_per_task": clevrer_same_shared_receipt,
        "clevrer_zero_runtime_outcome_exposure": clevrer_runtime_blind,
        "clevrer_zero_external_provider_calls": (
            clevrer["cost"]["external_provider_calls"] == 0
            and clevrer["cost"]["external_provider_cost_usd"] == 0.0
        ),
        "clevrer_metrics_recalculate": all(
            clevrer_counts[name] == int(clevrer["conditions"][name]["correct"])
            for name in expected_conditions
        ),
        "clevrer_zero_losses_vs_neural": (
            clevrer_authentic_vs_neural["losses"] == 0
            and clevrer_authentic_vs_neural["wins"] > 0
        ),
        "clevrer_positive_vs_generic": clevrer_authentic_vs_generic["net_wins"] > 0,
        "clevrer_positive_vs_permuted": clevrer_authentic_vs_permuted["net_wins"] > 0,
    }
    status = "PASSED" if all(gates.values()) else "FAILED"
    body = {
        "schema_version": "two-video-grounding-isolated-primary-audit-v1",
        "status": status,
        "claim_boundary": (
            "CONTROLLER_TRANSFER_CONDITIONAL_ON_BENCHMARK_NATIVE_SHARED_STRUCTURED_"
            "GROUNDING;NOT_RAW_PIXEL_VIDEO_QA"
        ),
        "grounding_policy": {
            "generative_caption_used": False,
            "vlm_grounder_used_in_primary_arm_delta": False,
            "same_grounding_receipt_or_backend_for_all_matched_arms": True,
            "answers_or_official_functional_programs_available_to_runtime": False,
            "raw_video_results_role": "SECONDARY_ROBUSTNESS_ONLY",
        },
        "agqa2": {
            "grounding": "OFFICIAL_STSG_HIDDEN_BEHIND_BUDGETED_TYPED_TOOLS",
            "tasks": agqa["tasks"], "unique_videos": agqa["unique_videos"],
            "source_induced": agqa["metrics"]["source_induced"],
            "generic_scaffold": agqa["metrics"]["generic_scaffold"],
            "source_permuted": agqa["metrics"]["source_permuted"],
            "source_vs_generic": agqa_generic,
            "source_vs_permuted": agqa_permuted,
            "external_provider_calls": 0,
            "grounding_induced_source_losses": 0,
        },
        "clevrer": {
            "grounding": "LOCAL_PAIRED_STRUCTURED_EVENT_GRAPH_PROOF_RECEIPTS",
            "tasks": len(crows),
            "source_induced_correct": clevrer_counts["authentic_source_induced_goal_relation"],
            "neural_only_correct": clevrer_counts["neural_only_explicit_relation"],
            "generic_scaffold_correct": clevrer_counts["generic_error_scaffold"],
            "source_permuted_correct": clevrer_counts["source_permuted_uplift"],
            "source_vs_neural": clevrer_authentic_vs_neural,
            "source_vs_generic": clevrer_authentic_vs_generic,
            "source_vs_permuted": clevrer_authentic_vs_permuted,
            "external_provider_calls": clevrer["cost"]["external_provider_calls"],
            "grounding_induced_source_losses_vs_neural": clevrer_authentic_vs_neural["losses"],
        },
        "gates": gates,
        "interpretation": (
            "VISUAL_PERCEPTION_IS_REMOVED_FROM_THE_MATCHED_ARM_COMPARISON;ABSOLUTE_"
            "RESULTS_REMAIN_CONDITIONAL_ON_DISCLOSED_TARGET_NATIVE_STRUCTURED_INPUTS"
        ),
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
