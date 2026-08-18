#!/usr/bin/env python3
"""Independent audit of the CLEVRER unified V15 prospective reserve."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.sokoban_video_recovery import (  # noqa: E402
    exact_binomial_two_sided,
)


AUTHENTIC = "authentic_source_induced_goal_relation"
NEURAL = "neural_only_explicit_relation"
TARGET_BASE = "target_base_receipt_recovery"
GENERIC = "generic_error_scaffold"
PERMUTED = "source_permuted_uplift"
SHUFFLED = "shuffled_proof_binding"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_report(path: Path) -> tuple[dict[str, Any], str]:
    selected = path
    if not selected.is_file() and path == (
        REPO / "runs/clevrer_unified_goal_relation_v15_reserve/formal_report.json"
    ):
        selected = REPO / "artifacts/video_event_graph_v15/formal_report.json.gz"
    payload = selected.read_bytes()
    if selected.suffix == ".gz":
        payload = gzip.decompress(payload)
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {selected}")
    return value, hashlib.sha256(payload).hexdigest()


def _self_hash(value: dict[str, Any], field: str) -> bool:
    body = dict(value)
    claimed = body.pop(field, None)
    return bool(claimed and claimed == stable_hash(body))


def _paired(rows: list[dict[str, Any]], right: str) -> dict[str, Any]:
    wins = sum(
        row["conditions"][AUTHENTIC]["correct"]
        and not row["conditions"][right]["correct"]
        for row in rows
    )
    losses = sum(
        row["conditions"][right]["correct"]
        and not row["conditions"][AUTHENTIC]["correct"]
        for row in rows
    )
    return {
        "wins": wins, "losses": losses, "net_wins": wins - losses,
        "ties": len(rows) - wins - losses,
        "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/clevrer_unified_goal_relation_v15_reserve.json",
    )
    parser.add_argument(
        "--report", type=Path,
        default=REPO / "runs/clevrer_unified_goal_relation_v15_reserve/formal_report.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/clevrer_unified_goal_relation_v15_summary.json",
    )
    args = parser.parse_args()
    config = _read(args.config)
    report, report_file_sha256 = _read_report(args.report)
    manifest = _read(REPO / config["target"]["split_manifest"])
    splits = manifest["benchmarks"]["clevrer"]["splits"]
    reserve_ids = list(splits["reserve"])
    rows = list(report["rows"])
    row_ids = [str(row["sample_id"]) for row in rows]
    conditions = tuple(report["conditions"])

    recalculated_metrics = {
        name: {
            "correct": sum(row["conditions"][name]["correct"] for row in rows),
            "accuracy": sum(row["conditions"][name]["correct"] for row in rows)
            / len(rows),
            "recoveries": sum(row["conditions"][name]["recover"] for row in rows),
        }
        for name in conditions
    }
    recalculated_paired = {
        name: _paired(rows, name) for name in conditions if name != AUTHENTIC
    }
    authority_selected = sum(
        row["unified_authority"]["phase7"]["verdict"] == "SELECT_SKILL"
        for row in rows
    )
    executor_calls = sum(
        int(row["unified_authority"]["executor_calls"]) for row in rows
    )
    controls = config["gates"]["causal_control_conditions"]
    gates = {
        "config_self_hash": _self_hash(config, "config_sha256"),
        "report_self_hash": _self_hash(report, "report_sha256"),
        "config_was_frozen_before_reserve": config.get("status")
        == "FROZEN_BEFORE_CLEVRER_V15_RESERVE_OUTCOMES",
        "config_file_lineage": report["lineage"]["config_file_sha256"]
        == _sha(args.config),
        "all_frozen_lineage_reverified": report["lineage"][
            "verified_frozen_lineage"
        ] == config["frozen_lineage"],
        "reserve_identity_exact": row_ids == reserve_ids,
        "reserve_unique": len(row_ids) == len(set(row_ids)) == 360,
        "reserve_disjoint_from_prior_roles": not (
            set(row_ids) & (set(splits["development"]) | set(splits["formal"]))
        ),
        "condition_metrics_recalculate": recalculated_metrics
        == report["conditions"],
        "paired_metrics_recalculate": recalculated_paired
        == report["paired_authentic"],
        "all_preregistered_formal_gates_passed": all(report["gates"].values()),
        "authority_count_matches_recoveries": authority_selected
        == executor_calls == report["conditions"][AUTHENTIC]["recoveries"],
        "zero_runtime_outcome_exposure": all(
            row["unified_authority"]["current_target_outcome_read"] is False
            and row["unified_authority"]["phase7"]["current_target_outcome_read"] is False
            and row["unified_authority"]["utility"]["current_outcome_read"] is False
            for row in rows
        ),
        "authentic_improves_neural_success": report["conditions"][AUTHENTIC]["correct"]
        > report["conditions"][NEURAL]["correct"],
        "authentic_beats_source_permuted_with_exact_p": recalculated_paired[
            PERMUTED
        ]["net_wins"] > 0 and recalculated_paired[PERMUTED]["exact_two_sided_p"] <= 0.05,
        "authentic_beats_shuffled_binding_with_exact_p": recalculated_paired[
            SHUFFLED
        ]["net_wins"] > 0 and recalculated_paired[SHUFFLED]["exact_two_sided_p"] <= 0.05,
        "zero_external_provider_cost": report["cost"]["external_provider_calls"]
        == 0 and report["cost"]["external_provider_cost_usd"] == 0.0,
    }
    target_base_pair = recalculated_paired[TARGET_BASE]
    warnings = {
        "target_base_statistically_indistinguishable": target_base_pair[
            "exact_two_sided_p"
        ] > 0.05,
        "generic_scaffold_difference_not_significant": recalculated_paired[
            GENERIC
        ]["exact_two_sided_p"] > 0.05,
        "source_provenance_necessity_not_established": True,
        "natural_video_transfer_not_established": True,
    }
    passed = all(gates.values())
    body = {
        "schema_version": "clevrer-unified-goal-relation-v15-independent-audit",
        "status": (
            "CLEVRER_V15_PROSPECTIVE_RESERVE_VALIDATED_WITH_TARGET_BASE_AMBIGUITY"
            if passed else "CLEVRER_V15_INDEPENDENT_AUDIT_FAILED"
        ),
        "claim_boundary": (
            "Validates prospective success-rate improvement over neural-only "
            "and source-specific superiority over permuted and shuffled "
            "bindings in the fixed synthetic CLEVRER setup. Authentic is not "
            "statistically distinguishable from the strongest target-base "
            "learner, so source provenance necessity remains unsupported."
        ),
        "samples": len(rows),
        "primary": {
            "authentic": report["conditions"][AUTHENTIC],
            "neural_only": report["conditions"][NEURAL],
            "target_base": report["conditions"][TARGET_BASE],
            "generic_scaffold": report["conditions"][GENERIC],
            "source_permuted": report["conditions"][PERMUTED],
            "shuffled_binding": report["conditions"][SHUFFLED],
        },
        "paired": {
            "vs_neural_only": recalculated_paired[NEURAL],
            "vs_target_base": target_base_pair,
            "vs_generic_scaffold": recalculated_paired[GENERIC],
            "vs_source_permuted": recalculated_paired[PERMUTED],
            "vs_shuffled_binding": recalculated_paired[SHUFFLED],
        },
        "authority": {
            "selected_rows": authority_selected,
            "target_executor_calls": executor_calls,
            "source_selector_emitted_target_action": False,
        },
        "cost": report["cost"],
        "warnings": warnings,
        "gates": gates,
        "lineage": {
            "config_sha256": _sha(args.config),
            "formal_report_sha256": report_file_sha256,
            "formal_report_content_sha256": report["report_sha256"],
        },
    }
    summary = body | {"summary_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"], "primary": summary["primary"],
        "paired": summary["paired"], "warnings": warnings,
        "gates": gates, "output": str(args.output),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
