#!/usr/bin/env python3
"""Induce V48 temporal-support reliability from 500 consumed target rows."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_aggregate_temporal_transfer import (  # noqa: E402
    bind_aggregate_temporal_pair_program,
)
from motif_transfer.agqa_interval_reliability_calibrator import (  # noqa: E402
    binding_geometry,
)
from motif_transfer.agqa_temporal_support_calibrator import (  # noqa: E402
    TemporalSupportExample,
    induce_temporal_support_rule,
    maximum_interval_span,
)
from motif_transfer.agqa_view_reliability_calibrator import (  # noqa: E402
    singleton_view_kind,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


INPUTS = (
    ("v38_consumed_development", "runs/agqa2_aggregate_temporal_v38_development/report.json", "runs/agqa2_robust_temporal_v36_development/base_report.json"),
    ("v40_failed_formal_consumed_training", "runs/agqa2_aggregate_temporal_v41_completion/report.json", "runs/agqa2_aggregate_temporal_v41_completion/base_report.json"),
    ("v44_failed_qualification_consumed_training", "runs/agqa2_view_reliability_v44_qualification/report.json", "runs/agqa2_view_reliability_v43_qualification/base_report.json"),
    ("v47_failed_qualification_consumed_training", "runs/agqa2_interval_reliability_v47_qualification/report.json", "runs/agqa2_interval_reliability_v46_qualification/base_report.json"),
)
OUTPUT = "configs/agqa2_temporal_support_v48/training_artifact.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path) -> dict:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed:
        raise ValueError(f"report hash mismatch: {path}")
    return value


def _evaluate(rule, rows):
    allowed = set(rule.allowed_singleton_views)
    selected = [
        row for row in rows
        if row.aggregate_authorized
        and (row.singleton_view is None or row.singleton_view in allowed)
        and row.minimum_cross_pair_gap >= rule.minimum_cross_pair_gap
        and row.maximum_within_operand_endpoint_spread
        <= rule.maximum_within_operand_endpoint_spread
        and row.maximum_interval_span >= rule.minimum_max_interval_span
    ]
    wins = sum(row.source_correct and not row.target_native_correct for row in selected)
    losses = sum(row.target_native_correct and not row.source_correct for row in selected)
    return {"authorizations": len(selected), "wins": wins, "losses": losses,
            "net_gain": wins - losses}


def main() -> None:
    examples, lineage = [], []
    for split, report_name, base_name in INPUTS:
        report_path, base_path = REPO_ROOT / report_name, REPO_ROOT / base_name
        report, base = _verified(report_path), _verified(base_path)
        base_rows = {str(row["task_id"]): row for row in base["rows"]}
        for outcome in report["rows_detail"]:
            row = base_rows[str(outcome["task_id"])]
            binding = bind_aggregate_temporal_pair_program(
                task_id=str(row["task_id"]),
                target_state_sha256=str(row["runtime_receipt_sha256"]),
                target_grounder_sha256=str(report["target_grounder_sha256"]),
                source_program_sha256=str(report["source_program_sha256"]),
                obligation_kind=str(row["query_plan"]["obligation_kind"]),
                operand_runs=row["operand_runs"], grounder_qualified=True,
                formal_outcome_read=False,
            )
            gap, spread = binding_geometry(binding)
            examples.append(TemporalSupportExample(
                split=split, task_id=str(row["task_id"]),
                aggregate_authorized=binding.authorized_relation is not None,
                singleton_view=singleton_view_kind(binding),
                minimum_cross_pair_gap=gap,
                maximum_within_operand_endpoint_spread=spread,
                maximum_interval_span=maximum_interval_span(binding),
                source_correct=bool(outcome["source_correct"]),
                target_native_correct=bool(outcome["target_native_correct"]),
            ))
        lineage.append({
            "split": split, "report": report_name,
            "report_sha256": report["report_sha256"],
            "report_file_sha256": _sha256(report_path),
            "base_report": base_name, "base_report_sha256": base["report_sha256"],
            "base_report_file_sha256": _sha256(base_path),
            "rows": len(report["rows_detail"]),
        })
    rule, candidates = induce_temporal_support_rule(examples)
    cross_validation = []
    for held_out in [row[0] for row in INPUTS]:
        training = [row for row in examples if row.split != held_out]
        testing = [row for row in examples if row.split == held_out]
        fold_rule, _ = induce_temporal_support_rule(training)
        cross_validation.append({
            "held_out_split": held_out,
            "selected_rule": asdict(fold_rule),
            "training_metrics": _evaluate(fold_rule, training),
            "held_out_metrics": _evaluate(fold_rule, testing),
        })
    cv_wins = sum(row["held_out_metrics"]["wins"] for row in cross_validation)
    cv_losses = sum(row["held_out_metrics"]["losses"] for row in cross_validation)
    core = {
        "schema_version": "agqa2-temporal-support-training-artifact-v48",
        "status": "V48_TRAINED_ON_500_CONSUMED_ROWS_BEFORE_NEW_QUALIFICATION",
        "training_authority": (
            "CONSUMED_V38_V40_V44_AND_FAILED_V47_ROWS_ONLY;NO_FUTURE_"
            "QUALIFICATION_OR_FORMAL_DATA"
        ),
        "source_program_or_ir_changed": False,
        "target_interval_or_relation_changed": False,
        "runtime_authority": "ABSTENTION_ONLY;CANNOT_INVENT_OR_EDIT_A_BINDING",
        "feature_space": [
            "SINGLETON_VIEW_KIND", "MINIMUM_CROSS_PAIR_GAP",
            "MAXIMUM_WITHIN_OPERAND_ENDPOINT_SPREAD", "MAXIMUM_INTERVAL_SPAN",
        ],
        "finite_candidate_count": len(candidates),
        "selection_objective": (
            "MINIMIZE_OBSERVED_NEGATIVE_TRANSFER_THEN_MAXIMIZE_NET_GAIN_"
            "WINS_AND_COVERAGE_WITH_FIXED_MDL_TIE_BREAK"
        ),
        "rule": asdict(rule),
        "candidate_rule_table": list(candidates),
        "leave_one_experiment_out": cross_validation,
        "leave_one_experiment_out_totals": {
            "wins": cv_wins, "losses": cv_losses,
            "net_gain": cv_wins - cv_losses,
        },
        "training_lineage": lineage,
        "training_example_count": len(examples),
        "future_policy": (
            "REQUIRE_ONE_NEW_VIDEO_DISJOINT_TRAIN_QUALIFICATION_BEFORE_"
            "ANY_NEW_TEST_FORMAL"
        ),
        "confirmatory_claim": False,
    }
    artifact = core | {"artifact_sha256": stable_hash(core)}
    output = REPO_ROOT / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": artifact["status"],
        "training_example_count": len(examples),
        "finite_candidate_count": len(candidates),
        "selected_rule": artifact["rule"],
        "leave_one_experiment_out_totals": artifact[
            "leave_one_experiment_out_totals"
        ],
        "artifact_sha256": artifact["artifact_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
