#!/usr/bin/env python3
"""Induce V45 interval reliability from 350 consumed target rows."""

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
    IntervalReliabilityExample,
    binding_geometry,
    induce_interval_reliability_rule,
)
from motif_transfer.agqa_view_reliability_calibrator import (  # noqa: E402
    singleton_view_kind,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


INPUTS = (
    (
        "v38_consumed_development",
        "runs/agqa2_aggregate_temporal_v38_development/report.json",
        "runs/agqa2_robust_temporal_v36_development/base_report.json",
    ),
    (
        "v40_failed_formal_consumed_training",
        "runs/agqa2_aggregate_temporal_v41_completion/report.json",
        "runs/agqa2_aggregate_temporal_v41_completion/base_report.json",
    ),
    (
        "v44_failed_qualification_consumed_training",
        "runs/agqa2_view_reliability_v44_qualification/report.json",
        "runs/agqa2_view_reliability_v43_qualification/base_report.json",
    ),
)
OUTPUT = "configs/agqa2_interval_reliability_v45/training_artifact.json"


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
            examples.append(IntervalReliabilityExample(
                task_id=str(row["task_id"]),
                aggregate_authorized=binding.authorized_relation is not None,
                singleton_view=singleton_view_kind(binding),
                minimum_cross_pair_gap=gap,
                maximum_within_operand_endpoint_spread=spread,
                source_correct=bool(outcome["source_correct"]),
                target_native_correct=bool(outcome["target_native_correct"]),
            ))
        lineage.append({
            "split": split, "report": report_name,
            "report_sha256": report["report_sha256"],
            "report_file_sha256": _sha256(report_path),
            "base_report": base_name,
            "base_report_sha256": base["report_sha256"],
            "base_report_file_sha256": _sha256(base_path),
            "rows": len(report["rows_detail"]),
        })
    rule, candidates = induce_interval_reliability_rule(examples)
    core = {
        "schema_version": "agqa2-interval-reliability-training-artifact-v45",
        "status": "V45_TRAINED_ON_350_CONSUMED_ROWS_BEFORE_NEW_QUALIFICATION",
        "training_authority": (
            "CONSUMED_V38_V40_AND_FAILED_V44_ROWS_ONLY;NO_FUTURE_"
            "QUALIFICATION_OR_FORMAL_DATA"
        ),
        "source_program_or_ir_changed": False,
        "target_interval_or_relation_changed": False,
        "runtime_authority": "ABSTENTION_ONLY;CANNOT_INVENT_A_BINDING",
        "feature_space": [
            "SINGLETON_VIEW_KIND", "MINIMUM_CROSS_PAIR_GAP",
            "MAXIMUM_WITHIN_OPERAND_ENDPOINT_SPREAD",
        ],
        "finite_candidate_count": len(candidates),
        "rule": asdict(rule),
        "candidate_rule_table": list(candidates),
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
        "artifact_sha256": artifact["artifact_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
