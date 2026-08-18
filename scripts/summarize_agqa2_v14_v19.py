#!/usr/bin/env python3
"""Build the immutable V14--V19 AGQA transfer audit summary."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


REPORTS = {
    "v13_reserve": "runs/agqa2_active_grounding_v13_reserve/report.json",
    "v15_development": "runs/agqa2_active_grounding_v15_development/report.json",
    "v15_replication": "runs/agqa2_active_grounding_v15_replication/report.json",
    "v16_development": "runs/agqa2_active_grounding_v16_development/report.json",
    "v16_reserve": "runs/agqa2_active_grounding_v16_reserve/report.json",
    "v17_powered_reserve": "runs/agqa2_active_grounding_v17_powered_reserve/report.json",
    "v18_development": "runs/agqa2_override_adjudicator_v18_development/report.json",
    "v19_development": "runs/agqa2_temporal_selective_v19_development/report.json",
    "v19_reserve": "runs/agqa2_temporal_selective_v19_reserve/report.json",
}

FRESH_REPLICATIONS = (
    "v15_replication",
    "v16_reserve",
    "v17_powered_reserve",
    "v19_reserve",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified(relative_path: str) -> tuple[Path, dict[str, Any]]:
    path = REPO_ROOT / relative_path
    report = json.loads(path.read_text())
    body = deepcopy(report)
    claimed = str(body.pop("report_sha256"))
    actual = stable_hash(body)
    if actual != claimed:
        raise ValueError(f"report hash mismatch for {relative_path}: {claimed} != {actual}")
    return path, report


def _row_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    return {
        "status": report["status"],
        "qualified": bool(report.get("grounder_qualified")),
        "rows": int(metrics.get("valid_runtime_rows", report.get("sample_count", 0))),
        "route_correct": metrics.get("route_correct"),
        "decisive_executions": metrics.get("decisive_executions"),
        "decisive_correct": metrics.get("decisive_correct"),
        "direct_correct": metrics.get("direct_correct"),
        "typed_fallback_correct": metrics.get("typed_fallback_correct"),
        "wins": metrics.get("typed_vs_direct_wins"),
        "losses": metrics.get("typed_vs_direct_losses"),
        "unified_harness_correct": metrics.get("unified_harness_correct"),
        "unified_harness_authorizations": metrics.get(
            "unified_harness_executor_authorizations"
        ),
        "provider_cost_usd": report.get("reported_provider_cost_usd"),
        "report_path": str(path.relative_to(REPO_ROOT)),
        "report_file_sha256": _file_sha256(path),
        "report_sha256": report["report_sha256"],
    }


def main() -> None:
    loaded = {name: _load_verified(path) for name, path in REPORTS.items()}
    experiments = {
        name: _row_summary(path, report)
        for name, (path, report) in loaded.items()
    }

    fresh = [loaded[name][1] for name in FRESH_REPLICATIONS]
    fresh_metrics = [report["metrics"] for report in fresh]
    pooled = {
        "replications": list(FRESH_REPLICATIONS),
        "all_replications_qualified": all(
            report["grounder_qualified"] for report in fresh
        ),
        "rows": sum(int(row["valid_runtime_rows"]) for row in fresh_metrics),
        "route_correct": sum(int(row["route_correct"]) for row in fresh_metrics),
        "decisive_executions": sum(
            int(row["decisive_executions"]) for row in fresh_metrics
        ),
        "decisive_correct": sum(int(row["decisive_correct"]) for row in fresh_metrics),
        "direct_correct": sum(int(row["direct_correct"]) for row in fresh_metrics),
        "typed_fallback_correct": sum(
            int(row["typed_fallback_correct"]) for row in fresh_metrics
        ),
        "counterfactual_typed_gain": sum(
            int(row["typed_fallback_correct"]) - int(row["direct_correct"])
            for row in fresh_metrics
        ),
        "wins": sum(int(row["typed_vs_direct_wins"]) for row in fresh_metrics),
        "losses": sum(int(row["typed_vs_direct_losses"]) for row in fresh_metrics),
        "unified_harness_correct": sum(
            int(row["unified_harness_correct"]) for row in fresh_metrics
        ),
        "unified_harness_authorizations": sum(
            int(row["unified_harness_executor_authorizations"])
            for row in fresh_metrics
        ),
        "source_permuted_abstentions": sum(
            int(report["controls"]["source_permuted_abstentions"])
            for report in fresh
        ),
        "target_written_equivalent_matches": sum(
            int(report["controls"]["target_written_equivalent_matches"])
            for report in fresh
        ),
        "provider_cost_usd": sum(
            float(report["reported_provider_cost_usd"]) for report in fresh
        ),
        "interpretation": (
            "AVERAGE_COUNTERFACTUAL_GAIN_WITH_NONZERO_NEGATIVE_TRANSFER;"
            "STRICT_SELECTIVE_TRANSFER_NOT_QUALIFIED;FAIL_CLOSED_HARNESS_HAS_NO_GAIN"
        ),
    }

    v19 = loaded["v19_reserve"][1]
    v19_changed_rows = []
    for row in v19["rows"]:
        if row["typed_fallback_correct"] == row["direct_correct"]:
            continue
        v19_changed_rows.append({
            "task_id": row["task_id"],
            "video_id": row["video_id"],
            "comparison": row["query_plan"]["comparison"],
            "direct_correct": row["direct_correct"],
            "typed_fallback_correct": row["typed_fallback_correct"],
            "typed_prediction": row["typed_fallback_prediction"],
            "gold_answer_evaluator_only": row["gold_answer_evaluator_only"],
        })

    body = {
        "schema_version": "agqa2-active-neurosymbolic-transfer-v14-v19-audit-v1",
        "status": "PIPELINE_EXECUTION_VALIDATED_TRANSFER_UTILITY_NOT_QUALIFIED",
        "claim_boundary": (
            "AGQA_TYPED_RUNTIME_AND_FAIL_CLOSED_UNIFIED_HARNESS_WORK_END_TO_END;"
            "GAME_TO_AGQA_SELECTIVE_SUCCESS_RATE_TRANSFER_IS_NOT_VALIDATED"
        ),
        "experiments": experiments,
        "pooled_fresh_v15_v19": pooled,
        "v19_final_changed_rows": v19_changed_rows,
        "query_object_status": (
            "NOT_STARTED_BY_PREREGISTERED_STOP_POLICY_BECAUSE_BASE_TEMPORAL_"
            "GROUNDER_DID_NOT_QUALIFY"
        ),
        "recommended_next_step": (
            "TRAIN_OR_CALIBRATE_A_TARGET_NATIVE_AGQA_ONTOLOGY_GROUNDER_ON_"
            "VIDEO_DISJOINT_TRAIN_OR_DEVELOPMENT_DATA;COMPARE_TO_A_MATCHED_"
            "STRONG_TARGET_ONLY_BASELINE;FREEZE_AND_RUN_ONE_FRESH_CONFIRMATION"
        ),
    }
    result = body | {"summary_sha256": stable_hash(body)}
    output = REPO_ROOT / "docs/results/agqa2_active_grounding_v14_v19_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output.relative_to(REPO_ROOT)),
        "status": result["status"],
        "pooled": pooled,
        "summary_sha256": result["summary_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
