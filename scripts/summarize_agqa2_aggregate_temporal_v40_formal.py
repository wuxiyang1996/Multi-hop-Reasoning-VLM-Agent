#!/usr/bin/env python3
"""Write the compact, failure-preserving V40 formal summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, field: str) -> dict:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = body.pop(field)
    if stable_hash(body) != claimed:
        raise ValueError(f"hash mismatch: {path}")
    return value


def main() -> None:
    report_path = REPO_ROOT / (
        "runs/agqa2_aggregate_temporal_v41_completion/report.json"
    )
    report = _verified(report_path, "report_sha256")
    if (
        report["status"]
        != "AGQA2_AGGREGATE_TEMPORAL_V40_FORMAL_NOT_QUALIFIED"
        or all(report["qualification_gates"].values())
    ):
        raise ValueError("V40 is not the expected failed formal endpoint")
    abort_path = REPO_ROOT / (
        "docs/results/agqa2_aggregate_temporal_v40_runtime_abort.json"
    )
    abort = _verified(abort_path, "result_sha256")
    completion_config = REPO_ROOT / (
        "configs/agqa2_aggregate_temporal_v41_completion.json"
    )
    rows = report["rows_detail"]
    strict = []
    for row in rows:
        authorized = (
            row["source_executor_authorized"]
            and row["operand_a_hypothesis_count"] >= 2
            and row["operand_b_hypothesis_count"] >= 2
        )
        source_correct = (
            row["source_correct"] if authorized
            else row["target_native_correct"]
        )
        strict.append((source_correct, row["target_native_correct"], authorized))
    strict_wins = sum(a and not b for a, b, _ in strict)
    strict_losses = sum(b and not a for a, b, _ in strict)
    core = {
        "schema_version": "agqa2-aggregate-temporal-v40-formal-summary-v1",
        "status": report["status"],
        "confirmatory_claim": False,
        "rows": report["rows"],
        "unique_video_count": report["unique_video_count"],
        "source_program_sha256": report["source_program_sha256"],
        "target_grounder_sha256": report["target_grounder_sha256"],
        "target_executor_sha256": report["target_executor_sha256"],
        "source_executor_authorizations": report[
            "source_executor_authorizations"
        ],
        "source_vs_target_native": report["source_vs_target_native"],
        "effect_shuffled_abstentions": report[
            "effect_shuffled_abstentions"
        ],
        "wrong_source_abstentions": report["wrong_source_abstentions"],
        "source_vs_generic_scaffold": report[
            "source_vs_generic_scaffold"
        ],
        "source_vs_target_written_equivalent": report[
            "source_vs_target_written_equivalent"
        ],
        "qualification_gates": report["qualification_gates"],
        "provider_calls": report["provider_calls"],
        "reported_provider_cost_usd": report[
            "reported_provider_cost_usd"
        ],
        "formal_report_sha256": report["report_sha256"],
        "formal_report_file_sha256": _sha256(report_path),
        "completion_integrity": {
            "v40_runtime_abort_result_sha256": abort["result_sha256"],
            "all_runtime_receipts_frozen_before_outcome_access": True,
            "formal_metrics_externalized_before_schema_repair": False,
            "schema_alias_only": True,
            "method_gate_sample_or_prediction_change": False,
            "completion_config_file_sha256": _sha256(completion_config),
        },
        "posthoc_failure_localization_not_a_claim": {
            "original_two_views_per_operand_authorizations": sum(
                c for _, _, c in strict
            ),
            "original_two_views_per_operand_wins": strict_wins,
            "original_two_views_per_operand_losses": strict_losses,
            "aggregate_only_incremental_wins": (
                report["source_vs_target_native"]["wins"] - strict_wins
            ),
            "aggregate_only_incremental_losses": (
                report["source_vs_target_native"]["losses"] - strict_losses
            ),
        },
        "claim_boundary": {
            "agqa_query_object_v32_remains_confirmed": True,
            "agqa_before_after_temporal_transfer_confirmed": False,
            "second_source_program_family_success_validated": False,
            "full_agqa_solved": False,
            "source_beats_generic_scaffold": False,
            "source_provenance_is_necessary": False,
        },
    }
    summary = core | {"summary_sha256": stable_hash(core)}
    output = REPO_ROOT / (
        "docs/results/agqa2_aggregate_temporal_v40_formal_summary.json"
    )
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
