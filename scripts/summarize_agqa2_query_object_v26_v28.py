#!/usr/bin/env python3
"""Summarize the preregistered V26--V28 source-specific AGQA study."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_query_object_source_specific import (  # noqa: E402
    exact_one_sided_pvalue,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402


def _report(relative: str) -> tuple[Path, dict[str, Any]]:
    path = REPO_ROOT / relative
    report = json.loads(path.read_text())
    body = deepcopy(report)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed:
        raise ValueError(f"report hash mismatch: {relative}")
    return path, report


def _candidate_audit(report: dict, manifest: dict) -> dict[str, Any]:
    groups = {row["task_id"]: row["relation_group"] for row in manifest["samples"]}
    wins = losses = 0
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    discordant = []
    for row in report["rows"]:
        source_correct = bool(row["typed_fallback_correct"])
        target_correct = bool(row["target_only_correct"])
        group = groups[row["task_id"]]
        metrics = by_group[group]
        metrics["rows"] += 1
        metrics["direct_correct"] += bool(row["direct_correct"])
        metrics["source_candidate_correct"] += source_correct
        metrics["target_only_correct"] += target_correct
        metrics["source_decisive"] += bool(row["decisive_execution"])
        metrics["source_decisive_correct"] += (
            bool(row["decisive_correct"]) if row["decisive_execution"] else 0
        )
        outcome = None
        if source_correct and not target_correct:
            wins += 1
            outcome = "SOURCE_WIN"
        elif target_correct and not source_correct:
            losses += 1
            outcome = "SOURCE_LOSS"
        if outcome:
            discordant.append({
                "task_id": row["task_id"],
                "relation_group": group,
                "relation": row["query_plan"]["operand_a"],
                "gold": row["gold_answer_evaluator_only"],
                "source_candidate_prediction": row["typed_fallback_prediction"],
                "target_only_prediction": row["target_only_prediction"],
                "outcome": outcome,
            })
    source_correct = report["metrics"]["typed_fallback_correct"]
    target_correct = report["source_specific_metrics"]["target_only_correct"]
    return {
        "status": "POSTHOC_PREAUTHORIZATION_DIAGNOSTIC_NOT_FORMAL_ENDPOINT",
        "source_candidate_correct": source_correct,
        "target_only_correct": target_correct,
        "source_minus_target_only_correct": source_correct - target_correct,
        "source_vs_target_only_wins": wins,
        "source_vs_target_only_losses": losses,
        "discordant_pairs": wins + losses,
        "exact_one_sided_pvalue": exact_one_sided_pvalue(
            source_wins=wins, source_losses=losses,
        ),
        "passes_original_source_specific_gates": False,
        "relation_group_metrics": {
            key: dict(value) for key, value in sorted(by_group.items())
        },
        "discordant_rows": discordant,
    }


def main() -> None:
    v26_abort = _verified_json(
        REPO_ROOT / "docs/results/agqa2_query_object_v26_runtime_abort.json",
        "abort_sha256",
    )
    v27_abort = _verified_json(
        REPO_ROOT / "docs/results/agqa2_query_object_v27_runtime_abort.json",
        "abort_sha256",
    )
    development_path, development = _report(
        "runs/agqa2_query_object_v28_development/report.json"
    )
    formal_path, formal = _report(
        "runs/agqa2_query_object_v28_reserve/report.json"
    )
    manifest = _verified_json(
        REPO_ROOT / "configs/agqa2_query_object_v28_reserve_manifest.json",
        "manifest_sha256",
    )
    if formal["source_specific_transfer_qualified"]:
        raise ValueError("V28 unexpectedly claims source-specific qualification")
    if any(formal["source_specific_qualification_gates"].values()) and all(
        formal["source_specific_qualification_gates"].values()
    ):
        raise ValueError("V28 gate state is internally inconsistent")
    candidate = _candidate_audit(formal, manifest)
    body = {
        "schema_version": "agqa2-query-object-v26-v28-source-specific-audit-v1",
        "status": "AGQA2_SOURCE_SPECIFIC_TRANSFER_NOT_VALIDATED",
        "claim_boundary": (
            "V25_ATOMIC_STRUCTURAL_MECHANISM_TRANSFER_REMAINS_QUALIFIED;"
            "V28_POWERED_SOURCE_SPECIFIC_CONFIRMATION_FAILED;DO_NOT_CLAIM_"
            "SOURCE_PROVENANCE_OR_FULL_AGQA"
        ),
        "v26_runtime_abort": v26_abort,
        "v27_runtime_abort": v27_abort,
        "v28_development": {
            "status": development["status"],
            "qualified": development["grounder_qualified"],
            "metrics": development["metrics"],
            "controls": development["controls"],
            "reported_provider_cost_usd": development[
                "reported_provider_cost_usd"
            ],
            "grounder_sha256": development["grounder_sha256"],
            "report_path": str(development_path.relative_to(REPO_ROOT)),
            "report_file_sha256": _sha256(development_path),
            "report_sha256": development["report_sha256"],
        },
        "v28_formal": {
            "status": formal["status"],
            "source_specific_transfer_qualified": formal[
                "source_specific_transfer_qualified"
            ],
            "grounder_sha256": formal["grounder_sha256"],
            "metrics": formal["metrics"],
            "controls": formal["controls"],
            "base_mechanism_gates": formal["qualification_gates"],
            "source_specific_metrics": formal["source_specific_metrics"],
            "source_specific_gates": formal[
                "source_specific_qualification_gates"
            ],
            "reported_provider_cost_usd": formal["reported_provider_cost_usd"],
            "report_path": str(formal_path.relative_to(REPO_ROOT)),
            "report_file_sha256": _sha256(formal_path),
            "report_sha256": formal["report_sha256"],
            "posthoc_preauthorization_candidate_audit": candidate,
        },
        "scientific_interpretation": {
            "structural_mechanism_transfer_still_supported_by_v25": True,
            "source_specific_transfer_supported_by_v28": False,
            "target_only_ontology_ensemble_is_a_strong_baseline": True,
            "candidate_source_trend_is_confirmatory": False,
            "candidate_source_gain_concentrated_in_perception": True,
            "another_reserve_seed_without_new_development_hypothesis_allowed": False,
        },
        "source_provenance_claim": False,
        "full_agqa_claim": False,
    }
    result = body | {"summary_sha256": stable_hash(body)}
    output = REPO_ROOT / "docs/results/agqa2_query_object_v26_v28_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output.relative_to(REPO_ROOT)),
        "status": result["status"],
        "formal_status": formal["status"],
        "formal_source_specific_metrics": formal["source_specific_metrics"],
        "posthoc_preauthorization_candidate": candidate,
        "summary_sha256": result["summary_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
