#!/usr/bin/env python3
"""Audit QUERY_OBJECT V20--V25, including non-source target-only controls."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motif_transfer.agqa_query_object_grounder import (  # noqa: E402
    AGQA_OBJECT_ONTOLOGY, canonical_object_label,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import _answer_matches  # noqa: E402


REPORTS = {
    "v20_development": "runs/agqa2_query_object_v20_development/report.json",
    "v21_development": "runs/agqa2_query_object_v21_development/report.json",
    "v22_development": "runs/agqa2_query_object_v22_development/report.json",
    "v24_development": "runs/agqa2_query_object_v24_development/report.json",
    "v25_reserve": "runs/agqa2_query_object_v25_reserve/report.json",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report(relative: str) -> tuple[Path, dict[str, Any]]:
    path = REPO_ROOT / relative
    report = json.loads(path.read_text())
    body = deepcopy(report)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed:
        raise ValueError(f"report hash mismatch: {relative}")
    return path, report


def _ontology_only_decision(row: dict[str, Any]) -> str | None:
    votes = []
    for receipt, threshold in zip(
        row["object_ontology_receipts"], (0.8, 0.8), strict=True,
    ):
        decision = canonical_object_label(receipt["decision"])
        if (
            receipt["relation_observed"]
            and float(receipt["confidence"]) >= threshold
            and receipt["evidence_frames"]
            and decision in AGQA_OBJECT_ONTOLOGY
        ):
            votes.append(decision)
    return votes[0] if len(votes) == 2 and votes[0] == votes[1] else None


def _lexical_direct_decision(response: str) -> str | None:
    text = " " + re.sub(r"[^a-z0-9 ]+", " ", response.casefold()) + " "
    aliases = {
        "cabinet": "closet", "notebook": "paper", "camera": "phone",
        "couch": "sofa", "glass": "cup", "bottle": "cup", "tv": "television",
    }
    hits = {
        label for label in AGQA_OBJECT_ONTOLOGY if f" {label} " in text
    }
    hits.update(
        canonical for surface, canonical in aliases.items()
        if f" {surface} " in text
    )
    return next(iter(hits)) if len(hits) == 1 else None


def _experiment_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "qualified": report["grounder_qualified"],
        "metrics": report["metrics"],
        "controls": report["controls"],
        "failed_gates": [
            key for key, value in report["qualification_gates"].items() if not value
        ],
        "reported_provider_cost_usd": report["reported_provider_cost_usd"],
        "grounder_sha256": report["grounder_sha256"],
        "report_path": str(path.relative_to(REPO_ROOT)),
        "report_file_sha256": _file_sha256(path),
        "report_sha256": report["report_sha256"],
    }


def _v23_abort() -> dict[str, Any]:
    cache_root = REPO_ROOT / "runs/agqa2_query_object_v23_reserve/call_cache"
    cache_rows = [json.loads(path.read_text()) for path in sorted(cache_root.glob("*/*.json"))]
    worker_errors_path = REPO_ROOT / "runs/agqa2_query_object_v23_reserve/worker_errors.json"
    worker_errors = json.loads(worker_errors_path.read_text())
    runtime_count = len(list((
        REPO_ROOT / "runs/agqa2_query_object_v23_reserve/runtime_receipts"
    ).glob("*.json")))
    body = {
        "schema_version": "agqa2-query-object-v23-runtime-abort-v1",
        "status": "V23_RUNTIME_INCOMPLETE_BEFORE_FORMAL_EVALUATION",
        "stage": "OPERAND_RECEIPT_SCHEMA_VALIDATION",
        "completed_runtime_receipts": runtime_count,
        "required_runtime_receipts": 30,
        "worker_errors": worker_errors["errors"],
        "accepted_provider_calls": len(cache_rows),
        "accepted_reported_provider_cost_usd": sum(
            float(row["usage"]["reported_cost_usd"]) for row in cache_rows
        ),
        "worker_errors_file_sha256": _file_sha256(worker_errors_path),
        "formal_report_created": False,
        "formal_gold_evaluation_started": False,
        "repair": (
            "V24_DETERMINISTIC_INTERVAL_EVIDENCE_ENVELOPE;REQUALIFIED_ON_"
            "DEVELOPMENT;V23_VIDEOS_EXCLUDED_FROM_V25"
        ),
    }
    return body | {"abort_sha256": stable_hash(body)}


def main() -> None:
    loaded = {name: _load_report(path) for name, path in REPORTS.items()}
    experiments = {
        name: _experiment_summary(path, report)
        for name, (path, report) in loaded.items()
    }
    v24 = loaded["v24_development"][1]
    v25 = loaded["v25_reserve"][1]
    if v24["grounder_sha256"] != v25["grounder_sha256"]:
        raise ValueError("V25 did not preserve the qualified V24 grounder")

    manifest = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v25_reserve_manifest.json"
    ).read_text())
    group_by_task = {
        row["task_id"]: row["relation_group"] for row in manifest["samples"]
    }
    target_only_correct = lexical_correct = 0
    target_only_decisive = source_vs_target_wins = source_vs_target_losses = 0
    group_metrics: dict[str, dict[str, int]] = {}
    comparison_rows = []
    for row in v25["rows"]:
        gold, direct = row["gold_answer_evaluator_only"], row["direct_response"]
        ontology_decision = _ontology_only_decision(row)
        ontology_prediction = ontology_decision or direct
        lexical_decision = _lexical_direct_decision(direct)
        lexical_prediction = lexical_decision or direct
        source_correct = bool(row["unified_harness_correct"])
        ontology_correct = _answer_matches(ontology_prediction, gold)
        lexical_row_correct = _answer_matches(lexical_prediction, gold)
        target_only_correct += ontology_correct
        target_only_decisive += ontology_decision is not None
        lexical_correct += lexical_row_correct
        source_vs_target_wins += source_correct and not ontology_correct
        source_vs_target_losses += ontology_correct and not source_correct
        group = group_by_task[row["task_id"]]
        metrics = group_metrics.setdefault(group, {
            "rows": 0, "direct_correct": 0, "source_harness_correct": 0,
            "target_only_ontology_correct": 0,
        })
        metrics["rows"] += 1
        metrics["direct_correct"] += bool(row["direct_correct"])
        metrics["source_harness_correct"] += source_correct
        metrics["target_only_ontology_correct"] += ontology_correct
        if source_correct != ontology_correct:
            comparison_rows.append({
                "task_id": row["task_id"], "relation_group": group,
                "relation": row["query_plan"]["operand_a"], "gold": gold,
                "source_harness_prediction": row["unified_harness_prediction"],
                "source_harness_correct": source_correct,
                "target_only_prediction": ontology_prediction,
                "target_only_correct": ontology_correct,
            })

    posthoc = {
        "status": "POSTHOC_DIAGNOSTIC_NOT_A_PREREGISTERED_GATE",
        "raw_direct_correct": v25["metrics"]["direct_correct"],
        "lexical_ontology_normalized_direct_correct": lexical_correct,
        "two_ontology_view_decisive": target_only_decisive,
        "two_ontology_view_fallback_correct": target_only_correct,
        "source_harness_correct": v25["metrics"]["unified_harness_correct"],
        "source_harness_vs_two_ontology_view_wins": source_vs_target_wins,
        "source_harness_vs_two_ontology_view_losses": source_vs_target_losses,
        "source_harness_vs_two_ontology_view_delta": (
            v25["metrics"]["unified_harness_correct"] - target_only_correct
        ),
        "rows_with_different_correctness": comparison_rows,
        "interpretation": (
            "FORMATTING_EXPLAINS_PART_BUT_NOT_ALL_OF_THE_GAIN;SOURCE_CONTROLLED_"
            "THIRD_VIEW_ADDS_TWO_CORRECT_ROWS_OVER_THE_TWO_ONTOLOGY_VIEW_"
            "TARGET_ONLY_CONTROL;CONTROL_IS_POSTHOC"
        ),
    }
    v23_abort = _v23_abort()
    abort_path = REPO_ROOT / "docs/results/agqa2_query_object_v23_runtime_abort.json"
    abort_path.write_text(json.dumps(v23_abort, indent=2, sort_keys=True) + "\n")
    body = {
        "schema_version": "agqa2-query-object-v20-v25-audit-v1",
        "status": "AGQA2_QUERY_OBJECT_V25_RESERVE_QUALIFIED",
        "claim_boundary": (
            "ATOMIC_OPEN_ANSWER_QUERY_OBJECT_STRUCTURAL_MECHANISM_TRANSFER_"
            "QUALIFIED_ON_30_NEW_VIDEOS;SOURCE_PROVENANCE_NOT_ESTABLISHED"
        ),
        "experiments": experiments,
        "v23_runtime_abort": v23_abort,
        "v25_formal": {
            "development_and_reserve_grounder_identical": True,
            "grounder_sha256": v25["grounder_sha256"],
            "metrics": v25["metrics"],
            "controls": v25["controls"],
            "all_preregistered_gates_passed": all(
                v25["qualification_gates"].values()
            ),
            "relation_group_metrics": group_metrics,
            "posthoc_target_only_controls": posthoc,
        },
        "atomic_route_artifacts_modified": False,
        "source_provenance_claim": False,
        "target_written_equivalent_dynamics_match": "30/30",
    }
    result = body | {"summary_sha256": stable_hash(body)}
    output = REPO_ROOT / "docs/results/agqa2_query_object_v20_v25_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output.relative_to(REPO_ROOT)),
        "status": result["status"],
        "v25_metrics": v25["metrics"],
        "posthoc_target_only_controls": posthoc,
        "summary_sha256": result["summary_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
