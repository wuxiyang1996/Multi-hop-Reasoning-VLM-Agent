#!/usr/bin/env python3
"""Audit the sealed V65-grounder/router AGQA formal run and remaining pool."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import joblib


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
from motif_transfer.agqa_active_frame_grounder import parse_public_question_plan  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, key: str) -> dict:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = body.pop(key)
    if stable_hash(body) != claimed:
        raise ValueError(f"content hash mismatch: {path}")
    return value


def _runtime_videos() -> tuple[set[str], dict[str, set[str]]]:
    by_run: dict[str, set[str]] = {}
    for directory in REPO_ROOT.glob("runs/agqa2*"):
        ids = set()
        for path in directory.glob("runtime_receipts/*.json"):
            try:
                row = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(row.get("video_id"), str):
                ids.add(row["video_id"])
        if ids:
            by_run[directory.name] = ids
    return set().union(*by_run.values()), by_run


def audit() -> dict:
    protocol_path = REPO_ROOT / "configs/agqa2_router_v65_grounder_formal_v1_protocol.json"
    selection_path = REPO_ROOT / "configs/agqa2_router_heldout_formal_v1_selection.json"
    manifest_path = REPO_ROOT / "configs/agqa2_router_v65_grounder_formal_v1_manifest.json"
    split_path = REPO_ROOT / "configs/agqa2_program_router_video_split_v1.json"
    router_report_path = REPO_ROOT / "runs/agqa2_program_router_v1/qualification_report_v2.json"
    router_model_path = REPO_ROOT / "runs/agqa2_program_router_v1/router.joblib"
    report_path = REPO_ROOT / "runs/agqa2_router_v65_grounder_formal_v1/base_report.json"
    evaluation_path = REPO_ROOT / "runs/agqa2_router_v65_grounder_formal_v1/formal_evaluation.json"
    quarantine_path = REPO_ROOT / "runs/agqa2_router_v65_grounder_formal_v1/invalid_cache_quarantine_v1.json"
    transport_path = REPO_ROOT / "runs/agqa2_router_v65_grounder_formal_v1/postruntime_missing_program_hash_transport_fix.json"

    protocol = _verified(protocol_path, "protocol_sha256")
    selection = _verified(selection_path, "manifest_sha256")
    manifest = _verified(manifest_path, "manifest_sha256")
    report = _verified(report_path, "report_sha256")
    evaluation = _verified(evaluation_path, "evaluation_sha256")
    quarantine = _verified(quarantine_path, "receipt_sha256")
    transport = json.loads(transport_path.read_text())
    split = json.loads(split_path.read_text())
    router_report = json.loads(router_report_path.read_text())
    router = joblib.load(router_model_path)

    formal_videos = set(split["partitions"]["formal_holdout"])
    runtime_videos, by_run = _runtime_videos()
    unseen = formal_videos - runtime_videos
    threshold = float(router_report["validation"]["selection"]["threshold"])
    unseen_exists_candidates = []
    with zipfile.ZipFile(split["archive_path"]) as bundle, bundle.open(split["entry"]) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            video_id = str(row["video_id"])
            if video_id not in unseen:
                continue
            question = str(row["question"])
            plan = parse_public_question_plan(question)
            if plan is None or plan.comparison != "EXISTS":
                continue
            score = float(router.predict_proba([question])[0, 1])
            if score >= threshold:
                unseen_exists_candidates.append({
                    "task_id": task_id, "video_id": video_id,
                    "question_sha256": stable_hash(question), "router_score": score,
                })
    formal_by_run = {
        name: len(ids & formal_videos) for name, ids in by_run.items()
        if ids & formal_videos
    }
    paired_losses = [
        row for row in report["rows"]
        if row["direct_correct"] and not row["unified_harness_correct"]
    ]
    paired_wins = [
        row for row in report["rows"]
        if row["unified_harness_correct"] and not row["direct_correct"]
    ]
    loss_false_negatives = sum(
        str(row["unified_harness_prediction"]).casefold().startswith("no")
        and str(row["gold_answer_evaluator_only"]).casefold() == "yes"
        for row in paired_losses
    )
    loss_false_positives = len(paired_losses) - loss_false_negatives
    gates = {
        "protocol_was_frozen": protocol["status"] == "FROZEN_BEFORE_ANY_FORMAL_PROVIDER_OR_OUTCOME_ACCESS",
        "fresh_80_video_cohort": evaluation["gates"]["fresh_video_heldout_cohort"],
        "v65_runtime_pinned": evaluation["gates"]["v65_runtime_pinned"],
        "router_and_controls_passed": all(evaluation["gates"][key] for key in (
            "all_routes_correct", "applicability_coverage",
            "source_permuted_abstains", "source_permuted_equals_neural_only",
            "target_written_equivalent_matches_source", "runtime_blindness",
        )),
        "cost_gate_passed": evaluation["gates"]["cost_within_cap"],
        "negative_transfer_gate_passed": evaluation["gates"]["negative_transfer_bound"],
        "success_gain_gate_passed": evaluation["gates"]["success_gain"],
        "transport_fixes_outcome_neutral": (
            quarantine["target_outcome_read"] is False
            and quarantine["request_input_or_prompt_changed"] is False
            and transport["provider_or_grounding_semantics_changed"] is False
            and transport["formal_prediction_or_gate_semantics_changed"] is False
        ),
        "second_untouched_compatible_cohort_available": bool(unseen_exists_candidates),
    }
    status = (
        "PASSED" if all(gates.values())
        else "V1_CONFIRMATORY_FAILED_AND_NO_SECOND_UNTOUCHED_COMPATIBLE_COHORT"
    )
    body = {
        "schema_version": "agqa2-router-v65-formal-v1-audit-v1",
        "status": status,
        "formal_result": {
            "sample_count": evaluation["sample_count"],
            "unique_video_count": evaluation["unique_video_count"],
            "source_authorizations": evaluation["source_authorizations"],
            "arm_correct": evaluation["arm_correct"],
            "source_vs_neural_only": evaluation["source_vs_neural_only"],
            "reported_provider_cost_usd": evaluation["reported_provider_cost_usd"],
            "failure_decomposition": {
                "paired_wins": len(paired_wins),
                "paired_losses": len(paired_losses),
                "grounder_false_negative_losses": loss_false_negatives,
                "grounder_false_positive_losses": loss_false_positives,
                "all_losses_are_visual_grounding_decisions": all(
                    row["target_native_execution"]["decision"] is not None
                    for row in paired_losses
                ),
            },
        },
        "remaining_inventory": {
            "formal_partition_videos": len(formal_videos),
            "currently_runtime_exposed_formal_videos": len(formal_videos & runtime_videos),
            "currently_unseen_formal_videos": len(unseen),
            "currently_unseen_router_qualified_exists_tasks": len(unseen_exists_candidates),
            "formal_exposure_by_run": dict(sorted(formal_by_run.items())),
            "untouched_program_or_answer_access_during_inventory_scan": False,
        },
        "transport": {
            "invalid_cache_quarantined": quarantine["quarantined_count"],
            "invalid_cache_receipt_sha256": quarantine["receipt_sha256"],
            "postruntime_finalizer_receipt_sha256": transport["receipt_sha256"],
            "accepted_call_reported_cost_usd": report["reported_provider_cost_usd"],
            "malformed_provider_attempt_cost_caveat": "NOT_PRESENT_IN_ACCEPTED_CALL_CACHE;PRIMARY_GATE_WAS_PREREGISTERED_ON_REPORTED_PROVIDER_COST",
        },
        "gates": gates,
        "claim": (
            "POSITIVE_POINT_ESTIMATE_53_VS_45_AND_CONTROLS_VALID;"
            "PREREGISTERED_SUCCESS_AND_NEGATIVE_TRANSFER_GATES_FAILED;"
            "NO_POSITIVE_CONFIRMATORY_AGQA_CLAIM"
        ),
        "lineage": {
            "protocol_file_sha256": _sha256(protocol_path),
            "selection_file_sha256": _sha256(selection_path),
            "manifest_file_sha256": _sha256(manifest_path),
            "collector_report_file_sha256": _sha256(report_path),
            "formal_evaluation_file_sha256": _sha256(evaluation_path),
            "router_model_file_sha256": _sha256(router_model_path),
        },
    }
    return body | {"audit_sha256": stable_hash(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "docs/results/agqa2_router_v65_formal_v1_audit.json",
    )
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
