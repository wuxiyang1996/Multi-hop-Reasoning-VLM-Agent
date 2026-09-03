#!/usr/bin/env python3
"""Freeze a pilot-disjoint Gemini grounder qualification after V75C."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
from motif_transfer.contracts import stable_hash  # noqa: E402


SOURCE_SELECTION = REPO_ROOT / "configs/agqa2_source_triggered_powered_qualification_v12_selection.json"
SOURCE_MANIFEST = REPO_ROOT / "configs/agqa2_source_triggered_powered_qualification_v12_manifest.json"
PILOT_SELECTION = REPO_ROOT / "configs/agqa2_gemini_grounder_v16_development_selection.json"
PILOT_REPORT = REPO_ROOT / "runs/agqa2_gemini_grounder_v16c_development/report.json"
BASE_CONFIG = REPO_ROOT / "configs/agqa2_gemini_grounder_v16c_development.json"
OUTPUT_SELECTION = REPO_ROOT / "configs/agqa2_gemini_grounder_v17_qualification_selection.json"
OUTPUT_MANIFEST = REPO_ROOT / "configs/agqa2_gemini_grounder_v17_qualification_manifest.json"
OUTPUT_PROTOCOL = REPO_ROOT / "configs/agqa2_gemini_grounder_v17_qualification_protocol.json"
OUTPUT_CONFIG = REPO_ROOT / "configs/agqa2_gemini_grounder_v17_qualification.json"
EVALUATOR = REPO_ROOT / "scripts/evaluate_agqa2_source_executor_v13.py"
COUNT = 48
NONCE = "post-v74-gemini-grounder-pilot-disjoint-v17-qualification"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(path: Path, hash_field: str) -> dict:
    value = json.loads(path.read_text())
    body = dict(value); claimed = body.pop(hash_field)
    if stable_hash(body) != claimed:
        raise ValueError(f"content hash mismatch: {path}")
    return value


def main() -> None:
    for path in (OUTPUT_SELECTION, OUTPUT_MANIFEST, OUTPUT_PROTOCOL, OUTPUT_CONFIG):
        if path.exists():
            raise FileExistsError(f"frozen artifact already exists: {path}")
    source_selection = verified(SOURCE_SELECTION, "manifest_sha256")
    source_manifest = verified(SOURCE_MANIFEST, "manifest_sha256")
    pilot_selection = verified(PILOT_SELECTION, "manifest_sha256")
    pilot_report = verified(PILOT_REPORT, "report_sha256")
    if not pilot_report.get("grounder_qualified"):
        raise ValueError("Gemini pilot did not qualify")
    excluded = {str(row["task_id"]) for row in pilot_selection["samples"]}
    pool = [row for row in source_selection["samples"] if str(row["task_id"]) not in excluded]
    ranked = sorted(pool, key=lambda row: stable_hash({"nonce": NONCE, "task_id": row["task_id"]}))[:COUNT]
    selected_ids = {str(row["task_id"]) for row in ranked}
    manifest_rows = [row for row in source_manifest["samples"] if str(row["task_id"]) in selected_ids]
    if len(ranked) != COUNT or len(manifest_rows) != COUNT:
        raise ValueError("could not form pilot-disjoint qualification")

    selection_body = {
        "schema_version": "agqa2-independent-grounder-qualification-selection-v1",
        "status": "FROZEN_V76_PILOT_DISJOINT_QUALIFICATION_BEFORE_PROVIDER_OR_OUTCOME_ACCESS",
        "split": "development",
        "selection_nonce": NONCE,
        "selection_rule": "HASH_RANK_WITHIN_CONSUMED_V71_DEVELOPMENT_EXCLUDING_V75C",
        "sample_count": COUNT,
        "unique_video_count": len({row["video_id"] for row in ranked}),
        "excluded_pilot_task_count": len(excluded),
        "archive_path": source_selection["archive_path"],
        "archive_sha256": source_selection["archive_sha256"],
        "entry": source_selection["entry"],
        "raw_video_archive": source_selection["raw_video_archive"],
        "answer_read_during_selection": False,
        "program_read_during_selection": False,
        "scene_graph_read_during_selection": False,
        "source_identity_read_during_selection": False,
        "post_v74_exploratory": True,
        "samples": ranked,
    }
    selection = selection_body | {"manifest_sha256": stable_hash(selection_body)}
    OUTPUT_SELECTION.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")

    manifest_body = {
        "schema_version": "agqa2-independent-grounder-qualification-manifest-v1",
        "status": "FROZEN_V76_PILOT_DISJOINT_MEDIA_BEFORE_PROVIDER_OR_OUTCOME_ACCESS",
        "split": "development",
        "selection_manifest_sha256": selection["manifest_sha256"],
        "archive_path": source_manifest["archive_path"],
        "archive_sha256": source_manifest["archive_sha256"],
        "entry": source_manifest["entry"],
        "sample_count": COUNT,
        "unique_video_count": len({row["video_id"] for row in manifest_rows}),
        "answer_read_during_activation": False,
        "program_read_during_activation": False,
        "scene_graph_read_during_activation": False,
        "samples": manifest_rows,
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    gate = {
        "minimum_wins": 12,
        "maximum_losses": 4,
        "minimum_net_gain": 8,
        "maximum_one_sided_exact_pvalue": 0.05,
        "minimum_route_accuracy": 1.0,
        "minimum_source_permuted_abstention_rate": 1.0,
        "minimum_target_written_equivalent_rate": 1.0,
        "maximum_cost_usd": 3.5,
    }
    protocol = {
        "schema_version": "agqa2-independent-grounder-qualification-protocol-v1",
        "status": "FROZEN_BEFORE_V76_PROVIDER_OR_OUTCOME_ACCESS",
        "claim_boundary": "48_PILOT_DISJOINT_CONSUMED_DEVELOPMENT_VIDEOS;POST_V74_EXPLORATORY_GROUNDER_QUALIFICATION",
        "sample_count": COUNT,
        "applicability_rule": "ALL_DECISIVE_TYPED_EXECUTIONS",
        "grounder_model": "google/gemini-3.1-pro-preview",
        "grounder_sha256": pilot_report["grounder_sha256"],
        "pilot_report_file_sha256": file_sha256(PILOT_REPORT),
        "evaluator_file_sha256": file_sha256(EVALUATOR),
        "qualification_gate": gate,
        "failure_policy": "IF_ANY_GATE_FAILS_DO_NOT_OPEN_THE_96_VIDEO_EXPLORATORY_RESERVE",
        "multiple_testing_disclosure": "POST_V74_NEW_GROUNDER_HYPOTHESIS;CANNOT_REPLACE_OR_OVERTURN_THE_V74_NEGATIVE_RESULT",
    }
    OUTPUT_PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    config = deepcopy(json.loads(BASE_CONFIG.read_text()))
    config.update({
        "status": "FROZEN_V76_GEMINI_GROUNDER_QUALIFICATION",
        "report_version": "GEMINI31_V76_QUAL",
        "claim_boundary": protocol["claim_boundary"],
        "preregistration": str(OUTPUT_SELECTION.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": file_sha256(OUTPUT_SELECTION),
        "expected_preregistration_status": selection["status"],
        "manifest": str(OUTPUT_MANIFEST.relative_to(REPO_ROOT)),
        "manifest_file_sha256": file_sha256(OUTPUT_MANIFEST),
        "expected_manifest_status": manifest["status"],
        "expected_grounder_sha256": pilot_report["grounder_sha256"],
        "formal_protocol": str(OUTPUT_PROTOCOL.relative_to(REPO_ROOT)),
        "formal_protocol_file_sha256": file_sha256(OUTPUT_PROTOCOL),
        "qualification_gates": {
            "required_valid_runtime_rows": COUNT,
            "minimum_route_correct": COUNT,
            "minimum_decisive_executions": 12,
            "minimum_decisive_accuracy": 0.65,
            "maximum_typed_vs_direct_losses": gate["maximum_losses"],
            "minimum_typed_vs_direct_wins": gate["minimum_wins"],
            "required_source_permuted_abstentions": COUNT,
            "required_target_written_equivalent_matches": COUNT,
            "maximum_reported_provider_cost_usd": gate["maximum_cost_usd"],
        },
    })
    OUTPUT_CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": config["status"], "sample_count": COUNT,
        "selection_file_sha256": file_sha256(OUTPUT_SELECTION),
        "manifest_file_sha256": file_sha256(OUTPUT_MANIFEST),
        "protocol_file_sha256": file_sha256(OUTPUT_PROTOCOL),
        "config_file_sha256": file_sha256(OUTPUT_CONFIG),
        "provider_calls_before_freeze": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
