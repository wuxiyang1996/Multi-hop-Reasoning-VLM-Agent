#!/usr/bin/env python3
"""Freeze a post-V74, consumed-development Gemini grounder pilot.

This is an explicitly exploratory grounder replacement study. It selects only
already-consumed V71 development videos and cannot alter the V74 verdict.
"""

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
BASE_CONFIG = REPO_ROOT / "configs/agqa2_source_binding_formal_v15.json"
OUTPUT_SELECTION = REPO_ROOT / "configs/agqa2_gemini_grounder_v16_development_selection.json"
OUTPUT_MANIFEST = REPO_ROOT / "configs/agqa2_gemini_grounder_v16_development_manifest.json"
OUTPUT_CONFIG = REPO_ROOT / "configs/agqa2_gemini_grounder_v16_development.json"
COUNT = 16
NONCE = "post-v74-independent-gemini-grounder-v16-development"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(path: Path, hash_field: str) -> dict:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = body.pop(hash_field)
    if stable_hash(body) != claimed:
        raise ValueError(f"content hash mismatch: {path}")
    return value


def main() -> None:
    for path in (OUTPUT_SELECTION, OUTPUT_MANIFEST, OUTPUT_CONFIG):
        if path.exists():
            raise FileExistsError(f"frozen artifact already exists: {path}")
    source_selection = verified(SOURCE_SELECTION, "manifest_sha256")
    source_manifest = verified(SOURCE_MANIFEST, "manifest_sha256")
    ranked = sorted(
        source_selection["samples"],
        key=lambda row: stable_hash({"nonce": NONCE, "task_id": row["task_id"]}),
    )[:COUNT]
    selected_ids = {str(row["task_id"]) for row in ranked}
    selected_manifest_rows = [
        row for row in source_manifest["samples"]
        if str(row["task_id"]) in selected_ids
    ]
    if len(ranked) != COUNT or len(selected_manifest_rows) != COUNT:
        raise ValueError("could not form the frozen development subset")

    selection_body = {
        "schema_version": "agqa2-independent-grounder-development-selection-v1",
        "status": "FROZEN_V75_CONSUMED_DEVELOPMENT_BEFORE_GEMINI_CALLS",
        "split": "development",
        "selection_nonce": NONCE,
        "selection_rule": "HASH_RANK_WITHIN_ALREADY_CONSUMED_V71_DEVELOPMENT",
        "sample_count": COUNT,
        "unique_video_count": len({row["video_id"] for row in ranked}),
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
        "schema_version": "agqa2-independent-grounder-development-manifest-v1",
        "status": "FROZEN_V75_CONSUMED_DEVELOPMENT_MEDIA_BEFORE_GEMINI_CALLS",
        "split": "development",
        "selection_manifest_sha256": selection["manifest_sha256"],
        "archive_path": source_manifest["archive_path"],
        "archive_sha256": source_manifest["archive_sha256"],
        "entry": source_manifest["entry"],
        "sample_count": COUNT,
        "unique_video_count": len({row["video_id"] for row in selected_manifest_rows}),
        "answer_read_during_activation": False,
        "program_read_during_activation": False,
        "scene_graph_read_during_activation": False,
        "samples": selected_manifest_rows,
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = deepcopy(json.loads(BASE_CONFIG.read_text()))
    for key in (
        "formal_protocol", "formal_protocol_file_sha256",
        "development_qualification_report", "development_qualification_file_sha256",
        "expected_grounder_sha256", "expected_evaluation_protocol_sha256",
    ):
        config.pop(key, None)
    config.update({
        "schema_version": "agqa2-independent-grounder-development-config-v1",
        "status": "FROZEN_V75_POST_V74_EXPLORATORY_GEMINI_DEVELOPMENT",
        "split": "development",
        "report_version": "GEMINI31_V75_DEV",
        "claim_boundary": (
            "16_ALREADY_CONSUMED_V71_DEVELOPMENT_VIDEOS;POST_V74_EXPLORATORY_"
            "OFF_THE_SHELF_GROUNDER_REPLACEMENT;CANNOT_CHANGE_V74_VERDICT"
        ),
        "preregistration": str(OUTPUT_SELECTION.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": file_sha256(OUTPUT_SELECTION),
        "expected_preregistration_status": selection["status"],
        "manifest": str(OUTPUT_MANIFEST.relative_to(REPO_ROOT)),
        "manifest_file_sha256": file_sha256(OUTPUT_MANIFEST),
        "expected_manifest_status": manifest["status"],
        "qualification_gates": {
            "required_valid_runtime_rows": COUNT,
            "minimum_route_correct": COUNT,
            "minimum_decisive_executions": 8,
            "minimum_decisive_accuracy": 0.65,
            "maximum_typed_vs_direct_losses": 1,
            "minimum_typed_vs_direct_wins": 3,
            "required_source_permuted_abstentions": COUNT,
            "required_target_written_equivalent_matches": COUNT,
            "maximum_reported_provider_cost_usd": 2.0
        },
    })
    config["model"]["id"] = "google/gemini-3.1-pro-preview"
    OUTPUT_CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": config["status"],
        "sample_count": COUNT,
        "selection_file_sha256": file_sha256(OUTPUT_SELECTION),
        "manifest_file_sha256": file_sha256(OUTPUT_MANIFEST),
        "config_file_sha256": file_sha256(OUTPUT_CONFIG),
        "provider_calls_before_freeze": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
