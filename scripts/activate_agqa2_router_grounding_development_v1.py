#!/usr/bin/env python3
"""Activate downloaded V67 router-qualified grounding development data."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.activate_agqa2_qwen235_fresh_reserve_v1 import _sha256, _verified  # noqa: E402


SELECTION = REPO_ROOT / "configs/agqa2_router_grounding_development_v1_selection.json"
DOWNLOAD = REPO_ROOT / "runs/agqa2_router_grounding_development_v1_download/receipt.json"
BASE = REPO_ROOT / "configs/agqa2_qwen235_tool_grounder_v1_development.json"
MANIFEST = REPO_ROOT / "configs/agqa2_router_grounding_development_v1_manifest.json"
CONFIG = REPO_ROOT / "configs/agqa2_router_grounding_development_v1.json"
MANIFEST_STATUS = "FROZEN_V67_DEVELOPMENT_BEFORE_PROVIDER_OR_OUTCOME_ACCESS"
CONFIG_STATUS = "FROZEN_V67_ROUTER_QUALIFIED_GROUNDING_DEVELOPMENT"
REPORT_VERSION = "QWEN235_V67"
CLAIM_BOUNDARY = "80_VIDEO_DISJOINT_ROUTER_VALIDATION_VIDEOS;GROUNDING_DEVELOPMENT_ONLY;NO_FORMAL_CLAIM"
SPLIT = "development"


def main() -> None:
    if MANIFEST.exists() or CONFIG.exists():
        raise FileExistsError("V67 activation artifacts are immutable")
    selection = _verified(SELECTION, "manifest_sha256")
    receipt = json.loads(DOWNLOAD.read_text())
    if receipt["status"] != "COMPLETE" or receipt["selection_manifest_sha256"] != selection["manifest_sha256"]:
        raise ValueError("V67 download receipt mismatch")
    downloaded = {row["video_id"]: row for row in receipt["videos"]}
    samples = []
    for row in selection["samples"]:
        video = downloaded[row["video_id"]]
        path = Path(row["video_path"])
        if _sha256(path) != video["sha256"]:
            raise ValueError(f"video hash mismatch: {path}")
        samples.append(dict(row) | {"video_sha256": video["sha256"], "video_bytes": video["file_size"]})
    manifest_body = {
        "schema_version": "agqa2-router-grounding-development-manifest-v1",
        "status": MANIFEST_STATUS,
        "split": SPLIT,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(DOWNLOAD),
        "archive_path": selection["archive_path"], "archive_sha256": selection["archive_sha256"], "entry": selection["entry"],
        "sample_count": len(samples), "unique_video_count": len(samples),
        "answer_read_during_activation": False, "program_read_during_activation": False, "scene_graph_read_during_activation": False,
        "samples": samples,
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    config = deepcopy(json.loads(BASE.read_text()))
    config["grounder"]["module_sha256"] = _sha256(REPO_ROOT / config["grounder"]["module"])
    config["grounder"]["collector_sha256"] = _sha256(REPO_ROOT / config["grounder"]["collector"])
    config.update({
        "schema_version": "agqa2-router-grounding-development-config-v1",
        "status": CONFIG_STATUS,
        "split": SPLIT, "report_version": REPORT_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "preregistration": str(SELECTION.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(SELECTION),
        "expected_preregistration_status": selection["status"],
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(MANIFEST),
        "expected_manifest_status": manifest["status"],
        "qualification_gates": {
            "required_valid_runtime_rows": len(samples), "minimum_route_correct": len(samples),
            "minimum_decisive_executions": 1, "minimum_decisive_accuracy": 0.0,
            "maximum_typed_vs_direct_losses": len(samples), "minimum_typed_vs_direct_wins": 0,
            "required_source_permuted_abstentions": len(samples), "required_target_written_equivalent_matches": len(samples),
            "maximum_reported_provider_cost_usd": 0.5
        }
    })
    CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": config["status"], "sample_count": len(samples), "manifest_file_sha256": _sha256(MANIFEST), "config_file_sha256": _sha256(CONFIG)}, indent=2))


if __name__ == "__main__":
    main()
