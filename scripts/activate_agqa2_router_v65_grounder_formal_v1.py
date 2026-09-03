#!/usr/bin/env python3
"""Bind the frozen router-selected cohort to the unchanged V65 grounder."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.activate_agqa2_qwen235_fresh_reserve_v1 import _sha256, _verified  # noqa: E402


SELECTION = REPO_ROOT / "configs/agqa2_router_heldout_formal_v1_selection.json"
DOWNLOAD = REPO_ROOT / "runs/agqa2_router_heldout_formal_v1_download/receipt.json"
BASE = REPO_ROOT / "configs/agqa2_qwen235_fresh_reserve_v1.json"
PROTOCOL = REPO_ROOT / "configs/agqa2_router_v65_grounder_formal_v1_protocol.json"
MANIFEST = REPO_ROOT / "configs/agqa2_router_v65_grounder_formal_v1_manifest.json"
CONFIG = REPO_ROOT / "configs/agqa2_router_v65_grounder_formal_v1.json"
V65_COMMIT = "ded7448839183851aa10c3cd3e12d253f04e1ceb"
V65_COLLECTOR_SHA = "c845a0446fe5edc60f29dedbbb8eca3527a1f0c087f130924529a64cb8cdd5f1"
V65_MODULE_SHA = "87a41b64a77aae9cd8899f714061276fd3fcee05e8950a050fffb8849b81761c"


def main() -> None:
    if MANIFEST.exists() or CONFIG.exists():
        raise FileExistsError("formal activation artifacts are immutable")
    selection = _verified(SELECTION, "manifest_sha256")
    protocol = _verified(PROTOCOL, "protocol_sha256")
    receipt = json.loads(DOWNLOAD.read_text())
    if receipt["status"] != "COMPLETE":
        raise ValueError("formal video download is incomplete")
    if receipt["selection_manifest_sha256"] != selection["manifest_sha256"]:
        raise ValueError("download receipt belongs to another selection")
    downloaded = {str(row["video_id"]): row for row in receipt["videos"]}
    samples = []
    for row in selection["samples"]:
        video = downloaded[str(row["video_id"])]
        path = Path(row["video_path"])
        if not path.is_file() or _sha256(path) != video["sha256"]:
            raise ValueError(f"video integrity mismatch: {path}")
        samples.append(dict(row) | {
            "video_sha256": video["sha256"], "video_bytes": video["file_size"],
        })
    manifest_body = {
        "schema_version": "agqa2-router-v65-grounder-formal-manifest-v1",
        "status": "FROZEN_BEFORE_ANY_FORMAL_PROVIDER_OR_OUTCOME_ACCESS",
        "split": "official_train_video_heldout_router_selected_relation_exists",
        "selection_manifest_sha256": selection["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(DOWNLOAD),
        "archive_path": selection["archive_path"],
        "archive_sha256": selection["archive_sha256"],
        "entry": selection["entry"],
        "sample_count": len(samples),
        "unique_video_count": len({row["video_id"] for row in samples}),
        "answer_read_during_activation": False,
        "program_read_during_activation": False,
        "scene_graph_read_during_activation": False,
        "samples": samples,
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = deepcopy(json.loads(BASE.read_text()))
    config.update({
        "schema_version": "agqa2-router-v65-grounder-formal-config-v1",
        "status": "FROZEN_ROUTER_V65_GROUNDER_FORMAL_V1",
        "split": "official_train_video_heldout_router_selected_relation_exists",
        "report_version": "QWEN235_V65_ROUTER_FORMAL_V1",
        "claim_boundary": protocol["claim_boundary"],
        "preregistration": str(SELECTION.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(SELECTION),
        "expected_preregistration_status": selection["status"],
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(MANIFEST),
        "expected_manifest_status": manifest["status"],
        "formal_protocol": str(PROTOCOL.relative_to(REPO_ROOT)),
        "formal_protocol_file_sha256": _sha256(PROTOCOL),
        "expected_grounder_sha256": "f8e3e500c273858b5cb70a2ae3e0551e51be8dff4d76e33c6144a578209cbed1",
        "frozen_runtime": {
            "git_commit": V65_COMMIT,
            "collector_sha256": V65_COLLECTOR_SHA,
            "grounder_module_sha256": V65_MODULE_SHA,
            "dependency_overlay_sha256": {
                "src/motif_transfer/phase3_source_function_induction.py": "5bd04fa4b0d9b3a90b61d9108e19b8366080b167a63e5ac2556d351356fdcd6d"
            },
            "post_v65_grounder_tuning_used": False,
        },
        # These gates only open the already V65-qualified fail-closed runtime.
        # The independent formal protocol below owns all success-gain gates.
        "qualification_gates": {
            "required_valid_runtime_rows": len(samples),
            "minimum_route_correct": len(samples),
            "minimum_decisive_executions": 1,
            "minimum_decisive_accuracy": 0.0,
            "maximum_typed_vs_direct_losses": len(samples),
            "minimum_typed_vs_direct_wins": 0,
            "required_source_permuted_abstentions": len(samples),
            "required_target_written_equivalent_matches": len(samples),
            "maximum_reported_provider_cost_usd": float(
                protocol["gates"]["maximum_reported_provider_cost_usd"]
            ),
        },
    })
    config["grounder"]["collector_sha256"] = V65_COLLECTOR_SHA
    config["grounder"]["module_sha256"] = V65_MODULE_SHA
    CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": config["status"], "sample_count": len(samples),
        "unique_video_count": manifest["unique_video_count"],
        "manifest_file_sha256": _sha256(MANIFEST),
        "config_file_sha256": _sha256(CONFIG),
        "protocol_file_sha256": _sha256(PROTOCOL),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
