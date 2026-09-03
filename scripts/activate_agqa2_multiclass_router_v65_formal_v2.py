#!/usr/bin/env python3
"""Bind frozen multi-route V2 videos to the unchanged historical V65 runtime."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.activate_agqa2_qwen235_fresh_reserve_v1 import _sha256, _verified  # noqa: E402

SELECTION = REPO_ROOT / "configs/agqa2_multiclass_router_formal_v2_selection.json"
PRIOR_SELECTION = REPO_ROOT / "configs/agqa2_router_heldout_formal_v1_selection.json"
DOWNLOAD = REPO_ROOT / "runs/agqa2_multiclass_router_formal_v2_download/receipt.json"
BASE = REPO_ROOT / "configs/agqa2_qwen235_fresh_reserve_v1.json"
PROTOCOL = REPO_ROOT / "configs/agqa2_multiclass_router_v65_formal_v2_protocol.json"
ROUTER_MODEL = REPO_ROOT / "runs/agqa2_multiclass_program_router_v2/router.joblib"
ROUTER_REPORT = REPO_ROOT / "runs/agqa2_multiclass_program_router_v2/qualification_report.json"
MANIFEST = REPO_ROOT / "configs/agqa2_multiclass_router_v65_formal_v2_manifest.json"
CONFIG = REPO_ROOT / "configs/agqa2_multiclass_router_v65_formal_v2.json"
V65_COMMIT = "ded7448839183851aa10c3cd3e12d253f04e1ceb"
V65_COLLECTOR_SHA = "c845a0446fe5edc60f29dedbbb8eca3527a1f0c087f130924529a64cb8cdd5f1"
V65_MODULE_SHA = "87a41b64a77aae9cd8899f714061276fd3fcee05e8950a050fffb8849b81761c"
EXPECTED_GROUNDER_SHA = "f8e3e500c273858b5cb70a2ae3e0551e51be8dff4d76e33c6144a578209cbed1"


def main() -> None:
    if MANIFEST.exists() or CONFIG.exists():
        raise FileExistsError("V2 activation artifacts are immutable")
    selection = _verified(SELECTION, "manifest_sha256")
    prior = _verified(PRIOR_SELECTION, "manifest_sha256")
    protocol = _verified(PROTOCOL, "protocol_sha256")
    router_report = _verified(ROUTER_REPORT, "report_sha256")
    receipt = json.loads(DOWNLOAD.read_text())
    if protocol["status"] != "FROZEN_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_FORMAL_LABEL_ACCESS":
        raise ValueError("V2 protocol is not prospective")
    if protocol["cohort"]["selection_manifest_sha256"] != selection["manifest_sha256"]:
        raise ValueError("protocol/selection mismatch")
    if protocol["cohort"]["prior_v1_selection_manifest_sha256"] != prior["manifest_sha256"]:
        raise ValueError("protocol/V1 exclusion mismatch")
    if receipt["status"] != "COMPLETE":
        raise ValueError("V2 video download is incomplete")
    if receipt["selection_manifest_sha256"] != selection["manifest_sha256"]:
        raise ValueError("download receipt belongs to another selection")
    if _sha256(ROUTER_MODEL) != protocol["lineage"]["program_router_model_sha256"]:
        raise ValueError("router model changed after protocol freeze")
    if _sha256(ROUTER_REPORT) != protocol["lineage"]["program_router_qualification_file_sha256"]:
        raise ValueError("router qualification changed after protocol freeze")
    if router_report["report_sha256"] != protocol["lineage"]["program_router_qualification_report_sha256"]:
        raise ValueError("router qualification body changed after protocol freeze")

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
    if len({row["video_id"] for row in samples}) != len(samples):
        raise ValueError("V2 is not one-question-per-video")
    if {row["video_id"] for row in samples} & {row["video_id"] for row in prior["samples"]}:
        raise ValueError("V2 overlaps V1")

    manifest_body = {
        "schema_version": "agqa2-multiclass-router-v65-formal-manifest-v2",
        "status": "FROZEN_BEFORE_ANY_FORMAL_PROVIDER_OR_OUTCOME_ACCESS",
        "split": "official_train_remaining_video_heldout_multiclass_router_v2",
        "selection_manifest_sha256": selection["manifest_sha256"],
        "prior_v1_selection_manifest_sha256": prior["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(DOWNLOAD),
        "archive_path": selection["archive_path"],
        "archive_sha256": selection["archive_sha256"],
        "entry": selection["entry"],
        "sample_count": len(samples),
        "unique_video_count": len(samples),
        "route_counts": selection["route_counts"],
        "answer_read_during_activation": False,
        "program_read_during_activation": False,
        "scene_graph_read_during_activation": False,
        "samples": samples,
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = deepcopy(json.loads(BASE.read_text()))
    config.update({
        "schema_version": "agqa2-multiclass-router-v65-formal-config-v2",
        "status": "FROZEN_MULTICLASS_ROUTER_V65_GROUNDER_FORMAL_V2",
        "split": manifest["split"],
        "report_version": "QWEN235_V65_MULTICLASS_ROUTER_FORMAL_V2",
        "claim_boundary": protocol["claim_boundary"],
        "preregistration": str(SELECTION.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(SELECTION),
        "expected_preregistration_status": selection["status"],
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(MANIFEST),
        "expected_manifest_status": manifest["status"],
        "formal_protocol": str(PROTOCOL.relative_to(REPO_ROOT)),
        "formal_protocol_file_sha256": _sha256(PROTOCOL),
        "expected_grounder_sha256": EXPECTED_GROUNDER_SHA,
        "target_native_program_router": {
            "role": "QUESTION_ONLY_ROUTE_SELECTOR_NOT_VISUAL_GROUNDER",
            "training_scope": "AGQA_OFFICIAL_TRAIN_DEVELOPMENT_PARTITIONS_ONLY",
            "model": str(ROUTER_MODEL.relative_to(REPO_ROOT)),
            "model_file_sha256": _sha256(ROUTER_MODEL),
            "qualification": str(ROUTER_REPORT.relative_to(REPO_ROOT)),
            "qualification_file_sha256": _sha256(ROUTER_REPORT),
            "qualification_report_sha256": router_report["report_sha256"],
            "formal_video_labels_read": False,
        },
        "frozen_runtime": {
            "git_commit": V65_COMMIT,
            "collector_sha256": V65_COLLECTOR_SHA,
            "grounder_module_sha256": V65_MODULE_SHA,
            "dependency_overlay_sha256": {
                "src/motif_transfer/phase3_source_function_induction.py": "5bd04fa4b0d9b3a90b61d9108e19b8366080b167a63e5ac2556d351356fdcd6d"
            },
            "post_v65_grounder_tuning_used": False,
        },
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
        "route_counts": selection["route_counts"],
        "manifest_file_sha256": _sha256(MANIFEST),
        "config_file_sha256": _sha256(CONFIG),
        "protocol_file_sha256": _sha256(PROTOCOL),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
