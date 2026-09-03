#!/usr/bin/env python3
"""Bind downloaded videos to the frozen Qwen235 V65 selection and config."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from motif_transfer.contracts import stable_hash  # noqa: E402


SELECTION = REPO_ROOT / "configs/agqa2_qwen235_fresh_reserve_v1_selection.json"
DOWNLOAD = REPO_ROOT / "runs/agqa2_qwen235_fresh_reserve_v1_download/receipt.json"
PREREG = REPO_ROOT / "configs/agqa2_qwen235_fresh_reserve_v1_preregistration.json"
BASE = REPO_ROOT / "configs/agqa2_qwen235_tool_grounder_v1_development.json"
MANIFEST = REPO_ROOT / "configs/agqa2_qwen235_fresh_reserve_v1_manifest.json"
CONFIG = REPO_ROOT / "configs/agqa2_qwen235_fresh_reserve_v1.json"


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


def main() -> None:
    if MANIFEST.exists() or CONFIG.exists():
        raise FileExistsError("V65 activation artifacts are immutable once written")
    selection = _verified(SELECTION, "manifest_sha256")
    receipt = json.loads(DOWNLOAD.read_text())
    if receipt["status"] != "COMPLETE":
        raise ValueError("download receipt is incomplete")
    if receipt["selection_manifest_sha256"] != selection["manifest_sha256"]:
        raise ValueError("download receipt belongs to another selection")
    downloaded = {row["video_id"]: row for row in receipt["videos"]}
    samples = []
    for row in selection["samples"]:
        video = downloaded[row["video_id"]]
        path = Path(row["video_path"])
        if not path.is_file() or _sha256(path) != video["sha256"]:
            raise ValueError(f"video integrity mismatch: {path}")
        samples.append(dict(row) | {
            "video_sha256": video["sha256"],
            "video_bytes": video["file_size"],
        })
    manifest_body = {
        "schema_version": "agqa2-qwen235-fresh-reserve-manifest-v1",
        "status": "FROZEN_V65_FRESH_RESERVE_BEFORE_PROVIDER_CALLS",
        "split": "official_test_fresh_video_relation_exists",
        "selection_manifest_sha256": selection["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(DOWNLOAD),
        "archive_path": selection["archive_path"],
        "archive_sha256": selection["archive_sha256"],
        "entry": selection["entry"],
        "sample_count": len(samples),
        "unique_video_count": len({row["video_id"] for row in samples}),
        "answer_read_during_activation": False,
        "scene_graph_read_during_activation": False,
        "samples": samples,
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = deepcopy(json.loads(BASE.read_text()))
    config.update({
        "schema_version": "agqa2-qwen235-fresh-reserve-config-v1",
        "status": "FROZEN_V65_FRESH_RESERVE",
        "split": manifest["split"],
        "report_version": "QWEN235_V65",
        "claim_boundary": "30_PREVIOUSLY_RUNTIME_UNEXPOSED_OFFICIAL_TEST_VIDEOS;ONE_RELATION_EXISTS_TASK_PER_VIDEO;SELECTIVE_TRANSFER_ONLY;NOT_FULL_AGQA_OR_SOTA",
        "preregistration": str(PREREG.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(PREREG),
        "expected_preregistration_status": "FROZEN_BEFORE_ANY_V65_PROVIDER_OR_OUTCOME_CALL",
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(MANIFEST),
        "expected_manifest_status": manifest["status"],
        "frozen_selective_authorizer": "configs/agqa2_qwen235_selective_authorizer_v1.json",
        "qualification_gates": {
            "required_valid_runtime_rows": 30,
            "minimum_route_correct": 30,
            "minimum_decisive_executions": 1,
            "minimum_decisive_accuracy": 0.0,
            "maximum_typed_vs_direct_losses": 30,
            "minimum_typed_vs_direct_wins": 0,
            "required_source_permuted_abstentions": 30,
            "required_target_written_equivalent_matches": 30,
            "maximum_reported_provider_cost_usd": 0.5
        },
    })
    CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": config["status"], "sample_count": len(samples),
        "manifest_file_sha256": _sha256(MANIFEST),
        "config_file_sha256": _sha256(CONFIG),
    }, indent=2))


if __name__ == "__main__":
    main()
