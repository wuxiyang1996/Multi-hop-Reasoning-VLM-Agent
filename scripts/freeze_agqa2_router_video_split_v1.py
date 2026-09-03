#!/usr/bin/env python3
"""Freeze video-disjoint AGQA train partitions before reading program labels."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402


ARCHIVE = Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip")
ENTRY = "AGQA_balanced/train_balanced.txt"
OUTPUT = REPO_ROOT / "configs/agqa2_program_router_video_split_v1.json"
NONCE = "agqa2-program-router-video-disjoint-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"split is immutable once written: {OUTPUT}")
    videos = set()
    with zipfile.ZipFile(ARCHIVE) as bundle, bundle.open(ENTRY) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8")
        for _, row in _iter_top_level_object(text):
            videos.add(str(row["video_id"]))
    partitions = {"router_train": [], "router_validation": [], "formal_holdout": []}
    for video_id in sorted(videos):
        bucket = int(stable_hash({"nonce": NONCE, "video_id": video_id})[:8], 16) % 100
        key = "router_train" if bucket < 80 else "router_validation" if bucket < 90 else "formal_holdout"
        partitions[key].append(video_id)
    body = {
        "schema_version": "agqa2-program-router-video-split-v1",
        "status": "FROZEN_BEFORE_ANY_TRAIN_PROGRAM_LABEL_ACCESS",
        "selection_nonce": NONCE,
        "archive_path": str(ARCHIVE),
        "archive_sha256": _sha256(ARCHIVE),
        "entry": ENTRY,
        "split_rule": "SHA256_NONCE_VIDEO_ID_MOD_100:0_79_TRAIN;80_89_VALIDATION;90_99_FORMAL",
        "program_read_during_split": False,
        "answer_read_during_split": False,
        "scene_graph_read_during_split": False,
        "counts": {key: len(value) for key, value in partitions.items()},
        "partitions": partitions,
    }
    result = body | {"split_sha256": stable_hash(body)}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "split_sha256": result["split_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
