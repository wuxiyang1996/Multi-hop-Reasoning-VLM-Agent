#!/usr/bin/env python3
"""Freeze an outcome-blind fresh AGQA actor qualification on local unused videos."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v16_reserve import _configured_video_ids  # noqa: E402


VIDEO_COUNT = 30
ROWS_PER_VIDEO = 3
NONCE = "agqa2-target-actor-v64-fresh-local-qualification"
ARCHIVE = Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip")
ENTRY = "AGQA_balanced/test_balanced.txt"
VIDEO_ROOT = Path("/fs/gamma-projects/vlm-robot/datasets/STAR-official/videos/charades")
SELECTION = REPO / "configs/agqa2_target_actor_v64_qualification_selection.json"
MANIFEST = REPO / "configs/agqa2_target_actor_v64_qualification_manifest.json"
CONFIG = REPO / "configs/agqa2_target_actor_v64_qualification.json"


def main() -> int:
    if any(path.exists() for path in (SELECTION, MANIFEST, CONFIG)):
        raise SystemExit("refusing to overwrite frozen V64 qualification")
    eligible_videos = {
        path.stem for path in VIDEO_ROOT.glob("*.mp4")
    } - _configured_video_ids()
    video_sha256 = {
        video_id: hashlib.sha256(
            (VIDEO_ROOT / f"{video_id}.mp4").read_bytes()
        ).hexdigest()
        for video_id in eligible_videos
    }
    by_video: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(ARCHIVE) as bundle:
        with bundle.open(ENTRY, "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            for task_id, row in _iter_top_level_object(text):
                video_id = str(row.get("video_id", ""))
                if video_id not in eligible_videos:
                    continue
                question = str(row.get("question", ""))
                by_video[video_id].append({
                    "task_id": str(task_id),
                    "video_id": video_id,
                    "video_path": str(VIDEO_ROOT / f"{video_id}.mp4"),
                    "video_sha256": video_sha256[video_id],
                    "question_sha256": stable_hash(question),
                    "priority": stable_hash(NONCE + "|row|" + str(task_id)),
                })
    selected_videos = sorted(
        by_video, key=lambda value: stable_hash(NONCE + "|video|" + value)
    )[:VIDEO_COUNT]
    if len(selected_videos) != VIDEO_COUNT:
        raise ValueError("not enough local unused AGQA videos")
    samples = []
    for video_id in sorted(selected_videos):
        rows = sorted(by_video[video_id], key=lambda row: row["priority"])[:ROWS_PER_VIDEO]
        if len(rows) != ROWS_PER_VIDEO:
            raise ValueError("selected video has fewer than three questions")
        samples.extend({key: value for key, value in row.items() if key != "priority"} for row in rows)
    selection_body = {
        "schema_version": "agqa2-target-actor-selection-v64",
        "status": "FROZEN_BEFORE_ANY_V64_PROVIDER_OR_OUTCOME_CALL",
        "split": "official_test_fresh_local_video_actor_qualification",
        "selection_nonce": NONCE,
        "selection_rule": "30_LOWEST_HASH_UNUSED_LOCAL_VIDEOS_X_3_LOWEST_HASH_QUESTIONS;NO_OPERATOR_ANSWER_PROGRAM_OR_SCENE_GRAPH_FILTER",
        "sample_count": len(samples),
        "unique_video_count": len(selected_videos),
        "answer_program_scene_graph_read_during_freeze": False,
        "samples": samples,
    }
    selection = selection_body | {"manifest_sha256": stable_hash(selection_body)}
    SELECTION.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_body = {
        "schema_version": "agqa2-target-actor-manifest-v64",
        "status": "FROZEN_FRESH_QUALIFICATION_BEFORE_PROVIDER_CALLS",
        "split": selection["split"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "sample_count": len(samples),
        "unique_video_count": len(selected_videos),
        "samples": samples,
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = {
        "schema_version": "agqa2-target-actor-qualification-v64",
        "status": "FROZEN_BEFORE_ANY_V64_PROVIDER_OR_OUTCOME_CALL",
        "manifest": str(MANIFEST.relative_to(REPO)),
        "manifest_file_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "baseline_model_id": "qwen/qwen3-vl-32b-instruct",
        "dataset": {"archive_path": str(ARCHIVE), "archive_sha256": "3cd4cc741864ac4bb875e3c1dc41d0ddea559259059b01d229527b08323bea0d", "entry": ENTRY},
        "media": {"frame_count": 48, "frame_max_side": 512, "frames_per_panel": 6, "panel_frame_width": 192, "jpeg_quality": 80},
        "models": [
            {"id": "qwen/qwen3-vl-32b-instruct", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key_name": "OPENROUTER_API_KEY", "temperature": 0, "max_direct_tokens": 80, "max_retries": 2, "schema_retries": 2, "timeout_seconds": 240},
            {"id": "google/gemini-3-flash-preview", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key_name": "OPENROUTER_API_KEY", "temperature": 0, "max_direct_tokens": 80, "max_retries": 2, "schema_retries": 2, "timeout_seconds": 240}
        ],
        "selection_gates": {"required_rows_per_model": 90, "minimum_accuracy_gain_over_baseline": 0.05, "minimum_net_paired_wins": 5, "maximum_reported_provider_cost_usd": 1.0},
        "claim_boundary": "FRESH_VIDEO_DISJOINT_TARGET_ACTOR_QUALIFICATION;NO_SOURCE_TRANSFER_CLAIM;NO_FORMAL_RESULT_CHANGE;WINNER_MUST_BE_FROZEN_BEFORE_A_SEPARATE_FORMAL SAMPLE"
    }
    CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": config["status"], "samples": len(samples), "videos": len(selected_videos), "selection_sha256": selection["manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
