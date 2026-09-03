#!/usr/bin/env python3
"""Freeze router-qualified grounding development tasks on validation videos."""

from __future__ import annotations

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
from scripts.freeze_agqa2_router_heldout_formal_v1 import _prior_runtime_videos, _sha256  # noqa: E402


SPLIT = REPO_ROOT / "configs/agqa2_program_router_video_split_v1.json"
QUALIFICATION = REPO_ROOT / "runs/agqa2_program_router_v1/qualification_report_v2.json"
MODEL = REPO_ROOT / "runs/agqa2_program_router_v1/router.joblib"
OUTPUT = REPO_ROOT / "configs/agqa2_router_grounding_development_v1_selection.json"
VIDEO_ROOT = Path("/fs/gamma-projects/vlm-robot/datasets/STAR-official/videos/charades")
COUNT = 80
NONCE = "agqa2-router-v2-grounding-development-v1"
STATUS = "FROZEN_V67_DEVELOPMENT_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_OUTCOME_ACCESS"
SPLIT_NAME = "official_train_router_validation_grounding_development"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("grounding-development selection is immutable")
    split = json.loads(SPLIT.read_text())
    qualification = json.loads(QUALIFICATION.read_text())
    threshold = float(qualification["validation"]["selection"]["threshold"])
    allowed_videos = set(split["partitions"]["router_validation"]) - _prior_runtime_videos()
    raw_candidates = []
    with zipfile.ZipFile(split["archive_path"]) as bundle, bundle.open(split["entry"]) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            video_id = str(row["video_id"])
            if video_id not in allowed_videos:
                continue
            question = str(row["question"])
            plan = parse_public_question_plan(question)
            if plan is not None and plan.comparison == "EXISTS":
                raw_candidates.append((task_id, video_id, question, plan))
    model = joblib.load(MODEL)
    scores = model.predict_proba([row[2] for row in raw_candidates])[:, 1]
    by_video = {}
    for (task_id, video_id, question, plan), score in zip(raw_candidates, scores):
        if float(score) < threshold:
            continue
        row = {
            "task_id": task_id, "video_id": video_id,
            "question_sha256": stable_hash(question),
            "public_parser_plan_sha256": stable_hash(plan.as_dict()),
            "router_score": float(score),
            "rank_sha256": stable_hash({"nonce": NONCE, "task_id": task_id}),
            "video_path": str(VIDEO_ROOT / f"{video_id}.mp4"),
        }
        prior = by_video.get(video_id)
        if prior is None or (row["router_score"], row["rank_sha256"]) > (prior["router_score"], prior["rank_sha256"]):
            by_video[video_id] = row
    selected = sorted(by_video.values(), key=lambda row: row["rank_sha256"])[:COUNT]
    if len(selected) != COUNT:
        raise ValueError(f"only {len(selected)} grounding-development videos qualify")
    body = {
        "schema_version": "agqa2-router-grounding-development-selection-v1",
        "status": STATUS,
        "split": SPLIT_NAME,
        "selection_nonce": NONCE,
        "sample_count": len(selected),
        "unique_video_count": len({row["video_id"] for row in selected}),
        "router_threshold": threshold,
        "router_model_file_sha256": _sha256(MODEL),
        "router_qualification_file_sha256": _sha256(QUALIFICATION),
        "archive_path": split["archive_path"], "archive_sha256": split["archive_sha256"], "entry": split["entry"],
        "raw_video_archive": {"url": "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip", "archive_prefix": "Charades_v1_480/"},
        "answer_read_during_selection": False,
        "program_read_during_selection": False,
        "scene_graph_read_during_selection": False,
        "samples": selected
    }
    result = body | {"manifest_sha256": stable_hash(body)}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "sample_count": len(selected), "already_local": sum(Path(row["video_path"]).is_file() for row in selected), "manifest_sha256": result["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
