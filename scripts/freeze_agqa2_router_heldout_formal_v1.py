#!/usr/bin/env python3
"""Freeze router-selected AGQA train formal questions without reading labels."""

from __future__ import annotations

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


SPLIT = REPO_ROOT / "configs/agqa2_program_router_video_split_v1.json"
QUALIFICATION = REPO_ROOT / "runs/agqa2_program_router_v1/qualification_report_v2.json"
MODEL = REPO_ROOT / "runs/agqa2_program_router_v1/router.joblib"
OUTPUT = REPO_ROOT / "configs/agqa2_router_heldout_formal_v1_selection.json"
VIDEO_ROOT = Path("/fs/gamma-projects/vlm-robot/datasets/STAR-official/videos/charades")
COUNT = 80
NONCE = "agqa2-router-v2-heldout-formal-v1"
STATUS = "FROZEN_V66_SELECTION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_FORMAL_LABEL_ACCESS"
SPLIT_NAME = "official_train_video_heldout_router_selected_relation_exists"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prior_runtime_videos() -> set[str]:
    output = set()
    for path in REPO_ROOT.glob("runs/agqa2*/runtime_receipts/*.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row.get("video_id"), str):
            output.add(row["video_id"])
    return output


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("formal selection is immutable once written")
    split = json.loads(SPLIT.read_text())
    qualification = json.loads(QUALIFICATION.read_text())
    if qualification["status"] != "PROGRAM_ROUTER_V2_QUALIFIED":
        raise ValueError("program router did not qualify")
    if qualification["model_file_sha256"] != _sha256(MODEL):
        raise ValueError("program router model hash mismatch")
    threshold = float(qualification["validation"]["selection"]["threshold"])
    formal_videos = set(split["partitions"]["formal_holdout"])
    prior_videos = _prior_runtime_videos()
    eligible_videos = formal_videos - prior_videos
    raw_candidates = []
    with zipfile.ZipFile(split["archive_path"]) as bundle, bundle.open(split["entry"]) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            video_id = str(row["video_id"])
            if video_id not in eligible_videos:
                continue
            question = str(row["question"])
            plan = parse_public_question_plan(question)
            if plan is None or plan.comparison != "EXISTS":
                continue
            raw_candidates.append((task_id, video_id, question, plan))
    model = joblib.load(MODEL)
    scores = model.predict_proba([row[2] for row in raw_candidates])[:, 1]
    candidates = []
    for (task_id, video_id, question, plan), score in zip(raw_candidates, scores):
        if float(score) < threshold:
            continue
        candidates.append({
            "task_id": task_id,
            "video_id": video_id,
            "question_sha256": stable_hash(question),
            "public_parser_plan_sha256": stable_hash(plan.as_dict()),
            "router_score": float(score),
            "rank_sha256": stable_hash({"nonce": NONCE, "task_id": task_id}),
            "video_path": str(VIDEO_ROOT / f"{video_id}.mp4"),
        })
    by_video = {}
    for row in candidates:
        prior = by_video.get(row["video_id"])
        if prior is None or (row["router_score"], row["rank_sha256"]) > (prior["router_score"], prior["rank_sha256"]):
            by_video[row["video_id"]] = row
    selected = sorted(by_video.values(), key=lambda row: row["rank_sha256"])[:COUNT]
    if len(selected) != COUNT:
        raise ValueError(f"only {len(selected)} qualified fresh formal videos available")
    body = {
        "schema_version": "agqa2-router-heldout-formal-selection-v1",
        "status": STATUS,
        "split": SPLIT_NAME,
        "selection_nonce": NONCE,
        "selection_rule": "FROZEN_ROUTER_V2_SCORE_GE_THRESHOLD;ONE_QUESTION_PER_FORMAL_VIDEO;HASH_RANK",
        "sample_count": len(selected),
        "unique_video_count": len({row["video_id"] for row in selected}),
        "formal_partition_video_count": len(formal_videos),
        "prior_runtime_video_count": len(prior_videos),
        "eligible_unexposed_formal_video_count": len(eligible_videos),
        "router_threshold": threshold,
        "router_model_file_sha256": _sha256(MODEL),
        "router_qualification_file_sha256": _sha256(QUALIFICATION),
        "router_qualification_report_sha256": qualification["report_sha256"],
        "archive_path": split["archive_path"],
        "archive_sha256": split["archive_sha256"],
        "entry": split["entry"],
        "raw_video_archive": {"url": "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip", "archive_prefix": "Charades_v1_480/"},
        "answer_read_during_selection": False,
        "program_read_during_selection": False,
        "scene_graph_read_during_selection": False,
        "source_identity_read_during_selection": False,
        "samples": selected,
    }
    result = body | {"manifest_sha256": stable_hash(body)}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "sample_count": len(selected), "unique_video_count": result["unique_video_count"], "eligible_unexposed_formal_video_count": len(eligible_videos), "already_local": sum(Path(row["video_path"]).is_file() for row in selected), "manifest_sha256": result["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
