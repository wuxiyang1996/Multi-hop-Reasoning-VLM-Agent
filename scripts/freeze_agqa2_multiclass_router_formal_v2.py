#!/usr/bin/env python3
"""Freeze a fresh multi-route AGQA cohort without reading formal labels."""

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
ROUTER_REPORT = REPO_ROOT / "runs/agqa2_multiclass_program_router_v2/qualification_report.json"
ROUTER_MODEL = REPO_ROOT / "runs/agqa2_multiclass_program_router_v2/router.joblib"
OUTPUT = REPO_ROOT / "configs/agqa2_multiclass_router_formal_v2_selection.json"
VIDEO_ROOT = Path("/fs/gamma-projects/vlm-robot/datasets/STAR-official/videos/charades")
NONCE = "agqa2-multiclass-router-remaining-untouched-formal-v2"
MINIMUM_VIDEOS = 60


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


def best_per_video(rows: list[dict]) -> list[dict]:
    by_video = {}
    for row in rows:
        prior = by_video.get(row["video_id"])
        key = (float(row["router_score"]), row["rank_sha256"])
        prior_key = (
            (float(prior["router_score"]), prior["rank_sha256"])
            if prior is not None else None
        )
        if prior_key is None or key > prior_key:
            by_video[row["video_id"]] = row
    return sorted(by_video.values(), key=lambda row: row["rank_sha256"])


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("formal V2 selection is immutable")
    split = json.loads(SPLIT.read_text())
    router_report = json.loads(ROUTER_REPORT.read_text())
    report_body = dict(router_report)
    claimed_report_hash = report_body.pop("report_sha256")
    if stable_hash(report_body) != claimed_report_hash:
        raise ValueError("multi-class router report hash mismatch")
    if router_report["status"] != "MULTICLASS_PROGRAM_ROUTER_V2_QUALIFIED":
        raise ValueError("multi-class router did not qualify")
    if router_report["model_file_sha256"] != _sha256(ROUTER_MODEL):
        raise ValueError("multi-class router model hash mismatch")
    thresholds = {
        route: float(value["threshold"])
        for route, value in router_report["validation"]["thresholds"].items()
    }
    formal_videos = set(split["partitions"]["formal_holdout"])
    prior_videos = _prior_runtime_videos()
    eligible_videos = formal_videos - prior_videos
    router = joblib.load(ROUTER_MODEL)
    classes = list(router.classes_)
    raw_candidates = []
    with zipfile.ZipFile(split["archive_path"]) as bundle, bundle.open(split["entry"]) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            video_id = str(row["video_id"])
            if video_id not in eligible_videos:
                continue
            question = str(row["question"])
            plan = parse_public_question_plan(question)
            if plan is None:
                continue
            probabilities = router.predict_proba([question])[0]
            class_index = int(probabilities.argmax())
            predicted_route = classes[class_index]
            score = float(probabilities[class_index])
            threshold = thresholds.get(predicted_route)
            if (
                threshold is None or score < threshold
                or plan.obligation_kind != predicted_route
            ):
                continue
            raw_candidates.append({
                "task_id": task_id,
                "video_id": video_id,
                "predicted_route": predicted_route,
                "parser_route": plan.obligation_kind,
                "comparison": plan.comparison,
                "router_score": score,
                "router_threshold": threshold,
                "question_sha256": stable_hash(question),
                "public_parser_plan_sha256": stable_hash(plan.as_dict()),
                "rank_sha256": stable_hash({"nonce": NONCE, "task_id": task_id}),
                "video_path": str(VIDEO_ROOT / f"{video_id}.mp4"),
            })
    selected = best_per_video(raw_candidates)
    if len(selected) < MINIMUM_VIDEOS:
        raise ValueError(
            f"only {len(selected)} untouched multi-route formal videos qualify"
        )
    route_counts = {}
    for row in selected:
        route_counts[row["predicted_route"]] = (
            route_counts.get(row["predicted_route"], 0) + 1
        )
    body = {
        "schema_version": "agqa2-multiclass-router-formal-v2-selection-v1",
        "status": "FROZEN_V78_SELECTION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_FORMAL_LABEL_ACCESS",
        "split": "official_train_remaining_video_heldout_multiclass_router_v2",
        "selection_nonce": NONCE,
        "selection_rule": "ALL_REMAINING_RUNTIME_UNSEEN_FORMAL_VIDEOS;QUESTION_ONLY_ROUTER_AND_DETERMINISTIC_PLAN_AGREE;PER_ROUTE_FROZEN_THRESHOLD;ONE_HIGHEST_SCORE_QUESTION_PER_VIDEO",
        "sample_count": len(selected),
        "unique_video_count": len({row["video_id"] for row in selected}),
        "route_counts": dict(sorted(route_counts.items())),
        "formal_partition_video_count": len(formal_videos),
        "prior_runtime_video_count": len(prior_videos),
        "eligible_unexposed_formal_video_count": len(eligible_videos),
        "qualified_public_question_count": len(raw_candidates),
        "router_model_file_sha256": _sha256(ROUTER_MODEL),
        "router_qualification_file_sha256": _sha256(ROUTER_REPORT),
        "router_qualification_report_sha256": router_report["report_sha256"],
        "archive_path": split["archive_path"],
        "archive_sha256": split["archive_sha256"],
        "entry": split["entry"],
        "raw_video_archive": {
            "url": "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip",
            "archive_prefix": "Charades_v1_480/",
        },
        "answer_read_during_selection": False,
        "program_read_during_selection": False,
        "scene_graph_read_during_selection": False,
        "source_identity_read_during_selection": False,
        "v1_formal_outcomes_used_for_selection": False,
        "samples": selected,
    }
    result = body | {"manifest_sha256": stable_hash(body)}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"], "sample_count": len(selected),
        "route_counts": result["route_counts"],
        "eligible_unexposed_formal_video_count": len(eligible_videos),
        "qualified_public_question_count": len(raw_candidates),
        "already_local": sum(Path(row["video_path"]).is_file() for row in selected),
        "manifest_sha256": result["manifest_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
