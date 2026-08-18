#!/usr/bin/env python3
"""Freeze a video-disjoint AGQA frame-grounding development pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_program_transfer import (  # noqa: E402
    RELATION_ROUTE,
    TEMPORAL_PAIR_ROUTE,
    TEMPORAL_SINGLE_ROUTE,
    profile_program,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import (  # noqa: E402
    _iter_top_level_object,
)


ROUTES = (RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE)
NONCE = "agqa2-frame-grounding-consumed-development-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(
    *, archive: Path, entry: str, video_root: Path, per_route: int,
) -> dict:
    available = {path.stem: path for path in video_root.glob("*.mp4")}
    candidates = defaultdict(list)
    with zipfile.ZipFile(archive) as bundle, bundle.open(entry, "r") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            for task_id, row in _iter_top_level_object(text):
                video_id = str(row.get("video_id", ""))
                if video_id not in available:
                    continue
                question = str(row.get("question", ""))
                program = str(row.get("program", ""))
                profile = profile_program(task_id=task_id, program=program)
                if profile.route_kind not in ROUTES:
                    continue
                candidates[profile.route_kind].append({
                    "task_id": task_id,
                    "video_id": video_id,
                    "oracle_route": profile.route_kind,
                    "question_sha256": stable_hash(question),
                    "program_sha256": stable_hash(program),
                    "rank_sha256": stable_hash(f"{NONCE}:{task_id}"),
                })
    for route in ROUTES:
        candidates[route].sort(key=lambda row: row["rank_sha256"])
    selected = []
    used_videos: set[str] = set()
    for route in ROUTES:
        for row in candidates[route]:
            if row["video_id"] in used_videos:
                continue
            video_path = available[row["video_id"]]
            selected.append(row | {
                "video_path": str(video_path),
                "video_sha256": _sha256(video_path),
                "video_bytes": video_path.stat().st_size,
            })
            used_videos.add(row["video_id"])
            if sum(x["oracle_route"] == route for x in selected) == per_route:
                break
    counts = {
        route: sum(row["oracle_route"] == route for row in selected)
        for route in ROUTES
    }
    if any(count != per_route for count in counts.values()):
        raise RuntimeError(f"insufficient video-disjoint candidates: {counts}")
    selected.sort(key=lambda row: (ROUTES.index(row["oracle_route"]), row["rank_sha256"]))
    core = {
        "schema_version": "agqa2-frame-grounding-manifest-v2",
        "status": "FROZEN_CONSUMED_METADATA_DEVELOPMENT_BEFORE_NEURAL_CALLS",
        "claim_boundary": (
            "FRAME_ONLY_GROUNDER_QUALIFICATION_ON_ALREADY_DOWNLOADED_CHARADES;"
            "NO_UNTOUCHED_OR_FORMAL_TRANSFER_CLAIM"
        ),
        "selection_nonce": NONCE,
        "selection_rule": (
            "SHA256_RANK_WITHIN_ORACLE_ROUTE_THEN_GLOBAL_VIDEO_DISJOINT;"
            "NO_ANSWER_OR_SCENE_GRAPH_READ"
        ),
        "archive_path": str(archive),
        "archive_sha256": _sha256(archive),
        "entry": entry,
        "video_root": str(video_root),
        "available_video_count": len(available),
        "per_route": per_route,
        "route_counts": counts,
        "samples": selected,
        "sample_count": len(selected),
        "unique_video_count": len(used_videos),
        "answer_read_during_freeze": False,
        "scene_graph_grounding_read_during_freeze": False,
        "functional_program_visible_to_grounder": False,
        "new_video_downloads": 0,
    }
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/"
            "AGQA_balanced.zip"
        ),
    )
    parser.add_argument("--entry", default="AGQA_balanced/test_balanced.txt")
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path(
            "/fs/gamma-projects/vlm-robot/datasets/STAR-official/videos/charades"
        ),
    )
    parser.add_argument("--per-route", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "configs/agqa2_frame_grounding_v2_manifest.json",
    )
    args = parser.parse_args()
    result = freeze(
        archive=args.archive.resolve(),
        entry=args.entry,
        video_root=args.video_root.resolve(),
        per_route=args.per_route,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "sample_count": result["sample_count"],
        "unique_video_count": result["unique_video_count"],
        "route_counts": result["route_counts"],
        "manifest_sha256": result["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
