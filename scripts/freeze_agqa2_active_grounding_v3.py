#!/usr/bin/env python3
"""Freeze AGQA V3 development and raw-video-unseen reserve manifests.

The V2 videos are the consumed development set.  The reserve excludes every
V2 video before ranking and is frozen before any V3 provider call.  AGQA test
metadata was already globally scanned by V1, so the reserve claim is limited
to previously unseen raw videos, not an untouched benchmark split.
"""

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
RESERVE_NONCE = "agqa2-active-grounding-raw-video-unseen-reserve-v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_core(
    *, source: dict, samples: list[dict], status: str, split: str,
    selection_rule: str, nonce: str,
) -> dict:
    counts = {
        route: sum(row["oracle_route"] == route for row in samples)
        for route in ROUTES
    }
    used = {row["video_id"] for row in samples}
    return {
        "schema_version": "agqa2-active-grounding-manifest-v3",
        "status": status,
        "split": split,
        "claim_boundary": (
            "V3_RAW_VIDEO_UNSEEN_AND_VIDEO_DISJOINT_RESERVE;AGQA_TEST_METADATA_"
            "WAS_PREVIOUSLY_GLOBALLY_SCANNED;NOT_AN_UNTOUCHED_BENCHMARK_SPLIT"
            if split == "reserve" else
            "CONSUMED_V2_DEVELOPMENT_VIDEOS;DEVELOPMENT_ONLY"
        ),
        "selection_nonce": nonce,
        "selection_rule": selection_rule,
        "archive_path": source["archive_path"],
        "archive_sha256": source["archive_sha256"],
        "entry": source["entry"],
        "video_root": source["video_root"],
        "available_video_count": source["available_video_count"],
        "per_route": 3,
        "route_counts": counts,
        "samples": samples,
        "sample_count": len(samples),
        "unique_video_count": len(used),
        "answer_read_during_freeze": False,
        "scene_graph_grounding_read_during_freeze": False,
        "functional_program_visible_to_grounder": False,
        "prior_v3_raw_video_exposure": False if split == "reserve" else True,
        "new_video_downloads": 0,
    }


def freeze(
    *, v2_manifest_path: Path, per_route: int = 3,
) -> tuple[dict, dict]:
    v2 = json.loads(v2_manifest_path.read_text())
    body = dict(v2)
    claimed = body.pop("manifest_sha256")
    if stable_hash(body) != claimed:
        raise ValueError("V2 development manifest content hash mismatch")
    if per_route != 3:
        raise ValueError("V3 preregistration fixes exactly three samples per route")

    development_samples = [dict(row) for row in v2["samples"]]
    development_core = _manifest_core(
        source=v2,
        samples=development_samples,
        status="FROZEN_CONSUMED_DEVELOPMENT_BEFORE_V3_NEURAL_CALLS",
        split="development",
        selection_rule="EXACT_V2_DEVELOPMENT_MANIFEST_REUSE",
        nonce="agqa2-active-grounding-consumed-development-v3",
    )
    development_core["parent_v2_manifest_sha256"] = claimed
    development = development_core | {
        "manifest_sha256": stable_hash(development_core),
    }

    video_root = Path(v2["video_root"])
    available = {path.stem: path for path in video_root.glob("*.mp4")}
    excluded_videos = {row["video_id"] for row in development_samples}
    candidates: dict[str, list[dict]] = defaultdict(list)
    archive = Path(v2["archive_path"])
    with zipfile.ZipFile(archive) as bundle, bundle.open(v2["entry"], "r") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            for task_id, row in _iter_top_level_object(text):
                video_id = str(row.get("video_id", ""))
                if video_id not in available or video_id in excluded_videos:
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
                    "rank_sha256": stable_hash(f"{RESERVE_NONCE}:{task_id}"),
                })
    for route in ROUTES:
        candidates[route].sort(key=lambda row: row["rank_sha256"])

    selected: list[dict] = []
    used_videos: set[str] = set()
    # Prefer scarce route/video combinations first while retaining a fixed
    # route presentation order in the final manifest.
    route_order = sorted(
        ROUTES,
        key=lambda route: len({row["video_id"] for row in candidates[route]}),
    )
    for route in route_order:
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
        availability = {
            route: len({row["video_id"] for row in candidates[route]})
            for route in ROUTES
        }
        raise RuntimeError(
            f"insufficient V3 reserve candidates: selected={counts}, "
            f"available_videos={availability}"
        )
    selected.sort(key=lambda row: (
        ROUTES.index(row["oracle_route"]), row["rank_sha256"],
    ))
    reserve_core = _manifest_core(
        source=v2,
        samples=selected,
        status="FROZEN_RAW_VIDEO_UNSEEN_RESERVE_BEFORE_V3_NEURAL_CALLS",
        split="reserve",
        selection_rule=(
            "EXCLUDE_ALL_V2_DEVELOPMENT_VIDEO_IDS;SHA256_RANK_WITHIN_ORACLE_"
            "ROUTE;SCARCE_ROUTE_FIRST;GLOBAL_VIDEO_DISJOINT;NO_ANSWER_OR_"
            "SCENE_GRAPH_READ"
        ),
        nonce=RESERVE_NONCE,
    )
    reserve_core["excluded_development_video_ids"] = sorted(excluded_videos)
    reserve_core["development_manifest_sha256"] = development[
        "manifest_sha256"
    ]
    reserve = reserve_core | {"manifest_sha256": stable_hash(reserve_core)}
    return development, reserve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2-manifest", type=Path,
        default=REPO_ROOT / "configs/agqa2_frame_grounding_v2_manifest.json",
    )
    parser.add_argument(
        "--development-output", type=Path,
        default=REPO_ROOT / "configs/agqa2_active_grounding_v3_development_manifest.json",
    )
    parser.add_argument(
        "--reserve-output", type=Path,
        default=REPO_ROOT / "configs/agqa2_active_grounding_v3_reserve_manifest.json",
    )
    args = parser.parse_args()
    development, reserve = freeze(v2_manifest_path=args.v2_manifest.resolve())
    for path, payload in (
        (args.development_output, development),
        (args.reserve_output, reserve),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "development_manifest_sha256": development["manifest_sha256"],
        "reserve_manifest_sha256": reserve["manifest_sha256"],
        "development_video_ids": [
            row["video_id"] for row in development["samples"]
        ],
        "reserve_video_ids": [row["video_id"] for row in reserve["samples"]],
        "video_overlap": sorted(
            {row["video_id"] for row in development["samples"]}
            & {row["video_id"] for row in reserve["samples"]}
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
