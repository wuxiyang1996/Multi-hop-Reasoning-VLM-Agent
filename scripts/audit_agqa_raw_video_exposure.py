#!/usr/bin/env python3
"""Audit whether any official AGQA split still has raw-video-fresh videos."""

from __future__ import annotations

import argparse
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
from scripts.freeze_agqa_query_grounder_v2_qualification import (  # noqa: E402
    _exact_raw_runtime_exposed_videos,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/test_balanced.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("exposure audit is immutable")

    exposed, evidence = _exact_raw_runtime_exposed_videos(REPO)
    split_videos: set[str] = set()
    query_videos: set[str] = set()
    with zipfile.ZipFile(args.archive) as bundle, bundle.open(args.entry) as raw:
        for _, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            video_id = str(row["video_id"])
            split_videos.add(video_id)
            if str(row.get("structural", "")).casefold() == "query":
                query_videos.add(video_id)
    exposed_split = split_videos & exposed
    exposed_query = query_videos & exposed
    body = {
        "schema_version": "agqa-raw-video-exposure-audit-v1",
        "status": (
            "NO_RAW_VIDEO_FRESH_OFFICIAL_SPLIT_REMAINS"
            if exposed_split == split_videos
            else "RAW_VIDEO_FRESH_OFFICIAL_VIDEOS_REMAIN"
        ),
        "archive_sha256": _sha256(args.archive),
        "archive_entry": args.entry,
        "evidence_scan": evidence,
        "official_split_video_count": len(split_videos),
        "official_query_video_count": len(query_videos),
        "raw_exposed_official_split_video_count": len(exposed_split),
        "raw_exposed_official_query_video_count": len(exposed_query),
        "raw_fresh_official_split_video_count": len(split_videos - exposed_split),
        "raw_fresh_official_query_video_count": len(query_videos - exposed_query),
        "exposed_official_video_ids": sorted(exposed_split),
        "exposed_official_video_ids_sha256": stable_hash(sorted(exposed_split)),
        "selection_or_outcomes_read": False,
        "audit_interpretation": (
            "AGQA can support a newly locked task-disjoint replication, but not "
            "a new untouched-video claim in this repository history."
        ),
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"],
        "official_split_video_count": body["official_split_video_count"],
        "raw_exposed_official_split_video_count": body["raw_exposed_official_split_video_count"],
        "raw_fresh_official_split_video_count": body["raw_fresh_official_split_video_count"],
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
