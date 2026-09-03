#!/usr/bin/env python3
"""Range-extract only the frozen AGQA V4 reserve videos from Charades."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import cv2
import fsspec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _probe(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode downloaded video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = capture.read()
    capture.release()
    if not ok or frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"downloaded video failed decode checks: {path}")
    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps,
        "width": width,
        "height": height,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", type=Path, required=True,
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--archive-url",
        help="Transport-only mirror override; does not alter the frozen selection.",
    )
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    body = dict(selection)
    claimed = body.pop("manifest_sha256")
    if _stable_hash(body) != claimed:
        raise ValueError("AGQA V4 reserve selection hash mismatch")
    selection_status = str(selection.get("status", ""))
    if not (
        re.fullmatch(
            r"FROZEN_V\d+_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V\d+_CALLS",
            selection_status,
        )
        or re.fullmatch(
            r"FROZEN_V\d+_SELECTION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_FORMAL_LABEL_ACCESS",
            selection_status,
        )
        or re.fullmatch(
            r"FROZEN_V\d+_(?:DEVELOPMENT|QUALIFICATION)_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_OUTCOME_ACCESS",
            selection_status,
        )
        or selection_status
        == "FROZEN_BEFORE_VIDEO_DOWNLOAD_GROUNDING_OR_OUTCOMES"
    ):
        raise ValueError("AGQA reserve selection is not frozen")
    archive = selection["raw_video_archive"]
    video_rows = {str(row["video_id"]): row for row in selection["samples"]}
    total_videos = len(video_rows)
    receipts = {}
    if args.receipt.is_file():
        cached = json.loads(args.receipt.read_text())
        if cached.get("selection_manifest_sha256") != claimed:
            raise ValueError("download receipt belongs to another selection")
        receipts = {str(row["video_id"]): row for row in cached.get("videos", [])}

    archive_url = str(args.archive_url or archive["url"])
    with fsspec.open(
        archive_url, "rb", block_size=1024 * 1024,
        cache_type="readahead",
    ) as remote, zipfile.ZipFile(remote) as bundle:
        names = set(bundle.namelist())
        prefix = str(archive["archive_prefix"])
        for index, video_id in enumerate(sorted(video_rows), start=1):
            target = Path(video_rows[video_id]["video_path"])
            prior = receipts.get(video_id)
            if prior and target.is_file() and _sha256(target) == prior["sha256"]:
                print(json.dumps({
                    "cached": video_id,
                    "progress": f"{index}/{total_videos}",
                }), flush=True)
                continue
            member = f"{prefix}{video_id}.mp4"
            if member not in names:
                raise ValueError(f"official archive lacks {member}")
            target.parent.mkdir(parents=True, exist_ok=True)
            info = bundle.getinfo(member)
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{video_id}.", suffix=".part",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                try:
                    with bundle.open(info) as source:
                        shutil.copyfileobj(source, temporary, length=1024 * 1024)
                except Exception:
                    temporary_path.unlink(missing_ok=True)
                    raise
            if temporary_path.stat().st_size != info.file_size:
                temporary_path.unlink(missing_ok=True)
                raise ValueError(f"size mismatch for {video_id}")
            os.replace(temporary_path, target)
            receipts[video_id] = {
                "video_id": video_id,
                "archive_member": member,
                "file_size": info.file_size,
                "crc32": info.CRC,
                "sha256": _sha256(target),
                "probe": _probe(target),
                "output": str(target),
            }
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps({
                "schema_version": "agqa2-v4-reserve-download-v1",
                "status": "IN_PROGRESS",
                "selection_manifest_sha256": claimed,
                "archive_url": archive_url,
                "frozen_archive_url": archive["url"],
                "videos": [receipts[key] for key in sorted(receipts)],
            }, indent=2, sort_keys=True) + "\n")
            print(json.dumps({
                "downloaded": video_id, "bytes": info.file_size,
                "progress": f"{index}/{total_videos}",
            }), flush=True)
    complete = {
        "schema_version": "agqa2-v4-reserve-download-v1",
        "status": "COMPLETE",
        "selection_manifest_sha256": claimed,
        "archive_url": archive_url,
        "frozen_archive_url": archive["url"],
        "videos": [receipts[key] for key in sorted(receipts)],
    }
    args.receipt.write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "COMPLETE", "video_count": len(receipts),
        "total_bytes": sum(row["file_size"] for row in receipts.values()),
        "receipt_sha256": _sha256(args.receipt),
    }, indent=2))


if __name__ == "__main__":
    main()
