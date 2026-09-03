#!/usr/bin/env python3
"""Range-extract only frozen V24 videos from the official Charades archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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


def _probe(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not decode downloaded video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    ok, _frame = capture.read()
    capture.release()
    if not ok or width <= 0 or height <= 0 or frame_count <= 0 or fps <= 0:
        raise ValueError(f"downloaded video failed decode metadata checks: {path}")
    return {
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps,
        "frame_count": frame_count,
        "fps": fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_V24_VIDEO_DOWNLOAD_OR_RUNTIME_OUTCOMES":
        raise ValueError("V24 fresh manifest is not sealed")
    expected_manifest = "a950ad10f4da7be6057c89131697a09a2f671fb0057838c23ff787e9613c61f2"
    if _sha256(args.manifest) != expected_manifest:
        raise ValueError("V24 fresh manifest hash mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = manifest["raw_video_archive"]
    url = str(archive["url"])
    prefix = str(archive["archive_prefix"])
    video_ids = sorted({str(row["video_id"]) for row in manifest["samples"]})
    existing_receipts = {}
    if args.receipt.is_file():
        prior = json.loads(args.receipt.read_text(encoding="utf-8"))
        if prior.get("manifest_sha256") != expected_manifest:
            raise ValueError("cached V24 download receipt uses another manifest")
        existing_receipts = {
            str(row["video_id"]): row for row in prior.get("videos", [])
        }

    with fsspec.open(
        url, "rb", block_size=1024 * 1024, cache_type="readahead",
    ) as remote:
        with zipfile.ZipFile(remote) as archive_zip:
            names = set(archive_zip.namelist())
            expected_names = {video_id: f"{prefix}{video_id}.mp4" for video_id in video_ids}
            missing = [
                video_id for video_id, name in expected_names.items() if name not in names
            ]
            if missing:
                raise ValueError(f"official Charades archive lacks V24 videos: {missing}")
            receipts = []
            for index, video_id in enumerate(video_ids, start=1):
                output = args.output_dir / f"{video_id}.mp4"
                cached = existing_receipts.get(video_id)
                if (
                    cached
                    and output.is_file()
                    and _sha256(output) == str(cached.get("sha256"))
                ):
                    receipts.append(cached)
                    print(json.dumps({
                        "cached": video_id, "progress": f"{index}/{len(video_ids)}",
                    }), flush=True)
                    continue
                info = archive_zip.getinfo(expected_names[video_id])
                with tempfile.NamedTemporaryFile(
                    dir=args.output_dir, prefix=f".{video_id}.", suffix=".part",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    try:
                        with archive_zip.open(info) as source:
                            shutil.copyfileobj(source, temporary, length=1024 * 1024)
                    except Exception:
                        temporary_path.unlink(missing_ok=True)
                        raise
                if temporary_path.stat().st_size != info.file_size:
                    temporary_path.unlink(missing_ok=True)
                    raise ValueError(f"range extraction size mismatch for {video_id}")
                os.replace(temporary_path, output)
                row = {
                    "video_id": video_id,
                    "archive_member": expected_names[video_id],
                    "compressed_size": int(info.compress_size),
                    "file_size": int(info.file_size),
                    "crc32": int(info.CRC),
                    "sha256": _sha256(output),
                    "ffprobe": _probe(output),
                    "output": str(output.resolve()),
                }
                receipts.append(row)
                args.receipt.parent.mkdir(parents=True, exist_ok=True)
                args.receipt.write_text(json.dumps({
                    "schema_version": 24,
                    "status": "V24_OFFICIAL_RANGE_DOWNLOAD_IN_PROGRESS",
                    "manifest": str(args.manifest.resolve()),
                    "manifest_sha256": expected_manifest,
                    "archive_url": url,
                    "archive_etag": archive["etag"],
                    "videos": receipts,
                }, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({
                    "downloaded": video_id,
                    "bytes": info.file_size,
                    "progress": f"{index}/{len(video_ids)}",
                }), flush=True)
    payload = {
        "schema_version": 24,
        "status": "V24_OFFICIAL_RANGE_DOWNLOAD_COMPLETE",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": expected_manifest,
        "archive_url": url,
        "archive_etag": archive["etag"],
        "video_count": len(receipts),
        "total_bytes": sum(int(row["file_size"]) for row in receipts),
        "videos": receipts,
    }
    args.receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "video_count": payload["video_count"],
        "total_bytes": payload["total_bytes"],
        "receipt_sha256": _sha256(args.receipt),
    }, indent=2))


if __name__ == "__main__":
    main()
