#!/usr/bin/env python3
"""Audit original-coordinate SGDET boxes against native AGQA video dimensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from motif_transfer.contracts import stable_hash
def _inside(box, width: int, height: int) -> bool:
    x1, y1, x2, y2 = (float(value) for value in box)
    return 0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height


def _intersects_after_native_clamp(box, width: int, height: int) -> bool:
    x1, y1, x2, y2 = (float(value) for value in box)
    return max(0.0, x1) < min(float(width), x2) and max(0.0, y1) < min(float(height), y2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("coordinate audit is immutable")
    sgdet = json.loads(args.sgdet.read_text())
    rows = []
    total = native_inside = native_intersecting = 0
    for video in sgdet["rows"]:
        capture = cv2.VideoCapture(str(video["video_path"]))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if width <= 0 or height <= 0:
            raise RuntimeError(f"invalid video dimensions: {video['video_path']}")
        boxes = [object_row["bbox_xyxy"] for object_row in video["objects"]]
        inside = sum(_inside(box, width, height) for box in boxes)
        intersecting = sum(
            _intersects_after_native_clamp(box, width, height) for box in boxes
        )
        total += len(boxes)
        native_inside += inside
        native_intersecting += intersecting
        rows.append({
            "video_id": video["video_id"], "native_width": width,
            "native_height": height, "box_count": len(boxes),
            "native_identity_in_bounds": inside,
            "native_identity_intersects_after_clamp": intersecting,
        })
    report = {
        "schema_version": "agqa-sgdet-native-box-coordinate-audit-v3",
        "status": (
            "PASS_NATIVE_IDENTITY"
            if native_inside == total
            else "PASS_NATIVE_IDENTITY_WITH_RENDER_CLAMP"
            if native_intersecting == total
            else "FAIL_NONINTERSECTING_NATIVE_BOXES"
        ),
        "videos": len(rows), "boxes": total,
        "native_identity_in_bounds_fraction": native_inside / total if total else 1.0,
        "native_identity_intersects_after_clamp_fraction": (
            native_intersecting / total if total else 1.0
        ),
        "coordinate_transform": "native_xyxy = receipt_bbox_xyxy (identity)",
        "producer_contract": (
            "official SGDET object_detector divides PRED_BOXES by preprocessing "
            "scale before entry['boxes'] is serialized"
        ),
        "answer_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read": False,
        "target_outcome_read": False, "rows": rows,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: report[key] for key in (
            "status", "videos", "boxes", "native_identity_in_bounds_fraction",
            "native_identity_intersects_after_clamp_fraction", "report_sha256",
        )
    }, indent=2))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
