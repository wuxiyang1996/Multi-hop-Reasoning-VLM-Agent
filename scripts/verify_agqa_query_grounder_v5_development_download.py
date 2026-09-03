#!/usr/bin/env python3
"""Verify the content-addressed videos acquired for the frozen AGQA V5 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.freeze_agqa_query_grounder_v2_qualification import _sha256  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _verified_body(document: dict, hash_key: str) -> dict:
    body = dict(document)
    claimed = body.pop(hash_key)
    if stable_hash(body) != claimed:
        raise ValueError(f"{hash_key} mismatch")
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("download verification is immutable")

    public = _read(args.cohort_dir / "public_cohort.json")
    selection = _read(args.cohort_dir / "download_selection.json")
    receipt = _read(args.cohort_dir / "download_receipt.json")
    _verified_body(selection, "manifest_sha256")
    if selection["public_cohort_sha256"] != public["cohort_sha256"]:
        raise ValueError("selection/public cohort mismatch")
    if receipt.get("status") != "COMPLETE":
        raise ValueError("download receipt is not complete")
    if receipt.get("selection_manifest_sha256") != selection["manifest_sha256"]:
        raise ValueError("download receipt belongs to another selection")

    expected = {str(row["video_id"]): row for row in selection["samples"]}
    received = {str(row["video_id"]): row for row in receipt["videos"]}
    if set(expected) != set(received):
        raise ValueError("download receipt video set mismatch")
    content = []
    for video_id in sorted(expected):
        path = Path(expected[video_id]["video_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = _sha256(path)
        if actual_sha != received[video_id]["sha256"]:
            raise ValueError(f"content hash mismatch for {video_id}")
        content.append({
            "video_id": video_id,
            "video_path": str(path),
            "video_sha256": actual_sha,
            "probe": received[video_id]["probe"],
        })

    report = {
        "schema_version": "agqa-query-grounder-v5-development-download-verification-v1",
        "status": "PASS",
        "public_cohort_sha256": public["cohort_sha256"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "video_count": len(content),
        "video_content_receipts": content,
        "gates": {
            "selection_hash_valid": True,
            "cohort_binding_valid": True,
            "receipt_complete": True,
            "video_set_exact": True,
            "all_videos_decode_probed": all(row["probe"] for row in content),
            "all_content_hashes_match": True,
        },
        "answers_read": False,
        "functional_programs_read": False,
        "official_scene_graph_read": False,
        "target_outcomes_read": False,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "video_count": report["video_count"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
