#!/usr/bin/env python3
"""Hash-freeze a fresh AGQA2 public cohort, then seal evaluator outcomes."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from scripts.freeze_agqa2_oracle_query_cohort_v1 import iter_top_level_object  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402


ARCHIVE = Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip")
MEMBER = "AGQA_balanced/train_balanced.txt"
PREREG = REPO / "configs/agqa2_oracle_query_mdp_v2_fresh_preregistration.json"
OUTPUT = REPO / "runs/agqa2_oracle_query_mdp_v2_fresh"


def _stream():
    archive = zipfile.ZipFile(ARCHIVE)
    binary = archive.open(MEMBER)
    text = io.TextIOWrapper(binary, encoding="utf-8")
    try:
        yield from iter_top_level_object(text)
    finally:
        text.close(); archive.close()


def _artifact(body):
    return body | {"artifact_sha256": stable_hash(body)}


def main() -> None:
    prereg = json.loads(PREREG.read_text())
    if prereg["status"] != "FROZEN_BEFORE_V2_SELECTION_OR_OUTCOMES":
        raise ValueError("fresh V2 preregistration is not frozen")
    nonce = prereg["selection"]["selection_nonce"]
    old = json.loads((REPO / "configs/agqa2_full_distribution_v62_manifest.json").read_text())
    excluded_videos = {str(row["video_id"]) for row in old["samples"]}
    # Keep the three smallest public hashes per non-consumed video. The code
    # does not access answer/program/sg_grounding in this selection pass.
    candidates: dict[str, list[tuple[str, dict]]] = {}
    for task_id, row in _stream():
        video_id = str(row["video_id"])
        if video_id in excluded_videos:
            continue
        public = {
            "task_id": task_id, "video_id": video_id,
            "question": str(row["question"]),
            "question_sha256": stable_hash(str(row["question"])),
            "answer_type": str(row.get("ans_type", "")),
            "semantic": str(row.get("semantic", "")),
            "structural": str(row.get("structural", "")),
            "runtime_answer_read": False,
            "runtime_functional_program_read": False,
            "runtime_sg_grounding_read": False,
        }
        rank = stable_hash([nonce, task_id, public["question_sha256"]])
        bucket = candidates.setdefault(video_id, [])
        bucket.append((rank, public)); bucket.sort(key=lambda value: value[0])
        del bucket[3:]
    eligible = [video for video, rows in candidates.items() if len(rows) == 3]
    selected_videos = sorted(
        eligible, key=lambda video: stable_hash([nonce, "video", video])
    )[:300]
    public_rows = [
        row for video in selected_videos for _, row in candidates[video]
    ]
    if len(public_rows) != 900 or len(set(selected_videos)) != 300:
        raise ValueError("fresh AGQA2 cohort cardinality failure")
    public_body = {
        "schema_version": "agqa2-oracle-query-public-cohort-v2-fresh",
        "role": "FRESH_RUNTIME_ONLY_FROZEN_BEFORE_EVALUATOR",
        "preregistration_sha256": stable_hash(prereg),
        "excluded_v62_video_count": len(excluded_videos),
        "selected_video_ids": selected_videos,
        "rows": sorted(public_rows, key=lambda row: row["task_id"]),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    public_artifact = _artifact(public_body)
    (OUTPUT / "public_cohort.json").write_text(
        json.dumps(public_artifact, indent=2, sort_keys=True) + "\n"
    )

    # Only after the public selection is durable do we reopen the official file
    # and project evaluator-only fields for the already frozen identities.
    selected_ids = {row["task_id"] for row in public_rows}
    evaluator_rows = []
    for task_id, row in _stream():
        if task_id not in selected_ids:
            continue
        evaluator_rows.append({
            "task_id": task_id, "answer": row["answer"],
            "functional_program_sha256": stable_hash(row.get("program")),
            "sg_grounding_sha256": stable_hash(row.get("sg_grounding")),
        })
        if len(evaluator_rows) == len(selected_ids):
            break
    evaluator_body = {
        "schema_version": "agqa2-oracle-query-evaluator-v2-fresh",
        "role": "EVALUATOR_ONLY_OPEN_AFTER_RUNTIME_PREDICTIONS_FREEZE",
        "public_artifact_sha256": public_artifact["artifact_sha256"],
        "rows": sorted(evaluator_rows, key=lambda row: row["task_id"]),
    }
    evaluator_artifact = _artifact(evaluator_body)
    (OUTPUT / "evaluator_only.json").write_text(
        json.dumps(evaluator_artifact, indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "schema_version": "agqa2-oracle-query-v2-fresh-freeze-receipt",
        "status": "PASSED",
        "tasks": 900, "unique_videos": 300,
        "all_videos_unseen_in_v62": not bool(set(selected_videos) & excluded_videos),
        "selection_outcome_fields_read": False,
        "public_artifact_sha256": public_artifact["artifact_sha256"],
        "evaluator_artifact_sha256": evaluator_artifact["artifact_sha256"],
    }
    (OUTPUT / "freeze_receipt.json").write_text(
        json.dumps(receipt | {"receipt_sha256": stable_hash(receipt)}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
