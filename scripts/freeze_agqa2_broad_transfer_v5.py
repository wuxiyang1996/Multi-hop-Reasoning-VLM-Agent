#!/usr/bin/env python3
"""Freeze a video-disjoint AGQA train reserve for broad-stack transfer V5."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from scripts.freeze_agqa2_oracle_query_fresh_v2 import _artifact, _stream  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402

PREREG = REPO / "configs/agqa2_broad_transfer_v5_preregistration.json"
OUTPUT = REPO / "runs/agqa2_broad_transfer_v5"


def main() -> None:
    prereg = json.loads(PREREG.read_text())
    if prereg["status"] != "FROZEN_BEFORE_V5_SELECTION_OR_OUTCOMES":
        raise ValueError("V5 preregistration is not frozen")
    excluded = set()
    for path in (
        REPO / "configs/agqa2_full_distribution_v62_manifest.json",
        REPO / "runs/agqa2_oracle_query_mdp_v2_fresh/public_cohort.json",
        REPO / "runs/agqa2_oracle_query_mdp_v3_transfer/public_cohort.json",
    ):
        value = json.loads(path.read_text())
        excluded.update(str(row["video_id"]) for row in value.get("samples", value.get("rows", ())))
        excluded.update(str(video) for video in value.get("selected_video_ids", ()))
    nonce = prereg["selection"]["selection_nonce"]
    candidates: dict[str, list[tuple[str, dict]]] = {}
    for task_id, row in _stream():
        video_id = str(row["video_id"])
        if video_id in excluded:
            continue
        public = {
            "task_id": task_id, "video_id": video_id,
            "question": str(row["question"]),
            "question_sha256": stable_hash(str(row["question"])),
            "runtime_answer_read": False,
            "runtime_functional_program_read": False,
            "runtime_sg_grounding_read": False,
        }
        rank = stable_hash([nonce, task_id, public["question_sha256"]])
        bucket = candidates.setdefault(video_id, [])
        bucket.append((rank, public)); bucket.sort(key=lambda item: item[0]); del bucket[3:]
    eligible = [video for video, rows in candidates.items() if len(rows) == 3]
    selected = sorted(eligible, key=lambda video: stable_hash([nonce, "video", video]))[:300]
    rows = sorted([row for video in selected for _, row in candidates[video]], key=lambda row: row["task_id"])
    if len(rows) != 900:
        raise ValueError("V5 cohort cardinality failure")
    public = _artifact({
        "schema_version": "agqa2-broad-transfer-public-v5",
        "role": "FRESH_RUNTIME_ONLY_FROZEN_BEFORE_EVALUATOR",
        "preregistration_sha256": stable_hash(prereg),
        "selected_video_ids": selected, "rows": rows,
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "public_cohort.json").write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    wanted = {row["task_id"] for row in rows}
    evaluator_rows = []
    for task_id, row in _stream():
        if task_id in wanted:
            evaluator_rows.append({"task_id": task_id, "answer": row["answer"]})
            if len(evaluator_rows) == len(wanted):
                break
    evaluator = _artifact({
        "schema_version": "agqa2-broad-transfer-evaluator-v5",
        "role": "EVALUATOR_ONLY_DO_NOT_OPEN_BEFORE_RUNTIME_FREEZE",
        "public_artifact_sha256": public["artifact_sha256"],
        "rows": sorted(evaluator_rows, key=lambda row: row["task_id"]),
    })
    (OUTPUT / "evaluator_only.json").write_text(json.dumps(evaluator, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "agqa2-broad-transfer-freeze-v5", "status": "PASSED",
        "tasks": len(rows), "unique_videos": len(selected),
        "all_videos_unseen_in_v62_v2_v3": not bool(set(selected) & excluded),
        "selection_outcome_fields_read": False,
        "public_artifact_sha256": public["artifact_sha256"],
        "evaluator_artifact_sha256": evaluator["artifact_sha256"],
    }
    (OUTPUT / "freeze_receipt.json").write_text(json.dumps(receipt | {"receipt_sha256": stable_hash(receipt)}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
