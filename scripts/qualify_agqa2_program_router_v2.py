#!/usr/bin/env python3
"""Fail-closed V2 qualification for the already-trained AGQA router."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
from motif_transfer.agqa_active_frame_grounder import parse_public_question_plan  # noqa: E402
from motif_transfer.agqa_program_transfer import RELATION_ROUTE, profile_program  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402
from scripts.train_agqa2_program_router_v1 import select_threshold  # noqa: E402


SPLIT = REPO_ROOT / "configs/agqa2_program_router_video_split_v1.json"
MODEL = REPO_ROOT / "runs/agqa2_program_router_v1/router.joblib"
PARENT = REPO_ROOT / "runs/agqa2_program_router_v1/qualification_report.json"
OUTPUT = REPO_ROOT / "runs/agqa2_program_router_v1/qualification_report_v2.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("V2 qualification is immutable once written")
    split = json.loads(SPLIT.read_text())
    validation_videos = set(split["partitions"]["router_validation"])
    formal_videos = set(split["partitions"]["formal_holdout"])
    texts, labels = [], []
    formal_rows_skipped = 0
    with zipfile.ZipFile(split["archive_path"]) as bundle, bundle.open(split["entry"]) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            video_id = str(row["video_id"])
            if video_id in formal_videos:
                formal_rows_skipped += 1
                continue
            if video_id not in validation_videos:
                continue
            question = str(row["question"])
            plan = parse_public_question_plan(question)
            if plan is None or plan.comparison != "EXISTS":
                continue
            texts.append(question)
            labels.append(int(profile_program(task_id=task_id, program=str(row["program"])).route_kind == RELATION_ROUTE))
    model = joblib.load(MODEL)
    y = np.asarray(labels, dtype=np.int64)
    scores = model.predict_proba(texts)[:, 1]
    selection = select_threshold(y, scores, minimum_precision=1.0, minimum_selected=100)
    body = {
        "schema_version": "agqa2-program-router-qualification-v2",
        "status": "PROGRAM_ROUTER_V2_QUALIFIED" if selection["precision"] == 1.0 and selection["selected"] >= 100 else "PROGRAM_ROUTER_V2_NOT_QUALIFIED",
        "fix": "MAXIMIZE_RECALL_THEN_PRECISION_INSTEAD_OF_RECALL_THEN_SELECTED_COUNT",
        "split_file_sha256": _sha256(SPLIT),
        "model_file_sha256": _sha256(MODEL),
        "parent_v1_report_file_sha256": _sha256(PARENT),
        "formal_video_labels_read": False,
        "formal_rows_skipped_before_program_access": formal_rows_skipped,
        "validation": {"rows": len(texts), "positive": int(y.sum()), "negative": len(y) - int(y.sum()), "roc_auc": float(roc_auc_score(y, scores)), "selection": selection},
        "runtime_visible_fields": ["public_question_text"],
        "runtime_forbidden_fields": ["answer", "program", "scene_graph", "video", "source_identity"],
    }
    result = body | {"report_sha256": stable_hash(body)}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "validation": result["validation"], "report_sha256": result["report_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
