#!/usr/bin/env python3
"""Train a question-only AGQA program-family router on video-disjoint train data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
from motif_transfer.agqa_active_frame_grounder import parse_public_question_plan  # noqa: E402
from motif_transfer.agqa_program_transfer import RELATION_ROUTE, profile_program  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_threshold(labels: np.ndarray, scores: np.ndarray, *, minimum_precision: float, minimum_selected: int) -> dict:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    candidates = []
    for index, threshold in enumerate(thresholds):
        selected = scores >= threshold
        count = int(selected.sum())
        if count < minimum_selected:
            continue
        true_positive = int(labels[selected].sum())
        candidate_precision = true_positive / count
        if candidate_precision >= minimum_precision:
            candidates.append((float(recall[index]), count, float(threshold), candidate_precision, true_positive))
    if not candidates:
        raise ValueError("no validation threshold satisfies frozen precision and support gates")
    recall_value, count, threshold, precision_value, true_positive = max(
        candidates,
        key=lambda item: (item[0], item[3], -item[1], item[2]),
    )
    return {
        "threshold": threshold,
        "precision": precision_value,
        "recall": recall_value,
        "selected": count,
        "true_positive": true_positive,
        "false_positive": count - true_positive,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="configs/agqa2_program_router_video_split_v1.json", type=Path)
    parser.add_argument("--output-dir", default="runs/agqa2_program_router_v1", type=Path)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    model_path = output / "router.joblib"
    report_path = output / "qualification_report.json"
    if report_path.exists():
        raise FileExistsError("qualified router report is immutable once written")
    resume_trained_model = model_path.exists()
    split_path = args.split if args.split.is_absolute() else REPO_ROOT / args.split
    split = json.loads(split_path.read_text())
    body = dict(split)
    claimed = body.pop("split_sha256")
    if stable_hash(body) != claimed:
        raise ValueError("router split content hash mismatch")
    train_videos = set(split["partitions"]["router_train"])
    validation_videos = set(split["partitions"]["router_validation"])
    formal_videos = set(split["partitions"]["formal_holdout"])
    train_texts, train_labels, validation_texts, validation_labels = [], [], [], []
    formal_rows_skipped_before_program_access = 0
    with zipfile.ZipFile(split["archive_path"]) as bundle, bundle.open(split["entry"]) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8")
        for task_id, row in _iter_top_level_object(text):
            video_id = str(row["video_id"])
            if video_id in formal_videos:
                formal_rows_skipped_before_program_access += 1
                continue
            if video_id not in train_videos and video_id not in validation_videos:
                continue
            question = str(row["question"])
            plan = parse_public_question_plan(question)
            if plan is None or plan.comparison != "EXISTS":
                continue
            label = int(profile_program(task_id=task_id, program=str(row["program"])).route_kind == RELATION_ROUTE)
            if video_id in train_videos:
                train_texts.append(question)
                train_labels.append(label)
            else:
                validation_texts.append(question)
                validation_labels.append(label)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120000, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=2.0, max_iter=300, solver="liblinear", random_state=0)),
    ])
    if resume_trained_model:
        pipeline = joblib.load(model_path)
    else:
        pipeline.fit(train_texts, train_labels)
    labels = np.asarray(validation_labels, dtype=np.int64)
    scores = pipeline.predict_proba(validation_texts)[:, 1]
    threshold = select_threshold(labels, scores, minimum_precision=0.98, minimum_selected=100)
    output.mkdir(parents=True, exist_ok=True)
    if not resume_trained_model:
        joblib.dump(pipeline, model_path, compress=3)
    report_body = {
        "schema_version": "agqa2-program-router-qualification-v1",
        "status": "PROGRAM_ROUTER_QUALIFIED" if threshold["precision"] >= 0.98 and threshold["selected"] >= 100 else "PROGRAM_ROUTER_NOT_QUALIFIED",
        "split_file_sha256": _sha256(split_path),
        "split_sha256": claimed,
        "formal_video_labels_read": False,
        "formal_rows_skipped_before_program_access": formal_rows_skipped_before_program_access,
        "resumed_existing_trained_model_after_report_path_error": resume_trained_model,
        "features": "QUESTION_TEXT_ONLY_CHAR_WB_3_5_TFIDF",
        "training": {"rows": len(train_texts), "positive": int(sum(train_labels)), "negative": len(train_labels) - int(sum(train_labels))},
        "validation": {"rows": len(validation_texts), "positive": int(labels.sum()), "negative": len(labels) - int(labels.sum()), "roc_auc": float(roc_auc_score(labels, scores)), "selection": threshold},
        "model_path": str(model_path.relative_to(REPO_ROOT)),
        "model_file_sha256": _sha256(model_path),
        "runtime_visible_fields": ["public_question_text"],
        "runtime_forbidden_fields": ["answer", "program", "scene_graph", "video", "source_identity"],
    }
    report = report_body | {"report_sha256": stable_hash(report_body)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "training": report["training"], "validation": report["validation"], "model_file_sha256": report["model_file_sha256"], "report_sha256": report["report_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
