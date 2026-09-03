#!/usr/bin/env python3
"""Train a question-only multi-class AGQA program-type router.

Only the frozen router-train and router-validation video partitions expose
functional programs.  Every formal-holdout row is skipped before accessing
its program, answer, or scene graph.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Sequence
import zipfile

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
from motif_transfer.agqa_active_frame_grounder import parse_public_question_plan  # noqa: E402
from motif_transfer.agqa_program_transfer import (  # noqa: E402
    COMPOSITE_ROUTE,
    RELATION_ROUTE,
    TEMPORAL_PAIR_ROUTE,
    TEMPORAL_SINGLE_ROUTE,
    profile_program,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402


SUPPORTED_ROUTES = (
    RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE,
)
ALL_ROUTES = SUPPORTED_ROUTES + (COMPOSITE_ROUTE,)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_class_threshold(
    *, labels: Sequence[str], predicted: Sequence[str], scores: Sequence[float],
    route: str, plan_routes: Sequence[str], minimum_precision: float,
    minimum_selected: int,
) -> dict:
    values = sorted({float(score) for score in scores}, reverse=True)
    candidates = []
    route_positives = sum(label == route for label in labels)
    for threshold in values:
        selected = [
            index for index, (prediction, score, plan_route) in enumerate(
                zip(predicted, scores, plan_routes)
            )
            if prediction == route and plan_route == route and score >= threshold
        ]
        if len(selected) < minimum_selected:
            continue
        true_positive = sum(labels[index] == route for index in selected)
        precision = true_positive / len(selected)
        if precision < minimum_precision:
            continue
        recall = true_positive / route_positives if route_positives else 0.0
        candidates.append((recall, precision, len(selected), threshold, {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "selected": len(selected),
            "true_positive": true_positive,
            "false_positive": len(selected) - true_positive,
            "route_positives": route_positives,
        }))
    if not candidates:
        raise ValueError(f"no qualified threshold for {route}")
    return max(candidates, key=lambda item: item[:4])[4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", type=Path,
        default=REPO_ROOT / "configs/agqa2_program_router_video_split_v1.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "runs/agqa2_multiclass_program_router_v2",
    )
    args = parser.parse_args()
    split_path = args.split.resolve()
    output = args.output_dir.resolve()
    model_path = output / "router.joblib"
    report_path = output / "qualification_report.json"
    if model_path.exists() or report_path.exists():
        raise FileExistsError("multi-class router artifacts are immutable")
    split = json.loads(split_path.read_text())
    split_body = dict(split)
    split_claimed = split_body.pop("split_sha256")
    if stable_hash(split_body) != split_claimed:
        raise ValueError("router split content hash mismatch")
    train_videos = set(split["partitions"]["router_train"])
    validation_videos = set(split["partitions"]["router_validation"])
    formal_videos = set(split["partitions"]["formal_holdout"])
    texts = {"train": [], "validation": []}
    labels = {"train": [], "validation": []}
    plan_routes = {"train": [], "validation": []}
    formal_rows_skipped_before_program_access = 0
    with zipfile.ZipFile(split["archive_path"]) as bundle, bundle.open(split["entry"]) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            video_id = str(row["video_id"])
            if video_id in formal_videos:
                formal_rows_skipped_before_program_access += 1
                continue
            bucket = (
                "train" if video_id in train_videos
                else "validation" if video_id in validation_videos
                else None
            )
            if bucket is None:
                continue
            question = str(row["question"])
            plan = parse_public_question_plan(question)
            if plan is None:
                continue
            oracle = profile_program(
                task_id=task_id, program=str(row["program"]),
            ).route_kind
            if oracle not in ALL_ROUTES:
                continue
            texts[bucket].append(question)
            labels[bucket].append(oracle)
            plan_routes[bucket].append(plan.obligation_kind)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3,
            max_features=120000, sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            C=2.0, max_iter=300, solver="liblinear", random_state=0,
        )),
    ])
    pipeline.fit(texts["train"], labels["train"])
    validation_probabilities = pipeline.predict_proba(texts["validation"])
    classes = list(pipeline.classes_)
    predicted_indices = np.argmax(validation_probabilities, axis=1)
    predicted = [classes[index] for index in predicted_indices]
    predicted_scores = [
        float(validation_probabilities[index, class_index])
        for index, class_index in enumerate(predicted_indices)
    ]
    thresholds = {
        route: select_class_threshold(
            labels=labels["validation"], predicted=predicted,
            scores=predicted_scores, route=route,
            plan_routes=plan_routes["validation"], minimum_precision=0.98,
            minimum_selected=100,
        )
        for route in SUPPORTED_ROUTES
    }
    selected_indices = [
        index for index, (prediction, score, plan_route) in enumerate(zip(
            predicted, predicted_scores, plan_routes["validation"],
        ))
        if prediction in thresholds
        and plan_route == prediction
        and score >= thresholds[prediction]["threshold"]
    ]
    selected_correct = sum(
        labels["validation"][index] == predicted[index]
        for index in selected_indices
    )
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(pipeline, model_path, compress=3)
    report_body = {
        "schema_version": "agqa2-multiclass-program-router-qualification-v2",
        "status": (
            "MULTICLASS_PROGRAM_ROUTER_V2_QUALIFIED"
            if selected_indices
            and selected_correct / len(selected_indices) >= 0.98
            and all(value["precision"] >= 0.98 for value in thresholds.values())
            else "MULTICLASS_PROGRAM_ROUTER_V2_NOT_QUALIFIED"
        ),
        "split_file_sha256": _sha256(split_path),
        "split_sha256": split_claimed,
        "formal_video_labels_read": False,
        "formal_rows_skipped_before_program_access": formal_rows_skipped_before_program_access,
        "v1_formal_outcomes_used_for_training_or_thresholds": False,
        "features": "QUESTION_TEXT_ONLY_CHAR_WB_3_5_TFIDF_PLUS_DETERMINISTIC_PLAN_AGREEMENT",
        "classes": classes,
        "supported_routes": list(SUPPORTED_ROUTES),
        "training": {
            "rows": len(texts["train"]),
            "class_counts": dict(sorted(Counter(labels["train"]).items())),
        },
        "validation": {
            "rows": len(texts["validation"]),
            "class_counts": dict(sorted(Counter(labels["validation"]).items())),
            "confusion_labels": classes,
            "confusion_matrix": confusion_matrix(
                labels["validation"], predicted, labels=classes,
            ).tolist(),
            "thresholds": thresholds,
            "joint_selection": {
                "selected": len(selected_indices),
                "correct": selected_correct,
                "precision": selected_correct / len(selected_indices),
            },
        },
        "model_path": str(model_path.relative_to(REPO_ROOT)),
        "model_file_sha256": _sha256(model_path),
        "runtime_visible_fields": ["public_question_text"],
        "runtime_forbidden_fields": [
            "answer", "program", "scene_graph", "video", "source_identity",
        ],
    }
    report = report_body | {"report_sha256": stable_hash(report_body)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "training": report["training"],
        "validation": report["validation"],
        "model_file_sha256": report["model_file_sha256"],
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
