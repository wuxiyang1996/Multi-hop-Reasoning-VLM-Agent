#!/usr/bin/env python3
"""Bind AGQA action-object queries with frozen Charades SlowFast logits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash
from scripts.compile_agqa_action_genome_sgdet_bindings import dense_window
from scripts.build_agqa_action_genome_sgdet_query_plans import native_temporal_window


ACTION_PREFIXES = {
    "taking": ("taking ",),
    "putting down": ("putting ",),
    "opening": ("opening ",),
    "holding": ("holding ", "someone is holding "),
    "closing": ("closing ",),
    "playing on": ("playing with ", "working/playing on "),
    "washing": ("washing ", "wash "),
    "grasping": ("grasping ",),
    "tidying": ("tidying ", "tidying up "),
    "throwing": ("throwing ",),
}

OBJECT_TERMS = (
    ("cup/glass/bottle", "cup/glass/bottle"),
    ("paper/notebook", "paper/notebook"),
    ("phone/camera", "phone/camera"),
    ("closet/cabinet", "closet/cabinet"),
    ("sofa/couch", "sofa/couch"),
    ("refrigerator", "refrigerator"),
    ("television", "television"),
    ("doorknob", "doorknob"),
    ("groceries", "groceries"),
    ("sandwich", "sandwich"),
    ("blanket", "blanket"),
    ("picture", "picture"),
    ("medicine", "medicine"),
    ("laptop", "laptop"),
    ("pillow", "pillow"),
    ("mirror", "mirror"),
    ("window", "window"),
    ("vacuum", "vacuum"),
    ("clothes", "clothes"),
    ("towel", "towel"),
    ("broom", "broom"),
    ("table", "table"),
    ("floor", "floor"),
    ("chair", "chair"),
    ("shoes", "shoe"),
    ("shoe", "shoe"),
    ("book", "book"),
    ("door", "door"),
    ("box", "box"),
    ("food", "food"),
    ("dish", "dish"),
    ("bag", "bag"),
)

# In these public Charades labels, the named object is a destination or an
# instrument rather than the queried patient/theme.  Fail closed on them.
ROLE_AMBIGUOUS_CLASS_IDS = {"c009", "c038", "c044", "c081", "c087", "c126", "c127"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_matches(predicate: str, phrase: str) -> bool:
    if predicate == "undressing":
        return phrase == "someone is undressing"
    return any(phrase.startswith(prefix) for prefix in ACTION_PREFIXES.get(predicate, ()))


def object_label(predicate: str, class_id: str, phrase: str):
    if predicate == "undressing" and phrase == "someone is undressing":
        return "clothes"
    if class_id in ROLE_AMBIGUOUS_CLASS_IDS:
        return None
    for token, label in OBJECT_TERMS:
        if token in phrase:
            return label
    return None


def eligible_windows(plan: dict) -> list[int]:
    native_window = native_temporal_window(plan)
    native_views = plan.get("native_frame_index_views", ())
    if native_window is not None and native_views:
        lower, upper = native_window
        centers = [sum(int(x) for x in view) / len(view) for view in native_views]
        eligible = [index for index, center in enumerate(centers)
                    if lower <= center <= upper]
        if eligible:
            return eligible
        midpoint = (lower + upper) / 2
        return [min(range(len(centers)), key=lambda index: abs(centers[index] - midpoint))]
    lower, upper = dense_window(plan)
    lower48, upper48 = lower * 47 / 63, upper * 47 / 63
    centers = (15.5, 23.5, 31.5)
    eligible = [index for index, center in enumerate(centers)
                if lower48 <= center <= upper48]
    if eligible:
        return eligible
    # Deterministic nearest-view fallback for very narrow endpoint windows.
    midpoint = (lower48 + upper48) / 2
    return [min(range(3), key=lambda index: abs(centers[index] - midpoint))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-grounding", type=Path, required=True)
    parser.add_argument("--query-plans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("SlowFast action binding receipt is immutable")
    action = json.loads(args.action_grounding.read_text())
    plans = json.loads(args.query_plans.read_text())
    if not action.get("all_class_scores_stored"):
        raise ValueError("full frozen Charades class scores are required")
    if action["answers_read"] or action["official_program_read"] or action["official_scene_graph_read"]:
        raise ValueError("action receipt violates isolation contract")
    by_task = {str(row["task_id"]): row for row in action["rows"]}
    outputs = []
    for plan in plans["rows"]:
        task_id = str(plan["task_id"])
        predicate = str(plan["predicate"]).strip().lower()
        windows = eligible_windows(plan)
        candidates = {}
        for value in by_task[task_id]["all_class_scores"]:
            phrase = str(value["phrase"]).strip().lower()
            class_id = str(value["class_id"])
            if not action_matches(predicate, phrase):
                continue
            label = object_label(predicate, class_id, phrase)
            if label is None:
                continue
            scores = [float(score) for score in value["window_scores"]]
            score = max(scores[index] for index in windows)
            current = candidates.get(label)
            if current is None or score > current["action_score"]:
                candidates[label] = {
                    "candidate_label": label,
                    "action_score": score,
                    "class_id": class_id,
                    "class_phrase": phrase,
                    "eligible_window_indices": windows,
                    "window_scores": scores,
                }
        ranked = sorted(candidates.values(), key=lambda value: (
            -value["action_score"], value["class_id"], value["candidate_label"]))
        outputs.append({
            "task_id": task_id,
            "predicate": predicate,
            "video_id": by_task[task_id].get("video_id"),
            "video_sha256": by_task[task_id].get("video_sha256"),
            "source_frame_count": by_task[task_id].get("source_frame_count"),
            "native_frame_index_views": by_task[task_id].get("native_frame_index_views", []),
            "presented_frame_receipts": by_task[task_id].get("presented_frame_receipts", []),
            "status": "BOUND" if ranked else "ABSTAIN",
            "top_candidate": ranked[0]["candidate_label"] if ranked else None,
            "candidates": ranked,
        })
    report = {
        "schema_version": "agqa-slowfast-action-binding-receipt-v1",
        "status": "SLOWFAST_ACTION_BINDINGS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
        "routing_rule": "public typed predicate prefix and unambiguous public Charades patient/theme noun",
        "scoring_rule": "max frozen SlowFast probability over temporally eligible views",
        "action_grounding_file_sha256": file_hash(args.action_grounding),
        "query_plans_file_sha256": file_hash(args.query_plans),
        "answer_read": False,
        "official_scene_graph_read": False,
        "functional_program_read": False,
        "source_controller_read": False,
        "target_outcome_read": False,
        "rows": outputs,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
