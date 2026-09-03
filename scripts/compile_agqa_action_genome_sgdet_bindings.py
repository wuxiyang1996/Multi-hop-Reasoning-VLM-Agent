#!/usr/bin/env python3
"""Compile answer-blind SGDET probabilities into AGQA candidate bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


PREDICATE_CHANNELS = {
    # Action Genome spatial logits are object -> person.  AGQA phrases below
    # ask for the object's relation to the person and therefore use inverses.
    "beneath": (("spatial_object_to_person", "above"),),
    "below": (("spatial_object_to_person", "above"),),
    "above": (("spatial_object_to_person", "beneath"),),
    "in front of": (("spatial_object_to_person", "behind"),),
    "behind": (("spatial_object_to_person", "in_front_of"),),
    "on the side of": (("spatial_object_to_person", "on_the_side_of"),),
    "in": (("spatial_object_to_person", "in"),),
    "touching": (("contact_person_to_object", "touching"),),
    "carrying": (("contact_person_to_object", "carrying"),),
    "holding": (("contact_person_to_object", "holding"),),
    "wearing": (("contact_person_to_object", "wearing"),),
    "sitting on": (("contact_person_to_object", "sitting_on"),),
    "standing on": (("contact_person_to_object", "standing_on"),),
    "leaning on": (("contact_person_to_object", "leaning_on"),),
    "watching": (("attention_person_to_object", "looking_at"),),
    "looking at": (("attention_person_to_object", "looking_at"),),
    "related to": (
        ("attention_person_to_object", "looking_at"),
        ("contact_person_to_object", "carrying"),
        ("contact_person_to_object", "holding"),
        ("contact_person_to_object", "touching"),
        ("contact_person_to_object", "wearing"),
        ("contact_person_to_object", "sitting_on"),
        ("contact_person_to_object", "standing_on"),
        ("contact_person_to_object", "leaning_on"),
        ("contact_person_to_object", "lying_on"),
        ("contact_person_to_object", "covered_by"),
    ),
    "taking": (
        ("contact_person_to_object", "carrying"),
        ("contact_person_to_object", "holding"),
    ),
    "putting down": (
        ("contact_person_to_object", "carrying"),
        ("contact_person_to_object", "holding"),
        ("contact_person_to_object", "touching"),
    ),
    "opening": (
        ("contact_person_to_object", "holding"),
        ("contact_person_to_object", "touching"),
    ),
    "closing": (
        ("contact_person_to_object", "holding"),
        ("contact_person_to_object", "touching"),
    ),
    "grasping": (
        ("contact_person_to_object", "holding"),
        ("contact_person_to_object", "touching"),
    ),
    "throwing": (
        ("contact_person_to_object", "carrying"),
        ("contact_person_to_object", "holding"),
    ),
    "washing": (
        ("contact_person_to_object", "wiping"),
        ("contact_person_to_object", "touching"),
    ),
    "tidying": (
        ("contact_person_to_object", "holding"),
        ("contact_person_to_object", "touching"),
    ),
    "working on": (
        ("attention_person_to_object", "looking_at"),
        ("contact_person_to_object", "touching"),
    ),
    "snuggling": (
        ("contact_person_to_object", "holding"),
        ("contact_person_to_object", "covered_by"),
    ),
    "undressing": (("contact_person_to_object", "wearing"),),
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_window(row: dict) -> list[int]:
    original = [int(value) for value in row["inspection_indices"]]
    lower = round(min(original) / 47 * 63)
    upper = round(max(original) / 47 * 63)
    obligations = [value for value in row.get("action_obligations", [])
                   if "argmax_window" in value]
    if obligations:
        anchor = int(obligations[0]["argmax_window"])
        anchor_lower = (0, 21, 42)[anchor]
        anchor_upper = (21, 42, 63)[anchor]
        temporal = str(row["temporal_operator"])
        if temporal == "BEFORE":
            lower, upper = 0, anchor_lower
        elif temporal == "AFTER":
            lower, upper = anchor_upper, 63
        else:
            lower, upper = anchor_lower, anchor_upper
        if lower == upper:
            lower, upper = anchor_lower, anchor_upper
    return [lower, upper]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("binding receipt is immutable")
    grounding = json.loads(args.grounding.read_text())
    temporal = json.loads(args.temporal.read_text())
    forbidden = (
        grounding["question_read"], grounding["answer_read"],
        grounding["functional_program_read"], grounding["official_scene_graph_read"],
        grounding["per_video_action_genome_annotation_read"],
        grounding["source_controller_read"], grounding["target_outcome_read"],
    )
    if any(forbidden) or grounding["mode"] != "sgdet":
        raise ValueError("grounding receipt violates prediction-only SGDET contract")
    by_video = {str(row["video_id"]): row for row in grounding["rows"]}
    object_scores = {
        video: {int(obj["detection_index"]): float(obj["score"])
                for obj in row["objects"]}
        for video, row in by_video.items()
    }
    outputs = []
    for task in temporal["rows"]:
        task_id = str(task["task_id"])
        video_id = task_id.split("-", 1)[0]
        if video_id not in by_video:
            continue
        predicate = str(task["predicate"]).strip().lower()
        channel_specs = PREDICATE_CHANNELS.get(predicate, ())
        window = dense_window(task)
        candidates = {}
        for channel, relation_name in channel_specs:
            for relation in by_video[video_id]["relations"]:
                frame = int(relation["sampled_frame_index"])
                if not window[0] <= frame <= window[1]:
                    continue
                label = str(relation["object_label"])
                relation_score = float(relation[channel][relation_name])
                detector_score = object_scores[video_id][int(relation["object_detection_index"])]
                joint = relation_score * detector_score
                current = candidates.get(label)
                if current is None or joint > current["joint_score"]:
                    candidates[label] = {
                        "candidate_label": label,
                        "joint_score": joint,
                        "relation_score": relation_score,
                        "object_score": detector_score,
                        "evidence_original_frame_index": int(relation["original_frame_index"]),
                    }
        ranked = sorted(candidates.values(), key=lambda value: (
            -value["joint_score"], -value["relation_score"],
            -value["object_score"], value["candidate_label"]))
        outputs.append({
            "task_id": task_id,
            "video_id": video_id,
            "predicate": predicate,
            "channels": [list(value) for value in channel_specs],
            "mapped_dense_window": window,
            "status": "BOUND" if ranked else "ABSTAIN",
            "top_candidate": ranked[0]["candidate_label"] if ranked else None,
            "candidates": ranked,
        })
    report = {
        "schema_version": "agqa-action-genome-sgdet-binding-receipt-v1",
        "status": "SGDET_BINDINGS_FROZEN_BEFORE_DEVELOPMENT_OUTCOME",
        "scoring_rule": "max_frame(relation_probability * predicted_object_probability)",
        "grounding_file_sha256": file_hash(args.grounding),
        "temporal_file_sha256": file_hash(args.temporal),
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
