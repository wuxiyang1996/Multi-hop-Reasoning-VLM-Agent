#!/usr/bin/env python3
"""Qualify frozen NS-DR perception on development annotations, never QA answers."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import zipfile

from motif_transfer.clevrer_nsdr_tool_grounder import (
    bind_cached_nsdr_prediction, load_prediction_payload,
)
from motif_transfer.contracts import stable_hash


ATTRS = ("color", "material", "shape")


def _signature(row: dict) -> tuple[str, str, str]:
    return tuple(str(row[key]) for key in ATTRS)  # type: ignore[return-value]


def _attribute_score(predicted: list[dict], gold: list[dict]) -> tuple[int, int]:
    if not predicted and not gold:
        return 0, 0
    left, right = (gold, predicted) if len(gold) <= len(predicted) else (predicted, gold)
    best = 0
    for assignment in itertools.permutations(range(len(right)), len(left)):
        score = sum(
            sum(str(row[key]) == str(right[index][key]) for key in ATTRS)
            for row, index in zip(left, assignment)
        )
        best = max(best, score)
    return best, 3 * max(len(predicted), len(gold))


def _collision_key(objects: list[dict]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted(_signature(row) for row in objects))


def _collision_matches(predicted: list[dict], gold: list[dict], tolerance: int = 5) -> int:
    candidates = []
    for p_index, pred in enumerate(predicted):
        for g_index, truth in enumerate(gold):
            if _collision_key(pred["objects"]) != _collision_key(truth["objects"]):
                continue
            delta = abs(int(pred["frame"]) - int(truth["frame"]))
            if delta <= tolerance:
                candidates.append((delta, p_index, g_index))
    used_pred = set(); used_gold = set()
    for _, p_index, g_index in sorted(candidates):
        if p_index not in used_pred and g_index not in used_gold:
            used_pred.add(p_index); used_gold.add(g_index)
    return len(used_pred)


def _annotation(bundle: zipfile.ZipFile, video_id: int) -> dict:
    suffix = f"annotation_{video_id}.json"
    names = [name for name in bundle.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"missing/ambiguous CLEVRER development annotation: {video_id}")
    return json.loads(bundle.read(names[0]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--grounder-config", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("CLEVRER grounder qualification is immutable")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    protocol = json.loads(args.preregistration.read_text(encoding="utf-8"))
    config = json.loads(args.grounder_config.read_text(encoding="utf-8"))
    if cohort.get("status") != "FROZEN_PUBLIC_COHORTS":
        raise ValueError("CLEVRER public cohort is not frozen")
    config_hash = stable_hash(config)
    exact = attribute_correct = attribute_total = collision_tp = collision_pred = collision_gold = 0
    counterfactual_complete = 0; rows = []
    raw_root = Path(config["raw_video_root"]); prediction_root = Path(config["prediction_root"])
    with zipfile.ZipFile(args.annotations) as bundle:
        for public in cohort["development"]:
            video_id = int(public["video_id"])
            video_matches = list(raw_root.glob(f"**/video_{video_id}.mp4"))
            if len(video_matches) != 1:
                raise ValueError(f"raw video lookup failed for {video_id}")
            prediction_path = prediction_root / f"sim_{video_id:05d}.json"
            receipt = bind_cached_nsdr_prediction(
                video_path=video_matches[0], prediction_path=prediction_path, config=config,
            )
            payload = load_prediction_payload(prediction_path)
            annotation = _annotation(bundle, video_id)
            # Development-only target-native grounder qualification.  No
            # question, functional program, or QA answer is accessed here.
            gold_objects = list(annotation["object_property"])
            predicted_objects = list(payload["objects"])
            exact_here = Counter(map(_signature, predicted_objects)) == Counter(map(_signature, gold_objects))
            correct_here, total_here = _attribute_score(predicted_objects, gold_objects)
            observed = next(row for row in payload["predictions"] if int(row["what_if"]) == -1)
            # The released NS-DR "what_if=-1" world rolls dynamics beyond the
            # decoded clip.  Events after the last raw frame are predictive,
            # not observed grounding claims, and are qualified separately by
            # the complete-world gate rather than counted as false positives.
            last_raw_frame = int(receipt.selected_frame_indices[-1])
            predicted_collisions = [
                row for row in observed.get("collisions") or ()
                if int(row["frame"]) <= last_raw_frame
            ]
            by_id = {int(row["object_id"]): row for row in gold_objects}
            gold_collisions = [{
                "frame": int(row["frame_id"]),
                "objects": [by_id[int(object_id)] for object_id in row["object_ids"]],
            } for row in annotation["collision"]]
            matches = _collision_matches(predicted_collisions, gold_collisions)
            exact += int(exact_here); attribute_correct += correct_here; attribute_total += total_here
            collision_tp += matches; collision_pred += len(predicted_collisions); collision_gold += len(gold_collisions)
            counterfactual_complete += int(receipt.counterfactual_worlds_complete)
            rows.append({
                "video_id": video_id, "grounder_receipt": asdict(receipt),
                "object_inventory_exact": exact_here,
                "attribute_correct": correct_here, "attribute_total": total_here,
                "predicted_collision_count": len(predicted_collisions),
                "observed_collision_boundary_source_frame": last_raw_frame,
                "gold_collision_count_development_only": len(gold_collisions),
                "matched_collision_count": matches,
                "qa_question_read": False, "qa_answer_read": False,
                "functional_program_read": False,
            })
    n = len(rows)
    metrics = {
        "development_videos": n,
        "object_inventory_exact_video_fraction": exact / n,
        "object_attribute_micro_accuracy": attribute_correct / attribute_total,
        "observed_collision_precision": collision_tp / collision_pred if collision_pred else 0.0,
        "observed_collision_recall": collision_tp / collision_gold if collision_gold else 0.0,
        "counterfactual_world_completeness": counterfactual_complete / n,
    }
    thresholds = protocol["grounder_qualification_gates"]
    metric_by_threshold = {
        "object_inventory_exact_video_fraction_minimum": "object_inventory_exact_video_fraction",
        "object_attribute_micro_accuracy_minimum": "object_attribute_micro_accuracy",
        "observed_collision_precision_minimum": "observed_collision_precision",
        "observed_collision_recall_minimum": "observed_collision_recall",
        "counterfactual_world_completeness": "counterfactual_world_completeness",
    }
    if set(thresholds) != set(metric_by_threshold):
        raise ValueError("grounder qualification threshold schema drift")
    gates = {
        name: metrics[metric_by_threshold[name]] >= float(value)
        for name, value in thresholds.items()
    }
    body = {
        "schema_version": "clevrer-nsdr-grounder-qualification-v1",
        "status": "CLEVRER_NSDR_GROUNDER_QUALIFIED" if all(gates.values()) else "CLEVRER_NSDR_GROUNDER_FAILED",
        "cohort_sha256": cohort["cohort_sha256"],
        "preregistration_sha256": stable_hash(protocol),
        "grounder_config_sha256": config_hash,
        "qualification_authority": "DEVELOPMENT_OBJECT_PROPERTIES_AND_COLLISIONS_ONLY;NO_QA_ANSWER_OR_PROGRAM",
        "rows": rows, "metrics": metrics, "thresholds": thresholds, "gates": gates,
        "reserve_read": False, "qa_answers_read": False, "functional_programs_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": body["status"], "metrics": metrics, "gates": gates,
                      "report_sha256": body["report_sha256"]}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
