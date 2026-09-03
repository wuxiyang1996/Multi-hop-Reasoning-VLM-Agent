#!/usr/bin/env python3
"""Select one global V3 confidence threshold on development outcomes only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from motif_transfer.agqa_query_grounder_v2 import query_grounding_v2_from_dict
from motif_transfer.agqa_query_object_grounder import canonical_object_label
from motif_transfer.contracts import stable_hash
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _answers(archive: Path, entry: str, wanted: set[str]) -> dict[str, str]:
    output = {}
    with zipfile.ZipFile(archive) as bundle, bundle.open(entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            if str(task_id) in wanted:
                output[str(task_id)] = canonical_object_label(str(row["answer"]))
                if len(output) == len(wanted):
                    break
    if set(output) != wanted:
        raise ValueError(f"missing {len(wanted - set(output))} development answers")
    return output


def wilson_lower(correct: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    p = correct / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (center - radius) / denominator


def select_threshold(curve: list[dict], constraints: dict) -> dict | None:
    eligible = [row for row in curve if (
        row["supported"] >= int(constraints["unique_supported_count_minimum"])
        and row["precision_wilson_95_lower"]
        >= float(constraints["unique_supported_precision_wilson_95_lower_bound_minimum"])
        and row["coverage"] >= float(constraints["unique_supported_coverage_minimum"])
    )]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["coverage"], row["threshold"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V3 calibration output is immutable")

    grounding = json.loads(args.grounding.read_text())
    protocol = json.loads(args.protocol.read_text())
    if grounding.get("status") != "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME":
        raise ValueError("development predictions were not frozen before outcomes")
    if any(grounding.get(key) for key in (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )):
        raise ValueError("development grounding crossed the authority boundary")
    if any(float(row["candidate_support_threshold"]) != 0.0 for row in grounding["rows"]):
        raise ValueError("calibration requires outcome-blind candidate generation at threshold zero")

    task_ids = {str(row["task_id"]) for row in grounding["rows"]}
    answers = _answers(args.archive, args.entry, task_ids)
    predictions = []
    for row in grounding["rows"]:
        receipt = query_grounding_v2_from_dict(row["receipt"])
        tracks = {track.track_id: track.canonical_label for track in receipt.tracks}
        candidate = receipt.candidates[0] if receipt.candidates else None
        predicted = tracks.get(candidate.track_id) if candidate else None
        confidence = float(candidate.confidence) if candidate else -1.0
        predictions.append({
            "task_id": receipt.task_id,
            "predicted": predicted,
            "confidence": confidence,
            "correct": predicted == answers[receipt.task_id] if predicted is not None else False,
        })

    selection = protocol["threshold_selection"]
    curve = []
    for threshold in selection["candidate_grid"]:
        selected = [row for row in predictions if row["confidence"] >= float(threshold)]
        correct = sum(bool(row["correct"]) for row in selected)
        total = len(selected)
        curve.append({
            "threshold": float(threshold),
            "supported": total,
            "correct": correct,
            "precision": correct / total if total else 0.0,
            "precision_wilson_95_lower": wilson_lower(correct, total),
            "coverage": total / len(predictions) if predictions else 0.0,
        })
    chosen = select_threshold(curve, selection["constraints"])
    body = {
        "schema_version": "agqa-query-grounder-v3-global-threshold-calibration-v1",
        "status": "V3_GLOBAL_THRESHOLD_QUALIFIED_ON_DEVELOPMENT" if chosen else "V3_CALIBRATION_FAILED",
        "grounding_report_sha256": grounding["report_sha256"],
        "grounding_file_sha256": _sha256(args.grounding),
        "protocol_file_sha256": _sha256(args.protocol),
        "development_tasks": len(predictions),
        "curve": curve,
        "selected": chosen,
        "selection_objective": selection["objective"],
        "selection_constraints": selection["constraints"],
        "one_global_threshold": True,
        "predicate_role_or_slice_specific_thresholds_used": False,
        "answers_read_for_development_calibration_only": True,
        "official_scene_graph_or_program_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"],
        "development_tasks": len(predictions),
        "selected": chosen,
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
