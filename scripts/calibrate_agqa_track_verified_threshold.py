#!/usr/bin/env python3
"""Calibrate one global stable-track verification threshold on development."""

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


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_lower(correct: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total
    )
    return (center - radius) / denominator


def select_threshold(curve: list[dict], constraints: dict) -> dict | None:
    eligible = [row for row in curve if (
        row["supported"] >= int(constraints["unique_supported_count_minimum"])
        and row["coverage"] >= float(constraints["unique_supported_coverage_minimum"])
        and row["precision_wilson_95_lower"]
        >= float(constraints["unique_supported_precision_wilson_95_lower_bound_minimum"])
    )]
    if not eligible:
        return None
    # Maximum coverage is the preregistered objective.  A higher threshold is
    # only a deterministic tie-break, never a predicate-specific exception.
    return max(eligible, key=lambda row: (row["coverage"], row["threshold"]))


def _answers(archive: Path, entry: str, wanted: set[str]) -> dict[str, str]:
    output = {}
    with zipfile.ZipFile(archive) as bundle, bundle.open(entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            task_id = str(task_id)
            if task_id in wanted:
                output[task_id] = canonical_object_label(str(row["answer"]))
                if len(output) == len(wanted):
                    break
    if set(output) != wanted:
        raise ValueError(f"missing {len(wanted - set(output))} development answers")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("track-verifier calibration output is immutable")

    grounding = json.loads(args.grounding.read_text())
    protocol = json.loads(args.protocol.read_text())
    if grounding.get("schema_version") != "agqa-query-grounder-v2-stable-track-verified-v1":
        raise ValueError("grounding does not contain the stable-track verifier")
    if grounding.get("candidate_verification", {}).get("fitted_weights") is not False:
        raise ValueError("candidate verifier must have no fitted weights")
    if grounding.get("candidate_verification", {}).get("source_controller_read") is not False:
        raise ValueError("candidate verifier must not read the source controller")
    if any(grounding.get(key) for key in (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )):
        raise ValueError("development grounding crossed its authority boundary")
    expected = protocol["development_grounder"]
    if expected["grounding_file_sha256"] != _file_hash(args.grounding):
        raise ValueError("development protocol is not bound to this grounding file")
    if expected["grounding_report_sha256"] != grounding["report_sha256"]:
        raise ValueError("development grounding report hash changed")

    wanted = {str(row["task_id"]) for row in grounding["rows"]}
    answers = _answers(args.archive, args.entry, wanted)
    scored = []
    for raw in grounding["rows"]:
        receipt = query_grounding_v2_from_dict(raw["receipt"])
        candidate = receipt.candidates[0] if receipt.candidates else None
        tracks = {row.track_id: row.canonical_label for row in receipt.tracks}
        predicted = tracks.get(candidate.track_id) if candidate else None
        scored.append({
            "task_id": receipt.task_id,
            "score": float(candidate.confidence) if candidate else 0.0,
            "candidate_exists": candidate is not None,
            "correct": predicted == answers[receipt.task_id] if predicted else False,
        })

    selection = protocol["threshold_selection"]
    curve = []
    for threshold in selection["candidate_grid"]:
        selected = [
            row for row in scored
            if row["candidate_exists"] and row["score"] >= float(threshold)
        ]
        correct = sum(bool(row["correct"]) for row in selected)
        total = len(selected)
        curve.append({
            "threshold": float(threshold),
            "supported": total,
            "correct": correct,
            "precision": correct / total if total else 0.0,
            "precision_wilson_95_lower": wilson_lower(correct, total),
            "coverage": total / len(scored) if scored else 0.0,
        })
    chosen = select_threshold(curve, selection["constraints"])
    body = {
        "schema_version": "agqa-track-verified-threshold-calibration-v1",
        "status": (
            "TRACK_VERIFIED_GLOBAL_THRESHOLD_QUALIFIED_ON_DEVELOPMENT"
            if chosen else "TRACK_VERIFIED_GLOBAL_THRESHOLD_NOT_QUALIFIED"
        ),
        "grounding_file_sha256": _file_hash(args.grounding),
        "grounding_report_sha256": grounding["report_sha256"],
        "protocol_file_sha256": _file_hash(args.protocol),
        "development_tasks": len(scored),
        "curve": curve,
        "selected": chosen,
        "selection_objective": selection["objective"],
        "selection_constraints": selection["constraints"],
        "one_global_threshold": True,
        "predicate_role_or_slice_specific_thresholds_used": False,
        "candidate_verifier_fitted_weights": False,
        "source_controller_read": False,
        "answers_read_for_development_calibration_only": True,
        "official_scene_graph_or_program_read": False,
        "target_outcome_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "development_tasks": len(scored),
        "selected": chosen, "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
