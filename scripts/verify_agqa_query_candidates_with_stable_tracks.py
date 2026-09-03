#!/usr/bin/env python3
"""Add answer-blind stable-track verification to frozen AGQA V2 receipts."""

from __future__ import annotations

from dataclasses import asdict, replace
import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_query_grounder_v2 import (
    QueryGroundingV2Receipt,
    query_grounding_v2_from_dict,
)
from motif_transfer.agqa_track_verified_candidate import (
    track_verified_candidate_score,
)
from motif_transfer.contracts import stable_hash


FORBIDDEN = (
    "answer_read", "official_scene_graph_read", "functional_program_read",
    "source_controller_read", "target_outcome_read",
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("track-verified grounding output is immutable")

    source = json.loads(args.grounding.read_text())
    if source.get("status") != "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME":
        raise ValueError("input is not a frozen V2 grounding report")
    if any(source.get(key) for key in FORBIDDEN):
        raise ValueError("input grounding crossed its authority boundary")
    budgets = source.get("component_frame_budgets", {})
    frame_budget = int(budgets.get("sgdet_unique_and_model_presentations", 0))
    if frame_budget <= 0:
        raise ValueError("frozen SGDET frame budget is missing")

    implementation_sha = _file_hash(Path(__file__))
    input_sha = _file_hash(args.grounding)
    backend_sha = stable_hash({
        "protocol": "AGQA_QUERY_GROUNDER_V2_STABLE_TRACK_VERIFIER_V1",
        "input_report_sha256": source["report_sha256"],
        "input_file_sha256": input_sha,
        "implementation_sha256": implementation_sha,
        "formula": "GEOMETRIC_MEAN_RELATION_ACTION_DETECTION_LOG_PERSISTENCE",
        "frame_budget": frame_budget,
        "fitted_weights": False,
    })

    rows = []
    for raw in source["rows"]:
        parent = query_grounding_v2_from_dict(raw["receipt"])
        tracks = {row.track_id: row for row in parent.tracks}
        candidates = []
        components = None
        for candidate in parent.candidates:
            track = tracks[candidate.track_id]
            value = track_verified_candidate_score(
                candidate.confidence,
                track.confidence,
                len(track.evidence_frames),
                frame_budget,
            )
            candidates.append(replace(candidate, confidence=value.score))
            if components is None:
                components = asdict(value)
        receipt = QueryGroundingV2Receipt.create(
            task_id=parent.task_id,
            video_sha256=parent.video_sha256,
            semantic_slots_sha256=parent.semantic_slots_sha256,
            selected_frame_indices=parent.selected_frame_indices,
            selected_frame_sha256s=parent.selected_frame_sha256s,
            tracks=parent.tracks,
            events=parent.events,
            candidates=candidates,
            public_ontology_sha256=parent.public_ontology_sha256,
            grounder_backend_sha256=backend_sha,
            provider_calls=parent.provider_calls,
        )
        score = candidates[0].confidence if candidates else 0.0
        rows.append({
            **raw,
            "receipt": asdict(receipt),
            "candidate_confidence": score,
            "candidate_verification": components,
            "candidate_verification_method": (
                "GEOMETRIC_MEAN_RELATION_ACTION_DETECTION_LOG_PERSISTENCE"
            ),
            "parent_query_grounding_receipt_sha256": parent.receipt_sha256,
        })

    report = {
        **source,
        "schema_version": "agqa-query-grounder-v2-stable-track-verified-v1",
        "grounder_backend_sha256": backend_sha,
        "rows": rows,
        "candidate_verification": {
            "method": "GEOMETRIC_MEAN_RELATION_ACTION_DETECTION_LOG_PERSISTENCE",
            "formula": "(relation_action_support * track_detection_confidence * log1p(track_frames)/log1p(sgdet_frame_budget)) ** (1/3)",
            "sgdet_frame_budget": frame_budget,
            "fitted_weights": False,
            "global_threshold_applied_downstream": True,
            "answer_blind_at_runtime": True,
            "source_controller_read": False,
        },
        "parent_grounder_backend_sha256": source["grounder_backend_sha256"],
        "parent_grounding_report_sha256": source["report_sha256"],
        "parent_grounding_file_sha256": input_sha,
        "stable_track_verifier_implementation_sha256": implementation_sha,
        "answer_blind_query_candidate_verification": True,
        "answer_read": False,
        "official_scene_graph_read": False,
        "functional_program_read": False,
        "source_controller_read": False,
        "target_outcome_read": False,
    }
    report.pop("report_sha256", None)
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "tasks": len(rows),
        "candidates": sum(bool(row["receipt"]["candidates"]) for row in rows),
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
