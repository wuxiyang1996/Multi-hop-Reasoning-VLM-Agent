#!/usr/bin/env python3
"""Apply frozen binary event verification to AGQA Query Grounding V2 receipts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_query_grounder_v2 import (
    QueryCandidateEvidence, QueryGroundingV2Receipt, query_grounding_v2_from_dict,
)
from motif_transfer.contracts import stable_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-grounding", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--consumed-development-pilot", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("binary-verified query grounding is immutable")
    candidate = json.loads(args.candidate_grounding.read_text())
    verification = json.loads(args.verification.read_text())
    protocol = json.loads(args.protocol.read_text())
    expected_candidate_status = (
        "CONSUMED_DEVELOPMENT_QUERY_GROUNDING_V2_NOT_TRANSFER_EVIDENCE"
        if args.consumed_development_pilot
        else "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME"
    )
    if candidate.get("status") != expected_candidate_status:
        raise ValueError("candidate grounding is not outcome blind")
    if verification.get("status") != "BINARY_EVENT_VERIFICATION_ANSWER_BLIND":
        raise ValueError("binary verification is not outcome blind")
    if bool(verification.get("consumed_development_pilot")) != bool(
        args.consumed_development_pilot
    ):
        raise ValueError("verification phase label differs from candidate grounding")
    if verification["candidate_grounding_report_sha256"] != candidate["report_sha256"]:
        raise ValueError("candidate grounding and verification differ")
    if protocol["immutable_inputs"]["candidate_grounding_file_sha256"] != _sha256(args.candidate_grounding):
        raise ValueError("candidate grounding differs from binary-verifier protocol")
    by_task = {str(row["task_id"]): row for row in verification["rows"]}
    if set(by_task) != {str(row["task_id"]) for row in candidate["rows"]}:
        raise ValueError("binary verification task set differs")
    backend_sha = stable_hash({
        "protocol": "AGQA_BINARY_VERIFIED_QUERY_GROUNDER_V1",
        "candidate_grounding_report_sha256": candidate["report_sha256"],
        "verification_report_sha256": verification["report_sha256"],
        "score": "MIN_QUESTION_BLIND_EVENT_AND_BINARY_VERIFIER_CONFIDENCE",
        "alternative_candidate_recovery": False,
    })
    outputs = []
    for raw in candidate["rows"]:
        task_id = str(raw["task_id"])
        old = query_grounding_v2_from_dict(raw["receipt"])
        decision = by_task[task_id]
        new_candidates = []
        new_events = []
        if old.candidates and decision["status"] == "SUPPORTED":
            old_candidate = old.candidates[0]
            if decision.get("candidate_track_id") != old_candidate.track_id:
                raise ValueError("binary verifier changed the candidate track")
            matching = [
                event for event in old.events
                if event.role_map.get(old_candidate.requested_role) == old_candidate.track_id
            ]
            evidence = tuple(sorted(set(int(x) for x in decision["evidence_frame_ids"])))
            if not matching or not evidence:
                raise ValueError("supported verification lacks a matching typed event")
            confidence = min(old_candidate.confidence, float(decision["confidence"]))
            template = max(matching, key=lambda event: (event.confidence, event.event_id))
            new_events.append(replace(
                template, event_id="R0", start_frame=min(evidence), end_frame=max(evidence),
                evidence_frames=evidence, confidence=confidence,
            ))
            new_candidates.append(QueryCandidateEvidence(
                track_id=old_candidate.track_id,
                requested_role=old_candidate.requested_role,
                status="SUPPORTED", confidence=confidence,
                evidence_frames=evidence,
            ))
        receipt = QueryGroundingV2Receipt.create(
            task_id=old.task_id, video_sha256=old.video_sha256,
            semantic_slots_sha256=old.semantic_slots_sha256,
            selected_frame_indices=old.selected_frame_indices,
            selected_frame_sha256s=old.selected_frame_sha256s,
            tracks=old.tracks, events=new_events, candidates=new_candidates,
            public_ontology_sha256=old.public_ontology_sha256,
            grounder_backend_sha256=backend_sha,
            provider_calls=old.provider_calls + int(decision["status"] != "ABSTAIN_NO_EVENT_CANDIDATE"),
        )
        row = dict(raw)
        row.update({
            "receipt": asdict(receipt),
            "candidate_confidence": (
                new_candidates[0].confidence if new_candidates else 0.0
            ),
            "candidate_support_threshold": 0.0,
            "provider_error": decision.get("provider_error"),
            "binary_verification": decision,
        })
        outputs.append(row)
    report = {
        **{key: value for key, value in candidate.items() if key not in {
            "rows", "report_sha256", "grounder_backend_sha256",
            "reported_receipt_provider_cost_usd", "provider_and_contract_success_fraction",
        }},
        "schema_version": "agqa-binary-verified-query-grounder-v1",
        "status": expected_candidate_status,
        "consumed_development_pilot": bool(args.consumed_development_pilot),
        "rows": outputs, "grounder_backend_sha256": backend_sha,
        "candidate_grounding_report_sha256": candidate["report_sha256"],
        "binary_verification_report_sha256": verification["report_sha256"],
        "reported_receipt_provider_cost_usd": (
            float(candidate.get("reported_receipt_provider_cost_usd", 0.0))
            + float(verification["reported_cost_usd"])
        ),
        "provider_and_contract_success_fraction": min(
            float(candidate.get("provider_and_contract_success_fraction", 1.0)),
            float(verification["provider_and_contract_success_fraction"]),
        ),
        "answer_blind_query_candidate_verification": True,
        "candidate_verification_status": "COMPLETED_INDEPENDENT_SINGLE_CANDIDATE_PIXEL_CHECK",
        "alternative_candidate_selection_allowed": False,
        "answer_read": False, "official_scene_graph_read": False,
        "functional_program_read": False, "source_controller_read": False,
        "target_outcome_read": False,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "tasks": len(outputs),
        "supported": sum(bool(row["receipt"]["candidates"]) for row in outputs),
        "reported_cost_usd": report["reported_receipt_provider_cost_usd"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
