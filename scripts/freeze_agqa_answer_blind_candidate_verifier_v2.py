#!/usr/bin/env python3
"""Freeze an independent single-candidate pixel verifier before outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--runtime-amendment", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--candidate-grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--consumed-development-pilot", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("candidate-verifier protocol is immutable")

    parent = json.loads(args.parent_protocol.read_text())
    amendment = json.loads(args.runtime_amendment.read_text())
    cohort = json.loads(args.cohort.read_text())
    candidate = json.loads(args.candidate_grounding.read_text())
    expected_status = (
        "CONSUMED_DEVELOPMENT_QUERY_GROUNDING_V2_NOT_TRANSFER_EVIDENCE"
        if args.consumed_development_pilot
        else "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME"
    )
    if candidate.get("status") != expected_status:
        raise ValueError("candidate grounding phase differs from verifier freeze")
    if candidate.get("answer_blind_query_candidate_verification"):
        raise ValueError("candidate grounding already claims independent verification")
    if candidate.get("candidate_verification_status") != (
        "PENDING_INDEPENDENT_SINGLE_CANDIDATE_PIXEL_CHECK"
    ):
        raise ValueError("candidate grounding does not request the frozen verifier")
    if candidate.get("cohort_sha256") != cohort.get("cohort_sha256"):
        raise ValueError("candidate grounding and cohort differ")
    forbidden = (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )
    if any(candidate.get(key) for key in forbidden):
        raise ValueError("candidate grounding crossed its authority boundary")
    if amendment.get("parent_acquisition_protocol_file_sha256") != _sha256(
        args.parent_protocol
    ):
        raise ValueError("runtime amendment and parent protocol differ")

    repo = Path(__file__).resolve().parents[1]
    body = {
        "schema_version": "agqa-answer-blind-single-candidate-verifier-protocol-v2",
        "status": "FROZEN_AFTER_PRELIMINARY_CANDIDATES_BEFORE_DEVELOPMENT_OR_TARGET_OUTCOMES",
        "claim_boundary": (
            "Consumed development qualification only. The verifier may reject or "
            "abstain on one frozen candidate but cannot name/select an alternative."
        ),
        "consumed_development_pilot": bool(args.consumed_development_pilot),
        "parent_acquisition_protocol_file_sha256": _sha256(args.parent_protocol),
        "parent_runtime_amendment_file_sha256": _sha256(args.runtime_amendment),
        "immutable_inputs": {
            "cohort_sha256": cohort["cohort_sha256"],
            "cohort_file_sha256": _sha256(args.cohort),
            "sgdet_file_sha256": _sha256(args.sgdet),
            "candidate_grounding_file_sha256": _sha256(args.candidate_grounding),
            "candidate_grounding_report_sha256": candidate["report_sha256"],
            "verifier_sha256": _sha256(
                repo / "scripts/verify_agqa_question_blind_event_candidates_v1.py"
            ),
            "merger_sha256": _sha256(
                repo / "scripts/merge_agqa_answer_blind_binary_event_verifier_v1.py"
            ),
            "verified_compiler_sha256": _sha256(
                repo / "scripts/compile_agqa_binary_verified_query_grounder_v1.py"
            ),
        },
        "binary_candidate_verifier": {
            "model": "qwen/qwen3-vl-235b-a22b-instruct",
            "provider": "parasail",
            "provider_allow_fallbacks": False,
            "seed": 0,
            "temperature": 0,
            "thinking": False,
            "maximum_frames": 12,
            "max_tokens": 256,
            "maximum_contract_attempts": 2,
            "candidate_count_per_task": 1,
            "alternative_candidate_selection_allowed": False,
            "supported_evidence_requires_same_frame_person_and_candidate_tracks": True,
        },
        "authority": {
            "raw_frames_read": True,
            "frozen_candidate_identity_read": True,
            "task_question_text_read": False,
            "alternative_candidates_read": False,
            "answer_read": False,
            "official_stsg_read": False,
            "functional_program_read": False,
            "source_controller_read": False,
            "target_outcome_read": False,
        },
        "development_outcomes_opened_before_freeze": False,
        "target_outcomes_opened_before_freeze": False,
    }
    body["protocol_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"],
        "candidate_grounding_report_sha256": candidate["report_sha256"],
        "protocol_sha256": body["protocol_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
