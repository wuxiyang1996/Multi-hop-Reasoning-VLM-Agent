#!/usr/bin/env python3
"""Freeze an answer-blind binary event-verifier protocol before calls/outcomes."""

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
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sgdet", type=Path, required=True)
    parser.add_argument("--candidate-grounding", type=Path, required=True)
    parser.add_argument("--candidate-protocol", type=Path, required=True)
    parser.add_argument("--verifier-script", type=Path, required=True)
    parser.add_argument("--compiler-script", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("binary event verifier protocol is immutable")
    cohort = json.loads(args.cohort.read_text())
    candidate = json.loads(args.candidate_grounding.read_text())
    candidate_protocol = json.loads(args.candidate_protocol.read_text())
    if candidate.get("status") != "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME":
        raise ValueError("candidate grounding is not frozen before outcome")
    if candidate.get("cohort_sha256") != cohort.get("cohort_sha256"):
        raise ValueError("candidate grounding and cohort differ")
    if any(candidate.get(key) for key in (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )):
        raise ValueError("candidate grounding crossed its authority boundary")
    body = {
        "schema_version": "agqa-answer-blind-binary-event-verifier-development-protocol-v1",
        "status": "FROZEN_ON_CONSUMED_DEVELOPMENT_BEFORE_BINARY_CALLS",
        "claim_boundary": (
            "Consumed AGQA train grounder development only; Stage-C aggregate "
            "diagnostics were already opened, and this artifact is never transfer evidence."
        ),
        "selection_context": (
            "The frozen question-blind inventory passed recall/role/dedup/contract "
            "gates but its provider self-confidence failed the global precision gate."
        ),
        "scientific_change": {
            "reason": "Question-blind inventory provides recall, while saturated provider self-confidence cannot by itself certify a unique query binding.",
            "method": "Verify exactly one already-frozen candidate as SUPPORTED, REFUTED, or UNKNOWN from pixels.",
            "alternative_candidate_selection": False,
            "question_text_supplied": False,
            "answer_or_outcome_supplied": False
        },
        "immutable_inputs": {
            "cohort_sha256": cohort["cohort_sha256"],
            "sgdet_file_sha256": _sha256(args.sgdet),
            "candidate_grounding_file_sha256": _sha256(args.candidate_grounding),
            "candidate_grounding_report_sha256": candidate["report_sha256"],
            "candidate_protocol_file_sha256": _sha256(args.candidate_protocol),
            "verifier_script_sha256": _sha256(args.verifier_script),
            "compiler_script_sha256": _sha256(args.compiler_script)
        },
        "binary_candidate_verifier": {
            "model": args.model,
            "maximum_frames": 12,
            "max_tokens": 160,
            "maximum_contract_attempts": 2,
            "temperature_parameter": "omitted for provider-constrained Qwen instruct decoding",
            "selection": "one top event-inventory candidate fixed before verifier call",
            "outputs": ["SUPPORTED", "REFUTED", "UNKNOWN"],
            "supported_requires_presented_pixel_evidence": True,
            "absence_means_unknown_not_refuted": True,
            "score": "minimum of inventory-event and verifier confidence"
        },
        "candidate_generation_threshold": 0.0,
        "threshold_selection": candidate_protocol["threshold_selection"],
        "grounding_gates": candidate_protocol["grounding_gates"],
        "authority": {
            "question_text_read": False,
            "answer_read": False,
            "official_stsg_read": False,
            "functional_program_read": False,
            "source_controller_read": False,
            "target_outcome_read": False
        }
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
