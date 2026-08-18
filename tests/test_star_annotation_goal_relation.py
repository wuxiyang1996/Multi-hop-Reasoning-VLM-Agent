from __future__ import annotations

import json
from pathlib import Path

from motif_transfer.contracts import stable_hash
from motif_transfer.natural_video_recovery import PROOF_KINDS
from motif_transfer.star_annotation_goal_relation import (
    build_harness,
    build_route,
    decide_recovery,
    relation_coverage_receipt,
    source_goal_relation_envelope,
    target_grounding,
)
from motif_transfer.unified_transfer_runtime import (
    PairedCalibration,
    TransferVerdict,
)


REPO = Path(__file__).resolve().parents[1]
QUESTION_PROGRAM = (
    {"function": "Situations", "value_input": []},
    {"function": "Actions", "value_input": []},
    {"function": "Query_Objs", "value_input": []},
)


def _proof():
    statuses = {
        "A": "REFUTED", "B": "SUPPORTED", "C": "UNKNOWN", "D": "UNKNOWN",
    }
    return {
        "answer": "B",
        "candidates": [
            {
                "slot": slot,
                "proof_steps": [
                    {
                        "kind": kind,
                        "status": statuses[slot],
                        "confidence": 0.9,
                    }
                    for kind in PROOF_KINDS
                ],
            }
            for slot in "ABCD"
        ],
    }


def _setup(calibration: PairedCalibration):
    artifact = json.loads((
        REPO / "runs/sokoban_goal_relation_macro_v3/artifact.json"
    ).read_text(encoding="utf-8"))
    confirmation = json.loads((
        REPO / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
    ).read_text(encoding="utf-8"))
    envelope = source_goal_relation_envelope(
        artifact, confirmation,
        inducer_artifact_sha256=stable_hash("source-inducer"),
    )
    grounder = stable_hash("star-grounder")
    executor = stable_hash("star-executor")
    route = build_route(
        source_program_sha256=envelope.contract.program_sha256,
        target_grounder_sha256=grounder,
        target_executor_sha256=executor,
        evidence_report_sha256=stable_hash("prior-star-development"),
        utility_vs_neural=calibration,
        authenticity_vs_source_permuted=calibration,
    )
    return envelope, build_harness(envelope, route), grounder, executor


def _target(envelope, grounder, *, rotation=0):
    coverage = relation_coverage_receipt(
        task_id="Interaction_T1_test",
        direct={"answer": "A"}, proof=_proof(),
        question_program=QUESTION_PROGRAM,
        binding_rotation=rotation,
    )
    target = target_grounding(
        contract=envelope.contract,
        target_grounder_sha256=grounder,
        coverage=coverage,
        proof_receipt_sha256=stable_hash(_proof()),
    )
    return coverage, target


def test_same_source_ir_authorizes_only_star_native_proof_policy():
    envelope, harness, grounder, executor = _setup(PairedCalibration(27, 5, 80))
    coverage, target = _target(envelope, grounder)
    decision = decide_recovery(
        harness=harness, target=target, target_executor_sha256=executor,
    )
    assert coverage.recurrent_update_count == 3
    assert coverage.terminal_relation_coverage is True
    assert coverage.gold_or_formal_outcome_read is False
    assert decision.phase7.verdict == TransferVerdict.SELECT_SKILL
    assert decision.selected_native_policy == "uniform_typed_proof"
    assert decision.executor_calls == 1


def test_weak_prior_star_calibration_fails_closed_before_executor():
    envelope, harness, grounder, executor = _setup(PairedCalibration(9, 7, 112))
    _, target = _target(envelope, grounder)
    decision = decide_recovery(
        harness=harness, target=target, target_executor_sha256=executor,
    )
    assert decision.phase7.verdict == TransferVerdict.ABSTAIN
    assert decision.phase7.reason == (
        "UTILITY_ROUTER:DIRECTIONAL_UTILITY_NOT_CALIBRATED"
    )
    assert decision.selected_native_policy == "uniform_direct"
    assert decision.executor_calls == 0


def test_rotated_target_binding_rejects_the_apparent_relation_update():
    envelope, harness, grounder, executor = _setup(PairedCalibration(27, 5, 80))
    coverage, target = _target(envelope, grounder, rotation=1)
    decision = decide_recovery(
        harness=harness, target=target, target_executor_sha256=executor,
    )
    assert coverage.terminal_relation_coverage is False
    assert decision.phase7.verdict == TransferVerdict.ABSTAIN
    assert decision.executor_calls == 0
    assert target.requirement.formal_outcome_read is False
    assert target.applicability.formal_outcome_read is False
