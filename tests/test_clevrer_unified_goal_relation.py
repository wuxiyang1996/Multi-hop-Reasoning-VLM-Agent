from __future__ import annotations

import json
from pathlib import Path

from motif_transfer.clevrer_unified_goal_relation import (
    build_harness,
    build_route,
    decide_recovery,
    source_goal_relation_envelope,
    target_grounding,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.unified_transfer_runtime import (
    PairedCalibration,
    TransferVerdict,
)
from motif_transfer.video_transfer_measurement import (
    GroundingMode,
    GroundingToolBudget,
    SharedVideoGroundingReceipt,
    assert_unified_target_uses_shared_grounding,
)


REPO = Path(__file__).resolve().parents[1]


def _read(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


def _setup():
    artifact = _read("runs/sokoban_goal_relation_macro_v3/artifact.json")
    confirmation = _read(
        "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
    )
    envelope = source_goal_relation_envelope(
        artifact, confirmation,
        inducer_artifact_sha256=stable_hash("source-inducer"),
    )
    grounder = stable_hash("clevrer-grounder")
    executor = stable_hash("clevrer-executor")
    route = build_route(
        source_program_sha256=envelope.contract.program_sha256,
        target_grounder_sha256=grounder,
        target_executor_sha256=executor,
        evidence_report_sha256=stable_hash("development-report"),
        utility_vs_neural=PairedCalibration(27, 5, 688),
        authenticity_vs_source_permuted=PairedCalibration(38, 11, 671),
    )
    return envelope, build_harness(envelope, route), grounder, executor


def test_positive_proof_delta_authorizes_only_native_trajectory_switch():
    envelope, harness, grounder, executor = _setup()
    target = target_grounding(
        task_id="video_1.mp4.Q1", contract=envelope.contract,
        target_grounder_sha256=grounder,
        proof_receipt_sha256=stable_hash("proof"),
        proof_predicted_uplift=0.3, decision_threshold=0.2,
    )
    decision = decide_recovery(
        harness=harness, target=target,
        target_executor_sha256=executor,
    )
    assert decision.phase7.verdict == TransferVerdict.SELECT_SKILL
    assert decision.selected_native_representation == "trajectory"
    assert decision.executor_calls == 1
    assert decision.phase7.target_action_emitted is False
    assert decision.phase7.current_target_outcome_read is False


def test_nonpositive_delta_abstains_to_explicit_baseline_without_executor():
    envelope, harness, grounder, executor = _setup()
    target = target_grounding(
        task_id="video_2.mp4.Q1", contract=envelope.contract,
        target_grounder_sha256=grounder,
        proof_receipt_sha256=stable_hash("proof-2"),
        proof_predicted_uplift=0.2, decision_threshold=0.2,
    )
    decision = decide_recovery(
        harness=harness, target=target,
        target_executor_sha256=executor,
    )
    assert decision.phase7.verdict == TransferVerdict.ABSTAIN
    assert decision.phase7.reason == (
        "UTILITY_ROUTER:CURRENT_STATE_STRUCTURAL_APPLICABILITY_FAILED"
    )
    assert decision.selected_native_representation == "explicit_relation"
    assert decision.executor_calls == 0


def test_target_receipt_is_outcome_blind_and_content_bound():
    envelope, _, grounder, _ = _setup()
    target = target_grounding(
        task_id="video_3.mp4.Q1", contract=envelope.contract,
        target_grounder_sha256=grounder,
        proof_receipt_sha256=stable_hash("proof-3"),
        proof_predicted_uplift=0.9, decision_threshold=0.2,
    )
    assert target.requirement.formal_outcome_read is False
    assert target.applicability.formal_outcome_read is False
    assert target.applicability.target_state_sha256 == stable_hash({
        "task_id": "video_3.mp4.Q1",
        "proof_receipt_sha256": stable_hash("proof-3"),
        "proof_predicted_uplift": 0.9,
        "decision_threshold": 0.2,
        "gold_or_formal_outcome": "NOT_READ",
    })


def test_clevrer_adapter_accepts_shared_answer_blind_video_state():
    envelope, _, grounder, _ = _setup()
    shared = SharedVideoGroundingReceipt.create(
        benchmark="clevrer", task_id="video_4.mp4.Q1", split="qualification",
        mode=GroundingMode.ORACLE_EVENT_GRAPH,
        state={"objects": [{"id": 0}], "events": [{"kind": "collision"}]},
        evidence_source_sha256=stable_hash("official-clevrer-scene-state"),
        tool_budget=GroundingToolBudget(0, 0, 0),
        official_scene_graph_read=True,
    )
    target = target_grounding(
        task_id=shared.task_id, contract=envelope.contract,
        target_grounder_sha256=grounder,
        proof_receipt_sha256=stable_hash("proof-4"),
        proof_predicted_uplift=0.9, decision_threshold=0.2,
        shared_target_state_sha256=shared.target_state_sha256,
    )
    assert_unified_target_uses_shared_grounding(shared, target)
