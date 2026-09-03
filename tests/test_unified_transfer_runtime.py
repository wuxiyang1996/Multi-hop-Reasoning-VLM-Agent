from __future__ import annotations

from dataclasses import replace

import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.unified_transfer_runtime import (
    PairedCalibration,
    SelectiveTargetExecutor,
    TargetGroundingReceipt,
    TransferVerdict,
    UnifiedNeurosymbolicTransferRuntime,
    UnifiedRoute,
    UnifiedRuntimeError,
    validate_authorization,
)


SOURCE = stable_hash("source")
GROUNDER = stable_hash("grounder")
EXECUTOR = stable_hash("executor")
REPORT = stable_hash("report")


def route(
    *, utility: PairedCalibration = PairedCalibration(16, 0, 16),
    authenticity: PairedCalibration = PairedCalibration(16, 0, 16),
    induced: bool = True,
) -> UnifiedRoute:
    return UnifiedRoute(
        route_id="sokoban-to-webshop-v21",
        target_domain="webshop",
        target_interface="option_relation_search",
        required_capabilities=("candidate_relations", "native_search"),
        source_program_sha256=SOURCE,
        source_program_induced_from_interventions=induced,
        source_program_qualified=True,
        target_grounder_sha256=GROUNDER,
        target_executor_sha256=EXECUTOR,
        target_grounder_id="webshop.target_native_grounder",
        target_executor_id="webshop.target_native_executor",
        evidence_report_sha256=REPORT,
        utility_vs_neural=utility,
        authenticity_vs_source_permuted=authenticity,
    )


def grounding(**overrides) -> TargetGroundingReceipt:
    arguments = {
        "task_id": "webshop.future.1",
        "target_domain": "webshop",
        "target_interface": "option_relation_search",
        "target_state_sha256": stable_hash("state"),
        "target_grounder_sha256": GROUNDER,
        "capabilities": ("candidate_relations", "native_search"),
        "candidate_ids": ("a", "b"),
        "structural_predicates": {
            "candidate_set_unique": True,
            "terminal_binding_unique": True,
        },
        "grounder_qualified": True,
        "formal_outcome_read": False,
    }
    arguments.update(overrides)
    return TargetGroundingReceipt.create(**arguments)


def test_strong_direction_and_authenticity_authorize_but_never_emit_action():
    runtime = UnifiedNeurosymbolicTransferRuntime([route()])
    authorization = runtime.decide(grounding())
    validate_authorization(authorization)
    assert authorization.verdict == TransferVerdict.SELECT_SKILL
    assert authorization.utility_lower_bound > 0.5
    assert authorization.authenticity_lower_bound > 0.5
    assert not hasattr(authorization, "action")
    assert authorization.current_outcome_read is False


def test_alfworld_like_authenticity_tie_abstains_even_with_safe_gain():
    runtime = UnifiedNeurosymbolicTransferRuntime([
        route(
            utility=PairedCalibration(2, 0, 70),
            authenticity=PairedCalibration(1, 1, 70),
        )
    ])
    authorization = runtime.decide(grounding())
    assert authorization.verdict == TransferVerdict.ABSTAIN
    # The small 2W/0L target delta is itself not directionally calibrated.
    assert authorization.reason == "DIRECTIONAL_UTILITY_NOT_CALIBRATED"


def test_authenticity_is_a_separate_gate_after_directional_utility():
    runtime = UnifiedNeurosymbolicTransferRuntime([
        route(authenticity=PairedCalibration(1, 1, 30))
    ])
    authorization = runtime.decide(grounding())
    assert authorization.verdict == TransferVerdict.ABSTAIN
    assert authorization.reason == "SOURCE_SPECIFIC_AUTHENTICITY_NOT_CALIBRATED"


def test_unknown_interface_bad_state_grounding_and_outcome_exposure_fail_closed():
    runtime = UnifiedNeurosymbolicTransferRuntime([route()])
    unknown = runtime.decide(grounding(target_interface="natural_video_qa"))
    assert unknown.reason == "NO_EXACT_TARGET_INTERFACE_ROUTE"

    structurally_invalid = runtime.decide(grounding(
        structural_predicates={"terminal_binding_unique": False},
    ))
    assert structurally_invalid.reason == (
        "CURRENT_STATE_STRUCTURAL_APPLICABILITY_FAILED"
    )

    exposed = runtime.decide(grounding(formal_outcome_read=True))
    assert exposed.reason == "CURRENT_TASK_OUTCOME_EXPOSURE"


def test_non_induced_source_program_and_grounder_drift_abstain():
    noninduced = UnifiedNeurosymbolicTransferRuntime([route(induced=False)])
    assert noninduced.decide(grounding()).reason == (
        "SOURCE_PROGRAM_NOT_INTERVENTION_INDUCED"
    )
    runtime = UnifiedNeurosymbolicTransferRuntime([route()])
    drifted = grounding(target_grounder_sha256=stable_hash("changed"))
    assert runtime.decide(drifted).reason == "TARGET_GROUNDER_HASH_MISMATCH"


class _Executor:
    artifact_sha256 = EXECUTOR

    def __init__(self, action: str):
        self.action = action

    def execute(self, authorization, target_grounding, native_actions):
        assert authorization.source_program_sha256 == SOURCE
        assert target_grounding.formal_outcome_read is False
        return self.action


def test_only_hash_matched_target_executor_may_emit_a_native_action():
    target_grounding = grounding()
    authorization = UnifiedNeurosymbolicTransferRuntime([route()]).decide(
        target_grounding
    )
    assert SelectiveTargetExecutor.execute(
        authorization, target_grounding, ("search", "click"), _Executor("click")
    ) == "click"
    with pytest.raises(UnifiedRuntimeError, match="non-native"):
        SelectiveTargetExecutor.execute(
            authorization, target_grounding, ("search", "click"),
            _Executor("source_action"),
        )
    with pytest.raises(UnifiedRuntimeError, match="hash mismatch"):
        bad = _Executor("click")
        bad.artifact_sha256 = stable_hash("wrong")
        SelectiveTargetExecutor.execute(
            authorization, target_grounding, ("search", "click"), bad,
        )


def test_abstention_never_calls_target_executor():
    target_grounding = grounding(
        structural_predicates={"terminal_binding_unique": False},
    )
    authorization = UnifiedNeurosymbolicTransferRuntime([route()]).decide(
        target_grounding
    )
    assert SelectiveTargetExecutor.execute(
        authorization, target_grounding, ("search", "click"), _Executor("click")
    ) is None


def test_receipt_and_authorization_hashes_fail_closed():
    target_grounding = grounding()
    with pytest.raises(UnifiedRuntimeError, match="receipt hash mismatch"):
        replace(target_grounding, task_id="tampered").validate()
    authorization = UnifiedNeurosymbolicTransferRuntime([route()]).decide(
        target_grounding
    )
    with pytest.raises(UnifiedRuntimeError, match="authorization hash mismatch"):
        validate_authorization(replace(authorization, reason="tampered"))
