from dataclasses import replace

import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.structural_ir_applicability import (
    OperatorSignature,
    SourceIRContract,
    TargetIRRequirement,
)
from motif_transfer.unified_neurosymbolic_harness import (
    InducedProgramEnvelope,
    UnifiedNeurosymbolicHarness,
    UnifiedTargetGrounding,
    validate_phase7_authorization,
)
from motif_transfer.unified_transfer_runtime import (
    PairedCalibration,
    TargetGroundingReceipt,
    TransferVerdict,
    UnifiedNeurosymbolicTransferRuntime,
    UnifiedRoute,
    UnifiedRuntimeError,
)


PROGRAM = stable_hash("program")
GROUNDER = stable_hash("grounder")
EXECUTOR = stable_hash("executor")


def _signature():
    return OperatorSignature("ADD", "ENTITY_SLOT", 1, "ENTITY_REFERENCE")


def _contract(*, program=PROGRAM, kind="FINITE"):
    return SourceIRContract.create(
        program_sha256=program,
        ir_kind=kind,
        operator_sequence=(_signature(),),
        recurrent=False,
        terminal_predicate_families=(),
        source_intervention_qualified=True,
        source_confirmation_sha256=stable_hash("confirmation"),
    )


def _envelope(*, program=PROGRAM, kind="FINITE", target_read=False):
    return InducedProgramEnvelope.create(
        contract=_contract(program=program, kind=kind),
        source_transition_receipts_sha256=stable_hash("source tuples"),
        inducer_artifact_sha256=stable_hash("inducer"),
        target_data_read=target_read,
    )


def _route(*, program=PROGRAM, utility=PairedCalibration(16, 0, 16)):
    return UnifiedRoute(
        route_id="source-to-target",
        target_domain="target",
        target_interface="native.v1",
        required_capabilities=("typed_candidates",),
        source_program_sha256=program,
        source_program_induced_from_interventions=True,
        source_program_qualified=True,
        target_grounder_sha256=GROUNDER,
        target_executor_sha256=EXECUTOR,
        target_grounder_id="target.grounder",
        target_executor_id="target.executor",
        evidence_report_sha256=stable_hash("report"),
        utility_vs_neural=utility,
        authenticity_vs_source_permuted=PairedCalibration(16, 0, 16),
    )


def _target(*, kind="FINITE"):
    requirement = TargetIRRequirement.create(
        task_id="target.task.1",
        target_domain="target",
        target_interface="native.v1",
        target_grounder_sha256=GROUNDER,
        ir_kind=kind,
        operator_sequence=(_signature(),),
        recurrent=False,
        terminal_predicate_families=(),
        grounder_qualified=True,
        formal_outcome_read=False,
    )
    applicability = TargetGroundingReceipt.create(
        task_id="target.task.1",
        target_domain="target",
        target_interface="native.v1",
        target_state_sha256=stable_hash("state"),
        target_grounder_sha256=GROUNDER,
        capabilities=("typed_candidates",),
        candidate_ids=("native-a", "native-b"),
        structural_predicates={"unique_binding": True},
        grounder_qualified=True,
        formal_outcome_read=False,
    )
    return UnifiedTargetGrounding.create(
        requirement=requirement, applicability=applicability,
    )


def _harness(*, envelope=None, route=None):
    return UnifiedNeurosymbolicHarness(
        (envelope or _envelope(),),
        UnifiedNeurosymbolicTransferRuntime((route or _route(),)),
    )


def test_phase7_authorizes_only_when_structure_route_and_utility_agree():
    authorization = _harness().decide(_target())
    validate_phase7_authorization(authorization)
    assert authorization.verdict == TransferVerdict.SELECT_SKILL
    assert authorization.selected_program_sha256 == PROGRAM
    assert authorization.reason == "STRUCTURE_AND_CALIBRATED_UTILITY_AGREE"
    assert authorization.current_target_outcome_read is False
    assert authorization.target_action_emitted is False


def test_structural_mismatch_abstains_before_utility():
    authorization = _harness().decide(_target(kind="OTHER"))
    assert authorization.verdict == TransferVerdict.ABSTAIN
    assert authorization.reason.startswith("STRUCTURAL_SELECTION:")
    assert authorization.utility_authorization_sha256 is None


def test_source_target_data_leak_disqualifies_envelope():
    authorization = _harness(envelope=_envelope(target_read=True)).decide(
        _target()
    )
    assert authorization.verdict == TransferVerdict.ABSTAIN
    assert authorization.reason == "STRUCTURAL_SELECTION:NO_SOURCE_CONTRACT_MATCHES"


def test_small_directional_gain_is_preserved_as_utility_abstention():
    authorization = _harness(route=_route(
        utility=PairedCalibration(2, 0, 70),
    )).decide(_target())
    assert authorization.verdict == TransferVerdict.ABSTAIN
    assert authorization.reason == (
        "UTILITY_ROUTER:DIRECTIONAL_UTILITY_NOT_CALIBRATED"
    )


def test_route_and_structural_source_disagreement_fails_closed():
    other = stable_hash("other program")
    authorization = _harness(route=_route(program=other)).decide(_target())
    assert authorization.verdict == TransferVerdict.ABSTAIN
    assert authorization.reason == "ROUTE_SOURCE_CONTRACT_MISMATCH"


class _Executor:
    artifact_sha256 = EXECUTOR

    def __init__(self):
        self.calls = 0

    def execute(self, authorization, grounding, native_actions):
        self.calls += 1
        return native_actions[-1]


def test_only_target_executor_emits_native_action_after_phase7_authorization():
    target = _target()
    harness = _harness()
    phase7 = harness.decide(target)
    utility = harness.runtime.decide(target.applicability)
    executor = _Executor()
    assert harness.execute(
        phase7, utility, target, ("native-a", "native-b"), executor,
    ) == "native-b"
    assert executor.calls == 1

    abstaining = _harness(route=_route(
        utility=PairedCalibration(0, 0, 20),
    ))
    phase7 = abstaining.decide(target)
    utility = abstaining.runtime.decide(target.applicability)
    assert abstaining.execute(
        phase7, utility, target, ("native-a", "native-b"), executor,
    ) is None
    assert executor.calls == 1


def test_grounder_dual_receipts_and_envelope_hashes_fail_closed():
    target = _target()
    bad_requirement = replace(target.requirement, task_id="tampered")
    with pytest.raises(ValueError, match="requirement hash mismatch"):
        UnifiedTargetGrounding.create(
            requirement=bad_requirement, applicability=target.applicability,
        )
    envelope = _envelope()
    with pytest.raises(UnifiedRuntimeError, match="envelope hash mismatch"):
        _harness(envelope=replace(envelope, target_data_read=True))


def test_phase7_authorization_hash_fails_closed():
    authorization = _harness().decide(_target())
    with pytest.raises(UnifiedRuntimeError, match="authorization hash mismatch"):
        validate_phase7_authorization(replace(authorization, reason="tampered"))
