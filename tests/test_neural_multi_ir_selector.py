from __future__ import annotations

import json

from motif_transfer.contracts import stable_hash
from motif_transfer.multi_ir_selector_training import execute_anonymous_selection
from motif_transfer.neural_multi_ir_selector import (
    EXACT_SHADOW,
    STRUCTURAL_ONLY,
    NeuralMultiIRSelector,
)
from motif_transfer.structural_ir_applicability import (
    OperatorSignature,
    SourceIRContract,
    TargetIRRequirement,
)
from motif_transfer.unified_neurosymbolic_harness import (
    InducedProgramEnvelope,
    UnifiedNeurosymbolicHarness,
    UnifiedTargetGrounding,
)
from motif_transfer.unified_transfer_runtime import (
    PairedCalibration,
    TargetGroundingReceipt,
    TransferVerdict,
    UnifiedNeurosymbolicTransferRuntime,
    UnifiedRoute,
)


PROGRAM = stable_hash("neural selector program")
OTHER = stable_hash("neural selector other")
GROUNDER = stable_hash("neural selector grounder")
EXECUTOR = stable_hash("neural selector executor")


def _signature(family: str) -> OperatorSignature:
    return OperatorSignature("UPDATE", family, 1, "COUNT")


def _contract(program: str, kind: str, family: str) -> SourceIRContract:
    return SourceIRContract.create(
        program_sha256=program,
        ir_kind=kind,
        operator_sequence=(_signature(family),),
        recurrent=False,
        terminal_predicate_families=(),
        source_intervention_qualified=True,
        source_confirmation_sha256=stable_hash(f"confirmation:{program}"),
    )


def _requirement() -> TargetIRRequirement:
    return TargetIRRequirement.create(
        task_id="task", target_domain="domain", target_interface="native.v1",
        target_grounder_sha256=GROUNDER, ir_kind="MATCHING_IR",
        operator_sequence=(_signature("MATCH"),), recurrent=False,
        terminal_predicate_families=(), grounder_qualified=True,
        formal_outcome_read=False,
    )


class _ExactGenerator:
    artifact_sha256 = stable_hash("exact neural selector")

    def generate(self, prompt: str) -> str:
        payload = json.loads(prompt.split(
            "SELECTOR_INPUT=", 1,
        )[1].split("\nOUTPUT_JSON=", 1)[0])
        return json.dumps(execute_anonymous_selection(
            program_catalog=payload["program_catalog"],
            target_requirement=payload[
                "target_native_structural_requirement"
            ],
        ), sort_keys=True, separators=(",", ":"))


class _WrongButWellFormedGenerator:
    artifact_sha256 = stable_hash("wrong neural selector")

    def generate(self, prompt: str) -> str:
        payload = json.loads(prompt.split(
            "SELECTOR_INPUT=", 1,
        )[1].split("\nOUTPUT_JSON=", 1)[0])
        selected = next(
            row["catalog_id"] for row in payload["program_catalog"]
            if row["ir_kind"] == "OTHER_IR"
        )
        return json.dumps({
            "decision": "SELECT_SKILL",
            "selected_catalog_id": selected,
            "reason": "MODEL_PROPOSAL",
        }, sort_keys=True, separators=(",", ":"))


def _envelope(contract: SourceIRContract) -> InducedProgramEnvelope:
    return InducedProgramEnvelope.create(
        contract=contract,
        source_transition_receipts_sha256=stable_hash(
            f"tuples:{contract.program_sha256}"
        ),
        inducer_artifact_sha256=stable_hash("inducer"),
    )


def _target(requirement: TargetIRRequirement) -> UnifiedTargetGrounding:
    applicability = TargetGroundingReceipt.create(
        task_id=requirement.task_id, target_domain=requirement.target_domain,
        target_interface=requirement.target_interface,
        target_state_sha256=stable_hash("target state"),
        target_grounder_sha256=requirement.target_grounder_sha256,
        capabilities=("typed_candidates",), candidate_ids=("a", "b"),
        structural_predicates={"qualified": True}, grounder_qualified=True,
        formal_outcome_read=False,
    )
    return UnifiedTargetGrounding.create(
        requirement=requirement, applicability=applicability,
    )


def _harness(contracts: tuple[SourceIRContract, ...]) -> UnifiedNeurosymbolicHarness:
    route = UnifiedRoute(
        route_id="route", target_domain="domain", target_interface="native.v1",
        required_capabilities=("typed_candidates",),
        source_program_sha256=PROGRAM,
        source_program_induced_from_interventions=True,
        source_program_qualified=True, target_grounder_sha256=GROUNDER,
        target_executor_sha256=EXECUTOR, target_grounder_id="grounder",
        target_executor_id="executor", evidence_report_sha256=stable_hash("evidence"),
        utility_vs_neural=PairedCalibration(8, 0, 8),
        authenticity_vs_source_permuted=PairedCalibration(8, 0, 8),
    )
    return UnifiedNeurosymbolicHarness(
        tuple(_envelope(row) for row in contracts),
        UnifiedNeurosymbolicTransferRuntime((route,)),
    )


def test_exact_shadow_never_authorizes_even_when_exact() -> None:
    contract = _contract(PROGRAM, "MATCHING_IR", "MATCH")
    receipt = NeuralMultiIRSelector(
        _ExactGenerator(), verification_mode=EXACT_SHADOW,
    ).decide(contracts=(contract,), requirement=_requirement())
    assert receipt.exact_symbolic_match is True
    assert receipt.source_program_authorized is False
    assert receipt.selected_program_sha256 is None


def test_structural_live_selector_authorizes_correct_program() -> None:
    contract = _contract(PROGRAM, "MATCHING_IR", "MATCH")
    selector = NeuralMultiIRSelector(
        _ExactGenerator(), verification_mode=STRUCTURAL_ONLY,
    )
    target = _target(_requirement())
    authorization = _harness((contract,)).decide_neural(target, selector)
    assert authorization.verdict == TransferVerdict.SELECT_SKILL
    assert authorization.selected_program_sha256 == PROGRAM
    assert authorization.reason == "NEURAL_STRUCTURE_AND_CALIBRATED_UTILITY_AGREE"


def test_live_mode_does_not_python_repair_legal_wrong_selection() -> None:
    matching = _contract(PROGRAM, "MATCHING_IR", "MATCH")
    other = _contract(OTHER, "OTHER_IR", "OTHER")
    selector = NeuralMultiIRSelector(
        _WrongButWellFormedGenerator(), verification_mode=STRUCTURAL_ONLY,
    )
    target = _target(_requirement())
    direct = selector.decide(
        contracts=(matching, other), requirement=target.requirement,
    )
    assert direct.structural_contract_valid is True
    assert direct.source_program_authorized is True
    assert direct.selected_program_sha256 == OTHER

    authorization = _harness((matching, other)).decide_neural(target, selector)
    assert authorization.verdict == TransferVerdict.ABSTAIN
    assert authorization.selected_program_sha256 == OTHER
    assert authorization.reason == "NEURAL_ROUTE_SOURCE_CONTRACT_MISMATCH"


def test_invalid_json_fails_closed() -> None:
    class Invalid:
        artifact_sha256 = stable_hash("invalid selector")

        @staticmethod
        def generate(prompt: str) -> str:
            return "not-json"

    contract = _contract(PROGRAM, "MATCHING_IR", "MATCH")
    receipt = NeuralMultiIRSelector(
        Invalid(), verification_mode=STRUCTURAL_ONLY,
    ).decide(contracts=(contract,), requirement=_requirement())
    assert receipt.source_program_authorized is False
    assert receipt.status == "NEURAL_MULTI_IR_OUTPUT_REJECTED"
