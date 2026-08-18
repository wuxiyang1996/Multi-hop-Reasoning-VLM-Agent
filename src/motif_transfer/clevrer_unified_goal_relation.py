"""Unified-harness adapter for source-induced CLEVRER proof recovery.

The source artifact is the template-free recurrent goal-relation program
induced from Sokoban ``(state, action, effect, next_state)`` tuples.  The
target side is deliberately narrow: a frozen CLEVRER-native proof grounder
binds that anonymous relation-update program to one of two native dynamics
representations.  The source selector can authorize a representation switch;
it never sees a video token, answer label, or target action.

This module contains no provider client and cannot make an external API call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .source_goal_relation_induction import (
    validate_goal_relation_macro_program,
)
from .structural_ir_applicability import (
    OperatorSignature,
    SourceIRContract,
    TargetIRRequirement,
)
from .unified_neurosymbolic_harness import (
    InducedProgramEnvelope,
    Phase7Authorization,
    UnifiedNeurosymbolicHarness,
    UnifiedTargetGrounding,
)
from .unified_transfer_runtime import (
    ExecutionAuthorization,
    PairedCalibration,
    TargetGroundingReceipt,
    TransferVerdict,
    UnifiedNeurosymbolicTransferRuntime,
    UnifiedRoute,
)


TARGET_DOMAIN = "clevrer"
TARGET_INTERFACE = "paired_event_graph_goal_relation_recovery_v15"
IR_KIND = "RECURRENT_GOAL_RELATION_PROGRAM"
CAPABILITIES = (
    "paired_neural_event_graphs",
    "typed_step_proof_receipts",
    "unique_native_representation_binding",
)
NATIVE_ACTIONS = ("explicit_relation", "trajectory")


def _validated_confirmation(
    artifact: Mapping[str, Any], confirmation: Mapping[str, Any],
) -> str:
    validate_goal_relation_macro_program(artifact)
    body = dict(confirmation)
    claimed = str(body.pop("report_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("source goal-relation confirmation hash mismatch")
    if confirmation.get("artifact_sha256") != artifact.get("artifact_sha256"):
        raise ValueError("source goal-relation artifact/confirmation mismatch")
    if confirmation.get("status") != "SOURCE_GOAL_RELATION_MACRO_FRESH_VALIDATED":
        raise ValueError("source goal-relation program lacks fresh validation")
    if confirmation.get("source_gate_passed") is not True:
        raise ValueError("source goal-relation gate did not pass")
    if not all((confirmation.get("gates") or {}).values()):
        raise ValueError("source goal-relation confirmation gate failed")
    return claimed


def source_goal_relation_contract(
    artifact: Mapping[str, Any], confirmation: Mapping[str, Any],
) -> SourceIRContract:
    """Extract an anonymous structural contract without target information."""

    confirmation_sha256 = _validated_confirmation(artifact, confirmation)
    operators = tuple(
        OperatorSignature(
            operation=str(row["operation"]),
            predicate_family=str(row["predicate_family"]),
            arity=int(row["arity"]),
            value_kind=str(row["value_kind"]),
        )
        for row in artifact["operator_types"]
    )
    transition = artifact["program"]["transitions"][0]
    terminal = tuple(
        str(row["predicate_family"])
        for row in artifact["program"]["terminal_predicates"]
    )
    return SourceIRContract.create(
        program_sha256=str(artifact["artifact_sha256"]),
        ir_kind=IR_KIND,
        operator_sequence=operators,
        recurrent=transition["cardinality"] == "ONE_OR_MORE",
        terminal_predicate_families=terminal,
        source_intervention_qualified=True,
        source_confirmation_sha256=confirmation_sha256,
    )


def source_goal_relation_envelope(
    artifact: Mapping[str, Any], confirmation: Mapping[str, Any], *,
    inducer_artifact_sha256: str,
) -> InducedProgramEnvelope:
    contract = source_goal_relation_contract(artifact, confirmation)
    return InducedProgramEnvelope.create(
        contract=contract,
        source_transition_receipts_sha256=str(
            artifact["source_receipts_sha256"]
        ),
        inducer_artifact_sha256=inducer_artifact_sha256,
        learned_from_state_action_effect_next_state=True,
        target_data_read=False,
        named_policy_template_used=False,
    )


def build_route(
    *, source_program_sha256: str, target_grounder_sha256: str,
    target_executor_sha256: str, evidence_report_sha256: str,
    utility_vs_neural: PairedCalibration,
    authenticity_vs_source_permuted: PairedCalibration,
) -> UnifiedRoute:
    return UnifiedRoute(
        route_id="sokoban-goal-relation-to-clevrer-proof-v15",
        target_domain=TARGET_DOMAIN,
        target_interface=TARGET_INTERFACE,
        required_capabilities=CAPABILITIES,
        source_program_sha256=source_program_sha256,
        source_program_induced_from_interventions=True,
        source_program_qualified=True,
        target_grounder_sha256=target_grounder_sha256,
        target_executor_sha256=target_executor_sha256,
        target_grounder_id="clevrer.paired-proof-uplift.v14-frozen",
        target_executor_id="clevrer.native-representation-switch.v15",
        evidence_report_sha256=evidence_report_sha256,
        utility_vs_neural=utility_vs_neural,
        authenticity_vs_source_permuted=authenticity_vs_source_permuted,
    )


def build_harness(
    envelope: InducedProgramEnvelope, route: UnifiedRoute,
) -> UnifiedNeurosymbolicHarness:
    return UnifiedNeurosymbolicHarness(
        (envelope,), UnifiedNeurosymbolicTransferRuntime((route,)),
    )


def target_grounding(
    *, task_id: str, contract: SourceIRContract,
    target_grounder_sha256: str, proof_receipt_sha256: str,
    proof_predicted_uplift: float, decision_threshold: float,
) -> UnifiedTargetGrounding:
    """Bind one outcome-blind proof receipt to the induced relation program."""

    positive_delta = float(proof_predicted_uplift) > float(decision_threshold)
    requirement = TargetIRRequirement.create(
        task_id=task_id,
        target_domain=TARGET_DOMAIN,
        target_interface=TARGET_INTERFACE,
        target_grounder_sha256=target_grounder_sha256,
        ir_kind=contract.ir_kind,
        operator_sequence=contract.operator_sequence,
        recurrent=contract.recurrent,
        terminal_predicate_families=contract.terminal_predicate_families,
        grounder_qualified=True,
        formal_outcome_read=False,
    )
    state_sha256 = stable_hash({
        "task_id": task_id,
        "proof_receipt_sha256": proof_receipt_sha256,
        "proof_predicted_uplift": float(proof_predicted_uplift),
        "decision_threshold": float(decision_threshold),
        "gold_or_formal_outcome": "NOT_READ",
    })
    applicability = TargetGroundingReceipt.create(
        task_id=task_id,
        target_domain=TARGET_DOMAIN,
        target_interface=TARGET_INTERFACE,
        target_state_sha256=state_sha256,
        target_grounder_sha256=target_grounder_sha256,
        capabilities=CAPABILITIES,
        candidate_ids=NATIVE_ACTIONS,
        structural_predicates={
            "positive_relation_delta": positive_delta,
            "proof_receipt_bound": bool(proof_receipt_sha256),
            "terminal_predicate_observable": True,
            "unique_native_representation_binding": True,
        },
        grounder_qualified=True,
        formal_outcome_read=False,
    )
    return UnifiedTargetGrounding.create(
        requirement=requirement, applicability=applicability,
    )


class ClevrerRepresentationExecutor:
    """The only authority that emits a CLEVRER-native representation choice."""

    def __init__(self, artifact_sha256: str):
        self.artifact_sha256 = artifact_sha256
        self.calls = 0

    def execute(
        self, authorization: ExecutionAuthorization,
        grounding: TargetGroundingReceipt,
        native_actions: Sequence[str],
    ) -> str:
        self.calls += 1
        actions = tuple(native_actions)
        if actions != NATIVE_ACTIONS:
            raise ValueError("CLEVRER native representation contract drift")
        return "trajectory"


@dataclass(frozen=True)
class UnifiedRecoveryDecision:
    selected_native_representation: str
    phase7: Phase7Authorization
    utility: ExecutionAuthorization
    executor_calls: int


def decide_recovery(
    *, harness: UnifiedNeurosymbolicHarness,
    target: UnifiedTargetGrounding, target_executor_sha256: str,
) -> UnifiedRecoveryDecision:
    phase7 = harness.decide(target)
    utility = harness.runtime.decide(target.applicability)
    executor = ClevrerRepresentationExecutor(target_executor_sha256)
    action = harness.execute(
        phase7, utility, target, NATIVE_ACTIONS, executor,
    )
    selected = action if action is not None else "explicit_relation"
    if (phase7.verdict == TransferVerdict.SELECT_SKILL) != (action is not None):
        raise AssertionError("CLEVRER authorization/execution mismatch")
    return UnifiedRecoveryDecision(
        selected_native_representation=selected,
        phase7=phase7,
        utility=utility,
        executor_calls=executor.calls,
    )


__all__ = [
    "CAPABILITIES", "IR_KIND", "NATIVE_ACTIONS", "TARGET_DOMAIN",
    "TARGET_INTERFACE", "ClevrerRepresentationExecutor",
    "UnifiedRecoveryDecision", "build_harness", "build_route",
    "decide_recovery", "source_goal_relation_contract",
    "source_goal_relation_envelope", "target_grounding",
]
