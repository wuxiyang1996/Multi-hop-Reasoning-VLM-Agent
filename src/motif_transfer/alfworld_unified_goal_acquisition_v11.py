"""Unified-harness authority wrapper for ALFWorld goal acquisition.

The source program and the target neural grounder remain unchanged from V10.
This module adds the Phase-7 authority chain: anonymous structural matching and
pre-existing paired calibration must authorize the exact route before a
source-active decision can reach the ALFWorld-native executor.  The selector
never receives or emits an ALFWorld action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .alfworld_goal_acquisition_v10 import (
    AUTHENTIC,
    choose_goal_relation_action as _choose_v10,
)
from .contracts import stable_hash
from .structural_ir_applicability import (
    TargetIRRequirement,
    goal_acquisition_artifact_contract,
)
from .unified_neurosymbolic_harness import (
    InducedProgramEnvelope,
    Phase7Authorization,
    UnifiedNeurosymbolicHarness,
    UnifiedTargetGrounding,
    validate_phase7_authorization,
)
from .unified_transfer_runtime import (
    ExecutionAuthorization,
    PairedCalibration,
    TargetGroundingReceipt,
    TransferVerdict,
    UnifiedNeurosymbolicTransferRuntime,
    UnifiedRoute,
    validate_authorization,
)


ROUTE_ID = "sokoban-goal-acquisition-to-alfworld-multiplicity-v11"
TARGET_INTERFACE = "multiplicity_goal_acquisition_relation_v11"
REQUIRED_CAPABILITIES = (
    "exact_relation_handle",
    "multiplicity_two",
    "native_action_candidates",
    "neural_search_ranking",
    "positive_binding_cardinality",
)


@dataclass(frozen=True)
class UnifiedALFWorldAuthorization:
    harness: UnifiedNeurosymbolicHarness
    target: UnifiedTargetGrounding
    phase7: Phase7Authorization
    utility: ExecutionAuthorization
    target_executor_sha256: str


@dataclass
class _BoundNativeExecutor:
    """The only object in the source-active path allowed to emit an action."""

    artifact_sha256: str
    selected_action: str
    calls: int = 0

    def execute(
        self, authorization: ExecutionAuthorization,
        grounding: TargetGroundingReceipt,
        native_actions: Sequence[str],
    ) -> str:
        self.calls += 1
        if authorization.route_id != ROUTE_ID:
            raise ValueError("ALFWorld executor received the wrong route")
        if grounding.formal_outcome_read:
            raise ValueError("ALFWorld executor received an exposed outcome")
        if self.selected_action not in native_actions:
            raise ValueError("ALFWorld grounder proposed a non-native action")
        return self.selected_action


_CONTEXT: UnifiedALFWorldAuthorization | None = None
_AUTHORITY_RECEIPTS: list[dict[str, Any]] = []


def build_unified_authorization(
    *, task_id: str, acquisition_artifact: Mapping[str, Any],
    acquisition_confirmation: Mapping[str, Any],
    target_grounder_sha256: str, target_executor_sha256: str,
    evidence_report_sha256: str, inducer_artifact_sha256: str,
    utility_vs_neural: PairedCalibration = PairedCalibration(7, 0, 17),
    authenticity_vs_source_permuted: PairedCalibration = PairedCalibration(
        7, 0, 17,
    ),
) -> UnifiedALFWorldAuthorization:
    """Build an outcome-blind authorization from frozen V10 calibration."""

    contract = goal_acquisition_artifact_contract(
        acquisition_artifact, confirmation=acquisition_confirmation,
    )
    envelope = InducedProgramEnvelope.create(
        contract=contract,
        source_transition_receipts_sha256=str(
            acquisition_artifact["source_receipts_sha256"]
        ),
        inducer_artifact_sha256=str(inducer_artifact_sha256),
    )
    route = UnifiedRoute(
        route_id=ROUTE_ID,
        target_domain="alfworld",
        target_interface=TARGET_INTERFACE,
        required_capabilities=REQUIRED_CAPABILITIES,
        source_program_sha256=contract.program_sha256,
        source_program_induced_from_interventions=True,
        source_program_qualified=contract.source_intervention_qualified,
        target_grounder_sha256=str(target_grounder_sha256),
        target_executor_sha256=str(target_executor_sha256),
        target_grounder_id="alfworld.goal_acquisition_neural_grounder.v10",
        target_executor_id="alfworld.goal_acquisition_native_executor.v11",
        evidence_report_sha256=str(evidence_report_sha256),
        utility_vs_neural=utility_vs_neural,
        authenticity_vs_source_permuted=authenticity_vs_source_permuted,
    )
    runtime = UnifiedNeurosymbolicTransferRuntime((route,))
    harness = UnifiedNeurosymbolicHarness((envelope,), runtime)
    requirement = TargetIRRequirement.create(
        task_id=str(task_id), target_domain="alfworld",
        target_interface=TARGET_INTERFACE,
        target_grounder_sha256=str(target_grounder_sha256),
        ir_kind=contract.ir_kind,
        operator_sequence=contract.operator_sequence,
        recurrent=contract.recurrent,
        terminal_predicate_families=contract.terminal_predicate_families,
        grounder_qualified=True, formal_outcome_read=False,
    )
    applicability = TargetGroundingReceipt.create(
        task_id=str(task_id), target_domain="alfworld",
        target_interface=TARGET_INTERFACE,
        target_state_sha256=stable_hash({
            "task_id": str(task_id),
            "interface": TARGET_INTERFACE,
            "outcome_blind": True,
        }),
        target_grounder_sha256=str(target_grounder_sha256),
        capabilities=REQUIRED_CAPABILITIES,
        candidate_ids=("BIND", "RELATE", "SEARCH"),
        structural_predicates={
            "exact_relation_handle_supported": True,
            "multiplicity_two_interface": True,
            "positive_binding_cardinality_observable": True,
            "target_native_search_candidates_available": True,
        },
        grounder_qualified=True, formal_outcome_read=False,
    )
    target = UnifiedTargetGrounding.create(
        requirement=requirement, applicability=applicability,
    )
    phase7 = harness.decide(target)
    utility = runtime.decide(applicability)
    validate_phase7_authorization(phase7)
    validate_authorization(utility)
    if phase7.verdict != TransferVerdict.SELECT_SKILL:
        raise ValueError(f"unified ALFWorld route abstained: {phase7.reason}")
    return UnifiedALFWorldAuthorization(
        harness=harness, target=target, phase7=phase7, utility=utility,
        target_executor_sha256=str(target_executor_sha256),
    )


def configure_unified_authorization(
    authorization: UnifiedALFWorldAuthorization,
) -> None:
    validate_phase7_authorization(authorization.phase7)
    validate_authorization(authorization.utility)
    if authorization.phase7.verdict != TransferVerdict.SELECT_SKILL:
        raise ValueError("cannot configure an abstaining ALFWorld route")
    global _CONTEXT
    _CONTEXT = authorization
    _AUTHORITY_RECEIPTS.clear()


def authority_receipts() -> tuple[Mapping[str, Any], ...]:
    return tuple(dict(row) for row in _AUTHORITY_RECEIPTS)


def choose_goal_relation_action(**kwargs) -> dict[str, Any]:
    """Require unified authorization at every source-active V10 decision."""

    decision = _choose_v10(**kwargs)
    if kwargs.get("condition") != AUTHENTIC or not (
        decision.get("program_active") or decision.get("source_admitted")
    ):
        return decision
    if _CONTEXT is None:
        raise RuntimeError("unified ALFWorld authorization was not configured")
    action = str(decision["action"])
    native_actions = tuple(map(str, kwargs["grounded"].keys()))
    executor = _BoundNativeExecutor(
        artifact_sha256=_CONTEXT.target_executor_sha256,
        selected_action=action,
    )
    emitted = _CONTEXT.harness.execute(
        _CONTEXT.phase7, _CONTEXT.utility, _CONTEXT.target,
        native_actions, executor,
    )
    if emitted != action or executor.calls != 1:
        raise RuntimeError("unified ALFWorld executor did not emit exactly once")
    receipt_body = {
        "route_id": ROUTE_ID,
        "phase7_authorization_sha256": _CONTEXT.phase7.authorization_sha256,
        "utility_authorization_sha256": _CONTEXT.utility.authorization_sha256,
        "target_grounding_receipt_sha256": _CONTEXT.target.receipt_sha256,
        "source_selector_action_emitted": False,
        "target_executor_calls": executor.calls,
        "target_native_action": emitted,
        "formal_outcome_read": False,
    }
    _AUTHORITY_RECEIPTS.append(
        receipt_body | {"receipt_sha256": stable_hash(receipt_body)}
    )
    return decision | {
        "action": emitted,
        "unified_route_id": ROUTE_ID,
        "unified_authorization_sha256": (
            _CONTEXT.phase7.authorization_sha256
        ),
        "unified_utility_authorization_sha256": (
            _CONTEXT.utility.authorization_sha256
        ),
        "action_authority": "ALFWORLD_GOAL_ACQUISITION_NATIVE_EXECUTOR_V11",
    }


__all__ = [
    "REQUIRED_CAPABILITIES", "ROUTE_ID", "TARGET_INTERFACE",
    "UnifiedALFWorldAuthorization", "authority_receipts",
    "build_unified_authorization",
    "choose_goal_relation_action", "configure_unified_authorization",
]
