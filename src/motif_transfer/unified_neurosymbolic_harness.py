"""Phase-7 composition of induction, grounding, utility, and execution.

The individual components remain deliberately separate authorities.  This
module supplies the narrow composition layer and enforces that all of them
agree on the same content-addressed program and target state before the
target-native executor can act.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

from .contracts import stable_hash
from .structural_ir_applicability import (
    SourceIRContract,
    TargetIRRequirement,
    select_source_contract,
)
from .neural_multi_ir_selector import NeuralMultiIRSelector
from .unified_transfer_runtime import (
    ExecutionAuthorization,
    SelectiveTargetExecutor,
    TargetGroundingReceipt,
    TargetNativeExecutor,
    TransferVerdict,
    UnifiedNeurosymbolicTransferRuntime,
    UnifiedRuntimeError,
    validate_authorization,
)


class SourceOnlyInducer(Protocol):
    """Shared contract for implementations that learn from source tuples."""

    artifact_sha256: str

    def induce(
        self, transitions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Learn a typed program without target data or named templates."""


@dataclass(frozen=True)
class InducedProgramEnvelope:
    contract: SourceIRContract
    source_transition_receipts_sha256: str
    inducer_artifact_sha256: str
    learned_from_state_action_effect_next_state: bool
    target_data_read: bool
    named_policy_template_used: bool
    envelope_sha256: str

    @classmethod
    def create(
        cls, *, contract: SourceIRContract,
        source_transition_receipts_sha256: str,
        inducer_artifact_sha256: str,
        learned_from_state_action_effect_next_state: bool = True,
        target_data_read: bool = False,
        named_policy_template_used: bool = False,
    ) -> "InducedProgramEnvelope":
        core = {
            "contract_sha256": contract.contract_sha256,
            "source_transition_receipts_sha256": str(
                source_transition_receipts_sha256
            ),
            "inducer_artifact_sha256": str(inducer_artifact_sha256),
            "learned_from_state_action_effect_next_state": bool(
                learned_from_state_action_effect_next_state
            ),
            "target_data_read": bool(target_data_read),
            "named_policy_template_used": bool(named_policy_template_used),
        }
        return cls(
            contract=contract,
            source_transition_receipts_sha256=core[
                "source_transition_receipts_sha256"
            ],
            inducer_artifact_sha256=core["inducer_artifact_sha256"],
            learned_from_state_action_effect_next_state=core[
                "learned_from_state_action_effect_next_state"
            ],
            target_data_read=core["target_data_read"],
            named_policy_template_used=core["named_policy_template_used"],
            envelope_sha256=stable_hash(core),
        )

    @property
    def admitted(self) -> bool:
        return (
            self.contract.source_intervention_qualified
            and self.learned_from_state_action_effect_next_state
            and not self.target_data_read
            and not self.named_policy_template_used
        )

    def validate(self) -> None:
        self.contract.validate()
        core = {
            "contract_sha256": self.contract.contract_sha256,
            "source_transition_receipts_sha256": (
                self.source_transition_receipts_sha256
            ),
            "inducer_artifact_sha256": self.inducer_artifact_sha256,
            "learned_from_state_action_effect_next_state": (
                self.learned_from_state_action_effect_next_state
            ),
            "target_data_read": self.target_data_read,
            "named_policy_template_used": self.named_policy_template_used,
        }
        if stable_hash(core) != self.envelope_sha256:
            raise UnifiedRuntimeError("induced program envelope hash mismatch")


@dataclass(frozen=True)
class UnifiedTargetGrounding:
    requirement: TargetIRRequirement
    applicability: TargetGroundingReceipt
    receipt_sha256: str

    @classmethod
    def create(
        cls, *, requirement: TargetIRRequirement,
        applicability: TargetGroundingReceipt,
    ) -> "UnifiedTargetGrounding":
        requirement.validate()
        applicability.validate()
        aligned = (
            requirement.task_id == applicability.task_id
            and requirement.target_domain == applicability.target_domain
            and requirement.target_interface == applicability.target_interface
            and requirement.target_grounder_sha256
            == applicability.target_grounder_sha256
            and requirement.formal_outcome_read
            == applicability.formal_outcome_read
        )
        if not aligned:
            raise UnifiedRuntimeError(
                "target structural requirement/applicability receipt mismatch"
            )
        core = {
            "requirement_sha256": requirement.requirement_sha256,
            "applicability_receipt_sha256": applicability.receipt_sha256,
        }
        return cls(
            requirement=requirement,
            applicability=applicability,
            receipt_sha256=stable_hash(core),
        )


@dataclass(frozen=True)
class Phase7Authorization:
    verdict: TransferVerdict
    reason: str
    task_id: str
    selected_program_sha256: str | None
    route_id: str | None
    structural_selection_receipt_sha256: str
    utility_authorization_sha256: str | None
    target_grounding_receipt_sha256: str
    current_target_outcome_read: bool
    target_action_emitted: bool
    authorization_sha256: str


class UnifiedNeurosymbolicHarness:
    """Fail-closed Phase-7 orchestrator; it never emits an action itself."""

    def __init__(
        self, programs: Sequence[InducedProgramEnvelope],
        runtime: UnifiedNeurosymbolicTransferRuntime,
    ):
        if not programs:
            raise UnifiedRuntimeError("unified harness requires source programs")
        for program in programs:
            program.validate()
        hashes = [row.contract.program_sha256 for row in programs]
        if len(hashes) != len(set(hashes)):
            raise UnifiedRuntimeError("duplicate source program in harness")
        self.programs = tuple(programs)
        self.runtime = runtime

    @staticmethod
    def _authorization(
        *, verdict: TransferVerdict, reason: str,
        target: UnifiedTargetGrounding,
        selection: Mapping[str, Any],
        utility: ExecutionAuthorization | None,
    ) -> Phase7Authorization:
        selected = selection.get("selected_program_sha256")
        body = {
            "verdict": verdict.value,
            "reason": str(reason),
            "task_id": target.applicability.task_id,
            "selected_program_sha256": selected,
            "route_id": utility.route_id if utility else None,
            "structural_selection_receipt_sha256": str(
                selection["receipt_sha256"]
            ),
            "utility_authorization_sha256": (
                utility.authorization_sha256 if utility else None
            ),
            "target_grounding_receipt_sha256": target.receipt_sha256,
            "current_target_outcome_read": False,
            "target_action_emitted": False,
        }
        return Phase7Authorization(
            verdict=verdict,
            reason=body["reason"],
            task_id=body["task_id"],
            selected_program_sha256=body["selected_program_sha256"],
            route_id=body["route_id"],
            structural_selection_receipt_sha256=body[
                "structural_selection_receipt_sha256"
            ],
            utility_authorization_sha256=body[
                "utility_authorization_sha256"
            ],
            target_grounding_receipt_sha256=body[
                "target_grounding_receipt_sha256"
            ],
            current_target_outcome_read=False,
            target_action_emitted=False,
            authorization_sha256=stable_hash(body),
        )

    def decide(self, target: UnifiedTargetGrounding) -> Phase7Authorization:
        eligible = [row.contract for row in self.programs if row.admitted]
        selection = select_source_contract(eligible, target.requirement)
        if selection["status"] != "UNIQUE_SOURCE_CONTRACT_SELECTED":
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason=f"STRUCTURAL_SELECTION:{selection['reason']}",
                target=target, selection=selection, utility=None,
            )
        utility = self.runtime.decide(target.applicability)
        validate_authorization(utility)
        if utility.verdict != TransferVerdict.SELECT_SKILL:
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason=f"UTILITY_ROUTER:{utility.reason}",
                target=target, selection=selection, utility=utility,
            )
        if utility.source_program_sha256 != selection[
            "selected_program_sha256"
        ]:
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason="ROUTE_SOURCE_CONTRACT_MISMATCH",
                target=target, selection=selection, utility=utility,
            )
        return self._authorization(
            verdict=TransferVerdict.SELECT_SKILL,
            reason="STRUCTURE_AND_CALIBRATED_UTILITY_AGREE",
            target=target, selection=selection, utility=utility,
        )

    def decide_neural(
        self, target: UnifiedTargetGrounding,
        selector: NeuralMultiIRSelector,
    ) -> Phase7Authorization:
        """Use a neural structural selector without Python exact-match repair.

        The frozen utility router remains a separate safety and calibration
        authority.  A legal but wrong neural catalog choice reaches the route
        agreement check and causes abstention; it is never replaced by the
        deterministic choice from :func:`select_source_contract`.
        """

        eligible = [row.contract for row in self.programs if row.admitted]
        neural = selector.decide(
            contracts=eligible, requirement=target.requirement,
        )
        selection = {
            "receipt_sha256": neural.receipt_sha256,
            "selected_program_sha256": neural.selected_program_sha256,
        }
        if not neural.source_program_authorized:
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason=f"NEURAL_STRUCTURAL_SELECTION:{neural.reason}",
                target=target, selection=selection, utility=None,
            )
        utility = self.runtime.decide(target.applicability)
        validate_authorization(utility)
        if utility.verdict != TransferVerdict.SELECT_SKILL:
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason=f"UTILITY_ROUTER:{utility.reason}",
                target=target, selection=selection, utility=utility,
            )
        if utility.source_program_sha256 != neural.selected_program_sha256:
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason="NEURAL_ROUTE_SOURCE_CONTRACT_MISMATCH",
                target=target, selection=selection, utility=utility,
            )
        return self._authorization(
            verdict=TransferVerdict.SELECT_SKILL,
            reason="NEURAL_STRUCTURE_AND_CALIBRATED_UTILITY_AGREE",
            target=target, selection=selection, utility=utility,
        )

    @staticmethod
    def execute(
        phase7: Phase7Authorization,
        utility: ExecutionAuthorization,
        target: UnifiedTargetGrounding,
        native_actions: Sequence[str],
        executor: TargetNativeExecutor,
    ) -> str | None:
        validate_phase7_authorization(phase7)
        validate_authorization(utility)
        if phase7.verdict != TransferVerdict.SELECT_SKILL:
            return None
        if phase7.utility_authorization_sha256 != utility.authorization_sha256:
            raise UnifiedRuntimeError("Phase-7/utility authorization mismatch")
        if phase7.selected_program_sha256 != utility.source_program_sha256:
            raise UnifiedRuntimeError("Phase-7/source program mismatch")
        return SelectiveTargetExecutor.execute(
            utility, target.applicability, native_actions, executor,
        )


def validate_phase7_authorization(value: Phase7Authorization) -> None:
    body = asdict(value)
    claimed = body.pop("authorization_sha256")
    body["verdict"] = value.verdict.value
    if stable_hash(body) != claimed:
        raise UnifiedRuntimeError("Phase-7 authorization hash mismatch")
    if value.current_target_outcome_read or value.target_action_emitted:
        raise UnifiedRuntimeError("Phase-7 selector crossed an authority boundary")
    if value.verdict == TransferVerdict.SELECT_SKILL and not all((
        value.selected_program_sha256,
        value.route_id,
        value.utility_authorization_sha256,
    )):
        raise UnifiedRuntimeError("Phase-7 selected authorization is incomplete")


__all__ = [
    "InducedProgramEnvelope", "Phase7Authorization", "SourceOnlyInducer",
    "UnifiedNeurosymbolicHarness", "UnifiedTargetGrounding",
    "validate_phase7_authorization",
]
