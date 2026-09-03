"""Execute source-induced transition structure as target grounding-tool control.

This layer does not infer when to rescan from benchmark names.  A target-native
grounder exposes the truth values of the source program's typed transition,
terminal, and abstention guards.  The source controller then executes those
guards unchanged and returns an anonymous intervention.  A separate target
adapter binds that intervention to a CLEVRER or AGQA tool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import re
from typing import Any, Mapping

from .contracts import stable_hash
from .structural_ir_applicability import SourceIRContract


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GroundingControlVerdict(str, Enum):
    APPLY_TRANSITION = "APPLY_TRANSITION"
    COMMIT = "COMMIT"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class TypedGroundingControlState:
    task_id: str
    target_domain: str
    target_state_sha256: str
    transition_guard_observable: bool
    transition_guard_satisfied: bool
    transition_effect_authenticated: bool
    terminal_guard_observable: bool
    terminal_guard_satisfied: bool
    abstention_guard_satisfied: bool
    interventions_used: int
    intervention_budget: int
    formal_outcome_read: bool

    def validate(self) -> None:
        if not self.task_id or not self.target_domain:
            raise ValueError("grounding-control state identity is incomplete")
        if _SHA256.fullmatch(self.target_state_sha256) is None:
            raise ValueError("grounding-control state is not content-addressed")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.interventions_used, self.intervention_budget)
        ):
            raise ValueError("intervention counts must be integers")
        if not 0 <= self.interventions_used <= self.intervention_budget:
            raise ValueError("invalid intervention usage")


@dataclass(frozen=True)
class GroundingControlAuthorization:
    verdict: GroundingControlVerdict
    reason: str
    task_id: str
    source_program_sha256: str
    target_state_sha256: str
    anonymous_intervention: str | None
    formal_outcome_read: bool
    authorization_sha256: str

    def validate(self) -> None:
        body = asdict(self)
        claimed = body.pop("authorization_sha256")
        body["verdict"] = self.verdict.value
        if stable_hash(body) != claimed:
            raise ValueError("grounding-control authorization hash mismatch")
        if self.formal_outcome_read and self.verdict != GroundingControlVerdict.ABSTAIN:
            raise ValueError("outcome-exposed controller did not abstain")
        if (
            self.verdict == GroundingControlVerdict.APPLY_TRANSITION
            and not self.anonymous_intervention
        ):
            raise ValueError("transition authorization lacks an intervention")


class SourceControlledGroundingPolicy:
    """Domain-blind interpreter for one qualified source IR contract."""

    def __init__(self, contract: SourceIRContract):
        contract.validate()
        if not contract.source_intervention_qualified:
            raise ValueError("source grounding controller needs qualified interventions")
        self.contract = contract

    def decide(
        self, state: TypedGroundingControlState,
    ) -> GroundingControlAuthorization:
        state.validate()
        if state.formal_outcome_read:
            verdict = GroundingControlVerdict.ABSTAIN
            reason = "CURRENT_TARGET_OUTCOME_EXPOSED"
            intervention = None
        elif state.abstention_guard_satisfied:
            verdict = GroundingControlVerdict.ABSTAIN
            reason = "SOURCE_ABSTENTION_GUARD_SATISFIED"
            intervention = None
        elif state.terminal_guard_observable and state.terminal_guard_satisfied:
            verdict = GroundingControlVerdict.COMMIT
            reason = "SOURCE_TERMINAL_GUARD_SATISFIED"
            intervention = None
        elif not state.transition_guard_observable:
            verdict = GroundingControlVerdict.ABSTAIN
            reason = "SOURCE_TRANSITION_GUARD_UNOBSERVABLE"
            intervention = None
        elif not state.transition_effect_authenticated:
            verdict = GroundingControlVerdict.ABSTAIN
            reason = "SOURCE_TRANSITION_EFFECT_UNAUTHENTICATED"
            intervention = None
        elif state.interventions_used >= state.intervention_budget:
            verdict = GroundingControlVerdict.ABSTAIN
            reason = "GROUNDING_INTERVENTION_BUDGET_EXHAUSTED"
            intervention = None
        elif not state.transition_guard_satisfied:
            verdict = GroundingControlVerdict.ABSTAIN
            reason = "SOURCE_TRANSITION_GUARD_NOT_SATISFIED"
            intervention = None
        else:
            verdict = GroundingControlVerdict.APPLY_TRANSITION
            reason = "SOURCE_TYPED_TRANSITION_AUTHORIZED"
            # Anonymous by construction: no benchmark or source action token.
            intervention = "APPLY_TYPED_TRANSITION"
        body = {
            "verdict": verdict.value,
            "reason": reason,
            "task_id": state.task_id,
            "source_program_sha256": self.contract.program_sha256,
            "target_state_sha256": state.target_state_sha256,
            "anonymous_intervention": intervention,
            "formal_outcome_read": state.formal_outcome_read,
        }
        result = GroundingControlAuthorization(
            verdict=verdict,
            reason=reason,
            task_id=state.task_id,
            source_program_sha256=self.contract.program_sha256,
            target_state_sha256=state.target_state_sha256,
            anonymous_intervention=intervention,
            formal_outcome_read=state.formal_outcome_read,
            authorization_sha256=stable_hash(body),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class TargetGroundingToolBinding:
    target_domain: str
    target_adapter_sha256: str
    transition_tool: str
    transition_arguments: Mapping[str, Any]

    def validate(self) -> None:
        if not self.target_domain or not self.transition_tool:
            raise ValueError("target grounding-tool binding is incomplete")
        if _SHA256.fullmatch(self.target_adapter_sha256) is None:
            raise ValueError("target grounding adapter is not content-addressed")


def bind_authorized_grounding_tool(
    authorization: GroundingControlAuthorization,
    binding: TargetGroundingToolBinding,
) -> Mapping[str, Any] | None:
    """Bind an anonymous source transition to a target-native tool only."""

    authorization.validate()
    binding.validate()
    if authorization.verdict != GroundingControlVerdict.APPLY_TRANSITION:
        return None
    body = {
        "task_id": authorization.task_id,
        "source_program_sha256": authorization.source_program_sha256,
        "source_authorization_sha256": authorization.authorization_sha256,
        "target_domain": binding.target_domain,
        "target_adapter_sha256": binding.target_adapter_sha256,
        "tool": binding.transition_tool,
        "arguments": dict(binding.transition_arguments),
        "gold_or_target_outcome": "NOT_READ",
    }
    return body | {"binding_receipt_sha256": stable_hash(body)}


__all__ = [
    "GroundingControlAuthorization",
    "GroundingControlVerdict",
    "SourceControlledGroundingPolicy",
    "TargetGroundingToolBinding",
    "TypedGroundingControlState",
    "bind_authorized_grounding_tool",
]
