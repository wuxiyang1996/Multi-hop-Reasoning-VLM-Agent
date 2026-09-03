"""Unified fail-closed runtime for source-induced neural-symbolic transfer.

The runtime deliberately separates four authorities:

* a source-only learner supplies a content-addressed symbolic program;
* a target-native grounder supplies current-state bindings and applicability;
* completed, *earlier* matched target trials calibrate directional utility and
  authenticity against a source-permuted control; and
* a target-native executor is the only component allowed to emit an action.

The selector returns an execution authorization, never a target action.  A
current task outcome is not accepted by any API in this module and therefore
cannot select the route that acts on that same task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Protocol, Sequence

from .contracts import stable_hash
from .online_transfer_utility import (
    ApplicabilityReceipt,
    OnlineTransferUtilityGate,
    PairedOutcome,
)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


class UnifiedRuntimeError(ValueError):
    """Raised when a frozen route or runtime receipt is inconsistent."""


class TransferVerdict(str, Enum):
    SELECT_SKILL = "SELECT_SKILL"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class PairedCalibration:
    """Completed paired evidence from tasks preceding the current decision."""

    wins: int
    losses: int
    ties: int

    def validate(self) -> None:
        values = (self.wins, self.losses, self.ties)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise UnifiedRuntimeError("paired calibration counts must be integers")
        if any(value < 0 for value in values):
            raise UnifiedRuntimeError("paired calibration counts must be nonnegative")

    @property
    def exposures(self) -> int:
        return self.wins + self.losses + self.ties

    def utility_gate(self, applicability: ApplicabilityReceipt):
        self.validate()
        gate = OnlineTransferUtilityGate()
        gate.update_many(
            [PairedOutcome(True, False)] * self.wins
            + [PairedOutcome(False, True)] * self.losses
            + [PairedOutcome(True, True)] * self.ties
        )
        return gate.decision(applicability)


@dataclass(frozen=True)
class UnifiedRoute:
    """One exact source-program/target-interface route."""

    route_id: str
    target_domain: str
    target_interface: str
    required_capabilities: tuple[str, ...]
    source_program_sha256: str
    source_program_induced_from_interventions: bool
    source_program_qualified: bool
    target_grounder_sha256: str
    target_executor_sha256: str
    target_grounder_id: str
    target_executor_id: str
    evidence_report_sha256: str
    utility_vs_neural: PairedCalibration
    authenticity_vs_source_permuted: PairedCalibration

    def validate(self) -> None:
        if not self.route_id or not self.target_domain or not self.target_interface:
            raise UnifiedRuntimeError("route identity is incomplete")
        if not self.required_capabilities:
            raise UnifiedRuntimeError("route has no required target capabilities")
        if len(set(self.required_capabilities)) != len(self.required_capabilities):
            raise UnifiedRuntimeError("route capabilities are not unique")
        for name, value in (
            ("source program", self.source_program_sha256),
            ("target grounder", self.target_grounder_sha256),
            ("target executor", self.target_executor_sha256),
            ("evidence report", self.evidence_report_sha256),
        ):
            if not _is_sha256(value):
                raise UnifiedRuntimeError(f"{name} is not a lowercase sha256")
        self.utility_vs_neural.validate()
        self.authenticity_vs_source_permuted.validate()


@dataclass(frozen=True)
class TargetGroundingReceipt:
    """Outcome-blind target-native applicability evidence for one state."""

    task_id: str
    target_domain: str
    target_interface: str
    target_state_sha256: str
    target_grounder_sha256: str
    capabilities: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    structural_predicates: tuple[tuple[str, bool], ...]
    grounder_qualified: bool
    formal_outcome_read: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls, *, task_id: str, target_domain: str, target_interface: str,
        target_state_sha256: str, target_grounder_sha256: str,
        capabilities: Sequence[str], candidate_ids: Sequence[str],
        structural_predicates: Mapping[str, bool], grounder_qualified: bool,
        formal_outcome_read: bool = False,
    ) -> "TargetGroundingReceipt":
        body = {
            "task_id": str(task_id),
            "target_domain": str(target_domain),
            "target_interface": str(target_interface),
            "target_state_sha256": str(target_state_sha256),
            "target_grounder_sha256": str(target_grounder_sha256),
            "capabilities": tuple(sorted({str(value) for value in capabilities})),
            "candidate_ids": tuple(map(str, candidate_ids)),
            "structural_predicates": tuple(sorted(
                (str(key), bool(value)) for key, value in structural_predicates.items()
            )),
            "grounder_qualified": bool(grounder_qualified),
            "formal_outcome_read": bool(formal_outcome_read),
        }
        receipt = cls(**body, receipt_sha256=stable_hash(body))
        receipt.validate()
        return receipt

    @property
    def structural_applicable(self) -> bool:
        return bool(self.structural_predicates) and all(
            value for _, value in self.structural_predicates
        )

    def validate(self) -> None:
        if not self.task_id or not self.target_domain or not self.target_interface:
            raise UnifiedRuntimeError("target grounding identity is incomplete")
        if not _is_sha256(self.target_state_sha256):
            raise UnifiedRuntimeError("target state is not content-addressed")
        if not _is_sha256(self.target_grounder_sha256):
            raise UnifiedRuntimeError("target grounder is not content-addressed")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise UnifiedRuntimeError("target candidate ids are not unique")
        if len({key for key, _ in self.structural_predicates}) != len(
            self.structural_predicates
        ):
            raise UnifiedRuntimeError("structural predicate names are not unique")
        body = asdict(self)
        claimed = body.pop("receipt_sha256")
        if stable_hash(body) != claimed:
            raise UnifiedRuntimeError("target grounding receipt hash mismatch")


@dataclass(frozen=True)
class ExecutionAuthorization:
    """Selector output.  It intentionally contains no action field."""

    verdict: TransferVerdict
    reason: str
    task_id: str
    target_state_sha256: str
    route_id: str | None
    source_program_sha256: str | None
    target_grounder_id: str | None
    target_executor_id: str | None
    target_executor_sha256: str | None
    utility_lower_bound: float
    authenticity_lower_bound: float
    current_outcome_read: bool
    authorization_sha256: str


class TargetNativeGrounder(Protocol):
    artifact_sha256: str

    def ground(self, state: Any) -> TargetGroundingReceipt:
        """Bind one target-native state without reading its formal outcome."""


class TargetNativeExecutor(Protocol):
    artifact_sha256: str

    def execute(
        self, authorization: ExecutionAuthorization,
        grounding: TargetGroundingReceipt,
        native_actions: Sequence[str],
    ) -> str:
        """Return a native action after a valid authorization."""


class UnifiedNeurosymbolicTransferRuntime:
    """Exact structural routing plus calibrated utility and authenticity."""

    def __init__(self, routes: Sequence[UnifiedRoute]):
        if not routes:
            raise UnifiedRuntimeError("runtime requires at least one route")
        for route in routes:
            route.validate()
        ids = [route.route_id for route in routes]
        if len(ids) != len(set(ids)):
            raise UnifiedRuntimeError("duplicate unified route id")
        self.routes = tuple(routes)

    @staticmethod
    def _authorization(
        *, verdict: TransferVerdict, reason: str,
        grounding: TargetGroundingReceipt, route: UnifiedRoute | None = None,
        utility_lower_bound: float = 0.0,
        authenticity_lower_bound: float = 0.0,
    ) -> ExecutionAuthorization:
        body = {
            "verdict": verdict.value,
            "reason": str(reason),
            "task_id": grounding.task_id,
            "target_state_sha256": grounding.target_state_sha256,
            "route_id": route.route_id if route else None,
            "source_program_sha256": route.source_program_sha256 if route else None,
            "target_grounder_id": route.target_grounder_id if route else None,
            "target_executor_id": route.target_executor_id if route else None,
            "target_executor_sha256": route.target_executor_sha256 if route else None,
            "utility_lower_bound": float(utility_lower_bound),
            "authenticity_lower_bound": float(authenticity_lower_bound),
            "current_outcome_read": False,
        }
        return ExecutionAuthorization(
            verdict=verdict,
            reason=body["reason"],
            task_id=body["task_id"],
            target_state_sha256=body["target_state_sha256"],
            route_id=body["route_id"],
            source_program_sha256=body["source_program_sha256"],
            target_grounder_id=body["target_grounder_id"],
            target_executor_id=body["target_executor_id"],
            target_executor_sha256=body["target_executor_sha256"],
            utility_lower_bound=body["utility_lower_bound"],
            authenticity_lower_bound=body["authenticity_lower_bound"],
            current_outcome_read=False,
            authorization_sha256=stable_hash(body),
        )

    def decide(self, grounding: TargetGroundingReceipt) -> ExecutionAuthorization:
        grounding.validate()
        if grounding.formal_outcome_read:
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason="CURRENT_TASK_OUTCOME_EXPOSURE",
                grounding=grounding,
            )
        capability_set = set(grounding.capabilities)
        candidates = [
            route for route in self.routes
            if route.target_domain == grounding.target_domain
            and route.target_interface == grounding.target_interface
            and set(route.required_capabilities) <= capability_set
        ]
        if len(candidates) > 1:
            raise UnifiedRuntimeError("ambiguous exact target route")
        if not candidates:
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason="NO_EXACT_TARGET_INTERFACE_ROUTE",
                grounding=grounding,
            )
        route = candidates[0]
        if not route.source_program_induced_from_interventions:
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason="SOURCE_PROGRAM_NOT_INTERVENTION_INDUCED",
                grounding=grounding, route=route,
            )
        if not route.source_program_qualified:
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason="SOURCE_PROGRAM_NOT_QUALIFIED",
                grounding=grounding, route=route,
            )
        if grounding.target_grounder_sha256 != route.target_grounder_sha256:
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason="TARGET_GROUNDER_HASH_MISMATCH",
                grounding=grounding, route=route,
            )
        if not grounding.grounder_qualified or not grounding.structural_applicable:
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason="CURRENT_STATE_STRUCTURAL_APPLICABILITY_FAILED",
                grounding=grounding, route=route,
            )
        applicability = ApplicabilityReceipt(True, True, True, True, True)
        utility = route.utility_vs_neural.utility_gate(applicability)
        authenticity = route.authenticity_vs_source_permuted.utility_gate(applicability)
        lower_utility = utility.posterior_lower_win_probability
        lower_authenticity = authenticity.posterior_lower_win_probability
        if utility.decision != "SELECT_SKILL":
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason="DIRECTIONAL_UTILITY_NOT_CALIBRATED",
                grounding=grounding, route=route,
                utility_lower_bound=lower_utility,
                authenticity_lower_bound=lower_authenticity,
            )
        if authenticity.decision != "SELECT_SKILL":
            return self._authorization(
                verdict=TransferVerdict.ABSTAIN,
                reason="SOURCE_SPECIFIC_AUTHENTICITY_NOT_CALIBRATED",
                grounding=grounding, route=route,
                utility_lower_bound=lower_utility,
                authenticity_lower_bound=lower_authenticity,
            )
        return self._authorization(
            verdict=TransferVerdict.SELECT_SKILL,
            reason="STRUCTURALLY_APPLICABLE_CALIBRATED_SOURCE_PROGRAM",
            grounding=grounding, route=route,
            utility_lower_bound=lower_utility,
            authenticity_lower_bound=lower_authenticity,
        )


class SelectiveTargetExecutor:
    """Enforce target-native action authority after route selection."""

    @staticmethod
    def execute(
        authorization: ExecutionAuthorization,
        grounding: TargetGroundingReceipt,
        native_actions: Sequence[str],
        executor: TargetNativeExecutor,
    ) -> str | None:
        if authorization.verdict != TransferVerdict.SELECT_SKILL:
            return None
        if authorization.current_outcome_read:
            raise UnifiedRuntimeError("authorization used a current target outcome")
        if authorization.target_state_sha256 != grounding.target_state_sha256:
            raise UnifiedRuntimeError("authorization/grounding state mismatch")
        if executor.artifact_sha256 != authorization.target_executor_sha256:
            raise UnifiedRuntimeError("target executor hash mismatch")
        actions = tuple(map(str, native_actions))
        if not actions or len(actions) != len(set(actions)):
            raise UnifiedRuntimeError("native action set must be nonempty and unique")
        action = str(executor.execute(authorization, grounding, actions))
        if action not in actions:
            raise UnifiedRuntimeError("target executor emitted a non-native action")
        return action


def validate_authorization(authorization: ExecutionAuthorization) -> None:
    body = asdict(authorization)
    claimed = body.pop("authorization_sha256")
    body["verdict"] = authorization.verdict.value
    if stable_hash(body) != claimed:
        raise UnifiedRuntimeError("execution authorization hash mismatch")
    for value in (
        authorization.utility_lower_bound,
        authorization.authenticity_lower_bound,
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise UnifiedRuntimeError("authorization probability is invalid")
    if authorization.current_outcome_read:
        raise UnifiedRuntimeError("execution authorization read current outcome")
    if authorization.verdict == TransferVerdict.SELECT_SKILL:
        if not all((
            authorization.route_id,
            authorization.source_program_sha256,
            authorization.target_grounder_id,
            authorization.target_executor_id,
            authorization.target_executor_sha256,
        )):
            raise UnifiedRuntimeError("selected authorization is incomplete")
        if hasattr(authorization, "action"):
            raise UnifiedRuntimeError("selector must not carry a target action")


__all__ = [
    "ExecutionAuthorization",
    "PairedCalibration",
    "SelectiveTargetExecutor",
    "TargetGroundingReceipt",
    "TargetNativeExecutor",
    "TargetNativeGrounder",
    "TransferVerdict",
    "UnifiedNeurosymbolicTransferRuntime",
    "UnifiedRoute",
    "UnifiedRuntimeError",
    "validate_authorization",
]
