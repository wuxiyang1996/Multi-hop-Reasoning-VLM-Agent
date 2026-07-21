"""Proof-carrying, domain-native multi-step trace programs.

This IR deliberately represents what was observed, not a hand-authored skill
ontology.  Agents may attach control hypotheses, but observational replay never
upgrades an unobserved guard, branch, retry, verification, or termination claim
to an intervention-verified fact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Sequence


class TraceProgramStatus(str, Enum):
    OBSERVED_TRACE = "OBSERVED_TRACE"
    AGENT_HYPOTHESIS = "AGENT_HYPOTHESIS"
    INTERVENTION_VERIFIED = "INTERVENTION_VERIFIED"


class ControlClaimKind(str, Enum):
    SKILL_BOUNDARY = "SKILL_BOUNDARY"
    GUARD = "GUARD"
    BRANCH = "BRANCH"
    LOOP = "LOOP"
    RETRY = "RETRY"
    VERIFY = "VERIFY"
    TERMINATE = "TERMINATE"


@dataclass(frozen=True)
class NativeTransitionReceipt:
    transition_id: str
    step_index: int
    state_sha256: str
    next_state_sha256: str
    available_actions_sha256: str
    action: str
    reward: float
    done: bool


@dataclass(frozen=True)
class ObservedOrderEdge:
    before_transition_id: str
    after_transition_id: str


@dataclass(frozen=True)
class ControlHypothesis:
    claim_id: str
    kind: ControlClaimKind
    anchor_transition_ids: Sequence[str]
    proposal_source: str
    proposal_receipt_sha256: str
    status: TraceProgramStatus = TraceProgramStatus.AGENT_HYPOTHESIS
    intervention_receipt_ids: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class BackboneCoverage:
    observation_receipted: bool
    admissibility_receipted: bool
    environment_step_receipted: bool
    native_delta_receipted: bool
    agent_proposal_receipted: bool
    continuation_decision_receipted: bool
    official_stop_receipted: bool


@dataclass
class TraceProgram:
    program_id: str
    game: str
    episode_id: str
    source_file_sha256: str
    transitions: Sequence[NativeTransitionReceipt]
    observed_order: Sequence[ObservedOrderEdge]
    coverage: BackboneCoverage
    hypotheses: Sequence[ControlHypothesis] = field(default_factory=tuple)
    status: TraceProgramStatus = TraceProgramStatus.OBSERVED_TRACE
    full_reset_to_stop_trace: bool = True
    official_success_verified: bool = False
    schema_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate_structure(self) -> None:
        if not self.program_id or not self.game or not self.episode_id:
            raise ValueError("trace program identity is incomplete")
        _validate_digest(self.source_file_sha256, "source_file_sha256")
        if len(self.transitions) < 2:
            raise ValueError("a multi-step trace program requires at least two transitions")
        ids = [item.transition_id for item in self.transitions]
        if len(ids) != len(set(ids)):
            raise ValueError("transition IDs must be unique")
        for item in self.transitions:
            for name, digest in (
                ("state_sha256", item.state_sha256),
                ("next_state_sha256", item.next_state_sha256),
                ("available_actions_sha256", item.available_actions_sha256),
            ):
                _validate_digest(digest, name)
            if not item.action:
                raise ValueError("observed transition action is empty")
        expected_edges = [
            ObservedOrderEdge(before, after) for before, after in zip(ids, ids[1:])
        ]
        if list(self.observed_order) != expected_edges:
            raise ValueError("observed_order must be the exact contiguous trace chain")
        known = set(ids)
        for claim in self.hypotheses:
            _validate_digest(claim.proposal_receipt_sha256, "proposal_receipt_sha256")
            if not set(claim.anchor_transition_ids).issubset(known):
                raise ValueError(f"control claim cites an unknown transition: {claim.claim_id}")
            if claim.status == TraceProgramStatus.INTERVENTION_VERIFIED and not claim.intervention_receipt_ids:
                raise ValueError("intervention-verified claim lacks intervention receipts")
        if self.status == TraceProgramStatus.INTERVENTION_VERIFIED:
            if not self.hypotheses or any(
                claim.status != TraceProgramStatus.INTERVENTION_VERIFIED
                for claim in self.hypotheses
            ):
                raise ValueError("program cannot outrank its control claims")

    def content_hash(self) -> str:
        payload = _jsonable(asdict(self))
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        payload = _jsonable(asdict(self))
        payload["program_hash"] = self.content_hash()
        return payload


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase sha256 digest")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "BackboneCoverage",
    "ControlClaimKind",
    "ControlHypothesis",
    "NativeTransitionReceipt",
    "ObservedOrderEdge",
    "TraceProgram",
    "TraceProgramStatus",
]
