from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Sequence


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Lifecycle(str, Enum):
    CANDIDATE = "CANDIDATE"
    SOURCE_SUPPORTED = "SOURCE_SUPPORTED"
    TARGET_PROVISIONAL = "TARGET_PROVISIONAL"
    POSITIVE_TRANSFER = "POSITIVE_TRANSFER"
    NEGATIVE_TRANSFER = "NEGATIVE_TRANSFER"
    GENERIC_ONLY = "GENERIC_ONLY"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    REJECTED = "REJECTED"


class AdvisoryVerdict(str, Enum):
    ADMIT = "ADMIT"
    REPLAN = "REPLAN"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class Observation:
    state: Mapping[str, Any]
    native_actions: tuple[str, ...]
    terminal: bool = False
    official_success: bool = False
    score: float = 0.0


@dataclass(frozen=True)
class DecisionProposal:
    proposal_id: str
    action: str
    prediction: str = ""
    rationale: str = ""
    agent_id: str = "decision-agent"


@dataclass(frozen=True)
class Advisory:
    verdict: AdvisoryVerdict
    reason: str
    evidence_receipt_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TransitionReceipt:
    receipt_id: str
    before_hash: str
    native_actions_hash: str
    proposal_hash: str
    action: str
    after_hash: str
    reward: float
    done: bool
    official_success: bool

    @classmethod
    def create(
        cls,
        before: Observation,
        proposal: DecisionProposal,
        after: Observation,
        reward: float,
    ) -> "TransitionReceipt":
        body = {
            "before_hash": stable_hash(before.state),
            "native_actions_hash": stable_hash(before.native_actions),
            "proposal_hash": stable_hash(asdict(proposal)),
            "action": proposal.action,
            "after_hash": stable_hash(after.state),
            "reward": reward,
            "done": after.terminal,
            "official_success": after.official_success,
        }
        return cls(receipt_id=stable_hash(body), **body)

    def validate(self) -> bool:
        body = asdict(self)
        receipt_id = body.pop("receipt_id")
        return stable_hash(body) == receipt_id


@dataclass(frozen=True)
class DecisionStepSignature:
    proposal_count: int
    selected_ordinal: int
    post_verdict: str
    continuation_decision: str


@dataclass(frozen=True)
class MotifNode:
    node_id: str
    transition_receipt_ids: tuple[str, ...]
    decision_signatures: tuple[DecisionStepSignature, ...] = ()


@dataclass(frozen=True)
class MotifEdge:
    source: str
    target: str
    replay_receipt_ids: tuple[str, ...]
    untrusted_claim: str = ""


@dataclass(frozen=True)
class MotifCandidate:
    motif_id: str
    source_lineage: tuple[str, ...]
    nodes: tuple[MotifNode, ...]
    edges: tuple[MotifEdge, ...]
    status: Lifecycle = Lifecycle.CANDIDATE
    untrusted_description: str = ""


@dataclass(frozen=True)
class BindingHypothesis:
    binding_id: str
    motif_id: str
    target_claim: str
    testable_prediction: str
    adaptation_receipt_ids: tuple[str, ...]
    status: Lifecycle = Lifecycle.TARGET_PROVISIONAL


@dataclass(frozen=True)
class ConditionOutcome:
    condition: str
    initial_state_hash: str
    prefix_hash: str
    policy_hash: str
    budget_hash: str
    official_success: bool
    official_score: float


@dataclass(frozen=True)
class TransferReport:
    status: Lifecycle
    reason: str
    outcomes: tuple[ConditionOutcome, ...] = field(default_factory=tuple)
