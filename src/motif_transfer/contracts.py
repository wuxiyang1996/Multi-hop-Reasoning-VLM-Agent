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


class EvidenceVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class ContinuationDecision(str, Enum):
    CONTINUE = "CONTINUE"
    REPLAN = "REPLAN"
    ABSTAIN = "ABSTAIN"
    TERMINATE = "TERMINATE"


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
class DecisionProposalSet:
    proposal_set_id: str
    proposals: tuple[DecisionProposal, ...]
    selected_proposal_id: str

    @property
    def selected(self) -> DecisionProposal:
        matches = [row for row in self.proposals if row.proposal_id == self.selected_proposal_id]
        if len(matches) != 1:
            raise ValueError("selected proposal must identify exactly one candidate")
        return matches[0]


@dataclass(frozen=True)
class Advisory:
    verdict: AdvisoryVerdict
    reason: str
    evidence_receipt_ids: tuple[str, ...] = ()
    current_role: str = ""
    open_hypotheses: tuple[str, ...] = ()
    information_need: str = ""
    expected_transition: str = ""
    failure_route: str = ""
    termination_test: str = ""


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
class SourceTransitionReceipt:
    """Receipt for an unchanged source policy action; no proposal is fabricated."""

    receipt_id: str
    episode_id: str
    step: int
    before_hash: str
    native_actions_hash: str
    selected_skill_hash: str | None
    action_response_hash: str
    action: str
    action_origin: str
    policy_adapter: str
    after_hash: str
    reward: float
    done: bool
    official_success: bool

    @classmethod
    def create(
        cls,
        before: Observation,
        *,
        episode_id: str,
        step: int,
        selected_skill_hash: str | None,
        action_response_hash: str,
        action: str,
        action_origin: str,
        policy_adapter: str,
        after: Observation,
        reward: float,
    ) -> "SourceTransitionReceipt":
        body = {
            "episode_id": episode_id,
            "step": step,
            "before_hash": stable_hash(before.state),
            "native_actions_hash": stable_hash(before.native_actions),
            "selected_skill_hash": selected_skill_hash,
            "action_response_hash": action_response_hash,
            "action": action,
            "action_origin": action_origin,
            "policy_adapter": policy_adapter,
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
class SourcePolicyStepRecord:
    episode_id: str
    step: int
    before: Observation
    selected_skill_id: str | None
    selected_skill_hash: str | None
    action_reasoning: str
    action_response_hash: str
    action: str
    action_origin: str
    policy_adapter: str
    after: Observation
    reward: float
    transition: SourceTransitionReceipt

    def validate(self) -> bool:
        expected = SourceTransitionReceipt.create(
            self.before,
            episode_id=self.episode_id,
            step=self.step,
            selected_skill_hash=self.selected_skill_hash,
            action_response_hash=self.action_response_hash,
            action=self.action,
            action_origin=self.action_origin,
            policy_adapter=self.policy_adapter,
            after=self.after,
            reward=self.reward,
        )
        return self.transition == expected


@dataclass(frozen=True)
class SourceSegmentReceipt:
    """Mechanically delimited span of native source-policy transitions."""

    receipt_id: str
    episode_id: str
    start_step: int
    end_step: int
    transition_receipt_ids: tuple[str, ...]
    boundary_skill_hash: str | None
    segmentation_rule: str = "MAXIMAL_RECORDED_SKILL_ID_RUN_V2"

    @classmethod
    def create(
        cls,
        episode_id: str,
        records: Sequence[SourcePolicyStepRecord],
    ) -> "SourceSegmentReceipt":
        if not records:
            raise ValueError("a source segment cannot be empty")
        if any(row.episode_id != episode_id for row in records):
            raise ValueError("source segment crosses episode boundary")
        steps = tuple(row.step for row in records)
        if steps != tuple(range(steps[0], steps[0] + len(steps))):
            raise ValueError("source segment steps must be contiguous")
        skill_ids = {row.selected_skill_id for row in records}
        if len(skill_ids) != 1:
            raise ValueError("source segment crosses selected-skill boundary")
        skill_id = next(iter(skill_ids))
        body = {
            "episode_id": episode_id,
            "start_step": steps[0],
            "end_step": steps[-1],
            "transition_receipt_ids": tuple(
                row.transition.receipt_id for row in records
            ),
            # The full guidance hash is dynamic (confidence, progress, and
            # execution hints can change while the selected skill is stable).
            # Bind the segment to exact recorded skill identity instead.
            "boundary_skill_hash": (
                stable_hash({"selected_skill_id": skill_id})
                if skill_id is not None else None
            ),
            "segmentation_rule": "MAXIMAL_RECORDED_SKILL_ID_RUN_V2",
        }
        return cls(receipt_id=stable_hash(body), **body)

    def validate(self) -> bool:
        body = asdict(self)
        receipt_id = body.pop("receipt_id")
        return stable_hash(body) == receipt_id


@dataclass(frozen=True)
class SkillRankingReceipt:
    """Strict, hash-bound output from the old segment LoRA's native task."""

    receipt_id: str
    segment_receipt_id: str
    candidate_bank_hash: str
    model_identity_hash: str
    ranking: tuple[str, ...]
    reasoning_hash: str
    raw_response_hash: str

    @classmethod
    def create(
        cls,
        *,
        segment_receipt_id: str,
        candidate_bank_hash: str,
        model_identity_hash: str,
        ranking: Sequence[str],
        reasoning: str,
        raw_response: str,
    ) -> "SkillRankingReceipt":
        body = {
            "segment_receipt_id": segment_receipt_id,
            "candidate_bank_hash": candidate_bank_hash,
            "model_identity_hash": model_identity_hash,
            "ranking": tuple(ranking),
            "reasoning_hash": stable_hash(reasoning),
            "raw_response_hash": stable_hash(raw_response),
        }
        return cls(receipt_id=stable_hash(body), **body)

    def validate(self) -> bool:
        body = asdict(self)
        receipt_id = body.pop("receipt_id")
        return stable_hash(body) == receipt_id


@dataclass(frozen=True)
class ReplayForkReceipt:
    receipt_id: str
    source_transition_id: str
    prefix_hash: str
    fork_state_hash: str
    admissible_actions_hash: str
    alternative_action: str
    alternative_after_hash: str

    @classmethod
    def create(
        cls,
        *,
        source_transition_id: str,
        prefix_hash: str,
        fork_state_hash: str,
        admissible_actions_hash: str,
        alternative_action: str,
        alternative_after_hash: str,
    ) -> "ReplayForkReceipt":
        body = {
            "source_transition_id": source_transition_id,
            "prefix_hash": prefix_hash,
            "fork_state_hash": fork_state_hash,
            "admissible_actions_hash": admissible_actions_hash,
            "alternative_action": alternative_action,
            "alternative_after_hash": alternative_after_hash,
        }
        return cls(receipt_id=stable_hash(body), **body)

    def validate(self) -> bool:
        body = asdict(self)
        receipt_id = body.pop("receipt_id")
        return stable_hash(body) == receipt_id


@dataclass(frozen=True)
class PostTransitionAssessment:
    verdict: EvidenceVerdict
    continuation: ContinuationDecision
    reason: str = ""


@dataclass(frozen=True)
class DecisionCycleReceipt:
    cycle_id: str
    proposal_set_hash: str
    selected_proposal_id: str
    transition_receipt_id: str
    assessment_hash: str

    @classmethod
    def create(
        cls,
        proposal_set: DecisionProposalSet,
        transition: TransitionReceipt,
        assessment: PostTransitionAssessment,
    ) -> "DecisionCycleReceipt":
        body = {
            "proposal_set_hash": stable_hash(asdict(proposal_set)),
            "selected_proposal_id": proposal_set.selected_proposal_id,
            "transition_receipt_id": transition.receipt_id,
            "assessment_hash": stable_hash(asdict(assessment)),
        }
        return cls(cycle_id=stable_hash(body), **body)

    def validate(self) -> bool:
        body = asdict(self)
        cycle_id = body.pop("cycle_id")
        return stable_hash(body) == cycle_id


@dataclass(frozen=True)
class DecisionCycleRecord:
    before: Observation
    proposal_set: DecisionProposalSet
    advisory: Advisory
    after: Observation
    reward: float
    transition: TransitionReceipt
    assessment: PostTransitionAssessment
    receipt: DecisionCycleReceipt

    def validate(self) -> bool:
        expected_transition = TransitionReceipt.create(
            self.before, self.proposal_set.selected, self.after, self.reward
        )
        expected_cycle = DecisionCycleReceipt.create(
            self.proposal_set, expected_transition, self.assessment
        )
        return self.transition == expected_transition and self.receipt == expected_cycle


@dataclass(frozen=True)
class DecisionStepSignature:
    proposal_count: int
    selected_ordinal: int
    post_verdict: str
    continuation_decision: str


@dataclass(frozen=True)
class SourceStepSignature:
    skill_conditioned: bool
    action_origin: str
    reward_sign: str
    terminal: bool
    # Anonymous first-occurrence class: preserves equality/change structure
    # under arbitrary skill-ID renaming without treating names as semantics.
    skill_class_ordinal: int | None = None


@dataclass(frozen=True)
class MotifNode:
    node_id: str
    transition_receipt_ids: tuple[str, ...]
    decision_signatures: tuple[DecisionStepSignature | SourceStepSignature, ...] = ()


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
    verifier_id: str = ""
    status: Lifecycle = Lifecycle.TARGET_PROVISIONAL


@dataclass(frozen=True)
class BindingEvidence:
    binding_id: str
    receipt_id: str
    verifier_id: str
    verdict: EvidenceVerdict


@dataclass(frozen=True)
class ConditionOutcome:
    condition: str
    initial_state_hash: str
    prefix_hash: str
    policy_hash: str
    budget_hash: str
    official_success: bool
    official_score: float
    pair_id: str = "pair-0"


@dataclass(frozen=True)
class TransferReport:
    status: Lifecycle
    reason: str
    outcomes: tuple[ConditionOutcome, ...] = field(default_factory=tuple)
    metrics: Mapping[str, float] = field(default_factory=dict)
