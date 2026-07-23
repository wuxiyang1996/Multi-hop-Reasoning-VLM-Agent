"""Closed-schema, Agent-native proposal and post-transition verdict receipts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


_JSON_OBJECT = re.compile(r"\A\s*\{.*\}\s*\Z", re.DOTALL)


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class AgentActionProposal:
    proposal_id: str
    action_number: int
    predicted_observable_delta: str
    rationale: str


@dataclass(frozen=True)
class AgentActionProposalSet:
    proposals: Sequence[AgentActionProposal]
    selected_proposal_id: str | None
    decision: str

    def validate(self, *, n_native_actions: int) -> None:
        if self.decision not in {"EXECUTE", "ABSTAIN"}:
            raise ValueError("INVALID_PROPOSAL_DECISION")
        if not 1 <= len(self.proposals) <= 3:
            raise ValueError("PROPOSAL_COUNT_OUT_OF_RANGE")
        ids = [item.proposal_id for item in self.proposals]
        if any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("PROPOSAL_IDS_NOT_UNIQUE")
        for item in self.proposals:
            if not 1 <= item.action_number <= n_native_actions:
                raise ValueError("PROPOSAL_ACTION_OUTSIDE_NATIVE_LIST")
            if not item.predicted_observable_delta or not item.rationale:
                raise ValueError("EMPTY_PROPOSAL_CLAIM")
        if self.decision == "EXECUTE" and self.selected_proposal_id not in ids:
            raise ValueError("SELECTED_PROPOSAL_UNKNOWN")
        if self.decision == "ABSTAIN" and self.selected_proposal_id is not None:
            raise ValueError("ABSTENTION_SELECTS_PROPOSAL")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "proposals": [asdict(item) for item in self.proposals],
            "selected_proposal_id": self.selected_proposal_id,
            "decision": self.decision,
        }

    def content_hash(self) -> str:
        return _hash(self.to_dict())

    def selected(self) -> AgentActionProposal | None:
        return next(
            (item for item in self.proposals if item.proposal_id == self.selected_proposal_id),
            None,
        )


@dataclass(frozen=True)
class AgentPostTransitionVerdict:
    proposal_id: str | None
    verdict: str
    decision: str
    evidence_claim: str

    def validate(self, *, expected_proposal_id: str | None) -> None:
        if self.verdict not in {"SUPPORTED", "REFUTED", "INCONCLUSIVE"}:
            raise ValueError("INVALID_POST_TRANSITION_VERDICT")
        if self.decision not in {"CONTINUE", "REPLAN", "ABSTAIN"}:
            raise ValueError("INVALID_POST_TRANSITION_DECISION")
        if self.proposal_id != expected_proposal_id:
            raise ValueError("POST_VERDICT_PROPOSAL_ID_MISMATCH")
        if not self.evidence_claim:
            raise ValueError("EMPTY_POST_TRANSITION_EVIDENCE_CLAIM")

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        return _hash(self.to_dict())


def parse_agent_action_proposal_set(
    raw: str, *, n_native_actions: int,
) -> AgentActionProposalSet:
    if _JSON_OBJECT.fullmatch(raw or "") is None:
        raise ValueError("PROPOSAL_NOT_EXACT_JSON_OBJECT")
    payload = json.loads(raw)
    if set(payload) != {"proposals", "selected_proposal_id", "decision"}:
        raise ValueError("WRONG_PROPOSAL_TOP_LEVEL_KEYS")
    if (
        not isinstance(payload["proposals"], list)
        or not isinstance(payload["decision"], str)
        or not (
            payload["selected_proposal_id"] is None
            or isinstance(payload["selected_proposal_id"], str)
        )
    ):
        raise ValueError("WRONG_PROPOSAL_FIELD_TYPES")
    proposals = []
    for row in payload["proposals"]:
        if not isinstance(row, Mapping):
            raise ValueError("ACTION_PROPOSAL_NOT_OBJECT")
        if set(row) != {
            "proposal_id", "action_number", "predicted_observable_delta", "rationale",
        }:
            raise ValueError("WRONG_ACTION_PROPOSAL_KEYS")
        if (
            not isinstance(row["proposal_id"], str)
            or type(row["action_number"]) is not int
            or not isinstance(row["predicted_observable_delta"], str)
            or not isinstance(row["rationale"], str)
        ):
            raise ValueError("WRONG_ACTION_PROPOSAL_FIELD_TYPES")
        proposals.append(AgentActionProposal(
            proposal_id=str(row["proposal_id"]),
            action_number=int(row["action_number"]),
            predicted_observable_delta=str(row["predicted_observable_delta"]),
            rationale=str(row["rationale"]),
        ))
    result = AgentActionProposalSet(
        proposals=tuple(proposals),
        selected_proposal_id=(
            str(payload["selected_proposal_id"])
            if payload["selected_proposal_id"] is not None else None
        ),
        decision=str(payload["decision"]),
    )
    result.validate(n_native_actions=n_native_actions)
    return result


def parse_agent_post_transition_verdict(
    raw: str, *, expected_proposal_id: str | None,
) -> AgentPostTransitionVerdict:
    if _JSON_OBJECT.fullmatch(raw or "") is None:
        raise ValueError("VERDICT_NOT_EXACT_JSON_OBJECT")
    payload = json.loads(raw)
    if set(payload) != {"proposal_id", "verdict", "decision", "evidence_claim"}:
        raise ValueError("WRONG_VERDICT_TOP_LEVEL_KEYS")
    if (
        not (payload["proposal_id"] is None or isinstance(payload["proposal_id"], str))
        or not isinstance(payload["verdict"], str)
        or not isinstance(payload["decision"], str)
        or not isinstance(payload["evidence_claim"], str)
    ):
        raise ValueError("WRONG_VERDICT_FIELD_TYPES")
    result = AgentPostTransitionVerdict(
        proposal_id=(str(payload["proposal_id"]) if payload["proposal_id"] is not None else None),
        verdict=str(payload["verdict"]), decision=str(payload["decision"]),
        evidence_claim=str(payload["evidence_claim"]),
    )
    result.validate(expected_proposal_id=expected_proposal_id)
    return result


__all__ = [
    "AgentActionProposal", "AgentActionProposalSet", "AgentPostTransitionVerdict",
    "parse_agent_action_proposal_set", "parse_agent_post_transition_verdict",
]
