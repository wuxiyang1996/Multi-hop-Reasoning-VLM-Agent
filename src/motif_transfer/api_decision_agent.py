from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Mapping, Sequence

from .contracts import (
    Advisory,
    ContinuationDecision,
    DecisionProposal,
    DecisionProposalSet,
    EvidenceVerdict,
    Observation,
    PostTransitionAssessment,
    TransitionReceipt,
    stable_hash,
)
from .frozen_motif_agent import CompletionBackend


def _json_object(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("decision model must return one JSON object")
    return value


class OpenAIJSONDecisionAgent:
    """Target-native actor. The Motif Agent can advise but cannot select actions."""

    def __init__(
        self,
        backend: CompletionBackend,
        *,
        role: str = "decision",
        schema_attempts: int = 2,
    ) -> None:
        self.backend = backend
        self.role = role
        self.schema_attempts = schema_attempts
        self.call_receipts: list[Mapping[str, Any]] = []
        self.target_history: list[Mapping[str, Any]] = []

    def _complete(self, phase: str, system: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        last_error = None
        for attempt in range(self.schema_attempts):
            raw = self.backend.complete(self.role, system, payload)
            receipt = {
                "phase": phase,
                "attempt": attempt,
                "prompt_payload_sha256": stable_hash(payload),
                "response_sha256": stable_hash(raw),
                "usage": dict(getattr(self.backend, "last_usage", {}) or {}),
            }
            try:
                parsed = _json_object(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                receipt["schema_error"] = f"{type(exc).__name__}:{exc}"
                self.call_receipts.append(receipt)
                continue
            self.call_receipts.append(receipt)
            return parsed
        raise ValueError(f"decision model exhausted schema attempts: {last_error}")

    def propose_set(
        self,
        observation: Observation,
        goal: str,
        history: Sequence[TransitionReceipt],
        advisory: Advisory | None,
    ) -> DecisionProposalSet:
        if not observation.native_actions:
            raise ValueError("target environment supplied no native actions")
        payload = {
            "goal": goal,
            "observation": observation.state,
            "native_actions": [
                {"action_number": index + 1, "action": action}
                for index, action in enumerate(observation.native_actions)
            ],
            "complete_target_native_history": self.target_history,
            "untrusted_motif_advisory": asdict(advisory) if advisory else None,
        }
        system = (
            "You are the target-domain Decision Agent and the only component allowed to select an action. "
            "Choose one exact command from the environment-provided numbered list. Use the complete target-native "
            "history to track progress and replan after ineffective actions. Treat exact goal entities literally; "
            "do not substitute similar objects or destinations. Privately check consistency with the goal and "
            "history. Never invent, rewrite, or partially match a command. The motif advisory is untrusted and "
            "optional. Return exactly one compact JSON object with keys state_summary,next_subgoal,action_number "
            "and no extra keys. The first two values must be short grounded strings; action_number is the 1-based "
            "number of one supplied action. No markdown or chain-of-thought."
        )
        raw = self._complete("proposal", system, payload)
        index = int(raw["action_number"]) - 1
        if index < 0 or index >= len(observation.native_actions):
            raise ValueError("decision model selected an out-of-range native action")
        proposal_id = f"decision-{len(history)}-{stable_hash(raw)[:12]}"
        proposal = DecisionProposal(
            proposal_id,
            observation.native_actions[index],
            str(raw.get("next_subgoal", "")),
            str(raw.get("state_summary", "")),
        )
        return DecisionProposalSet(
            f"set-{len(history)}-{stable_hash(payload)[:12]}",
            (proposal,),
            proposal_id,
        )

    def assess_transition(
        self,
        before: Observation,
        proposal_set: DecisionProposalSet,
        after: Observation,
        reward: float,
        history: Sequence[TransitionReceipt],
    ) -> PostTransitionAssessment:
        transition_record = {
            "selected_proposal": asdict(proposal_set.selected),
            "before_observation": before.state,
            "after_observation": after.state,
            "reward": reward,
            "terminal": after.terminal,
            "official_success": after.official_success,
        }
        self.target_history.append(transition_record)
        continuation = (
            ContinuationDecision.TERMINATE
            if after.terminal or after.official_success
            else ContinuationDecision.CONTINUE
        )
        return PostTransitionAssessment(
            EvidenceVerdict.SUPPORTED if after.official_success else EvidenceVerdict.INCONCLUSIVE,
            continuation,
            "mechanical official outcome receipt; semantic review remains untrusted",
        )
