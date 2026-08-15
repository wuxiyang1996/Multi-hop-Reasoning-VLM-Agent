"""Fail-closed target binding for the source-learned Sokoban search automaton.

The transferable object is deliberately small: an event-to-control-action map.
Target domains own every semantic predicate and every native action.  In
particular, a target may emit ``ACTIVE_CANDIDATE_VERIFIED`` only from a
target-native terminal/commit predicate, not merely because an observation
changed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .sokoban_search_automaton_v16 import (
    ACTIONS,
    EVENTS,
    REFUTED,
    UNBOUND,
    VERIFIED,
)


SOURCE_SCHEMA = "sokoban-search-automaton-artifact-v16"
SOURCE_STATUS = "SOURCE_SEARCH_AUTOMATON_FROZEN"

TARGET_DOMAINS = frozenset({"webshop", "alfworld", "discoveryworld", "tirbench"})
OUTCOME_REFUTED = "REFUTED"
OUTCOME_NONTERMINAL_EFFECT = "NONTERMINAL_EFFECT"
OUTCOME_TERMINAL_VERIFIED = "TERMINAL_VERIFIED"
OUTCOMES = frozenset({
    OUTCOME_REFUTED,
    OUTCOME_NONTERMINAL_EFFECT,
    OUTCOME_TERMINAL_VERIFIED,
})

_FORBIDDEN_SOURCE_FIELDS = frozenset({
    "box_coordinate",
    "candidate_rank",
    "direction_sequence",
    "path_length",
    "sokoban_action",
    "sokoban_coordinate",
    "source_candidate_id",
    "source_path",
})


@dataclass(frozen=True)
class NativeBinding:
    """One target-owned realization of an abstract source action."""

    abstract_action: str
    native_action_id: str
    native_action: Any
    grounding_confidence: float
    target_evidence_sha256: str

    def validate(self) -> None:
        if self.abstract_action not in ACTIONS:
            raise ValueError(f"unknown abstract action: {self.abstract_action}")
        if not self.native_action_id:
            raise ValueError("target native action ID is empty")
        if not 0.0 <= float(self.grounding_confidence) <= 1.0:
            raise ValueError("grounding confidence must be in [0, 1]")
        if len(self.target_evidence_sha256) != 64:
            raise ValueError("target evidence hash is malformed")
        _reject_source_native_payload(self.native_action)


@dataclass(frozen=True)
class TargetEvent:
    """Target-native grounding of one source automaton event."""

    domain: str
    episode_id: str
    decision_index: int
    event: str
    evidence_kind: str
    evidence_payload: Mapping[str, Any]
    grounding_confidence: float

    @property
    def evidence_sha256(self) -> str:
        return stable_hash({
            "domain": self.domain,
            "episode_id": self.episode_id,
            "decision_index": self.decision_index,
            "event": self.event,
            "evidence_kind": self.evidence_kind,
            "evidence_payload": dict(self.evidence_payload),
        })

    def validate(self) -> None:
        if self.domain not in TARGET_DOMAINS:
            raise ValueError(f"unsupported target domain: {self.domain}")
        if self.event not in EVENTS:
            raise ValueError(f"unknown target event: {self.event}")
        if not self.episode_id or self.decision_index < 0:
            raise ValueError("invalid target event identity")
        if not self.evidence_kind:
            raise ValueError("target event lacks an evidence kind")
        if not 0.0 <= float(self.grounding_confidence) <= 1.0:
            raise ValueError("event grounding confidence must be in [0, 1]")
        _reject_source_native_payload(self.evidence_payload)


@dataclass(frozen=True)
class RoutedDecision:
    domain: str
    episode_id: str
    decision_index: int
    target_event: str
    source_action: str | None
    native_action_id: str | None
    native_action: Any | None
    admitted: bool
    reason: str
    source_artifact_sha256: str
    target_evidence_sha256: str
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        event: TargetEvent,
        source_action: str | None,
        binding: NativeBinding | None,
        admitted: bool,
        reason: str,
        source_artifact_sha256: str,
    ) -> "RoutedDecision":
        body = {
            "domain": event.domain,
            "episode_id": event.episode_id,
            "decision_index": event.decision_index,
            "target_event": event.event,
            "source_action": source_action,
            "native_action_id": (
                binding.native_action_id if binding is not None else None
            ),
            "native_action": binding.native_action if binding is not None else None,
            "admitted": bool(admitted),
            "reason": reason,
            "source_artifact_sha256": source_artifact_sha256,
            "target_evidence_sha256": event.evidence_sha256,
        }
        return cls(**body, receipt_sha256=stable_hash(body))

    def validate(self) -> bool:
        body = asdict(self)
        claimed = body.pop("receipt_sha256")
        return claimed == stable_hash(body)


class SourceSearchAutomaton:
    """Validated source policy with no access to target semantics."""

    def __init__(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_sha256: str | None = None,
        minimum_grounding_confidence: float = 0.5,
    ) -> None:
        body = dict(artifact)
        claimed = str(body.pop("artifact_sha256", ""))
        if not claimed or stable_hash(body) != claimed:
            raise ValueError("source search-automaton artifact self-hash mismatch")
        if expected_sha256 is not None and claimed != expected_sha256:
            raise ValueError("source search-automaton artifact hash mismatch")
        if artifact.get("schema_version") != SOURCE_SCHEMA:
            raise ValueError("wrong source search-automaton schema")
        if artifact.get("status") != SOURCE_STATUS:
            raise ValueError("source search automaton is not frozen")
        if artifact.get("target_authorized") is not True:
            raise ValueError("source search automaton is not target-authorized")
        policy = {
            str(event): str(action)
            for event, action in (artifact.get("learned_policy") or {}).items()
        }
        if set(policy) != set(EVENTS) or set(policy.values()) != set(ACTIONS):
            raise ValueError("source policy must cover each event and action exactly once")
        self.artifact_sha256 = claimed
        self.policy = policy
        self.minimum_grounding_confidence = float(minimum_grounding_confidence)
        if not 0.0 <= self.minimum_grounding_confidence <= 1.0:
            raise ValueError("minimum grounding confidence must be in [0, 1]")

    def route(
        self,
        event: TargetEvent,
        bindings: Mapping[str, NativeBinding],
    ) -> RoutedDecision:
        event.validate()
        source_action = self.policy.get(event.event)
        if event.grounding_confidence < self.minimum_grounding_confidence:
            return RoutedDecision.create(
                event=event,
                source_action=source_action,
                binding=None,
                admitted=False,
                reason="ABSTAIN_LOW_EVENT_GROUNDING_CONFIDENCE",
                source_artifact_sha256=self.artifact_sha256,
            )
        binding = bindings.get(str(source_action))
        if binding is None:
            return RoutedDecision.create(
                event=event,
                source_action=source_action,
                binding=None,
                admitted=False,
                reason="ABSTAIN_ABSTRACT_ACTION_NOT_TARGET_REALIZABLE",
                source_artifact_sha256=self.artifact_sha256,
            )
        binding.validate()
        if binding.abstract_action != source_action:
            raise ValueError("target binding key/action mismatch")
        if binding.target_evidence_sha256 != event.evidence_sha256:
            raise ValueError("target action was grounded from different event evidence")
        if binding.grounding_confidence < self.minimum_grounding_confidence:
            return RoutedDecision.create(
                event=event,
                source_action=source_action,
                binding=None,
                admitted=False,
                reason="ABSTAIN_LOW_ACTION_GROUNDING_CONFIDENCE",
                source_artifact_sha256=self.artifact_sha256,
            )
        return RoutedDecision.create(
            event=event,
            source_action=source_action,
            binding=binding,
            admitted=True,
            reason="SOURCE_EVENT_ROUTE_TARGET_NATIVE_REALIZATION",
            source_artifact_sha256=self.artifact_sha256,
        )


class AttemptLedger:
    """Target-native candidate ledger; candidate order never comes from source."""

    def __init__(self) -> None:
        self.scope_id: str | None = None
        self.active_candidate_id: str | None = None
        self.tried: set[str] = set()
        self.refuted: set[str] = set()

    def begin_scope(self, scope_id: str) -> None:
        if not scope_id:
            raise ValueError("target scope ID is empty")
        if scope_id != self.scope_id:
            self.scope_id = scope_id
            self.active_candidate_id = None
            self.tried.clear()
            self.refuted.clear()

    def next_untried(self, ranked_candidate_ids: Sequence[str]) -> str | None:
        if self.active_candidate_id is not None:
            raise RuntimeError("cannot explore while a target candidate is active")
        for raw_candidate_id in ranked_candidate_ids:
            candidate_id = str(raw_candidate_id)
            if candidate_id and candidate_id not in self.tried:
                self.tried.add(candidate_id)
                self.active_candidate_id = candidate_id
                return candidate_id
        return None

    def observe(self, candidate_id: str, outcome: str) -> str | None:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown target candidate outcome: {outcome}")
        if self.active_candidate_id != candidate_id:
            raise ValueError("outcome does not refer to the active target candidate")
        self.active_candidate_id = None
        if outcome == OUTCOME_REFUTED:
            self.refuted.add(candidate_id)
            return REFUTED
        if outcome == OUTCOME_TERMINAL_VERIFIED:
            return VERIFIED
        return None

    def unbound_event(self, ranked_candidate_ids: Sequence[str]) -> str | None:
        if self.active_candidate_id is not None:
            return None
        return UNBOUND if any(
            str(candidate_id) not in self.tried
            for candidate_id in ranked_candidate_ids
        ) else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "active_candidate_id": self.active_candidate_id,
            "tried_target_candidate_ids": sorted(self.tried),
            "refuted_target_candidate_ids": sorted(self.refuted),
        }


def ground_target_event(
    *,
    domain: str,
    episode_id: str,
    decision_index: int,
    untried_candidate_available: bool,
    active_candidate_refuted: bool,
    terminal_commit_verified: bool,
    evidence_kind: str,
    evidence_payload: Mapping[str, Any],
    grounding_confidence: float,
) -> TargetEvent | None:
    """Ground exactly one mutually-exclusive event or fail closed.

    ``terminal_commit_verified`` must mean that the target-native predicate for
    a terminal or irreversible commit is satisfied.  Mere state change is not
    sufficient and belongs to a new target scope instead.
    """

    flags = (
        bool(untried_candidate_available),
        bool(active_candidate_refuted),
        bool(terminal_commit_verified),
    )
    if sum(flags) == 0:
        return None
    if sum(flags) != 1:
        raise ValueError("target event predicates are not mutually exclusive")
    event = (
        UNBOUND if flags[0]
        else REFUTED if flags[1]
        else VERIFIED
    )
    grounded = TargetEvent(
        domain=domain,
        episode_id=episode_id,
        decision_index=decision_index,
        event=event,
        evidence_kind=evidence_kind,
        evidence_payload=dict(evidence_payload),
        grounding_confidence=float(grounding_confidence),
    )
    grounded.validate()
    return grounded


def bind_native_action(
    event: TargetEvent,
    *,
    abstract_action: str,
    native_action_id: str,
    native_action: Any,
    grounding_confidence: float,
) -> NativeBinding:
    binding = NativeBinding(
        abstract_action=abstract_action,
        native_action_id=native_action_id,
        native_action=native_action,
        grounding_confidence=float(grounding_confidence),
        target_evidence_sha256=event.evidence_sha256,
    )
    binding.validate()
    return binding


def _reject_source_native_payload(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if key in _FORBIDDEN_SOURCE_FIELDS:
                raise ValueError(f"source-native field crossed boundary: {path}.{key}")
            _reject_source_native_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_source_native_payload(child, path=f"{path}[{index}]")


__all__ = [
    "AttemptLedger",
    "NativeBinding",
    "OUTCOME_NONTERMINAL_EFFECT",
    "OUTCOME_REFUTED",
    "OUTCOME_TERMINAL_VERIFIED",
    "RoutedDecision",
    "SourceSearchAutomaton",
    "TARGET_DOMAINS",
    "TargetEvent",
    "bind_native_action",
    "ground_target_event",
]
