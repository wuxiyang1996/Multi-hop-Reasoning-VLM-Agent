"""Typed, evidence-carrying intermediate representation for skill transfer.

The legacy skill banks in this repository contain useful proposals, but many
records marked ``VALIDATED`` have no executable receipts.  This module keeps
the new transfer path deliberately separate: a program is source-verified only
when every executable step cites immutable source-transition evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Sequence


class ProgramStatus(str, Enum):
    LEGACY_PROPOSAL = "LEGACY_PROPOSAL"
    SOURCE_VERIFIED = "SOURCE_VERIFIED"
    SUSPENDED = "SUSPENDED"


class Operator(str, Enum):
    OBSERVE = "OBSERVE"
    DECIDE = "DECIDE"
    COMMIT = "COMMIT"
    VERIFY = "VERIFY"
    RECOVER = "RECOVER"


class EffectKind(str, Enum):
    ADD = "ADD"
    DELETE = "DELETE"
    CHANGE = "CHANGE"
    EVENT = "EVENT"


@dataclass(frozen=True)
class SourceStepKey:
    game: str
    episode_id: str
    step_index: int
    provider_or_run: str = "unknown"

    def stable_id(self) -> str:
        raw = f"{self.game}\0{self.episode_id}\0{self.step_index}\0{self.provider_or_run}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TransitionEvidenceRef:
    key: SourceStepKey
    source_file_sha256: str
    state_sha256: str
    next_state_sha256: str
    action: str
    reward: float
    done: bool

    def validate(self) -> None:
        for label, value in (
            ("source_file_sha256", self.source_file_sha256),
            ("state_sha256", self.state_sha256),
            ("next_state_sha256", self.next_state_sha256),
        ):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{label} must be a lowercase sha256 digest")
        if not self.action:
            raise ValueError("transition evidence must name the executed action")


@dataclass(frozen=True)
class StateFact:
    predicate: str
    value_type: str = "bool"
    value: Any = True


@dataclass(frozen=True)
class ActionSchema:
    name: str
    argument_types: Mapping[str, str] = field(default_factory=dict)
    observed_source_actions: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class TypedEffect:
    kind: EffectKind
    predicate: str
    value_type: str
    evidence_step_ids: Sequence[str]


@dataclass(frozen=True)
class ProgramStep:
    step_id: str
    operator: Operator
    action: ActionSchema | None = None
    preconditions: Sequence[StateFact] = field(default_factory=tuple)
    effects: Sequence[TypedEffect] = field(default_factory=tuple)
    evidence_step_ids: Sequence[str] = field(default_factory=tuple)


@dataclass
class CanonicalSkillProgram:
    program_id: str
    name: str
    source_skill_ids: List[str]
    steps: List[ProgramStep]
    evidence: List[TransitionEvidenceRef]
    source_games: List[str]
    status: ProgramStatus = ProgramStatus.SOURCE_VERIFIED
    schema_version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.program_id or not self.name:
            raise ValueError("program_id and name are required")
        if not self.source_skill_ids or not self.source_games:
            raise ValueError("source skill ids and source games are required")
        if not self.steps:
            raise ValueError("a program must contain at least one step")
        for item in self.evidence:
            item.validate()
        known = {item.key.stable_id() for item in self.evidence}
        cited: set[str] = set()
        for step in self.steps:
            cited.update(step.evidence_step_ids)
            for effect in step.effects:
                cited.update(effect.evidence_step_ids)
        if self.status == ProgramStatus.SOURCE_VERIFIED:
            if not known:
                raise ValueError("SOURCE_VERIFIED programs require transition evidence")
            missing = cited - known
            if missing:
                raise ValueError(f"program cites missing evidence: {sorted(missing)}")
            if any(not step.evidence_step_ids for step in self.steps):
                raise ValueError("every SOURCE_VERIFIED step needs evidence")
        elif self.status == ProgramStatus.LEGACY_PROPOSAL and self.evidence:
            raise ValueError("LEGACY_PROPOSAL must not masquerade as verified evidence")

    def to_dict(self) -> Dict[str, Any]:
        payload = _jsonable(asdict(self))
        payload["program_hash"] = self.content_hash()
        return payload

    def content_hash(self) -> str:
        payload = _jsonable(asdict(self))
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def evidence_by_id(
    evidence: Iterable[TransitionEvidenceRef],
) -> Dict[str, TransitionEvidenceRef]:
    return {item.key.stable_id(): item for item in evidence}


def canonical_program_from_dict(payload: Mapping[str, Any]) -> CanonicalSkillProgram:
    """Load the stable JSON representation emitted by :meth:`to_dict`."""
    evidence = []
    for item in payload.get("evidence", []):
        key = item["key"]
        evidence.append(
            TransitionEvidenceRef(
                key=SourceStepKey(
                    game=str(key["game"]),
                    episode_id=str(key["episode_id"]),
                    step_index=int(key["step_index"]),
                    provider_or_run=str(key.get("provider_or_run") or "unknown"),
                ),
                source_file_sha256=str(item["source_file_sha256"]),
                state_sha256=str(item["state_sha256"]),
                next_state_sha256=str(item["next_state_sha256"]),
                action=str(item["action"]),
                reward=float(item.get("reward") or 0.0),
                done=bool(item.get("done")),
            )
        )
    steps = []
    for item in payload.get("steps", []):
        action_raw = item.get("action")
        action = None
        if isinstance(action_raw, Mapping):
            action = ActionSchema(
                name=str(action_raw["name"]),
                argument_types=dict(action_raw.get("argument_types") or {}),
                observed_source_actions=list(action_raw.get("observed_source_actions") or []),
            )
        steps.append(
            ProgramStep(
                step_id=str(item["step_id"]),
                operator=Operator(item["operator"]),
                action=action,
                preconditions=[StateFact(**fact) for fact in item.get("preconditions", [])],
                effects=[
                    TypedEffect(
                        kind=EffectKind(effect["kind"]),
                        predicate=str(effect["predicate"]),
                        value_type=str(effect["value_type"]),
                        evidence_step_ids=list(effect.get("evidence_step_ids") or []),
                    )
                    for effect in item.get("effects", [])
                ],
                evidence_step_ids=list(item.get("evidence_step_ids") or []),
            )
        )
    program = CanonicalSkillProgram(
        program_id=str(payload["program_id"]),
        name=str(payload["name"]),
        source_skill_ids=[str(item) for item in payload.get("source_skill_ids", [])],
        steps=steps,
        evidence=evidence,
        source_games=[str(item) for item in payload.get("source_games", [])],
        status=ProgramStatus(payload.get("status", ProgramStatus.SOURCE_VERIFIED.value)),
        schema_version=int(payload.get("schema_version", 1)),
        metadata=dict(payload.get("metadata") or {}),
    )
    expected_hash = payload.get("program_hash")
    if expected_hash and expected_hash != program.content_hash():
        raise ValueError(f"program hash mismatch for {program.program_id}")
    program.validate()
    return program


__all__ = [
    "ActionSchema",
    "CanonicalSkillProgram",
    "EffectKind",
    "Operator",
    "ProgramStatus",
    "ProgramStep",
    "SourceStepKey",
    "StateFact",
    "TransitionEvidenceRef",
    "TypedEffect",
    "evidence_by_id",
    "canonical_program_from_dict",
]
