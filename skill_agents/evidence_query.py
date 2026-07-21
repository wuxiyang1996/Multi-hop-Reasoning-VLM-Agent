"""Content-addressed source evidence access for untrusted Agents.

Agents may request exact transition receipts by immutable ID.  This module has
no semantic search, skill labels, similarity scores, or source/target mapping.
Every response is hash-addressed so a proposal can cite precisely what the
Agent was shown.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from skill_bank.trace_program_ir import TraceProgram


def _digest(value: Any) -> str:
    raw = value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceQuery:
    query_id: str
    program_id: str
    program_hash: str
    transition_ids: Sequence[str]

    def validate(self) -> None:
        if not self.query_id or not self.transition_ids:
            raise ValueError("evidence query must have an ID and transition IDs")
        if len(self.transition_ids) != len(set(self.transition_ids)):
            raise ValueError("evidence query transition IDs must be unique")


@dataclass(frozen=True)
class EvidenceResponse:
    query_id: str
    program_id: str
    program_hash: str
    transitions: Sequence[Mapping[str, Any]]
    response_sha256: str

    def payload_without_hash(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "program_id": self.program_id,
            "program_hash": self.program_hash,
            "transitions": [dict(item) for item in self.transitions],
        }

    def validate_hash(self) -> None:
        if _digest(self.payload_without_hash()) != self.response_sha256:
            raise ValueError("evidence response hash mismatch")


class ContentAddressedEvidenceSession:
    """Read-only exact-ID query interface over one frozen TraceProgram."""

    def __init__(
        self,
        program: TraceProgram,
        *,
        native_evidence_by_transition_id: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        program.validate_structure()
        self.program = program
        self.program_hash = program.content_hash()
        native = dict(native_evidence_by_transition_id or {})
        self._by_id = {}
        for item in program.transitions:
            row = asdict(item)
            if item.transition_id in native:
                row["native_evidence"] = dict(native[item.transition_id])
            self._by_id[item.transition_id] = row
        self.query_log: list[EvidenceResponse] = []

    @classmethod
    def from_source_episode(
        cls, program: TraceProgram, source_path: str | Path,
    ) -> "ContentAddressedEvidenceSession":
        path = Path(source_path)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != program.source_file_sha256:
            raise ValueError("source episode hash mismatch")
        payload = json.loads(raw)
        experiences = list(payload.get("experiences") or ())
        if len(experiences) != len(program.transitions):
            raise ValueError("source episode transition count mismatch")
        native: Dict[str, Mapping[str, Any]] = {}
        for ordinal, (step, receipt) in enumerate(zip(experiences, program.transitions)):
            state = step.get("raw_state", step.get("state", ""))
            next_state = step.get("raw_next_state", step.get("next_state", ""))
            actions = list(step.get("available_actions") or ())
            checks = (
                int(step.get("idx", ordinal)) == receipt.step_index,
                _digest(state) == receipt.state_sha256,
                _digest(next_state) == receipt.next_state_sha256,
                _digest(actions) == receipt.available_actions_sha256,
                str(step.get("action") or "").strip() == receipt.action,
            )
            if not all(checks):
                raise ValueError(f"source native evidence mismatch at transition {ordinal}")
            native[receipt.transition_id] = {
                "state": state,
                "next_state": next_state,
                "available_actions": actions,
                "reward": float(step.get("reward") or 0.0),
                "done": bool(step.get("done")),
            }
        return cls(program, native_evidence_by_transition_id=native)

    def query(self, request: EvidenceQuery) -> EvidenceResponse:
        request.validate()
        if request.program_id != self.program.program_id:
            raise ValueError("query program ID mismatch")
        if request.program_hash != self.program_hash:
            raise ValueError("query program hash mismatch")
        unknown = [item for item in request.transition_ids if item not in self._by_id]
        if unknown:
            raise ValueError(f"query cites unknown transition IDs: {unknown}")
        payload = {
            "query_id": request.query_id,
            "program_id": request.program_id,
            "program_hash": request.program_hash,
            # Preserve requested order.  The Agent cannot silently receive a
            # reordered or semantically filtered view.
            "transitions": [self._by_id[item] for item in request.transition_ids],
        }
        response = EvidenceResponse(**payload, response_sha256=_digest(payload))
        self.query_log.append(response)
        return response


__all__ = [
    "ContentAddressedEvidenceSession",
    "EvidenceQuery",
    "EvidenceResponse",
]
