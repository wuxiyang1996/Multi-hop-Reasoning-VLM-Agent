from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import BindingEvidence, BindingHypothesis, EvidenceVerdict


@dataclass(frozen=True)
class BindingState:
    hypothesis: BindingHypothesis
    evidence: tuple[BindingEvidence, ...] = ()
    viable: bool = True


class BindingVersionSpace:
    """Keeps every binding consistent with deterministic target evidence.

    It never ranks candidates by LLM confidence or natural-language similarity.
    """

    def __init__(self, hypotheses: Iterable[BindingHypothesis]):
        rows = tuple(hypotheses)
        if not rows:
            raise ValueError("a version space requires at least one hypothesis")
        if len({row.binding_id for row in rows}) != len(rows):
            raise ValueError("duplicate binding id")
        self._states = {row.binding_id: BindingState(row) for row in rows}

    def record(self, evidence: BindingEvidence) -> BindingState:
        if evidence.binding_id not in self._states:
            raise KeyError(evidence.binding_id)
        state = self._states[evidence.binding_id]
        if evidence.verifier_id != state.hypothesis.verifier_id:
            raise ValueError("evidence was produced by a different verifier")
        viable = state.viable and evidence.verdict != EvidenceVerdict.REFUTED
        updated = BindingState(state.hypothesis, state.evidence + (evidence,), viable)
        self._states[evidence.binding_id] = updated
        return updated

    def viable(self) -> tuple[BindingHypothesis, ...]:
        return tuple(state.hypothesis for state in self._states.values() if state.viable)

    def all_states(self) -> tuple[BindingState, ...]:
        return tuple(self._states.values())
