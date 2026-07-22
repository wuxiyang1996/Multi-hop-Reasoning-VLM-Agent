from __future__ import annotations

from typing import Protocol, Sequence

from .contracts import (
    Advisory,
    BindingHypothesis,
    DecisionProposal,
    MotifCandidate,
    Observation,
    TransitionReceipt,
)


class MotifHarnessAgent(Protocol):
    """Proposal-only role. Its return schemas deliberately contain no action field."""

    def propose_motifs(self, receipts: Sequence[TransitionReceipt]) -> Sequence[MotifCandidate]: ...

    def initialize_binding(
        self,
        motif: MotifCandidate,
        adaptation_receipts: Sequence[TransitionReceipt],
    ) -> BindingHypothesis | None: ...

    def review(
        self,
        proposal: DecisionProposal,
        observation: Observation,
        binding: BindingHypothesis | None,
        history: Sequence[TransitionReceipt],
    ) -> Advisory: ...
