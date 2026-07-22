from __future__ import annotations

from typing import Protocol, Sequence

from .contracts import (
    Advisory,
    BindingHypothesis,
    DecisionProposal,
    DecisionCycleRecord,
    MotifCandidate,
    Observation,
    ReplayForkReceipt,
    TransitionReceipt,
)


class MotifHarnessAgent(Protocol):
    """Proposal-only role. Its return schemas deliberately contain no action field."""

    def propose_motifs(
        self,
        records: Sequence[DecisionCycleRecord],
        replay_receipts: Sequence[ReplayForkReceipt],
    ) -> Sequence[MotifCandidate]: ...

    def initialize_binding(
        self,
        motif: MotifCandidate,
        adaptation_records: Sequence[DecisionCycleRecord],
    ) -> BindingHypothesis | None: ...

    def review(
        self,
        proposal: DecisionProposal,
        observation: Observation,
        binding: BindingHypothesis | None,
        history: Sequence[TransitionReceipt],
    ) -> Advisory: ...
