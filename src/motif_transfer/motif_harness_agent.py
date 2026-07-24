from __future__ import annotations

from typing import Protocol, Sequence

from .contracts import (
    Advisory,
    BindingHypothesis,
    BindingEvidence,
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

    def initialize_binding_set_from_example(
        self,
        motif: MotifCandidate,
        adaptation_example,
        *,
        max_candidates: int = 4,
        require_alpha_invariance: bool = True,
        induction_repetitions: int = 2,
    ) -> Sequence[BindingHypothesis]: ...

    def review(
        self,
        proposal: DecisionProposal,
        observation: Observation,
        binding: BindingHypothesis | None,
        history: Sequence[TransitionReceipt],
    ) -> Advisory: ...

    def verify_transition(
        self,
        binding: BindingHypothesis,
        before: Observation,
        proposal: DecisionProposal,
        after: Observation,
        transition: TransitionReceipt,
        history: Sequence[TransitionReceipt],
    ) -> BindingEvidence: ...
