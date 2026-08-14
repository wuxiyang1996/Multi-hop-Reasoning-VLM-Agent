"""Outcome-calibrated, fail-closed utility gate for transfer routes.

The gate estimates whether a paired disagreement is more likely to be a win
than a loss.  Ties are retained as exposure/coverage evidence but do not alter
the beta-binomial directional posterior.  A route is selected only when the
one-sided posterior lower bound exceeds the neutral probability 0.5 and every
target-native structural applicability predicate is valid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

from scipy.stats import beta as beta_distribution


Decision = Literal["SELECT_SKILL", "ABSTAIN"]


@dataclass(frozen=True)
class PairedOutcome:
    """A target task result observed only after both matched arms finish."""

    transfer_success: bool
    target_success: bool

    @property
    def utility(self) -> int:
        return int(self.transfer_success) - int(self.target_success)


@dataclass(frozen=True)
class UtilityState:
    wins: int = 0
    losses: int = 0
    ties: int = 0

    @property
    def exposures(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def discordant(self) -> int:
        return self.wins + self.losses


@dataclass(frozen=True)
class ApplicabilityReceipt:
    """Outcome-blind exact-route predicates supplied before target execution."""

    route_registered: bool
    target_interface_match: bool
    source_artifact_valid: bool
    target_grounder_valid: bool
    target_executor_valid: bool

    @property
    def eligible(self) -> bool:
        return all(asdict(self).values())


@dataclass(frozen=True)
class UtilityDecision:
    decision: Decision
    reason: str
    state: UtilityState
    posterior_mean_win_probability: float
    posterior_lower_win_probability: float
    posterior_upper_loss_probability: float
    observed_disagreement_rate: float
    applicability: ApplicabilityReceipt


class OnlineTransferUtilityGate:
    """Beta-binomial directional utility posterior with exact abstention.

    The uniform beta prior and credibility level are protocol parameters, not
    fitted thresholds.  Updates are chronological and accept only completed
    paired task outcomes.  The current task's outcome can never affect its own
    decision.
    """

    def __init__(
        self, *, prior_win: float = 1.0, prior_loss: float = 1.0,
        credibility: float = 0.95, neutral_probability: float = 0.5,
        state: UtilityState | None = None,
    ) -> None:
        if prior_win <= 0 or prior_loss <= 0:
            raise ValueError("beta prior parameters must be positive")
        if not 0.5 < credibility < 1.0:
            raise ValueError("credibility must be in (0.5,1)")
        if not 0.0 < neutral_probability < 1.0:
            raise ValueError("neutral probability must be in (0,1)")
        self.prior_win = float(prior_win)
        self.prior_loss = float(prior_loss)
        self.credibility = float(credibility)
        self.neutral_probability = float(neutral_probability)
        self.state = state or UtilityState()

    def update(self, outcome: PairedOutcome) -> UtilityState:
        utility = outcome.utility
        self.state = UtilityState(
            wins=self.state.wins + int(utility > 0),
            losses=self.state.losses + int(utility < 0),
            ties=self.state.ties + int(utility == 0),
        )
        return self.state

    def update_many(self, outcomes: Iterable[PairedOutcome]) -> UtilityState:
        for outcome in outcomes:
            self.update(outcome)
        return self.state

    def decision(self, applicability: ApplicabilityReceipt) -> UtilityDecision:
        alpha = self.prior_win + self.state.wins
        beta = self.prior_loss + self.state.losses
        tail = 1.0 - self.credibility
        lower = float(beta_distribution.ppf(tail, alpha, beta))
        mean = alpha / (alpha + beta)
        # Loss probability is one minus win probability on a discordant pair.
        upper_loss = 1.0 - lower
        disagreement_rate = (
            self.state.discordant / self.state.exposures
            if self.state.exposures else 0.0
        )
        if not applicability.eligible:
            decision: Decision = "ABSTAIN"
            reason = "STRUCTURAL_APPLICABILITY_FAILED"
        elif lower <= self.neutral_probability:
            decision = "ABSTAIN"
            reason = "POSITIVE_DIRECTION_NOT_CALIBRATED"
        else:
            decision = "SELECT_SKILL"
            reason = "POSTERIOR_DIRECTIONAL_UTILITY_LOWER_BOUND_POSITIVE"
        return UtilityDecision(
            decision=decision,
            reason=reason,
            state=self.state,
            posterior_mean_win_probability=mean,
            posterior_lower_win_probability=lower,
            posterior_upper_loss_probability=upper_loss,
            observed_disagreement_rate=disagreement_rate,
            applicability=applicability,
        )


__all__ = [
    "ApplicabilityReceipt",
    "OnlineTransferUtilityGate",
    "PairedOutcome",
    "UtilityDecision",
    "UtilityState",
]
