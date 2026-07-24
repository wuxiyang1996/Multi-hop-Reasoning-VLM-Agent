from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .contracts import AdvisoryVerdict, ContinuationDecision
from .runtime import EpisodeResult


@dataclass(frozen=True)
class AdaptationMetrics:
    steps: int
    official_success: bool
    official_score: float
    repeated_actions: int
    no_observable_progress: int
    advisory_replans: int
    decision_replans: int


def measure_episode(result: EpisodeResult) -> AdaptationMetrics:
    actions = [record.proposal_set.selected.action for record in result.records]
    action_counts = Counter(actions)
    repeated = sum(max(0, count - 1) for count in action_counts.values())
    no_progress = sum(
        record.transition.before_hash == record.transition.after_hash for record in result.records
    )
    return AdaptationMetrics(
        steps=len(result.records),
        official_success=result.final_observation.official_success,
        official_score=result.final_observation.score,
        repeated_actions=repeated,
        no_observable_progress=no_progress,
        advisory_replans=sum(
            record.advisory.verdict == AdvisoryVerdict.REPLAN for record in result.records
        ),
        decision_replans=sum(
            record.assessment.continuation == ContinuationDecision.REPLAN
            for record in result.records
        ),
    )
