"""Source-specific applicability prior for the shared induced symbolic IR.

The anonymous operators may be structurally identical across source games.  A
real source-specific transfer claim therefore cannot use a game-name feature or
different hand-authored controllers.  This module derives the only permitted
difference from intervention outcomes already stored in each frozen source
artifact: the empirical distribution of the rank of the verified intervention.

The distribution is projected by rank quantile onto a target-native candidate
set.  This changes the order in which the shared AttemptLedger IR instantiates
its anonymous ``active := observed candidate`` state delta.  It does not change
operators, target actions, effect grounding, or success evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import floor
from typing import Any, Mapping, Sequence

from .contracts import stable_hash


def _validated_rank_counts(profile: Mapping[str, Any]) -> tuple[int, ...]:
    raw = profile.get("verified_rank_distribution")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("source profile has no verified-rank distribution")
    candidate_count = int(profile.get("candidate_count_median") or 0)
    if candidate_count < 2:
        raise ValueError("source profile must cover at least two candidates")
    counts = []
    for rank in range(candidate_count):
        value = raw.get(str(rank), 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("verified-rank counts must be nonnegative integers")
        counts.append(value)
    if sum(counts) <= 0:
        raise ValueError("source profile verified-rank distribution is empty")
    unexpected = set(map(str, raw)) - {str(rank) for rank in range(candidate_count)}
    if unexpected:
        raise ValueError(f"source profile has out-of-range ranks: {sorted(unexpected)}")
    return tuple(counts)


def projected_rank_scores(
    source_counts: Sequence[int], target_candidate_count: int,
) -> tuple[float, ...]:
    """Linearly project an empirical source rank prior by normalized rank.

    Interpolation, rather than a hand-selected source-to-target mapping, makes
    the operation well-defined for every target candidate multiplicity while
    preserving endpoints and the empirical ordering as closely as possible.
    """

    counts = tuple(int(value) for value in source_counts)
    if len(counts) < 2 or any(value < 0 for value in counts) or sum(counts) <= 0:
        raise ValueError("invalid source rank counts")
    target_count = int(target_candidate_count)
    if target_count < 2:
        raise ValueError("applicability requires at least two target candidates")
    total = float(sum(counts))
    probabilities = tuple(value / total for value in counts)
    maximum_source_rank = len(probabilities) - 1
    scores = []
    for target_rank in range(target_count):
        quantile = target_rank / (target_count - 1)
        position = quantile * maximum_source_rank
        lower = floor(position)
        upper = min(maximum_source_rank, lower + 1)
        fraction = position - lower
        scores.append(
            probabilities[lower] * (1.0 - fraction)
            + probabilities[upper] * fraction
        )
    return tuple(scores)


@dataclass(frozen=True)
class SourceApplicabilityPrior:
    source_profile_sha256: str
    source_candidate_count: int
    source_rank_counts: tuple[int, ...]
    prior_sha256: str

    @classmethod
    def from_profile(
        cls, profile: Mapping[str, Any],
    ) -> "SourceApplicabilityPrior":
        body = dict(profile)
        claimed = str(body.pop("profile_sha256", ""))
        if not claimed or stable_hash(body) != claimed:
            raise ValueError("source-only profile hash mismatch")
        counts = _validated_rank_counts(profile)
        prior_body = {
            "source_profile_sha256": claimed,
            "source_candidate_count": len(counts),
            "source_rank_counts": list(counts),
            "projection": "LINEAR_NORMALIZED_RANK_INTERPOLATION",
        }
        return cls(
            source_profile_sha256=claimed,
            source_candidate_count=len(counts),
            source_rank_counts=counts,
            prior_sha256=stable_hash(prior_body),
        )

    def rank_scores(self, target_candidate_count: int) -> tuple[float, ...]:
        return projected_rank_scores(
            self.source_rank_counts, target_candidate_count,
        )

    def trial_order(self, target_candidate_count: int) -> tuple[int, ...]:
        scores = self.rank_scores(target_candidate_count)
        return tuple(sorted(range(len(scores)), key=lambda rank: (-scores[rank], rank)))

    def applicability_receipt(
        self, *, target_candidate_ids: Sequence[str], target_grounding_sha256: str,
    ) -> dict[str, Any]:
        ids = tuple(map(str, target_candidate_ids))
        if len(ids) < 2 or len(set(ids)) != len(ids):
            body = {
                "admitted": False,
                "abstention_reason": "TARGET_CANDIDATE_SET_NOT_MULTIPLE_AND_UNIQUE",
                "source_profile_sha256": self.source_profile_sha256,
                "source_prior_sha256": self.prior_sha256,
                "target_candidate_ids": list(ids),
                "target_grounding_sha256": str(target_grounding_sha256),
                "target_outcome_read": False,
            }
        else:
            scores = self.rank_scores(len(ids))
            order = self.trial_order(len(ids))
            body = {
                "admitted": True,
                "abstention_reason": None,
                "source_profile_sha256": self.source_profile_sha256,
                "source_prior_sha256": self.prior_sha256,
                "target_candidate_ids": list(ids),
                "target_grounding_sha256": str(target_grounding_sha256),
                "projected_rank_scores": list(scores),
                "trial_order": list(order),
                "ordered_target_candidate_ids": [ids[rank] for rank in order],
                "target_outcome_read": False,
            }
        return body | {"applicability_receipt_sha256": stable_hash(body)}


def prior_from_frozen_artifact(
    artifact: Mapping[str, Any],
) -> SourceApplicabilityPrior:
    body = dict(artifact)
    claimed = str(body.pop("artifact_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("frozen source artifact hash mismatch")
    if artifact.get("target_data_read_for_program_induction") is not False:
        raise ValueError("source artifact does not attest target-free induction")
    profile = artifact.get("source_only_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("frozen source artifact omitted source-only profile")
    return SourceApplicabilityPrior.from_profile(profile)


def maximum_profile_contrast_derangement(
    artifacts: Mapping[str, Mapping[str, Any]],
    *, candidate_counts: Sequence[int] = (2, 3, 4),
) -> dict[str, str]:
    """Return the source-only derangement with maximal executable contrast."""

    games = tuple(sorted(map(str, artifacts)))
    if len(games) < 2:
        raise ValueError("source permutation requires at least two artifacts")
    counts = tuple(sorted({int(value) for value in candidate_counts}))
    if not counts or counts[0] < 2:
        raise ValueError("candidate counts must all be at least two")
    priors = {
        game: prior_from_frozen_artifact(artifacts[game]) for game in games
    }
    signatures = {
        game: tuple(priors[game].trial_order(count) for count in counts)
        for game in games
    }
    eligible = []
    for permuted in permutations(games):
        if any(left == right for left, right in zip(games, permuted)):
            continue
        mapping = dict(zip(games, permuted))
        first_choice_contrast = sum(
            signatures[source][-1][0] != signatures[control][-1][0]
            for source, control in mapping.items()
        )
        full_order_contrast = sum(
            signatures[source][index] != signatures[control][index]
            for source, control in mapping.items()
            for index in range(len(counts))
        )
        eligible.append((
            first_choice_contrast, full_order_contrast,
            stable_hash(mapping), mapping,
        ))
    if not eligible:
        raise ValueError("no source derangement exists")
    return max(eligible, key=lambda row: (row[0], row[1], row[2]))[3]


__all__ = [
    "SourceApplicabilityPrior",
    "prior_from_frozen_artifact",
    "projected_rank_scores",
    "maximum_profile_contrast_derangement",
]
