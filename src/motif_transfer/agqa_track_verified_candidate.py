"""Answer-blind stable-track verification for AGQA query candidates.

The frozen V2 candidate score measures action/relation support, but a query
candidate is an entity track.  This module combines that score with two
independent properties already present in the prediction-only receipt:

* the detector confidence of the selected stable track; and
* the number of sampled frames supporting that track.

All three factors are mapped to ``[0, 1]`` and combined with an unweighted
geometric mean.  There are no predicate-, answer-, task-, or source-specific
parameters.  A single global decision threshold is calibrated separately on
development data and then frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrackVerifiedCandidateScore:
    relation_action_support: float
    track_detection_confidence: float
    track_persistence: float
    score: float


def track_persistence_score(evidence_frames: int, frame_budget: int) -> float:
    """Return a concave, budget-normalized persistence score.

    ``log1p`` prevents long-lived background objects from dominating while
    still requiring more evidence than a single-frame detection.  The only
    scale is the frozen SGDET frame budget, not a fitted target constant.
    """

    count = int(evidence_frames)
    budget = int(frame_budget)
    if budget <= 0:
        raise ValueError("frame budget must be positive")
    if count <= 0 or count > budget:
        raise ValueError("track evidence count must be in [1, frame_budget]")
    return math.log1p(count) / math.log1p(budget)


def track_verified_candidate_score(
    relation_action_support: float,
    track_detection_confidence: float,
    evidence_frames: int,
    frame_budget: int,
) -> TrackVerifiedCandidateScore:
    """Combine three prediction-only evidence channels without fitted weights."""

    relation = float(relation_action_support)
    detection = float(track_detection_confidence)
    if not math.isfinite(relation) or not 0.0 <= relation <= 1.0:
        raise ValueError("relation/action support must be finite and in [0,1]")
    if not math.isfinite(detection) or not 0.0 <= detection <= 1.0:
        raise ValueError("track detection confidence must be finite and in [0,1]")
    persistence = track_persistence_score(evidence_frames, frame_budget)
    score = (relation * detection * persistence) ** (1.0 / 3.0)
    return TrackVerifiedCandidateScore(
        relation_action_support=relation,
        track_detection_confidence=detection,
        track_persistence=persistence,
        score=score,
    )


__all__ = [
    "TrackVerifiedCandidateScore",
    "track_persistence_score",
    "track_verified_candidate_score",
]
