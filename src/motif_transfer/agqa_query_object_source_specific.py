"""Frozen target-only comparator for AGQA QUERY_OBJECT source-transfer tests."""

from __future__ import annotations

from math import comb
from typing import Any, Mapping, Sequence

from .agqa_query_object_grounder import (
    AGQA_OBJECT_ONTOLOGY,
    canonical_object_label,
)


POLICY_VERSION = "AGQA_QUERY_OBJECT_TARGET_ONLY_TWO_ONTOLOGY_V1"


def target_only_ontology_decision(
    receipts: Sequence[Mapping[str, Any]],
    minimum_confidences: Sequence[float],
) -> str | None:
    """Return a label only when the two target-native ontology views agree.

    This function deliberately has no gold-answer, source-view, question, or
    functional-program argument.  Falling back to matched direct is handled by
    the evaluator after this candidate-blind decision has frozen.
    """

    if len(receipts) != 2 or len(minimum_confidences) != 2:
        raise ValueError("target-only comparator requires exactly two ontology views")
    votes: list[str] = []
    for receipt, threshold in zip(receipts, minimum_confidences, strict=True):
        if not 0 <= float(threshold) <= 1:
            raise ValueError("ontology confidence thresholds must be in [0,1]")
        decision = canonical_object_label(str(receipt.get("decision") or ""))
        observed = receipt.get("relation_observed")
        evidence = receipt.get("evidence_frames")
        confidence = float(receipt.get("confidence", -1.0))
        if (
            observed is True
            and isinstance(evidence, list)
            and bool(evidence)
            and confidence >= float(threshold)
            and decision in AGQA_OBJECT_ONTOLOGY
        ):
            votes.append(decision)
    return votes[0] if len(votes) == 2 and votes[0] == votes[1] else None


def exact_one_sided_pvalue(*, source_wins: int, source_losses: int) -> float:
    """Exact paired sign/McNemar tail under equal discordant probabilities."""

    if (
        isinstance(source_wins, bool)
        or isinstance(source_losses, bool)
        or not isinstance(source_wins, int)
        or not isinstance(source_losses, int)
        or source_wins < 0
        or source_losses < 0
    ):
        raise ValueError("paired counts must be non-negative integers")
    discordant = source_wins + source_losses
    if discordant == 0:
        return 1.0
    return sum(
        comb(discordant, value)
        for value in range(source_wins, discordant + 1)
    ) / (2 ** discordant)


__all__ = [
    "POLICY_VERSION", "exact_one_sided_pvalue",
    "target_only_ontology_decision",
]
