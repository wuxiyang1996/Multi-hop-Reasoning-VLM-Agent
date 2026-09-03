"""Cross-model consensus executor for AGQA QUERY_OBJECT grounding."""

from __future__ import annotations

from typing import Any, Sequence

from .agqa_query_object_grounder import (
    AGQA_OBJECT_ONTOLOGY,
    AGQAObjectOntologyReceipt,
    canonical_object_label,
)


def calibrate_query_object_consensus(
    *, base_decision: str | None, direct_response: str,
    ontology_receipts: Sequence[AGQAObjectOntologyReceipt],
    minimum_confidences: Sequence[float], minimum_neural_votes: int = 2,
) -> dict[str, Any]:
    """Execute only a label supported by two independently prompted views."""

    if len(ontology_receipts) != len(minimum_confidences):
        raise ValueError("each ontology receipt requires a frozen confidence threshold")
    if minimum_neural_votes < 2:
        raise ValueError("QUERY_OBJECT consensus requires at least two neural votes")
    votes = []
    base = canonical_object_label(base_decision or "")
    if base in AGQA_OBJECT_ONTOLOGY:
        votes.append({"view": "isolated_relation", "decision": base})
    for index, (receipt, threshold) in enumerate(
        zip(ontology_receipts, minimum_confidences, strict=True)
    ):
        decision = canonical_object_label(receipt.decision)
        if (
            decision in AGQA_OBJECT_ONTOLOGY
            and receipt.relation_observed
            and receipt.confidence >= threshold
            and bool(receipt.evidence_frames)
        ):
            votes.append({"view": f"ontology_{index}", "decision": decision})
    counts = {
        label: sum(row["decision"] == label for row in votes)
        for label in AGQA_OBJECT_ONTOLOGY
    }
    winners = [
        label for label, count in counts.items() if count >= minimum_neural_votes
    ]
    decision = winners[0] if len(winners) == 1 else None
    authorization_class = "ABSTAIN"
    reason = "NO_UNIQUE_TWO_OF_THREE_NEURAL_OBJECT_CONSENSUS"
    if decision is not None:
        if canonical_object_label(direct_response) == decision:
            authorization_class = "AGREEMENT"
            reason = "DIRECT_AND_NEURAL_OBJECT_CONSENSUS_AGREE"
        else:
            authorization_class = "SOURCE_TYPED_OVERRIDE"
            reason = "SOURCE_RECURRENCE_WITH_TWO_OF_THREE_NEURAL_OBJECT_CONSENSUS"
    return {
        "schema_version": "agqa-query-object-calibration-v2",
        "decision": decision,
        "authorization_class": authorization_class,
        "reason": reason,
        "neural_votes": votes,
        "winning_vote_count": counts.get(decision, 0) if decision else 0,
        "minimum_neural_votes": minimum_neural_votes,
        "minimum_confidences": list(minimum_confidences),
        "answer_read": False,
        "functional_program_read": False,
        "scene_graph_read": False,
        "source_identity_read": False,
    }


__all__ = ["calibrate_query_object_consensus"]
