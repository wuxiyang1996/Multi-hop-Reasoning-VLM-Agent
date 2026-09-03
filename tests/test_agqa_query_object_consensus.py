from motif_transfer.agqa_query_object_consensus import (
    calibrate_query_object_consensus,
)
from motif_transfer.agqa_query_object_grounder import parse_object_ontology_receipt


def _receipt(decision, confidence=0.9):
    return parse_object_ontology_receipt({
        "decision": decision,
        "relation_observed": True,
        "confidence": confidence,
        "evidence_frames": [4, 12],
        "visual_description": "visible relation",
        "uncertainty": "",
    }, frame_count=48)


def test_two_of_three_neural_consensus_does_not_use_direct_as_a_vote():
    consensus = calibrate_query_object_consensus(
        base_decision=None, direct_response="bed",
        ontology_receipts=[_receipt("chair"), _receipt("chair", 0.85)],
        minimum_confidences=[0.8, 0.8],
    )
    assert consensus["decision"] == "chair"
    assert consensus["authorization_class"] == "SOURCE_TYPED_OVERRIDE"
    assert consensus["winning_vote_count"] == 2

    abstained = calibrate_query_object_consensus(
        base_decision=None, direct_response="chair",
        ontology_receipts=[_receipt("bed"), _receipt("table")],
        minimum_confidences=[0.8, 0.8],
    )
    assert abstained["decision"] is None
    assert all(row["view"] != "direct" for row in abstained["neural_votes"])


def test_base_vote_can_form_consensus_with_one_independent_ontology_view():
    result = calibrate_query_object_consensus(
        base_decision="laptop", direct_response="tablet screen",
        ontology_receipts=[_receipt("laptop"), _receipt("television")],
        minimum_confidences=[0.8, 0.8],
    )
    assert result["decision"] == "laptop"
    assert result["winning_vote_count"] == 2
