import pytest

from motif_transfer.agqa_active_frame_grounder import parse_public_question_plan
from motif_transfer.agqa_query_object_grounder import (
    atomic_query_object_plan,
    calibrate_query_object_execution,
    canonical_object_label,
    parse_object_ontology_receipt,
)


def _receipt(**updates):
    payload = {
        "decision": "chair",
        "relation_observed": True,
        "confidence": 0.9,
        "evidence_frames": [12, 18],
        "visual_description": "person is visibly seated on a chair",
        "uncertainty": "",
    }
    payload.update(updates)
    return parse_object_ontology_receipt(payload, frame_count=48)


def test_atomic_query_object_excludes_temporal_or_nested_subqueries():
    simple = parse_public_question_plan("What was the person sitting on?")
    temporal = parse_public_question_plan(
        "Which object were they sitting on after holding a vacuum?"
    )
    nested = parse_public_question_plan(
        "What was the person holding after watching the thing they leaned on?"
    )
    assert simple is not None and atomic_query_object_plan(simple)
    assert temporal is not None and not atomic_query_object_plan(temporal)
    assert nested is not None and not atomic_query_object_plan(nested)


def test_official_surface_aliases_canonicalize_without_per_question_candidates():
    assert canonical_object_label("the cabinet") == "closet"
    assert canonical_object_label("paper/notebook") == "paper"
    assert canonical_object_label("TV") == "television"


def test_ontology_receipt_is_closed_and_rejects_leakage():
    assert _receipt().decision == "chair"
    with pytest.raises(ValueError, match="outside"):
        _receipt(decision="countertop")
    with pytest.raises(ValueError, match="forbidden"):
        _receipt(gold_answer="chair")


def test_observed_ontology_decision_requires_evidence():
    with pytest.raises(ValueError, match="requires"):
        _receipt(evidence_frames=[])
    with pytest.raises(ValueError, match="cannot name"):
        _receipt(relation_observed=False)


def test_dual_neural_view_agreement_can_override_direct():
    result = calibrate_query_object_execution(
        base_decision="chair", direct_response="a bed",
        ontology_receipt=_receipt(), minimum_confidence=0.8,
    )
    assert result["decision"] == "chair"
    assert result["authorization_class"] == "SOURCE_TYPED_OVERRIDE"
    assert result["answer_read"] is False


def test_disagreement_or_low_confidence_abstains():
    mismatch = calibrate_query_object_execution(
        base_decision="bed", direct_response="chair",
        ontology_receipt=_receipt(), minimum_confidence=0.8,
    )
    low = calibrate_query_object_execution(
        base_decision="chair", direct_response="bed",
        ontology_receipt=_receipt(confidence=0.7), minimum_confidence=0.8,
    )
    assert mismatch["decision"] is None
    assert low["decision"] is None
