from scripts.summarize_agqa2_query_object_v20_v25 import (
    _lexical_direct_decision,
    _ontology_only_decision,
)


def _ontology(decision, confidence=0.9, observed=True):
    return {
        "decision": decision,
        "confidence": confidence,
        "relation_observed": observed,
        "evidence_frames": [2, 8] if observed else [],
    }


def test_target_only_control_requires_two_ontology_views_to_agree():
    row = {"object_ontology_receipts": [
        _ontology("chair"), _ontology("chair"),
    ]}
    assert _ontology_only_decision(row) == "chair"
    row["object_ontology_receipts"][1]["decision"] = "table"
    assert _ontology_only_decision(row) is None


def test_lexical_direct_normalization_abstains_on_multiple_objects():
    assert _lexical_direct_decision("a laptop screen") == "laptop"
    assert _lexical_direct_decision("cabinet door") is None
    assert _lexical_direct_decision("a chair or couch armrest") is None
