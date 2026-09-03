import pytest

from motif_transfer.agqa_query_object_source_specific import (
    exact_one_sided_pvalue,
    target_only_ontology_decision,
)


def _receipt(decision="chair", *, confidence=0.9, observed=True):
    return {
        "decision": decision,
        "confidence": confidence,
        "relation_observed": observed,
        "evidence_frames": [3] if observed else [],
    }


def test_target_only_comparator_requires_two_valid_agreeing_views():
    assert target_only_ontology_decision(
        [_receipt(), _receipt()], [0.8, 0.8],
    ) == "chair"
    assert target_only_ontology_decision(
        [_receipt(), _receipt("table")], [0.8, 0.8],
    ) is None
    assert target_only_ontology_decision(
        [_receipt(), _receipt(confidence=0.79)], [0.8, 0.8],
    ) is None


def test_target_only_comparator_cannot_accept_source_or_gold_inputs():
    with pytest.raises(ValueError):
        target_only_ontology_decision([_receipt()], [0.8])


def test_exact_one_sided_pvalue_matches_frozen_paired_test():
    assert exact_one_sided_pvalue(source_wins=5, source_losses=0) == 0.03125
    assert exact_one_sided_pvalue(source_wins=7, source_losses=1) == 0.03515625
    assert exact_one_sided_pvalue(source_wins=6, source_losses=1) == 0.0625
    assert exact_one_sided_pvalue(source_wins=0, source_losses=0) == 1.0
    with pytest.raises(ValueError):
        exact_one_sided_pvalue(source_wins=-1, source_losses=0)
