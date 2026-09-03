from motif_transfer.agqa_query_object_source_specific import exact_one_sided_pvalue


def test_v28_candidate_audit_is_not_formally_significant():
    assert exact_one_sided_pvalue(source_wins=5, source_losses=1) == 0.109375
    assert exact_one_sided_pvalue(source_wins=5, source_losses=0) == 0.03125
