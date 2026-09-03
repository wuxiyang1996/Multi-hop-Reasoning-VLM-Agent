from motif_transfer.agqa_typed_evidence_contract import authorize_typed_candidate


def test_native_relation_uses_sgdet_without_action_view() -> None:
    row = authorize_typed_candidate("in front of", ["sgdet"])
    assert row.evidence_kind == "NATIVE_RELATION"
    assert row.native_projection == "inverse(spatial:behind)"
    assert row.authorized is True


def test_action_proxy_requires_same_entity_cross_view_agreement() -> None:
    assert authorize_typed_candidate("grasping", ["sgdet"]).authorized is False
    row = authorize_typed_candidate("grasping", ["sgdet", "slowfast"])
    assert row.evidence_kind == "ACTION_PROXY"
    assert row.same_entity_cross_view_agreement is True
    assert row.authorized is True


def test_source_like_provenance_cannot_authorize_candidate() -> None:
    row = authorize_typed_candidate("taking", ["sgdet", "source"])
    assert row.slowfast_exact_action_supported is False
    assert row.authorized is False
