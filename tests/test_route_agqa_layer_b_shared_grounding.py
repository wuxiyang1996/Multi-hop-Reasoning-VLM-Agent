from scripts.route_agqa_layer_b_shared_grounding import choose_candidate


def test_router_uses_first_generic_commit():
    assert choose_candidate(("ABSTAINED", "COMMITTED")) == 1
    assert choose_candidate(("COMMITTED", "COMMITTED")) == 0


def test_router_fails_closed_to_declared_first_candidate():
    assert choose_candidate(("ABSTAINED", "ABSTAINED")) == 0
