from scripts.collect_agqa2_transition_verifier_v7 import transition_authorized


def test_transition_authorization_requires_complete_typed_evidence():
    payload = {"supported": True, "confidence": 0.95, "precondition_observed": True, "transition_observed": True, "effect_observed": True, "same_entity_binding": True}
    assert transition_authorized(payload)
    payload["effect_observed"] = False
    assert not transition_authorized(payload)
