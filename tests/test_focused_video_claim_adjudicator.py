from motif_transfer.focused_video_claim_adjudicator import (
    focus_indices,
    fuse_supported_receipt,
    parse_focused_adjudication,
)


def _receipt():
    return {
        "claim_status": "SUPPORTED",
        "confidence": 0.9,
        "checks": [
            {
                "kind": kind,
                "status": "SUPPORTED",
                "confidence": 0.9,
                "basis": "OBSERVED",
                "evidence_frames": [4, 8],
                "fact": kind,
            }
            for kind in (
                "ENTITY_BINDING", "PRECONDITION", "POSTCONDITION",
                "DIRECTIONAL_OR_CAUSAL_LINK", "CLAIM_ENTAILMENT",
            )
        ],
        "uncertainties": [],
        "reason": "initial",
    }


def test_focus_indices_expand_transition_evidence():
    assert focus_indices(_receipt(), frame_count=12, radius=1) == (0, 3, 4, 5, 7, 8, 9, 11)


def test_adjudicator_rejects_binding_leak():
    payload = {
        "slot": "A",
        "entity_binding": "SUPPORTED",
        "precondition": "SUPPORTED",
        "postcondition": "SUPPORTED",
        "transition_direction": "SUPPORTED",
        "claim_entailment": "SUPPORTED",
        "evidence_frames": [1, 2],
        "alternative_explanation": "",
        "confidence": 0.9,
        "reason": "ok",
    }
    try:
        parse_focused_adjudication(payload, frame_count=10)
    except ValueError as exc:
        assert "leaked" in str(exc)
    else:
        raise AssertionError("binding leak was accepted")


def test_refutation_prunes_initial_support():
    adjudication = parse_focused_adjudication({
        "entity_binding": "SUPPORTED",
        "precondition": "SUPPORTED",
        "postcondition": "REFUTED",
        "transition_direction": "REFUTED",
        "claim_entailment": "REFUTED",
        "evidence_frames": [4, 8],
        "alternative_explanation": "orientation change",
        "confidence": 0.8,
        "reason": "no closure transition",
    }, frame_count=12)
    fused = fuse_supported_receipt(_receipt(), adjudication, frame_count=12)
    assert fused["claim_status"] == "REFUTED"
    assert fused["checks"][-1]["status"] == "REFUTED"


def test_supported_claim_requires_all_transition_premises():
    payload = {
        "entity_binding": "SUPPORTED",
        "precondition": "SUPPORTED",
        "postcondition": "SUPPORTED",
        "transition_direction": "UNKNOWN",
        "claim_entailment": "SUPPORTED",
        "evidence_frames": [1, 2],
        "alternative_explanation": "",
        "confidence": 0.9,
        "reason": "bad",
    }
    try:
        parse_focused_adjudication(payload, frame_count=10)
    except ValueError as exc:
        assert "premise" in str(exc)
    else:
        raise AssertionError("incomplete supported claim was accepted")
