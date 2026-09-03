import pytest

from motif_transfer.agqa_override_adjudicator import (
    adjudication_supports_typed_override,
    parse_override_adjudication,
)


def _payload(**updates):
    payload = {
        "decision": "before",
        "confidence": 0.9,
        "evidence_frames": [3, 12],
        "observed_events": ["event A at F3", "event B at F12"],
        "ambiguity": "",
        "reason": "A visibly precedes B",
    }
    payload.update(updates)
    return payload


def test_closed_adjudication_supports_only_matching_high_confidence_decision():
    parsed = parse_override_adjudication(
        _payload(), allowed_decisions=("before", "after"), frame_count=48,
    )
    assert adjudication_supports_typed_override(
        parsed, typed_decision="before", minimum_confidence=0.8,
    )
    assert not adjudication_supports_typed_override(
        parsed, typed_decision="after", minimum_confidence=0.8,
    )


def test_unknown_may_abstain_without_evidence_but_never_authorizes():
    parsed = parse_override_adjudication(
        _payload(decision="unknown", evidence_frames=[], confidence=0.95),
        allowed_decisions=("yes", "no"), frame_count=48,
    )
    assert not adjudication_supports_typed_override(
        parsed, typed_decision="yes", minimum_confidence=0.8,
    )


@pytest.mark.parametrize("leak", [
    {"gold": "before"},
    {"functional_program": "Compare(...)"},
    {"typed_prediction": "before"},
    {"direct_response": "after"},
])
def test_adjudication_rejects_evaluator_or_candidate_leakage(leak):
    with pytest.raises(ValueError, match="forbidden"):
        parse_override_adjudication(
            _payload(**leak), allowed_decisions=("before", "after"),
            frame_count=48,
        )


def test_decisive_adjudication_requires_valid_chronological_evidence():
    with pytest.raises(ValueError, match="chronological"):
        parse_override_adjudication(
            _payload(evidence_frames=[12, 3]),
            allowed_decisions=("before", "after"), frame_count=48,
        )
    with pytest.raises(ValueError, match="requires cited"):
        parse_override_adjudication(
            _payload(evidence_frames=[]),
            allowed_decisions=("before", "after"), frame_count=48,
        )
