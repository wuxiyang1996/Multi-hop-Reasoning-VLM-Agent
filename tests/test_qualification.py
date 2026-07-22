from dataclasses import replace

from motif_transfer.qualification import SourceOutcome, SourceQualifier


def outcome(condition, success, score=0):
    return SourceOutcome(condition, "pair", "state", "prefix", "policy", "budget", success, score)


def test_source_support_requires_authentic_to_exceed_every_control():
    rows = [
        outcome("authentic_skill_loaded", True, 2),
        outcome("skill_disabled", False, 1),
        outcome("generic_protocol", False, 1),
        outcome("shuffled_topology", False, 0),
        outcome("other_source", False, 0),
    ]
    assert SourceQualifier().evaluate(rows).supported


def test_source_identity_mismatch_fails_closed():
    rows = [
        outcome("authentic_skill_loaded", True),
        replace(outcome("skill_disabled", False), initial_state_hash="different"),
        outcome("generic_protocol", False),
        outcome("shuffled_topology", False),
        outcome("other_source", False),
    ]
    report = SourceQualifier().evaluate(rows)
    assert not report.supported
    assert "identity mismatch" in report.reason
