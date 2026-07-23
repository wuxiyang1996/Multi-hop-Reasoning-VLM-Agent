from __future__ import annotations

import json

import pytest

from harness.agent_reasoning_cycle import (
    parse_agent_action_proposal_set,
    parse_agent_post_transition_verdict,
)


def test_agent_reasoning_cycle_parses_exact_native_grounded_receipts() -> None:
    proposal = parse_agent_action_proposal_set(json.dumps({
        "proposals": [{
            "proposal_id": "p0", "action_number": 2,
            "predicted_observable_delta": "the visible counter may change",
            "rationale": "this action tests the current plan",
        }],
        "selected_proposal_id": "p0", "decision": "EXECUTE",
    }), n_native_actions=3)
    assert proposal.selected().action_number == 2
    verdict = parse_agent_post_transition_verdict(json.dumps({
        "proposal_id": "p0", "verdict": "REFUTED", "decision": "REPLAN",
        "evidence_claim": "the observed counter did not change",
    }), expected_proposal_id="p0")
    assert verdict.decision == "REPLAN"
    assert proposal.content_hash() and verdict.content_hash()


def test_agent_reasoning_cycle_fails_closed_on_native_or_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="PROPOSAL_ACTION_OUTSIDE_NATIVE_LIST"):
        parse_agent_action_proposal_set(json.dumps({
            "proposals": [{
                "proposal_id": "p0", "action_number": 4,
                "predicted_observable_delta": "x", "rationale": "y",
            }],
            "selected_proposal_id": "p0", "decision": "EXECUTE",
        }), n_native_actions=3)
    with pytest.raises(ValueError, match="POST_VERDICT_PROPOSAL_ID_MISMATCH"):
        parse_agent_post_transition_verdict(json.dumps({
            "proposal_id": "invented", "verdict": "SUPPORTED",
            "decision": "CONTINUE", "evidence_claim": "x",
        }), expected_proposal_id="p0")


def test_agent_reasoning_cycle_rejects_prose_wrapped_json() -> None:
    with pytest.raises(ValueError, match="PROPOSAL_NOT_EXACT_JSON_OBJECT"):
        parse_agent_action_proposal_set(
            'Here is the result: {"proposals": []}', n_native_actions=1,
        )


def test_agent_reasoning_cycle_does_not_coerce_action_number_types() -> None:
    with pytest.raises(ValueError, match="WRONG_ACTION_PROPOSAL_FIELD_TYPES"):
        parse_agent_action_proposal_set(json.dumps({
            "proposals": [{
                "proposal_id": "p0", "action_number": "1",
                "predicted_observable_delta": "x", "rationale": "y",
            }],
            "selected_proposal_id": "p0", "decision": "EXECUTE",
        }), n_native_actions=1)
