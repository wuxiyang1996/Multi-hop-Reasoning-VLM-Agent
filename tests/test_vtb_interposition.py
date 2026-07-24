import pytest

from motif_transfer.vtb_interposition import (
    VTBHarnessReview,
    VTBHarnessVerification,
    VTBInterpositionHarness,
    VTBReviewVerdict,
    VTBToolProposal,
    VTBToolReceipt,
    VTBVerificationVerdict,
    parse_harness_review,
    pad_json_to_exact_tokens,
)


TOOLS = ({
    "type": "function",
    "function": {
        "name": "calculator",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
},)


def test_only_decision_agent_can_construct_schema_valid_tool_call() -> None:
    harness = VTBInterpositionHarness(TOOLS, "contract")
    proposal = VTBToolProposal.create(0, "call-1", "calculator", {"expression": "1+1"})
    harness.validate_proposal(proposal)
    with pytest.raises(ValueError, match="invalid official tool arguments"):
        harness.validate_proposal(VTBToolProposal.create(0, "call-2", "calculator", {}))
    with pytest.raises(ValueError, match="only Decision Agent"):
        harness.validate_proposal(VTBToolProposal(**{**proposal.__dict__, "agent_id": "motif-agent"}))


def test_review_schema_cannot_smuggle_action_and_receipts_must_exist() -> None:
    with pytest.raises(ValueError, match="forbidden fields"):
        parse_harness_review({
            "verdict": "ADMIT", "reason": "x", "expected_transition": "y",
            "termination_test": "z", "tool_name": "calculator",
        })
    review = VTBHarnessReview(
        VTBReviewVerdict.ADMIT, "supported", "observable output", "answerable",
        ("source-1",), ("live-1",),
    )
    harness = VTBInterpositionHarness(TOOLS, "contract")
    harness.validate_review(
        review, condition="authentic_game_source",
        known_source_receipts={"source-1"}, known_live_receipts={"live-1"},
    )
    with pytest.raises(ValueError, match="non-source"):
        harness.validate_review(
            review, condition="generic_reasoning",
            known_source_receipts={"source-1"}, known_live_receipts={"live-1"},
        )


def test_execution_and_verification_are_hash_and_lineage_bound() -> None:
    harness = VTBInterpositionHarness(TOOLS, "contract")
    proposal = VTBToolProposal.create(0, "call-1", "calculator", {"expression": "1+1"})
    receipt = VTBToolReceipt.create(
        proposal, tool_contract_sha256="contract", observation={"result": "2"}, success=True,
    )
    harness.validate_receipt(proposal, receipt)
    verification = VTBHarnessVerification(
        VTBVerificationVerdict.SUPPORTED, "prediction matched", receipt.receipt_id,
    )
    harness.validate_verification(verification, {receipt.receipt_id})
    with pytest.raises(ValueError, match="fabricated"):
        harness.validate_verification(
            VTBHarnessVerification(VTBVerificationVerdict.SUPPORTED, "x", "missing"),
            {receipt.receipt_id},
        )


def test_harness_payload_padding_is_exact_and_content_preserving() -> None:
    import json
    import tiktoken

    text = pad_json_to_exact_tokens({"condition": "authentic", "evidence": ["r1"]}, 128)
    assert len(tiktoken.get_encoding("o200k_base").encode(text)) == 128
    value = json.loads(text)
    assert value["condition"] == "authentic"
    assert value["evidence"] == ["r1"]
