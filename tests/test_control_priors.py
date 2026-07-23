import pytest
import json

from motif_transfer.binding import BindingVersionSpace
from motif_transfer.contracts import Lifecycle, Observation, TransferObjectKind
from motif_transfer.control_priors import (
    ControlKnowledgeRole,
    ReceiptGroundedClause,
    ReceiptGroundedKnowledge,
    ReceiptKnowledgeHarnessAgent,
    audit_receipt_grounded_knowledge,
    compile_weak_prior_controls,
    initialize_weak_prior_hypothesis,
    knowledge_from_mapping,
    knowledge_to_mapping,
    weak_prior_view,
)
from motif_transfer.decision_agent import FirstNativeDecisionAgent
from motif_transfer.runtime import TwoAgentRuntime


def supported_knowledge():
    clauses = (
        ReceiptGroundedClause.create(
            ControlKnowledgeRole.VERIFICATION_ROUTINE,
            "A proposed transition should be checked against the next observation.",
            ("r0", "r1"),
        ),
        ReceiptGroundedClause.create(
            ControlKnowledgeRole.FAILURE_SIGNATURE,
            "Contradictory evidence should disable this hypothesis.",
            ("r2", "r3"),
        ),
    )
    return ReceiptGroundedKnowledge.create(
        ("episode-a", "episode-b"), clauses, status=Lifecycle.SOURCE_SUPPORTED
    )


def test_receipt_grounded_knowledge_requires_cross_episode_support():
    knowledge = supported_knowledge()
    accepted = audit_receipt_grounded_knowledge(
        knowledge,
        receipt_to_episode={"r0": "a", "r1": "b", "r2": "a", "r3": "b"},
    )
    assert accepted.accepted
    rejected = audit_receipt_grounded_knowledge(
        knowledge,
        receipt_to_episode={"r0": "a", "r1": "a", "r2": "a", "r3": "b"},
    )
    assert not rejected.accepted
    assert any(code.startswith("INSUFFICIENT_EPISODE_RECURRENCE") for code in rejected.failure_codes)


def test_weak_prior_view_has_no_topology_or_action_mapping():
    view = weak_prior_view(supported_knowledge())
    serialized = str(view)
    assert "nodes" not in view
    assert "edges" not in view
    assert "node_alignment" not in serialized
    assert "edge_alignment" not in serialized
    assert "action" not in serialized.lower()


def test_weak_prior_hypothesis_cannot_carry_topology():
    hypothesis = initialize_weak_prior_hypothesis(
        supported_knowledge(),
        adaptation_receipt_ids=("target-r0",),
        target_claim="additional evidence is required",
        testable_prediction="the next live receipt will support or refute the claim",
        verifier_id="live-transition",
    )
    assert hypothesis.transfer_object_kind == TransferObjectKind.WEAK_CONTROL_PRIOR
    assert not hypothesis.node_alignment
    assert not hypothesis.edge_alignment
    BindingVersionSpace((hypothesis,))
    invalid = hypothesis.__class__(
        **{
            **hypothesis.__dict__,
            "node_alignment": ((0, (0,)),),
        }
    )
    with pytest.raises(ValueError, match="may not carry target topology"):
        BindingVersionSpace((invalid,))


def test_compiled_controls_separate_receipts_content_and_other_game():
    authentic = supported_knowledge()
    other = ReceiptGroundedKnowledge.create(
        ("other-a", "other-b"),
        (
            ReceiptGroundedClause.create(
                ControlKnowledgeRole.APPLICABILITY_BOUNDARY,
                "Disable the prior when its live prediction is contradicted.",
                ("o0", "o1"),
            ),
        ),
        status=Lifecycle.SOURCE_SUPPORTED,
    )
    controls = compile_weak_prior_controls(authentic, other)
    assert set(controls) == {
        "generic_reasoning",
        "source_receipts_only",
        "authentic_weak_control_prior",
        "shuffled_evidence_prior",
        "other_game_control_prior",
    }
    assert controls["source_receipts_only"]["payload"]["clauses"][0]["untrusted_hypothesis"] == ""
    authentic_clauses = controls["authentic_weak_control_prior"]["payload"]["clauses"]
    shuffled_clauses = controls["shuffled_evidence_prior"]["payload"]["clauses"]
    assert shuffled_clauses[0]["source_receipt_ids"] == authentic_clauses[1]["source_receipt_ids"]
    assert shuffled_clauses[1]["source_receipt_ids"] == authentic_clauses[0]["source_receipt_ids"]
    assert (
        controls["authentic_weak_control_prior"]["payload"]["knowledge_id"]
        != controls["other_game_control_prior"]["payload"]["knowledge_id"]
    )


def test_knowledge_artifact_round_trip_rejects_tampering():
    payload = knowledge_to_mapping(supported_knowledge())
    assert knowledge_from_mapping(payload) == supported_knowledge()
    payload["clauses"][0]["untrusted_hypothesis"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        knowledge_from_mapping(payload)


class KnowledgeBackend:
    def complete(self, role, system, payload):
        if role == "knowledge_initialize":
            clause_id = payload["receipt_grounded_knowledge"]["clauses"][0]["clause_id"]
            return json.dumps({
                "abstain": False,
                "candidates": [{
                    "target_claim": "the selected proposal needs live verification",
                    "testable_prediction": "the next receipt distinguishes support from contradiction",
                    "verifier_id": "live",
                    "cited_clause_ids": [clause_id],
                }],
            })
        if role == "knowledge_review":
            clause_id = payload["source_knowledge"][0]["clauses"][0]["clause_id"]
            return json.dumps({"candidate_verdicts": [{
                "binding_id": row["binding_id"],
                "verdict": "ADMIT",
                "reason": "prediction is testable",
                "cited_clause_ids": [clause_id],
                "current_role": "verify",
                "open_hypotheses": ["state may change"],
                "information_need": "next observation",
                "expected_transition": "observable state delta",
                "failure_route": "source-off",
                "termination_test": "official terminal receipt",
            } for row in payload["hypotheses"]]})
        if role == "knowledge_verify":
            return json.dumps({"candidate_evidence": [{
                "binding_id": row["binding_id"],
                "verdict": "SUPPORTED",
                "reason": "live receipt is observable",
            } for row in payload["hypotheses"]]})
        raise AssertionError(role)


class OneStepEnv:
    def reset(self):
        return Observation({"state": 0}, ("native",))

    def step(self, action):
        assert action == "native"
        return Observation({"state": 1}, (), True, True, 1.0), 1.0


def test_weak_prior_runs_online_without_action_authority_or_topology():
    knowledge = supported_knowledge()
    agent = ReceiptKnowledgeHarnessAgent(
        KnowledgeBackend(), (knowledge,), allowed_verifier_ids=("live",)
    )
    hypotheses = agent.initialize_from_example(
        knowledge.knowledge_id,
        {"summary": "one frozen adaptation example"},
        adaptation_receipt_ids=("target-adaptation-r0",),
    )
    result = TwoAgentRuntime(FirstNativeDecisionAgent(), agent).run(
        OneStepEnv(), "goal", bindings=hypotheses
    )
    assert result.final_observation.official_success
    assert len(result.binding_evidence) == 1
    assert result.source_failures == ()


def test_knowledge_initialization_rejects_action_field():
    class ActionBackend(KnowledgeBackend):
        def complete(self, role, system, payload):
            value = json.loads(super().complete(role, system, payload))
            if role == "knowledge_initialize":
                value["candidates"][0]["action"] = "forbidden"
            return json.dumps(value)

    knowledge = supported_knowledge()
    agent = ReceiptKnowledgeHarnessAgent(
        ActionBackend(), (knowledge,), allowed_verifier_ids=("live",)
    )
    with pytest.raises(ValueError, match="forbidden fields"):
        agent.initialize_from_example(
            knowledge.knowledge_id,
            {},
            adaptation_receipt_ids=("target-adaptation-r0",),
        )
