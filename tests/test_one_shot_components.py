import json
import pytest

from motif_transfer.api_decision_agent import OpenAIJSONDecisionAgent
from motif_transfer.contracts import (
    AdvisoryVerdict, BindingHypothesis, DecisionProposal, Lifecycle,
    MotifCandidate, MotifEdge, MotifNode, Observation,
)
from motif_transfer.binding import BindingAttribution
from motif_transfer.frozen_motif_agent import FrozenJSONMotifAgent, MemoizedCompletionBackend


class Backend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.last_usage = {"total_tokens": 1}

    @property
    def identity(self):
        return {"backend": "fake"}

    def complete(self, role, system, payload):
        return json.dumps(next(self.responses))


def test_memoized_backend_reuses_only_exact_requests():
    backend = Backend([{"value": 1}, {"value": 2}])
    memo = MemoizedCompletionBackend(backend)
    first = memo.complete("r", "system", {"x": 1})
    assert memo.last_usage["cache_hit"] is False
    assert memo.complete("r", "system", {"x": 1}) == first
    assert memo.last_usage["cache_hit"] is True
    assert memo.complete("r", "system", {"x": 2}) != first


def test_decision_agent_selects_only_numbered_native_action():
    backend = Backend([
        {"action_number": 2, "state_summary": "state", "next_subgoal": "delta"},
    ])
    agent = OpenAIJSONDecisionAgent(backend)
    proposal_set = agent.propose_set(Observation({"x": 1}, ("a", "b")), "goal", (), None)
    assert proposal_set.selected.action == "b"


def test_decision_schema_retry_changes_request_and_carries_error_receipt():
    class RecordingBackend(Backend):
        def __init__(self):
            super().__init__([
                "not an object",
                {"action_number": 1, "state_summary": "state", "next_subgoal": "retry"},
            ])
            self.payloads = []

        def complete(self, role, system, payload):
            self.payloads.append(payload)
            return super().complete(role, system, payload)

    backend = RecordingBackend()
    agent = OpenAIJSONDecisionAgent(backend)
    proposal_set = agent.propose_set(
        Observation({"x": 1}, ("a",)), "goal", (), None
    )
    assert proposal_set.selected.action == "a"
    assert "_schema_retry" not in backend.payloads[0]
    assert backend.payloads[1]["_schema_retry"]["attempt"] == 1
    assert "decision model must return one JSON object" in (
        backend.payloads[1]["_schema_retry"]["previous_error"]
    )


def test_one_shot_binding_receives_real_graph_not_shape_only():
    response = {
        "abstain": False,
        "bindings": [{
            "node_alignment": [
                {"source_node_ordinal": 0, "target_cycle_indices": [0]},
                {"source_node_ordinal": 1, "target_cycle_indices": [1]},
            ],
            "edge_alignment": [
                {"source_edge_ordinal": 0, "target_boundary": [0, 1]},
            ],
            "target_claim": "provisional",
            "testable_prediction": "observable",
            "verifier_id": "official_transition_and_outcome",
        }],
    }
    backend = Backend([response, response, response, response])
    agent = FrozenJSONMotifAgent(
        backend, allowed_verifier_ids=("official_transition_and_outcome",)
    )
    motif = MotifCandidate(
        "m", ("receipt",),
        (MotifNode("n0", ("r0",)), MotifNode("n1", ("r1",))),
        (MotifEdge("n0", "n1", ("fork",)),),
        Lifecycle.GENERIC_ONLY,
    )
    example = {
        "official_success": True,
        "transitions": [
            {"action": "look", "before_native_actions": ["look"], "after_native_actions": ["go"]},
            {"action": "go", "before_native_actions": ["go"], "after_native_actions": []},
        ],
    }
    binding = agent.initialize_binding_from_example(motif, example)
    assert binding is not None
    assert binding.status == Lifecycle.TARGET_PROVISIONAL
    assert binding.node_alignment == ((0, (0,)), (1, (1,)))
    assert any(row["phase"] == "one_shot_binding_stability_gate" for row in agent.call_receipts)


def test_alpha_difference_marks_target_grounding_but_does_not_override_raw_stability():
    first = {
        "abstain": False,
        "bindings": [{
            "node_alignment": [
                {"source_node_ordinal": 0, "target_cycle_indices": [0]},
                {"source_node_ordinal": 1, "target_cycle_indices": [1, 2]},
            ],
            "edge_alignment": [{"source_edge_ordinal": 0, "target_boundary": [0, 1]}],
            "verifier_id": "v",
        }],
    }
    second = {
        "abstain": False,
        "bindings": [{
            "node_alignment": [
                {"source_node_ordinal": 0, "target_cycle_indices": [0, 1]},
                {"source_node_ordinal": 1, "target_cycle_indices": [2]},
            ],
            "edge_alignment": [{"source_edge_ordinal": 0, "target_boundary": [1, 2]}],
            "verifier_id": "v",
        }],
    }
    motif = MotifCandidate(
        "m", (), (MotifNode("n0", ("r0",)), MotifNode("n1", ("r1",))),
        (MotifEdge("n0", "n1", ("f",)),), Lifecycle.CANDIDATE,
    )
    example = {"transitions": [
        {"action": "a"}, {"action": "b"}, {"action": "c"},
    ]}
    agent = FrozenJSONMotifAgent(
        Backend([first, second, first, second]), allowed_verifier_ids=("v",)
    )
    artifact = agent.build_binding_artifact(motif, example)
    assert len(artifact.bindings) == 1
    assert artifact.bindings[0].attribution == BindingAttribution.TARGET_GROUNDED_PROVISIONAL
    assert artifact.validate()


def test_unstable_raw_binding_fails_closed_even_when_alpha_is_stable():
    left = {
        "abstain": False,
        "bindings": [{
            "node_alignment": [
                {"source_node_ordinal": 0, "target_cycle_indices": [0]},
                {"source_node_ordinal": 1, "target_cycle_indices": [1, 2]},
            ],
            "edge_alignment": [{"source_edge_ordinal": 0, "target_boundary": [0, 1]}],
            "verifier_id": "v",
        }],
    }
    right = {
        "abstain": False,
        "bindings": [{
            "node_alignment": [
                {"source_node_ordinal": 0, "target_cycle_indices": [0, 1]},
                {"source_node_ordinal": 1, "target_cycle_indices": [2]},
            ],
            "edge_alignment": [{"source_edge_ordinal": 0, "target_boundary": [1, 2]}],
            "verifier_id": "v",
        }],
    }
    motif = MotifCandidate(
        "m", (), (MotifNode("n0", ("r0",)), MotifNode("n1", ("r1",))),
        (MotifEdge("n0", "n1", ("f",)),), Lifecycle.CANDIDATE,
    )
    example = {"transitions": [{"action": "a"}, {"action": "b"}, {"action": "c"}]}
    # Call order is raw0, alpha0, raw1, alpha1.
    agent = FrozenJSONMotifAgent(Backend([left, left, right, left]), allowed_verifier_ids=("v",))
    assert agent.build_binding_artifact(motif, example).hypotheses == ()


def test_review_must_cite_a_receipt_from_registered_source_node():
    motif = MotifCandidate(
        "m", ("r0",), (MotifNode("n0", ("r0",)),), (), Lifecycle.GENERIC_ONLY,
    )
    binding = BindingHypothesis("b", "m", "claim", "prediction", ("demo",), "v")
    valid = {
        "reason": "untrusted",
        "candidate_verdicts": [{
            "binding_id": "b",
            "verdict": "ADMIT",
            "active_source_node_ordinal": 0,
            "cited_source_receipt_ordinals": [0],
        }],
    }
    agent = FrozenJSONMotifAgent(Backend([valid]))
    agent.register_motif(motif)
    assert agent.review(DecisionProposal("p", "a"), Observation({}, ("a",)), binding, ()).verdict.value == "ADMIT"

    invalid = dict(valid, candidate_verdicts=[dict(
        valid["candidate_verdicts"][0], cited_source_receipt_ordinals=[1],
    )])
    agent = FrozenJSONMotifAgent(Backend([invalid]))
    agent.register_motif(motif)
    with pytest.raises(ValueError, match="out-of-range"):
        agent.review(DecisionProposal("p", "a"), Observation({}, ("a",)), binding, ())


def test_binding_disagreement_falls_back_instead_of_selecting_one_hypothesis():
    motif = MotifCandidate(
        "m", ("r0",), (MotifNode("n0", ("r0",)),), (), Lifecycle.GENERIC_ONLY,
    )
    bindings = tuple(
        BindingHypothesis(name, "m", "claim", "prediction", ("demo",), "v")
        for name in ("b0", "b1")
    )
    response = {"candidate_verdicts": [
        {
            "binding_id": "b0", "verdict": "ADMIT", "active_source_node_ordinal": 0,
            "cited_source_receipt_ordinals": [0],
        },
        {
            "binding_id": "b1", "verdict": "REPLAN", "active_source_node_ordinal": 0,
            "cited_source_receipt_ordinals": [0],
        },
    ]}
    agent = FrozenJSONMotifAgent(Backend([response]))
    agent.register_motif(motif)
    advisory = agent.review_bindings(
        DecisionProposal("p", "a"), Observation({}, ("a",)), bindings, (),
    )
    assert advisory.verdict == AdvisoryVerdict.ABSTAIN
    assert agent.call_receipts[-1]["unanimous"] is False


def test_only_exactly_shared_binding_advice_reaches_decision_agent():
    motif = MotifCandidate(
        "m", ("r0",), (MotifNode("n0", ("r0",)),), (), Lifecycle.GENERIC_ONLY,
    )
    bindings = tuple(
        BindingHypothesis(name, "m", "claim", "prediction", ("demo",), "v")
        for name in ("b0", "b1")
    )
    response = {"candidate_verdicts": [
        {
            "binding_id": name, "verdict": "ADMIT", "active_source_node_ordinal": 0,
            "cited_source_receipt_ordinals": [0], "current_role": "shared",
            "information_need": need, "open_hypotheses": ["shared", unique],
        }
        for name, need, unique in (("b0", "left", "x"), ("b1", "right", "y"))
    ]}
    agent = FrozenJSONMotifAgent(Backend([response]))
    agent.register_motif(motif)
    advisory = agent.review_bindings(
        DecisionProposal("p", "a"), Observation({}, ("a",)), bindings, (),
    )
    assert advisory.verdict == AdvisoryVerdict.ADMIT
    assert advisory.current_role == "shared"
    assert advisory.information_need == ""
    assert advisory.open_hypotheses == ("shared",)
