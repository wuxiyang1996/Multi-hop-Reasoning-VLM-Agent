import json

from motif_transfer.api_decision_agent import OpenAIJSONDecisionAgent
from motif_transfer.contracts import Advisory, AdvisoryVerdict, Lifecycle, MotifCandidate, MotifEdge, MotifNode, Observation
from motif_transfer.frozen_motif_agent import FrozenJSONMotifAgent


class Backend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.last_usage = {"total_tokens": 1}

    @property
    def identity(self):
        return {"backend": "fake"}

    def complete(self, role, system, payload):
        return json.dumps(next(self.responses))


def test_decision_agent_selects_only_numbered_native_action():
    backend = Backend([
        {"action_number": 2, "state_summary": "state", "next_subgoal": "delta"},
    ])
    agent = OpenAIJSONDecisionAgent(backend)
    proposal_set = agent.propose_set(Observation({"x": 1}, ("a", "b")), "goal", (), None)
    assert proposal_set.selected.action == "b"


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


def test_one_shot_binding_fails_closed_when_renamed_structure_differs():
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
    assert agent.initialize_binding_set_from_example(motif, example) == ()
