from dataclasses import replace
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
    backend = Backend([{
        "abstain": False,
        "target_claim": "provisional",
        "testable_prediction": "observable",
        "verifier_id": "official_transition_and_outcome",
    }])
    agent = FrozenJSONMotifAgent(
        backend, allowed_verifier_ids=("official_transition_and_outcome",)
    )
    motif = MotifCandidate(
        "m", ("receipt",),
        (MotifNode("n0", ("r0",)), MotifNode("n1", ("r1",))),
        (MotifEdge("n0", "n1", ("fork",)),),
        Lifecycle.GENERIC_ONLY,
    )
    binding = agent.initialize_binding_from_example(motif, {"official_success": True})
    assert binding is not None
    assert binding.status == Lifecycle.TARGET_PROVISIONAL
