import json

import pytest

from motif_transfer.contracts import (
    Advisory, AdvisoryVerdict, Observation, ReplayForkReceipt,
    SourcePolicyStepRecord, SourceTransitionReceipt,
)
from motif_transfer.decision_agent import FirstNativeDecisionAgent
from motif_transfer.frozen_motif_agent import FrozenJSONMotifAgent, PromptCondition
from motif_transfer.replay import replay_all_observed_alternatives
from motif_transfer.runtime import TwoAgentRuntime


class Env:
    def __init__(self):
        self.n = 0

    def reset(self, *, seed=3):
        self.n = 0
        return Observation({"n": 0, "seed": seed}, ("a", "b"))

    def step(self, action):
        self.n += 1
        done = self.n == 2
        return Observation({"n": self.n, "seed": 3}, ("a", "b"), done), 0


class Admit:
    def review(self, proposal, observation, binding, history):
        return Advisory(AdvisoryVerdict.ADMIT, "fixture")


class Backend:
    identity = {"model": "frozen-fixture"}

    def complete(self, role, system, payload):
        if role == "segment":
            return json.dumps(
                {
                    "motifs": [
                        {
                            "nodes": [
                                {"node_id": "n0", "cycle_indices": [0]},
                                {"node_id": "n1", "cycle_indices": [1]},
                            ],
                            "edges": [{"source": "n0", "target": "n1", "fork_indices": [0]}],
                        }
                    ]
                }
            )
        if role == "binding":
            return json.dumps(
                {
                    "abstain": False,
                    "target_claim": "untrusted",
                    "testable_prediction": "check delta",
                    "verifier_id": "delta-v1",
                }
            )
        return json.dumps({"verdict": "ADMIT", "reason": "untrusted"})


def test_frozen_agent_uses_indices_but_emits_real_receipt_references():
    result = TwoAgentRuntime(FirstNativeDecisionAgent(), Admit()).run(Env(), "goal")
    forks = replay_all_observed_alternatives(Env, result.records, seed=3)
    agent = FrozenJSONMotifAgent(
        Backend(), condition=PromptCondition.RECEIPT_ONLY, allowed_verifier_ids=("delta-v1",)
    )
    candidate = agent.propose_motifs(result.records, forks)[0]
    assert candidate.nodes[0].transition_receipt_ids == (result.receipts[0].receipt_id,)
    assert candidate.edges[0].replay_receipt_ids == (forks[0].receipt_id,)
    binding = agent.initialize_binding(candidate, result.records[:1])
    assert binding is not None and binding.verifier_id == "delta-v1"


def test_negative_model_indices_fail_closed():
    class NegativeBackend(Backend):
        def complete(self, role, system, payload):
            if role == "segment":
                return json.dumps(
                    {
                        "motifs": [
                            {
                                "nodes": [{"node_id": "n", "cycle_indices": [-1]}],
                                "edges": [],
                            }
                        ]
                    }
                )
            return super().complete(role, system, payload)

    result = TwoAgentRuntime(FirstNativeDecisionAgent(), Admit()).run(Env(), "goal")
    agent = FrozenJSONMotifAgent(NegativeBackend())
    with pytest.raises(ValueError, match="out-of-range"):
        agent.propose_motifs(result.records, ())


def test_source_motif_uses_native_policy_receipts_without_proposals():
    before0 = Observation({"n": 0}, ("a", "b"))
    after0 = Observation({"n": 1}, ("a", "b"))
    before1 = after0
    after1 = Observation({"n": 2}, ("a", "b"), terminal=True)

    def record(step, before, after, reward):
        transition = SourceTransitionReceipt.create(
            before,
            episode_id="episode",
            step=step,
            selected_skill_hash="skill-hash",
            action_response_hash=f"response-{step}",
            action="a",
            action_origin="AGENT",
            policy_adapter="action_taking",
            after=after,
            reward=reward,
        )
        return SourcePolicyStepRecord(
            "episode", step, before, "skill", "skill-hash", "reasoning",
            f"response-{step}", "a", "AGENT", "action_taking", after,
            reward, transition,
        )

    records = (record(0, before0, after0, 0), record(1, before1, after1, 1))
    fork = ReplayForkReceipt.create(
        source_transition_id=records[0].transition.receipt_id,
        prefix_hash="prefix", fork_state_hash="fork", admissible_actions_hash="actions",
        alternative_action="b", alternative_after_hash="after-b",
    )

    class SourceBackend(Backend):
        def complete(self, role, system, payload):
            if role == "segment":
                return json.dumps({"motifs": [{
                    "nodes": [
                        {"node_id": "n0", "step_indices": [0]},
                        {"node_id": "n1", "step_indices": [1]},
                    ],
                    "edges": [{"source": "n0", "target": "n1", "fork_indices": [0]}],
                }]})
            return super().complete(role, system, payload)

    candidate = FrozenJSONMotifAgent(SourceBackend()).propose_source_motifs(records, (fork,))[0]
    assert candidate.nodes[0].transition_receipt_ids == (records[0].transition.receipt_id,)
    assert candidate.edges[0].replay_receipt_ids == (fork.receipt_id,)
