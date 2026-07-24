from motif_transfer.contracts import Advisory, AdvisoryVerdict, Observation
from motif_transfer.decision_agent import FirstNativeDecisionAgent
from motif_transfer.replay import ReplayMismatch, replay_all_observed_alternatives
from motif_transfer.runtime import TwoAgentRuntime


class MotifAgent:
    def review(self, proposal, observation, binding, history):
        return Advisory(AdvisoryVerdict.ADMIT, "record")


class ReplayEnv:
    def __init__(self):
        self.n = 0

    def reset(self, *, seed=None):
        self.n = 0
        return Observation({"n": self.n, "seed": seed}, ("a", "b"))

    def step(self, action):
        self.n += 1 if action == "a" else 2
        return Observation({"n": self.n, "seed": 11}, ("a", "b"), self.n >= 1), 0


def test_exhaustive_replay_captures_native_alternative():
    # Use a wrapper so the original episode and every replay receive the same seed state.
    class Seeded(ReplayEnv):
        def reset(self, *, seed=11):
            return super().reset(seed=seed)

    result = TwoAgentRuntime(FirstNativeDecisionAgent(), MotifAgent()).run(Seeded(), "goal")
    forks = replay_all_observed_alternatives(Seeded, result.records, seed=11)
    assert len(forks) == 1
    assert forks[0].alternative_action == "b"
    assert forks[0].validate()
