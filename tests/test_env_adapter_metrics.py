from motif_transfer.contracts import Advisory, AdvisoryVerdict
from motif_transfer.decision_agent import FirstNativeDecisionAgent
from motif_transfer.env_adapter import GymLikeTextAdapter
from motif_transfer.metrics import measure_episode
from motif_transfer.runtime import TwoAgentRuntime


class LegacyEnv:
    def __init__(self):
        self.n = 0

    def reset(self, seed=None):
        self.n = 0
        return "start", {"action_names": ["go"], "structured_state": {"seed": seed}}

    def step(self, action):
        self.n += 1
        return "done", 1, True, False, {"action_names": [], "won": [True], "score": 3}


class MotifAgent:
    def review(self, proposal, observation, binding, history):
        return Advisory(AdvisoryVerdict.ADMIT, "fixture")


def test_gym_like_adapter_preserves_native_actions_and_official_success():
    result = TwoAgentRuntime(FirstNativeDecisionAgent(), MotifAgent()).run(
        GymLikeTextAdapter(LegacyEnv(), seed=4), "goal"
    )
    metrics = measure_episode(result)
    assert metrics.official_success
    assert metrics.official_score == 3
    assert metrics.steps == 1
