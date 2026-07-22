from motif_transfer.contracts import Advisory, AdvisoryVerdict, Observation
from motif_transfer.decision_agent import FirstNativeDecisionAgent
from motif_transfer.evidence import episode_artifact, validate_episode_artifact
from motif_transfer.runtime import TwoAgentRuntime


class MotifAgent:
    def review(self, proposal, observation, binding, history):
        return Advisory(AdvisoryVerdict.ADMIT, "record")


class Env:
    def reset(self):
        return Observation({"n": 0}, ("go",))

    def step(self, action):
        return Observation({"n": 1}, (), True, True, 1), 1


def test_full_episode_artifact_is_hash_bound():
    result = TwoAgentRuntime(FirstNativeDecisionAgent(), MotifAgent()).run(Env(), "goal")
    artifact = episode_artifact(
        result,
        episode_id="episode",
        environment_id="test-env",
        policy_identity={"model": "fixture", "checkpoint_sha256": "hash"},
        seed=7,
    )
    assert validate_episode_artifact(artifact)
    assert artifact["records"][0]["proposal_set"]["proposals"][0]["action"] == "go"
    artifact["seed"] = 8
    assert not validate_episode_artifact(artifact)
