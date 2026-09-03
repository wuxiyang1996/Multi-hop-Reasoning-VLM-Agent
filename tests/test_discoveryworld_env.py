from __future__ import annotations

from copy import deepcopy
import random

import pytest

from motif_transfer.discoveryworld_env import DiscoveryWorldEnvironment


class _World:
    def __init__(self):
        self.worldHistory = [b"initial"]
        self.state = {"step": 0, "objects": {"seen": set()}, "runtime_seconds": 1.2}

    def getWorldHistoryAtStep(self, index):
        assert index == len(self.worldHistory) - 1
        return deepcopy(self.state)


class _API:
    def __init__(self, thread_id, *, leak=False):
        self.thread_id = thread_id
        self.FRAME_DIR = ""
        self.world = _World()
        self.steps = 0
        self.leak = leak
        self.scorecard_calls = 0
        self.last_result = None

    def loadScenario(self, **kwargs):
        self.kwargs = kwargs
        return True

    def getAgentObservation(self, agentIdx):
        assert agentIdx == 0
        ui = {
            "agentLocation": {"x": self.steps, "y": 0},
            "taskProgress": [{
                "taskName": "fake",
                "description": "test a causal effect",
                "completed": self.steps >= 2,
                "completedSuccessfully": self.steps >= 2,
            }],
            "world_steps": self.steps,
            "dialog_box": {},
        }
        if self.leak:
            ui["criticalHypotheses"] = ["oracle"]
        return {"errors": [], "ui": ui, "vision": {"base64_no_grid": "x"}}

    def listKnownActions(self, limited=False):
        assert limited is False
        return {
            "MOVE_DIRECTION": {"args": ["arg1"]},
            "DISCOVERY_FEED_GET_UPDATES": {"args": []},
        }

    def listTeleportLocationsDict(self):
        return {"lab": {"gridX": 1, "gridY": 2}}

    def isAgentInDialog(self, agentIdx):
        assert agentIdx == 0
        return False

    def performAgentAction(self, agentIdx, actionJSON):
        assert agentIdx == 0
        self.last_result = {"errors": [], "success": True, "echo": dict(actionJSON)}
        return self.last_result

    def tick(self):
        self.steps += 1
        self.world.state = {
            "step": self.steps,
            "objects": {"seen": {self.steps}},
            "random_draw": random.random(),
            "runtime_seconds": 9.9,
        }
        self.world.worldHistory.append(f"step-{self.steps}".encode())
        return {"errors": [], "success": True}

    def getTaskScorecard(self):
        self.scorecard_calls += 1
        return [{"scoreNormalized": 1.0, "criticalHypotheses": ["secret"]}]


def _factory(created, *, leak=False):
    def make(thread_id):
        api = _API(thread_id, leak=leak)
        created.append(api)
        return api
    return make


def _environment(created, *, leak=False, max_steps=2):
    return DiscoveryWorldEnvironment(
        scenario="Proteomics", difficulty="Easy", seed=0, max_steps=max_steps,
        api_factory=_factory(created, leak=leak),
    )


def test_policy_channel_excludes_oracle_and_evaluation_is_terminal_only():
    created = []
    env = _environment(created)
    observation = env.reset()
    assert "score" not in str(observation.policy_payload()).lower()
    with pytest.raises(RuntimeError, match="only after"):
        env.finalize_evaluation()
    assert created[0].scorecard_calls == 0
    env.step({"action": "MOVE_DIRECTION", "arg1": "north"})
    terminal, receipt = env.step({"action": "DISCOVERY_FEED_GET_UPDATES"})
    assert terminal.terminal and terminal.official_success
    assert receipt.validate()
    evaluation = env.finalize_evaluation()
    assert created[0].scorecard_calls == 1
    assert evaluation.policy_runtime_saw_oracle_scorecard is False


def test_oracle_field_in_official_observation_fails_closed():
    env = _environment([], leak=True)
    with pytest.raises(ValueError, match="oracle field"):
        env.reset()


def test_action_schema_rejects_unknown_missing_and_extra_arguments():
    env = _environment([])
    env.reset()
    with pytest.raises(ValueError, match="unknown"):
        env.step({"action": "WAIT"})
    with pytest.raises(ValueError, match="missing"):
        env.step({"action": "MOVE_DIRECTION"})
    with pytest.raises(ValueError, match="extra"):
        env.step({"action": "DISCOVERY_FEED_GET_UPDATES", "arg1": 1})


def test_replay_prefix_reconstructs_policy_and_hidden_fork_state():
    created = []
    env = _environment(created, max_steps=4)
    actions = (
        {"action": "MOVE_DIRECTION", "arg1": "north"},
        {"action": "DISCOVERY_FEED_GET_UPDATES"},
    )
    first, receipts = env.replay_prefix(actions)
    expected_policy = first.policy_state_sha256
    expected_hidden = env.current_audit_hash
    second, replay = env.replay_prefix(
        actions,
        expected_policy_state_sha256=expected_policy,
        expected_audit_world_sha256=expected_hidden,
    )
    assert len(created) == 2
    assert first.policy_state_sha256 == second.policy_state_sha256
    assert receipts == replay
    assert all(row.runtime_saw_oracle_scorecard is False for row in replay)


def test_audit_hash_ignores_runtime_seconds_but_not_world_state():
    env = _environment([])
    env.reset()
    initial = env.current_audit_hash
    env.api.world.state["runtime_seconds"] = 10000.0
    assert env._audit_world_hash() == initial
    env.api.world.state["spriteNames"] = [{"spriteName": "cosmetic-variant"}]
    assert env._audit_world_hash() == initial
    env.api.world.state["associatedNotes"] = "unordered diagnostic rendering"
    assert env._audit_world_hash() == initial
    env.api.world.state["step"] = 99
    assert env._audit_world_hash() != initial


def test_process_random_is_restored_and_episode_rng_is_replayable():
    process_state = random.getstate()
    first_env = _environment([], max_steps=4)
    first, first_receipts = first_env.replay_prefix((
        {"action": "DISCOVERY_FEED_GET_UPDATES"},
    ))
    assert random.getstate() == process_state

    # Unrelated process-global draws must not perturb another episode.
    for _ in range(100):
        random.random()
    second_env = _environment([], max_steps=4)
    second, second_receipts = second_env.replay_prefix((
        {"action": "DISCOVERY_FEED_GET_UPDATES"},
    ))
    assert first.policy_state_sha256 == second.policy_state_sha256
    assert first_env.current_audit_hash == second_env.current_audit_hash
    assert first_receipts == second_receipts


def test_audit_hash_commits_future_random_state():
    env = _environment([])
    env.reset()
    initial = env.current_audit_hash
    env._global_random_state = random.Random(123456).getstate()
    assert env._audit_world_hash() != initial
