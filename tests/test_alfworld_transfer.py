from __future__ import annotations

import copy
import sys
import types

import pytest

from common.enums import SkillSourceType, SkillType
from common.state_schema import StateSchema
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from env_wrappers.alfworld_nl_wrapper import (
    ALFWorldNLWrapper,
    alfworld_obs_to_natural_language,
    extract_alfworld_task_goal,
    make_alfworld_env,
)
from harness import AdapterRegistry, FewShotAdapter, FewShotDemo, HarnessConfig, SkillHarness
from harness.adapters.alfworld_adapter import AlfworldAdapter, bind_alfworld_executor
from labeling_supplement._failure_synth.alfworld import from_sample
from labeling_supplement._phase4_target_dispatch import registered_target_domains
from trainer.coevolution._state_to_markup import state_to_markup
from scripts.skillbridge_eval.eval_alfworld import _aggregate, run_episode
from scripts.skillbridge_eval.eval_aggregator import _DOMAIN_PRIMARY
from scripts.skillbridge_eval.run_few_shot_sweep import (
    _DRIVER_MODULE as FEW_SHOT_DRIVERS,
)
from scripts.skillbridge_eval.run_transfer_matrix import (
    DEFAULT_DOMAINS as MATRIX_DOMAINS,
    _DRIVER_MODULE as MATRIX_DRIVERS,
)


class FakeALFWorldEnv:
    def __init__(self) -> None:
        self.actions = []
        self.closed = False

    def reset(self):
        return ["You are in a kitchen."], {
            "admissible_commands": [["look", "open fridge 1"]],
        }

    def step(self, actions):
        self.actions.extend(actions)
        return ["The fridge is open."], [1.0], [True], {
            "admissible_commands": [["look", "take apple 1 from fridge 1"]],
        }

    def close(self):
        self.closed = True


class FakePartialRewardEnv:
    """Expose a shaped score that must not be accumulated across steps."""

    def reset(self):
        return ["You are in a kitchen."], {
            "admissible_commands": [["open fridge 1"]],
        }

    def step(self, actions):
        return ["The task is not complete."], [0.6], [False], {
            "admissible_commands": [["open fridge 1"]],
            "won": [False],
        }


class FakeActor:
    def act(self, **kwargs):
        assert "open fridge 1" in kwargs["action_names"]
        return types.SimpleNamespace(action="open fridge 1")


def _skill() -> SkillRecord:
    return SkillRecord.new(
        name="open_target_receptacle",
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.MINED,
        feasible_domains=["gymv", "alfworld"],
        source_domains=["gymv"],
        transfer_target_domains=["alfworld"],
        protocol=[
            {"action": "OBSERVE", "payload": {}},
            {"action": "OPEN", "payload": {"target": "fridge 1"}},
        ],
        contract=SkillContract(expected_evidence_roles=["GATHER"]),
    )


def test_wrapper_exposes_admissible_commands_as_action_names() -> None:
    raw = FakeALFWorldEnv()
    env = ALFWorldNLWrapper(raw, max_steps=3)
    observation, info = env.reset()
    assert "You are in a kitchen." in observation
    assert info["domain"] == "alfworld"
    assert info["env_name"] == "alfworld"
    assert info["action_names"] == ["look", "open fridge 1"]
    assert info["structured_state"]["won"] is False

    _, reward, terminated, truncated, next_info = env.step("open fridge 1")
    assert raw.actions == ["open fridge 1"]
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert next_info["action_names"] == ["look", "take apple 1 from fridge 1"]


def test_library_factory_loads_repo_config_without_parsing_cli(monkeypatch) -> None:
    captured = {}

    class Factory:
        def __init__(self, config, train_eval):
            captured["config"] = config
            captured["split"] = train_eval

        def init_env(self, batch_size):
            captured["batch_size"] = batch_size
            return FakeALFWorldEnv()

    env_module = types.ModuleType("alfworld.agents.environment")
    env_module.get_environment = lambda env_type: Factory
    monkeypatch.setitem(sys.modules, "alfworld", types.ModuleType("alfworld"))
    monkeypatch.setitem(sys.modules, "alfworld.agents", types.ModuleType("alfworld.agents"))
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", env_module)

    env = make_alfworld_env(split="eval_out_of_distribution")
    assert isinstance(env, ALFWorldNLWrapper)
    assert captured["split"] == "eval_out_of_distribution"
    assert captured["batch_size"] == 1
    assert captured["config"]["env"]["type"] == "AlfredTWEnv"
    assert "$ALFWORLD_DATA" in captured["config"]["dataset"]["data_path"]


def test_library_factory_fails_before_textworld_reset_when_no_games(monkeypatch) -> None:
    class EmptyFactory:
        game_files = []

        def __init__(self, config, train_eval):
            pass

        def init_env(self, batch_size):
            raise AssertionError("empty game list must fail before TextWorld registration")

    env_module = types.ModuleType("alfworld.agents.environment")
    env_module.get_environment = lambda env_type: EmptyFactory
    monkeypatch.setitem(sys.modules, "alfworld", types.ModuleType("alfworld"))
    monkeypatch.setitem(sys.modules, "alfworld.agents", types.ModuleType("alfworld.agents"))
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", env_module)

    with pytest.raises(RuntimeError, match="resolved zero games"):
        make_alfworld_env(split="train")


def test_observation_renderer_handles_batch_shape() -> None:
    text = alfworld_obs_to_natural_language(
        ["You see a sink."],
        {"admissible_commands": [["look", "go to sink 1"]]},
    )
    assert "You see a sink." in text
    assert "go to sink 1" in text


def test_observation_renderer_accepts_unbatched_commands() -> None:
    text = alfworld_obs_to_natural_language(
        "You see a sink.",
        {"admissible_commands": ["look", "go to sink 1"]},
    )
    assert "go to sink 1" in text


def test_official_task_line_is_extracted_without_semantic_rewrite() -> None:
    observation = (
        "You are in a kitchen.\n\n"
        "Your task is to: put some butterknife on drawer.\n\n"
        "Admissible actions: look; inventory"
    )
    assert extract_alfworld_task_goal(observation) == (
        "put some butterknife on drawer."
    )


def test_alfworld_state_markup_preserves_commands_and_status() -> None:
    env = ALFWorldNLWrapper(FakeALFWorldEnv())
    observation, info = env.reset()
    markup = state_to_markup(
        obs_nl=observation,
        info=info,
        game="eval_out_of_distribution",
        step=0,
    )
    assert "domain=alfworld" in markup
    assert "task_status=in_progress" in markup
    assert "open fridge 1" in markup


def test_post_training_eval_uses_admissible_action_and_env_success() -> None:
    row = run_episode(
        env=ALFWorldNLWrapper(FakeALFWorldEnv()),
        actor=FakeActor(),
        split="eval_out_of_distribution",
        episode_idx=0,
        max_steps=3,
    )
    assert row["success"] is True
    assert row["score"] == 1.0
    assert row["actions"] == ["open fridge 1"]
    assert _aggregate([row])["success_rate"] == 1.0


def test_real_adapter_binding_resolves_only_admissible_command() -> None:
    raw = FakeALFWorldEnv()
    env = ALFWorldNLWrapper(raw)
    env.reset()

    adapter = AlfworldAdapter()
    bind_alfworld_executor(adapter, env=env)
    registry = AdapterRegistry()
    registry.register(adapter)
    harness = SkillHarness(registry, config=HarnessConfig())

    episode = harness.run_skill(
        _skill(),
        StateSchema(task="open the fridge", domain="alfworld"),
        parent_run_id="test-run",
    )
    assert episode.outcome is not None
    assert episode.outcome.success is True
    assert episode.outcome.score == 1.0
    assert raw.actions == ["open fridge 1"]


def test_executor_rejects_unique_but_partial_action_match() -> None:
    raw = FakeALFWorldEnv()
    env = ALFWorldNLWrapper(raw)
    adapter = AlfworldAdapter()
    bind_alfworld_executor(adapter, env=env)
    registry = AdapterRegistry()
    registry.register(adapter)
    harness = SkillHarness(registry, config=HarnessConfig())
    skill = _skill()
    skill.protocol[-1] = {"action": "OPEN", "payload": {"target": "fridge"}}

    episode = harness.run_skill(
        skill,
        StateSchema(task="partial command must abstain", domain="alfworld"),
        parent_run_id="test-run",
    )
    assert episode.outcome is not None
    assert episode.outcome.success is False
    assert "no_exact_admissible_match" in (episode.outcome.abort_reason or "")
    assert raw.actions == []


def test_few_shot_gate_uses_real_reward_expectation() -> None:
    raw = FakeALFWorldEnv()
    env = ALFWorldNLWrapper(raw)
    env.reset()
    adapter = AlfworldAdapter()
    bind_alfworld_executor(adapter, env=env)
    registry = AdapterRegistry()
    registry.register(adapter)
    harness = SkillHarness(registry, config=HarnessConfig())

    result = FewShotAdapter(harness=harness).adapt(
        skill=_skill(),
        target_domain="alfworld",
        demos=[FewShotDemo(
            state=StateSchema(task="open the fridge", domain="alfworld"),
            expected={"min_reward": 1.0},
        )],
    )
    assert result.n_success == 1
    assert result.pass_rate == 1.0


def test_executor_resets_environment_for_each_skill_episode() -> None:
    raw = FakeALFWorldEnv()
    env = ALFWorldNLWrapper(raw)
    adapter = AlfworldAdapter()
    bind_alfworld_executor(adapter, env=env)
    registry = AdapterRegistry()
    registry.register(adapter)
    harness = SkillHarness(registry, config=HarnessConfig())

    for index in range(2):
        episode = harness.run_skill(
            _skill(),
            StateSchema(task=f"probe-{index}", domain="alfworld"),
            parent_run_id="test-run",
        )
        assert episode.outcome is not None and episode.outcome.success
    assert raw.actions == ["open fridge 1", "open fridge 1"]


def test_executor_does_not_sum_shaped_episode_scores() -> None:
    env = ALFWorldNLWrapper(FakePartialRewardEnv())
    adapter = AlfworldAdapter()
    bind_alfworld_executor(adapter, env=env)
    registry = AdapterRegistry()
    registry.register(adapter)
    harness = SkillHarness(registry, config=HarnessConfig())
    skill = copy.deepcopy(_skill())
    skill.protocol.append({"action": "OPEN", "payload": {"target": "fridge 1"}})

    episode = harness.run_skill(
        skill,
        StateSchema(task="try twice", domain="alfworld"),
        parent_run_id="test-run",
    )
    assert episode.outcome is not None
    assert episode.outcome.score == 0.6
    assert episode.final_state is not None
    assert episode.final_state["facts"]["task_status"] == "in_progress"


def test_active_dispatcher_includes_alfworld_not_osworld() -> None:
    domains = registered_target_domains()
    assert "alfworld" in domains
    assert "osworld" not in domains


def test_post_training_eval_registries_include_alfworld() -> None:
    assert "alfworld" in MATRIX_DOMAINS
    assert MATRIX_DRIVERS["alfworld"].endswith("eval_alfworld")
    assert FEW_SHOT_DRIVERS["alfworld"].endswith("eval_alfworld")
    assert _DOMAIN_PRIMARY["alfworld"] == ("success_rate", "overall")


def test_alfworld_failure_synthesizer_uses_completion_signal() -> None:
    traces = from_sample({
        "task_id": "task-1",
        "reward": 0.0,
        "success": False,
        "observation": "The apple is still on the counter.",
    })
    assert len(traces) == 1
    assert traces[0].domain == "alfworld"
    assert traces[0].extra["synthesis_signal"] == "TASK_INCOMPLETE"
