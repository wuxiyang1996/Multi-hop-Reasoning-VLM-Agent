from __future__ import annotations

import pytest

from env_wrappers.subprocess_env import SubprocessEnv


def _unstarted(kind: str):
    env = object.__new__(SubprocessEnv)
    env._env_kind = kind
    env._action_names = []
    env._proc = None
    return env


def test_gymv_subprocess_reset_forwards_seed_and_options() -> None:
    env = _unstarted("gymv")
    calls = []

    def call(request, timeout):
        calls.append((request, timeout))
        return {"obs": "o", "info": {"action_names": ["LEFT"]}}

    env._call = call
    observation, info = env.reset(seed=17, options={"mode": "probe"})
    assert observation == "o"
    assert info["action_names"] == ["LEFT"]
    assert calls == [({"cmd": "reset", "seed": 17, "options": {"mode": "probe"}}, 120.0)]


def test_orak_seed_does_not_silently_degrade_to_unseeded_reset() -> None:
    env = _unstarted("orak")
    env._call = lambda *_args, **_kwargs: pytest.fail("RPC must not be sent")
    with pytest.raises(NotImplementedError, match="RESET_SEED_UNSUPPORTED"):
        env.reset(seed=17)
