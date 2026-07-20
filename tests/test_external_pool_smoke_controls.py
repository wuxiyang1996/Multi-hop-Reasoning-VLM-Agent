from __future__ import annotations

from trainer.coevolution.config import CoEvolutionConfig
from trainer.coevolution.rollout_collector import GAME_MAX_STEPS, build_lpt_schedule


def test_external_vllm_url_accepts_comma_separated_pool(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URLS", raising=False)
    cfg = CoEvolutionConfig(
        manage_vllm=False,
        vllm_base_url="http://a/v1, http://b/v1,",
    )
    assert cfg.vllm_base_urls == ["http://a/v1", "http://b/v1"]


def test_env_pool_takes_precedence_over_cli_pool(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_BASE_URLS", "http://env-a/v1,http://env-b/v1")
    cfg = CoEvolutionConfig(
        manage_vllm=False,
        vllm_base_url="http://cli/v1",
    )
    assert cfg.vllm_base_urls == ["http://env-a/v1", "http://env-b/v1"]


def test_smoke_step_cap_is_opt_in(monkeypatch) -> None:
    game = "twenty_forty_eight"
    monkeypatch.delenv("COEVO_MAX_EPISODE_STEPS", raising=False)
    normal = build_lpt_schedule([game], 1)
    assert normal[0].max_steps == GAME_MAX_STEPS[game]

    monkeypatch.setenv("COEVO_MAX_EPISODE_STEPS", "5")
    smoke = build_lpt_schedule([game], 1)
    assert smoke[0].max_steps == 5

    monkeypatch.setenv("COEVO_MAX_EPISODE_STEPS", "not-a-number")
    invalid = build_lpt_schedule([game], 1)
    assert invalid[0].max_steps == GAME_MAX_STEPS[game]
