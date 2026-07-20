from __future__ import annotations

import os

from trainer.coevolution.config import CoEvolutionConfig
from trainer.coevolution.orchestrator import _unpin_actor_from_multi_server_pool


def _config() -> CoEvolutionConfig:
    return CoEvolutionConfig(
        model_name="actor",
        manage_vllm=False,
        vllm_base_url="http://a/v1,http://b/v1",
    )


def test_actor_is_unpinned_but_judge_stays_mapped(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URLS", raising=False)
    monkeypatch.delenv("VLLM_PIN_ACTOR_URL", raising=False)
    monkeypatch.setenv(
        "VLLM_BASE_URL_MAP",
        "actor=http://a/v1,judge=http://judge/v1",
    )
    _unpin_actor_from_multi_server_pool(_config())
    assert os.environ["VLLM_BASE_URL_MAP"] == "judge=http://judge/v1"


def test_explicit_actor_pin_is_preserved(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URLS", raising=False)
    raw = "actor=http://a/v1,judge=http://judge/v1"
    monkeypatch.setenv("VLLM_BASE_URL_MAP", raw)
    monkeypatch.setenv("VLLM_PIN_ACTOR_URL", "1")
    _unpin_actor_from_multi_server_pool(_config())
    assert os.environ["VLLM_BASE_URL_MAP"] == raw


def test_single_server_actor_pin_is_preserved(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URLS", raising=False)
    monkeypatch.delenv("VLLM_PIN_ACTOR_URL", raising=False)
    cfg = _config()
    cfg.vllm_base_url = "http://a/v1"
    raw = "actor=http://a/v1,judge=http://judge/v1"
    monkeypatch.setenv("VLLM_BASE_URL_MAP", raw)
    _unpin_actor_from_multi_server_pool(cfg)
    assert os.environ["VLLM_BASE_URL_MAP"] == raw
