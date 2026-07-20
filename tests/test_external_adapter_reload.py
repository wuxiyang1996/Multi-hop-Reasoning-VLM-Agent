from __future__ import annotations

import asyncio

import httpx

from trainer.coevolution.vllm_server import reload_adapters_at_urls


def test_remote_adapter_reload_normalizes_v1_url_and_auth(
    tmp_path, monkeypatch,
) -> None:
    adapter = tmp_path / "decision" / "action_taking"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    calls = []

    class Response:
        status_code = 200
        text = "ok"

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    result = asyncio.run(
        reload_adapters_at_urls(
            str(tmp_path),
            ["http://rollout-a:8000/v1", "http://rollout-b:8010"],
        )
    )

    assert result == (2, 0)
    assert [call[0] for call in calls] == [
        "http://rollout-a:8000/v1/unload_lora_adapter",
        "http://rollout-a:8000/v1/load_lora_adapter",
        "http://rollout-b:8010/v1/unload_lora_adapter",
        "http://rollout-b:8010/v1/load_lora_adapter",
    ]
    assert all(
        call[1]["headers"] == {"Authorization": "Bearer test-key"}
        for call in calls
    )
