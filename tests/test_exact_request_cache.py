import pytest

from motif_transfer.exact_request_cache import ExactRequestCache


def test_exact_cache_persists_and_refuses_identity_or_response_drift(tmp_path) -> None:
    path = tmp_path / "cache.json"
    cache = ExactRequestCache(path, {"model": "m", "temperature": 0})
    request = {"messages": [{"role": "user", "content": "x"}]}
    response = {"content": None, "tool_calls": [{"name": "calculator"}]}
    assert cache.get(request) is None
    cache.put(request, response)
    assert ExactRequestCache(path, {"model": "m", "temperature": 0}).get(request) == response
    with pytest.raises(ValueError, match="different response"):
        cache.put(request, {"content": "changed"})
    with pytest.raises(ValueError, match="different backend identity"):
        ExactRequestCache(path, {"model": "other", "temperature": 0})
