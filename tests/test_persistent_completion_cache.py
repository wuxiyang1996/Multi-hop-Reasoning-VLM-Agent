from __future__ import annotations

from motif_transfer.frozen_motif_agent import MemoizedCompletionBackend


class Backend:
    identity = {"model": "fixed"}

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = {}

    def complete(self, role, system, payload):
        self.calls += 1
        self.last_usage = {"call": self.calls}
        return f"completion-{self.calls}"


def test_persistent_cache_reuses_completion_across_instances(tmp_path) -> None:
    path = tmp_path / "cache.json"
    first_backend = Backend()
    first = MemoizedCompletionBackend(first_backend, cache_path=path)
    assert first.complete("decision", "system", {"x": 1}) == "completion-1"
    assert first_backend.calls == 1

    second_backend = Backend()
    second = MemoizedCompletionBackend(second_backend, cache_path=path)
    assert second.complete("decision", "system", {"x": 1}) == "completion-1"
    assert second_backend.calls == 0
    assert second.last_usage["cache_hit"] is True
