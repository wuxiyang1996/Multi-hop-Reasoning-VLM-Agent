from __future__ import annotations

import pytest

from motif_transfer.alfworld_env import resolve_game_index


def test_resolve_frozen_alfworld_game_id() -> None:
    files = [
        "/data/valid_seen/task-a/trial-1/game.tw-pddl",
        "/data/valid_seen/task-b/trial-2/game.tw-pddl",
    ]
    assert resolve_game_index(files, "task-b/trial-2/game.tw-pddl") == 1


def test_resolve_game_id_fails_on_missing_or_ambiguous_suffix() -> None:
    with pytest.raises(ValueError):
        resolve_game_index(["/a/x/game.tw-pddl", "/b/x/game.tw-pddl"], "x/game.tw-pddl")
    with pytest.raises(ValueError):
        resolve_game_index(["/a/x/game.tw-pddl"], "missing/game.tw-pddl")
