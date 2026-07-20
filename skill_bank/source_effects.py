"""Deterministic source-state effect extraction; no model labels involved."""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Sequence, Tuple


AGENT_LOCATION_CHANGED = "agent_location_changed"
MOVABLE_LOCATION_CHANGED = "movable_location_changed"
POSSESSION_ACQUIRED = "possession_acquired"
RECEPTACLE_OPENED = "receptacle_opened"
SOURCE_STATE_CHANGED = "source_state_changed"


_TABLE_ROW = re.compile(
    r"^\s*\d+\s*\|\s*(?P<kind>[A-Za-z ]+?)\s*\|\s*"
    r"\((?P<x>-?\d+)\s*,\s*(?P<y>-?\d+)\)\s*$",
    re.MULTILINE,
)
_MARIO = re.compile(r"(?:Position of Mario:|mario=)\s*\((?P<x>-?\d+)\s*,\s*(?P<y>-?\d+)\)", re.I)


def _sokoban_entities(text: str) -> Mapping[str, Tuple[Tuple[int, int], ...]]:
    values: Dict[str, List[Tuple[int, int]]] = {}
    for match in _TABLE_ROW.finditer(text):
        kind = match.group("kind").strip().lower()
        values.setdefault(kind, []).append((int(match.group("x")), int(match.group("y"))))
    return {key: tuple(sorted(items)) for key, items in values.items()}


def extract_source_effects(
    *,
    game: str,
    state: str,
    next_state: str,
    action: str,
    reward: float,
    done: bool,
) -> Sequence[str]:
    """Return only effects directly decidable from domain state snapshots."""
    del action, reward, done
    effects: List[str] = []
    if state != next_state:
        effects.append(SOURCE_STATE_CHANGED)
    normalized_game = game.lower().strip()
    if normalized_game == "sokoban":
        before = _sokoban_entities(state)
        after = _sokoban_entities(next_state)
        if before.get("worker") and after.get("worker") and before["worker"] != after["worker"]:
            effects.append(AGENT_LOCATION_CHANGED)
        if before.get("box") and after.get("box") and before["box"] != after["box"]:
            effects.append(MOVABLE_LOCATION_CHANGED)
    elif normalized_game in {"super_mario", "mario"}:
        before_mario = _MARIO.search(state)
        after_mario = _MARIO.search(next_state)
        if before_mario and after_mario and before_mario.groups() != after_mario.groups():
            effects.append(AGENT_LOCATION_CHANGED)
        # The source game exposes no inventory fact. A mushroom disappearing
        # from the viewport is not sufficient evidence of possession, so this
        # parser intentionally never emits POSSESSION_ACQUIRED.
    return tuple(sorted(set(effects)))


__all__ = [
    "AGENT_LOCATION_CHANGED",
    "MOVABLE_LOCATION_CHANGED",
    "POSSESSION_ACQUIRED",
    "RECEPTACLE_OPENED",
    "SOURCE_STATE_CHANGED",
    "extract_source_effects",
]
