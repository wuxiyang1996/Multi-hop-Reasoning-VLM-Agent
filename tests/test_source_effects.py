from __future__ import annotations

from skill_bank.source_effects import (
    AGENT_LOCATION_CHANGED,
    MOVABLE_LOCATION_CHANGED,
    POSSESSION_ACQUIRED,
    extract_source_effects,
)


def test_sokoban_effects_come_from_state_delta_not_skill_name() -> None:
    before = "1 | Worker | (1, 1)\n2 | Box | (2, 1)"
    after = "1 | Worker | (2, 1)\n2 | Box | (3, 1)"
    effects = extract_source_effects(
        game="sokoban", state=before, next_state=after,
        action="push right", reward=-0.1, done=False,
    )
    assert AGENT_LOCATION_CHANGED in effects
    assert MOVABLE_LOCATION_CHANGED in effects


def test_mario_viewport_disappearance_is_not_possession_evidence() -> None:
    effects = extract_source_effects(
        game="super_mario",
        state="Position of Mario: (1, 2)\n- Item Mushrooms: (4, 2)",
        next_state="Position of Mario: (2, 2)\n- Item Mushrooms: None",
        action="Jump Level: 0", reward=500.0, done=False,
    )
    assert POSSESSION_ACQUIRED not in effects
