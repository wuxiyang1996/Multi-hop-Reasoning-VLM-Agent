"""Anonymous counterfactual grounding for a cyclic-group inverse program."""

from __future__ import annotations

from typing import Mapping

from .tetris_rotation_transfer import circular_distance


AUTHENTIC_CONDITIONS = {
    "authentic_tetris_inverse",
    "alpha_renamed_authentic",
    "target_written_isomorphic",
}


def execute_counterfactual_inverse(
    *, panel_to_slot: Mapping[str, str], selected_identity_panel: str,
    options: Mapping[str, float], condition: str,
) -> str:
    """Bind a neural identity witness to a target-native group action.

    The neural model sees only anonymous panels.  The executor retains the
    frozen panel/action binding and can apply destructive group controls without
    another model call.
    """

    tokens = tuple(panel_to_slot)
    if selected_identity_panel not in panel_to_slot:
        raise ValueError("selected panel is absent from the frozen binding")
    if set(panel_to_slot.values()) != set(options):
        raise ValueError("panel binding and target option slots differ")
    selected_slot = panel_to_slot[selected_identity_panel]
    selected_degrees = float(options[selected_slot]) % 360.0
    if condition in AUTHENTIC_CONDITIONS:
        return selected_slot
    if condition == "binding_rotation_control":
        index = tokens.index(selected_identity_panel)
        return panel_to_slot[tokens[(index + 1) % len(tokens)]]
    if condition == "opposite_group_control":
        requested = (-selected_degrees) % 360.0
    elif condition == "half_turn_marginal_control":
        requested = 180.0
    else:
        raise ValueError(f"unknown counterfactual condition: {condition}")
    return min(
        options,
        key=lambda slot: (circular_distance(float(options[slot]), requested), slot),
    )


__all__ = ["AUTHENTIC_CONDITIONS", "execute_counterfactual_inverse"]
