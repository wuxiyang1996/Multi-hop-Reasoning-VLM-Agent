from __future__ import annotations

import pytest

from motif_transfer.tetris_counterfactual_rotation import execute_counterfactual_inverse


OPTIONS = {"A": 40.0, "B": 55.0, "C": 20.0, "D": 25.0, "E": 10.0, "F": 70.0}
PANELS = {"P0": "D", "P1": "A", "P2": "F", "P3": "B", "P4": "E", "P5": "C"}


@pytest.mark.parametrize(
    "condition",
    ["authentic_tetris_inverse", "alpha_renamed_authentic", "target_written_isomorphic"],
)
def test_authentic_programs_bind_identity_panel(condition: str) -> None:
    assert execute_counterfactual_inverse(
        panel_to_slot=PANELS, selected_identity_panel="P1", options=OPTIONS,
        condition=condition,
    ) == "A"


def test_binding_rotation_is_a_destructive_anonymous_control() -> None:
    assert execute_counterfactual_inverse(
        panel_to_slot=PANELS, selected_identity_panel="P1", options=OPTIONS,
        condition="binding_rotation_control",
    ) == "F"


def test_opposite_and_marginal_are_group_controls() -> None:
    assert execute_counterfactual_inverse(
        panel_to_slot=PANELS, selected_identity_panel="P1", options=OPTIONS,
        condition="opposite_group_control",
    ) == "E"
    assert execute_counterfactual_inverse(
        panel_to_slot=PANELS, selected_identity_panel="P1", options=OPTIONS,
        condition="half_turn_marginal_control",
    ) == "F"


def test_invalid_panel_binding_fails_closed() -> None:
    with pytest.raises(ValueError):
        execute_counterfactual_inverse(
            panel_to_slot={"P0": "A"}, selected_identity_panel="P0",
            options=OPTIONS, condition="authentic_tetris_inverse",
        )
