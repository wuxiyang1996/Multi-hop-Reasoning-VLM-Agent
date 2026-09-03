from __future__ import annotations

from motif_transfer.alfworld_search_automaton_v16 import (
    classify_target_outcome,
    target_policy_rank,
    target_scope_id,
)
from motif_transfer.search_automaton_transfer_v16 import (
    OUTCOME_NONTERMINAL_EFFECT,
    OUTCOME_REFUTED,
    OUTCOME_TERMINAL_VERIFIED,
)


def test_target_policy_rank_is_target_only_and_repeat_optional() -> None:
    grounded = {
        "look": {"policy": 0.9, "applicability": 1.0},
        "go to drawer 1": {"policy": 0.8, "applicability": 1.0},
    }
    assert target_policy_rank(grounded, [], discount_repeats=True)[0] == "look"
    assert target_policy_rank(grounded, ["look"], discount_repeats=True)[0] == (
        "go to drawer 1"
    )
    assert target_policy_rank(grounded, ["look"], discount_repeats=False)[0] == "look"


def test_target_effect_classification_uses_native_feedback() -> None:
    assert classify_target_outcome(
        goal="put a mug in cabinet.", selected_action="put mug 1 in cabinet 1",
        selected_grounding={"option": "PLACE", "required_option": "PLACE"},
        effect_history=["take mug 1 from countertop 1"],
        before_observation="room", after_observation="done",
        before_native_actions=["look"], after_native_actions=[],
        official_success_after=True,
    ) == OUTCOME_TERMINAL_VERIFIED
    assert classify_target_outcome(
        goal="put a mug in cabinet.", selected_action="look",
        selected_grounding={"option": "SEARCH", "required_option": "SEARCH"},
        effect_history=[],
        before_observation="room", after_observation="Nothing happens.",
        before_native_actions=["look"], after_native_actions=["look"],
        official_success_after=False,
    ) == OUTCOME_REFUTED
    assert classify_target_outcome(
        goal="put a mug in cabinet.",
        selected_action="take mug 1 from countertop 1",
        selected_grounding={"option": "ACQUIRE", "required_option": "ACQUIRE"},
        effect_history=[],
        before_observation="room", after_observation="You pick up mug 1.",
        before_native_actions=["take mug 1 from countertop 1"],
        after_native_actions=["go to cabinet 1"],
        official_success_after=False,
    ) == OUTCOME_NONTERMINAL_EFFECT


def test_scope_ignores_surface_response_but_tracks_target_affordances() -> None:
    kwargs = {
        "goal": "put a mug in cabinet.",
        "native_actions": ["look", "go to cabinet 1"],
        "history": [],
    }
    assert target_scope_id(**kwargs) == target_scope_id(**kwargs)
    assert target_scope_id(**kwargs) != target_scope_id(
        **(kwargs | {"native_actions": ["look"]})
    )
