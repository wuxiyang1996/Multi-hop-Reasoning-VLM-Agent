from __future__ import annotations

from motif_transfer.webshop_neural_grounder_v5 import (
    action_verb,
    element_text_for_bid,
    target_action_features,
    url_phase,
    validate_browsergym_action,
)


def test_target_action_features_are_target_native() -> None:
    values = target_action_features(
        observation_text="[12] button Search bid=12",
        url="http://127.0.0.1:3000/search_results/fixed_1/query/1",
        goal="find a blue cotton shirt",
        action="click('12')",
        step_index=2,
        maximum_steps=20,
        previous_action="fill('8', 'blue cotton shirt')",
    )
    assert action_verb("click('12')") == "click"
    assert url_phase("http://x/item_page/session/item") == "item_page"
    assert len(values) == 32
    assert all(-1.0 <= value <= 1.0 for value in values)
    assert element_text_for_bid("[12] button 'Search'", "12") == "[12] button 'Search'"


def test_browsergym_action_validation_is_fail_closed() -> None:
    bids = {"12", "18"}
    assert validate_browsergym_action("click('12')", bids)
    assert validate_browsergym_action("fill('18', 'blue shirt')", bids)
    assert validate_browsergym_action("scroll(0, 300)", bids)
    assert not validate_browsergym_action("click('999')", bids)
    assert not validate_browsergym_action("__import__('os').system('id')", bids)
    assert not validate_browsergym_action("click('12')} trailing", bids)
