from motif_transfer.webshop_applicability_v8 import (
    candidate_semantics,
    exact_stall,
    safe_recovery_indices,
)


def test_candidate_semantics_detects_irreversible_commit() -> None:
    row = candidate_semantics(
        observation_text="[82] button 'Buy Now'",
        url="http://x/item_page/fixed_1/a/q/1/{}",
        goal="buy red shoes",
        action="click('82')",
    )
    assert row["is_commit"]
    assert not row["is_navigation"]


def test_exact_stall_requires_same_state_and_repeated_rank_zero() -> None:
    assert exact_stall(
        previous_before_hash="same",
        previous_after_hash="same",
        rank_zero_action="click('2')",
        previous_action="click('2')",
    )
    assert not exact_stall(
        previous_before_hash="new",
        previous_after_hash="old",
        rank_zero_action="click('2')",
        previous_action="click('2')",
    )


def test_constraint_rank_zero_is_preserved_and_commit_is_never_recovery() -> None:
    rows = [
        candidate_semantics(
            observation_text="[40] radio '10.5'\n[82] button 'Buy Now'",
            url="http://x/item_page/fixed_1/a/q/1/{}",
            goal="men's size 10.5 loafers",
            action=action,
        )
        for action in ("click('40')", "click('82')")
    ]
    safe, reason = safe_recovery_indices(rows)
    assert safe == ()
    assert reason == "preserve_target_constraint_action"


def test_search_navigation_can_be_a_safe_recovery() -> None:
    text = "[20] link 'same result'\n[22] link 'different result'\n[30] button 'Buy Now'"
    rows = [
        candidate_semantics(
            observation_text=text,
            url="http://x/search_results/fixed_1/q/1",
            goal="find a product",
            action=action,
        )
        for action in ("click('20')", "click('22')", "click('30')")
    ]
    safe, reason = safe_recovery_indices(rows)
    assert safe == (1,)
    assert reason is None


def test_adjacent_label_is_grounded_as_paired_constraint_recovery() -> None:
    text = "[30] radio '60x40x40cm', checked='false'\n[31] LabelText ''"
    rows = [
        candidate_semantics(
            observation_text=text,
            url="http://x/item_page/fixed_29/a/q/1/{}",
            goal="storage box size 60x40x40cm",
            action=action,
        )
        for action in ("click('30')", "click('31')")
    ]
    safe, reason = safe_recovery_indices(rows)
    assert rows[1]["paired_constraint_bid"] == "30"
    assert rows[1]["is_goal_constraint"]
    assert not rows[0]["is_selected"]
    assert safe == (1,)
    assert reason is None


def test_true_checked_attribute_is_selected() -> None:
    row = candidate_semantics(
        observation_text="[30] radio '60x40x40cm', checked='true'",
        url="http://x/item_page/fixed_29/a/q/1/{}",
        goal="storage box size 60x40x40cm",
        action="click('30')",
    )
    assert row["is_selected"]
