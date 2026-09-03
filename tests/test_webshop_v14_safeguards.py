from __future__ import annotations

import pytest

from motif_transfer.webshop_constraint_coverage_v14 import (
    ConstraintCoverage,
    augment_with_constraint_labels,
    augment_with_product_backtrack,
    ground_structured_goal_constraints,
    visible_goal_constraint_label_actions,
)
from motif_transfer.webshop_semantic_reserve import (
    ReserveIndependenceError,
    audit_semantic_reserve,
    require_semantic_reserve,
)


def _constraint(token: str, *, selected: bool = False) -> dict:
    return {
        "is_goal_constraint": True,
        "goal_overlap_tokens": [token],
        "is_selected": selected,
    }


def test_semantic_reserve_rejects_distinct_ids_that_repeat_goals() -> None:
    rows = [
        {"task_id": "webshop.1", "instruction_text": "Buy black shoes"},
        {"task_id": "webshop.2", "instruction_text": " buy   BLACK shoes "},
    ]
    audit = audit_semantic_reserve(rows)
    assert audit["unique_goal_semantics"] == 1
    assert not audit["gates"]["one_task_per_goal_semantics"]
    with pytest.raises(ReserveIndependenceError):
        require_semantic_reserve(rows)


def test_semantic_reserve_rejects_consumed_semantics() -> None:
    candidate = [{"instruction_text": "Buy blue shoes", "asin": "new"}]
    consumed = [{"instruction_text": "BUY BLUE SHOES", "asin": "old"}]
    audit = audit_semantic_reserve(candidate, consumed_rows=consumed)
    assert not audit["gates"]["instruction_disjoint_from_consumed"]


def test_semantic_reserve_accepts_unique_disjoint_goals() -> None:
    candidate = [
        {"instruction_text": "Buy blue shoes", "asin": "a"},
        {"instruction_text": "Buy a red lamp", "asin": "b"},
    ]
    consumed = [{"instruction_text": "Buy a desk", "asin": "c"}]
    assert require_semantic_reserve(
        candidate,
        consumed_rows=consumed,
        required_unique_goals=2,
        require_asin_disjointness=True,
        require_unique_candidate_asins=True,
    )["passed"]


def test_semantic_reserve_can_require_one_product_per_task() -> None:
    rows = [
        {"instruction_text": "Blue, size 8", "asin": "same"},
        {"instruction_text": "Black, size 9", "asin": "same"},
    ]
    audit = audit_semantic_reserve(rows, require_unique_candidate_asins=True)
    assert not audit["gates"]["one_task_per_asin"]


def test_commit_requires_every_observed_constraint_to_change_state() -> None:
    ledger = ConstraintCoverage()
    size, color = _constraint("10.5"), _constraint("black")
    ledger.begin_decision([size, color], prior_action_had_no_effect=False)
    ledger.record_selected(size)

    # A direct radio click was a no-op, so size is not verified.
    ledger.begin_decision([color], prior_action_had_no_effect=True)
    ledger.record_selected(color)
    ledger.begin_decision([], prior_action_had_no_effect=False)
    assert ledger.verified == {"black"}
    assert ledger.missing == ("10.5",)
    assert not ledger.commit_authorized


def test_commit_authorized_after_all_distinct_constraints_are_verified() -> None:
    ledger = ConstraintCoverage()
    size, color = _constraint("10.5"), _constraint("black")
    ledger.begin_decision([size, color], prior_action_had_no_effect=False)
    ledger.record_selected(size)
    ledger.begin_decision([color], prior_action_had_no_effect=False)
    ledger.record_selected(color)
    ledger.begin_decision([], prior_action_had_no_effect=False)
    assert ledger.commit_authorized
    assert ledger.missing == ()


def test_visible_goal_constraints_add_working_paired_label_actions() -> None:
    tree = """\
[28] radio 'black', checked='false'
[29] LabelText ''
[50] radio 'small', checked='false'
[51] LabelText ''
[70] button 'Buy Now'
"""
    assert visible_goal_constraint_label_actions(
        tree, "Find black, size large shorts"
    ) == ("click('29')",)
    assert augment_with_constraint_labels(
        ("click('28')", "click('70')"),
        axtree=tree,
        goal="Find black shorts",
    ) == ("click('28')", "click('70')", "click('29')")


def test_structured_goal_options_reject_partial_large_match() -> None:
    rows = [
        {
            "is_constraint": True,
            "is_goal_constraint": True,
            "goal_overlap_tokens": ["large"],
            "element_text": "[45] radio 'large', checked='false'",
            "paired_constraint_text": "",
        },
        {
            "is_constraint": True,
            "is_goal_constraint": True,
            "goal_overlap_tokens": ["3x", "large"],
            "element_text": "[51] radio '3x-large', checked='false'",
            "paired_constraint_text": "",
        },
    ]
    ground_structured_goal_constraints(rows, {"size": "3x-large"})
    assert not rows[0]["is_goal_constraint"]
    assert rows[1]["goal_constraint_signature"] == "size:3x large"


def test_structured_augmentation_adds_only_exact_goal_label() -> None:
    tree = """\
[45] radio 'large', checked='false'
[46] LabelText ''
[51] radio '3x-large', checked='false'
[52] LabelText ''
"""
    assert visible_goal_constraint_label_actions(
        tree,
        "Find black shorts, size 3x-large",
        goal_options={"size": "3x-large"},
    ) == ("click('52')",)


def test_product_backtrack_is_added_without_reranking() -> None:
    assert augment_with_product_backtrack(
        ("click('70')",), url="http://server/item_page/session/ASIN/query/1/{}",
    ) == ("click('70')", "go_back()")
    assert augment_with_product_backtrack(
        ("click('20')",), url="http://server/search_results/session/query/1",
    ) == ("click('20')",)
