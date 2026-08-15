from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from motif_transfer.search_automaton_transfer_v16 import SourceSearchAutomaton
from motif_transfer.sokoban_search_automaton_v16 import COMMIT, EXPLORE
from motif_transfer.webshop_search_automaton_v16 import (
    AUTHENTIC,
    PERMUTED,
    WebShopSearchAutomatonController,
)


REPO = Path(__file__).resolve().parents[1]


def _source() -> SourceSearchAutomaton:
    return SourceSearchAutomaton(json.loads(
        (REPO / "runs/sokoban_search_automaton_v16/artifact.json").read_text()
    ))


def _row(*, signature: str | None = None, paired: bool = False, commit=False):
    option = None if signature is None else signature.split("|")[0]
    return {
        "is_goal_constraint": signature is not None,
        "goal_overlap_tokens": [] if signature is None else signature.split("|"),
        "is_selected": False,
        "paired_constraint_bid": "10" if paired else None,
        "paired_constraint_text": (
            f"[10] radio '{option}', checked='false'" if paired else ""
        ),
        "element_text": (
            f"[10] radio '{option}', checked='false'" if option else ""
        ),
        "is_constraint": signature is not None,
        "is_commit": commit,
        "is_noop": False,
    }


def _call(
    controller, semantics, *, prior_no_effect=False,
    remaining_fraction=0.5, predictions=None,
):
    predictions = (
        np.zeros((len(semantics), 4)) if predictions is None else predictions
    )
    return controller(
        condition=controller.condition,
        predictions=predictions,
        semantics=semantics,
        source_models={"artifact": {}},
        visible_satisfied=False,
        visible_unsatisfied=True,
        prior_no_effect=prior_no_effect,
        remaining_fraction=remaining_fraction,
        previous_action=None,
        candidates=[f"action-{index}" for index in range(len(semantics))],
        uncertainty_scale=0.0,
        decision_margin=0.0,
    )


def test_authentic_routes_target_coverage_explore_then_commit() -> None:
    controller = WebShopSearchAutomatonController(
        AUTHENTIC, _source(), "webshop.4", goal_options={"color": "black"},
    )
    prepare = _call(controller, [
        _row(commit=True), _row(signature="black", paired=True),
    ])
    assert prepare.selected_index == 1
    commit = _call(controller, [_row(commit=True)])
    assert commit.selected_index == 0
    counts = controller.as_dict()["source_action_counts"]
    assert counts[EXPLORE] == 1
    assert counts[COMMIT] == 1


def test_permuted_event_fails_closed_to_raw_target() -> None:
    controller = WebShopSearchAutomatonController(
        PERMUTED, _source(), "webshop.4", goal_options={"color": "black"},
    )
    decision = _call(controller, [
        _row(commit=True), _row(signature="black", paired=True),
    ])
    assert decision.selected_index == 0
    assert decision.reason == "event_permuted_source_abstain_to_raw_target"
    assert controller.target_fallbacks == 1


def test_budget_infeasible_commit_is_target_native_abstention() -> None:
    controller = WebShopSearchAutomatonController(
        AUTHENTIC,
        _source(),
        "webshop.5",
        goal_options={"color": "black", "size": "xx-large tall"},
        maximum_steps=12,
    )
    semantics = [_row(signature="black"), _row(commit=True), _row()]
    predictions = np.zeros((3, 4))
    predictions[1, 2] = 0.9
    decision = _call(
        controller,
        semantics,
        remaining_fraction=1 / 12,
        predictions=predictions,
    )
    assert decision.selected_index == 1
    assert decision.source_abstained
    assert decision.reason.startswith(
        "target_budget_infeasible_immediate_reward_salvage:"
    )
    assert controller.target_fallbacks == 1
    assert COMMIT not in controller.as_dict()["source_action_counts"]
