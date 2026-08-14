from __future__ import annotations

import numpy as np

from motif_transfer.webshop_coverage_transfer_v14 import (
    AUTHENTIC_COVERAGE,
    COMMIT_AVAILABILITY_COVERAGE,
    CoverageTransferController,
    INVERTED_COVERAGE,
    POSITION_PRIOR_COVERAGE,
    TARGET_COVERAGE,
    TARGET_ONLY,
)


def _row(
    *, signature: str | None = None, paired: bool = False,
    commit: bool = False, noop: bool = False,
) -> dict:
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
        "is_noop": noop,
    }


def _call(
    controller: CoverageTransferController,
    semantics: list[dict],
    *, prior_no_effect: bool = False,
    predictions: np.ndarray | None = None,
    visible_satisfied: bool = False,
    visible_unsatisfied: bool = True,
    candidates: list[str] | None = None,
):
    predictions = (
        np.zeros((len(semantics), 4), dtype=float)
        if predictions is None else predictions
    )
    return controller(
        condition=controller.condition,
        predictions=predictions,
        semantics=semantics,
        source_models={"artifact": {}},
        visible_satisfied=visible_satisfied,
        visible_unsatisfied=visible_unsatisfied,
        prior_no_effect=prior_no_effect,
        remaining_fraction=0.5,
        previous_action=None,
        candidates=(
            [f"action-{index}" for index in range(len(semantics))]
            if candidates is None else candidates
        ),
        uncertainty_scale=0.0,
        decision_margin=0.0,
    )


def test_target_only_preserves_rank_zero_with_augmented_candidates() -> None:
    controller = CoverageTransferController(TARGET_ONLY)
    rows = [_row(commit=True), _row(signature="black", paired=True)]
    decision = _call(controller, rows)
    assert decision.selected_index == 0
    assert decision.reason == "target_rank_zero"
    assert controller.coverage_interventions == 0


def test_coverage_prefers_paired_label_then_allows_commit_after_effect() -> None:
    controller = CoverageTransferController(
        TARGET_COVERAGE, goal_options={"color": "black"},
    )
    rows = [
        _row(signature="black"),
        _row(signature="black", paired=True),
        _row(commit=True),
    ]
    prepare = _call(controller, rows)
    assert prepare.selected_index == 1
    assert prepare.reason == "target_coverage_prepare:color:black"

    commit = _call(
        controller, [_row(commit=True)], prior_no_effect=False,
        visible_satisfied=True, visible_unsatisfied=False,
    )
    assert controller.ledger.commit_authorized
    assert commit.selected_index == 0
    assert commit.reason == "target_rank_zero"


def test_no_effect_does_not_verify_constraint() -> None:
    controller = CoverageTransferController(
        TARGET_COVERAGE, goal_options={"size": "3x-large"},
    )
    rows = [_row(signature="3x-large", paired=True), _row(commit=True)]
    _call(controller, rows)
    repeat = _call(controller, rows, prior_no_effect=True)
    assert repeat.selected_index == 0
    assert controller.ledger.verified == set()
    assert not controller.ledger.commit_authorized


def test_authentic_source_receives_authority_only_after_coverage() -> None:
    controller = CoverageTransferController(
        AUTHENTIC_COVERAGE, goal_options={"color": "black"},
    )
    _call(controller, [_row(signature="black", paired=True), _row(commit=True)])
    rows = [_row(commit=True), _row()]
    predictions = np.array([
        [0.9, 0.1, 0.9, 0.2],
        [0.8, 0.1, 0.0, 0.9],
    ])
    decision = _call(
        controller,
        rows,
        prior_no_effect=False,
        predictions=predictions,
        visible_satisfied=True,
        visible_unsatisfied=False,
    )
    assert decision.selected_index == 0
    assert not decision.source_abstained
    assert controller.source_calls_after_coverage == 1


def test_all_source_controls_share_the_same_target_coverage_gate() -> None:
    for condition in (
        COMMIT_AVAILABILITY_COVERAGE,
        INVERTED_COVERAGE,
        POSITION_PRIOR_COVERAGE,
    ):
        controller = CoverageTransferController(
            condition, goal_options={"color": "black"},
        )
        decision = _call(
            controller,
            [_row(commit=True), _row(signature="black", paired=True)],
        )
        assert decision.selected_index == 1
        assert decision.reason == "target_coverage_prepare:color:black"
        assert decision.source_abstained


def test_incomplete_product_is_rejected_then_untried_product_is_opened() -> None:
    controller = CoverageTransferController(
        TARGET_COVERAGE,
        goal_options={"color": "black", "size": "3x-large"},
    )
    search_rows = [
        {
            **_row(),
            "element_role": "link",
            "element_text": "[35] link 'B000000001'",
        },
        {
            **_row(),
            "element_role": "link",
            "element_text": "[47] link 'B000000002'",
        },
    ]
    first = _call(controller, search_rows)
    assert first.selected_index == 0
    assert controller.last_selected_product_id == "B000000001"

    incomplete = [
        _row(signature="black", paired=True),
        _row(commit=True),
        _row(),
    ]
    back = _call(
        controller, incomplete,
        candidates=["click('29')", "click('70')", "go_back()"],
    )
    # The final row stands for the deterministically appended go_back action.
    assert back.selected_index == 2
    assert back.reason == "target_coverage_reject_incomplete_product"

    second = _call(controller, search_rows)
    assert second.selected_index == 1
    assert second.reason == "target_coverage_explore_untried_product"
