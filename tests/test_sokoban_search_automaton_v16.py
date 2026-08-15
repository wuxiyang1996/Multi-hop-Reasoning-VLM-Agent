from __future__ import annotations

from motif_transfer.sokoban_search_automaton_v16 import (
    BACKTRACK,
    COMMIT,
    EVENTS,
    EXPLORE,
    REFUTED,
    UNBOUND,
    VERIFIED,
    alpha_renaming_invariant,
    evaluate_policy,
    induce_event_policy,
    matched_decision_rows,
    permute_policy,
)


def _receipts() -> list[dict]:
    return [
        {
            "snapshot_id": "one",
            "episode_id": "episode-one",
            "candidate_count": 4,
            "verified_candidate_rank": 2,
        },
        {
            "snapshot_id": "two",
            "episode_id": "episode-two",
            "candidate_count": 4,
            "verified_candidate_rank": 0,
        },
    ]


def test_matched_rows_identify_all_three_event_actions() -> None:
    rows = matched_decision_rows(_receipts())
    policy = induce_event_policy(rows)
    assert policy == {
        UNBOUND: EXPLORE,
        REFUTED: BACKTRACK,
        VERIFIED: COMMIT,
    }
    assert {row["event"] for row in rows} == set(EVENTS)


def test_authentic_policy_executes_closed_loop_and_controls_fail() -> None:
    receipts = _receipts()
    policy = {UNBOUND: EXPLORE, REFUTED: BACKTRACK, VERIFIED: COMMIT}
    authentic = evaluate_policy(receipts, policy)
    assert authentic["successes"] == 2
    assert evaluate_policy(receipts, permute_policy(policy))["successes"] == 0
    assert evaluate_policy(receipts, policy, ledger_blind=True)["successes"] == 1


def test_alpha_renaming_does_not_change_execution() -> None:
    policy = {UNBOUND: EXPLORE, REFUTED: BACKTRACK, VERIFIED: COMMIT}
    assert alpha_renaming_invariant(_receipts(), policy)
