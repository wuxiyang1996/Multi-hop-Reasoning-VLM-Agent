from __future__ import annotations

import pytest

from motif_transfer.source_triage_gate_v15 import (
    COMMON_CONTINUATION,
    FULL_REGIME,
    PERSIST,
    SWITCH,
    build_source_triage_report,
    matched_switch_effects,
)


def _row(
    split: str,
    mode: str,
    snapshot: str,
    treatment: str,
    value: float,
) -> dict:
    return {
        "split": split,
        "mode": mode,
        "snapshot_id": snapshot,
        "treatment": treatment,
        "status": "INTERVENTION_OBSERVED",
        "game": "game-a",
        "event": "STALL",
        "fork_step": 4,
        "episode_seed": 17,
        "cumulative_returns": {"h8": value},
    }


def _matched_rows(*, switch_wins: bool) -> list[dict]:
    rows = []
    for split in ("qualification", "held_out"):
        for mode in (COMMON_CONTINUATION, FULL_REGIME):
            snapshot = f"{split}-{mode}"
            rows.extend([
                _row(split, mode, snapshot, SWITCH, 2.0 if switch_wins else 0.0),
                _row(split, mode, snapshot, PERSIST, 1.0),
            ])
    return rows


def _artifact() -> dict:
    return {
        "program": {
            "rules": [
                {"select": "COMMIT"},
                {"select": "POSITION"},
                {"select": "REPLAN_OR_ABSTAIN"},
            ]
        }
    }


def _confirmation(*, replan_examples: int) -> dict:
    return {
        "source_gate_passed": True,
        "optimal_option_counts": {
            "COMMIT": 10,
            "POSITION": 20,
            "REPLAN_OR_ABSTAIN": replan_examples,
        },
        "condition_metrics": {
            "authentic_effect_guard": {"accuracy": 0.95},
            "commit_availability_only": {"accuracy": 0.70},
            "inverted_effect_guard": {"accuracy": 0.50},
            "position_occupancy_prior": {"accuracy": 0.67},
        },
    }


def test_pairs_switch_and_persist_and_computes_relative_effect() -> None:
    effects = matched_switch_effects(_matched_rows(switch_wins=True))
    assert len(effects) == 4
    assert {row["winner"] for row in effects} == {"SWITCH"}
    assert {row["switch_minus_persist"] for row in effects} == {1.0}


def test_incomplete_matched_cell_is_an_integrity_error() -> None:
    with pytest.raises(ValueError, match="incomplete matched source cell"):
        matched_switch_effects([
            _row("qualification", COMMON_CONTINUATION, "one", SWITCH, 1.0)
        ])


def test_gate_fails_when_written_replan_has_no_examples_and_switch_loses() -> None:
    report = build_source_triage_report(
        sokoban_artifact=_artifact(),
        sokoban_confirmation=_confirmation(replan_examples=0),
        microcontroller_summary={"status": "REJECTED"},
        microcontroller_rows=_matched_rows(switch_wins=False),
    )
    assert not report["source_gate_passed"]
    assert not report["target_execution"]["authorized"]
    assert report["target_execution"]["target_files_read"] == []
    assert set(report["failed_branches"]) == {
        "INFEASIBLE__BACKTRACK_REPLAN",
        "FEASIBLE_AND_UNTRIED__EXPLORE",
    }


def test_gate_can_pass_only_when_all_three_source_branches_are_supported() -> None:
    report = build_source_triage_report(
        sokoban_artifact=_artifact(),
        sokoban_confirmation=_confirmation(replan_examples=5),
        microcontroller_summary={"status": "SUPPORTED"},
        microcontroller_rows=_matched_rows(switch_wins=True),
    )
    assert report["source_gate_passed"]
    assert report["target_execution"]["authorized"]
    assert report["failed_branches"] == []
