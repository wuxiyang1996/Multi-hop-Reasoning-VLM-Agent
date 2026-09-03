from __future__ import annotations

from pathlib import Path

from scripts.analyze_phase16_gap_closure_v30 import run
from scripts.analyze_tetris_cyclic_source_induction_v28 import (
    run as run_cyclic,
)


REPO = Path(__file__).resolve().parents[1]


def test_third_program_family_fresh_source_reserve_is_stable() -> None:
    report = run_cyclic(
        REPO / "configs/tetris_cyclic_source_induction_v28.json"
    )
    assert report["status"] == (
        "THIRD_PROGRAM_FAMILY_SOURCE_RESERVE_VALIDATED"
    )
    assert all(report["qualification_gates"].values())
    assert all(report["reserve_gates"].values())
    assert report["development"]["first_qualified"][
        "complete_source_intervention_episodes"
    ] == 3


def test_phase16_declared_gap_closure_is_hash_stable() -> None:
    report = run(REPO / "configs/phase16_gap_closure_v30.json")
    assert report["status"] == (
        "DECLARED_PHASE14_15_GAPS_CLOSED_WITH_BOUNDARIES"
    )
    assert all(report["gates"].values())
    assert report["complete_source_fork_cost"][
        "all_candidate_primitive_transitions"
    ] == 108
    assert report["target_schema_synthesis_baseline"][
        "strict_exact_matches"
    ] == 0
    assert report["target_schema_synthesis_baseline"][
        "family_matches"
    ] == 8
