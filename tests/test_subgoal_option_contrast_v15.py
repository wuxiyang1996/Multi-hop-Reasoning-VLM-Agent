from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from enumerate_subgoal_option_contrasts_v15 import summarize_gates  # noqa: E402
from audit_subgoal_option_contrasts_v15 import strict_gates  # noqa: E402
from freeze_subgoal_option_contrast_pool_v15 import (  # noqa: E402
    DESTINATIONS,
    _destination,
    select_tasks,
)


def _task(destination: str, index: int) -> str:
    return (
        f"pick_two_obj_and_place-Pencil-None-{destination}-{index}/"
        f"trial_T{index:06d}/game.tw-pddl"
    )


def test_select_tasks_is_stratified_deterministic_and_excludes() -> None:
    tasks = [
        _task(destination, index)
        for destination in DESTINATIONS
        for index in range(10)
    ]
    excluded = {_task(DESTINATIONS[0], 0)}
    first, counts = select_tasks(
        tasks, excluded=excluded, seed=17, per_destination=8
    )
    second, _ = select_tasks(
        list(reversed(tasks)), excluded=excluded, seed=17, per_destination=8
    )
    assert first == second
    assert all(len(rows) == 8 for rows in first.values())
    assert counts[DESTINATIONS[0]] == 9
    assert excluded.isdisjoint({row for rows in first.values() for row in rows})
    assert _destination(first[DESTINATIONS[1]][0]) == DESTINATIONS[1]


def test_summarize_gates_requires_breadth_and_second_cycle() -> None:
    tasks = []
    for destination_index, destination in enumerate(DESTINATIONS[:4]):
        for index in range(8):
            tasks.append({
                "destination": destination,
                "authentic_action_contrast_count": 1,
                "authentic_phase_action_contrast_count": int(index < 4),
                "second_cycle_authentic_contrast_count": int(
                    destination_index < 2
                ),
            })
    requirements = {
        "minimum_tasks_with_authentic_action_contrast": 32,
        "minimum_tasks_with_authentic_phase_action_contrast": 16,
        "minimum_tasks_with_second_cycle_authentic_contrast": 16,
        "minimum_destination_groups_with_four_authentic_contrasts": 4,
    }
    assert all(summarize_gates(tasks, requirements).values())
    tasks[0]["authentic_action_contrast_count"] = 0
    assert not summarize_gates(tasks, requirements)[
        "minimum_tasks_with_authentic_action_contrast"
    ]


def test_strict_gates_require_source_specific_option_contrasts() -> None:
    tasks = []
    for destination in ("drawer", "cabinet", "shelf", "desk"):
        for _index in range(8):
            tasks.append({
                "destination": destination,
                "option_contrast_count": 1,
                "source_specific_option_contrast_count": 1,
                "second_cycle_option_contrast_count": 1,
            })
    requirements = {
        "minimum_tasks_with_option_contrast": 32,
        "minimum_tasks_with_source_specific_option_contrast": 16,
        "minimum_tasks_with_second_cycle_option_contrast": 16,
        "minimum_destination_groups_with_four_source_specific_tasks": 4,
    }
    assert all(strict_gates(tasks, requirements).values())
    for row in tasks:
        row["source_specific_option_contrast_count"] = 0
    gates = strict_gates(tasks, requirements)
    assert not gates["minimum_tasks_with_source_specific_option_contrast"]
    assert not gates[
        "minimum_destination_groups_with_four_source_specific_tasks"
    ]
