from __future__ import annotations

from motif_transfer.sokoban_commit_skill import (
    NATIVE_ACTIONS,
    option_features,
    parse_state,
    shortest_solution,
    simulate,
    transition_reward,
)


def _table(rows: tuple[str, ...]) -> str:
    names = {
        "#": "Wall",
        " ": "Empty",
        ".": "Dock",
        "$": "Box",
        "*": "Box on Dock",
        "@": "Worker",
        "+": "Unknown",
    }
    lines = ["ID  | Item Type    | Position", "-----------------------------"]
    index = 1
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            lines.append(f"{index:<3} | {names[value]:<12} | ({x}, {y})")
            index += 1
    return "\n".join(lines)


def test_parser_and_simulator_reproduce_push_and_reward() -> None:
    state = parse_state(_table((
        "#####",
        "# @ #",
        "# $ #",
        "# . #",
        "#####",
    )))
    transition = simulate(state, "push down")
    assert transition.worker_moved
    assert transition.box_moved
    assert transition.boxes_on_docks_delta == 1
    assert transition.after.solved
    assert transition.role == "TERMINAL_COMMIT"
    assert transition_reward(transition) == 10.9


def test_player_on_dock_unknown_is_preserved() -> None:
    state = parse_state(_table((
        "#####",
        "# + #",
        "# $ #",
        "#   #",
        "#####",
    )))
    assert state.worker == (2, 1)
    assert (2, 1) in state.docks
    moved = simulate(state, "left").after
    assert moved.worker == (1, 1)
    assert (2, 1) in moved.docks


def test_illegal_push_is_inapplicable_and_detects_corner_deadlock() -> None:
    blocked = parse_state(_table((
        "#####",
        "#@$##",
        "# . #",
        "#   #",
        "#####",
    )))
    illegal = simulate(blocked, "push right")
    assert not illegal.state_changed
    assert illegal.role == "INAPPLICABLE"

    risky = parse_state(_table((
        "######",
        "# .  #",
        "# @$ #",
        "#   ##",
        "######",
    )))
    pushed = simulate(risky, "push right")
    assert pushed.box_moved
    assert pushed.created_static_deadlock
    assert pushed.role == "RISKY_COMMIT"


def test_shortest_solution_exposes_position_then_commit_structure() -> None:
    state = parse_state(_table((
        "#######",
        "#@    #",
        "#  $  #",
        "#  .  #",
        "#######",
    )))
    plan = shortest_solution(state, maximum_nodes=10_000)
    assert plan is not None
    assert plan[-1] == "push down"
    assert not plan[0].startswith("push ")
    assert simulate(state, plan[0]).option == "POSITION"
    assert len(option_features(state, "POSITION")) == len(
        option_features(state, "COMMIT")
    )
    assert set(NATIVE_ACTIONS) == {
        "up", "down", "left", "right", "push up", "push down",
        "push left", "push right", "no_op",
    }
