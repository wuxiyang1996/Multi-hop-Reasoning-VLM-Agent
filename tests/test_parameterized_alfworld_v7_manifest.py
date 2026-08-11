from __future__ import annotations

from scripts.freeze_parameterized_alfworld_v7_manifest import (
    _family,
    _task_ids_in_text,
)


def test_manifest_scanner_extracts_only_supported_task_identity() -> None:
    task_id = (
        "pick_cool_then_place_in_recep-Mug-None-Cabinet-6/"
        "trial_T20190908_224438_121165/game.tw-pddl"
    )
    value = f'{{"task_id": "{task_id}", "official_success": true}}'
    assert _task_ids_in_text(value) == {task_id}
    assert _task_ids_in_text('{"task_id": "not-an-alfworld-task"}') == set()


def test_family_comes_only_from_relative_task_identity() -> None:
    task_id = (
        "pick_two_obj_and_place-CellPhone-None-Shelf-320/"
        "trial_T20190908_053533_960820/game.tw-pddl"
    )
    assert _family(task_id) == "pick_two_obj_and_place"
