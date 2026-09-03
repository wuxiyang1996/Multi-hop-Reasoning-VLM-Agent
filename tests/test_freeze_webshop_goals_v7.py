from __future__ import annotations

from scripts.freeze_webshop_goals_v7 import _consumed_tasks, _task_index


def test_task_index_parses_webshop_task() -> None:
    assert _task_index("webshop.28") == 28


def test_consumed_roles_never_include_held_out() -> None:
    manifest = {
        "targets": {"webshop": {"partition": {"roles": {
            "adaptation": ["webshop.1"],
            "qualification": ["webshop.2"],
            "reserve": ["webshop.3"],
            "held_out": ["webshop.4"],
        }}}},
    }
    roles = _consumed_tasks(manifest)
    assert set(roles) == {"adaptation", "qualification", "reserve"}
    assert "webshop.4" not in {task for tasks in roles.values() for task in tasks}
