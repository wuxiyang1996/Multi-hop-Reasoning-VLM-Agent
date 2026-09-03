from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FREEZER = REPO / "scripts/freeze_alfworld_unified_goal_acquisition_v13.py"


def _module():
    spec = importlib.util.spec_from_file_location("alfworld_v13_freezer", FREEZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deterministic_reserve_is_order_invariant_and_unique() -> None:
    freezer = _module()
    candidates = ["task-c", "task-a", "task-b", "task-d", "task-a"]
    selected = freezer.deterministic_reserve(
        candidates, seed=486312, count=3,
    )
    assert selected == freezer.deterministic_reserve(
        reversed(candidates), seed=486312, count=3,
    )
    assert len(selected) == len(set(selected)) == 3
    assert set(selected) <= set(candidates)


def test_deterministic_reserve_rejects_undersized_population() -> None:
    freezer = _module()
    try:
        freezer.deterministic_reserve(["only-one"], seed=1, count=2)
    except ValueError as exc:
        assert "untouched candidates" in str(exc)
    else:
        raise AssertionError("undersized reserve should fail closed")
