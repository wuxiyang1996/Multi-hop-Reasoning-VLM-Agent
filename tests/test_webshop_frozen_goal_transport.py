from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.webshop_frozen_goal_transport import (
    install_frozen_goal_overrides,
)


def _manifest(goal: dict) -> dict:
    row = {
        "server_goal_index": 1,
        "goal": goal,
        "goal_sha256": stable_hash(goal),
    }
    body = {"roles": {"qualification": [row], "formal": []}}
    return body | {"artifact_sha256": stable_hash(body)}


def test_frozen_goal_replays_sampled_price_only() -> None:
    native = {
        "asin": "A", "attributes": ["soft"], "goal_options": {"size": "m"},
        "instruction_text": "native under 40", "price_upper": 40.0,
    }
    frozen = dict(native, instruction_text="frozen under 50", price_upper=50.0)
    app = SimpleNamespace(
        GOAL_SEED=0, get_goals=lambda *args, **kwargs: [{}, native],
    )
    install_frozen_goal_overrides(app, _manifest(frozen))
    goals = app.get_goals()
    random.Random(app.GOAL_SEED).shuffle(goals)
    assert goals[1] == frozen
    assert app._STRUCTURAL_FROZEN_GOALS_APPLIED == 1


def test_frozen_goal_rejects_native_semantic_drift() -> None:
    frozen = {
        "asin": "A", "attributes": ["soft"], "goal_options": {"size": "m"},
        "instruction_text": "goal under 50", "price_upper": 50.0,
    }
    native = dict(frozen, asin="B")
    app = SimpleNamespace(
        GOAL_SEED=0, get_goals=lambda *args, **kwargs: [{}, native],
    )
    install_frozen_goal_overrides(app, _manifest(frozen))
    with pytest.raises(RuntimeError, match="semantic identity drift"):
        app.get_goals()
