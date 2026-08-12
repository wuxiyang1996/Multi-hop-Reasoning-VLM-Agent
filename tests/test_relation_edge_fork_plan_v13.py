from __future__ import annotations

import importlib
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
_first_opportunity = importlib.import_module(
    "freeze_relation_edge_fork_plan_v13"
)._first_opportunity
_relative_game_matches = importlib.import_module(
    "run_relation_edge_intervention_forks_v13"
)._relative_game_matches


EDGE = {
    "kind": "EDGE",
    "from": "BIND",
    "to": "RELATE",
    "guard": "TARGET_NATIVE_SLOT_READY_FOR_RELATION",
    "graph_sha256": "graph",
}


def _record(step: int, *, actionable: bool) -> dict:
    decision = {
        "action": "look" if step == 0 else "move potato 2 to fridge 1",
        "fallback_action": "close fridge 1",
        "source_applicability": {"source_edge_candidate": True},
        "candidate_source_transition": EDGE,
    }
    if actionable:
        decision.update({
            "best_realization_score": 0.9,
            "target_policy_ratio": 0.8,
            "source_transition": EDGE,
        })
    return {
        "step": step,
        "goal": "put two potato in fridge",
        "before": {"observation": f"state {step}"},
        "native_actions": ["close fridge 1", "move potato 2 to fridge 1"],
        "ledger_before": {"required_count": 2},
        "property_probabilities": {"NONE": 1.0},
        "decision": decision,
    }


def test_plan_skips_routed_but_not_actionable_edge() -> None:
    row, reason = _first_opportunity(
        version="v12",
        episode={
            "task_id": "family/task/game.tw-pddl",
            "task_family": "family",
            "records": [
                _record(0, actionable=False),
                _record(1, actionable=True),
            ],
        },
        max_steps=60,
    )
    assert reason is None
    assert row is not None
    assert row["fork_step"] == 1
    assert row["prefix_actions"] == ["look"]
    assert row["expected_source_edge"]["kind"] == "EDGE"


def test_plan_excludes_actionable_edge_outside_endpoint() -> None:
    records = [
        {
            **_record(step, actionable=(step == 2)),
            "decision": {
                **_record(step, actionable=(step == 2))["decision"],
                "action": f"look {step}",
            },
        }
        for step in range(3)
    ]
    row, reason = _first_opportunity(
        version="v9",
        episode={
            "task_id": "family/task/game.tw-pddl",
            "task_family": "family",
            "records": records,
        },
        max_steps=2,
    )
    assert row is None
    assert reason == "FIRST_CANDIDATE_OUTSIDE_V13_ENDPOINT"


def test_actual_game_file_maps_by_relative_identity() -> None:
    assert _relative_game_matches(
        "/dataset/train/family/trial/game.tw-pddl",
        "family/trial/game.tw-pddl",
    )
    assert not _relative_game_matches(
        "/dataset/train/other/trial/game.tw-pddl",
        "family/trial/game.tw-pddl",
    )
