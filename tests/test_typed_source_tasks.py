from __future__ import annotations

import pytest

from motif_transfer.typed_source_tasks import (
    TypedEffect,
    build_effect_ir,
    classify_typed_effects,
    summarize_typed_source_gate,
)


def _state(*, carrying=None, objects=(), position=(1, 1), relations=()):
    return {
        "agent_position": list(position),
        "carrying": carrying,
        "objects": list(objects),
        "relations": list(relations),
    }


def test_labels_are_derived_from_state_deltas() -> None:
    key = {"type": "key", "color": "blue", "is_open": None, "is_locked": None}
    door_closed = {
        "position": [2, 1], "type": "door", "color": "yellow",
        "is_open": False, "is_locked": True,
    }
    door_open = {**door_closed, "is_open": True, "is_locked": False}
    before = _state(carrying=key, objects=[door_closed])
    after = _state(carrying=key, objects=[door_open], position=(1, 2))
    assert classify_typed_effects(before, after) == (
        TypedEffect.POSITION,
        TypedEffect.MUTATE,
    )


def test_relate_requires_released_carried_entity() -> None:
    key = {"type": "key", "color": "blue", "is_open": None, "is_locked": None}
    relation = [["box", "purple"], ["key", "blue"]]
    assert classify_typed_effects(
        _state(carrying=key), _state(carrying=None, relations=[relation])
    ) == (TypedEffect.RELATE,)
    assert classify_typed_effects(
        _state(carrying=None), _state(carrying=None, relations=[relation])
    ) == ()


def _collection(effect: str, *, before_bound: bool) -> dict:
    return {
        "task_id": effect.lower(),
        "seed": 0,
        "split": "development",
        "required_effects": [effect],
        "replay_mismatches": 0,
        "rows": [
            {
                "group_id": "g",
                "typed_effects": ["BIND", effect] if effect != "BIND" else ["BIND"],
                "before_carrier_bound": before_bound,
                "before_replay_state_sha256": "a" * 64,
                "source_action_ordinal": 3,
            },
            {
                "group_id": "g",
                "typed_effects": [],
                "before_carrier_bound": before_bound,
                "before_replay_state_sha256": "a" * 64,
                "source_action_ordinal": 0,
            },
        ],
    }


def test_gate_requires_a_matched_negative_and_ir_strips_actions() -> None:
    collections = [
        _collection("BIND", before_bound=False),
        _collection("MUTATE", before_bound=True),
        _collection("RELATE", before_bound=True),
    ]
    gate = summarize_typed_source_gate(collections)
    assert gate["status"] == "SOURCE_TYPED_GATE_PASSED"
    ir = build_effect_ir(collections, gate)
    assert {tuple((edge["from"], edge["to"])) for edge in ir["edges"]} == {
        ("BIND", "MUTATE"), ("BIND", "RELATE")
    }
    assert "source_action_ordinal" not in ir
    assert all("source_action_ordinal" not in edge for edge in ir["edges"])


def test_failed_gate_cannot_freeze_ir() -> None:
    collection = _collection("MUTATE", before_bound=True)
    collection["rows"] = collection["rows"][:1]
    gate = summarize_typed_source_gate([collection])
    assert gate["status"] == "SOURCE_TYPED_GATE_FAILED"
    with pytest.raises(ValueError):
        build_effect_ir([collection], gate)


def test_ir_lineage_uses_development_only() -> None:
    development = _collection("BIND", before_bound=False)
    heldout = _collection("BIND", before_bound=False)
    heldout["split"] = "heldout"
    heldout["seed"] = 1
    for row in heldout["rows"]:
        row["before_replay_state_sha256"] = "b" * 64
    gate = summarize_typed_source_gate([development, heldout])
    ir = build_effect_ir([development, heldout], gate)
    assert ir["induction_split"] == "development"
    assert ir["validation_splits"] == ["heldout"]
    assert ir["source_lineage"] == ["a" * 64]
