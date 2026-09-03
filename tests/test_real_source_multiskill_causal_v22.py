from __future__ import annotations

from motif_transfer.real_source_multiskill_causal_v22 import (
    ACTION_FEATURE_NAMES,
    RECEIPT_BY_EFFECT,
    UTILITY_FEATURE_NAMES,
    action_causal_features,
    utility_features,
)
from motif_transfer.relation_edge_value_v13 import FEATURE_NAMES


def _ledger(property_name: str = "CLEAN") -> dict[str, object]:
    return {
        "goal_spec": {
            "goal_object_type": "mug",
            "target_receptacle_type": "cabinet",
            "required_property": property_name,
            "required_count": 1,
        },
        "completed_objects": [],
    }


def test_effect_conditioning_changes_only_typed_coordinates() -> None:
    common = {
        "action": "clean mug 1 with sinkbasin 1",
        "grounded_scores": {
            "policy": 0.8,
            "completion": 0.9,
            "binding": 0.95,
            "applicability": 0.85,
        },
        "ledger": _ledger(),
        "history": (),
        "step": 12,
        "max_steps": 60,
        "required_property": "CLEAN",
    }
    mutate = action_causal_features(requested_effect="MUTATE", **common)
    relate = action_causal_features(requested_effect="RELATE", **common)
    assert tuple(mutate) == ACTION_FEATURE_NAMES
    assert mutate["requested_mutate"] == 1.0
    assert mutate["requested_relate"] == 0.0
    assert relate["requested_mutate"] == 0.0
    assert relate["requested_relate"] == 1.0
    assert mutate["required_clean"] == relate["required_clean"] == 1.0


def test_utility_features_include_symbol_and_neural_grounding() -> None:
    source = action_causal_features(
        action="clean mug 1 with sinkbasin 1",
        grounded_scores={
            "policy": 0.8, "completion": 0.9, "binding": 0.95,
            "applicability": 0.85,
        },
        ledger=_ledger(), history=(), step=12, max_steps=60,
        requested_effect="MUTATE", required_property="CLEAN",
    )
    fallback = action_causal_features(
        action="close cabinet 1",
        grounded_scores={
            "policy": 0.9, "completion": 0.2, "binding": 0.1,
            "applicability": 0.3,
        },
        ledger=_ledger(), history=(), step=12, max_steps=60,
        requested_effect="MUTATE", required_property="CLEAN",
    )
    base = {name: 0.0 for name in FEATURE_NAMES}
    values = utility_features(
        base_features=base,
        requested_effect="MUTATE",
        required_property="CLEAN",
        source_effect_probability=0.9,
        fallback_effect_probability=0.1,
        source_action_features=source,
        fallback_action_features=fallback,
    )
    assert tuple(values) == UTILITY_FEATURE_NAMES
    assert values["causal_effect_margin"] == 0.8
    assert values["requested_mutate"] == 1.0
    assert values["required_clean"] == 1.0


def test_receipts_are_typed_not_generic_success_labels() -> None:
    assert RECEIPT_BY_EFFECT == {
        "BIND": "BIND_INSTANCE",
        "MUTATE": "MUTATE_REQUIRED_PROPERTY",
        "RELATE": "RELATE_SLOT_CLOSED",
    }
