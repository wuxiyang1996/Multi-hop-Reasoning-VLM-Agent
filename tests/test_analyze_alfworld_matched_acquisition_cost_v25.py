from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/analyze_alfworld_matched_acquisition_cost_v25.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase14_v25", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_source_prefix_first_qualifies_at_four_episodes() -> None:
    module = _load_module()
    config = {
        "source_discovery_acquisition": (
            "runs/sokoban_goal_acquisition_v1/"
            "discovery_acquisition_interventions.json"
        ),
        "source_fresh_acquisition": (
            "runs/sokoban_goal_acquisition_v1/"
            "fresh_acquisition_interventions.json"
        ),
        "source_discovery_relation": (
            "runs/sokoban_goal_relation_macro_v3/"
            "discovery_macro_interventions.json"
        ),
        "source_fresh_relation": (
            "runs/sokoban_goal_relation_macro_v3/"
            "fresh_macro_interventions.json"
        ),
        "source_budget": {
            "maximum_episodes": 4,
            "primary_order_namespace": (
                "phase14-matched-acquisition-order-v1"
            ),
            "robustness_repetitions": 0,
            "robustness_namespace_prefix": (
                "phase14-matched-acquisition-order-v1/repeat/"
            ),
        },
    }

    result, _ = module.source_analysis(config)
    first = result["primary_first_qualified"]

    assert first is not None
    assert first["complete_source_intervention_episodes"] == 4
    assert first["source_target_data_read"] is False
    assert first["terminal_candidate_authority"][
        "named_terminal_feature_provided"
    ] is False
    assert first["heldout"]["relation_gate_passed"] is True
    assert first["heldout"]["acquisition_gate_passed"] is True
    assert first["heldout"]["relation_shuffled_effect_bindings"] == 0
    assert first["heldout"]["acquisition_shuffled_effect_bindings"] == 0


def test_source_structural_normal_form_excludes_target_adapter_rules() -> None:
    module = _load_module()
    assert "relation_argument_rule" not in module.source_structural_normal_form(
        {
            "operator_types": [
                {
                    "operator_type_id": "control",
                    "predicate_family": "CONTROL_STATE",
                }
            ],
            "program": {
                "acquisition_operator_type_ids": ["control"],
                "binding_operator_type_id": "bind",
                "relation_operator_type_id": "relate",
                "transition_graph": [{
                    "from_operator_type_id": "control",
                    "to_operator_type_id": "control",
                }],
                "binding_to_relation": {
                    "from_operator_type_id": "bind",
                    "to_operator_type_id": "relate",
                },
                "relation_binding_cardinality": {"value": 1},
                "abstention_rule": {
                    "binding_cardinality_above_induced_value": "ABSTAIN",
                },
            },
        },
        {
            "program": {
                "transitions": [{
                    "from_operator_type_id": "relate",
                    "to_operator_type_id": "relate",
                    "cardinality": "ONE_OR_MORE",
                }],
                "terminal_predicates": [{
                    "predicate_family": "ENTITY_GOAL_RELATION",
                    "operator": "EQ",
                    "value": 1.0,
                }],
                "abstention_rule": {
                    "multiple_target_bindings": "ABSTAIN",
                },
            }
        },
    )
