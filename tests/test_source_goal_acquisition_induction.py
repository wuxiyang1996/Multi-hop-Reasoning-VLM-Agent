from __future__ import annotations

import json
from pathlib import Path

import motif_transfer.source_goal_acquisition_induction as module


REPO = Path(__file__).resolve().parents[1]


def _real_inputs() -> tuple[dict, dict]:
    plan = json.loads((
        REPO / "runs/sokoban_effect_program_v2/fresh_confirmation_plan.json"
    ).read_text(encoding="utf-8"))
    relation = json.loads((
        REPO / "runs/sokoban_goal_relation_macro_v3/artifact.json"
    ).read_text(encoding="utf-8"))
    return plan, relation


def test_source_only_induction_learns_acquire_binding_relation_graph() -> None:
    plan, relation = _real_inputs()
    dataset = module.build_goal_acquisition_dataset(plan, relation)
    artifact = module.induce_goal_acquisition_program(dataset)
    module.validate_goal_acquisition_dataset(dataset)
    module.validate_goal_acquisition_program(artifact)
    program = artifact["program"]
    descriptors = {
        row["operator_type_id"]: row for row in artifact["operator_types"]
    }
    assert any(
        descriptors[type_id]["predicate_family"] == "CONTROL_STATE"
        for type_id in program["acquisition_operator_type_ids"]
    )
    assert descriptors[program["binding_operator_type_id"]][
        "predicate_family"
    ] == "POSITIVE_EFFECT_BINDING"
    assert descriptors[program["relation_operator_type_id"]][
        "predicate_family"
    ] == "ENTITY_GOAL_RELATION"
    assert program["relation_binding_cardinality"] == {
        "operator": "EQ", "value": 1,
    }
    assert program["binding_to_relation"]["discovery_precision"] == 1.0
    assert artifact["named_controller_template_used"] is False
    assert artifact["target_data_read"] is False


def test_previous_heldout_rejects_shuffled_effect_control() -> None:
    discovery_plan, relation = _real_inputs()
    heldout = json.loads((
        REPO / "configs/sokoban_goal_relation_macro_v3_frozen/fresh_source_plan.json"
    ).read_text(encoding="utf-8"))
    artifact = module.induce_goal_acquisition_program(
        module.build_goal_acquisition_dataset(discovery_plan, relation)
    )
    report = module.confirm_goal_acquisition_program(
        artifact, module.build_goal_acquisition_dataset(heldout, relation),
    )
    assert report["source_gate_passed"] is True
    metrics = report["metrics"]
    assert metrics["authentic_conformance_rate"] >= 0.95
    assert metrics["authentic_conformance_rate"] > (
        metrics["shuffled_conformance_rate"] + 0.20
    )
    assert metrics["shuffled_effect_bindings"] == 0
