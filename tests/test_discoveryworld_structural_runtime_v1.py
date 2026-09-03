from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from motif_transfer.discoveryworld_structural_runtime_v1 import (
    choose_structural_action,
    target_commit_guard_satisfied,
)
from motif_transfer.target_structural_induction import (
    ADD_ENTITY_SLOT,
    ADD_OBSERVATION_RELATION,
    REMOVE_ENTITY_SLOT,
    export_mlp_grounder,
    induce_target_partial_order_program,
)


def _grounder():
    # Position/control, add, remove, observation. Commit always predicts remove.
    model = SimpleNamespace(
        coefs_=[np.zeros((26, 3)), np.zeros((3, 4))],
        intercepts_=[np.zeros(3), np.array([-10.0, -10.0, 10.0, -10.0])],
    )
    return export_mlp_grounder(model, threshold=0.5)


def _program():
    add = ADD_ENTITY_SLOT["operator_type_id"]
    obs = ADD_OBSERVATION_RELATION["operator_type_id"]
    remove = REMOVE_ENTITY_SLOT["operator_type_id"]
    return induce_target_partial_order_program(
        [(add, add, obs, obs, obs, remove)] * 2,
        development_receipts_sha256="dev",
        learned_target_guards=({
            "operator_type_id": remove,
            "target_relation_from_agent": "east",
            "target_distance": 1,
            "minimum_inventory_cardinality": 2,
        },),
    )


def _facts(relation="north", distance=2):
    return {
        "inventory": [{"uuid": 1}, {"uuid": 2}],
        "accessible_objects": [],
        "salient_relative_objects": [{
            "uuid": 9, "relation_from_agent": relation, "distance": distance,
        }],
        "agent_location": {"directions_you_can_move": ["north"]},
    }


def test_source_subgraph_uses_target_guard_and_permuted_abstains():
    add = ADD_ENTITY_SLOT["operator_type_id"]
    remove = REMOVE_ENTITY_SLOT["operator_type_id"]
    source = {"induced_sequence": [add, remove], "program_sha256": "source"}
    counts = {add: 2, ADD_OBSERVATION_RELATION["operator_type_id"]: 3}
    decision, _ = choose_structural_action(
        condition="source_induced", facts=_facts(), target_uuid=9,
        commit_action={"action": "DROP", "arg1": 2},
        position_action={"action": "TELEPORT_TO_OBJECT", "arg1": 9},
        grounder=_grounder(), target_program=_program(),
        source_program=source, prefix_counts=counts,
    )
    assert decision.action["action"] == "TELEPORT_TO_OBJECT"
    assert not target_commit_guard_satisfied(_program(), _facts(), target_uuid=9)

    wrong = {"induced_sequence": [add, "wrong", remove], "program_sha256": "wrong"}
    decision, _ = choose_structural_action(
        condition="source_permuted", facts=_facts(), target_uuid=9,
        commit_action={"action": "DROP", "arg1": 2},
        position_action={"action": "TELEPORT_TO_OBJECT", "arg1": 9},
        grounder=_grounder(), target_program=_program(),
        source_program=wrong, prefix_counts=counts,
    )
    assert decision.kind == "ABSTAIN"


def test_source_commits_only_after_learned_target_guard():
    add = ADD_ENTITY_SLOT["operator_type_id"]
    remove = REMOVE_ENTITY_SLOT["operator_type_id"]
    decision, _ = choose_structural_action(
        condition="source_induced", facts=_facts("east", 1), target_uuid=9,
        commit_action={"action": "DROP", "arg1": 2},
        position_action={"action": "TELEPORT_TO_OBJECT", "arg1": 9},
        grounder=_grounder(), target_program=_program(),
        source_program={"induced_sequence": [add, remove], "program_sha256": "source"},
        prefix_counts={add: 2, ADD_OBSERVATION_RELATION["operator_type_id"]: 3},
    )
    assert decision.action["action"] == "DROP"
