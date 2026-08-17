from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from motif_transfer.target_structural_induction import (
    ADD_ENTITY_SLOT,
    ADD_OBSERVATION_RELATION,
    REMOVE_ENTITY_SLOT,
    discoveryworld_core_sequence,
    export_mlp_grounder,
    grounded_operator_ids,
    induce_target_partial_order_program,
    source_sequence_support,
    target_program_supports,
)


def _facts(*, x=0, inventory=(), message=""):
    return {
        "agent_location": {"x": x, "y": 0, "directions_you_can_move": []},
        "inventory": [{"uuid": value} for value in inventory],
        "accessible_objects": [], "salient_relative_objects": [],
        "last_action_message": message,
    }


def _step(action, before, after):
    return {
        "action": action,
        "before_target_native_facts": before,
        "after_target_native_facts": after,
    }


def test_target_sequence_retains_domain_specific_observation_multiplicity():
    steps = [
        _step({"action": "PICKUP", "arg1": 7}, _facts(), _facts(inventory=(7,))),
        _step({"action": "USE"}, _facts(inventory=(7,)), _facts(
            inventory=(7,), message="You use it to investigate the alpha. Protein A: 1",
        )),
        _step({"action": "USE"}, _facts(inventory=(7,)), _facts(
            inventory=(7,), message="You use it to investigate the beta. Protein A: 2",
        )),
        _step({"action": "USE"}, _facts(inventory=(7,)), _facts(
            inventory=(7,), message="You use it to investigate the beta. Protein A: 2",
        )),
        _step({"action": "DROP", "arg1": 7}, _facts(inventory=(7,)), _facts()),
    ]
    assert discoveryworld_core_sequence(steps) == (
        ADD_ENTITY_SLOT["operator_type_id"],
        ADD_OBSERVATION_RELATION["operator_type_id"],
        ADD_OBSERVATION_RELATION["operator_type_id"],
        REMOVE_ENTITY_SLOT["operator_type_id"],
    )


def test_target_program_is_partial_order_extension_of_source_motif():
    add = ADD_ENTITY_SLOT["operator_type_id"]
    observe = ADD_OBSERVATION_RELATION["operator_type_id"]
    remove = REMOVE_ENTITY_SLOT["operator_type_id"]
    paths = [(add, observe, observe, observe, add, remove)] * 2
    program = induce_target_partial_order_program(
        paths, development_receipts_sha256="dev",
    )
    assert target_program_supports(program, paths[0])
    assert source_sequence_support((add, remove), paths[0])
    assert not source_sequence_support((add, "missing"), paths[0])
    assert program["source_program_copied_as_target_body"] is False
    assert {row["operator_type_id"]: row["minimum_count"] for row in program[
        "operator_requirements"
    ]}[observe] == 3


def test_exported_neural_grounder_round_trip():
    feature_count = 26
    output_count = 4
    model = SimpleNamespace(
        coefs_=[np.zeros((feature_count, 3)), np.zeros((3, output_count))],
        intercepts_=[np.zeros(3), np.array([-10.0, 10.0, -10.0, -10.0])],
    )
    artifact = export_mlp_grounder(model, threshold=0.5)
    step = _step({"action": "PICKUP", "arg1": 7}, _facts(), _facts(inventory=(7,)))
    assert grounded_operator_ids(artifact, step) == (
        ADD_ENTITY_SLOT["operator_type_id"],
    )
