from __future__ import annotations

from motif_transfer.alfworld_structural_induction import ADD_ID, REMOVE_ID, UPDATE_ID
from motif_transfer.alfworld_structural_runtime_v1 import ALFWorldStructuralSelector


def _row(add=0.01, remove=0.01, update=0.01, entity=0.01, destination=0.01, behavior=0.1):
    return {
        "operator_probabilities": {ADD_ID: add, REMOVE_ID: remove, UPDATE_ID: update},
        "entity_binding_probability": entity,
        "destination_binding_probability": destination,
        "behavior_probability": behavior,
    }


def test_source_sequence_uses_neural_entity_and_destination_bindings() -> None:
    selector = ALFWorldStructuralSelector(
        condition="source_induced",
        target_sequence=(ADD_ID, REMOVE_ID, ADD_ID, REMOVE_ID),
        source_sequence=(ADD_ID, REMOVE_ID), threshold=0.5,
    )
    rows = {
        "take mug 1": _row(add=.99, entity=.95, behavior=.2),
        "take pillow 1": _row(add=.99, entity=.1, behavior=.9),
        "look": _row(behavior=.5),
    }
    decision = selector.select(rows=rows, history=(), goal="put two mug in cabinet.")
    assert decision["action"] == "take mug 1"
    receipt = selector.observe_transition(after_observation="You pick up the mug 1.")
    assert receipt["advanced"]
    rows = {
        "move mug 1 to cabinet 1": _row(remove=.99, entity=.95, destination=.9),
        "move mug 1 to desk 1": _row(remove=.99, entity=.95, destination=.1, behavior=.9),
    }
    assert selector.select(
        rows=rows, history=("take mug 1",), goal="put two mug in cabinet.",
    )["action"].endswith("cabinet 1")


def test_failed_transition_does_not_advance_symbolic_cursor() -> None:
    selector = ALFWorldStructuralSelector(
        condition="source_permuted", target_sequence=(ADD_ID, REMOVE_ID) * 2,
        source_sequence=(ADD_ID, UPDATE_ID, REMOVE_ID, ADD_ID), threshold=.5,
    )
    rows = {"take mug": _row(add=.9, entity=.9)}
    selector.select(rows=rows, history=(), goal="put two mug in cabinet.")
    receipt = selector.observe_transition(after_observation="Nothing happens.")
    assert not receipt["advanced"]
    assert selector.cursor == 0


def test_missing_source_binding_abstains_to_nonconflicting_search() -> None:
    selector = ALFWorldStructuralSelector(
        condition="source_induced", target_sequence=(ADD_ID, REMOVE_ID) * 2,
        source_sequence=(ADD_ID, REMOVE_ID), threshold=.5,
    )
    rows = {
        "go to desk 1": _row(behavior=.8),
        "take pillow 1": _row(add=.99, entity=.1, behavior=.95),
    }
    decision = selector.select(rows=rows, history=(), goal="put two mug in cabinet.")
    assert decision["action"] == "take pillow 1"
    assert not decision["source_admitted"]
