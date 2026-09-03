import pytest

from motif_transfer.typed_operator_vm import (
    TypedValue, ValueKind, VMAbstention, execute_operator, execute_program,
)


EVENTS = TypedValue(ValueKind.EVENT_SET, (
    {"id": "e1", "kind": "open", "start": 1, "end": 2, "duration": 1},
    {"id": "e2", "kind": "hold", "start": 4, "end": 8, "duration": 4},
    {"id": "e3", "kind": "open", "start": 10, "end": 12, "duration": 2},
))


def test_temporal_filter_unique_and_projection_compose():
    program = {
        "nodes": [
            {"id": "opens", "op": "FILTER_EQ", "args": ["events", "kind"],
             "params": {"field": "kind"}},
            {"id": "late", "op": "TEMPORAL_SELECT", "args": ["opens", "window"],
             "params": {"relation": "AFTER"}},
            {"id": "one", "op": "UNIQUE", "args": ["late"]},
            {"id": "answer", "op": "PROJECT", "args": ["one"],
             "params": {"field": "id", "output_kind": "SYMBOL"}},
        ],
        "output": "answer",
    }
    receipt = execute_program(program, {
        "events": EVENTS,
        "kind": TypedValue(ValueKind.SYMBOL, "open"),
        "window": TypedValue(ValueKind.INTERVAL, (3, 5)),
    }, authorized_operators=("FILTER_EQ", "TEMPORAL_SELECT", "UNIQUE", "PROJECT"))
    assert receipt.status == "COMMITTED"
    assert receipt.output == TypedValue(ValueKind.SYMBOL, "e3")


def test_boolean_choice_and_comparison_are_typed():
    greater = execute_operator("COMPARE", [
        TypedValue(ValueKind.NUMBER, 4), TypedValue(ValueKind.NUMBER, 2),
    ], {"relation": "GT"})
    both = execute_operator("AND", [greater, TypedValue(ValueKind.BOOLEAN, True)])
    answer = execute_operator("CHOOSE", [
        both, TypedValue(ValueKind.SYMBOL, "longer"), TypedValue(ValueKind.SYMBOL, "shorter"),
    ])
    assert answer.value == "longer"


def test_first_last_and_arg_extrema_fail_closed_on_ties():
    assert execute_operator("FIRST", [EVENTS]).value["id"] == "e1"
    assert execute_operator("LAST", [EVENTS]).value["id"] == "e3"
    assert execute_operator("ARGMAX", [EVENTS], {"field": "duration"}).value["id"] == "e2"
    tied = TypedValue(ValueKind.ENTITY_SET, ({"id": "a", "score": 1}, {"id": "b", "score": 1}))
    with pytest.raises(VMAbstention, match="ARGMAX_NOT_UNIQUE"):
        execute_operator("ARGMAX", [tied], {"field": "score"})


def test_source_capability_gate_prevents_target_compiler_from_authorizing_ops():
    program = {
        "nodes": [{"id": "answer", "op": "EXISTS", "args": ["events"]}],
        "output": "answer",
    }
    receipt = execute_program(program, {"events": EVENTS}, authorized_operators=())
    assert receipt.status == "ABSTAINED"
    assert receipt.abstention_reason == "SOURCE_OPERATOR_NOT_AUTHORIZED:EXISTS"


def test_invalid_type_abstains_without_partial_answer():
    program = {
        "nodes": [{"id": "answer", "op": "NOT", "args": ["value"]}],
        "output": "answer",
    }
    receipt = execute_program(
        program, {"value": TypedValue(ValueKind.NUMBER, 1)}, authorized_operators=("NOT",),
    )
    assert receipt.status == "ABSTAINED"
    assert receipt.output is None
    assert "TYPE_MISMATCH" in receipt.abstention_reason
