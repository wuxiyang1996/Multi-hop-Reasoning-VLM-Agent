import pytest

from motif_transfer.agqa_typed_program import (
    ProgramSyntaxError, compile_receipt, parse_program,
    required_vm_operators, serialize_program,
)


PROGRAM = (
    "Choose(food, window, IterateUntil(backward, video, "
    "XOR(Exists(food, Filter(frame, [relations, behind, objects])), "
    "Exists(window, Filter(frame, [relations, behind, objects]))), "
    "Filter(frame, [relations, behind, objects])))"
)


def test_round_trip_nested_agqa_program():
    ast = parse_program(PROGRAM)
    assert serialize_program(ast) == PROGRAM
    assert set(required_vm_operators(ast)) == {
        "CHOOSE", "EXISTS", "FILTER_EQ", "LAST", "XOR",
    }


def test_superlative_min_uses_canonical_signed_argmax():
    ast = parse_program(
        "Superlative(min, [Filter(video, [actions, running somewhere]), "
        "Filter(video, [actions, holding clothes])], "
        "Subtract(Query(end, action), Query(start, action)))"
    )
    required = set(required_vm_operators(ast))
    assert "ARGMAX" in required
    assert "ARGMIN" not in required
    assert "PROJECT" in required


def test_capability_check_fails_closed():
    receipt = compile_receipt(PROGRAM, ("CHOOSE", "EXISTS", "FILTER_EQ", "XOR"))
    assert receipt["status"] == "ABSTAINED"
    assert receipt["missing_operators"] == ["LAST"]


@pytest.mark.parametrize("program", ["Query(class", "[a, b", "X(a,,b)"])
def test_malformed_program_rejected(program):
    with pytest.raises(ProgramSyntaxError):
        parse_program(program)
