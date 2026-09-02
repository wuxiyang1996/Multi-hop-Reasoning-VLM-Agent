"""Parse AGQA's public DSL and expose its target-independent VM obligations.

The parser is syntax-only.  It reads neither answers nor scene graphs.  The
lowering separates target-native graph access (``Filter``, ``Iterate``, and
literal binding) from transferable control operations.  In particular,
``Superlative(min, ...)`` is represented as ARGMAX over a target-native signed
score, so target vocabulary never changes the source capability inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterator, Sequence

from .contracts import stable_hash


@dataclass(frozen=True)
class Atom:
    value: str


@dataclass(frozen=True)
class Array:
    values: tuple["Expression", ...]


@dataclass(frozen=True)
class Call:
    function: str
    arguments: tuple["Expression", ...]


Expression = Atom | Array | Call


class ProgramSyntaxError(ValueError):
    pass


_DELIMITERS = frozenset("(),[]")


def _tokens(text: str) -> Iterator[str]:
    start = 0
    for index, char in enumerate(text):
        if char not in _DELIMITERS:
            continue
        atom = text[start:index].strip()
        if atom:
            yield atom
        yield char
        start = index + 1
    atom = text[start:].strip()
    if atom:
        yield atom


def parse_program(text: str) -> Expression:
    tokens = list(_tokens(str(text)))
    cursor = 0

    def expression() -> Expression:
        nonlocal cursor
        if cursor >= len(tokens):
            raise ProgramSyntaxError("unexpected end of program")
        token = tokens[cursor]
        if token == "[":
            cursor += 1
            values: list[Expression] = []
            while cursor < len(tokens) and tokens[cursor] != "]":
                values.append(expression())
                if cursor < len(tokens) and tokens[cursor] == ",":
                    cursor += 1
                elif cursor >= len(tokens) or tokens[cursor] != "]":
                    raise ProgramSyntaxError("array separator missing")
            if cursor >= len(tokens):
                raise ProgramSyntaxError("array not closed")
            cursor += 1
            return Array(tuple(values))
        if token in {",", ")", "]", "("}:
            raise ProgramSyntaxError(f"unexpected token: {token}")
        cursor += 1
        if cursor < len(tokens) and tokens[cursor] == "(":
            cursor += 1
            arguments: list[Expression] = []
            while cursor < len(tokens) and tokens[cursor] != ")":
                arguments.append(expression())
                if cursor < len(tokens) and tokens[cursor] == ",":
                    cursor += 1
                elif cursor >= len(tokens) or tokens[cursor] != ")":
                    raise ProgramSyntaxError("call separator missing")
            if cursor >= len(tokens):
                raise ProgramSyntaxError("call not closed")
            cursor += 1
            return Call(token, tuple(arguments))
        return Atom(token)

    result = expression()
    if cursor != len(tokens):
        raise ProgramSyntaxError(f"trailing token: {tokens[cursor]}")
    return result


def serialize_program(expression: Expression) -> str:
    if isinstance(expression, Atom):
        return expression.value
    if isinstance(expression, Array):
        return "[" + ", ".join(serialize_program(x) for x in expression.values) + "]"
    return expression.function + "(" + ", ".join(
        serialize_program(x) for x in expression.arguments
    ) + ")"


_DIRECT_VM = {
    "Choose": ("CHOOSE",),
    "Query": ("PROJECT",),
    "OnlyItem": ("UNIQUE",),
    "Exists": ("EXISTS",),
    "HasItem": ("EXISTS",),
    "Compare": ("COMPARE", "CHOOSE"),
    "Equals": ("COMPARE",),
    "XOR": ("XOR",),
    "AND": ("AND",),
    "Superlative": ("ARGMAX",),
    "Localize": ("INTERVAL_OF", "TEMPORAL_SELECT"),
    "ToAction": ("FILTER_EQ", "UNIQUE", "PROJECT"),
}


def required_vm_operators(expression: Expression) -> tuple[str, ...]:
    """Return source-controlled operations after target-native grounding."""

    result: set[str] = set()

    def visit(node: Expression) -> None:
        if isinstance(node, Atom):
            return
        if isinstance(node, Array):
            for child in node.values:
                visit(child)
            return
        result.update(_DIRECT_VM.get(node.function, ()))
        if node.function == "Filter" and len(node.arguments) == 2:
            query = node.arguments[1]
            if isinstance(query, Array) and len(query.values) > 1:
                result.add("FILTER_EQ")
        if node.function == "IterateUntil":
            direction = node.arguments[0] if node.arguments else None
            if isinstance(direction, Atom) and direction.value == "forward":
                result.add("FIRST")
            elif isinstance(direction, Atom) and direction.value == "backward":
                result.add("LAST")
            else:
                # Unknown order is unsafe, and neither direction may be guessed.
                result.update(("FIRST", "LAST"))
        for child in node.arguments:
            visit(child)

    visit(expression)
    return tuple(sorted(result))


def required_compositions(expression: Expression) -> tuple[tuple[str, str], ...]:
    """Return typed child-output -> parent-consumer composition edges."""

    edges: set[tuple[str, str]] = set()

    def walk(node: Expression) -> set[str]:
        if isinstance(node, Atom): return set()
        if isinstance(node, Array):
            output = set()
            for child in node.values: output.update(walk(child))
            return output
        child_outputs = [walk(child) for child in node.arguments]
        internal = list(_DIRECT_VM.get(node.function, ()))
        if node.function == "Filter" and len(node.arguments) == 2:
            query = node.arguments[1]
            if isinstance(query, Array) and len(query.values) > 1: internal = ["FILTER_EQ"]
        if node.function == "IterateUntil":
            direction = node.arguments[0] if node.arguments else None
            internal = ["FIRST" if isinstance(direction, Atom) and direction.value == "forward" else "LAST"]
        if internal:
            for left, right in zip(internal, internal[1:]): edges.add((left, right))
            consumer = internal[0]
            for outputs in child_outputs:
                for output in outputs: edges.add((output, consumer))
            return {internal[-1]}
        output = set()
        for values in child_outputs: output.update(values)
        return output

    walk(expression)
    return tuple(sorted(edges))


def compile_receipt(
    program: str, authorized_operators: Sequence[str],
    authorized_compositions: Sequence[Sequence[str]] | None = None,
) -> dict[str, object]:
    """Check exact capability coverage without executing or reading outcomes."""

    try:
        ast = parse_program(program)
        canonical = serialize_program(ast)
        required = required_vm_operators(ast)
        missing = sorted(set(required) - {str(x).upper() for x in authorized_operators})
        compositions = required_compositions(ast)
        allowed_edges = None if authorized_compositions is None else {
            (str(edge[0]).upper(), str(edge[1]).upper())
            for edge in authorized_compositions if len(edge) == 2
        }
        missing_edges = [] if allowed_edges is None else sorted(set(compositions) - allowed_edges)
        body: dict[str, object] = {
            "status": "COMPILED" if not missing and not missing_edges else "ABSTAINED",
            "canonical_program": canonical,
            "required_operators": list(required),
            "missing_operators": missing,
            "required_compositions": [list(edge) for edge in compositions],
            "missing_compositions": [list(edge) for edge in missing_edges],
            "answer_read": False,
            "scene_graph_read": False,
        }
    except ProgramSyntaxError as exc:
        body = {
            "status": "ABSTAINED", "canonical_program": None,
            "required_operators": [], "missing_operators": [],
            "required_compositions": [], "missing_compositions": [],
            "reason": f"PROGRAM_SYNTAX:{exc}", "answer_read": False,
            "scene_graph_read": False,
        }
    body["receipt_sha256"] = stable_hash(body)
    return body


__all__ = [
    "Array", "Atom", "Call", "Expression", "ProgramSyntaxError",
    "compile_receipt", "parse_program", "required_compositions", "required_vm_operators",
    "serialize_program",
]
