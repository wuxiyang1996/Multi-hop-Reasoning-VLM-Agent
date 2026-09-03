"""Lower AGQA supervision programs to operator-free semantic slot graphs.

This module is a target-side *training-data builder*.  Runtime parsers receive
question text only and emit the resulting shallow semantic representation.
The representation deliberately omits AGQA function names and transferable VM
operators; the Harness must plan executable operations from these constraints.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Mapping

from .agqa_layer_b_contracts import AGQASemanticSlotReceipt, SemanticSlotNode
from .agqa_typed_program import Array, Atom, Call, Expression, parse_program, serialize_program


_NON_ACTION_ANCHOR_LITERALS = frozenset({
    "before", "after", "while", "between", "forward", "backward",
    "video", "frame", "frames", "action", "actions", "relation",
    "relations", "object", "objects", "class", "start", "end",
    "temporal tag",
})


_SEMANTICS: Mapping[str, tuple[str, str]] = {
    "Query": ("QUERY_GOAL", "request an attribute of the selected item"),
    "OnlyItem": ("ORDINAL_CONSTRAINT", "require one unambiguous referenced item"),
    "Iterate": ("RELATION", "collect observations satisfying a relation"),
    "Localize": ("TEMPORAL_CONSTRAINT", "restrict observations to an anchored time window"),
    "Filter": ("RELATION", "match a typed relation description"),
    "Exists": ("QUERY_GOAL", "ask whether a grounded event or relation exists"),
    "HasItem": ("QUERY_GOAL", "ask whether a grounded set contains an item"),
    "IterateUntil": ("ORDINAL_CONSTRAINT", "select an endpoint in temporal order"),
    "ToAction": ("ACTION", "bind a referenced entity to its action occurrence"),
    "Choose": ("CHOICE", "select between explicitly named alternatives"),
    "Equals": ("LOGICAL_CONSTRAINT", "test semantic equality"),
    "Compare": ("DURATION_CONSTRAINT", "compare two grounded durations"),
    "XOR": ("LOGICAL_CONSTRAINT", "require exactly one condition"),
    "AND": ("LOGICAL_CONSTRAINT", "require both conditions"),
    "Superlative": ("ORDINAL_CONSTRAINT", "select an extremal grounded duration"),
    "Subtract": ("DURATION_CONSTRAINT", "form a relative duration quantity"),
}

_COMPACT_TAGS: Mapping[str, str] = {
    "Query": "goal", "OnlyItem": "single_reference", "Iterate": "observations",
    "Localize": "time_window", "Filter": "relation_description",
    "Exists": "presence_question", "HasItem": "membership_question",
    "IterateUntil": "ordered_endpoint", "ToAction": "action_reference",
    "Choose": "alternatives", "Equals": "equality_condition",
    "Compare": "duration_choice", "XOR": "exclusive_condition",
    "AND": "joint_condition", "Superlative": "duration_extremum",
    "Subtract": "relative_duration",
}
_COMPACT_SEMANTICS = {_COMPACT_TAGS[key]: value for key, value in _SEMANTICS.items()}


def answer_kind(expression: Expression) -> str:
    if not isinstance(expression, Call):
        return "ENTITY"
    if expression.function in {"Exists", "HasItem", "Equals", "XOR", "AND"}:
        return "BOOLEAN"
    if expression.function in {"Choose", "Compare"}:
        return "CHOICE"
    if expression.function == "Query" and expression.arguments:
        head = expression.arguments[0]
        if isinstance(head, Atom) and head.value.casefold() in {"action", "actions"}:
            return "ACTION"
        return "ENTITY"
    return "ENTITY"


def semantic_nodes(expression: Expression) -> tuple[tuple[SemanticSlotNode, ...], str]:
    """Return a post-order semantic graph and its root slot ID."""

    rows: list[SemanticSlotNode] = []

    def lower(node: Expression) -> str:
        if isinstance(node, Atom):
            slot_id = f"S{len(rows)}"
            rows.append(SemanticSlotNode(slot_id, "LITERAL", node.value))
            return slot_id
        if isinstance(node, Array):
            children = tuple(lower(child) for child in node.values)
            slot_id = f"S{len(rows)}"
            rows.append(SemanticSlotNode(
                slot_id, "RELATION", "ordered semantic tuple", children,
                (("tuple_arity", str(len(children))),),
            ))
            return slot_id
        children = tuple(lower(child) for child in node.arguments)
        try:
            kind, surface = _SEMANTICS[node.function]
        except KeyError as exc:
            raise ValueError(f"unsupported AGQA semantic function: {node.function}") from exc
        slot_id = f"S{len(rows)}"
        rows.append(SemanticSlotNode(
            slot_id, kind, surface, children,
            (("argument_count", str(len(children))),),
        ))
        return slot_id

    root = lower(expression)
    return tuple(rows), root


def semantic_supervision_target(program: str) -> dict[str, object]:
    """Create the operator-free target serialized for semantic-parser SFT."""

    expression = parse_program(program)
    slots, root = semantic_nodes(expression)
    return {
        "answer_kind": answer_kind(expression),
        "root_slot_id": root,
        "slots": [asdict(row) for row in slots],
        "functional_program_in_target": False,
        "operator_sequence_in_target": False,
    }


def serialize_semantic_target(program: str) -> str:
    return json.dumps(semantic_supervision_target(program), separators=(",", ":"), sort_keys=True)


def compact_semantic_expression(expression: Expression) -> Expression:
    if isinstance(expression, Atom):
        return expression
    if isinstance(expression, Array):
        return Call("semantic_tuple", tuple(compact_semantic_expression(x) for x in expression.values))
    try:
        tag = _COMPACT_TAGS[expression.function]
    except KeyError as exc:
        raise ValueError(f"unsupported AGQA semantic function: {expression.function}") from exc
    return Call(tag, tuple(compact_semantic_expression(x) for x in expression.arguments))


def serialize_compact_semantic_target(program: str) -> str:
    """Compact operator-free target suitable for seq2seq generation."""

    return serialize_program(compact_semantic_expression(parse_program(program)))


def _nodes_from_compact(expression: Expression) -> tuple[tuple[SemanticSlotNode, ...], str]:
    rows: list[SemanticSlotNode] = []

    def lower(node: Expression) -> str:
        if isinstance(node, Atom):
            slot_id = f"S{len(rows)}"; rows.append(SemanticSlotNode(slot_id, "LITERAL", node.value)); return slot_id
        if isinstance(node, Array):
            raise ValueError("compact semantic targets use semantic_tuple calls, not arrays")
        children = tuple(lower(child) for child in node.arguments)
        if node.function == "semantic_tuple":
            kind, surface = "RELATION", "ordered semantic tuple"
        else:
            try: kind, surface = _COMPACT_SEMANTICS[node.function]
            except KeyError as exc: raise ValueError(f"unknown compact semantic tag: {node.function}") from exc
        slot_id = f"S{len(rows)}"
        rows.append(SemanticSlotNode(slot_id, kind, surface, children,
                                     (("argument_count", str(len(children))),)))
        return slot_id

    root = lower(expression)
    return tuple(rows), root


def parse_compact_semantic_target(
    text: str, *, task_id: str, question_sha256: str, parser_sha256: str,
    parser_training_authority: str,
) -> AGQASemanticSlotReceipt:
    expression = parse_program(text)
    slots, root = _nodes_from_compact(expression)
    root_call = expression.function if isinstance(expression, Call) else ""
    answer_by_root = {
        "presence_question": "BOOLEAN", "membership_question": "BOOLEAN",
        "equality_condition": "BOOLEAN", "exclusive_condition": "BOOLEAN",
        "joint_condition": "BOOLEAN", "alternatives": "CHOICE", "duration_choice": "CHOICE",
        "goal": "ENTITY",
    }
    return AGQASemanticSlotReceipt.create(
        task_id=task_id, question_sha256=question_sha256,
        answer_kind=answer_by_root.get(root_call, "ENTITY"), root_slot_id=root,
        slots=slots, parser_sha256=parser_sha256,
        parser_training_authority=parser_training_authority,
    )


def parse_semantic_target(
    text: str, *, task_id: str, question_sha256: str, parser_sha256: str,
    parser_training_authority: str,
) -> AGQASemanticSlotReceipt:
    payload = json.loads(text)
    if payload.get("functional_program_in_target") is not False:
        raise ValueError("semantic parser target crossed the functional-program boundary")
    if payload.get("operator_sequence_in_target") is not False:
        raise ValueError("semantic parser target crossed the operator-sequence boundary")
    slots = tuple(SemanticSlotNode(
        slot_id=str(row["slot_id"]), kind=str(row["kind"]), surface=str(row["surface"]),
        children=tuple(str(value) for value in row.get("children", ())),
        attributes=tuple((str(k), str(v)) for k, v in row.get("attributes", ())),
    ) for row in payload["slots"])
    return AGQASemanticSlotReceipt.create(
        task_id=task_id, question_sha256=question_sha256,
        answer_kind=str(payload["answer_kind"]), root_slot_id=str(payload["root_slot_id"]),
        slots=slots, parser_sha256=parser_sha256,
        parser_training_authority=parser_training_authority,
    )


def action_anchor_obligations(
    semantic: AGQASemanticSlotReceipt,
) -> tuple[tuple[str, str], ...]:
    """Return exact action phrases required by an operator-free task graph.

    This is deliberately outcome-blind.  In particular, it handles both a
    scalar temporal anchor and ``between`` anchors represented by a nested
    ordered semantic tuple.
    """

    by_id = {slot.slot_id: slot for slot in semantic.slots}
    output: list[tuple[str, str]] = []

    def add(slot: SemanticSlotNode) -> None:
        phrase = slot.surface.casefold().strip()
        if (
            slot.kind == "LITERAL"
            and phrase not in _NON_ACTION_ANCHOR_LITERALS
            and phrase not in {existing for existing, _ in output}
        ):
            output.append((phrase, slot.slot_id))

    def add_literal_descendants(slot: SemanticSlotNode) -> None:
        if slot.kind == "LITERAL":
            add(slot)
            return
        for child_id in slot.children:
            add_literal_descendants(by_id[child_id])

    for slot in semantic.slots:
        if slot.kind == "TEMPORAL_CONSTRAINT" and len(slot.children) >= 2:
            add_literal_descendants(by_id[slot.children[1]])
        if slot.kind == "ACTION" and slot.children:
            add(by_id[slot.children[0]])
        if slot.kind == "RELATION" and slot.surface == "ordered semantic tuple":
            children = [by_id[child_id] for child_id in slot.children]
            if any(
                child.kind == "LITERAL"
                and child.surface.casefold().strip() == "actions"
                for child in children
            ):
                for child in children:
                    add(child)
    return tuple(output)


def relation_grounding_obligations(
    semantic: AGQASemanticSlotReceipt,
) -> tuple[tuple[str, str], ...]:
    """Extract explicit visual relation predicates from operator-free slots."""
    by_id = {slot.slot_id: slot for slot in semantic.slots}
    output: list[tuple[str, str]] = []
    categories = {"relation", "relations", "objects", "actions"}
    for slot in semantic.slots:
        if not (
            slot.kind == "RELATION"
            and slot.surface.startswith("match a typed relation description")
            and len(slot.children) >= 2
        ):
            continue
        relation_tuple = by_id[slot.children[1]]
        if relation_tuple.kind != "RELATION" or relation_tuple.surface != "ordered semantic tuple":
            continue
        literals = [by_id[child_id] for child_id in relation_tuple.children]
        if not literals or literals[0].kind != "LITERAL":
            continue
        category = literals[0].surface.casefold().strip()
        if category not in {"relation", "relations"} or len(literals) < 2:
            continue
        predicate = literals[1]
        phrase = predicate.surface.casefold().strip()
        if predicate.kind != "LITERAL" or phrase in categories or phrase in _NON_ACTION_ANCHOR_LITERALS:
            continue
        if phrase not in {existing for existing, _ in output}:
            output.append((phrase, predicate.slot_id))
    return tuple(output)


__all__ = [
    "action_anchor_obligations", "answer_kind", "compact_semantic_expression", "parse_compact_semantic_target",
    "parse_semantic_target", "semantic_nodes", "semantic_supervision_target",
    "relation_grounding_obligations", "serialize_compact_semantic_target", "serialize_semantic_target",
]
