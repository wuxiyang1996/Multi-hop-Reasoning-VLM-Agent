"""Adapter from frozen raw-video events and semantic slots to the shared AGQA VM."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from .agqa_layer_b_contracts import (
    AGQASemanticSlotReceipt, GroundedEvent, RawVideoEventGraphReceipt,
)
from .agqa_stsg_typed_executor import AGQATypedSTSGExecutor, STSGExecutionReceipt
from .agqa_typed_program import Array, Atom, Call, Expression, parse_program, serialize_program
from .contracts import stable_hash


_TAG_TO_FUNCTION = {
    "goal": "Query", "single_reference": "OnlyItem", "observations": "Iterate",
    "time_window": "Localize", "relation_description": "Filter",
    "presence_question": "Exists", "membership_question": "HasItem",
    "ordered_endpoint": "IterateUntil", "action_reference": "ToAction",
    "alternatives": "Choose", "equality_condition": "Equals",
    "duration_choice": "Compare", "exclusive_condition": "XOR",
    "joint_condition": "AND", "duration_extremum": "Superlative",
    "relative_duration": "Subtract",
}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).replace("_", " ").strip().casefold())


def _verb_forms(value: str) -> frozenset[str]:
    """Small deterministic inflection normalizer for grounded action heads."""
    word = _clean(value).split(maxsplit=1)[0] if _clean(value) else ""
    forms = {word}
    if word.endswith("ing") and len(word) > 4:
        stem = word[:-3]
        forms.update({stem, stem + "e"})
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            forms.add(stem[:-1])
    return frozenset(form for form in forms if form)


def semantic_program_from_compact(text: str) -> str:
    """Compile parser semantics to the legacy DSL without reading an official program."""

    def lift(node: Expression) -> Expression:
        if isinstance(node, Atom):
            return node
        if isinstance(node, Array):
            raise ValueError("compact semantic parser unexpectedly emitted an array")
        children = tuple(lift(child) for child in node.arguments)
        if node.function == "semantic_tuple":
            return Array(children)
        try:
            name = _TAG_TO_FUNCTION[node.function]
        except KeyError as exc:
            raise ValueError(f"unknown operator-free semantic tag: {node.function}") from exc
        return Call(name, children)

    return serialize_program(lift(parse_program(text)))


def _bound_literals(
    event: GroundedEvent, semantic: AGQASemanticSlotReceipt | None,
) -> tuple[str, ...]:
    if semantic is None:
        return ()
    by_id = {slot.slot_id: slot for slot in semantic.slots}
    return tuple(
        _clean(by_id[slot_id].surface) for slot_id in event.semantic_slot_ids
        if slot_id in by_id and by_id[slot_id].kind == "LITERAL"
    )


def _action_phrase(
    event: GroundedEvent, semantic: AGQASemanticSlotReceipt | None = None,
) -> str:
    predicate, obj = _clean(event.predicate), _clean(event.object)
    candidates = [value for value in _bound_literals(event, semantic)
                  if predicate in value and (not obj or obj in value)]
    if candidates:
        return max(candidates, key=len)
    # Target parsers may lexicalize a visual verb with arguments inserted
    # between its head and object ("putting some food somewhere" versus a
    # grounder event ``predicate=putting down, object=food``).  The bound slot
    # is authoritative for target phrasing; require both the same verb head
    # and the grounded object so unrelated actions cannot be substituted.
    head_forms = _verb_forms(predicate)
    bound_head_matches = [value for value in _bound_literals(event, semantic)
                          if head_forms & _verb_forms(value) and obj and obj in value]
    if bound_head_matches:
        return max(bound_head_matches, key=len)
    if not obj or obj in {"none", "n/a", "person", "someone"}:
        return predicate
    if obj in predicate:
        return predicate
    return f"{predicate} {obj}"


def event_graph_to_pseudo_stsg(
    receipt: RawVideoEventGraphReceipt,
    semantic: AGQASemanticSlotReceipt | None = None,
) -> tuple[dict[str, Any], dict[str, str], str]:
    """Expose pixel-grounded events through the target-independent typed executor API."""

    receipt.validate()
    graph: dict[str, Any] = {
        str(frame): {"type": "frame"} for frame in range(len(receipt.selected_frame_indices))
    }
    id_to_text: dict[str, str] = {}
    text_to_id: dict[str, str] = {}

    def symbol(prefix: str, text: str) -> str:
        value = _clean(text)
        key = f"{prefix}:{value}"
        if key not in text_to_id:
            identifier = f"{prefix}{len(text_to_id)}"
            text_to_id[key] = identifier
            id_to_text[identifier] = value
        return text_to_id[key]

    action_groups: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(receipt.events):
        bound_literals = _bound_literals(event, semantic)
        raw_predicate = _clean(event.predicate)
        predicate_candidates = [
            value for value in bound_literals
            if value in raw_predicate
            or raw_predicate in value
            or bool(_verb_forms(value) & _verb_forms(raw_predicate))
        ]
        predicate = min(predicate_candidates, key=len) if predicate_candidates else raw_predicate
        obj = _clean(event.object)
        predicate_id = symbol("p", predicate)
        object_id = symbol("o", obj) if obj else symbol("o", "none")
        phrase = _action_phrase(event, semantic)
        action = action_groups.setdefault(phrase, {
            "type": "action", "phrase": phrase, "all_f": set(),
            "verb_id": predicate_id, "object_id": object_id,
            "confidence": event.confidence, "semantic_slot_ids": set(),
        })
        action["all_f"].update(range(event.start_frame, event.end_frame + 1))
        action["confidence"] = max(float(action["confidence"]), event.confidence)
        action["semantic_slot_ids"].update(event.semantic_slot_ids)
        for frame in range(event.start_frame, event.end_frame + 1):
            object_ref = f"{object_id}/{frame:06d}"
            graph[object_ref] = {"type": "object", "class": object_id}
            graph[f"r{index}/{frame:06d}"] = {
                "type": "relation", "class": predicate_id, "objects": [object_ref],
                "confidence": event.confidence,
                "semantic_slot_ids": list(event.semantic_slot_ids),
            }
    for action_index, phrase in enumerate(sorted(action_groups)):
        action = action_groups[phrase]
        graph[f"a{action_index}"] = {
            **action, "all_f": sorted(action["all_f"]),
            "semantic_slot_ids": sorted(action["semantic_slot_ids"]),
        }
    graph_sha = stable_hash({
        "raw_event_graph_receipt_sha256": receipt.receipt_sha256,
        "adapter": "PIXEL_EVENTS_TO_SHARED_AGQA_TYPED_VM_V1",
    })
    return graph, id_to_text, graph_sha


@dataclass(frozen=True)
class LayerBExecutionResult:
    receipt: STSGExecutionReceipt
    semantic_program_sha256: str
    raw_event_graph_receipt_sha256: str
    adapter_sha256: str


def execute_layer_b_semantics(
    *, compact_semantics: str, grounding: RawVideoEventGraphReceipt,
    semantic: AGQASemanticSlotReceipt | None = None,
    authorized_operators: Sequence[str],
    authorized_compositions: Sequence[Sequence[str]] | None = None,
    ambiguity_policy: str = "STRICT",
) -> LayerBExecutionResult:
    program = semantic_program_from_compact(compact_semantics)
    graph, id_to_text, graph_sha = event_graph_to_pseudo_stsg(grounding, semantic)
    executor = AGQATypedSTSGExecutor(
        graph=graph, id_to_text=id_to_text, graph_sha256=graph_sha,
        authorized_operators=authorized_operators,
        authorized_compositions=authorized_compositions,
        ambiguity_policy=ambiguity_policy,
    )
    result = executor.execute(program, functional_program_source="PREDICTED_OPERATOR_FREE_SEMANTICS")
    return LayerBExecutionResult(
        receipt=result, semantic_program_sha256=stable_hash(program),
        raw_event_graph_receipt_sha256=grounding.receipt_sha256,
        adapter_sha256=stable_hash("PIXEL_EVENTS_TO_SHARED_AGQA_TYPED_VM_V1"),
    )


__all__ = [
    "LayerBExecutionResult", "event_graph_to_pseudo_stsg",
    "execute_layer_b_semantics", "semantic_program_from_compact",
]
