"""Layer-B adapter for action literals nested in relation descriptions.

V2 correctly kept direct temporal anchors lexicalized by their visual verb, but
AGQA duration programs encode full action phrases inside
``semantic_tuple(actions, <action literal>)``.  This adapter discovers those
typed slots from the operator-free semantic tree and uses the grounded object
to disambiguate action literals that share a verb head.  It never reads an
answer, official program, or target outcome.
"""

from __future__ import annotations

from typing import Any, Sequence

from . import agqa_layer_b_executor as v1
from .agqa_layer_b_contracts import AGQASemanticSlotReceipt, RawVideoEventGraphReceipt
from .contracts import stable_hash


def _typed_full_action_slots(
    semantic: AGQASemanticSlotReceipt | None,
) -> frozenset[str]:
    if semantic is None:
        return frozenset()
    by_id = {row.slot_id: row for row in semantic.slots}
    slots = {
        row.children[0] for row in semantic.slots
        if row.kind == "ACTION" and len(row.children) >= 2
    }
    for row in semantic.slots:
        if row.kind != "RELATION" or len(row.children) < 2:
            continue
        children = [by_id.get(slot_id) for slot_id in row.children]
        if not any(child is not None and child.kind == "LITERAL"
                   and v1._clean(child.surface) == "actions" for child in children):
            continue
        slots.update(
            child.slot_id for child in children
            if child is not None and child.kind == "LITERAL"
            and v1._clean(child.surface) != "actions"
        )
    return frozenset(slots)


def _action_phrase(event, semantic, full_action_slots: frozenset[str]) -> str:
    if set(event.semantic_slot_ids) & full_action_slots:
        return v1._action_phrase(event, semantic)
    return v1._clean(event.predicate)


def event_graph_to_pseudo_stsg_v3(
    receipt: RawVideoEventGraphReceipt,
    semantic: AGQASemanticSlotReceipt | None = None,
) -> tuple[dict[str, Any], dict[str, str], str]:
    receipt.validate()
    graph: dict[str, Any] = {
        str(frame): {"type": "frame"}
        for frame in range(len(receipt.selected_frame_indices))
    }
    id_to_text: dict[str, str] = {}
    text_to_id: dict[str, str] = {}

    def symbol(prefix: str, text: str) -> str:
        value = v1._clean(text)
        key = f"{prefix}:{value}"
        if key not in text_to_id:
            identifier = f"{prefix}{len(text_to_id)}"
            text_to_id[key] = identifier
            id_to_text[identifier] = value
        return text_to_id[key]

    full_action_slots = _typed_full_action_slots(semantic)
    action_groups: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(receipt.events):
        bound_literals = v1._bound_literals(event, semantic)
        raw_predicate = v1._clean(event.predicate)
        predicate_candidates = [
            value for value in bound_literals
            if value in raw_predicate or raw_predicate in value
            or bool(v1._verb_forms(value) & v1._verb_forms(raw_predicate))
        ]
        predicate = min(predicate_candidates, key=len) if predicate_candidates else raw_predicate
        obj = v1._clean(event.object)
        predicate_id = symbol("p", predicate)
        object_id = symbol("o", obj) if obj else symbol("o", "none")
        phrase = _action_phrase(event, semantic, full_action_slots)
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
                "type": "relation", "class": predicate_id,
                "objects": [object_ref], "confidence": event.confidence,
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
        "adapter": "PIXEL_EVENTS_TO_SHARED_AGQA_TYPED_VM_V3_DESCRIPTION_ACTION_TYPED",
    })
    return graph, id_to_text, graph_sha


def execute_layer_b_semantics_v3(
    *, compact_semantics: str, grounding: RawVideoEventGraphReceipt,
    semantic: AGQASemanticSlotReceipt | None = None,
    authorized_operators: Sequence[str],
    authorized_compositions: Sequence[Sequence[str]] | None = None,
    ambiguity_policy: str = "STRICT",
) -> v1.LayerBExecutionResult:
    program = v1.semantic_program_from_compact(compact_semantics)
    graph, id_to_text, graph_sha = event_graph_to_pseudo_stsg_v3(grounding, semantic)
    executor = v1.AGQATypedSTSGExecutor(
        graph=graph, id_to_text=id_to_text, graph_sha256=graph_sha,
        authorized_operators=authorized_operators,
        authorized_compositions=authorized_compositions,
        ambiguity_policy=ambiguity_policy,
    )
    result = executor.execute(
        program, functional_program_source="PREDICTED_OPERATOR_FREE_SEMANTICS",
    )
    return v1.LayerBExecutionResult(
        receipt=result,
        semantic_program_sha256=stable_hash(program),
        raw_event_graph_receipt_sha256=grounding.receipt_sha256,
        adapter_sha256=stable_hash(
            "PIXEL_EVENTS_TO_SHARED_AGQA_TYPED_VM_V3_DESCRIPTION_ACTION_TYPED"
        ),
    )


__all__ = [
    "_typed_full_action_slots", "event_graph_to_pseudo_stsg_v3",
    "execute_layer_b_semantics_v3",
]
