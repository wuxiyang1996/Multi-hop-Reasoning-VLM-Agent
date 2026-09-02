"""Five-arm Harness planning over frozen AGQA Layer-B semantic slots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .agqa_layer_b_contracts import AGQASemanticSlotReceipt, SemanticSlotNode
from .contracts import stable_hash


ARMS = (
    "neural_only", "generic_scaffold", "source_permuted",
    "source_induced", "target_written_isomorphic",
)


def source_permuted_compositions(
    operators: Sequence[str], compositions: Sequence[Sequence[str]],
) -> tuple[tuple[str, str], ...]:
    """Fixed derangement of source effect/composition lineage.

    The control preserves the exact operator inventory and edge count.  It
    changes only which measured source effect is bound to which operator node.
    """
    ordered = tuple(sorted({str(value) for value in operators}))
    if len(ordered) < 2:
        raise ValueError("source permutation needs at least two operators")
    mapping = {value: ordered[(index + 1) % len(ordered)]
               for index, value in enumerate(ordered)}
    return tuple(sorted({
        (mapping[str(edge[0])], mapping[str(edge[1])])
        for edge in compositions if len(edge) == 2
    }))


def _node_ops(node: SemanticSlotNode, answer_kind: str) -> tuple[str, ...]:
    surface = node.surface
    if node.kind == "QUERY_GOAL":
        return ("PROJECT",) if surface.startswith("request an attribute") else ("EXISTS",)
    if node.kind == "ORDINAL_CONSTRAINT":
        if surface.startswith("require one unambiguous"): return ("UNIQUE",)
        if surface.startswith("select an endpoint"): return ()
        if surface.startswith("select an extremal"): return ("ARGMAX",)
    if node.kind == "TEMPORAL_CONSTRAINT": return ("INTERVAL_OF", "TEMPORAL_SELECT")
    if node.kind == "CHOICE": return ("CHOOSE",)
    if node.kind == "DURATION_CONSTRAINT":
        return ("COMPARE", "CHOOSE") if surface.startswith("compare two") else ()
    if node.kind == "LOGICAL_CONSTRAINT":
        if surface.startswith("require exactly one"): return ("XOR",)
        if surface.startswith("require both"): return ("AND",)
        # Source EFFECT_RANKING evidence authorizes ordered score/duration
        # comparison, not semantic identity.  Keep the types separate so a
        # target equality question cannot borrow an unrelated game operator.
        if surface.startswith("test semantic equality"): return ("SEMANTIC_EQUALS",)
    if node.kind == "ACTION": return ("FILTER_EQ", "UNIQUE", "PROJECT")
    if node.kind == "RELATION" and surface.startswith("match a typed relation"):
        return ("FILTER_EQ",)
    return ()


def semantic_operator_plan(receipt: AGQASemanticSlotReceipt) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Plan VM obligations from semantics, not from an AGQA functional program."""

    receipt.validate(); by_id={row.slot_id:row for row in receipt.slots}; edges=set(); operators=set()

    def walk(slot_id: str) -> set[str]:
        node=by_id[slot_id]; child_outputs=[walk(child) for child in node.children]
        if node.kind == "ORDINAL_CONSTRAINT" and node.surface.startswith("select an endpoint"):
            direction = by_id[node.children[0]].surface.casefold() if node.children else ""
            if direction == "forward": internal=["FIRST"]
            elif direction == "backward": internal=["LAST"]
            else:
                # Unknown order is not guessed.  Preserve both obligations so
                # the runtime can fail closed before selecting an endpoint.
                operators.update(("FIRST","LAST"))
                return {"FIRST","LAST"}
        else:
            internal=list(_node_ops(node,receipt.answer_kind))
        operators.update(internal)
        if internal:
            for left,right in zip(internal,internal[1:]): edges.add((left,right))
            for outputs in child_outputs:
                for output in outputs: edges.add((output,internal[0]))
            return {internal[-1]}
        output=set()
        for values in child_outputs: output.update(values)
        return output

    walk(receipt.root_slot_id)
    return tuple(sorted(operators)), tuple(sorted(edges))


@dataclass(frozen=True)
class LayerBHarnessPlan:
    task_id: str
    arm: str
    status: str
    required_operators: tuple[str, ...]
    required_compositions: tuple[tuple[str, str], ...]
    missing_operators: tuple[str, ...]
    missing_compositions: tuple[tuple[str, str], ...]
    commit_policy: str
    semantic_receipt_sha256: str
    source_capability_sha256: str | None
    target_outcome_read: bool
    plan_sha256: str


def plan_harness_arm(
    receipt: AGQASemanticSlotReceipt, *, arm: str,
    source_capabilities: Mapping[str, object], all_vm_operators: Sequence[str],
) -> LayerBHarnessPlan:
    if arm not in ARMS: raise ValueError("unknown Layer-B Harness arm")
    required, compositions=semantic_operator_plan(receipt)
    source_ops={str(x) for x in source_capabilities["authorized_operators"]}
    source_edges={tuple(str(v) for v in edge) for edge in source_capabilities["authorized_compositions"]}
    if arm == "neural_only":
        allowed_ops=set(); allowed_edges=set(); policy="NEURAL_FALLBACK_ONLY"
    elif arm == "source_permuted":
        # Matched-complexity causal control: same primitives and same number of
        # source graph edges, but a fixed derangement breaks effect lineage.
        allowed_ops=source_ops
        allowed_edges=set(source_permuted_compositions(
            sorted(source_ops), sorted(source_edges),
        ))
        policy="FIXED_DERANGEMENT_OF_SOURCE_EFFECT_COMPOSITION_LINEAGE"
    elif arm == "generic_scaffold":
        # Generic symbolic computation is deliberately not crippled.  It has
        # every VM primitive/composition but lacks intervention-derived safety.
        allowed_ops={str(x) for x in all_vm_operators}
        allowed_edges={(left,right) for left in allowed_ops for right in allowed_ops}
        policy="EAGER_COMMIT_WITHOUT_INTERVENTION_ABSTENTION"
    else:
        allowed_ops=source_ops; allowed_edges=source_edges
        policy="UNIQUE_EFFECT_CONFIRMED_ELSE_ABSTAIN"
    missing_ops=tuple(sorted(set(required)-allowed_ops))
    # Alternative FIRST/LAST edges are admitted if either direction-specific
    # execution will be legal; unknown direction still fails in the executor.
    missing_edges=tuple(sorted(set(compositions)-allowed_edges))
    status="PLANNED" if not missing_ops and not missing_edges and arm!="neural_only" else "ABSTAINED"
    body={
        "task_id":receipt.task_id,"arm":arm,"status":status,
        "required_operators":required,"required_compositions":compositions,
        "missing_operators":missing_ops,"missing_compositions":missing_edges,
        "commit_policy":policy,"semantic_receipt_sha256":receipt.receipt_sha256,
        "source_capability_sha256":(
            str(source_capabilities["artifact_sha256"]) if arm in {"source_induced","source_permuted"} else None
        ),"target_outcome_read":False,
    }
    return LayerBHarnessPlan(**body,plan_sha256=stable_hash(body))


__all__=[
    "ARMS", "LayerBHarnessPlan", "plan_harness_arm", "semantic_operator_plan",
    "source_permuted_compositions",
]
