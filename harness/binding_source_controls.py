"""Content-independent source treatments applied before target binding generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BindingSourceControlReceipt:
    treatment: str
    seed: int
    input_graphs_sha256: str
    output_graphs_sha256: str
    identity_mapping: Sequence[Mapping[str, str]]
    receipt_sha256: str

    def unsigned_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload.pop("receipt_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.receipt_sha256:
            raise ValueError("binding source control receipt hash mismatch")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate_hash()
        return asdict(self)


def apply_binding_source_control(
    graphs: Sequence[Mapping[str, Any]], *, treatment: str, seed: int,
) -> tuple[Sequence[Mapping[str, Any]], BindingSourceControlReceipt]:
    """Apply a registered transformation without inspecting target data or semantics.

    ``correct`` is an identity treatment. ``renamed`` deterministically replaces
    source hypothesis and node identifiers while retaining evidence payloads and
    topology. ``wrong`` is represented by passing graphs loaded from a separately
    frozen source artifact; this function records but does not choose that artifact.
    """
    if treatment not in {"empty", "correct", "wrong", "renamed", "randomized"}:
        raise ValueError("unsupported binding source treatment")
    rows = json.loads(json.dumps(list(graphs), ensure_ascii=False))
    controlled = [] if treatment == "empty" else rows
    mapping: list[Mapping[str, str]] = []
    if treatment == "renamed":
        controlled = []
        for graph_index, graph in enumerate(rows):
            updated = dict(graph)
            old_hypothesis = str(graph["source_hypothesis_hash"])
            new_hypothesis = _hash({
                "seed": int(seed), "kind": "hypothesis", "index": graph_index,
                "identity": old_hypothesis,
            })
            node_mapping = {
                str(node["node_id"]): "node_" + _hash({
                    "seed": int(seed), "kind": "node", "graph": graph_index,
                    "identity": str(node["node_id"]),
                })[:16]
                for node in graph.get("nodes") or ()
            }
            updated["source_hypothesis_hash"] = new_hypothesis
            updated["nodes"] = [
                {**dict(node), "node_id": node_mapping[str(node["node_id"])]}
                for node in graph.get("nodes") or ()
            ]
            updated["edges"] = [{
                **dict(edge),
                "source_node_id": node_mapping[str(edge["source_node_id"])],
                "target_node_id": node_mapping[str(edge["target_node_id"])],
            } for edge in graph.get("edges") or ()]
            controlled.append(updated)
            mapping.append({
                "old_identity": old_hypothesis,
                "new_identity": new_hypothesis,
            })
            mapping.extend({
                "old_identity": old, "new_identity": new,
            } for old, new in node_mapping.items())
    elif treatment == "randomized":
        controlled = []
        for graph_index, graph in enumerate(rows):
            nodes = [dict(node) for node in graph.get("nodes") or ()]
            if len(nodes) < 2:
                raise ValueError("randomized source control requires at least two nodes")
            offset = 1 + (int(seed) + graph_index) % (len(nodes) - 1)
            rotated = []
            for node_index, node in enumerate(nodes):
                donor = nodes[(node_index + offset) % len(nodes)]
                updated = dict(node)
                updated["observed_transitions"] = json.loads(json.dumps(
                    donor.get("observed_transitions") or (), ensure_ascii=False,
                ))
                rotated.append(updated)
                mapping.append({
                    "old_identity": (
                        f"{graph['source_hypothesis_hash']}:{node['node_id']}:content"
                    ),
                    "new_identity": (
                        f"{graph['source_hypothesis_hash']}:{donor['node_id']}:content"
                    ),
                })
            controlled.append({**dict(graph), "nodes": rotated})
    unsigned = {
        "treatment": str(treatment),
        "seed": int(seed),
        "input_graphs_sha256": _hash(rows),
        "output_graphs_sha256": _hash(controlled),
        "identity_mapping": mapping,
    }
    receipt = BindingSourceControlReceipt(
        treatment=unsigned["treatment"], seed=unsigned["seed"],
        input_graphs_sha256=unsigned["input_graphs_sha256"],
        output_graphs_sha256=unsigned["output_graphs_sha256"],
        identity_mapping=tuple(mapping), receipt_sha256=_hash(unsigned),
    )
    receipt.validate_hash()
    return tuple(controlled), receipt


__all__ = ["BindingSourceControlReceipt", "apply_binding_source_control"]
