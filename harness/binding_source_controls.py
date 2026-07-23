"""Content-independent source treatments applied before target binding generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _mask_text(value: str, *, seed: int, identity: str) -> str:
    stream = ""
    counter = 0
    while len(stream) < len(value):
        stream += _hash({"seed": seed, "identity": identity, "counter": counter})
        counter += 1
    out = []
    for index, char in enumerate(value):
        token = stream[index]
        if char.isdigit():
            out.append(str(int(token, 16) % 10))
        elif char.isalpha():
            base = ord("A") if char.isupper() else ord("a")
            out.append(chr(base + int(token, 16) % 26))
        else:
            out.append(char)
    return "".join(out)


def _mask_claim(value: Any, *, seed: int, identity: str) -> Any:
    if isinstance(value, str):
        return _mask_text(value, seed=seed, identity=identity)
    if isinstance(value, Mapping):
        return {
            str(key): _mask_claim(
                item, seed=seed, identity=f"{identity}:{key}",
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _mask_claim(item, seed=seed, identity=f"{identity}:{index}")
            for index, item in enumerate(value)
        ]
    # Preserve numeric/bool/null types and ranges; placement is controlled
    # separately where it carries transition information.
    return value


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
    if treatment not in {
        "empty", "correct", "wrong", "renamed", "randomized", "receipt_null",
    }:
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
    elif treatment == "receipt_null":
        controlled = []
        for graph_index, graph in enumerate(rows):
            all_rewards = [
                float(transition["reward"])
                for node in graph.get("nodes") or ()
                for transition in node.get("observed_transitions") or ()
            ]
            reward_offset = (
                1 + (int(seed) + graph_index) % (len(all_rewards) - 1)
                if len(all_rewards) > 1 else 0
            )
            reward_cursor = 0
            digest_map: dict[str, str] = {}

            def null_digest(value: str) -> str:
                digest_map.setdefault(value, _hash({
                    "seed": int(seed), "graph": graph_index,
                    "kind": "null_digest", "identity": value,
                }))
                return digest_map[value]

            nodes = []
            for node_index, node in enumerate(graph.get("nodes") or ()):
                transitions = []
                for transition_index, transition in enumerate(
                    node.get("observed_transitions") or ()
                ):
                    identity = f"g{graph_index}:n{node_index}:t{transition_index}"
                    updated_transition = dict(transition)
                    updated_transition["action"] = _mask_text(
                        str(transition["action"]), seed=int(seed),
                        identity=f"{identity}:action",
                    )
                    if all_rewards:
                        updated_transition["reward"] = all_rewards[
                            (reward_cursor + reward_offset) % len(all_rewards)
                        ]
                    reward_cursor += 1
                    updated_transition["state_sha256"] = null_digest(
                        str(transition["state_sha256"])
                    )
                    updated_transition["next_state_sha256"] = null_digest(
                        str(transition["next_state_sha256"])
                    )
                    if "agent_reasoning_claim" in transition:
                        updated_transition["agent_reasoning_claim"] = _mask_text(
                            str(transition["agent_reasoning_claim"]), seed=int(seed),
                            identity=f"{identity}:agent_reasoning_claim",
                        )
                    if "agent_response_sha256" in transition:
                        updated_transition["agent_response_sha256"] = null_digest(
                            str(transition["agent_response_sha256"])
                        )
                    for receipt_key in (
                        "action_proposal_receipt", "post_transition_verdict_receipt",
                    ):
                        if receipt_key in transition:
                            updated_transition[receipt_key] = _mask_claim(
                                transition[receipt_key], seed=int(seed),
                                identity=f"{identity}:{receipt_key}",
                            )
                    for hash_key in (
                        "action_proposal_event_sha256",
                        "post_transition_verdict_event_sha256",
                    ):
                        if hash_key in transition:
                            updated_transition[hash_key] = null_digest(
                                str(transition[hash_key])
                            )
                    transitions.append(updated_transition)
                nodes.append({**dict(node), "observed_transitions": transitions})
            edges = []
            for edge_index, edge in enumerate(graph.get("edges") or ()):
                updated_edge = dict(edge)
                updated_edge["agent_claim"] = _mask_claim(
                    edge.get("agent_claim") or {}, seed=int(seed),
                    identity=f"g{graph_index}:edge{edge_index}:claim",
                )
                updated_edge["intervention_receipt_sha256s"] = [
                    null_digest(str(item))
                    for item in edge.get("intervention_receipt_sha256s") or ()
                ]
                edges.append(updated_edge)
            updated_graph = {**dict(graph), "nodes": nodes, "edges": edges}
            if graph.get("source_reasoning_trace_sha256"):
                updated_graph["source_reasoning_trace_sha256"] = null_digest(
                    str(graph["source_reasoning_trace_sha256"])
                )
            controlled.append(updated_graph)
            mapping.append({
                "old_identity": str(graph["source_hypothesis_hash"]),
                "new_identity": "receipt_null:" + _hash({
                    "seed": int(seed), "graph": graph_index,
                }),
            })
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
