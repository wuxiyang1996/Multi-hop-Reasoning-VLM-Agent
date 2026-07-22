from __future__ import annotations

from dataclasses import replace

from .contracts import MotifCandidate, MotifEdge


def generic_protocol(candidate: MotifCandidate) -> dict[str, object]:
    """Content-free control with the same node/edge counts and context shape."""
    return {
        "kind": "generic_protocol",
        "node_slots": len(candidate.nodes),
        "edge_slots": len(candidate.edges),
        "instruction": "propose, execute, observe, update, branch or terminate",
    }


def shuffled_topology(candidate: MotifCandidate) -> MotifCandidate:
    node_ids = [node.node_id for node in candidate.nodes]
    reversed_ids = list(reversed(node_ids))
    mapping = dict(zip(node_ids, reversed_ids))
    edges = tuple(
        MotifEdge(mapping[edge.source], mapping[edge.target], edge.replay_receipt_ids, "")
        for edge in candidate.edges
    )
    return replace(candidate, motif_id=f"{candidate.motif_id}:shuffled", edges=edges, untrusted_description="")
