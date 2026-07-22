from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .contracts import BindingEvidence, BindingHypothesis, EvidenceVerdict, MotifCandidate, stable_hash


class BindingAttribution(str, Enum):
    """What the alpha control can, and cannot, establish about a binding."""

    GENERIC_STRUCTURAL = "GENERIC_STRUCTURAL"
    TARGET_GROUNDED_PROVISIONAL = "TARGET_GROUNDED_PROVISIONAL"


class BindingArtifactStatus(str, Enum):
    ADMITTED = "ADMITTED"
    REJECTED_UNSTABLE = "REJECTED_UNSTABLE"


@dataclass(frozen=True)
class AttributedBinding:
    hypothesis: BindingHypothesis
    attribution: BindingAttribution


@dataclass(frozen=True)
class FrozenBindingArtifact:
    """Immutable output of adaptation; evaluation must not regenerate it."""

    schema_version: int
    motif_id: str
    adaptation_example_sha256: str
    induction_repetitions: int
    raw_signature_sets: tuple[tuple[str, ...], ...]
    alpha_signature_sets: tuple[tuple[str, ...], ...]
    bindings: tuple[AttributedBinding, ...]
    status: BindingArtifactStatus
    backend_identity_sha256: str
    call_receipt_hashes: tuple[str, ...]
    artifact_hash: str

    @staticmethod
    def _hypothesis_dict(row: BindingHypothesis) -> dict[str, object]:
        payload = asdict(row)
        payload["status"] = row.status.value
        return payload

    def unsigned_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "motif_id": self.motif_id,
            "adaptation_example_sha256": self.adaptation_example_sha256,
            "induction_repetitions": self.induction_repetitions,
            "raw_signature_sets": self.raw_signature_sets,
            "alpha_signature_sets": self.alpha_signature_sets,
            "bindings": [
                {
                    "hypothesis": self._hypothesis_dict(row.hypothesis),
                    "attribution": row.attribution.value,
                }
                for row in self.bindings
            ],
            "status": self.status.value,
            "backend_identity_sha256": self.backend_identity_sha256,
            "call_receipt_hashes": self.call_receipt_hashes,
        }

    def validate(self) -> bool:
        return self.schema_version == 1 and stable_hash(self.unsigned_payload()) == self.artifact_hash

    @property
    def hypotheses(self) -> tuple[BindingHypothesis, ...]:
        return tuple(row.hypothesis for row in self.bindings)


def alignment_signature(
    node_alignment: Sequence[tuple[int, Sequence[int]]],
    edge_alignment: Sequence[tuple[int, Sequence[int]]],
) -> str:
    return stable_hash({
        "nodes": [(int(node), tuple(int(index) for index in indices)) for node, indices in node_alignment],
        "edges": [(int(edge), tuple(int(index) for index in boundary)) for edge, boundary in edge_alignment],
    })


def validate_structural_binding(
    motif: MotifCandidate,
    *,
    target_cycle_count: int,
    node_alignment: Sequence[tuple[int, Sequence[int]]],
    edge_alignment: Sequence[tuple[int, Sequence[int]]],
) -> str:
    """Validate only references, coverage and graph order; never target semantics."""
    nodes = [(int(node), tuple(int(index) for index in indices)) for node, indices in node_alignment]
    edges = [(int(edge), tuple(int(index) for index in boundary)) for edge, boundary in edge_alignment]
    node_ordinals = [node for node, _ in nodes]
    if sorted(node_ordinals) != list(range(len(motif.nodes))):
        raise ValueError("binding must align every source node exactly once")
    target_indices = [index for _, indices in nodes for index in indices]
    if sorted(target_indices) != list(range(target_cycle_count)):
        raise ValueError("binding node alignment must partition the full target example")
    if any(tuple(sorted(indices)) != indices or not indices for _, indices in nodes):
        raise ValueError("each target node span must be non-empty and ordered")
    if any(indices != tuple(range(indices[0], indices[-1] + 1)) for _, indices in nodes):
        raise ValueError("each target node alignment must be one contiguous span")
    edge_ordinals = [edge for edge, _ in edges]
    if sorted(edge_ordinals) != list(range(len(motif.edges))):
        raise ValueError("binding must align every source edge exactly once")
    for edge_ordinal, boundary in edges:
        if len(boundary) != 2:
            raise ValueError("target edge boundary must contain two indices")
        source_edge = motif.edges[edge_ordinal]
        motif_node_ordinals = {node.node_id: index for index, node in enumerate(motif.nodes)}
        source_node = motif_node_ordinals[source_edge.source]
        target_node = motif_node_ordinals[source_edge.target]
        aligned = dict(nodes)
        if boundary[0] not in aligned[source_node] or boundary[1] not in aligned[target_node]:
            raise ValueError("target boundary is incompatible with its source graph edge")
        if boundary[0] >= boundary[1]:
            raise ValueError("target edge boundary must preserve observed order")
    return alignment_signature(nodes, edges)


def alpha_rename_target_actions(example: Mapping[str, object]) -> dict[str, object]:
    """Remove all target action semantics while preserving exact equality structure."""
    aliases: dict[str, str] = {}

    def alias(value: object) -> object:
        if not isinstance(value, str):
            return value
        if value not in aliases:
            aliases[value] = f"TARGET_ACTION_{len(aliases)}"
        return aliases[value]

    result = dict(example)
    transitions = []
    for raw in example.get("transitions", []):
        row = dict(raw)
        row["action"] = alias(row.get("action"))
        row["before_native_actions"] = [alias(value) for value in row.get("before_native_actions", [])]
        row["after_native_actions"] = [alias(value) for value in row.get("after_native_actions", [])]
        transitions.append(row)
    result["transitions"] = transitions
    result["renaming_control"] = "FULL_ACTION_ALPHA_RENAMING_V1"
    return result


@dataclass(frozen=True)
class BindingState:
    hypothesis: BindingHypothesis
    evidence: tuple[BindingEvidence, ...] = ()
    viable: bool = True


class BindingVersionSpace:
    """Keeps every binding consistent with deterministic target evidence.

    It never ranks candidates by LLM confidence or natural-language similarity.
    """

    def __init__(self, hypotheses: Iterable[BindingHypothesis]):
        rows = tuple(hypotheses)
        if not rows:
            raise ValueError("a version space requires at least one hypothesis")
        if len({row.binding_id for row in rows}) != len(rows):
            raise ValueError("duplicate binding id")
        self._states = {row.binding_id: BindingState(row) for row in rows}

    def record(self, evidence: BindingEvidence) -> BindingState:
        if evidence.binding_id not in self._states:
            raise KeyError(evidence.binding_id)
        state = self._states[evidence.binding_id]
        if evidence.verifier_id != state.hypothesis.verifier_id:
            raise ValueError("evidence was produced by a different verifier")
        viable = state.viable and evidence.verdict != EvidenceVerdict.REFUTED
        updated = BindingState(state.hypothesis, state.evidence + (evidence,), viable)
        self._states[evidence.binding_id] = updated
        return updated

    def viable(self) -> tuple[BindingHypothesis, ...]:
        return tuple(state.hypothesis for state in self._states.values() if state.viable)

    def all_states(self) -> tuple[BindingState, ...]:
        return tuple(self._states.values())
