"""Content-addressed IR for receipt-grounded neural-symbolic programs.

The symbolic layer owns control flow.  Neural probes may only return a
calibrated score for either a pre-action guard or a post-action effect.  The
runtime converts that score to ``SUPPORTED``, ``REFUTED`` or ``UNKNOWN`` and
always abstains on ``UNKNOWN``.

Probe descriptions and node labels are explicitly untrusted.  They are useful
for inspection, but never participate in a runtime decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Mapping, Sequence

from .contracts import stable_hash


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


class ProbeInputKind(str, Enum):
    BEFORE = "BEFORE"
    TRANSITION = "TRANSITION"


class ProbeVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    UNKNOWN = "UNKNOWN"


class RouteKind(str, Enum):
    NEXT_NODE = "NEXT_NODE"
    REPLAN = "REPLAN"
    ABSTAIN = "ABSTAIN"
    TERMINATE = "TERMINATE"


@dataclass(frozen=True)
class NeuralProbeSpec:
    """Frozen contract for one learned operational predicate."""

    probe_id: str
    input_kind: ProbeInputKind
    model_artifact_sha256: str
    source_receipt_ids: tuple[str, ...]
    refuted_max_score: float = 0.2
    supported_min_score: float = 0.8
    untrusted_description: str = ""

    @classmethod
    def create(
        cls,
        *,
        input_kind: ProbeInputKind,
        model_artifact_sha256: str,
        source_receipt_ids: Sequence[str],
        refuted_max_score: float = 0.2,
        supported_min_score: float = 0.8,
        untrusted_description: str = "",
    ) -> "NeuralProbeSpec":
        body = {
            "input_kind": input_kind.value,
            "model_artifact_sha256": model_artifact_sha256,
            "source_receipt_ids": tuple(source_receipt_ids),
            "refuted_max_score": refuted_max_score,
            "supported_min_score": supported_min_score,
            "untrusted_description": untrusted_description,
        }
        probe = cls(stable_hash(body), input_kind, model_artifact_sha256,
                    tuple(source_receipt_ids), refuted_max_score,
                    supported_min_score, untrusted_description)
        probe.validate()
        return probe

    def validate(self) -> None:
        if not _is_sha256(self.model_artifact_sha256):
            raise ValueError("probe model artifact must be a lowercase sha256")
        if not self.source_receipt_ids:
            raise ValueError("probe requires source receipt provenance")
        if len(set(self.source_receipt_ids)) != len(self.source_receipt_ids):
            raise ValueError("probe source receipt ids must be unique")
        if not all(_is_sha256(receipt_id) for receipt_id in self.source_receipt_ids):
            raise ValueError("probe source receipt ids must be lowercase sha256 values")
        if not all(math.isfinite(value) for value in (
            self.refuted_max_score, self.supported_min_score,
        )):
            raise ValueError("probe thresholds must be finite")
        if not (
            0.0 <= self.refuted_max_score
            < self.supported_min_score <= 1.0
        ):
            raise ValueError("probe thresholds must define a non-empty UNKNOWN band")
        body = asdict(self)
        expected = body.pop("probe_id")
        body["input_kind"] = self.input_kind.value
        if expected != stable_hash(body):
            raise ValueError("probe id does not match its frozen content")

    def verdict(self, score: float | None) -> ProbeVerdict:
        if score is None:
            return ProbeVerdict.UNKNOWN
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("probe score must be finite and in [0, 1]")
        if score >= self.supported_min_score:
            return ProbeVerdict.SUPPORTED
        if score <= self.refuted_max_score:
            return ProbeVerdict.REFUTED
        return ProbeVerdict.UNKNOWN


@dataclass(frozen=True)
class ControlRoute:
    kind: RouteKind
    target_node_id: str | None = None

    def validate(self) -> None:
        if self.kind == RouteKind.NEXT_NODE and not self.target_node_id:
            raise ValueError("NEXT_NODE route requires a target node")
        if self.kind != RouteKind.NEXT_NODE and self.target_node_id is not None:
            raise ValueError(f"{self.kind.value} route cannot name a target node")


@dataclass(frozen=True)
class NeuroSymbolicNode:
    node_id: str
    before_guard_probe_id: str
    transition_effect_probe_id: str
    on_guard_refuted: ControlRoute
    on_effect_supported: ControlRoute
    on_effect_refuted: ControlRoute
    untrusted_role: str = ""


@dataclass(frozen=True)
class NeuroSymbolicProgram:
    program_id: str
    entry_node_id: str
    probes: tuple[NeuralProbeSpec, ...]
    nodes: tuple[NeuroSymbolicNode, ...]
    source_lineage: tuple[str, ...]
    schema_version: int = 1
    untrusted_description: str = ""

    @classmethod
    def create(
        cls,
        *,
        entry_node_id: str,
        probes: Sequence[NeuralProbeSpec],
        nodes: Sequence[NeuroSymbolicNode],
        source_lineage: Sequence[str],
        untrusted_description: str = "",
    ) -> "NeuroSymbolicProgram":
        body = {
            "entry_node_id": entry_node_id,
            "probes": tuple(asdict(probe) for probe in probes),
            "nodes": tuple(asdict(node) for node in nodes),
            "source_lineage": tuple(source_lineage),
            "schema_version": 1,
            "untrusted_description": untrusted_description,
        }
        program = cls(
            stable_hash(body), entry_node_id, tuple(probes), tuple(nodes),
            tuple(source_lineage), 1, untrusted_description,
        )
        program.validate()
        return program

    @property
    def probe_by_id(self) -> Mapping[str, NeuralProbeSpec]:
        return {probe.probe_id: probe for probe in self.probes}

    @property
    def node_by_id(self) -> Mapping[str, NeuroSymbolicNode]:
        return {node.node_id: node for node in self.nodes}

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported neural-symbolic program schema")
        if not self.source_lineage:
            raise ValueError("program requires source-only lineage")
        if len(set(self.source_lineage)) != len(self.source_lineage):
            raise ValueError("program source lineage must be unique")
        if not self.probes or not self.nodes:
            raise ValueError("program requires probes and nodes")
        for probe in self.probes:
            probe.validate()
        probe_ids = [probe.probe_id for probe in self.probes]
        node_ids = [node.node_id for node in self.nodes]
        if len(probe_ids) != len(set(probe_ids)):
            raise ValueError("duplicate probe id")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("duplicate node id")
        if self.entry_node_id not in node_ids:
            raise ValueError("entry node does not exist")
        probes = self.probe_by_id
        has_termination_route = False
        for node in self.nodes:
            guard = probes.get(node.before_guard_probe_id)
            effect = probes.get(node.transition_effect_probe_id)
            if guard is None or guard.input_kind != ProbeInputKind.BEFORE:
                raise ValueError("node guard must reference a BEFORE probe")
            if effect is None or effect.input_kind != ProbeInputKind.TRANSITION:
                raise ValueError("node effect must reference a TRANSITION probe")
            for route in (
                node.on_guard_refuted,
                node.on_effect_supported,
                node.on_effect_refuted,
            ):
                route.validate()
                if (
                    route.kind == RouteKind.NEXT_NODE
                    and route.target_node_id not in node_ids
                ):
                    raise ValueError("route references an unknown target node")
                has_termination_route |= route.kind == RouteKind.TERMINATE
        if not has_termination_route:
            raise ValueError("program requires an explicit termination route")

        reachable = {self.entry_node_id}
        frontier = [self.entry_node_id]
        while frontier:
            current = self.node_by_id[frontier.pop()]
            for route in (
                current.on_guard_refuted,
                current.on_effect_supported,
                current.on_effect_refuted,
            ):
                target = route.target_node_id
                if route.kind == RouteKind.NEXT_NODE and target not in reachable:
                    reachable.add(str(target))
                    frontier.append(str(target))
        if reachable != set(node_ids):
            raise ValueError("program contains unreachable nodes")
        body = asdict(self)
        expected = body.pop("program_id")
        if expected != stable_hash(body):
            raise ValueError("program id does not match its frozen content")


__all__ = [
    "ControlRoute",
    "NeuralProbeSpec",
    "NeuroSymbolicNode",
    "NeuroSymbolicProgram",
    "ProbeInputKind",
    "ProbeVerdict",
    "RouteKind",
]
