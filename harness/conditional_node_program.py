"""Receipt-grounded conditional node programs with explicit native gaps.

Untrusted Agents propose segmentations.  The Harness performs no semantic
alignment: it checks exact source identity, complete target-transition
coverage, source-node/edge order, and equality of target-native node schemas
across registered adaptation examples.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from harness.alfworld_grammar import parse_alfworld_action
from harness.multistep_binding import build_demo_transition_contract_receipt
from harness.online_transfer_runtime import NativeTransitionEvidence
from harness.receipt_version_space import transition_supported_evidence
from harness.skill_admission import TargetDemoReceipt


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _valid_hash(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


class SegmentKind(str, Enum):
    SOURCE_NODE = "SOURCE_NODE"
    TARGET_NATIVE_GAP = "TARGET_NATIVE_GAP"


class ConditionalRuntimeVerdict(str, Enum):
    SOURCE_READY = "SOURCE_READY"
    TARGET_NATIVE_GAP_REQUIRED = "TARGET_NATIVE_GAP_REQUIRED"
    SOURCE_PROGRAM_FINISHED = "SOURCE_PROGRAM_FINISHED"
    SUPPORTED = "SUPPORTED"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"


class ConditionalAdmissionStatus(str, Enum):
    READY = "READY"
    NEED_MORE_AGENT_PROPOSALS = "NEED_MORE_AGENT_PROPOSALS"


@dataclass(frozen=True)
class ProposedSegment:
    segment_id: str
    kind: SegmentKind
    source_node_id: str | None
    target_transition_indices: Sequence[int]


@dataclass(frozen=True)
class ExampleSegmentation:
    demo_id: str
    segments: Sequence[ProposedSegment]


@dataclass(frozen=True)
class ConditionalProgramProposal:
    proposal_id: str
    proposal_source: str
    proposal_receipt_sha256: str
    source_hypothesis_hash: str
    examples: Sequence[ExampleSegmentation]

    def content_hash(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class TargetStepSchema:
    target_operator: str
    argument_types: Mapping[str, str]


@dataclass(frozen=True)
class StepReceiptWitness:
    demo_hash: str
    target_transition_index: int
    receipt_sha256: str
    supported_evidence: Sequence[str]


@dataclass(frozen=True)
class ConditionalNode:
    node_id: str
    steps: Sequence[TargetStepSchema]
    witnesses: Sequence[Sequence[StepReceiptWitness]]
    source_conditioning: Mapping[str, Any]


@dataclass(frozen=True)
class NativeGapWitness:
    demo_hash: str
    gap_id: str
    predecessor_source_node_id: str | None
    successor_source_node_id: str | None
    steps: Sequence[TargetStepSchema]
    receipt_sha256s: Sequence[str]


@dataclass(frozen=True)
class SourceEdgeWitness:
    source_node_id: str
    target_node_id: str
    kind: str
    intervention_receipt_sha256s: Sequence[str]
    edge_sha256: str


@dataclass(frozen=True)
class ConditionalTransitionReceipt:
    artifact_hash: str
    transition_receipt_sha256: str
    node_index_before: int
    step_index_before: int
    executed_command: str
    verdict: ConditionalRuntimeVerdict
    receipt_sha256: str

    def to_dict(self) -> Mapping[str, Any]:
        unsigned = asdict(self)
        unsigned["verdict"] = self.verdict.value
        unsigned.pop("receipt_sha256")
        if _hash(unsigned) != self.receipt_sha256:
            raise ValueError("conditional transition receipt hash mismatch")
        return {**unsigned, "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class QualifiedConditionalProgram:
    proposal_hash: str
    source_hypothesis_hash: str
    nodes: Sequence[ConditionalNode]
    native_gaps: Sequence[NativeGapWitness]
    source_edges: Sequence[SourceEdgeWitness]
    checks: Mapping[str, bool]

    def content_hash(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class ConditionalProgramArtifact:
    adaptation_set_id: str
    target_domain: str
    task_family: str
    demo_ids: Sequence[str]
    demo_hashes: Sequence[str]
    source_treatment: str
    candidates: Sequence[QualifiedConditionalProgram]
    rejected_candidates: Sequence[Mapping[str, Any]]
    status: ConditionalAdmissionStatus
    schema_version: int = 1
    semantic_alignment_claimed: bool = False

    def unsigned_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    def to_dict(self) -> Mapping[str, Any]:
        payload = dict(self.unsigned_payload())
        payload["artifact_hash"] = _hash(payload)
        return payload

    @property
    def artifact_hash(self) -> str:
        return str(self.to_dict()["artifact_hash"])


def proposal_from_dict(payload: Mapping[str, Any]) -> ConditionalProgramProposal:
    return ConditionalProgramProposal(
        proposal_id=str(payload["proposal_id"]),
        proposal_source=str(payload["proposal_source"]),
        proposal_receipt_sha256=str(payload["proposal_receipt_sha256"]),
        source_hypothesis_hash=str(payload["source_hypothesis_hash"]),
        examples=tuple(ExampleSegmentation(
            demo_id=str(example["demo_id"]),
            segments=tuple(ProposedSegment(
                segment_id=str(segment["segment_id"]),
                kind=SegmentKind(str(segment["kind"])),
                source_node_id=(
                    str(segment["source_node_id"])
                    if segment.get("source_node_id") is not None else None
                ),
                target_transition_indices=tuple(
                    int(item) for item in segment["target_transition_indices"]
                ),
            ) for segment in example["segments"]),
        ) for example in payload["examples"]),
    )


def _edge_witness(edge: Mapping[str, Any]) -> SourceEdgeWitness:
    unsigned = {
        "source_node_id": str(edge["source_node_id"]),
        "target_node_id": str(edge["target_node_id"]),
        "kind": str(edge["kind"]),
        "intervention_receipt_sha256s": tuple(
            str(item) for item in edge.get("intervention_receipt_sha256s") or ()
        ),
    }
    return SourceEdgeWitness(**unsigned, edge_sha256=_hash(unsigned))


def admit_conditional_programs(
    *, adaptation_set_id: str, proposals: Sequence[ConditionalProgramProposal],
    demos: Sequence[TargetDemoReceipt], source_graphs: Sequence[Mapping[str, Any]],
    known_proposal_receipt_hashes: Sequence[str], source_treatment: str,
) -> ConditionalProgramArtifact:
    """Admit only exact multi-example segmentations; retain the full set."""
    if not adaptation_set_id or len(demos) < 2:
        raise ValueError("conditional admission requires a registered multi-example set")
    for demo in demos:
        demo.validate_for_admission()
    if len({demo.content_hash() for demo in demos}) != len(demos):
        raise ValueError("adaptation examples must be distinct")
    if len({(demo.target_domain, demo.task_family) for demo in demos}) != 1:
        raise ValueError("adaptation examples must share target protocol identity")
    graph_by_hash = {str(graph["source_hypothesis_hash"]): graph for graph in source_graphs}
    known_receipts = set(known_proposal_receipt_hashes)
    demo_by_id = {demo.demo_id: demo for demo in demos}
    qualified = []
    rejected = []
    for proposal in proposals:
        graph = graph_by_hash.get(proposal.source_hypothesis_hash)
        source_node_ids = [str(node["node_id"]) for node in (graph or {}).get("nodes", ())]
        examples_by_id = {example.demo_id: example for example in proposal.examples}
        checks: dict[str, bool] = {
            "proposal_identity": bool(proposal.proposal_id),
            "proposal_receipt_known": (
                _valid_hash(proposal.proposal_receipt_sha256)
                and proposal.proposal_receipt_sha256 in known_receipts
            ),
            "source_hypothesis_known": graph is not None,
            "examples_exact": (
                len(examples_by_id) == len(proposal.examples)
                and set(examples_by_id) == set(demo_by_id)
            ),
        }
        diagnostics: dict[str, Any] = {}
        per_demo_nodes: dict[str, dict[str, ProposedSegment]] = {}
        gap_witnesses: list[NativeGapWitness] = []
        if checks["examples_exact"] and graph is not None:
            for demo in demos:
                example = examples_by_id[demo.demo_id]
                flattened = [
                    index for segment in example.segments
                    for index in segment.target_transition_indices
                ]
                node_segments = [
                    segment for segment in example.segments
                    if segment.kind == SegmentKind.SOURCE_NODE
                ]
                node_ids = [segment.source_node_id for segment in node_segments]
                segment_ids = [segment.segment_id for segment in example.segments]
                checks[f"{demo.demo_id}:segment_ids_unique"] = (
                    len(segment_ids) == len(set(segment_ids))
                )
                checks[f"{demo.demo_id}:complete_ordered_coverage"] = (
                    flattened == list(range(len(demo.actions)))
                )
                checks[f"{demo.demo_id}:segments_nonempty"] = all(
                    segment.target_transition_indices for segment in example.segments
                )
                checks[f"{demo.demo_id}:source_nodes_exact_once"] = (
                    node_ids == source_node_ids
                )
                diagnostics[f"{demo.demo_id}:source_node_order"] = {
                    "expected": source_node_ids, "observed": node_ids,
                }
                diagnostics[f"{demo.demo_id}:transition_coverage"] = {
                    "expected": list(range(len(demo.actions))), "observed": flattened,
                }
                checks[f"{demo.demo_id}:gap_identity_closed"] = all(
                    (
                        segment.kind == SegmentKind.SOURCE_NODE
                        and segment.source_node_id in source_node_ids
                    ) or (
                        segment.kind == SegmentKind.TARGET_NATIVE_GAP
                        and segment.source_node_id is None
                    )
                    for segment in example.segments
                )
                per_demo_nodes[demo.demo_id] = {
                    str(segment.source_node_id): segment for segment in node_segments
                }
                for position, segment in enumerate(example.segments):
                    if segment.kind != SegmentKind.TARGET_NATIVE_GAP:
                        continue
                    previous_nodes = [
                        item.source_node_id for item in example.segments[:position]
                        if item.kind == SegmentKind.SOURCE_NODE
                    ]
                    next_nodes = [
                        item.source_node_id for item in example.segments[position + 1:]
                        if item.kind == SegmentKind.SOURCE_NODE
                    ]
                    gap_witnesses.append(NativeGapWitness(
                        demo_hash=demo.content_hash(), gap_id=segment.segment_id,
                        predecessor_source_node_id=(previous_nodes[-1] if previous_nodes else None),
                        successor_source_node_id=(next_nodes[0] if next_nodes else None),
                        steps=tuple(TargetStepSchema(
                            demo.actions[index].operator,
                            dict(demo.actions[index].argument_types),
                        ) for index in segment.target_transition_indices),
                        receipt_sha256s=tuple(
                            build_demo_transition_contract_receipt(
                                demo.actions[index]
                            ).receipt_sha256
                            for index in segment.target_transition_indices
                        ),
                    ))
        nodes: list[ConditionalNode] = []
        if all(checks.values()) and graph is not None:
            for source_node in graph["nodes"]:
                node_id = str(source_node["node_id"])
                schemas = []
                all_witnesses = []
                for demo in demos:
                    segment = per_demo_nodes[demo.demo_id][node_id]
                    schema = tuple(TargetStepSchema(
                        demo.actions[index].operator,
                        dict(demo.actions[index].argument_types),
                    ) for index in segment.target_transition_indices)
                    schemas.append(schema)
                    all_witnesses.append(tuple(StepReceiptWitness(
                        demo_hash=demo.content_hash(),
                        target_transition_index=index,
                        receipt_sha256=build_demo_transition_contract_receipt(
                            demo.actions[index]
                        ).receipt_sha256,
                        supported_evidence=tuple(
                            build_demo_transition_contract_receipt(
                                demo.actions[index]
                            ).supported_evidence
                        ),
                    ) for index in segment.target_transition_indices))
                checks[f"node:{node_id}:schema_exact_across_examples"] = all(
                    schema == schemas[0] for schema in schemas[1:]
                )
                diagnostics[f"node:{node_id}:observed_schemas"] = {
                    demo.demo_id: [asdict(step) for step in schema]
                    for demo, schema in zip(demos, schemas)
                }
                if schemas:
                    nodes.append(ConditionalNode(
                        node_id=node_id, steps=schemas[0], witnesses=tuple(all_witnesses),
                        source_conditioning={
                            "observed_transitions": list(source_node["observed_transitions"]),
                        },
                    ))
        graph_edges = {(str(edge["source_node_id"]), str(edge["target_node_id"])): edge
                       for edge in (graph or {}).get("edges", ())}
        edge_witnesses = []
        for left, right in zip(source_node_ids, source_node_ids[1:]):
            edge = graph_edges.get((left, right))
            checks[f"edge:{left}->{right}:registered"] = edge is not None
            checks[f"edge:{left}->{right}:has_intervention_receipts"] = bool(
                (edge or {}).get("intervention_receipt_sha256s")
            ) and all(_valid_hash(str(item)) for item in (
                (edge or {}).get("intervention_receipt_sha256s") or ()
            ))
            if edge is not None:
                edge_witnesses.append(_edge_witness(edge))
        if all(checks.values()):
            candidate = QualifiedConditionalProgram(
                proposal_hash=proposal.content_hash(),
                source_hypothesis_hash=proposal.source_hypothesis_hash,
                nodes=tuple(nodes), native_gaps=tuple(gap_witnesses),
                source_edges=tuple(edge_witnesses), checks=checks,
            )
            qualified.append(candidate)
        else:
            rejected.append({
                "proposal_id": proposal.proposal_id,
                "failure_codes": [key for key, value in checks.items() if not value],
                "diagnostics": diagnostics,
            })
    by_hash = {item.content_hash(): item for item in qualified}
    first = demos[0]
    artifact = ConditionalProgramArtifact(
        adaptation_set_id=adaptation_set_id, target_domain=first.target_domain,
        task_family=first.task_family, demo_ids=tuple(demo.demo_id for demo in demos),
        demo_hashes=tuple(demo.content_hash() for demo in demos),
        source_treatment=source_treatment,
        candidates=tuple(by_hash[key] for key in sorted(by_hash)),
        rejected_candidates=tuple(rejected),
        status=(
            ConditionalAdmissionStatus.READY
            if by_hash else ConditionalAdmissionStatus.NEED_MORE_AGENT_PROPOSALS
        ),
    )
    conditional_artifact_from_dict(artifact.to_dict())
    return artifact


def conditional_artifact_from_dict(payload: Mapping[str, Any]) -> ConditionalProgramArtifact:
    candidates = []
    for row in payload.get("candidates") or ():
        candidates.append(QualifiedConditionalProgram(
            proposal_hash=str(row["proposal_hash"]),
            source_hypothesis_hash=str(row["source_hypothesis_hash"]),
            nodes=tuple(ConditionalNode(
                node_id=str(node["node_id"]),
                steps=tuple(TargetStepSchema(
                    str(step["target_operator"]), dict(step["argument_types"]),
                ) for step in node["steps"]),
                witnesses=tuple(tuple(StepReceiptWitness(
                    demo_hash=str(item["demo_hash"]),
                    target_transition_index=int(item["target_transition_index"]),
                    receipt_sha256=str(item["receipt_sha256"]),
                    supported_evidence=tuple(item["supported_evidence"]),
                ) for item in witness) for witness in node["witnesses"]),
                source_conditioning=dict(node["source_conditioning"]),
            ) for node in row["nodes"]),
            native_gaps=tuple(NativeGapWitness(
                demo_hash=str(gap["demo_hash"]), gap_id=str(gap["gap_id"]),
                predecessor_source_node_id=gap.get("predecessor_source_node_id"),
                successor_source_node_id=gap.get("successor_source_node_id"),
                steps=tuple(TargetStepSchema(
                    str(step["target_operator"]), dict(step["argument_types"]),
                ) for step in gap["steps"]),
                receipt_sha256s=tuple(gap["receipt_sha256s"]),
            ) for gap in row["native_gaps"]),
            source_edges=tuple(SourceEdgeWitness(
                source_node_id=str(edge["source_node_id"]),
                target_node_id=str(edge["target_node_id"]), kind=str(edge["kind"]),
                intervention_receipt_sha256s=tuple(edge["intervention_receipt_sha256s"]),
                edge_sha256=str(edge["edge_sha256"]),
            ) for edge in row["source_edges"]),
            checks=dict(row["checks"]),
        ))
    artifact = ConditionalProgramArtifact(
        adaptation_set_id=str(payload["adaptation_set_id"]),
        target_domain=str(payload["target_domain"]), task_family=str(payload["task_family"]),
        demo_ids=tuple(payload["demo_ids"]), demo_hashes=tuple(payload["demo_hashes"]),
        source_treatment=str(payload["source_treatment"]), candidates=tuple(candidates),
        rejected_candidates=tuple(payload.get("rejected_candidates") or ()),
        status=ConditionalAdmissionStatus(str(payload["status"])),
        schema_version=int(payload.get("schema_version", 0)),
        semantic_alignment_claimed=bool(payload.get("semantic_alignment_claimed", True)),
    )
    if artifact.schema_version != 1 or artifact.semantic_alignment_claimed:
        raise ValueError("unsupported conditional artifact")
    claimed = str(payload.get("artifact_hash") or "")
    if artifact.artifact_hash != claimed:
        raise ValueError("conditional artifact hash mismatch")
    expected_status = (
        ConditionalAdmissionStatus.READY if artifact.candidates
        else ConditionalAdmissionStatus.NEED_MORE_AGENT_PROPOSALS
    )
    if artifact.status != expected_status:
        raise ValueError("conditional admission status is inconsistent")
    if not artifact.candidates:
        return artifact
    for candidate in artifact.candidates:
        if not _valid_hash(candidate.proposal_hash) or not all(candidate.checks.values()):
            raise ValueError("invalid qualified conditional candidate")
        if [node.node_id for node in candidate.nodes] == []:
            raise ValueError("conditional candidate has no nodes")
        for edge in candidate.source_edges:
            unsigned = {
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id, "kind": edge.kind,
                "intervention_receipt_sha256s": edge.intervention_receipt_sha256s,
            }
            if edge.edge_sha256 != _hash(unsigned):
                raise ValueError("conditional source edge hash mismatch")
        for node in candidate.nodes:
            if len(node.witnesses) != len(artifact.demo_hashes):
                raise ValueError("conditional node witness count mismatch")
            for step_index in range(len(node.steps)):
                if any(len(witness) != len(node.steps) for witness in node.witnesses):
                    raise ValueError("invalid conditional node receipt witness")
                if {
                    witness[step_index].demo_hash for witness in node.witnesses
                } != set(artifact.demo_hashes):
                    raise ValueError("conditional node witness coverage mismatch")
                if any(
                    not _valid_hash(witness[step_index].receipt_sha256)
                    for witness in node.witnesses
                ):
                    raise ValueError("invalid conditional node receipt witness")
    return artifact


class ConditionalNodeRuntime:
    """Execute only common node-local signatures; target Actor fills entry gaps."""

    def __init__(self, artifact: ConditionalProgramArtifact) -> None:
        conditional_artifact_from_dict(artifact.to_dict())
        if not artifact.candidates:
            raise ValueError("conditional runtime requires admitted candidates")
        node_orders = {tuple(node.node_id for node in item.nodes) for item in artifact.candidates}
        if len(node_orders) != 1:
            raise ValueError("conditional candidates disagree on source node order")
        self.artifact = artifact
        self.node_index = 0
        self.step_index = 0
        self.node_started = False
        self.paused = False

    def _current_steps(self) -> Sequence[TargetStepSchema]:
        if self.node_index >= len(self.artifact.candidates[0].nodes):
            return ()
        return tuple(
            candidate.nodes[self.node_index].steps[self.step_index]
            for candidate in self.artifact.candidates
        )

    def allowed_actions(self, admissible: Sequence[str]) -> tuple[
        ConditionalRuntimeVerdict, tuple[str, ...]
    ]:
        if self.node_index >= len(self.artifact.candidates[0].nodes):
            return ConditionalRuntimeVerdict.SOURCE_PROGRAM_FINISHED, ()
        steps = self._current_steps()
        if len({(step.target_operator, tuple(sorted(step.argument_types.items())))
                for step in steps}) != 1:
            self.paused = True
            return ConditionalRuntimeVerdict.NEED_MORE_EVIDENCE, ()
        expected = steps[0]
        allowed = []
        for command in admissible:
            try:
                parsed = parse_alfworld_action(command, admissible=admissible)
            except ValueError:
                continue
            if (parsed.operator == expected.target_operator
                    and dict(parsed.argument_types) == dict(expected.argument_types)):
                allowed.append(command)
        if not allowed and not self.node_started:
            return ConditionalRuntimeVerdict.TARGET_NATIVE_GAP_REQUIRED, ()
        if not allowed:
            self.paused = True
            return ConditionalRuntimeVerdict.NEED_MORE_EVIDENCE, ()
        return ConditionalRuntimeVerdict.SOURCE_READY, tuple(allowed)

    def source_conditioning(self) -> Sequence[Mapping[str, Any]]:
        return tuple({
            "conditional_candidate_hash": candidate.content_hash(),
            "source_hypothesis_hash": candidate.source_hypothesis_hash,
            "node_id": candidate.nodes[self.node_index].node_id,
            "node_step_index": self.step_index,
            "source_conditioning": dict(candidate.nodes[self.node_index].source_conditioning),
        } for candidate in self.artifact.candidates)

    def observe_source_transition(
        self, transition: NativeTransitionEvidence, *, executed_command: str,
        before_admissible: Sequence[str],
    ) -> ConditionalTransitionReceipt:
        transition.validate_hash()
        node_before, step_before = self.node_index, self.step_index
        if executed_command not in before_admissible:
            raise ValueError("conditional source command was not native-admissible")
        parsed = parse_alfworld_action(
            executed_command, admissible=before_admissible,
        )
        steps = self._current_steps()
        if not steps or any(
            parsed.operator != step.target_operator
            or dict(parsed.argument_types) != dict(step.argument_types)
            for step in steps
        ):
            raise ValueError("conditional source command violates current exact schema")
        observed = transition_supported_evidence(transition)
        verdict = ConditionalRuntimeVerdict.SUPPORTED
        for candidate in self.artifact.candidates:
            node = candidate.nodes[self.node_index]
            known = {
                tuple(witness[self.step_index].supported_evidence)
                for witness in node.witnesses
            }
            if observed not in known:
                self.paused = True
                verdict = ConditionalRuntimeVerdict.NEED_MORE_EVIDENCE
                break
        if verdict == ConditionalRuntimeVerdict.SUPPORTED:
            self.node_started = True
            self.step_index += 1
            node_length = len(self.artifact.candidates[0].nodes[self.node_index].steps)
            if self.step_index >= node_length:
                self.node_index += 1
                self.step_index = 0
                self.node_started = False
        unsigned = {
            "artifact_hash": self.artifact.artifact_hash,
            "transition_receipt_sha256": transition.receipt_sha256,
            "node_index_before": node_before, "step_index_before": step_before,
            "executed_command": executed_command, "verdict": verdict.value,
        }
        receipt = ConditionalTransitionReceipt(
            artifact_hash=unsigned["artifact_hash"],
            transition_receipt_sha256=unsigned["transition_receipt_sha256"],
            node_index_before=node_before, step_index_before=step_before,
            executed_command=executed_command, verdict=verdict,
            receipt_sha256=_hash(unsigned),
        )
        receipt.to_dict()
        return receipt


__all__ = [
    "ConditionalAdmissionStatus", "ConditionalNodeRuntime", "ConditionalProgramArtifact",
    "ConditionalProgramProposal", "ConditionalRuntimeVerdict",
    "ConditionalTransitionReceipt",
    "ExampleSegmentation", "ProposedSegment", "SegmentKind",
    "admit_conditional_programs", "conditional_artifact_from_dict",
    "proposal_from_dict",
]
