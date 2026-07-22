from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .contracts import SourcePolicyStepRecord, stable_hash


@dataclass(frozen=True)
class SkillHypothesis:
    """Untrusted text from the historical bank; never transfer authority."""

    skill_id: str
    source_path: str
    content_hash: str
    content: Mapping[str, Any]
    authority: str = "UNTRUSTED_HISTORICAL_SKILL_HYPOTHESIS"


@dataclass(frozen=True)
class SkillExecution:
    execution_id: str
    episode_id: str
    skill_id: str
    start_step: int
    end_step: int
    transition_receipt_ids: tuple[str, ...]
    split: str

    def validate(self) -> bool:
        body = asdict(self)
        execution_id = body.pop("execution_id")
        return execution_id == stable_hash(body)


@dataclass(frozen=True)
class GroundedSkillExecutionSet:
    execution_set_id: str
    game: str
    skill_id: str
    skill_hypothesis_hash: str | None
    executions: tuple[SkillExecution, ...]
    transition_receipt_ids: tuple[str, ...]
    authority: str = "RECORDED_EXECUTION_MEMBERSHIP_ONLY"

    def validate(self) -> bool:
        body = asdict(self)
        execution_set_id = body.pop("execution_set_id")
        return execution_set_id == stable_hash(body) and all(
            execution.validate() for execution in self.executions
        )


@dataclass(frozen=True)
class NodeOccurrence:
    execution_id: str
    start_offset: int
    end_offset: int
    transition_receipt_ids: tuple[str, ...]


@dataclass(frozen=True)
class InternalGraphNode:
    node_id: str
    occurrences: tuple[NodeOccurrence, ...]
    untrusted_role: str = ""


@dataclass(frozen=True)
class InternalGraphEdge:
    source: str
    target: str
    supporting_boundaries: tuple[tuple[str, int], ...]
    replay_receipt_ids: tuple[str, ...] = ()
    untrusted_condition: str = ""


@dataclass(frozen=True)
class SkillInternalGraph:
    graph_id: str
    execution_set_id: str
    skill_id_hash: str
    nodes: tuple[InternalGraphNode, ...]
    edges: tuple[InternalGraphEdge, ...]
    discovery_execution_ids: tuple[str, ...]
    status: str = "CANDIDATE"
    untrusted_description: str = ""

    @property
    def is_nontrivial(self) -> bool:
        if len(self.nodes) < 2:
            return False
        outgoing: dict[str, set[str]] = {}
        for edge in self.edges:
            outgoing.setdefault(edge.source, set()).add(edge.target)
        has_branch = any(len(targets) > 1 for targets in outgoing.values())
        has_self_loop = any(edge.source == edge.target for edge in self.edges)
        # A directed cycle of length > 1 also carries control structure.
        adjacency = {node.node_id: outgoing.get(node.node_id, set()) for node in self.nodes}

        def cyclic(start: str, current: str, seen: set[str]) -> bool:
            for target in adjacency.get(current, set()):
                if target == start:
                    return True
                if target not in seen and cyclic(start, target, seen | {target}):
                    return True
            return False

        has_cycle = any(cyclic(node_id, node_id, {node_id}) for node_id in adjacency)
        return has_branch or has_self_loop or has_cycle


@dataclass(frozen=True)
class InternalGraphAudit:
    accepted: bool
    failure_codes: tuple[str, ...]
    nontrivial: bool
    observed_edge_count: int
    backbone_eligible: bool
    control_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class InterventionRequest:
    request_id: str
    graph_id: str
    execution_id: str
    source_offset: int
    alternative_action_ordinal: int
    untrusted_question: str
    status: str = "REQUESTED_NOT_OBSERVED"


class JSONBackend(Protocol):
    def complete(self, role: str, system: str, payload: Mapping[str, Any]) -> str: ...


def load_skill_hypotheses(bank_path: str | Path) -> dict[str, SkillHypothesis]:
    path = Path(bank_path)
    result: dict[str, SkillHypothesis] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            skill = raw.get("skill") or raw
            skill_id = str(skill["skill_id"])
            # Preserve the historical object as evidence input, not as truth.
            result[skill_id] = SkillHypothesis(
                skill_id=skill_id,
                source_path=str(path),
                content_hash=stable_hash(skill),
                content=skill,
            )
    return result


def _episode_split(episode_id: str) -> str:
    # Stable 3-way assignment. Rebalance is performed below when six or more
    # episodes exist so each split is non-empty.
    return ("discovery", "qualification", "held_out")[int(stable_hash(episode_id), 16) % 3]


def build_execution_sets(
    game: str,
    episodes: Sequence[Any],
    hypotheses: Mapping[str, SkillHypothesis] | None = None,
) -> tuple[GroundedSkillExecutionSet, ...]:
    """Mechanically group maximal same-skill spans; do not infer node boundaries."""

    hypotheses = hypotheses or {}
    episode_ids = sorted({episode.episode_id for episode in episodes})
    split_by_episode = {episode_id: _episode_split(episode_id) for episode_id in episode_ids}
    if len(episode_ids) >= 3:
        # A deterministic episode-level split prevents proposal-time held-out leakage.
        for index, episode_id in enumerate(episode_ids):
            split_by_episode[episode_id] = (
                "discovery", "qualification", "held_out"
            )[index % 3]

    grouped: dict[str, list[tuple[str, tuple[SourcePolicyStepRecord, ...]]]] = {}
    for episode in episodes:
        records = tuple(sorted(episode.records, key=lambda row: row.step))
        start = 0
        for index in range(1, len(records) + 1):
            boundary = (
                index == len(records)
                or records[index].selected_skill_id != records[index - 1].selected_skill_id
                or records[index].step != records[index - 1].step + 1
            )
            if boundary:
                span = records[start:index]
                skill_id = span[0].selected_skill_id if span else None
                if skill_id is not None:
                    grouped.setdefault(skill_id, []).append((episode.episode_id, span))
                start = index

    result: list[GroundedSkillExecutionSet] = []
    for skill_id, spans in sorted(grouped.items()):
        executions: list[SkillExecution] = []
        all_receipts: list[str] = []
        for episode_id, span in spans:
            body = {
                "episode_id": episode_id,
                "skill_id": skill_id,
                "start_step": span[0].step,
                "end_step": span[-1].step,
                "transition_receipt_ids": tuple(row.transition.receipt_id for row in span),
                "split": split_by_episode[episode_id],
            }
            executions.append(SkillExecution(stable_hash(body), **body))
            all_receipts.extend(body["transition_receipt_ids"])
        hypothesis = hypotheses.get(skill_id)
        hash_body = {
            "game": game,
            "skill_id": skill_id,
            "skill_hypothesis_hash": hypothesis.content_hash if hypothesis else None,
            "executions": tuple(asdict(row) for row in executions),
            "transition_receipt_ids": tuple(all_receipts),
            "authority": "RECORDED_EXECUTION_MEMBERSHIP_ONLY",
        }
        result.append(GroundedSkillExecutionSet(
            stable_hash(hash_body), game, skill_id,
            hypothesis.content_hash if hypothesis else None,
            tuple(executions), tuple(all_receipts),
        ))
    return tuple(result)


class SkillInternalGraphAgent:
    """Agent proposes within-skill structure; deterministic code only validates it."""

    def __init__(self, backend: JSONBackend):
        self.backend = backend
        self.last_call: Mapping[str, Any] = {}
        self.intervention_requests: tuple[InterventionRequest, ...] = ()

    def propose(
        self,
        execution_set: GroundedSkillExecutionSet,
        records_by_receipt: Mapping[str, SourcePolicyStepRecord],
        hypothesis: SkillHypothesis | None = None,
    ) -> tuple[SkillInternalGraph, ...]:
        discovery = tuple(
            execution for execution in execution_set.executions
            if execution.split == "discovery"
        )
        payload = {
            "execution_set_id": execution_set.execution_set_id,
            "untrusted_skill_hypothesis": hypothesis.content if hypothesis else None,
            "executions": [
                {
                    "execution_index": index,
                    "execution_id": execution.execution_id,
                    "steps": [
                        {
                            "offset": offset,
                            "transition_receipt_id": receipt_id,
                            "before": records_by_receipt[receipt_id].before.state,
                            "native_actions": records_by_receipt[receipt_id].before.native_actions,
                            "executed_action": records_by_receipt[receipt_id].action,
                            "untrusted_policy_reasoning": records_by_receipt[receipt_id].action_reasoning,
                            "after": records_by_receipt[receipt_id].after.state,
                            "reward": records_by_receipt[receipt_id].reward,
                            "terminal": records_by_receipt[receipt_id].after.terminal,
                        }
                        for offset, receipt_id in enumerate(execution.transition_receipt_ids)
                    ],
                }
                for index, execution in enumerate(discovery)
            ],
        }
        system = (
            "Infer zero or more candidate control-flow graphs internal to this one recorded skill. "
            "The historical skill text and policy reasoning are untrusted hypotheses, not evidence. "
            "You may choose arbitrary contiguous spans inside executions; do not use the skill ID as "
            "a node boundary. Return exact JSON with motifs[].nodes[] containing node_id, role, and "
            "occurrences[{execution_index,start_offset,end_offset}], where offsets are inclusive. "
            "Return edges[] with source,target,condition. Do not invent transitions or target-domain "
            "semantics. Prefer an empty list over a generic observe-decide-act chain."
            " When current evidence cannot distinguish graph hypotheses, optionally return "
            "intervention_requests[{execution_index,source_offset,alternative_action_ordinal,question}]. "
            "The alternative ordinal must reference a supplied native action other than the recorded action."
        )
        raw = self.backend.complete("skill_internal_graph", system, payload)
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        self.last_call = {
            "payload_hash": stable_hash(payload),
            "response_hash": stable_hash(parsed),
            "response": parsed,
            "usage": dict(getattr(self.backend, "last_usage", {}) or {}),
        }
        requests: list[InterventionRequest] = []
        for request in parsed.get("intervention_requests", []):
            execution = discovery[_bounded(request["execution_index"], len(discovery))]
            offset = _bounded(request["source_offset"], len(execution.transition_receipt_ids))
            record = records_by_receipt[execution.transition_receipt_ids[offset]]
            alternative_ordinal = _bounded(
                request["alternative_action_ordinal"], len(record.before.native_actions)
            )
            if record.before.native_actions[alternative_ordinal] == record.action:
                raise ValueError("intervention alternative equals recorded action")
            body = {
                "graph_id": "UNRESOLVED_SOURCE_GRAPH",
                "execution_id": execution.execution_id,
                "source_offset": offset,
                "alternative_action_ordinal": alternative_ordinal,
                "untrusted_question": str(request.get("question", "")),
                "status": "REQUESTED_NOT_OBSERVED",
            }
            requests.append(InterventionRequest(stable_hash(body), **body))
        self.intervention_requests = tuple(requests)
        candidates: list[SkillInternalGraph] = []
        for motif in parsed.get("motifs", []):
            nodes: list[InternalGraphNode] = []
            for raw_node in motif.get("nodes", []):
                occurrences: list[NodeOccurrence] = []
                for raw_occurrence in raw_node.get("occurrences", []):
                    execution = discovery[_bounded(raw_occurrence["execution_index"], len(discovery))]
                    start = _bounded(raw_occurrence["start_offset"], len(execution.transition_receipt_ids))
                    end = _bounded(raw_occurrence["end_offset"], len(execution.transition_receipt_ids))
                    if end < start:
                        raise ValueError("node occurrence has reversed offsets")
                    occurrences.append(NodeOccurrence(
                        execution.execution_id, start, end,
                        execution.transition_receipt_ids[start:end + 1],
                    ))
                nodes.append(InternalGraphNode(
                    str(raw_node["node_id"]), tuple(occurrences), str(raw_node.get("role", ""))
                ))
            edges = _derive_supported_edges(nodes, discovery, motif.get("edges", []))
            hash_content = {
                "execution_set_id": execution_set.execution_set_id,
                "skill_id_hash": stable_hash(execution_set.skill_id),
                "nodes": [asdict(node) for node in nodes],
                "edges": [asdict(edge) for edge in edges],
                "discovery_execution_ids": tuple(row.execution_id for row in discovery),
                "status": "CANDIDATE",
                "untrusted_description": str(motif.get("description", "")),
            }
            candidates.append(SkillInternalGraph(
                stable_hash(hash_content), execution_set.execution_set_id,
                stable_hash(execution_set.skill_id), tuple(nodes), tuple(edges),
                tuple(row.execution_id for row in discovery), "CANDIDATE",
                str(motif.get("description", "")),
            ))
        return tuple(candidates)


def _bounded(value: Any, length: int) -> int:
    index = int(value)
    if index < 0 or index >= length:
        raise ValueError(f"out-of-range index {index} for length {length}")
    return index


def _derive_supported_edges(
    nodes: Sequence[InternalGraphNode],
    executions: Sequence[SkillExecution],
    proposed_edges: Sequence[Mapping[str, Any]],
) -> tuple[InternalGraphEdge, ...]:
    node_ids = {node.node_id for node in nodes}
    occurrence_at_start: dict[tuple[str, int], str] = {}
    occurrence_at_end: dict[tuple[str, int], str] = {}
    for node in nodes:
        for occurrence in node.occurrences:
            occurrence_at_start[(occurrence.execution_id, occurrence.start_offset)] = node.node_id
            occurrence_at_end[(occurrence.execution_id, occurrence.end_offset)] = node.node_id
    result: list[InternalGraphEdge] = []
    for raw in proposed_edges:
        source, target = str(raw["source"]), str(raw["target"])
        if source not in node_ids or target not in node_ids:
            raise ValueError("edge references unknown node")
        boundaries: list[tuple[str, int]] = []
        for execution in executions:
            for offset in range(len(execution.transition_receipt_ids) - 1):
                if (
                    occurrence_at_end.get((execution.execution_id, offset)) == source
                    and occurrence_at_start.get((execution.execution_id, offset + 1)) == target
                ):
                    boundaries.append((execution.execution_id, offset))
        if not boundaries:
            raise ValueError("edge has no observed adjacent occurrence")
        result.append(InternalGraphEdge(
            source, target, tuple(boundaries), (), str(raw.get("condition", ""))
        ))
    return tuple(result)


def audit_internal_graph(
    graph: SkillInternalGraph,
    execution_set: GroundedSkillExecutionSet,
    records_by_receipt: Mapping[str, SourcePolicyStepRecord] | None = None,
) -> InternalGraphAudit:
    failures: list[str] = []
    if graph.execution_set_id != execution_set.execution_set_id:
        failures.append("EXECUTION_SET_MISMATCH")
    known_executions = {row.execution_id: row for row in execution_set.executions}
    node_ids = [node.node_id for node in graph.nodes]
    if len(node_ids) != len(set(node_ids)):
        failures.append("DUPLICATE_NODE_ID")
    occupied: set[tuple[str, int]] = set()
    for node in graph.nodes:
        if not node.occurrences:
            failures.append("EMPTY_NODE")
        for occurrence in node.occurrences:
            execution = known_executions.get(occurrence.execution_id)
            if execution is None or execution.split != "discovery":
                failures.append("NON_DISCOVERY_OCCURRENCE")
                continue
            expected = execution.transition_receipt_ids[
                occurrence.start_offset:occurrence.end_offset + 1
            ]
            if expected != occurrence.transition_receipt_ids:
                failures.append("OCCURRENCE_RECEIPT_MISMATCH")
            for offset in range(occurrence.start_offset, occurrence.end_offset + 1):
                key = (occurrence.execution_id, offset)
                if key in occupied:
                    failures.append("OVERLAPPING_OCCURRENCES")
                occupied.add(key)
    if not graph.is_nontrivial:
        failures.append("GENERIC_LINEAR_OR_TRIVIAL_GRAPH")
    control_flags: list[str] = []
    if records_by_receipt is not None and graph.nodes:
        action_sets = []
        for node in graph.nodes:
            actions = {
                records_by_receipt[receipt_id].action
                for occurrence in node.occurrences
                for receipt_id in occurrence.transition_receipt_ids
                if receipt_id in records_by_receipt
            }
            action_sets.append(actions)
        if (
            all(len(actions) == 1 for actions in action_sets)
            and len({next(iter(actions)) for actions in action_sets}) == len(action_sets)
        ):
            control_flags.append("ACTION_IDENTITY_EXPLAINS_NODE_PARTITION")
    accepted = not failures
    return InternalGraphAudit(
        accepted,
        tuple(sorted(set(failures))),
        graph.is_nontrivial,
        len(graph.edges),
        accepted and not control_flags,
        tuple(control_flags),
    )
