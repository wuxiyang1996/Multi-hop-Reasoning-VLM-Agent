from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping, Protocol, Sequence

from .contracts import stable_hash
from .instrumented_import import ImportedSourceEpisode
from .source_ranking import segment_native_policy


SPLITS = ("discovery", "qualification", "held_out")


def _sign(value: float) -> str:
    return "POSITIVE" if value > 0 else "NEGATIVE" if value < 0 else "ZERO"


@dataclass(frozen=True)
class ExecutionSignature:
    skill_relation: str
    length: int
    return_sign: str
    reward_sign_sequence: tuple[str, ...]
    action_repeat_sequence: tuple[str, ...]
    terminal: bool


@dataclass(frozen=True)
class SkillExecutionEvidence:
    execution_id: str
    segment_receipt_id: str
    episode_id: str
    start_step: int
    end_step: int
    transition_receipt_ids: tuple[str, ...]
    skill_token: str
    action_tokens: tuple[str, ...]
    signature: ExecutionSignature
    untrusted_reasoning: tuple[str, ...]

    def validate(self) -> bool:
        body = asdict(self)
        execution_id = body.pop("execution_id")
        return execution_id == stable_hash(body)


@dataclass(frozen=True)
class ExecutionTrace:
    trace_id: str
    game: str
    episode_id: str
    split: str
    executions: tuple[SkillExecutionEvidence, ...]

    def validate(self) -> bool:
        body = asdict(self)
        trace_id = body.pop("trace_id")
        return (
            trace_id == stable_hash(body)
            and self.split in SPLITS
            and all(row.validate() for row in self.executions)
            and all(
                right.start_step == left.end_step + 1
                for left, right in zip(self.executions, self.executions[1:])
            )
        )


@dataclass(frozen=True)
class ExecutionMotifNode:
    node_id: str
    execution_ids: tuple[str, ...]
    untrusted_role: str = ""


@dataclass(frozen=True)
class ExecutionMotifEdge:
    source: str
    target: str
    supporting_boundaries: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ExecutionMotifGraph:
    graph_id: str
    game: str
    nodes: tuple[ExecutionMotifNode, ...]
    edges: tuple[ExecutionMotifEdge, ...]
    discovery_trace_ids: tuple[str, ...]
    untrusted_description: str = ""


@dataclass(frozen=True)
class ExecutionGraphAudit:
    accepted: bool
    failure_codes: tuple[str, ...]
    discovery_execution_coverage: float
    observed_edges: int
    nodes: int
    edges: int
    has_branch: bool
    has_cycle: bool
    recurrent_nodes: bool
    recurrent_edges: bool
    single_field_shortcuts: tuple[str, ...]


class JSONBackend(Protocol):
    def complete(self, role: str, system: str, payload: Mapping[str, Any]) -> str: ...


def _split_by_episode(episodes: Sequence[ImportedSourceEpisode]) -> dict[str, str]:
    ids = sorted({episode.episode_id for episode in episodes})
    return {episode_id: SPLITS[index % 3] for index, episode_id in enumerate(ids)}


def build_execution_traces(
    episodes: Sequence[ImportedSourceEpisode],
) -> tuple[ExecutionTrace, ...]:
    """Promote old maximal skill-run sub-episodes to receipt-bound graph events."""

    split_by_episode = _split_by_episode(episodes)
    traces = []
    for episode in sorted(episodes, key=lambda row: row.episode_id):
        skill_aliases: dict[str | None, str] = {}
        action_aliases: dict[str, str] = {}
        previous_skill: str | None = None
        executions = []
        for segment, records in segment_native_policy(episode.records):
            skill_id = records[0].selected_skill_id
            if skill_id not in skill_aliases:
                skill_aliases[skill_id] = f"S{len(skill_aliases)}"
            for record in records:
                if record.action not in action_aliases:
                    action_aliases[record.action] = f"A{len(action_aliases)}"
            actions = tuple(action_aliases[row.action] for row in records)
            repeats = tuple(
                "START" if index == 0 else
                ("SAME" if actions[index] == actions[index - 1] else "CHANGED")
                for index in range(len(actions))
            )
            signature = ExecutionSignature(
                skill_relation=(
                    "START" if previous_skill is None else
                    ("SAME" if skill_id == previous_skill else "CHANGED")
                ),
                length=len(records),
                return_sign=_sign(sum(row.reward for row in records)),
                reward_sign_sequence=tuple(_sign(row.reward) for row in records),
                action_repeat_sequence=repeats,
                terminal=bool(records[-1].after.terminal),
            )
            body = {
                "segment_receipt_id": segment.receipt_id,
                "episode_id": episode.episode_id,
                "start_step": segment.start_step,
                "end_step": segment.end_step,
                "transition_receipt_ids": segment.transition_receipt_ids,
                "skill_token": skill_aliases[skill_id],
                "action_tokens": actions,
                "signature": asdict(signature),
                # Reasoning is an untrusted proposal feature, never verifier evidence.
                "untrusted_reasoning": tuple(
                    row.action_reasoning for row in records if row.action_reasoning
                ),
            }
            executions.append(SkillExecutionEvidence(
                stable_hash(body), **{**body, "signature": signature}
            ))
            previous_skill = skill_id
        trace_body = {
            "game": episode.game,
            "episode_id": episode.episode_id,
            "split": split_by_episode[episode.episode_id],
            "executions": tuple(asdict(row) for row in executions),
        }
        traces.append(ExecutionTrace(
            stable_hash(trace_body), episode.game, episode.episode_id,
            split_by_episode[episode.episode_id], tuple(executions),
        ))
    return tuple(traces)


def compact_execution_payload(
    traces: Sequence[ExecutionTrace],
    *,
    include_reasoning: bool,
) -> list[dict[str, Any]]:
    return [{
        "trace_id": trace.trace_id,
        "executions": [{
            "offset": offset,
            "execution_id": row.execution_id,
            "segment_receipt_id": row.segment_receipt_id,
            "skill": row.skill_token,
            "actions": row.action_tokens,
            "signature": asdict(row.signature),
            **({"untrusted_reasoning": row.untrusted_reasoning} if include_reasoning else {}),
        } for offset, row in enumerate(trace.executions)],
    } for trace in traces]


def graph_from_response(
    game: str,
    traces: Sequence[ExecutionTrace],
    response: Mapping[str, Any],
) -> ExecutionMotifGraph:
    discovery = [trace for trace in traces if trace.split == "discovery"]
    nodes = tuple(ExecutionMotifNode(
        str(raw["node_id"]),
        tuple(str(item) for item in raw.get("execution_ids", ())),
        str(raw.get("role", "")),
    ) for raw in response.get("nodes", ()))
    assignment = {
        execution_id: node.node_id
        for node in nodes for execution_id in node.execution_ids
    }
    edges = []
    for raw in response.get("edges", ()):
        source, target = str(raw["source"]), str(raw["target"])
        boundaries = []
        for trace in discovery:
            for offset, (left, right) in enumerate(
                zip(trace.executions, trace.executions[1:])
            ):
                if (
                    assignment.get(left.execution_id) == source
                    and assignment.get(right.execution_id) == target
                ):
                    boundaries.append((trace.trace_id, offset))
        edges.append(ExecutionMotifEdge(source, target, tuple(boundaries)))
    body = {
        "game": game,
        "nodes": tuple(asdict(row) for row in nodes),
        "edges": tuple(asdict(row) for row in edges),
        "discovery_trace_ids": tuple(trace.trace_id for trace in discovery),
        "untrusted_description": str(response.get("description", "")),
    }
    return ExecutionMotifGraph(
        stable_hash(body), **{**body, "nodes": nodes, "edges": tuple(edges)}
    )


def audit_execution_graph(
    graph: ExecutionMotifGraph,
    traces: Sequence[ExecutionTrace],
) -> ExecutionGraphAudit:
    discovery = [trace for trace in traces if trace.split == "discovery"]
    event_index = {
        row.execution_id: row
        for trace in discovery for row in trace.executions
    }
    failures = []
    node_ids = [node.node_id for node in graph.nodes]
    if len(node_ids) != len(set(node_ids)):
        failures.append("DUPLICATE_NODE_ID")
    assignment = {}
    support_by_node: dict[str, list[SkillExecutionEvidence]] = defaultdict(list)
    for node in graph.nodes:
        for execution_id in node.execution_ids:
            if execution_id not in event_index:
                failures.append("NODE_REFERENCES_NON_DISCOVERY_EXECUTION")
            if execution_id in assignment:
                failures.append("EXECUTION_ASSIGNED_TO_MULTIPLE_NODES")
            assignment[execution_id] = node.node_id
            if execution_id in event_index:
                support_by_node[node.node_id].append(event_index[execution_id])
    recurrent_nodes = bool(graph.nodes) and all(
        len({row.episode_id for row in support_by_node[node.node_id]}) >= 2
        for node in graph.nodes
    )
    if not recurrent_nodes:
        failures.append("NODE_NOT_RECURRENT_ACROSS_DISCOVERY_EPISODES")
    observed_edges = 0
    adjacency: dict[str, set[str]] = defaultdict(set)
    recurrent_edges = bool(graph.edges)
    for edge in graph.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            failures.append("EDGE_REFERENCES_UNKNOWN_NODE")
            continue
        expected = []
        for trace in discovery:
            for offset, (left, right) in enumerate(
                zip(trace.executions, trace.executions[1:])
            ):
                if (
                    assignment.get(left.execution_id) == edge.source
                    and assignment.get(right.execution_id) == edge.target
                ):
                    expected.append((trace.trace_id, offset))
        if not expected:
            failures.append("EDGE_HAS_NO_OBSERVED_EXECUTION_BOUNDARY")
        if tuple(expected) != edge.supporting_boundaries:
            failures.append("EDGE_BOUNDARY_RECEIPT_MISMATCH")
        if len({trace_id for trace_id, _ in expected}) < 2:
            recurrent_edges = False
        observed_edges += len(expected)
        adjacency[edge.source].add(edge.target)
    if not recurrent_edges:
        failures.append("EDGE_NOT_RECURRENT_ACROSS_DISCOVERY_EPISODES")
    has_branch = any(len(targets) > 1 for targets in adjacency.values())

    def reaches(start: str, current: str, seen: set[str]) -> bool:
        for target in adjacency.get(current, set()):
            if target == start:
                return True
            if target not in seen and reaches(start, target, seen | {target}):
                return True
        return False

    has_cycle = any(reaches(node_id, node_id, {node_id}) for node_id in node_ids)
    if len(node_ids) < 2 or not graph.edges or not (has_branch or has_cycle):
        failures.append("TRIVIAL_EXECUTION_GRAPH")
    total = len(event_index)
    coverage = len(assignment) / total if total else 0.0
    if coverage == 0:
        failures.append("ZERO_DISCOVERY_COVERAGE")
    shortcut_extractors = {
        "SKILL_TOKEN": lambda row: row.skill_token,
        "LENGTH": lambda row: row.signature.length,
        "RETURN_SIGN": lambda row: row.signature.return_sign,
        "ACTION_SEQUENCE": lambda row: row.action_tokens,
        "ACTION_REPEAT_SEQUENCE": lambda row: row.signature.action_repeat_sequence,
        "REWARD_SIGN_SEQUENCE": lambda row: row.signature.reward_sign_sequence,
    }
    shortcuts = []
    assigned_rows = [
        (event_index[execution_id], node_id)
        for execution_id, node_id in assignment.items()
        if execution_id in event_index
    ]
    if len(set(assignment.values())) >= 2:
        for name, extractor in shortcut_extractors.items():
            labels_by_value: dict[Any, set[str]] = defaultdict(set)
            for row, node_id in assigned_rows:
                labels_by_value[extractor(row)].add(node_id)
            # A lookup on one recorded scalar/sequence exactly recovers every
            # proposed node label. This is a measurable shortcut, not a
            # semantic judgement about what a role name means.
            if labels_by_value and all(
                len(labels) == 1 for labels in labels_by_value.values()
            ):
                shortcuts.append(name)
    for shortcut in shortcuts:
        failures.append(f"SINGLE_FIELD_SHORTCUT_{shortcut}")
    return ExecutionGraphAudit(
        not failures, tuple(sorted(set(failures))), coverage, observed_edges,
        len(node_ids), len(graph.edges), has_branch, has_cycle,
        recurrent_nodes, recurrent_edges, tuple(shortcuts),
    )


class SourceExecutionMotifAgent:
    def __init__(self, backend: JSONBackend):
        self.backend = backend
        self.last_call: dict[str, Any] = {}

    def propose(
        self,
        game: str,
        traces: Sequence[ExecutionTrace],
        *,
        include_reasoning: bool = True,
    ) -> ExecutionMotifGraph:
        discovery = [trace for trace in traces if trace.split == "discovery"]
        payload = {
            "schema_version": "SOURCE_SKILL_EXECUTION_MOTIF_DISCOVERY_V1",
            "game_identity_hash": stable_hash(game),
            "evidence_granularity": "MAXIMAL_RECORDED_SKILL_EXECUTION",
            "traces": compact_execution_payload(
                discovery, include_reasoning=include_reasoning,
            ),
        }
        system = (
            "Propose at most one non-trivial control motif over complete recorded skill "
            "executions. Each supplied execution is already a receipt-bound sub-episode; "
            "never split it into motor actions. Skill and action tokens are episode-local "
            "alpha aliases without semantics. Nodes must list disjoint exact execution_ids. "
            "Edges must be observed adjacency between executions. A node partition explained "
            "only by action identity, skill identity, duration, or reward sign is not a reasoning "
            "motif; return an empty graph if no stronger branching, recovery, verification, or "
            "termination organization is supported. Do not mention or infer a target domain. "
            "Return JSON with description, nodes [{node_id,execution_ids,role}], and edges "
            "[{source,target}]."
        )
        raw = self.backend.complete("source_execution_motif", system, payload)
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        self.last_call = {
            "role": "source_execution_motif",
            "include_reasoning": include_reasoning,
            "payload_hash": stable_hash(payload),
            "response_hash": stable_hash(parsed),
            "usage": dict(getattr(self.backend, "last_usage", {}) or {}),
        }
        return graph_from_response(game, traces, parsed)


def execution_affordance_report(
    traces: Sequence[ExecutionTrace],
) -> dict[str, Any]:
    result = {}
    for split in SPLITS:
        selected = [trace for trace in traces if trace.split == split]
        result[split] = {
            "episodes": len(selected),
            "executions": sum(len(trace.executions) for trace in selected),
            "cross_execution_boundaries": sum(
                max(0, len(trace.executions) - 1) for trace in selected
            ),
            "episodes_with_at_least_three_executions": sum(
                len(trace.executions) >= 3 for trace in selected
            ),
            "positive_return_executions": sum(
                row.signature.return_sign == "POSITIVE"
                for trace in selected for row in trace.executions
            ),
        }
    return {
        "schema_version": "SOURCE_EXECUTION_AFFORDANCE_V1",
        "traces_valid": all(trace.validate() for trace in traces),
        "split_stats": result,
    }


__all__ = [
    "ExecutionSignature", "SkillExecutionEvidence", "ExecutionTrace",
    "ExecutionMotifNode", "ExecutionMotifEdge", "ExecutionMotifGraph",
    "ExecutionGraphAudit", "build_execution_traces", "compact_execution_payload",
    "graph_from_response", "audit_execution_graph", "SourceExecutionMotifAgent",
    "execution_affordance_report",
]
