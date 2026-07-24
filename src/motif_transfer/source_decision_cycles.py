from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping, Protocol, Sequence

from .contracts import SourcePolicyStepRecord, stable_hash
from .instrumented_import import ImportedSourceEpisode


SPLITS = ("discovery", "qualification", "held_out")
SIGNATURE_FIELDS = (
    "skill_relation", "action_relation", "action_origin", "reward_sign", "terminal",
)


def _relation(previous: str | None, current: str | None) -> str:
    if current is None:
        return "MISSING"
    if previous is None:
        return "START"
    return "SAME" if previous == current else "CHANGED"


def _reward_sign(value: float) -> str:
    return "POSITIVE" if value > 0 else "NEGATIVE" if value < 0 else "ZERO"


@dataclass(frozen=True)
class AnonymousEventSignature:
    """Interface-free event attributes already present in source receipts."""

    skill_relation: str
    action_relation: str
    action_origin: str
    reward_sign: str
    terminal: bool


@dataclass(frozen=True)
class SourceDecisionEvent:
    event_id: str
    episode_id: str
    step: int
    transition_receipt_id: str
    skill_token: str
    action_token: str
    signature: AnonymousEventSignature
    untrusted_reasoning: str

    def validate(self) -> bool:
        body = asdict(self)
        event_id = body.pop("event_id")
        return event_id == stable_hash(body)


@dataclass(frozen=True)
class SourceDecisionTrace:
    trace_id: str
    game: str
    episode_id: str
    split: str
    events: tuple[SourceDecisionEvent, ...]

    def validate(self) -> bool:
        body = asdict(self)
        trace_id = body.pop("trace_id")
        return (
            trace_id == stable_hash(body)
            and self.split in SPLITS
            and all(event.validate() for event in self.events)
            and all(
                right.step == left.step + 1
                for left, right in zip(self.events, self.events[1:])
            )
        )


@dataclass(frozen=True)
class DecisionGraphNode:
    node_id: str
    event_ids: tuple[str, ...]
    untrusted_role: str = ""


@dataclass(frozen=True)
class DecisionGraphEdge:
    source: str
    target: str
    supporting_boundaries: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class SourceDecisionGraph:
    graph_id: str
    game: str
    nodes: tuple[DecisionGraphNode, ...]
    edges: tuple[DecisionGraphEdge, ...]
    discovery_trace_ids: tuple[str, ...]
    untrusted_description: str = ""

    @property
    def is_nontrivial(self) -> bool:
        if len(self.nodes) < 2 or not self.edges:
            return False
        outgoing: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            outgoing[edge.source].add(edge.target)
        has_branch = any(len(targets) > 1 for targets in outgoing.values())
        has_loop = any(edge.source == edge.target for edge in self.edges)
        adjacency = {node.node_id: outgoing[node.node_id] for node in self.nodes}

        def reaches_start(start: str, current: str, seen: set[str]) -> bool:
            for target in adjacency.get(current, set()):
                if target == start:
                    return True
                if target not in seen and reaches_start(start, target, seen | {target}):
                    return True
            return False

        return has_branch or has_loop or any(
            reaches_start(node.node_id, node.node_id, {node.node_id})
            for node in self.nodes
        )


@dataclass(frozen=True)
class DecisionGraphAudit:
    accepted: bool
    failure_codes: tuple[str, ...]
    discovery_event_coverage: float
    observed_edges: int
    nontrivial: bool
    crosses_recorded_skill_boundary: bool


@dataclass(frozen=True)
class BlindPrediction:
    query_id: str
    graph_id: str
    trace_id: str
    prefix_end_offset: int
    current_node_id: str
    predicted_next_node_id: str
    predicted_signature: AnonymousEventSignature
    abstained: bool = False
    condition: str = "AUTHENTIC_GRAPH"


class JSONBackend(Protocol):
    def complete(self, role: str, system: str, payload: Mapping[str, Any]) -> str: ...


def _split_by_episode(episodes: Sequence[ImportedSourceEpisode]) -> dict[str, str]:
    ids = sorted({episode.episode_id for episode in episodes})
    return {episode_id: SPLITS[index % 3] for index, episode_id in enumerate(ids)}


def build_decision_traces(
    episodes: Sequence[ImportedSourceEpisode],
) -> tuple[SourceDecisionTrace, ...]:
    """Build whole decision cycles; never cut at a selected-skill boundary."""

    split_by_episode = _split_by_episode(episodes)
    traces: list[SourceDecisionTrace] = []
    for episode in sorted(episodes, key=lambda row: row.episode_id):
        ordered = sorted(episode.records, key=lambda row: row.step)
        # An episode may contain import gaps. Preserve maximal contiguous spans
        # instead of inventing adjacency across a missing transition.
        spans: list[list[SourcePolicyStepRecord]] = []
        current: list[SourcePolicyStepRecord] = []
        for record in ordered:
            if current and record.step != current[-1].step + 1:
                spans.append(current)
                current = []
            current.append(record)
        if current:
            spans.append(current)
        for span_index, span in enumerate(spans):
            skill_tokens: dict[str | None, str] = {}
            action_tokens: dict[str, str] = {}
            events: list[SourceDecisionEvent] = []
            previous_skill: str | None = None
            previous_action: str | None = None
            for record in span:
                if record.selected_skill_id not in skill_tokens:
                    skill_tokens[record.selected_skill_id] = f"S{len(skill_tokens)}"
                if record.action not in action_tokens:
                    action_tokens[record.action] = f"A{len(action_tokens)}"
                signature = AnonymousEventSignature(
                    skill_relation=_relation(previous_skill, record.selected_skill_id),
                    action_relation=_relation(previous_action, record.action),
                    action_origin=record.action_origin,
                    reward_sign=_reward_sign(record.reward),
                    terminal=record.after.terminal,
                )
                body = {
                    "episode_id": episode.episode_id,
                    "step": record.step,
                    "transition_receipt_id": record.transition.receipt_id,
                    "skill_token": skill_tokens[record.selected_skill_id],
                    "action_token": action_tokens[record.action],
                    "signature": asdict(signature),
                    "untrusted_reasoning": record.action_reasoning,
                }
                events.append(SourceDecisionEvent(stable_hash(body), **{
                    **body, "signature": signature,
                }))
                previous_skill, previous_action = record.selected_skill_id, record.action
            trace_body = {
                "game": episode.game,
                "episode_id": episode.episode_id,
                "split": split_by_episode[episode.episode_id],
                "events": tuple(asdict(event) for event in events),
            }
            traces.append(SourceDecisionTrace(
                stable_hash(trace_body), episode.game, episode.episode_id,
                split_by_episode[episode.episode_id], tuple(events),
            ))
    return tuple(traces)


def compact_trace_payload(
    traces: Sequence[SourceDecisionTrace],
    *,
    include_reasoning: bool,
) -> list[dict[str, Any]]:
    return [{
        "trace_id": trace.trace_id,
        "events": [{
            "offset": offset,
            "event_id": event.event_id,
            "skill": event.skill_token,
            "action": event.action_token,
            "signature": asdict(event.signature),
            **({"untrusted_reasoning": event.untrusted_reasoning} if include_reasoning else {}),
        } for offset, event in enumerate(trace.events)],
    } for trace in traces]


def audit_graph(
    graph: SourceDecisionGraph,
    traces: Sequence[SourceDecisionTrace],
) -> DecisionGraphAudit:
    discovery = [trace for trace in traces if trace.split == "discovery"]
    event_index = {
        event.event_id: (trace, offset, event)
        for trace in discovery for offset, event in enumerate(trace.events)
    }
    failures: list[str] = []
    node_ids = [node.node_id for node in graph.nodes]
    if len(node_ids) != len(set(node_ids)):
        failures.append("DUPLICATE_NODE_ID")
    assigned: dict[str, str] = {}
    for node in graph.nodes:
        for event_id in node.event_ids:
            if event_id not in event_index:
                failures.append("NODE_REFERENCES_NON_DISCOVERY_EVENT")
            if event_id in assigned:
                failures.append("EVENT_ASSIGNED_TO_MULTIPLE_NODES")
            assigned[event_id] = node.node_id
    observed_edges = 0
    crosses_skill = False
    for edge in graph.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            failures.append("EDGE_REFERENCES_UNKNOWN_NODE")
            continue
        expected = []
        for trace in discovery:
            for offset in range(len(trace.events) - 1):
                left, right = trace.events[offset:offset + 2]
                if assigned.get(left.event_id) == edge.source and assigned.get(right.event_id) == edge.target:
                    expected.append((trace.trace_id, offset))
                    crosses_skill |= right.signature.skill_relation == "CHANGED"
        if not expected:
            failures.append("EDGE_HAS_NO_OBSERVED_BOUNDARY")
        if tuple(expected) != edge.supporting_boundaries:
            failures.append("EDGE_BOUNDARY_RECEIPT_MISMATCH")
        observed_edges += len(expected)
    total = sum(len(trace.events) for trace in discovery)
    coverage = len(assigned) / total if total else 0.0
    nontrivial = graph.is_nontrivial
    if not nontrivial:
        failures.append("TRIVIAL_GRAPH")
    if coverage == 0:
        failures.append("ZERO_DISCOVERY_COVERAGE")
    return DecisionGraphAudit(
        not failures, tuple(sorted(set(failures))), coverage,
        observed_edges, nontrivial, crosses_skill,
    )


def graph_from_agent_response(
    game: str,
    traces: Sequence[SourceDecisionTrace],
    response: Mapping[str, Any],
) -> SourceDecisionGraph:
    discovery = [trace for trace in traces if trace.split == "discovery"]
    by_id = {event.event_id: event for trace in discovery for event in trace.events}
    nodes = tuple(DecisionGraphNode(
        str(raw["node_id"]),
        tuple(str(event_id) for event_id in raw.get("event_ids", ())),
        str(raw.get("role", "")),
    ) for raw in response.get("nodes", ()))
    assignment = {
        event_id: node.node_id for node in nodes for event_id in node.event_ids
    }
    edges = []
    for raw in response.get("edges", ()):
        source, target = str(raw["source"]), str(raw["target"])
        boundaries = []
        for trace in discovery:
            for offset in range(len(trace.events) - 1):
                left, right = trace.events[offset:offset + 2]
                if assignment.get(left.event_id) == source and assignment.get(right.event_id) == target:
                    boundaries.append((trace.trace_id, offset))
        edges.append(DecisionGraphEdge(source, target, tuple(boundaries)))
    body = {
        "game": game,
        "nodes": tuple(asdict(node) for node in nodes),
        "edges": tuple(asdict(edge) for edge in edges),
        "discovery_trace_ids": tuple(trace.trace_id for trace in discovery),
        "untrusted_description": str(response.get("description", "")),
    }
    graph = SourceDecisionGraph(stable_hash(body), **{
        **body, "nodes": nodes, "edges": tuple(edges),
    })
    # Fail immediately on fabricated event IDs rather than silently dropping.
    unknown = set(assignment) - set(by_id)
    if unknown:
        raise ValueError(f"agent referenced {len(unknown)} unknown discovery events")
    return graph


class SourceDecisionCycleAgent:
    def __init__(self, backend: JSONBackend):
        self.backend = backend
        self.last_calls: list[dict[str, Any]] = []

    def propose(
        self,
        game: str,
        traces: Sequence[SourceDecisionTrace],
    ) -> SourceDecisionGraph:
        discovery = [trace for trace in traces if trace.split == "discovery"]
        payload = {
            "schema_version": "SOURCE_DECISION_CYCLE_DISCOVERY_V1",
            "game_identity_hash": stable_hash(game),
            "traces": compact_trace_payload(discovery, include_reasoning=True),
        }
        system = (
            "Propose one opaque multi-step control graph from discovery source receipts. "
            "The skill/action tokens are alpha-renamed within each episode and their names have "
            "no semantics. Nodes must list exact supplied event_ids. Edges must reflect observed "
            "adjacency. Look across skill boundaries when the recorded evidence supports it. "
            "Do not invent target concepts. Return one JSON object with description, nodes "
            "[{node_id,event_ids,role}], and edges [{source,target}]. Prefer an empty graph over "
            "a repeated motor pattern or a generic linear observe-act chain."
        )
        raw = self.backend.complete("source_decision_graph", system, payload)
        parsed = __import__("json").loads(raw) if isinstance(raw, str) else raw
        self.last_calls.append({
            "role": "source_decision_graph",
            "payload_hash": stable_hash(payload),
            "response_hash": stable_hash(parsed),
            "usage": dict(getattr(self.backend, "last_usage", {}) or {}),
        })
        return graph_from_agent_response(game, traces, parsed)

    def predict(
        self,
        graph: SourceDecisionGraph,
        traces: Sequence[SourceDecisionTrace],
        *,
        split: str,
        maximum_queries_per_trace: int = 8,
        condition: str = "AUTHENTIC_GRAPH",
        edge_override: Sequence[tuple[str, str]] | None = None,
    ) -> tuple[BlindPrediction, ...]:
        if split not in {"qualification", "held_out"}:
            raise ValueError("blind prediction split must be qualification or held_out")
        if maximum_queries_per_trace < 1:
            raise ValueError("maximum_queries_per_trace must be positive")
        evaluation = [trace for trace in traces if trace.split == split]
        discovery_events = {
            event.event_id: event
            for trace in traces if trace.split == "discovery"
            for event in trace.events
        }
        graph_edges = (
            tuple(edge_override) if edge_override is not None
            else tuple((edge.source, edge.target) for edge in graph.edges)
        )
        node_evidence = [{
            "node_id": node.node_id,
            "receipts": [{
                "event_id": event_id,
                "signature": asdict(discovery_events[event_id].signature),
                "skill": discovery_events[event_id].skill_token,
                "action": discovery_events[event_id].action_token,
            } for event_id in node.event_ids if event_id in discovery_events],
        } for node in graph.nodes]
        queries: list[dict[str, Any]] = []
        for trace in evaluation:
            offsets = list(range(len(trace.events) - 1))
            offsets.sort(key=lambda offset: stable_hash({
                "trace_id": trace.trace_id,
                "offset": offset,
                "sampling_contract": "OUTCOME_BLIND_HASH_ORDER_V1",
            }))
            for offset in sorted(offsets[:maximum_queries_per_trace]):
                prefix = trace.events[:offset + 1]
                query_id = stable_hash({
                    "graph_id": graph.graph_id,
                    "trace_id": trace.trace_id,
                    "prefix_event_ids": tuple(event.event_id for event in prefix),
                    "condition": condition,
                })
                queries.append({
                    "query_id": query_id,
                    "trace_id": trace.trace_id,
                    "prefix_end_offset": offset,
                    "prefix": compact_trace_payload([
                        SourceDecisionTrace(
                            trace.trace_id, trace.game, trace.episode_id,
                            trace.split, prefix,
                        )
                    ], include_reasoning=False)[0]["events"],
                })
        system = (
            "Use the frozen discovery graph to make one blind next-event prediction. The request "
            "contains exactly one evaluation prefix; the next event and every later event are "
            "hidden. Discovery node receipts define the opaque nodes but are not target semantics. "
            "Return one JSON object under prediction with the exact query_id, "
            "current_node_id, predicted_next_node_id, predicted_signature, and abstained. "
            "Node IDs must come from the frozen graph. Do not add nodes or target semantics."
        )
        node_ids = {node.node_id for node in graph.nodes}
        predictions = []
        for query in queries:
            payload = {
                "schema_version": "SOURCE_DECISION_CYCLE_BLIND_PREDICTION_V2",
                "condition": condition,
                "graph": {
                    "graph_id": graph.graph_id,
                    "nodes": node_evidence,
                    "edges": [
                        {"source": source, "target": target}
                        for source, target in graph_edges
                    ],
                },
                "query": query,
                "allowed_signature_values": {
                    "skill_relation": ["MISSING", "START", "SAME", "CHANGED"],
                    "action_relation": ["MISSING", "START", "SAME", "CHANGED"],
                    "action_origin": ["AGENT", "POLICY_POSTPROCESSOR", "FALLBACK"],
                    "reward_sign": ["NEGATIVE", "ZERO", "POSITIVE"],
                    "terminal": [False, True],
                },
            }
            raw = self.backend.complete("source_decision_prediction", system, payload)
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            self.last_calls.append({
                "role": "source_decision_prediction",
                "split": split,
                "condition": condition,
                "query_id": query["query_id"],
                "payload_hash": stable_hash(payload),
                "response_hash": stable_hash(parsed),
                "usage": dict(getattr(self.backend, "last_usage", {}) or {}),
            })
            raw_prediction = parsed.get("prediction", parsed)
            query_id = str(raw_prediction["query_id"])
            if query_id != query["query_id"]:
                raise ValueError("prediction references unknown query")
            current = str(raw_prediction["current_node_id"])
            following = str(raw_prediction["predicted_next_node_id"])
            if current not in node_ids or following not in node_ids:
                raise ValueError("prediction references unknown graph node")
            signature = AnonymousEventSignature(**raw_prediction["predicted_signature"])
            predictions.append(BlindPrediction(
                query_id, graph.graph_id, str(query["trace_id"]),
                int(query["prefix_end_offset"]), current, following, signature,
                bool(raw_prediction.get("abstained", False)), condition,
            ))
        return tuple(predictions)


def shuffled_graph_edges(graph: SourceDecisionGraph) -> tuple[tuple[str, str], ...]:
    """Deterministic topology control; does not inspect evaluation outcomes."""

    node_ids = sorted(node.node_id for node in graph.nodes)
    if len(node_ids) < 2:
        return tuple((edge.source, edge.target) for edge in graph.edges)
    shift = 1 + int(stable_hash(graph.graph_id)[:8], 16) % (len(node_ids) - 1)
    replacement = {
        node_id: node_ids[(index + shift) % len(node_ids)]
        for index, node_id in enumerate(node_ids)
    }
    return tuple((edge.source, replacement[edge.target]) for edge in graph.edges)


def score_blind_predictions(
    graph: SourceDecisionGraph,
    traces: Sequence[SourceDecisionTrace],
    predictions: Sequence[BlindPrediction],
    *,
    split: str,
    edge_override: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    evaluation = {trace.trace_id: trace for trace in traces if trace.split == split}
    edge_set = (
        set(edge_override) if edge_override is not None
        else {(edge.source, edge.target) for edge in graph.edges}
    )
    discovery_next = [
        event.signature for trace in traces if trace.split == "discovery"
        for event in trace.events[1:]
    ]
    majority = {}
    for field in SIGNATURE_FIELDS:
        values = [getattr(signature, field) for signature in discovery_next]
        majority[field] = Counter(values).most_common(1)[0][0] if values else None
    scored = []
    for prediction in predictions:
        trace = evaluation.get(prediction.trace_id)
        if trace is None or prediction.prefix_end_offset + 1 >= len(trace.events):
            raise ValueError("prediction cannot be joined to hidden next receipt")
        actual = trace.events[prediction.prefix_end_offset + 1].signature
        matches = {
            field: getattr(prediction.predicted_signature, field) == getattr(actual, field)
            for field in SIGNATURE_FIELDS
        }
        null_matches = {
            field: majority[field] == getattr(actual, field) for field in SIGNATURE_FIELDS
        }
        scored.append({
            "query_id": prediction.query_id,
            "abstained": prediction.abstained,
            "edge_allowed": (
                prediction.current_node_id, prediction.predicted_next_node_id
            ) in edge_set,
            "field_matches": matches,
            "exact": all(matches.values()),
            "null_field_matches": null_matches,
            "null_exact": all(null_matches.values()),
            "actual_next_event_id": trace.events[prediction.prefix_end_offset + 1].event_id,
        })
    admitted = [row for row in scored if not row["abstained"]]
    denominator = len(admitted)
    field_total = denominator * len(SIGNATURE_FIELDS)
    return {
        "split": split,
        "queries_expected": sum(
            max(0, len(trace.events) - 1) for trace in evaluation.values()
        ),
        "queries_sampled": len(predictions),
        "predictions_returned": len(predictions),
        "admitted": denominator,
        "coverage": len(predictions) / max(1, sum(
            max(0, len(trace.events) - 1) for trace in evaluation.values()
        )),
        "exact_accuracy": sum(row["exact"] for row in admitted) / denominator if denominator else 0.0,
        "null_exact_accuracy": (
            sum(row["null_exact"] for row in admitted) / denominator if denominator else 0.0
        ),
        "field_accuracy": (
            sum(sum(row["field_matches"].values()) for row in admitted) / field_total
            if field_total else 0.0
        ),
        "null_field_accuracy": (
            sum(sum(row["null_field_matches"].values()) for row in admitted) / field_total
            if field_total else 0.0
        ),
        "graph_edge_validity": (
            sum(row["edge_allowed"] for row in admitted) / denominator if denominator else 0.0
        ),
        "rows": scored,
        "claim_boundary": (
            "This is blind anonymous next-event prediction, not source value support. "
            "SOURCE_SUPPORTED additionally requires matched multi-horizon authentic advantage."
        ),
    }


def structural_affordance_report(
    traces: Sequence[SourceDecisionTrace],
) -> dict[str, Any]:
    by_split = {}
    for split in SPLITS:
        selected = [trace for trace in traces if trace.split == split]
        events = [event for trace in selected for event in trace.events]
        next_signatures: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
        for trace in selected:
            for left, right in zip(trace.events, trace.events[1:]):
                left_key = tuple(getattr(left.signature, field) for field in SIGNATURE_FIELDS)
                right_key = tuple(getattr(right.signature, field) for field in SIGNATURE_FIELDS)
                next_signatures[left_key].add(right_key)
        by_split[split] = {
            "traces": len(selected),
            "events": len(events),
            "skill_switch_events": sum(
                event.signature.skill_relation == "CHANGED" for event in events
            ),
            "action_switch_events": sum(
                event.signature.action_relation == "CHANGED" for event in events
            ),
            "positive_reward_events": sum(
                event.signature.reward_sign == "POSITIVE" for event in events
            ),
            "terminal_events": sum(event.signature.terminal for event in events),
            "anonymous_branch_opportunities": sum(
                len(targets) > 1 for targets in next_signatures.values()
            ),
        }
    return {
        "schema_version": "SOURCE_DECISION_CYCLE_AFFORDANCE_V1",
        "games": sorted({trace.game for trace in traces}),
        "traces_valid": all(trace.validate() for trace in traces),
        "split_stats": by_split,
        "claim_boundary": (
            "Branch opportunities are mechanical extraction diagnostics only. "
            "They are not Agent-proposed motifs and do not establish transfer."
        ),
    }


__all__ = [
    "AnonymousEventSignature", "SourceDecisionEvent", "SourceDecisionTrace",
    "DecisionGraphNode", "DecisionGraphEdge", "SourceDecisionGraph",
    "DecisionGraphAudit", "BlindPrediction", "SourceDecisionCycleAgent",
    "build_decision_traces", "compact_trace_payload", "audit_graph",
    "graph_from_agent_response", "score_blind_predictions",
    "shuffled_graph_edges", "structural_affordance_report",
]
