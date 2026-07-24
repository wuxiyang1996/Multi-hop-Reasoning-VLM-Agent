from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .contracts import DecisionCycleRecord, stable_hash
from .instrumented_import import ImportedEpisode


def _reward_sign(value: float) -> str:
    return "POSITIVE" if value > 0 else "NEGATIVE" if value < 0 else "ZERO"


@dataclass(frozen=True)
class TargetEpisodeView:
    episode_id: str
    domain: str
    split: str
    records: tuple[DecisionCycleRecord, ...]

    @classmethod
    def from_imported(
        cls,
        episode: ImportedEpisode,
        *,
        split: str,
    ) -> "TargetEpisodeView":
        if split not in {"adaptation", "test"}:
            raise ValueError("target split must be adaptation or test")
        return cls(episode.episode_id, episode.game, split, episode.records)


@dataclass(frozen=True)
class TargetExecutionSpan:
    span_id: str
    episode_id: str
    start_offset: int
    end_offset: int
    cycle_receipt_ids: tuple[str, ...]
    untrusted_intent: str = ""


@dataclass(frozen=True)
class TargetNativeNode:
    node_id: str
    span_ids: tuple[str, ...]
    untrusted_role: str = ""


@dataclass(frozen=True)
class TargetNativeEdge:
    source: str
    target: str
    supporting_boundaries: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TargetNativeMotif:
    motif_id: str
    domain: str
    spans: tuple[TargetExecutionSpan, ...]
    nodes: tuple[TargetNativeNode, ...]
    edges: tuple[TargetNativeEdge, ...]
    adaptation_episode_ids: tuple[str, ...]
    untrusted_description: str = ""


@dataclass(frozen=True)
class TargetNativeMotifAudit:
    accepted: bool
    failure_codes: tuple[str, ...]
    adaptation_episodes: int
    spans: int
    nodes: int
    edges: int
    has_branch: bool
    has_cycle: bool
    recurrent_nodes: bool
    recurrent_edges: bool
    single_field_shortcuts: tuple[str, ...]


def compact_target_adaptation_payload(
    episodes: Sequence[TargetEpisodeView],
) -> list[dict[str, Any]]:
    """Expose target-native semantics to the proposing Agent, never to the audit."""

    return [{
        "episode_id": episode.episode_id,
        "records": [{
            "offset": offset,
            "cycle_receipt_id": record.receipt.cycle_id,
            "before": dict(record.before.state),
            "native_actions": record.before.native_actions,
            "proposals": [{
                "proposal_id": proposal.proposal_id,
                "action": proposal.action,
                "prediction": proposal.prediction,
                "rationale": proposal.rationale,
            } for proposal in record.proposal_set.proposals],
            "selected_proposal_id": record.proposal_set.selected_proposal_id,
            "after": dict(record.after.state),
            "reward": record.reward,
            "terminal": record.after.terminal,
            "official_success": record.after.official_success,
        } for offset, record in enumerate(episode.records)],
    } for episode in episodes if episode.split == "adaptation"]


def target_motif_from_agent_response(
    domain: str,
    episodes: Sequence[TargetEpisodeView],
    response: Mapping[str, Any],
) -> TargetNativeMotif:
    adaptation = {
        episode.episode_id: episode
        for episode in episodes if episode.split == "adaptation"
    }
    spans = []
    for raw in response.get("spans", ()):
        episode_id = str(raw["episode_id"])
        start, end = int(raw["start_offset"]), int(raw["end_offset"])
        episode = adaptation.get(episode_id)
        cycle_ids = (
            tuple(
                row.receipt.cycle_id for row in episode.records[start:end + 1]
            )
            if episode is not None and 0 <= start <= end < len(episode.records)
            else ()
        )
        body = {
            "episode_id": episode_id,
            "start_offset": start,
            "end_offset": end,
            "cycle_receipt_ids": cycle_ids,
            "untrusted_intent": str(raw.get("intent", "")),
        }
        proposed_id = str(raw.get("span_id", ""))
        span_id = proposed_id or stable_hash(body)
        spans.append(TargetExecutionSpan(span_id, **body))
    nodes = tuple(TargetNativeNode(
        str(raw["node_id"]),
        tuple(str(span_id) for span_id in raw.get("span_ids", ())),
        str(raw.get("role", "")),
    ) for raw in response.get("nodes", ()))
    node_by_span = {
        span_id: node.node_id for node in nodes for span_id in node.span_ids
    }
    edges = []
    for raw in response.get("edges", ()):
        source, target = str(raw["source"]), str(raw["target"])
        boundaries = []
        for left in spans:
            for right in spans:
                if (
                    left.episode_id == right.episode_id
                    and left.end_offset + 1 == right.start_offset
                    and node_by_span.get(left.span_id) == source
                    and node_by_span.get(right.span_id) == target
                ):
                    boundaries.append((left.span_id, right.span_id))
        edges.append(TargetNativeEdge(source, target, tuple(sorted(boundaries))))
    body = {
        "domain": domain,
        "spans": tuple(asdict(span) for span in spans),
        "nodes": tuple(asdict(node) for node in nodes),
        "edges": tuple(asdict(edge) for edge in edges),
        "adaptation_episode_ids": tuple(sorted(adaptation)),
        "untrusted_description": str(response.get("description", "")),
    }
    return TargetNativeMotif(
        stable_hash(body), **{
            **body, "spans": tuple(spans), "nodes": nodes,
            "edges": tuple(edges),
        },
    )


def _span_signature(
    span: TargetExecutionSpan,
    episodes: Mapping[str, TargetEpisodeView],
) -> dict[str, Any]:
    episode = episodes[span.episode_id]
    records = episode.records[span.start_offset:span.end_offset + 1]
    return {
        "length": len(records),
        "reward_sign_sequence": tuple(_reward_sign(row.reward) for row in records),
        "terminal_sequence": tuple(row.after.terminal for row in records),
        "proposal_count_sequence": tuple(
            len(row.proposal_set.proposals) for row in records
        ),
        "selected_ordinal_sequence": tuple(
            next(
                index for index, proposal in enumerate(row.proposal_set.proposals)
                if proposal.proposal_id == row.proposal_set.selected_proposal_id
            )
            for row in records
        ),
    }


def audit_target_native_motif(
    motif: TargetNativeMotif,
    episodes: Sequence[TargetEpisodeView],
) -> TargetNativeMotifAudit:
    by_episode = {episode.episode_id: episode for episode in episodes}
    adaptation = {
        episode.episode_id: episode
        for episode in episodes if episode.split == "adaptation"
    }
    failures = []
    span_ids = [span.span_id for span in motif.spans]
    if len(span_ids) != len(set(span_ids)):
        failures.append("DUPLICATE_SPAN_ID")
    occupied: dict[tuple[str, int], str] = {}
    valid_spans = {}
    for span in motif.spans:
        episode = by_episode.get(span.episode_id)
        if episode is None:
            failures.append("SPAN_REFERENCES_UNKNOWN_EPISODE")
            continue
        if episode.split != "adaptation":
            failures.append("SPAN_REFERENCES_TEST_EPISODE")
            continue
        if not (0 <= span.start_offset <= span.end_offset < len(episode.records)):
            failures.append("SPAN_OFFSET_OUT_OF_RANGE")
            continue
        expected = tuple(
            row.receipt.cycle_id
            for row in episode.records[span.start_offset:span.end_offset + 1]
        )
        if expected != span.cycle_receipt_ids:
            failures.append("SPAN_CYCLE_RECEIPT_MISMATCH")
            continue
        for offset in range(span.start_offset, span.end_offset + 1):
            key = (span.episode_id, offset)
            if key in occupied:
                failures.append("OVERLAPPING_TARGET_SPANS")
            occupied[key] = span.span_id
        valid_spans[span.span_id] = span
    node_ids = [node.node_id for node in motif.nodes]
    if len(node_ids) != len(set(node_ids)):
        failures.append("DUPLICATE_NODE_ID")
    node_by_span = {}
    support_by_node: dict[str, list[TargetExecutionSpan]] = defaultdict(list)
    for node in motif.nodes:
        for span_id in node.span_ids:
            if span_id not in valid_spans:
                failures.append("NODE_REFERENCES_INVALID_SPAN")
                continue
            if span_id in node_by_span:
                failures.append("SPAN_ASSIGNED_TO_MULTIPLE_NODES")
            node_by_span[span_id] = node.node_id
            support_by_node[node.node_id].append(valid_spans[span_id])
    recurrent_nodes = bool(motif.nodes) and all(
        len({span.episode_id for span in support_by_node[node.node_id]}) >= 2
        for node in motif.nodes
    )
    if not recurrent_nodes:
        failures.append("TARGET_NODE_NOT_RECURRENT_ACROSS_ADAPTATION_EPISODES")
    adjacency: dict[str, set[str]] = defaultdict(set)
    recurrent_edges = bool(motif.edges)
    for edge in motif.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            failures.append("EDGE_REFERENCES_UNKNOWN_NODE")
            recurrent_edges = False
            continue
        expected = []
        for left in valid_spans.values():
            for right in valid_spans.values():
                if (
                    left.episode_id == right.episode_id
                    and left.end_offset + 1 == right.start_offset
                    and node_by_span.get(left.span_id) == edge.source
                    and node_by_span.get(right.span_id) == edge.target
                ):
                    expected.append((left.span_id, right.span_id))
        expected_tuple = tuple(sorted(expected))
        if expected_tuple != edge.supporting_boundaries:
            failures.append("TARGET_EDGE_BOUNDARY_MISMATCH")
        edge_episodes = {
            valid_spans[left].episode_id
            for left, _ in expected_tuple if left in valid_spans
        }
        if len(edge_episodes) < 2:
            recurrent_edges = False
        adjacency[edge.source].add(edge.target)
    if not recurrent_edges:
        failures.append("TARGET_EDGE_NOT_RECURRENT_ACROSS_ADAPTATION_EPISODES")
    has_branch = any(len(targets) > 1 for targets in adjacency.values())

    def reaches(start: str, current: str, seen: set[str]) -> bool:
        for target in adjacency.get(current, set()):
            if target == start:
                return True
            if target not in seen and reaches(start, target, seen | {target}):
                return True
        return False

    has_cycle = any(reaches(node, node, {node}) for node in node_ids)
    if len(node_ids) < 2 or not motif.edges or not (has_branch or has_cycle):
        failures.append("TRIVIAL_TARGET_MOTIF")
    shortcuts = []
    fields = (
        "length", "reward_sign_sequence", "terminal_sequence",
        "proposal_count_sequence", "selected_ordinal_sequence",
    )
    assigned = [
        (valid_spans[span_id], node_id)
        for span_id, node_id in node_by_span.items()
        if span_id in valid_spans
    ]
    if len(set(node_by_span.values())) >= 2:
        for field in fields:
            labels_by_value: dict[Any, set[str]] = defaultdict(set)
            for span, node_id in assigned:
                labels_by_value[
                    _span_signature(span, adaptation)[field]
                ].add(node_id)
            if labels_by_value and all(
                len(labels) == 1 for labels in labels_by_value.values()
            ):
                shortcuts.append(field)
                failures.append(f"TARGET_SINGLE_FIELD_SHORTCUT_{field.upper()}")
    return TargetNativeMotifAudit(
        not failures, tuple(sorted(set(failures))), len(adaptation),
        len(motif.spans), len(motif.nodes), len(motif.edges), has_branch,
        has_cycle, recurrent_nodes, recurrent_edges, tuple(shortcuts),
    )


__all__ = [
    "TargetEpisodeView", "TargetExecutionSpan", "TargetNativeNode",
    "TargetNativeEdge", "TargetNativeMotif", "TargetNativeMotifAudit",
    "compact_target_adaptation_payload", "target_motif_from_agent_response",
    "audit_target_native_motif",
]
