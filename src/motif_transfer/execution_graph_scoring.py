from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .source_execution_motifs import (
    ExecutionMotifGraph,
    ExecutionSignature,
    ExecutionTrace,
)


EMISSION_FIELDS = ("skill_relation", "return_sign", "terminal")
FIELD_VOCABULARY = {
    "skill_relation": ("START", "SAME", "CHANGED", "__UNSEEN__"),
    "return_sign": ("NEGATIVE", "ZERO", "POSITIVE", "__UNSEEN__"),
    "terminal": (False, True, "__UNSEEN__"),
}


def _value(signature: ExecutionSignature, field: str) -> Any:
    value = getattr(signature, field)
    return value if value in FIELD_VOCABULARY[field] else "__UNSEEN__"


@dataclass(frozen=True)
class FrozenExecutionGraphModel:
    node_ids: tuple[str, ...]
    emission_counts: Mapping[str, Mapping[str, Mapping[Any, int]]]
    emission_totals: Mapping[str, int]
    transition_counts: Mapping[str, Mapping[str, int]]
    transition_totals: Mapping[str, int]
    null_counts: Mapping[str, Mapping[Any, int]]
    null_total: int
    alpha: float


def fit_execution_graph_model(
    graph: ExecutionMotifGraph,
    traces: Sequence[ExecutionTrace],
    *,
    alpha: float = 1.0,
) -> FrozenExecutionGraphModel:
    """Fit only from discovery receipts assigned by the frozen proposal."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    discovery = [trace for trace in traces if trace.split == "discovery"]
    event_index = {
        row.execution_id: row
        for trace in discovery for row in trace.executions
    }
    assignment = {
        execution_id: node.node_id
        for node in graph.nodes for execution_id in node.execution_ids
    }
    node_ids = tuple(sorted(node.node_id for node in graph.nodes))
    emissions: dict[str, dict[str, Counter]] = {
        node_id: {field: Counter() for field in EMISSION_FIELDS}
        for node_id in node_ids
    }
    totals = Counter()
    null = {field: Counter() for field in EMISSION_FIELDS}
    null_total = 0
    for execution_id, node_id in assignment.items():
        row = event_index.get(execution_id)
        if row is None:
            continue
        totals[node_id] += 1
        null_total += 1
        for field in EMISSION_FIELDS:
            item = _value(row.signature, field)
            emissions[node_id][field][item] += 1
            null[field][item] += 1
    transitions: dict[str, Counter] = {node_id: Counter() for node_id in node_ids}
    transition_totals = Counter()
    for trace in discovery:
        for left, right in zip(trace.executions, trace.executions[1:]):
            source = assignment.get(left.execution_id)
            target = assignment.get(right.execution_id)
            if source is not None and target is not None:
                transitions[source][target] += 1
                transition_totals[source] += 1
    return FrozenExecutionGraphModel(
        node_ids=node_ids,
        emission_counts={
            node: {
                field: dict(emissions[node][field]) for field in EMISSION_FIELDS
            } for node in node_ids
        },
        emission_totals=dict(totals),
        transition_counts={
            node: dict(transitions[node]) for node in node_ids
        },
        transition_totals=dict(transition_totals),
        null_counts={field: dict(null[field]) for field in EMISSION_FIELDS},
        null_total=null_total,
        alpha=float(alpha),
    )


def _categorical_probability(
    counts: Mapping[Any, int],
    total: int,
    value: Any,
    vocabulary_size: int,
    alpha: float,
) -> float:
    return (int(counts.get(value, 0)) + alpha) / (
        total + alpha * vocabulary_size
    )


def _node_emission_probability(
    model: FrozenExecutionGraphModel,
    node_id: str,
    signature: ExecutionSignature,
) -> float:
    probability = 1.0
    total = int(model.emission_totals.get(node_id, 0))
    for field in EMISSION_FIELDS:
        probability *= _categorical_probability(
            model.emission_counts[node_id][field],
            total,
            _value(signature, field),
            len(FIELD_VOCABULARY[field]),
            model.alpha,
        )
    return probability


def _current_node(
    model: FrozenExecutionGraphModel,
    signature: ExecutionSignature,
) -> str | None:
    if not model.node_ids:
        return None
    scored = [
        (_node_emission_probability(model, node_id, signature), node_id)
        for node_id in model.node_ids
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _transition_probability(
    model: FrozenExecutionGraphModel,
    source: str,
    target: str,
    *,
    shuffled_target: Mapping[str, str] | None,
) -> float:
    lookup_target = (
        next(
            (
                original for original, replacement in shuffled_target.items()
                if replacement == target
            ),
            target,
        )
        if shuffled_target else target
    )
    return _categorical_probability(
        model.transition_counts.get(source, {}),
        int(model.transition_totals.get(source, 0)),
        lookup_target,
        len(model.node_ids),
        model.alpha,
    )


def _next_signature_probability(
    model: FrozenExecutionGraphModel,
    current_node: str,
    signature: ExecutionSignature,
    *,
    shuffled_target: Mapping[str, str] | None = None,
) -> float:
    return sum(
        _transition_probability(
            model, current_node, target, shuffled_target=shuffled_target,
        ) * _node_emission_probability(model, target, signature)
        for target in model.node_ids
    )


def _null_probability(
    model: FrozenExecutionGraphModel,
    signature: ExecutionSignature,
) -> float:
    probability = 1.0
    for field in EMISSION_FIELDS:
        probability *= _categorical_probability(
            model.null_counts[field],
            model.null_total,
            _value(signature, field),
            len(FIELD_VOCABULARY[field]),
            model.alpha,
        )
    return probability


def score_execution_graph_blind(
    model: FrozenExecutionGraphModel,
    traces: Sequence[ExecutionTrace],
    *,
    split: str,
) -> dict[str, Any]:
    if split not in {"qualification", "held_out"}:
        raise ValueError("split must be qualification or held_out")
    if len(model.node_ids) < 2:
        raise ValueError("a nontrivial frozen graph is required")
    shift = 1
    shuffled = {
        node_id: model.node_ids[(index + shift) % len(model.node_ids)]
        for index, node_id in enumerate(model.node_ids)
    }
    rows = []
    for trace in traces:
        if trace.split != split:
            continue
        for offset, (current, following) in enumerate(
            zip(trace.executions, trace.executions[1:])
        ):
            current_node = _current_node(model, current.signature)
            if current_node is None:
                continue
            authentic = max(
                _next_signature_probability(
                    model, current_node, following.signature,
                ),
                1e-300,
            )
            shuffled_probability = max(
                _next_signature_probability(
                    model, current_node, following.signature,
                    shuffled_target=shuffled,
                ),
                1e-300,
            )
            null = max(_null_probability(model, following.signature), 1e-300)
            rows.append({
                "trace_id": trace.trace_id,
                "prefix_end_execution_offset": offset,
                "hidden_next_execution_id": following.execution_id,
                "current_node": current_node,
                "authentic_log_score": math.log(authentic),
                "shuffled_log_score": math.log(shuffled_probability),
                "null_log_score": math.log(null),
            })
    count = len(rows)
    def mean(field: str) -> float | None:
        return (
            sum(float(row[field]) for row in rows) / count
            if count else None
        )
    authentic_mean = mean("authentic_log_score")
    shuffled_mean = mean("shuffled_log_score")
    null_mean = mean("null_log_score")
    return {
        "split": split,
        "blind_boundaries": count,
        "authentic_mean_log_score": authentic_mean,
        "shuffled_mean_log_score": shuffled_mean,
        "null_mean_log_score": null_mean,
        "authentic_gain_over_shuffled": (
            authentic_mean - shuffled_mean if count else None
        ),
        "authentic_gain_over_null": (
            authentic_mean - null_mean if count else None
        ),
        "passes_predictive_ordering": bool(
            count
            and authentic_mean > shuffled_mean
            and authentic_mean > null_mean
        ),
        "rows": rows,
        "claim_boundary": (
            "Model fitting uses discovery assignments only. Evaluation reveals "
            "each next execution solely to the mechanical scorer."
        ),
    }


__all__ = [
    "EMISSION_FIELDS", "FrozenExecutionGraphModel",
    "fit_execution_graph_model", "score_execution_graph_blind",
]
