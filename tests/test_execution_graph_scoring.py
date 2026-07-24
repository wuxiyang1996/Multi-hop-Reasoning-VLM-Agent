from __future__ import annotations

from motif_transfer.execution_graph_scoring import (
    fit_execution_graph_model,
    score_execution_graph_blind,
)
from motif_transfer.source_execution_motifs import (
    ExecutionMotifEdge,
    ExecutionMotifGraph,
    ExecutionMotifNode,
    ExecutionSignature,
    ExecutionTrace,
    SkillExecutionEvidence,
)


def _execution(eid, episode, offset, skill_relation, return_sign):
    signature = ExecutionSignature(
        skill_relation, 1, return_sign, (return_sign,), ("START",), False,
    )
    body = {
        "segment_receipt_id": f"segment-{eid}",
        "episode_id": episode,
        "start_step": offset,
        "end_step": offset,
        "transition_receipt_ids": (f"transition-{eid}",),
        "skill_token": "S",
        "action_tokens": ("A",),
        "signature": signature,
        "untrusted_reasoning": (),
    }
    from motif_transfer.contracts import stable_hash
    return SkillExecutionEvidence(stable_hash({
        **body, "signature": {
            "skill_relation": signature.skill_relation,
            "length": signature.length,
            "return_sign": signature.return_sign,
            "reward_sign_sequence": signature.reward_sign_sequence,
            "action_repeat_sequence": signature.action_repeat_sequence,
            "terminal": signature.terminal,
        }
    }), **body)


def _trace(episode, split, pattern):
    rows = tuple(
        _execution(f"{episode}-{i}", episode, i, relation, reward)
        for i, (relation, reward) in enumerate(pattern)
    )
    from dataclasses import asdict
    from motif_transfer.contracts import stable_hash
    body = {
        "game": "g", "episode_id": episode, "split": split,
        "executions": tuple(asdict(row) for row in rows),
    }
    return ExecutionTrace(stable_hash(body), "g", episode, split, rows)


def test_blind_graph_score_uses_discovery_only_and_beats_shuffled_on_pattern():
    pattern = (
        ("START", "ZERO"),
        ("CHANGED", "POSITIVE"),
        ("CHANGED", "ZERO"),
        ("CHANGED", "POSITIVE"),
    )
    traces = (
        _trace("d1", "discovery", pattern),
        _trace("d2", "discovery", pattern),
        _trace("q1", "qualification", pattern),
        _trace("h1", "held_out", pattern),
    )
    discovery = traces[:2]
    zero_ids = tuple(
        row.execution_id for trace in discovery for row in trace.executions
        if row.signature.return_sign == "ZERO"
    )
    positive_ids = tuple(
        row.execution_id for trace in discovery for row in trace.executions
        if row.signature.return_sign == "POSITIVE"
    )
    graph = ExecutionMotifGraph(
        "graph", "g",
        (ExecutionMotifNode("zero", zero_ids), ExecutionMotifNode("positive", positive_ids)),
        (
            ExecutionMotifEdge("zero", "positive", ()),
            ExecutionMotifEdge("positive", "zero", ()),
        ),
        tuple(trace.trace_id for trace in discovery),
    )
    model = fit_execution_graph_model(graph, traces)
    report = score_execution_graph_blind(model, traces, split="qualification")
    assert report["blind_boundaries"] == 3
    assert report["authentic_gain_over_shuffled"] > 0
