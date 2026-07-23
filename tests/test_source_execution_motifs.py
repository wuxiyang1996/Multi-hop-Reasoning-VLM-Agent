from __future__ import annotations

from motif_transfer.contracts import Observation, SourcePolicyStepRecord, SourceTransitionReceipt
from motif_transfer.instrumented_import import ImportedSourceEpisode
from motif_transfer.source_execution_motifs import (
    audit_execution_graph,
    build_execution_traces,
    graph_from_response,
)


def _record(episode, step, skill, action, reward=0.0, terminal=False):
    before = Observation({"step": step}, ("L", "R"))
    after = Observation({"step": step + 1}, ("L", "R"), terminal=terminal)
    receipt = SourceTransitionReceipt.create(
        before, episode_id=episode, step=step,
        selected_skill_hash=skill, action_response_hash=f"r{step}",
        action=action, action_origin="AGENT", policy_adapter="action_taking",
        after=after, reward=reward,
    )
    return SourcePolicyStepRecord(
        episode, step, before, skill, skill, "reason", f"r{step}", action,
        "AGENT", "action_taking", after, reward, receipt,
    )


def _episodes():
    episodes = []
    for index in range(6):
        eid = f"e{index}"
        records = (
            _record(eid, 0, "a", "L"),
            _record(eid, 1, "a", "R"),
            _record(eid, 2, "b", "R", 1),
            _record(eid, 3, "a", "L", terminal=True),
        )
        episodes.append(ImportedSourceEpisode(eid, "game", records, (), 1, True, ()))
    return tuple(episodes)


def test_old_skill_spans_become_atomic_execution_events():
    traces = build_execution_traces(_episodes())
    assert all(trace.validate() for trace in traces)
    assert all(len(trace.executions) == 3 for trace in traces)
    first = traces[0].executions[0]
    assert first.start_step == 0 and first.end_step == 1
    assert len(first.transition_receipt_ids) == 2


def test_execution_graph_edges_cross_old_subepisode_boundaries():
    traces = build_execution_traces(_episodes())
    discovery = [trace for trace in traces if trace.split == "discovery"]
    graph = graph_from_response("game", traces, {
        "nodes": [
            {"node_id": "start", "execution_ids": [
                trace.executions[0].execution_id for trace in discovery
            ]},
            {"node_id": "check", "execution_ids": [
                trace.executions[1].execution_id for trace in discovery
            ]},
            {"node_id": "return", "execution_ids": [
                trace.executions[2].execution_id for trace in discovery
            ]},
        ],
        "edges": [
            {"source": "start", "target": "check"},
            {"source": "check", "target": "return"},
        ],
    })
    audit = audit_execution_graph(graph, traces)
    # A linear three-node graph is deliberately not enough for a backbone claim.
    assert not audit.accepted
    assert audit.observed_edges == 4
    assert "TRIVIAL_EXECUTION_GRAPH" in audit.failure_codes
    assert audit.recurrent_nodes
    assert audit.recurrent_edges
