from __future__ import annotations

from motif_transfer.contracts import (
    Observation, SourcePolicyStepRecord, SourceTransitionReceipt,
)
from motif_transfer.instrumented_import import ImportedSourceEpisode
from motif_transfer.source_decision_cycles import (
    AnonymousEventSignature,
    BlindPrediction,
    audit_graph,
    build_decision_traces,
    graph_from_agent_response,
    score_blind_predictions,
    shuffled_graph_edges,
)


def _record(episode: str, step: int, skill: str, action: str, reward=0.0, terminal=False):
    before = Observation({"step": step}, ("L", "R"))
    after = Observation({"step": step + 1}, ("L", "R"), terminal=terminal)
    receipt = SourceTransitionReceipt.create(
        before, episode_id=episode, step=step,
        selected_skill_hash=f"hash-{skill}", action_response_hash=f"response-{step}",
        action=action, action_origin="AGENT", policy_adapter="action_taking",
        after=after, reward=reward,
    )
    return SourcePolicyStepRecord(
        episode, step, before, skill, f"hash-{skill}", "untrusted",
        f"response-{step}", action, "AGENT", "action_taking", after, reward, receipt,
    )


def _episodes():
    result = []
    for index in range(6):
        episode_id = f"episode-{index}"
        records = (
            _record(episode_id, 0, "skill-a", "L"),
            _record(episode_id, 1, "skill-a", "L"),
            _record(episode_id, 2, "skill-b", "R", reward=1.0),
            _record(episode_id, 3, "skill-a", "L", terminal=True),
        )
        result.append(ImportedSourceEpisode(
            episode_id, "game", records, (), 1.0, True, (),
        ))
    return tuple(result)


def test_decision_cycles_cross_skill_boundaries_and_alpha_rename_actions():
    traces = build_decision_traces(_episodes())
    assert len(traces) == 6
    assert {trace.split for trace in traces} == {
        "discovery", "qualification", "held_out",
    }
    assert all(trace.validate() for trace in traces)
    assert traces[0].events[2].signature.skill_relation == "CHANGED"
    assert traces[0].events[2].action_token == "A1"


def test_agent_graph_is_receipt_bound_and_blind_scoring_beats_bad_prediction():
    traces = build_decision_traces(_episodes())
    discovery = [trace for trace in traces if trace.split == "discovery"]
    first = [trace.events[0].event_id for trace in discovery]
    middle = [
        event.event_id for trace in discovery for event in trace.events[1:3]
    ]
    final = [trace.events[3].event_id for trace in discovery]
    graph = graph_from_agent_response("game", traces, {
        "nodes": [
            {"node_id": "n0", "event_ids": first},
            {"node_id": "n1", "event_ids": middle},
            {"node_id": "n2", "event_ids": final},
        ],
        "edges": [
            {"source": "n0", "target": "n1"},
            {"source": "n1", "target": "n1"},
            {"source": "n1", "target": "n2"},
        ],
    })
    audit = audit_graph(graph, traces)
    assert audit.accepted
    assert audit.crosses_recorded_skill_boundary
    trace = next(trace for trace in traces if trace.split == "qualification")
    actual = trace.events[1].signature
    prediction = BlindPrediction(
        "q", graph.graph_id, trace.trace_id, 0, "n0", "n1", actual,
    )
    score = score_blind_predictions(
        graph, traces, (prediction,), split="qualification",
    )
    assert score["exact_accuracy"] == 1.0
    assert score["graph_edge_validity"] == 1.0
    bad = BlindPrediction(
        "q2", graph.graph_id, trace.trace_id, 0, "n0", "n2",
        AnonymousEventSignature("CHANGED", "CHANGED", "FALLBACK", "NEGATIVE", True),
    )
    bad_score = score_blind_predictions(
        graph, traces, (bad,), split="qualification",
    )
    assert bad_score["exact_accuracy"] == 0.0
    assert bad_score["graph_edge_validity"] == 0.0


def test_shuffled_control_changes_topology_without_changing_edge_count():
    traces = build_decision_traces(_episodes())
    discovery = [trace for trace in traces if trace.split == "discovery"]
    graph = graph_from_agent_response("game", traces, {
        "nodes": [
            {"node_id": "n0", "event_ids": [trace.events[0].event_id for trace in discovery]},
            {"node_id": "n1", "event_ids": [
                event.event_id for trace in discovery for event in trace.events[1:]
            ]},
        ],
        "edges": [
            {"source": "n0", "target": "n1"},
            {"source": "n1", "target": "n1"},
        ],
    })
    authentic = tuple((edge.source, edge.target) for edge in graph.edges)
    shuffled = shuffled_graph_edges(graph)
    assert len(shuffled) == len(authentic)
    assert shuffled != authentic
