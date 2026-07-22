from dataclasses import dataclass, replace
import json

import pytest

from motif_transfer.contracts import Observation, SourcePolicyStepRecord, SourceTransitionReceipt
from motif_transfer.skill_internal import (
    SourcePromptCondition,
    SkillInternalGraphAgent,
    audit_frozen_alignment,
    audit_internal_graph,
    build_execution_sets,
    build_skill_off_control_set,
    load_skill_hypotheses,
    structural_fingerprint,
)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    records: tuple[SourcePolicyStepRecord, ...]


def _record(episode_id, step, skill_id="skill-a"):
    before = Observation({"n": step}, ("a", "b"))
    after = Observation({"n": step + 1}, ("a", "b"), terminal=step == 2)
    action = "b" if step == 1 else "a"
    receipt = SourceTransitionReceipt.create(
        before,
        episode_id=episode_id,
        step=step,
        selected_skill_hash="hash-a",
        action_response_hash=f"response-{episode_id}-{step}",
        action=action,
        action_origin="AGENT",
        policy_adapter="action_taking",
        after=after,
        reward=float(step == 2),
    )
    return SourcePolicyStepRecord(
        episode_id, step, before, skill_id, "hash-a", f"reason-{step}",
        f"response-{episode_id}-{step}", action, "AGENT", "action_taking",
        after, float(step == 2), receipt,
    )


def _episodes():
    return tuple(
        Episode(f"episode-{index}", tuple(_record(f"episode-{index}", step) for step in range(3)))
        for index in range(6)
    )


def test_build_execution_sets_reuses_bank_as_untrusted_sidecar(tmp_path):
    bank = tmp_path / "skill_bank.jsonl"
    bank.write_text(json.dumps({"skill": {
        "skill_id": "skill-a", "protocol": {"steps": ["possibly hallucinated"]}
    }}) + "\n")
    hypotheses = load_skill_hypotheses(bank)
    execution_set = build_execution_sets("game", _episodes(), hypotheses)[0]
    assert execution_set.validate()
    assert execution_set.skill_hypothesis_hash == hypotheses["skill-a"].content_hash
    assert {row.split for row in execution_set.executions} == {
        "discovery", "qualification", "held_out"
    }
    assert len(execution_set.transition_receipt_ids) == 18


def test_skill_off_control_is_explicitly_not_a_source_skill():
    control = build_skill_off_control_set("game", _episodes(), "skill-a")
    assert control.validate()
    assert control.authority == "MATCHED_SKILL_OFF_EPISODE_CONTROL_ONLY"
    assert control.skill_hypothesis_hash is None
    assert control.skill_id.startswith("SKILL_OFF_CONTROL::")


class CyclicBackend:
    def complete(self, role, system, payload):
        assert role == "skill_internal_graph"
        assert "arbitrary contiguous spans" in system
        assert len(payload["executions"]) == 2  # discovery only
        return json.dumps({"motifs": [{
            "nodes": [
                {"node_id": "n0", "role": "untrusted-a", "occurrences": [
                    {"execution_index": 0, "start_offset": 0, "end_offset": 0},
                    {"execution_index": 0, "start_offset": 2, "end_offset": 2},
                ]},
                {"node_id": "n1", "role": "untrusted-b", "occurrences": [
                    {"execution_index": 0, "start_offset": 1, "end_offset": 1},
                ]},
            ],
            "edges": [
                {"source": "n0", "target": "n1", "condition": "untrusted"},
                {"source": "n1", "target": "n0", "condition": "untrusted"},
            ],
        }]})


class RequestBackend:
    def complete(self, role, system, payload):
        return json.dumps({
            "motifs": [],
            "intervention_requests": [{
                "execution_index": 0,
                "source_offset": 0,
                "alternative_action_ordinal": 1,
                "question": "untrusted discriminating question",
            }],
        })


def test_agent_can_split_inside_one_skill_and_audit_nontrivial_graph():
    episodes = _episodes()
    execution_set = build_execution_sets("game", episodes)[0]
    records = {
        row.transition.receipt_id: row
        for episode in episodes for row in episode.records
    }
    graph = SkillInternalGraphAgent(CyclicBackend()).propose(
        execution_set, records
    )[0]
    audit = audit_internal_graph(graph, execution_set, records)
    assert audit.accepted
    assert audit.nontrivial
    assert not audit.backbone_eligible
    assert audit.control_flags == ("ACTION_IDENTITY_EXPLAINS_NODE_PARTITION",)
    assert len(graph.nodes[0].occurrences) == 2
    assert all(
        occurrence.execution_id in graph.discovery_execution_ids
        for node in graph.nodes for occurrence in node.occurrences
    )


def test_edge_without_observed_adjacency_fails_closed():
    class UnsupportedEdgeBackend(CyclicBackend):
        def complete(self, role, system, payload):
            return json.dumps({"motifs": [{
                "nodes": [
                    {"node_id": "n0", "occurrences": [
                        {"execution_index": 0, "start_offset": 0, "end_offset": 0}
                    ]},
                    {"node_id": "n1", "occurrences": [
                        {"execution_index": 0, "start_offset": 2, "end_offset": 2}
                    ]},
                ],
                "edges": [{"source": "n0", "target": "n1"}],
            }]})

    episodes = _episodes()
    execution_set = build_execution_sets("game", episodes)[0]
    records = {
        row.transition.receipt_id: row
        for episode in episodes for row in episode.records
    }
    with pytest.raises(ValueError, match="no observed adjacent occurrence"):
        SkillInternalGraphAgent(UnsupportedEdgeBackend()).propose(execution_set, records)


def test_agent_can_request_but_not_fabricate_an_intervention():
    episodes = _episodes()
    execution_set = build_execution_sets("game", episodes)[0]
    records = {
        row.transition.receipt_id: row
        for episode in episodes for row in episode.records
    }
    agent = SkillInternalGraphAgent(RequestBackend())
    assert agent.propose(execution_set, records) == ()
    request = agent.intervention_requests[0]
    assert request.status == "REQUESTED_NOT_OBSERVED"
    assert request.alternative_action_ordinal == 1
    assert request.execution_id in {
        row.execution_id for row in execution_set.executions if row.split == "discovery"
    }


def test_frozen_controls_mask_information_without_changing_receipt_lineage():
    class CaptureBackend:
        def __init__(self):
            self.payload = None

        def complete(self, role, system, payload):
            self.payload = payload
            return json.dumps({"motifs": []})

    episodes = _episodes()
    execution_set = build_execution_sets("game", episodes)[0]
    records = {
        row.transition.receipt_id: row
        for episode in episodes for row in episode.records
    }
    backend = CaptureBackend()
    SkillInternalGraphAgent(backend).propose(
        execution_set, records, condition=SourcePromptCondition.ACTION_ALPHA_RENAMED
    )
    payload = backend.payload
    assert payload["schema_version"] == "SKILL_INTERNAL_V1_FROZEN"
    assert payload["untrusted_skill_hypothesis"] is None
    step = payload["executions"][0]["steps"][0]
    assert step["executed_action"] == "ACTION_0"
    assert step["native_actions"] == ("ACTION_0", "ACTION_1")
    assert "untrusted_policy_reasoning" not in step
    expected_receipts = [
        row.transition_receipt_ids
        for row in execution_set.executions if row.split == "discovery"
    ]
    assert [
        tuple(step["transition_receipt_id"] for step in execution["steps"])
        for execution in payload["executions"]
    ] == expected_receipts


def test_frozen_graph_alignment_cannot_edit_graph_and_recurrence_is_mechanical():
    episodes = _episodes()
    execution_set = build_execution_sets("game", episodes)[0]
    records = {
        row.transition.receipt_id: row
        for episode in episodes for row in episode.records
    }
    graph = SkillInternalGraphAgent(CyclicBackend()).propose(execution_set, records)[0]

    class AlignmentBackend:
        def complete(self, role, system, payload):
            assert role == "skill_internal_alignment"
            assert payload["schema_version"] == "SKILL_INTERNAL_ALIGNMENT_V1_FROZEN"
            return json.dumps({
                "abstain": False,
                "nodes": [
                    {"node_id": "n0", "occurrences": [
                        {"execution_index": 0, "start_offset": 0, "end_offset": 0},
                        {"execution_index": 0, "start_offset": 2, "end_offset": 2},
                    ]},
                    {"node_id": "n1", "occurrences": [
                        {"execution_index": 0, "start_offset": 1, "end_offset": 1}
                    ]},
                ],
            })

    alignment = SkillInternalGraphAgent(AlignmentBackend()).align_frozen_graph(
        graph, execution_set, records, split="qualification"
    )
    audit = audit_frozen_alignment(alignment, graph, execution_set)
    assert audit.accepted
    assert audit.node_coverage == 1.0
    assert audit.edge_recurrence == 1.0


def test_alignment_unknown_node_fails_closed():
    episodes = _episodes()
    execution_set = build_execution_sets("game", episodes)[0]
    records = {
        row.transition.receipt_id: row
        for episode in episodes for row in episode.records
    }
    graph = SkillInternalGraphAgent(CyclicBackend()).propose(execution_set, records)[0]

    class BadAlignmentBackend:
        def complete(self, role, system, payload):
            return json.dumps({
                "abstain": False,
                "nodes": [{"node_id": "invented", "occurrences": []}],
            })

    with pytest.raises(ValueError, match="unknown frozen node"):
        SkillInternalGraphAgent(BadAlignmentBackend()).align_frozen_graph(
            graph, execution_set, records, split="qualification"
        )


def test_structural_fingerprint_is_invariant_to_node_names_and_order():
    episodes = _episodes()
    execution_set = build_execution_sets("game", episodes)[0]
    records = {
        row.transition.receipt_id: row
        for episode in episodes for row in episode.records
    }
    graph = SkillInternalGraphAgent(CyclicBackend()).propose(execution_set, records)[0]
    renamed_nodes = (
        replace(graph.nodes[1], node_id="second"),
        replace(graph.nodes[0], node_id="first"),
    )
    renamed_edges = tuple(
        replace(
            edge,
            source={"n0": "first", "n1": "second"}[edge.source],
            target={"n0": "first", "n1": "second"}[edge.target],
        )
        for edge in reversed(graph.edges)
    )
    renamed = replace(graph, nodes=renamed_nodes, edges=renamed_edges)
    assert structural_fingerprint(graph) == structural_fingerprint(renamed)
