from __future__ import annotations

from motif_transfer.contracts import (
    Advisory, AdvisoryVerdict, ContinuationDecision, DecisionCycleReceipt,
    DecisionCycleRecord, DecisionProposal, DecisionProposalSet,
    EvidenceVerdict, Observation, PostTransitionAssessment, TransitionReceipt,
)
from motif_transfer.target_native_motifs import (
    TargetEpisodeView,
    audit_target_native_motif,
    target_motif_from_agent_response,
)


def _record(step):
    before = Observation({"step": step}, ("A", "B"))
    after = Observation({"step": step + 1}, ("A", "B"))
    proposal = DecisionProposal(f"p{step}", "A")
    proposals = DecisionProposalSet(f"ps{step}", (proposal,), proposal.proposal_id)
    assessment = PostTransitionAssessment(
        EvidenceVerdict.SUPPORTED, ContinuationDecision.CONTINUE,
    )
    transition = TransitionReceipt.create(before, proposal, after, float(step % 2))
    receipt = DecisionCycleReceipt.create(proposals, transition, assessment)
    return DecisionCycleRecord(
        before, proposals, Advisory(AdvisoryVerdict.ADMIT, "test"),
        after, float(step % 2), transition, assessment, receipt,
    )


def _episodes():
    return tuple(
        TargetEpisodeView(
            f"e{i}", "target", "adaptation",
            tuple(_record(step) for step in range(4)),
        )
        for i in range(3)
    )


def test_target_motif_rejects_single_field_shortcut_even_when_recurrent():
    episodes = _episodes()
    spans = []
    for episode in episodes:
        spans.extend([
            {
                "span_id": f"{episode.episode_id}-a",
                "episode_id": episode.episode_id,
                "start_offset": 0, "end_offset": 0,
            },
            {
                "span_id": f"{episode.episode_id}-b",
                "episode_id": episode.episode_id,
                "start_offset": 1, "end_offset": 2,
            },
            {
                "span_id": f"{episode.episode_id}-c",
                "episode_id": episode.episode_id,
                "start_offset": 3, "end_offset": 3,
            },
        ])
    motif = target_motif_from_agent_response("target", episodes, {
        "spans": spans,
        "nodes": [
            {"node_id": "short", "span_ids": [
                f"e{i}-{suffix}" for i in range(3) for suffix in ("a", "c")
            ]},
            {"node_id": "long", "span_ids": [f"e{i}-b" for i in range(3)]},
        ],
        "edges": [
            {"source": "short", "target": "long"},
            {"source": "long", "target": "short"},
        ],
    })
    audit = audit_target_native_motif(motif, episodes)
    assert not audit.accepted
    assert audit.recurrent_nodes and audit.recurrent_edges
    assert "length" in audit.single_field_shortcuts


def test_target_motif_rejects_test_episode_reference():
    episodes = list(_episodes())
    episodes[0] = TargetEpisodeView(
        episodes[0].episode_id, "target", "test", episodes[0].records,
    )
    motif = target_motif_from_agent_response("target", episodes, {
        "spans": [{
            "span_id": "bad", "episode_id": "e0",
            "start_offset": 0, "end_offset": 1,
        }],
        "nodes": [{"node_id": "n", "span_ids": ["bad"]}],
        "edges": [],
    })
    audit = audit_target_native_motif(motif, episodes)
    # The parser cannot bind cycle receipts from a test episode.
    assert not audit.accepted
