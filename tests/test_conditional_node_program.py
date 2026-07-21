from __future__ import annotations

import hashlib
import json

import pytest

from harness.conditional_node_program import (
    ConditionalAdmissionStatus,
    ConditionalNodeRuntime,
    ConditionalProgramProposal,
    ConditionalRuntimeVerdict,
    ExampleSegmentation,
    ProposedSegment,
    SegmentKind,
    admit_conditional_programs,
    conditional_artifact_from_dict,
)
from harness.online_transfer_runtime import NativeTransitionEvidence
from harness.skill_admission import TargetActionEvidence, TargetDemoReceipt


def _hash(value) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _action(index, command, operator, types, before, after, *, success=False):
    return TargetActionEvidence(
        transition_index=index, action=command, operator=operator,
        arguments={}, argument_types=types,
        before_admissible_actions=before, after_admissible_actions=after,
        admissible_actions_sha256=_hash(before),
        next_admissible_actions_sha256=_hash(after),
        state_sha256=_hash(f"state-{index}"),
        next_state_sha256=_hash(f"state-{index + 1}"), reward=float(success),
        terminated=success, truncated=False, official_success_after=success,
    )


def _demo(name, with_gap):
    specs = [
        ("look", "LOOK", {}),
        ("go to shelf 1", "GOTO", {"location": "location"}),
    ]
    if with_gap:
        specs.append(("go to table 1", "GOTO", {"location": "location"}))
    specs += [
        ("take apple 1 from table 1", "TAKE", {
            "object": "object", "receptacle": "receptacle",
        }),
        ("go to box 1", "GOTO", {"location": "location"}),
        ("put apple 1 in/on box 1", "MOVE_TO", {
            "object": "object", "receptacle": "receptacle",
        }),
    ]
    action_sets = [[spec[0], "look"] for spec in specs] + [["look"]]
    actions = tuple(_action(
        i, *spec, action_sets[i], action_sets[i + 1],
        success=i == len(specs) - 1,
    ) for i, spec in enumerate(specs))
    return TargetDemoReceipt(
        demo_id=name, target_domain="alfworld", task_family="pick_and_place",
        split="train", episode_id=name, source_file_sha256=_hash(name),
        executor_kind="real", evaluator="alfworld_official",
        official_success=True, official_score=1.0, actions=actions,
    )


def _graph():
    return {
        "source_hypothesis_hash": _hash("source"),
        "nodes": [
            {"node_id": "N0", "observed_transitions": [{"action": "left"}]},
            {"node_id": "N1", "observed_transitions": [{"action": "right"}]},
        ],
        "edges": [{
            "source_node_id": "N0", "target_node_id": "N1", "kind": "GUARD",
            "intervention_receipt_sha256s": [_hash("intervention")],
        }],
    }


def _proposal(demos):
    receipt = _hash("proposal-receipt")
    examples = []
    for demo in demos:
        gap = len(demo.actions) == 6
        segments = [ProposedSegment(
            "node-N0", SegmentKind.SOURCE_NODE, "N0", (0, 1),
        )]
        if gap:
            segments.append(ProposedSegment(
                "native-gap-0", SegmentKind.TARGET_NATIVE_GAP, None, (2,),
            ))
        start = 3 if gap else 2
        segments.append(ProposedSegment(
            "node-N1", SegmentKind.SOURCE_NODE, "N1", (start, start + 1, start + 2),
        ))
        examples.append(ExampleSegmentation(demo.demo_id, tuple(segments)))
    return ConditionalProgramProposal(
        proposal_id="agent-proposal", proposal_source="35b:test",
        proposal_receipt_sha256=receipt,
        source_hypothesis_hash=_hash("source"), examples=tuple(examples),
    )


def _artifact():
    demos = (_demo("demo-a", True), _demo("demo-b", False))
    proposal = _proposal(demos)
    return admit_conditional_programs(
        adaptation_set_id="two-example", proposals=(proposal,), demos=demos,
        source_graphs=(_graph(),),
        known_proposal_receipt_hashes=(proposal.proposal_receipt_sha256,),
        source_treatment="correct",
    )


def _transition(command):
    return NativeTransitionEvidence.build(
        step=0, command=command,
        before_observation_sha256=_hash("before"),
        after_observation_sha256=_hash("after"),
        before_actions_sha256=_hash([command, "look"]),
        after_actions_sha256=_hash(["look"]), reward=0.0,
        official_success=False, command_was_admissible=True,
        executed_action_admissible_after=(command == "look"),
        terminated=False, truncated=False,
    )


def test_conditional_admission_preserves_gap_and_common_node_programs():
    artifact = _artifact()
    assert len(artifact.candidates) == 1
    assert artifact.status == ConditionalAdmissionStatus.READY
    candidate = artifact.candidates[0]
    assert [[step.target_operator for step in node.steps] for node in candidate.nodes] == [
        ["LOOK", "GOTO"], ["TAKE", "GOTO", "MOVE_TO"],
    ]
    assert len(candidate.native_gaps) == 1
    assert candidate.native_gaps[0].steps[0].target_operator == "GOTO"
    assert conditional_artifact_from_dict(artifact.to_dict()).artifact_hash == artifact.artifact_hash


def test_runtime_waits_for_entry_signature_and_never_calls_gap_source():
    runtime = ConditionalNodeRuntime(_artifact())
    verdict, allowed = runtime.allowed_actions(["look", "go to shelf 1"])
    assert verdict == ConditionalRuntimeVerdict.SOURCE_READY
    assert allowed == ("look",)
    receipt = runtime.observe_source_transition(
        _transition("look"), executed_command="look",
        before_admissible=["look", "go to shelf 1"],
    )
    assert receipt.verdict == ConditionalRuntimeVerdict.SUPPORTED
    verdict, allowed = runtime.allowed_actions(["go to shelf 1", "look"])
    assert verdict == ConditionalRuntimeVerdict.SOURCE_READY
    runtime.observe_source_transition(
        _transition("go to shelf 1"), executed_command="go to shelf 1",
        before_admissible=["go to shelf 1", "look"],
    )
    verdict, allowed = runtime.allowed_actions(["go to table 1", "look"])
    assert verdict == ConditionalRuntimeVerdict.TARGET_NATIVE_GAP_REQUIRED
    assert allowed == ()
    verdict, allowed = runtime.allowed_actions([
        "take apple 1 from table 1", "look",
    ])
    assert verdict == ConditionalRuntimeVerdict.SOURCE_READY
    assert allowed == ("take apple 1 from table 1",)


def test_admission_rejects_uncovered_transition_instead_of_dropping_it():
    demos = (_demo("demo-a", True), _demo("demo-b", False))
    proposal = _proposal(demos)
    broken_examples = list(proposal.examples)
    broken_examples[0] = ExampleSegmentation(
        broken_examples[0].demo_id,
        tuple(segment for segment in broken_examples[0].segments
              if segment.kind != SegmentKind.TARGET_NATIVE_GAP),
    )
    broken = ConditionalProgramProposal(
        proposal.proposal_id, proposal.proposal_source,
        proposal.proposal_receipt_sha256, proposal.source_hypothesis_hash,
        tuple(broken_examples),
    )
    artifact = admit_conditional_programs(
        adaptation_set_id="two-example", proposals=(broken,), demos=demos,
        source_graphs=(_graph(),),
        known_proposal_receipt_hashes=(broken.proposal_receipt_sha256,),
        source_treatment="correct",
    )
    assert not artifact.candidates
    assert artifact.status == ConditionalAdmissionStatus.NEED_MORE_AGENT_PROPOSALS
    assert any("complete_ordered_coverage" in code
               for code in artifact.rejected_candidates[0]["failure_codes"])


def test_missing_source_edge_receipt_and_unseen_live_effect_fail_closed():
    demos = (_demo("demo-a", True), _demo("demo-b", False))
    proposal = _proposal(demos)
    graph = _graph()
    graph["edges"][0]["intervention_receipt_sha256s"] = []
    rejected = admit_conditional_programs(
        adaptation_set_id="two-example", proposals=(proposal,), demos=demos,
        source_graphs=(graph,),
        known_proposal_receipt_hashes=(proposal.proposal_receipt_sha256,),
        source_treatment="correct",
    )
    assert rejected.status == ConditionalAdmissionStatus.NEED_MORE_AGENT_PROPOSALS
    assert "edge:N0->N1:has_intervention_receipts" in (
        rejected.rejected_candidates[0]["failure_codes"]
    )

    runtime = ConditionalNodeRuntime(_artifact())
    unseen = NativeTransitionEvidence.build(
        step=0, command="look", before_observation_sha256=_hash("before"),
        after_observation_sha256=_hash("after"),
        before_actions_sha256=_hash("same-actions"),
        after_actions_sha256=_hash("same-actions"), reward=0.0,
        official_success=False, command_was_admissible=True,
        executed_action_admissible_after=True, terminated=False, truncated=False,
    )
    receipt = runtime.observe_source_transition(
        unseen, executed_command="look",
        before_admissible=["look", "go to shelf 1"],
    )
    assert receipt.verdict == ConditionalRuntimeVerdict.NEED_MORE_EVIDENCE
    assert runtime.step_index == 0


def test_loader_rejects_rehashed_inconsistent_status():
    payload = _artifact().to_dict()
    payload["status"] = "NEED_MORE_AGENT_PROPOSALS"
    payload["artifact_hash"] = _hash({
        key: value for key, value in payload.items() if key != "artifact_hash"
    })
    with pytest.raises(ValueError, match="status is inconsistent"):
        conditional_artifact_from_dict(payload)
