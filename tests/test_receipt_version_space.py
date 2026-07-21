from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from harness.multistep_binding import (
    MultiStepBindingCandidate,
    MultiStepTargetAdmission,
    ProgramOrigin,
    TargetNodeBinding,
    TargetStepBinding,
)
from harness.online_transfer_runtime import NativeTransitionEvidence
from harness.receipt_version_space import (
    ReceiptVersionSpaceRuntime,
    VersionSpaceStatus,
    VersionTransitionVerdict,
    build_receipt_version_space,
    receipt_version_space_from_dict,
)
from harness.skill_admission import TargetActionEvidence, TargetDemoReceipt


def _hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _action(index, command, operator, types, before, after, *, success=False):
    return TargetActionEvidence(
        transition_index=index, action=command, operator=operator,
        arguments={}, argument_types=types,
        before_admissible_actions=before, after_admissible_actions=after,
        admissible_actions_sha256=_hash(before),
        next_admissible_actions_sha256=_hash(after),
        state_sha256=_hash(f"state-{index}"),
        next_state_sha256=_hash(f"state-{index + 1}"),
        reward=float(success), terminated=success, truncated=False,
        official_success_after=success,
    )


def _demo(example: int) -> TargetDemoReceipt:
    before = ["go to fridge 1", "look"]
    middle = ["open fridge 1", "look"]
    after = ["look"]
    return TargetDemoReceipt(
        demo_id=f"demo-{example}", target_domain="alfworld",
        task_family="pick_and_place", split="train",
        episode_id=f"episode-{example}", source_file_sha256=_hash(f"file-{example}"),
        executor_kind="real", evaluator="alfworld_official",
        official_success=True, official_score=1.0,
        actions=(
            _action(0, "go to fridge 1", "GOTO", {"location": "location"}, before, middle),
            _action(1, "open fridge 1", "OPEN", {"receptacle": "receptacle"}, middle, after, success=True),
        ),
    )


def _candidate(identity: str) -> MultiStepBindingCandidate:
    source_hash = _hash(f"source-{identity}")
    return MultiStepBindingCandidate(
        candidate_id=f"candidate-{identity}",
        origin=ProgramOrigin.SOURCE_HYPOTHESIS,
        proposal_source="untrusted-agent",
        proposal_receipt_sha256=_hash(f"proposal-{identity}"),
        source_hypothesis_hash=source_hash,
        nodes=(
            TargetNodeBinding("N0", (
                TargetStepBinding(0, "GOTO", {"location": "location"}),
            ), {"observed_transitions": [{"action": "left"}]}),
            TargetNodeBinding("N1", (
                TargetStepBinding(1, "OPEN", {"receptacle": "receptacle"}),
            ), {"observed_transitions": [{"action": "up"}]}),
        ),
    )


def _admit(example: int, identity: str):
    candidate = _candidate(identity)
    return MultiStepTargetAdmission().admit(
        candidates=(candidate,), demo=_demo(example),
        known_proposal_receipt_hashes=(candidate.proposal_receipt_sha256,),
        known_source_hypothesis_nodes={
            candidate.source_hypothesis_hash: ("N0", "N1"),
        },
        known_source_node_conditioning={
            candidate.source_hypothesis_hash: {
                node.node_id: dict(node.source_conditioning)
                for node in candidate.nodes
            },
        },
        source_treatment="correct",
        source_control_receipt_sha256=_hash("source-control"),
    )


def _transition(*, known_signature: bool = True) -> NativeTransitionEvidence:
    return NativeTransitionEvidence.build(
        step=0, command="go to fridge 1",
        before_observation_sha256=_hash("before"),
        after_observation_sha256=_hash("after"),
        before_actions_sha256=_hash("before-actions"),
        after_actions_sha256=(
            _hash("after-actions") if known_signature else _hash("before-actions")
        ),
        reward=0.0, official_success=False, command_was_admissible=True,
        executed_action_admissible_after=not known_signature,
        terminated=False, truncated=False,
    )


def test_one_example_initializes_provisional_version_space() -> None:
    space = build_receipt_version_space(
        adaptation_set_id="adapt-2", artifacts=(_admit(0, "a"),),
        expected_example_count=2,
    )
    assert space.status == VersionSpaceStatus.PROVISIONAL
    assert len(space.examples) == 1
    assert len(space.versions) == len(space.viable_schema_hashes) == 1
    loaded = receipt_version_space_from_dict(space.to_dict())
    assert loaded.artifact_hash == space.artifact_hash


def test_exact_schema_intersection_is_ready_without_clustering() -> None:
    space = build_receipt_version_space(
        adaptation_set_id="adapt-2",
        artifacts=(_admit(0, "a"), _admit(1, "a")),
        expected_example_count=2,
    )
    assert space.status == VersionSpaceStatus.READY
    assert len(space.viable_schema_hashes) == 1
    version = space.versions[0]
    assert version.supporting_example_indices == (0, 1)
    assert {row.demo_hash for row in version.step_evidence} == {
        _demo(0).content_hash(), _demo(1).content_hash(),
    }

    runtime = ReceiptVersionSpaceRuntime(space)
    receipt = runtime.observe_transition(
        transition=_transition(), observed_operator="GOTO",
        observed_argument_types={"location": "location"},
    )
    assert receipt.verdict == VersionTransitionVerdict.SUPPORTED
    assert runtime.cursor == 1
    receipt.validate_hash()


def test_unseen_live_signature_requests_evidence_without_refuting_version() -> None:
    space = build_receipt_version_space(
        adaptation_set_id="adapt-2",
        artifacts=(_admit(0, "a"), _admit(1, "a")),
        expected_example_count=2,
    )
    runtime = ReceiptVersionSpaceRuntime(space)
    receipt = runtime.observe_transition(
        transition=_transition(known_signature=False), observed_operator="GOTO",
        observed_argument_types={"location": "location"},
    )
    assert receipt.verdict == VersionTransitionVerdict.NEED_MORE_EVIDENCE
    assert runtime.cursor == 0
    assert runtime.paused_for_evidence
    assert len(space.viable_schema_hashes) == 1


def test_no_common_schema_is_bounded_not_applicable_claim() -> None:
    incomplete = build_receipt_version_space(
        adaptation_set_id="adapt-3",
        artifacts=(_admit(0, "a"), _admit(1, "b")),
        expected_example_count=3,
    )
    assert incomplete.status == VersionSpaceStatus.NEED_MORE_EVIDENCE

    complete = build_receipt_version_space(
        adaptation_set_id="adapt-2",
        artifacts=(_admit(0, "a"), _admit(1, "b")),
        expected_example_count=2,
    )
    assert complete.status == (
        VersionSpaceStatus.NOT_APPLICABLE_WITHIN_REGISTERED_ADAPTATION_SET
    )
    assert not complete.viable_schema_hashes


def test_version_space_rejects_duplicate_examples_and_tampering() -> None:
    artifact = _admit(0, "a")
    with pytest.raises(ValueError, match="distinct demo hashes"):
        build_receipt_version_space(
            adaptation_set_id="duplicate", artifacts=(artifact, artifact),
            expected_example_count=2,
        )
    space = build_receipt_version_space(
        adaptation_set_id="adapt-2", artifacts=(artifact,),
        expected_example_count=2,
    )
    tampered = space.to_dict()
    tampered["viable_schema_hashes"] = []
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        receipt_version_space_from_dict(tampered)

    internally_inconsistent = space.to_dict()
    internally_inconsistent["versions"][0]["supporting_example_indices"] = []
    internally_inconsistent["artifact_hash"] = _hash({
        key: value for key, value in internally_inconsistent.items()
        if key != "artifact_hash"
    })
    with pytest.raises(ValueError, match="supporting examples are inconsistent"):
        receipt_version_space_from_dict(internally_inconsistent)
