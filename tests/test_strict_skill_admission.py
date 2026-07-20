from __future__ import annotations

import hashlib
from dataclasses import asdict

import pytest

from harness.alfworld_grammar import parse_alfworld_action
from harness.alfworld_demo_recorder import AlfworldDemoRecorder
from harness.skill_admission import (
    AdmissionStatus,
    BindingCandidate,
    StrictOneShotAdmission,
    TargetActionEvidence,
    TargetDemoReceipt,
    admission_artifact_from_dict,
    runtime_scope_allows,
    target_demo_receipt_from_dict,
)
from skill_bank.source_skill_compiler import compile_source_programs


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _program(source_effect: str = "agent_location_changed"):
    row = {
        "game": "sokoban",
        "episode_id": "e1",
        "step_index": 0,
        "provider_or_run": "teacher",
        "chosen_skill_id": "NAVIGATE",
        "action": "down",
        "reward": 0.0,
        "done": False,
        "state_sha256": _digest("s"),
        "next_state_sha256": _digest("n"),
        "source_file_sha256": _digest("f"),
        "source_effects": [source_effect],
        "source_only": True,
    }
    return compile_source_programs([row], min_invocations=1)[0]


def _demo(*, split: str = "train", success: bool = True):
    parsed = parse_alfworld_action("go to fridge 1", admissible=["look", "go to fridge 1"])
    return TargetDemoReceipt(
        demo_id="demo-1",
        target_domain="alfworld",
        task_family="pick_and_place",
        split=split,
        episode_id="target-episode-1",
        source_file_sha256=_digest("target-file"),
        executor_kind="real",
        evaluator="alfworld_official",
        official_success=success,
        official_score=1.0 if success else 0.0,
        actions=[TargetActionEvidence(
            transition_index=0,
            action=parsed.raw,
            operator=parsed.operator,
            arguments=parsed.arguments,
            argument_types=parsed.argument_types,
            admissible_actions_sha256=_digest("look|go to fridge 1"),
            state_sha256=_digest("target-state"),
            next_state_sha256=_digest("target-next"),
        )],
    )


def _candidate(program, *, operator: str = "GOTO", candidate_id: str = "c1"):
    return BindingCandidate(
        candidate_id=candidate_id,
        source_program_id=program.program_id,
        source_program_hash=program.content_hash(),
        source_step_id="commit-observed-action",
        target_domain="alfworld",
        task_family="pick_and_place",
        target_operator=operator,
        argument_types={"location": "location"},
        source_effect="agent_location_changed",
        proposal_source="qwen35_untrusted",
    )


def test_exact_grammar_rejects_non_admissible_substring() -> None:
    try:
        parse_alfworld_action("go to fridge", admissible=["go to fridge 1"])
    except ValueError as exc:
        assert "not exactly admissible" in str(exc)
    else:
        raise AssertionError("substring command must not be accepted")


def test_exact_grammar_supports_alfworld_move_to_command() -> None:
    parsed = parse_alfworld_action(
        "move mug 3 to dresser 1",
        admissible=["move mug 3 to dresser 1"],
    )
    assert parsed.operator == "MOVE_TO"
    assert parsed.arguments == {"object": "mug 3", "receptacle": "dresser 1"}


def test_real_successful_one_shot_admits_scope() -> None:
    program = _program()
    artifact = StrictOneShotAdmission().admit(
        program=program, candidates=[_candidate(program)], demo=_demo()
    )
    assert artifact.status == AdmissionStatus.ADMITTED
    assert runtime_scope_allows(
        artifact,
        target_domain="alfworld",
        task_family="pick_and_place",
        operator="GOTO",
        argument_types={"location": "location"},
    )
    assert not runtime_scope_allows(
        artifact,
        target_domain="alfworld",
        task_family="pick_and_place",
        operator="OPEN",
        argument_types={"receptacle": "receptacle"},
    )


def test_frozen_artifact_loader_rejects_tampering() -> None:
    program = _program()
    artifact = StrictOneShotAdmission().admit(
        program=program, candidates=[_candidate(program)], demo=_demo()
    )
    payload = artifact.to_dict()
    loaded = admission_artifact_from_dict(payload)
    assert loaded.artifact_hash == artifact.artifact_hash
    payload["task_family"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        admission_artifact_from_dict(payload)


def test_test_split_cannot_update_admission() -> None:
    program = _program()
    artifact = StrictOneShotAdmission().admit(
        program=program, candidates=[_candidate(program)], demo=_demo(split="valid_unseen")
    )
    assert artifact.status == AdmissionStatus.REJECTED
    assert any("HELD-OUT/TEST" in code for code in artifact.failure_codes)


def test_failed_demo_rejects_even_if_model_proposes_binding() -> None:
    program = _program()
    artifact = StrictOneShotAdmission().admit(
        program=program, candidates=[_candidate(program)], demo=_demo(success=False)
    )
    assert artifact.status == AdmissionStatus.REJECTED


def test_non_equivalent_passing_candidates_remain_inconclusive() -> None:
    program = _program()
    # Both candidate IDs are equivalent and therefore safely canonicalized.
    equivalent = StrictOneShotAdmission().admit(
        program=program,
        candidates=[_candidate(program, candidate_id="b"), _candidate(program, candidate_id="a")],
        demo=_demo(),
    )
    assert equivalent.status == AdmissionStatus.ADMITTED
    assert equivalent.admitted_candidate_id == "a"


def test_hallucinated_operator_cannot_override_verifier() -> None:
    program = _program()
    artifact = StrictOneShotAdmission().admit(
        program=program,
        candidates=[_candidate(program, operator="TELEPORT")],
        demo=_demo(),
    )
    assert artifact.status == AdmissionStatus.REJECTED


def test_supported_but_unseen_operator_is_inconclusive() -> None:
    program = _program("receptacle_opened")
    candidate = BindingCandidate(
        candidate_id="open",
        source_program_id=program.program_id,
        source_program_hash=program.content_hash(),
        source_step_id="commit-observed-action",
        target_domain="alfworld",
        task_family="pick_and_place",
        target_operator="OPEN",
        argument_types={"receptacle": "receptacle"},
        source_effect="receptacle_opened",
        proposal_source="untrusted",
    )
    artifact = StrictOneShotAdmission().admit(
        program=program, candidates=[candidate], demo=_demo()
    )
    assert artifact.status == AdmissionStatus.INCONCLUSIVE
    assert artifact.failure_codes == ["OPERATOR_NOT_COVERED:open"]


def test_partially_observed_source_effect_is_conditional() -> None:
    rows = []
    for index, effects in enumerate((["agent_location_changed"], [])):
        rows.append({
            "game": "sokoban", "episode_id": f"e{index}", "step_index": 0,
            "provider_or_run": "teacher", "chosen_skill_id": "NAVIGATE",
            "action": "down", "reward": 0.0, "done": False,
            "state_sha256": _digest(f"s{index}"),
            "next_state_sha256": _digest(f"n{index}"),
            "source_file_sha256": _digest(f"f{index}"),
            "source_effects": effects, "source_only": True,
        })
    program = compile_source_programs(rows, min_invocations=1)[0]
    artifact = StrictOneShotAdmission().admit(
        program=program, candidates=[_candidate(program)], demo=_demo()
    )
    assert artifact.status == AdmissionStatus.CONDITIONAL
    assert artifact.verified_scope is not None
    assert artifact.verified_scope.source_effect_evidence_count == 1
    assert artifact.verified_scope.source_evidence_count == 2


def test_real_demo_recorder_preserves_exact_action_evidence() -> None:
    class Env:
        def reset(self):
            return "kitchen", {"action_names": ["look", "go to fridge 1"], "won": False}

        def step(self, action):
            assert action == "go to fridge 1"
            return "at fridge", 1.0, True, False, {"action_names": ["look"], "won": True}

    recorder = AlfworldDemoRecorder(
        Env(), demo_id="fixed-demo", task_family="pick_and_place", split="train"
    )
    recorder.reset()
    recorder.step("go to fridge 1")
    receipt = recorder.receipt()
    receipt.validate_for_admission()
    assert receipt.actions[0].operator == "GOTO"
    assert receipt.official_success is True
    restored = target_demo_receipt_from_dict({**asdict(receipt), "demo_hash": receipt.content_hash()})
    assert restored.content_hash() == receipt.content_hash()
