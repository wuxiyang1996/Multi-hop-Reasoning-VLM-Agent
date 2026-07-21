from __future__ import annotations

import json
from dataclasses import replace

import pytest

from skill_bank.trace_program_ir import (
    ControlClaimKind,
    ControlHypothesis,
    TraceProgramStatus,
)
from skill_bank.trace_program_validator import (
    TraceProgramValidator,
    compile_observed_episode,
)


def _episode(*, label: str = "human-label") -> dict:
    return {
        "game_name": "opaque-game",
        "episode_id": "episode-1",
        "outcome": True,
        "experiences": [
            {
                "idx": 0,
                "raw_state": "s0",
                "raw_next_state": "s1",
                "action": "a",
                "available_actions": ["a", "b"],
                "reward": 0.0,
                "done": False,
                "skills": [label],
                "intentions": [label],
            },
            {
                "idx": 1,
                "raw_state": "s1",
                "raw_next_state": "s2",
                "action": "b",
                "available_actions": ["b"],
                "reward": 1.0,
                "done": True,
                "skills": [label],
                "intentions": [label],
            },
        ],
    }


def _write(tmp_path, payload: dict, name: str = "episode.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_full_episode_compiles_without_segmentation_or_success_claim(tmp_path) -> None:
    path = _write(tmp_path, _episode())
    program = compile_observed_episode(path)
    result = TraceProgramValidator().validate(program, path)
    assert result.passed is True
    assert len(program.transitions) == 2
    assert program.metadata["segmentation"] == "none_full_environment_episode"
    assert program.official_success_verified is False
    assert program.coverage.agent_proposal_receipted is False
    assert program.coverage.official_stop_receipted is False


def test_legacy_skill_and_intention_labels_do_not_change_trace_content(tmp_path) -> None:
    first = compile_observed_episode(_write(tmp_path, _episode(label="NAVIGATE"), "a.json"))
    second = compile_observed_episode(_write(tmp_path, _episode(label="ACQUIRE"), "b.json"))
    assert first.transitions == second.transitions
    assert first.observed_order == second.observed_order


def test_noncontiguous_or_inadmissible_transition_fails_replay(tmp_path) -> None:
    payload = _episode()
    payload["experiences"][1]["available_actions"] = ["not-b"]
    path = _write(tmp_path, payload)
    program = compile_observed_episode(path)
    result = TraceProgramValidator().validate(program, path)
    assert result.passed is False
    assert any("action_admissible" in failure for failure in result.failures)


def test_unobserved_control_claim_cannot_be_intervention_verified(tmp_path) -> None:
    program = compile_observed_episode(_write(tmp_path, _episode()))
    first_id = program.transitions[0].transition_id
    claim = ControlHypothesis(
        claim_id="branch-1",
        kind=ControlClaimKind.BRANCH,
        anchor_transition_ids=[first_id],
        proposal_source="untrusted-agent",
        proposal_receipt_sha256="0" * 64,
        status=TraceProgramStatus.INTERVENTION_VERIFIED,
        intervention_receipt_ids=[],
    )
    program.hypotheses = [claim]
    with pytest.raises(ValueError, match="lacks intervention receipts"):
        program.validate_structure()


def test_tampered_step_index_fails_replay(tmp_path) -> None:
    path = _write(tmp_path, _episode())
    program = compile_observed_episode(path)
    program.transitions = [program.transitions[0], replace(program.transitions[1], step_index=7)]
    result = TraceProgramValidator().validate(program, path)
    assert result.passed is False
    assert any("step_index" in failure for failure in result.failures)
