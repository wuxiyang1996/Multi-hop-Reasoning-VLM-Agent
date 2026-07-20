from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_game_transfer_assets import index_episodes
from skill_bank.legacy_source_import import is_source_game_task
from skill_bank.program_ir import (
    ActionSchema,
    CanonicalSkillProgram,
    Operator,
    ProgramStatus,
    ProgramStep,
    SourceStepKey,
    TransitionEvidenceRef,
    canonical_program_from_dict,
)
from skill_bank.source_skill_compiler import compile_source_programs
from skill_bank.source_replay_validator import SourceReplayValidator


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _row(step: int = 0) -> dict:
    return {
        "game": "tetris",
        "episode_id": f"episode-{step}",
        "step_index": step,
        "provider_or_run": "teacher",
        "chosen_skill_id": "opening:BUILD",
        "action": "hard_drop",
        "reward": 1.0,
        "done": False,
        "state_sha256": _digest(f"s{step}"),
        "next_state_sha256": _digest(f"n{step}"),
        "source_file_sha256": _digest(f"f{step}"),
        "source_only": True,
    }


def test_exact_compiler_produces_replay_anchored_program() -> None:
    programs = compile_source_programs([_row(0), _row(1)])
    assert len(programs) == 1
    program = programs[0]
    assert program.status == ProgramStatus.SOURCE_VERIFIED
    assert program.steps[0].action is not None
    assert program.steps[0].action.observed_source_actions == ["hard_drop"]
    program.validate()
    restored = canonical_program_from_dict(program.to_dict())
    assert restored.content_hash() == program.content_hash()


def test_source_verified_program_cannot_cite_missing_evidence() -> None:
    key = SourceStepKey("tetris", "e", 0, "teacher")
    evidence = TransitionEvidenceRef(
        key=key,
        source_file_sha256=_digest("file"),
        state_sha256=_digest("state"),
        next_state_sha256=_digest("next"),
        action="left",
        reward=0.0,
        done=False,
    )
    program = CanonicalSkillProgram(
        program_id="p",
        name="p",
        source_skill_ids=["s"],
        source_games=["tetris"],
        evidence=[evidence],
        steps=[ProgramStep("commit", Operator.COMMIT, ActionSchema("a"), evidence_step_ids=["missing"])],
    )
    with pytest.raises(ValueError, match="missing evidence"):
        program.validate()


def test_legacy_proposal_cannot_carry_verified_evidence() -> None:
    key = SourceStepKey("tetris", "e", 0)
    evidence = TransitionEvidenceRef(
        key, _digest("f"), _digest("s"), _digest("n"), "left", 0.0, False
    )
    program = CanonicalSkillProgram(
        program_id="legacy",
        name="legacy",
        source_skill_ids=["s"],
        source_games=["tetris"],
        evidence=[evidence],
        steps=[ProgramStep("x", Operator.COMMIT)],
        status=ProgramStatus.LEGACY_PROPOSAL,
    )
    with pytest.raises(ValueError, match="must not masquerade"):
        program.validate()


def test_target_tasks_are_excluded_from_source_lineage() -> None:
    assert is_source_game_task("tetris")
    assert is_source_game_task("gymv_columns")
    assert not is_source_game_task("alfworld")
    assert not is_source_game_task("visual_toolbench")


def test_episode_index_hashes_real_transition_shape(tmp_path: Path) -> None:
    path = tmp_path / "tetris"
    path.mkdir()
    payload = {
        "episode_id": "e1",
        "game_name": "tetris",
        "metadata": {"agent_type": "teacher"},
        "experiences": [
            {
                "idx": 0,
                "state": "before",
                "next_state": "after",
                "action": "left",
                "reward": 1,
                "skills": {"skill_id": "POSITION"},
            }
        ],
    }
    (path / "episode_000.json").write_text(json.dumps(payload))
    episodes, rows = index_episodes(tmp_path)
    assert episodes[0]["source_only"] is True
    assert rows[0]["chosen_skill_id"] == "POSITION"
    assert rows[0]["state_sha256"] == _digest("before")
    program = compile_source_programs(rows, min_invocations=1)[0]
    receipt = SourceReplayValidator(tmp_path).validate(program)
    assert receipt.passed
    assert receipt.n_verified == 1
