"""Compile and replay full-episode TracePrograms without skill segmentation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from skill_bank.trace_program_ir import (
    BackboneCoverage,
    NativeTransitionReceipt,
    ObservedOrderEdge,
    TraceProgram,
)


def _stable_hash(value: Any) -> str:
    if isinstance(value, str):
        raw = value
    else:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compile_observed_episode(path: str | Path) -> TraceProgram:
    source = Path(path)
    raw = source.read_bytes()
    payload = json.loads(raw)
    game = str(payload.get("game_name") or payload.get("env_name") or source.parent.name)
    episode_id = str(payload.get("episode_id") or source.stem)
    experiences = list(payload.get("experiences") or [])
    transitions: List[NativeTransitionReceipt] = []
    for ordinal, step in enumerate(experiences):
        index = int(step.get("idx", ordinal))
        transition_id = hashlib.sha256(
            f"{game}\0{episode_id}\0{index}".encode("utf-8")
        ).hexdigest()
        transitions.append(NativeTransitionReceipt(
            transition_id=transition_id,
            step_index=index,
            state_sha256=_stable_hash(step.get("raw_state", step.get("state", ""))),
            next_state_sha256=_stable_hash(
                step.get("raw_next_state", step.get("next_state", ""))
            ),
            available_actions_sha256=_stable_hash(list(step.get("available_actions") or [])),
            action=str(step.get("action") or "").strip(),
            reward=float(step.get("reward") or 0.0),
            done=bool(step.get("done")),
        ))
    ids = [item.transition_id for item in transitions]
    program = TraceProgram(
        program_id="trace." + hashlib.sha256(
            f"{game}\0{episode_id}".encode("utf-8")
        ).hexdigest()[:24],
        game=game,
        episode_id=episode_id,
        source_file_sha256=hashlib.sha256(raw).hexdigest(),
        transitions=transitions,
        observed_order=[
            ObservedOrderEdge(before, after) for before, after in zip(ids, ids[1:])
        ],
        coverage=BackboneCoverage(
            observation_receipted=True,
            admissibility_receipted=True,
            environment_step_receipted=True,
            native_delta_receipted=True,
            agent_proposal_receipted=False,
            continuation_decision_receipted=False,
            official_stop_receipted=False,
        ),
        official_success_verified=False,
        metadata={
            "compiler": "full_reset_to_stop_observed_trace_v1",
            "segmentation": "none_full_environment_episode",
            "legacy_outcome_not_treated_as_official_success": True,
        },
    )
    program.validate_structure()
    return program


@dataclass
class TraceReplayResult:
    program_id: str
    passed: bool
    verified_transitions: int
    failures: Sequence[str] = field(default_factory=tuple)


class TraceProgramValidator:
    def validate(self, program: TraceProgram, source_path: str | Path) -> TraceReplayResult:
        failures: List[str] = []
        try:
            program.validate_structure()
        except ValueError as exc:
            return TraceReplayResult(program.program_id, False, 0, [f"IR:{exc}"])
        source = Path(source_path)
        raw = source.read_bytes()
        payload: Mapping[str, Any] = json.loads(raw)
        if hashlib.sha256(raw).hexdigest() != program.source_file_sha256:
            failures.append("SOURCE_FILE_HASH_MISMATCH")
        if str(payload.get("episode_id") or source.stem) != program.episode_id:
            failures.append("EPISODE_ID_MISMATCH")
        experiences = list(payload.get("experiences") or [])
        if len(experiences) != len(program.transitions):
            failures.append("TRANSITION_COUNT_MISMATCH")
        verified = 0
        for ordinal, (step, receipt) in enumerate(zip(experiences, program.transitions)):
            available = [str(item) for item in step.get("available_actions") or []]
            checks = {
                "step_index": int(step.get("idx", ordinal)) == receipt.step_index == ordinal,
                "state": _stable_hash(step.get("raw_state", step.get("state", ""))) == receipt.state_sha256,
                "next_state": _stable_hash(
                    step.get("raw_next_state", step.get("next_state", ""))
                ) == receipt.next_state_sha256,
                "available": _stable_hash(available) == receipt.available_actions_sha256,
                "action": str(step.get("action") or "").strip() == receipt.action,
                "action_admissible": receipt.action in available,
                "reward": float(step.get("reward") or 0.0) == receipt.reward,
                "done": bool(step.get("done")) == receipt.done,
            }
            bad = [name for name, passed in checks.items() if not passed]
            if bad:
                failures.append(f"STEP_{ordinal}:" + ",".join(bad))
            else:
                verified += 1
        for index, (previous, current) in enumerate(
            zip(program.transitions, program.transitions[1:])
        ):
            if previous.next_state_sha256 != current.state_sha256:
                failures.append(f"BROKEN_STATE_CHAIN:{index}->{index + 1}")
        if any(item.done for item in program.transitions[:-1]):
            failures.append("DONE_BEFORE_FINAL_TRANSITION")
        return TraceReplayResult(
            program_id=program.program_id,
            passed=not failures and verified == len(program.transitions),
            verified_transitions=verified,
            failures=failures,
        )


__all__ = ["TraceProgramValidator", "TraceReplayResult", "compile_observed_episode"]
