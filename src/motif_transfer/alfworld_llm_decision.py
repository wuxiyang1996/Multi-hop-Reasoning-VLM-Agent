"""An LLM Decision Agent for ALFWorld.

The validated ALFWorld structural runner has no language model in the loop: action
authority belongs to a frozen MLP grounder over admissible commands.  That leaves
nothing for a prompt-injected memory baseline to influence, so ExpeL, AWM, and
ReasoningBank cannot be evaluated against it at all.

This module supplies the missing piece: a Decision Agent that selects one of the
environment's own admissible commands through the shared ``CompletionBackend``.
Because it routes through that backend, ``MemoryAugmentedDecisionBackend`` wraps
it without modification, and the target-only arm is the same loop unwrapped.

Action authority stays with the environment: the model chooses an index into
``admissible_commands`` and any out-of-range or non-admissible reply is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .frozen_motif_agent import CompletionBackend

DECISION_ROLE = "decision"

DECISION_SYSTEM = (
    "You are a target-native ALFWorld Decision Agent. You are given the current text "
    "observation, the task goal, the interaction history, and the environment's "
    "admissible commands. Choose exactly one command by its integer index in that "
    "list. You may not invent, rewrite, or combine commands. Return exactly one JSON "
    'object {"action_index": <integer>, "reason": "<one short sentence>"}. '
    "Do not mention games, source skills, or latent options."
)


class InadmissibleActionError(ValueError):
    """The model returned something that is not an environment-admissible command."""


@dataclass(frozen=True)
class ALFWorldDecision:
    action: str
    action_index: int
    reason: str
    response_sha256: str
    attempts: int


def decide_alfworld_action(
    backend: CompletionBackend,
    *,
    observation_text: str,
    task_goal: str,
    admissible_commands: Sequence[str],
    history: Sequence[Mapping[str, Any]] = (),
    step: int = 0,
    maximum_steps: int = 60,
    schema_retries: int = 3,
) -> ALFWorldDecision:
    """Ask the model to pick one admissible command, refusing anything else."""
    commands = [str(row) for row in admissible_commands]
    if not commands:
        raise InadmissibleActionError("environment offered no admissible commands")
    payload: dict[str, Any] = {
        "task_goal": str(task_goal),
        "observation": str(observation_text),
        "admissible_commands": commands,
        # History is action-only; ALFWorld's score is a target outcome and must not
        # be shown to the Decision Agent or to any memory retrieval built from it.
        "history": [
            {"step": int(row.get("step", index)), "action": str(row.get("action", ""))}
            for index, row in enumerate(history)
        ],
        "step": int(step),
        "maximum_steps": int(maximum_steps),
    }
    last_error = ""
    for attempt in range(1, schema_retries + 1):
        request = payload if not last_error else payload | {"previous_error": last_error}
        raw = backend.complete(DECISION_ROLE, DECISION_SYSTEM, request)
        try:
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                raise ValueError("response is not a JSON object")
            index = int(value["action_index"])
            if not 0 <= index < len(commands):
                raise ValueError(
                    f"action_index {index} is outside 0..{len(commands) - 1}"
                )
            return ALFWorldDecision(
                action=commands[index],
                action_index=index,
                reason=str(value.get("reason") or "")[:300],
                response_sha256=stable_hash(raw),
                attempts=attempt,
            )
        except (ValueError, KeyError, TypeError) as error:
            last_error = str(error)[:300]
    raise InadmissibleActionError(
        f"model failed the admissible-command schema {schema_retries} times: {last_error}"
    )


__all__ = [
    "ALFWorldDecision", "DECISION_ROLE", "DECISION_SYSTEM",
    "InadmissibleActionError", "decide_alfworld_action",
]
