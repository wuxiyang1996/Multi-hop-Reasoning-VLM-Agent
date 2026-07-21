"""Failure-trace synthesis for ALFWorld text-household episodes."""

from __future__ import annotations

from typing import Any, Dict, List

from legacy.crafter.failure_memory import SEMANTIC_BUCKET_EXTRA_KEY
from data_structure.extensions.failure_trace import FailureTrace


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def from_sample(
    sample: Dict[str, Any],
    *,
    domain: str = "alfworld",
    sample_id: str = "",
    max_failures: int = 4,
) -> List[FailureTrace]:
    """Convert completion and invalid-command signals into bounded traces."""
    sid = sample_id or str(sample.get("sample_id") or sample.get("task_id") or "")
    info = sample.get("info") if isinstance(sample.get("info"), dict) else {}
    reward = float(
        _first(sample.get("total_reward", sample.get("reward", info.get("won", 0.0))))
        or 0.0
    )
    success = bool(_first(
        sample.get("success", sample.get("won", info.get("won", False)))
    ))
    observation = str(sample.get("observation") or sample.get("last_observation") or "")
    last_action = str(sample.get("last_action") or "")
    admissible = list(
        sample.get("pre_admissible_actions")
        or sample.get("admissible_actions_before_action")
        or []
    )
    abort_reason = str(sample.get("abort_reason") or "")
    out: List[FailureTrace] = []

    if not success and reward < 1.0:
        out.append(FailureTrace(
            skill_id=str(sample.get("skill_id") or ""),
            skill_episode_id=f"{sid}#task_incomplete",
            domain=domain,
            failed_step_index=int(sample.get("failed_step_index", 0) or 0),
            failure_class="INVARIANT_VIOLATION",
            abort_reason=f"alfworld task incomplete; reward={reward:g}",
            extra={
                SEMANTIC_BUCKET_EXTRA_KEY: "task_incomplete/alfworld/text",
                "synthesis_signal": "TASK_INCOMPLETE",
                "observation": observation[:400],
                "last_action": last_action,
            },
        ))

    invalid_command = bool(sample.get("invalid_action")) or (
        "unresolved_alfworld_command" in abort_reason
    )
    if invalid_command:
        out.append(FailureTrace(
            skill_id=str(sample.get("skill_id") or ""),
            skill_episode_id=f"{sid}#invalid_command",
            domain=domain,
            failed_step_index=int(sample.get("failed_step_index", 0) or 0),
            failure_class="PRECONDITION_VIOLATION",
            abort_reason=f"command not admissible: {last_action}",
            extra={
                SEMANTIC_BUCKET_EXTRA_KEY: "invalid_command/alfworld/text",
                "synthesis_signal": "INVALID_COMMAND",
                "last_action": last_action,
                "admissible_actions": admissible[:40],
            },
        ))

    return out[:max(0, int(max_failures))]


__all__ = ["from_sample"]
