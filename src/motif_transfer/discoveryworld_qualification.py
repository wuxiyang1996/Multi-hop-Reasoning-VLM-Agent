"""Outcome-blind fork selection for DiscoveryWorld qualification episodes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .discoveryworld_env import stable_hash


def select_first_commit_fork(
    episode: Mapping[str, Any], allowed_commit_actions: Sequence[str],
) -> dict[str, Any]:
    """Select the state before the first predeclared native commit proposal.

    Selection reads task metadata and policy steps only. It deliberately does
    not consult action success, terminal outcome, evaluation, or scorecard.
    """

    allowed = frozenset(str(value) for value in allowed_commit_actions)
    if not allowed:
        raise ValueError("allowed_commit_actions is empty")
    steps = episode.get("steps")
    if not isinstance(steps, list):
        raise ValueError("episode steps must be a list")
    task = episode.get("task")
    task_id = str(episode.get("task_id") or "")
    episode_sha256 = str(episode.get("episode_sha256") or "")
    if not isinstance(task, Mapping) or not task_id or not episode_sha256:
        raise ValueError("episode is missing frozen task identity")
    selected = None
    for index, row in enumerate(steps):
        if not isinstance(row, Mapping):
            raise ValueError(f"step {index} is not an object")
        action = row.get("action")
        if not isinstance(action, Mapping):
            raise ValueError(f"step {index} action is not an object")
        if str(action.get("action") or "") in allowed:
            episode_step = row.get("episode_step")
            if not isinstance(episode_step, int) or isinstance(episode_step, bool):
                raise ValueError(f"step {index} has invalid episode_step")
            fork_after = episode_step - 1
            selected = {
                "eligible": fork_after >= 1,
                "reason": (
                    "FIRST_PREDECLARED_COMMIT_ACTION"
                    if fork_after >= 1 else "COMMIT_AT_INITIAL_STATE_UNSUPPORTED"
                ),
                "fork_after_episode_step": fork_after,
                "selected_episode_step": episode_step,
                "selected_action": dict(action),
                "selected_action_sha256": stable_hash(dict(action)),
            }
            break
    if selected is None:
        selected = {
            "eligible": False,
            "reason": "NO_PREDECLARED_COMMIT_ACTION",
            "fork_after_episode_step": None,
            "selected_episode_step": None,
            "selected_action": None,
            "selected_action_sha256": None,
        }
    body = {
        "schema_version": "discoveryworld-qualification-fork-v1",
        "task_id": task_id,
        "task": dict(task),
        "reference_episode_sha256": episode_sha256,
        "allowed_commit_actions": sorted(allowed),
        **selected,
        "outcome_fields_read_for_eligibility": False,
    }
    body["fork_receipt_sha256"] = stable_hash(body)
    return body


__all__ = ["select_first_commit_fork"]
