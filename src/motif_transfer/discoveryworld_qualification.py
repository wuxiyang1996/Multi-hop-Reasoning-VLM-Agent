"""Outcome-blind fork selection for DiscoveryWorld qualification episodes."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .discoveryworld_env import stable_hash
from .discoveryworld_policy import target_native_facts

if TYPE_CHECKING:
    from .discoveryworld_env import DiscoveryWorldObservation
    from .discoveryworld_sokoban_transfer import (
        DiscoveryWorldGroundedCandidate,
        DiscoveryWorldTargetBinding,
    )


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


def assess_effect_guard_applicability(
    observation: "DiscoveryWorldObservation",
    target_binding: "DiscoveryWorldTargetBinding",
    candidates: Sequence["DiscoveryWorldGroundedCandidate"],
    *,
    allowed_commit_actions: Sequence[str],
    minimum_binding_confidence: float,
    prerequisite_threshold: float,
    positive_effect_threshold: float,
) -> dict[str, Any]:
    """Assess a source-effect-guard disagreement without reading outcomes.

    This is deliberately a *conditional applicability* definition, not a task
    success selector.  An eligible state contains both target-native actions,
    target-native myopic grounding prefers COMMIT, and the source-qualified
    exact effect guard prefers POSITION because no positive commit-effect
    witness is present.
    """

    from .discoveryworld_sokoban_transfer import (
        commit_available,
        positive_commit_effect_kind,
        select_candidate,
        target_bound_position,
    )

    if not 0.0 <= minimum_binding_confidence <= 1.0:
        raise ValueError("minimum_binding_confidence must be in [0, 1]")
    allowed = frozenset(str(value) for value in allowed_commit_actions)
    if not allowed:
        raise ValueError("allowed_commit_actions is empty")
    bound_commit_action = str(target_binding.commit_action.get("action") or "")
    target_name_tokens = re.findall(r"[a-z0-9]+", target_binding.target_name.lower())
    target_name_head = target_name_tokens[0] if target_name_tokens else ""
    task_text = " ".join(
        str(row.get("description") or row.get("taskDescription") or "")
        for row in target_native_facts(observation)["task_progress"]
        if isinstance(row, Mapping)
    ).lower()
    task_tokens = frozenset(re.findall(r"[a-z0-9]+", task_text))
    target_head_task_supported = bool(
        target_name_head and target_name_head in task_tokens
    )
    rows = tuple(candidates)
    available_commits = tuple(
        row for row in rows if commit_available(row, target_binding)
    )
    bound_positions = tuple(
        row for row in rows if target_bound_position(row, target_binding)
    )
    positive_effect_commits = tuple(
        row for row in available_commits
        if positive_commit_effect_kind(row, observation, target_binding) is not None
    )

    myopic = authentic = None
    myopic_receipt = authentic_receipt = None
    if rows:
        myopic, myopic_receipt = select_candidate(
            "target_native_myopic",
            rows,
            observation,
            target_binding=target_binding,
            prerequisite_threshold=prerequisite_threshold,
            positive_effect_threshold=positive_effect_threshold,
        )
        authentic, authentic_receipt = select_candidate(
            "authentic_sokoban_effect_plus_target",
            rows,
            observation,
            target_binding=target_binding,
            prerequisite_threshold=prerequisite_threshold,
            positive_effect_threshold=positive_effect_threshold,
        )

    reason = "FIRST_SOURCE_EFFECT_GUARD_DISAGREEMENT"
    eligible = True
    if target_binding.confidence < minimum_binding_confidence:
        eligible, reason = False, "LOW_TARGET_BINDING_CONFIDENCE"
    elif bound_commit_action not in allowed:
        eligible, reason = False, "UNSUPPORTED_BOUND_COMMIT_ACTION"
    elif not target_head_task_supported:
        eligible, reason = False, "TARGET_BINDING_HEAD_NOT_TASK_SUPPORTED"
    elif len(rows) < 2:
        eligible, reason = False, "DEGENERATE_CANDIDATE_SET"
    elif not available_commits:
        eligible, reason = False, "NO_BOUND_COMMIT_CANDIDATE"
    elif not bound_positions:
        eligible, reason = False, "NO_TARGET_BOUND_POSITION_CANDIDATE"
    elif positive_effect_commits:
        eligible, reason = False, "POSITIVE_COMMIT_EFFECT_ALREADY_WITNESSED"
    elif myopic is None or myopic.target_role != "COMMIT":
        eligible, reason = False, "MYOPIC_DOES_NOT_PREFER_COMMIT"
    elif authentic is None or authentic.target_role != "POSITION":
        eligible, reason = False, "AUTHENTIC_DOES_NOT_PREFER_POSITION"
    elif dict(myopic.action) == dict(authentic.action):
        eligible, reason = False, "NO_POLICY_ACTION_DISAGREEMENT"

    body = {
        "schema_version": "discoveryworld-effect-guard-applicability-v1",
        "eligible": eligible,
        "reason": reason,
        "binding_sha256": target_binding.binding_sha256,
        "binding_confidence": target_binding.confidence,
        "allowed_commit_actions": sorted(allowed),
        "bound_commit_action": bound_commit_action,
        "target_name_head": target_name_head,
        "target_head_task_supported": target_head_task_supported,
        "candidate_bundle_sha256": stable_hash(
            [row.candidate_sha256 for row in rows]
        ),
        "candidate_count": len(rows),
        "available_commit_count": len(available_commits),
        "target_bound_position_count": len(bound_positions),
        "positive_effect_commit_count": len(positive_effect_commits),
        "myopic_selected_role": myopic.target_role if myopic else None,
        "myopic_selected_action": dict(myopic.action) if myopic else None,
        "myopic_selection_receipt_sha256": (
            myopic_receipt.receipt_sha256 if myopic_receipt else None
        ),
        "authentic_selected_role": authentic.target_role if authentic else None,
        "authentic_selected_action": dict(authentic.action) if authentic else None,
        "authentic_selection_receipt_sha256": (
            authentic_receipt.receipt_sha256 if authentic_receipt else None
        ),
        "outcome_fields_read_for_eligibility": False,
    }
    body["applicability_receipt_sha256"] = stable_hash(body)
    return body


__all__ = ["assess_effect_guard_applicability", "select_first_commit_fork"]
