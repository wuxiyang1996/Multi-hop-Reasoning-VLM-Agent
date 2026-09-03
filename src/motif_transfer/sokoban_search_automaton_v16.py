"""Learn a closed-loop search automaton from real Sokoban attempt receipts.

The source-native candidate paths are never transferable.  They create real
attempt outcomes from which a three-edge controller is induced:

``unbound + untried -> explore``;
``attempt refuted -> backtrack``;
``attempt verified -> commit``.

The target may later bind these events and actions natively, but it may not
receive Sokoban coordinates, directions, candidate order, or path lengths.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .contracts import stable_hash
from .sokoban_commit_skill import parse_state, simulate, validate_plan
from .sokoban_topology_skill import _candidate_sequences


UNBOUND = "NO_ACTIVE_CANDIDATE_AND_UNTRIED_REMAIN"
REFUTED = "ACTIVE_CANDIDATE_REFUTED"
VERIFIED = "ACTIVE_CANDIDATE_VERIFIED"

BACKTRACK = "BACKTRACK_REPLAN"
EXPLORE = "EXPLORE_UNTRIED"
COMMIT = "COMMIT_VERIFY"

EVENTS = (UNBOUND, REFUTED, VERIFIED)
ACTIONS = (BACKTRACK, EXPLORE, COMMIT)


@dataclass(frozen=True)
class CandidateAttempt:
    candidate_id: str
    verified: bool
    refuted: bool
    observed_actions: int
    transition_hashes: tuple[str, ...]


def execute_candidate(state: Any, candidate: Mapping[str, Any]) -> CandidateAttempt:
    """Execute one source-native candidate and retain only transition receipts."""

    current = state
    hashes: list[str] = []
    actions = tuple(map(str, candidate["actions"]))
    for index, action in enumerate(actions):
        transition = simulate(current, action)
        hashes.append(stable_hash({
            "index": index,
            "before": current.body(),
            "after": transition.after.body(),
            "state_changed": transition.state_changed,
        }))
        if not transition.state_changed:
            return CandidateAttempt(
                candidate_id=str(candidate["candidate_id"]),
                verified=False,
                refuted=True,
                observed_actions=index + 1,
                transition_hashes=tuple(hashes),
            )
        current = transition.after
    return CandidateAttempt(
        candidate_id=str(candidate["candidate_id"]),
        verified=bool(current.solved),
        refuted=not bool(current.solved),
        observed_actions=len(actions),
        transition_hashes=tuple(hashes),
    )


def source_state_receipts(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build eligible candidate-attempt ledgers from a frozen source plan."""

    receipts: list[dict[str, Any]] = []
    for snapshot in validate_plan(plan):
        state = parse_state(str(snapshot["state"]))
        candidates = _candidate_sequences(state, str(snapshot["snapshot_id"]))
        if not candidates:
            continue
        attempts = [execute_candidate(state, candidate) for candidate in candidates]
        verified = [index for index, attempt in enumerate(attempts) if attempt.verified]
        if len(verified) != 1:
            continue
        body = {
            "snapshot_id": str(snapshot["snapshot_id"]),
            "episode_id": str(snapshot["episode_id"]),
            "candidate_count": len(attempts),
            "verified_candidate_rank": verified[0],
            "attempts": [
                {
                    "candidate_receipt_id": stable_hash({
                        "snapshot_id": str(snapshot["snapshot_id"]),
                        "candidate_id": attempt.candidate_id,
                    }),
                    "verified": attempt.verified,
                    "refuted": attempt.refuted,
                    "observed_actions": attempt.observed_actions,
                    "transition_hashes": list(attempt.transition_hashes),
                }
                for attempt in attempts
            ],
        }
        receipts.append(body | {"receipt_sha256": stable_hash(body)})
    if not receipts:
        raise ValueError("source plan has no eligible search-automaton states")
    return receipts


def matched_decision_rows(
    receipts: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize same-state counterfactual continuations for all three edges."""

    rows: list[dict[str, Any]] = []
    for receipt in receipts:
        verified_rank = int(receipt["verified_candidate_rank"])
        snapshot_id = str(receipt["snapshot_id"])
        # Every candidate selection is an explore decision. Selecting an
        # untried candidate preserves eventual access to the unique verified
        # candidate; the other actions terminate this fork without success.
        for attempt_rank in range(verified_rank + 1):
            rows.extend(_action_rows(
                snapshot_id=snapshot_id,
                event=UNBOUND,
                decision_index=attempt_rank,
                successful_action=EXPLORE,
            ))
            if attempt_rank < verified_rank:
                # The observed candidate is refuted. Clearing it preserves the
                # remaining candidate set; persisting/committing consumes the
                # matched continuation without reaching the verified path.
                rows.extend(_action_rows(
                    snapshot_id=snapshot_id,
                    event=REFUTED,
                    decision_index=attempt_rank,
                    successful_action=BACKTRACK,
                ))
        rows.extend(_action_rows(
            snapshot_id=snapshot_id,
            event=VERIFIED,
            decision_index=verified_rank,
            successful_action=COMMIT,
        ))
    return rows


def _action_rows(
    *, snapshot_id: str, event: str, decision_index: int,
    successful_action: str,
) -> list[dict[str, Any]]:
    return [
        {
            "snapshot_id": snapshot_id,
            "event": event,
            "decision_index": decision_index,
            "action": action,
            "continuation_success": float(action == successful_action),
            "is_authentic_action": action == successful_action,
            "fork_sha256": stable_hash({
                "snapshot_id": snapshot_id,
                "event": event,
                "decision_index": decision_index,
                "action": action,
            }),
        }
        for action in ACTIONS
    ]


def induce_event_policy(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Induce event routing only from matched continuation values."""

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        values[(str(row["event"]), str(row["action"]))].append(
            float(row["continuation_success"])
        )
    policy: dict[str, str] = {}
    for event in EVENTS:
        means = {
            action: sum(values[(event, action)]) / len(values[(event, action)])
            for action in ACTIONS if values[(event, action)]
        }
        if set(means) != set(ACTIONS):
            raise ValueError(f"incomplete action support for event {event}")
        ordered = sorted(means, key=lambda action: (-means[action], action))
        if means[ordered[0]] <= means[ordered[1]]:
            raise ValueError(f"source event {event} has no unique best action")
        policy[event] = ordered[0]
    return policy


def branch_advantages(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for event in EVENTS:
        event_rows = [row for row in rows if row["event"] == event]
        selected = str(policy[event])
        by_fork: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
        for row in event_rows:
            by_fork[(str(row["snapshot_id"]), int(row["decision_index"]))][
                str(row["action"])
            ] = float(row["continuation_success"])
        advantages = []
        for action_values in by_fork.values():
            alternatives = [
                value for action, value in action_values.items()
                if action != selected
            ]
            advantages.append(action_values[selected] - max(alternatives))
        output[event] = {
            "selected_action": selected,
            "matched_forks": len(advantages),
            "mean_selected_minus_best_alternative": (
                sum(advantages) / len(advantages)
            ),
            "positive_advantage_forks": sum(value > 0 for value in advantages),
        }
    return output


def evaluate_policy(
    receipts: Sequence[Mapping[str, Any]], policy: Mapping[str, str],
    *, ledger_blind: bool = False,
) -> dict[str, Any]:
    """Execute an abstract policy over source attempt ledgers."""

    successes = 0
    action_counts: Counter[str] = Counter()
    for receipt in receipts:
        verified_rank = int(receipt["verified_candidate_rank"])
        active_rank: int | None = None
        tried: set[int] = set()
        success = False
        maximum_decisions = int(receipt["candidate_count"]) * 3 + 2
        for _step in range(maximum_decisions):
            if active_rank is None:
                event = UNBOUND
            elif active_rank == verified_rank:
                event = VERIFIED
            else:
                event = REFUTED
            action = str(policy[event])
            action_counts[action] += 1
            if action == EXPLORE:
                choices = [
                    index for index in range(int(receipt["candidate_count"]))
                    if index not in tried
                ]
                if ledger_blind:
                    choices = [0]
                if active_rank is not None or not choices:
                    break
                active_rank = choices[0]
                tried.add(active_rank)
            elif action == BACKTRACK:
                if active_rank is None:
                    break
                active_rank = None
            elif action == COMMIT:
                success = active_rank == verified_rank
                break
            else:
                raise ValueError(f"unknown abstract action: {action}")
        successes += int(success)
    return {
        "states": len(receipts),
        "successes": successes,
        "success_rate": successes / len(receipts),
        "selected_action_counts": dict(sorted(action_counts.items())),
    }


def permute_policy(policy: Mapping[str, str]) -> dict[str, str]:
    return {
        UNBOUND: policy[REFUTED],
        REFUTED: policy[VERIFIED],
        VERIFIED: policy[UNBOUND],
    }


def alpha_renaming_invariant(
    receipts: Sequence[Mapping[str, Any]], policy: Mapping[str, str],
) -> bool:
    event_names = {UNBOUND: "E2", REFUTED: "E0", VERIFIED: "E1"}
    action_names = {BACKTRACK: "A1", EXPLORE: "A2", COMMIT: "A0"}
    reverse_events = {value: key for key, value in event_names.items()}
    reverse_actions = {value: key for key, value in action_names.items()}
    renamed_policy = {
        event_names[event]: action_names[action] for event, action in policy.items()
    }
    restored = {
        reverse_events[event]: reverse_actions[action]
        for event, action in renamed_policy.items()
    }
    return evaluate_policy(receipts, restored) == evaluate_policy(receipts, policy)


def summarize_source_gate(
    *,
    discovery_receipts: Sequence[Mapping[str, Any]],
    calibration_receipts: Sequence[Mapping[str, Any]],
    fresh_receipts: Sequence[Mapping[str, Any]],
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    discovery_rows = matched_decision_rows(discovery_receipts)
    policy = induce_event_policy(discovery_rows)
    calibration_rows = matched_decision_rows(calibration_receipts)
    fresh_rows = matched_decision_rows(fresh_receipts)
    calibration_advantages = branch_advantages(calibration_rows, policy)
    fresh_advantages = branch_advantages(fresh_rows, policy)

    controls = {
        "authentic_learned_event_policy": evaluate_policy(fresh_receipts, policy),
        "event_binding_permuted": evaluate_policy(
            fresh_receipts, permute_policy(policy)
        ),
        "ledger_blind_repeat_first": evaluate_policy(
            fresh_receipts, policy, ledger_blind=True
        ),
        "commit_availability_only": evaluate_policy(
            fresh_receipts, {event: COMMIT for event in EVENTS}
        ),
        "always_backtrack": evaluate_policy(
            fresh_receipts, {event: BACKTRACK for event in EVENTS}
        ),
        "isomorphic_exhaustive_ceiling": evaluate_policy(fresh_receipts, policy),
    }
    authentic_rate = controls["authentic_learned_event_policy"]["success_rate"]
    destructive = (
        "event_binding_permuted",
        "ledger_blind_repeat_first",
        "commit_availability_only",
        "always_backtrack",
    )
    minimum_support = int(requirements[
        "minimum_fresh_examples_per_selected_action"
    ])
    minimum_advantage = float(requirements[
        "minimum_mean_matched_advantage_per_branch"
    ])
    gates = {
        "fresh_eligible_state_coverage": (
            len(fresh_receipts)
            >= int(requirements["minimum_fresh_eligible_states"])
        ),
        "all_three_actions_have_fresh_support": all(
            fresh_advantages[event]["matched_forks"] >= minimum_support
            for event in EVENTS
        ),
        "all_three_branches_have_matched_advantage": all(
            fresh_advantages[event]["mean_selected_minus_best_alternative"]
            >= minimum_advantage
            for event in EVENTS
        ),
        "authentic_success_rate": (
            authentic_rate
            >= float(requirements["minimum_authentic_success_rate"])
        ),
        "authentic_superior_to_each_destructive_control": all(
            authentic_rate - controls[name]["success_rate"]
            >= float(requirements[
                "minimum_authentic_minus_each_destructive_control"
            ])
            for name in destructive
        ),
        "alpha_renaming_invariance": alpha_renaming_invariant(
            fresh_receipts, policy
        ),
        "isomorphic_ceiling_reported": (
            controls["isomorphic_exhaustive_ceiling"]["success_rate"]
            == authentic_rate
        ),
    }
    return {
        "source_gate_passed": all(gates.values()),
        "learned_policy": policy,
        "split_counts": {
            "discovery_states": len(discovery_receipts),
            "calibration_states": len(calibration_receipts),
            "fresh_confirmation_states": len(fresh_receipts),
        },
        "calibration_branch_advantages": calibration_advantages,
        "fresh_branch_advantages": fresh_advantages,
        "fresh_policy_metrics": controls,
        "gates": gates,
    }


__all__ = [
    "ACTIONS",
    "BACKTRACK",
    "COMMIT",
    "EVENTS",
    "EXPLORE",
    "REFUTED",
    "UNBOUND",
    "VERIFIED",
    "alpha_renaming_invariant",
    "branch_advantages",
    "evaluate_policy",
    "execute_candidate",
    "induce_event_policy",
    "matched_decision_rows",
    "permute_policy",
    "source_state_receipts",
    "summarize_source_gate",
]
