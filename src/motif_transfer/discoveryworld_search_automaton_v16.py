"""V16 equivalence replay over frozen DiscoveryWorld target receipts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .active_video_transfer import exact_binomial_two_sided
from .contracts import stable_hash
from .search_automaton_transfer_v16 import (
    SourceSearchAutomaton,
    bind_native_action,
    ground_target_event,
)
from .sokoban_search_automaton_v16 import BACKTRACK, COMMIT, EXPLORE


AUTHENTIC = "authentic_sokoban_effect_plus_target"
TARGET = "target_native_myopic"
CONTROLS = (
    "commit_availability_control_plus_target",
    "inverted_effect_control_plus_target",
    "position_prior_control_plus_target",
)


def _route(
    source: SourceSearchAutomaton,
    *,
    episode_id: str,
    decision_index: int,
    event_name: str,
    evidence_kind: str,
    evidence_payload: Mapping[str, Any],
    abstract_action: str,
    native_action_id: str,
    native_action: Any,
) -> dict[str, Any]:
    event = ground_target_event(
        domain="discoveryworld",
        episode_id=episode_id,
        decision_index=decision_index,
        untried_candidate_available=event_name == "UNBOUND",
        active_candidate_refuted=event_name == "REFUTED",
        terminal_commit_verified=event_name == "VERIFIED",
        evidence_kind=evidence_kind,
        evidence_payload=evidence_payload,
        grounding_confidence=1.0,
    )
    if event is None:
        raise RuntimeError("DiscoveryWorld event unexpectedly abstained")
    binding = bind_native_action(
        event,
        abstract_action=abstract_action,
        native_action_id=native_action_id,
        native_action=native_action,
        grounding_confidence=1.0,
    )
    return asdict(source.route(event, {abstract_action: binding}))


def relineage_discovery_episode(
    result: Mapping[str, Any],
    *,
    source: SourceSearchAutomaton,
) -> dict[str, Any]:
    """Show that V16 routes every recorded authentic target decision online.

    This does not counterfactually change a historical action.  It verifies
    program equivalence on policy-visible receipts and therefore remains a
    retrospective analysis, even when the underlying rollout was originally
    prospective.
    """

    task = result["task"]
    episode_id = (
        f"{str(task['scenario']).lower().replace(' ', '_')}."
        f"{str(task['difficulty']).lower()}.seed{int(task['seed'])}"
    )
    arm = result["conditions"][AUTHENTIC]
    decisions: list[dict[str, Any]] = []
    prior_realized_ids: set[str] = set()
    for step in arm.get("recovery") or ():
        selection = step["selection"]
        transition = step["transition"]
        realization = step["target_native_realization"]
        selected_role = str(selection["selected_role"])
        realized_action = dict(realization["realized_action"])
        realized_id = stable_hash(realized_action)
        if selected_role == "COMMIT":
            if not selection.get("positive_commit_effect_witnessed"):
                body = {
                    "domain": "discoveryworld",
                    "episode_id": episode_id,
                    "decision_index": len(decisions),
                    "target_event": None,
                    "source_action": None,
                    "native_action_id": realized_id,
                    "native_action": realized_action,
                    "admitted": False,
                    "reason": "ABSTAIN_UNVERIFIED_COMMIT_TO_HISTORICAL_TARGET_FALLBACK",
                    "source_artifact_sha256": source.artifact_sha256,
                    "target_evidence_sha256": stable_hash({
                        "selection_receipt_sha256": selection["receipt_sha256"],
                        "positive_commit_effect_witnessed": False,
                    }),
                }
                routed = body | {"receipt_sha256": stable_hash(body)}
            else:
                routed = _route(
                    source,
                    episode_id=episode_id,
                    decision_index=len(decisions),
                    event_name="VERIFIED",
                    evidence_kind="target_positive_commit_effect_witness",
                    evidence_payload={
                        "selection_receipt_sha256": selection["receipt_sha256"],
                        "positive_commit_effect_kind": selection[
                            "positive_commit_effect_kind"
                        ],
                    },
                    abstract_action=COMMIT,
                    native_action_id=realized_id,
                    native_action=realized_action,
                )
        elif selected_role == "POSITION":
            if selection.get("positive_commit_effect_witnessed"):
                raise ValueError("POSITION cannot carry a commit-effect witness")
            routed = _route(
                source,
                episode_id=episode_id,
                decision_index=len(decisions),
                event_name="UNBOUND",
                evidence_kind="target_grounder_untried_position_candidate",
                evidence_payload={
                    "selection_receipt_sha256": selection["receipt_sha256"],
                    "candidate_bundle_sha256": selection[
                        "candidate_bundle_sha256"
                    ],
                    "target_policy_state_sha256": transition[
                        "before_policy_state_sha256"
                    ],
                },
                abstract_action=EXPLORE,
                native_action_id=realized_id,
                native_action=realized_action,
            )
        else:
            raise ValueError(f"unsupported DiscoveryWorld target role: {selected_role}")
        allowed_target_fallback = (
            not routed["admitted"]
            and routed["reason"]
            == "ABSTAIN_UNVERIFIED_COMMIT_TO_HISTORICAL_TARGET_FALLBACK"
        )
        if (
            not routed["admitted"] and not allowed_target_fallback
        ) or routed["native_action"] != realized_action:
            raise ValueError("V16 runtime did not reproduce target-native action")
        decisions.append(routed)

        realizer_refuted = str(realization.get("reason") or "") in {
            "NO_AVAILABLE_MOVE_STRICTLY_REDUCES_RELATION_ERROR",
            "BOUND_TARGET_RELATION_NOT_CURRENTLY_VISIBLE",
            "RELATION_DOES_NOT_HAVE_A_UNIQUE_GOAL_VECTOR",
        }
        repeated_without_commit = (
            realized_id in prior_realized_ids
            and not selection.get("positive_commit_effect_witnessed")
        )
        action_refuted = (
            not bool(transition["action_succeeded"])
            or realizer_refuted
            or repeated_without_commit
        )
        if selected_role == "POSITION" and action_refuted:
            backtrack = _route(
                source,
                episode_id=episode_id,
                decision_index=len(decisions),
                event_name="REFUTED",
                evidence_kind="target_position_effect_refuted",
                evidence_payload={
                    "transition_receipt_sha256": transition["receipt_sha256"],
                    "target_realization_receipt_sha256": realization[
                        "receipt_sha256"
                    ],
                    "realizer_refuted": realizer_refuted,
                    "repeated_target_action": repeated_without_commit,
                    "action_succeeded": bool(transition["action_succeeded"]),
                },
                abstract_action=BACKTRACK,
                native_action_id="recompute_target_candidate_bundle",
                native_action={"operation": "target_native_recompute"},
            )
            if not backtrack["admitted"]:
                raise ValueError("V16 rejected target-native replan")
            decisions.append(backtrack)
        prior_realized_ids.add(realized_id)

    action_counts = Counter(
        row["source_action"] for row in decisions if row["admitted"]
    )
    body = {
        "task_id": episode_id,
        "historical_authentic_official_success": bool(arm["official_success"]),
        "historical_runtime_saw_oracle_scorecard": bool(
            result["policy_runtime_saw_oracle_scorecard"]
        ),
        "v16_source_action_counts": dict(sorted(action_counts.items())),
        "v16_decisions": decisions,
        "recorded_recovery_steps": len(arm.get("recovery") or ()),
        "v16_route_reproduced_every_recorded_action": True,
        "v16_source_abstentions": sum(
            not row["admitted"] for row in decisions
        ),
    }
    return body | {"relineage_sha256": stable_hash(body)}


def evaluate_discovery_relineage(
    *,
    source: SourceSearchAutomaton,
    summary: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    relineage = [
        relineage_discovery_episode(result, source=source) for result in results
    ]
    per_task = {str(row["task_id"]): row for row in summary["per_task"]}
    eligible = tuple(map(str, summary["eligible_task_ids"]))
    if {row["task_id"] for row in relineage} != set(eligible):
        raise ValueError("DiscoveryWorld V16 relineage task coverage differs")
    actions = {
        action
        for row in relineage
        for action in row["v16_source_action_counts"]
    }
    success_counts = dict(summary["success_counts"])
    paired = {}
    for comparator in (TARGET, *CONTROLS):
        wins = losses = 0
        for task_id in eligible:
            authentic = bool(per_task[task_id][AUTHENTIC])
            control = bool(per_task[task_id][comparator])
            wins += authentic and not control
            losses += control and not authentic
        paired[comparator] = {
            "wins": wins,
            "losses": losses,
            "ties": len(eligible) - wins - losses,
            "net_wins": wins - losses,
            "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
        }
    gates = {
        "historical_replication_gates_passed": bool(
            summary["all_predeclared_gates_passed"]
        ),
        "exact_eligible_task_coverage": len(results) == len(eligible),
        "all_v16_routes_or_explicit_target_fallbacks_reproduced_actions": all(
            row["v16_route_reproduced_every_recorded_action"] for row in relineage
        ),
        "all_three_source_actions_exercised": actions
        == {BACKTRACK, COMMIT, EXPLORE},
        "zero_policy_oracle_scorecard_use": not any(
            row["historical_runtime_saw_oracle_scorecard"] for row in relineage
        ),
        "historical_authentic_gain_over_target": (
            success_counts[AUTHENTIC] > success_counts[TARGET]
        ),
        "historical_zero_negative_transfer": paired[TARGET]["losses"] == 0,
        "historical_strict_control_superiority": all(
            success_counts[AUTHENTIC] > success_counts[name]
            for name in CONTROLS
        ),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "discoveryworld-search-automaton-transfer-v16",
        "status": (
            "RETROSPECTIVE_EQUIVALENCE_PASSED_NOT_NEW_CONFIRMATION"
            if passed else "TRANSFER_EQUIVALENCE_GATE_FAILED"
        ),
        "claim_boundary": (
            "RETROSPECTIVE_V16_PROGRAM_EQUIVALENCE_ON_PREVIOUSLY_FRESH_"
            "DISCOVERYWORLD_REPLICATION_RECEIPTS; OUTCOMES REMAIN VALID FOR THE "
            "ORIGINAL SOURCE PROGRAM BUT DO NOT BECOME NEW PROSPECTIVE V16 EVIDENCE"
        ),
        "source_artifact_sha256": source.artifact_sha256,
        "historical_source_program_sha256": summary["source_program_sha256"],
        "historical_success_counts": success_counts,
        "paired": paired,
        "gates": gates,
        "relineage": relineage,
        "missing_confirmation": (
            "A prospectively frozen V16 DiscoveryWorld reserve is still required "
            "for a source-artifact-specific claim."
        ),
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "evaluate_discovery_relineage",
    "relineage_discovery_episode",
]
