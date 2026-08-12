#!/usr/bin/env python3
"""Run matched source-edge/target-abstain forks on consumed ALFWorld tasks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment
from motif_transfer.alfworld_masked_effect_grounder import (
    score_actions,
    validate_artifact as validate_target_artifact,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.parameterized_alfworld_harness import (
    property_router_probabilities,
    validate_property_router,
)
from motif_transfer.relation_edge_value_v13 import (
    extract_relation_edge_features,
)
from motif_transfer.slot_aware_alfworld_harness import (
    initialize_slot_ledger,
    observe_target_transition,
    reconcile_visible_target_objects,
    slot_state,
    validate_slot_source_ir,
)
from motif_transfer.slot_aware_alfworld_harness_v10 import (
    choose_slot_aware_action,
    condition_required_property,
)

from run_slot_aware_alfworld_v8 import (
    _read,
    _sha256,
    _validate_dependency,
    _validate_file_receipt,
    _validate_hash,
    deepcopy_json,
)


TREATMENTS = ("SOURCE_EDGE", "TARGET_ABSTAIN")


def _relative_game_matches(actual: str, expected: str) -> bool:
    normalized = str(actual).replace("\\", "/")
    target = str(expected).replace("\\", "/").lstrip("/")
    return normalized == target or normalized.endswith("/" + target)


def _source_transition(decision: Mapping[str, Any]) -> Mapping[str, Any]:
    transition = decision.get("source_transition")
    if not isinstance(transition, dict) or transition.get("kind") != "EDGE":
        raise RuntimeError("live V13 fork did not reconstruct source EDGE")
    return transition


def _choose(
    *,
    condition: str,
    grounded: Mapping[str, Mapping[str, Any]],
    history: list[str],
    ledger: Mapping[str, Any],
    source_ir: Mapping[str, Any],
    probabilities: Mapping[str, float],
    thresholds: Mapping[str, Any],
    allowed_source_effects: tuple[str, ...],
    active_required_properties: tuple[str, ...],
) -> dict[str, Any]:
    return choose_slot_aware_action(
        condition=condition,
        grounded=grounded,
        history=history,
        ledger=ledger,
        source_ir=source_ir,
        property_probabilities=probabilities,
        minimum_property_confidence=float(
            thresholds["minimum_property_confidence_diagnostic_only"]
        ),
        minimum_role_binding=float(thresholds["minimum_role_binding"]),
        minimum_realization_score=float(
            thresholds["minimum_realization_score"]
        ),
        minimum_target_policy_ratio=float(
            thresholds["minimum_target_policy_ratio"]
        ),
        allowed_source_effects=allowed_source_effects,
        active_required_properties=active_required_properties,
    )


def _replay_prefix(
    *,
    environment: ALFWorldTextBatchEnvironment,
    observation: Any,
    ledger: Mapping[str, Any],
    prefix_actions: list[str],
) -> tuple[Any, dict[str, Any], list[str], float]:
    history: list[str] = []
    total_return = 0.0
    mutable_ledger = dict(ledger)
    for action in prefix_actions:
        mutable_ledger = reconcile_visible_target_objects(
            mutable_ledger,
            str(observation.state.get("observation", "")),
        )
        if action not in observation.native_actions:
            raise RuntimeError(
                f"V13 prefix action is no longer admissible: {action!r}"
            )
        observation, reward = environment.step(action)
        total_return += float(reward)
        mutable_ledger, _ = observe_target_transition(
            mutable_ledger,
            action=action,
            after_observation=str(
                observation.state.get("observation", "")
            ),
        )
        history.append(action)
        if observation.terminal or observation.official_success:
            raise RuntimeError("V13 prefix terminated before frozen fork")
    return observation, mutable_ledger, history, total_return


def _run_branch(
    *,
    environment: ALFWorldTextBatchEnvironment,
    observation: Any,
    opportunity: Mapping[str, Any],
    treatment: str,
    target: Mapping[str, Any],
    router: Mapping[str, Any],
    source_ir: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    allowed_source_effects: tuple[str, ...],
    active_required_properties: tuple[str, ...],
    max_steps: int,
) -> dict[str, Any]:
    goal = str(observation.state.get("task_goal", ""))
    probabilities = property_router_probabilities(goal, router)
    required_property, _, _ = condition_required_property(
        goal, probabilities, "authentic_slot_ir"
    )
    ledger = initialize_slot_ledger(
        goal,
        required_property=required_property,
        initial_observation=str(observation.state.get("observation", "")),
    )
    observation, ledger, history, total_return = _replay_prefix(
        environment=environment,
        observation=observation,
        ledger=ledger,
        prefix_actions=list(map(str, opportunity["prefix_actions"])),
    )
    ledger = reconcile_visible_target_objects(
        ledger, str(observation.state.get("observation", ""))
    )
    if len(history) != int(opportunity["fork_step"]):
        raise RuntimeError("V13 live prefix length differs from frozen step")
    state_body = {
        "task_id": str(opportunity["task_id"]),
        "step": len(history),
        "goal": goal,
        "before": dict(observation.state),
        "native_actions": list(observation.native_actions),
        "ledger_before": deepcopy_json(ledger),
        "history": history,
        "property_probabilities": probabilities,
    }
    state_hash = stable_hash(state_body)
    if state_hash != opportunity["expected_fork_state_sha256"]:
        raise RuntimeError("V13 prefix replay fork-state hash mismatch")
    grounded = score_actions(
        goal=goal,
        observation=str(observation.state.get("observation", "")),
        native_actions=observation.native_actions,
        step=len(history),
        action_history=history,
        artifact=target,
    )
    if not grounded:
        raise RuntimeError("V13 target grounder excluded every fork action")
    proposal = _choose(
        condition="authentic_slot_ir",
        grounded=grounded,
        history=history,
        ledger=ledger,
        source_ir=source_ir,
        probabilities=probabilities,
        thresholds=thresholds,
        allowed_source_effects=allowed_source_effects,
        active_required_properties=active_required_properties,
    )
    transition = _source_transition(proposal)
    if str(transition["graph_sha256"]) != str(
        opportunity["expected_source_graph_sha256"]
    ):
        raise RuntimeError("V13 live source graph differs from frozen receipt")
    expected_edge = opportunity["expected_source_edge"]
    if any(transition[key] != expected_edge[key] for key in expected_edge):
        raise RuntimeError("V13 live source edge differs from frozen receipt")
    if str(proposal["fallback_action"]) != str(
        opportunity["expected_fallback_action"]
    ):
        raise RuntimeError("V13 live fallback differs from frozen trace")
    source_action = str(proposal["action"])
    control_action = str(proposal["fallback_action"])
    selected_action = (
        source_action if treatment == "SOURCE_EDGE" else control_action
    )
    if selected_action not in observation.native_actions:
        raise RuntimeError("V13 frozen fork action is not native/admissible")
    features = extract_relation_edge_features(
        decision=proposal,
        grounded=grounded,
        ledger=ledger,
        step=len(history),
        max_steps=max_steps,
        native_action_count=len(observation.native_actions),
    )
    before_ledger = deepcopy_json(ledger)
    observation, reward = environment.step(selected_action)
    total_return += float(reward)
    ledger, fork_receipt = observe_target_transition(
        ledger,
        action=selected_action,
        after_observation=str(observation.state.get("observation", "")),
    )
    history.append(selected_action)
    trajectory = [{
        "step": len(history) - 1,
        "action": selected_action,
        "policy": "FROZEN_FORK_TREATMENT",
        "target_effect_receipt": fork_receipt,
        "official_success_after": bool(observation.official_success),
    }]
    while (
        len(history) < max_steps
        and not observation.terminal
        and not observation.official_success
    ):
        ledger = reconcile_visible_target_objects(
            ledger, str(observation.state.get("observation", ""))
        )
        grounded = score_actions(
            goal=goal,
            observation=str(observation.state.get("observation", "")),
            native_actions=observation.native_actions,
            step=len(history),
            action_history=history,
            artifact=target,
        )
        if not grounded:
            raise RuntimeError("V13 continuation grounder excluded every action")
        continuation = _choose(
            condition="edge_permuted_ir",
            grounded=grounded,
            history=history,
            ledger=ledger,
            source_ir=source_ir,
            probabilities=probabilities,
            thresholds=thresholds,
            allowed_source_effects=allowed_source_effects,
            active_required_properties=active_required_properties,
        )
        action = str(continuation["action"])
        observation, reward = environment.step(action)
        total_return += float(reward)
        ledger, receipt = observe_target_transition(
            ledger,
            action=action,
            after_observation=str(
                observation.state.get("observation", "")
            ),
        )
        history.append(action)
        trajectory.append({
            "step": len(history) - 1,
            "action": action,
            "policy": "EDGE_PERMUTED_NODE_ONLY_CONTINUATION",
            "target_effect_receipt": receipt,
            "official_success_after": bool(observation.official_success),
        })
    final = slot_state(ledger)
    required = max(int(final["required_count"]), 1)
    branch_body = {
        "treatment": treatment,
        "task_id": str(opportunity["task_id"]),
        "fork_id": str(opportunity["fork_id"]),
        "fork_state_sha256": state_hash,
        "source_action": source_action,
        "control_action": control_action,
        "selected_action": selected_action,
        "informative_action_contrast": source_action != control_action,
        "features": features,
        "features_sha256": stable_hash(features),
        "fork_ledger_before": before_ledger,
        "fork_target_effect_receipt": fork_receipt,
        "fork_relation_postcondition_observed": (
            fork_receipt == "RELATE_SLOT_CLOSED"
        ),
        "official_success": bool(observation.official_success),
        "steps": len(history),
        "return": total_return,
        "completed_fraction": float(final["completed_count"]) / required,
        "final_slot_state": final,
        "trajectory": trajectory,
    }
    return branch_body | {"branch_sha256": stable_hash(branch_body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V13 fork report: {args.output}")
    plan = _read(args.plan)
    _validate_hash(plan, "plan_sha256")
    if plan.get("status") != "FROZEN_BEFORE_ANY_V13_FORK_OUTCOME":
        raise SystemExit("V13 fork plan was not frozen before outcomes")
    if plan.get("existing_valid_unseen_heldout_read"):
        raise SystemExit("V13 plan crossed heldout boundary")
    for receipt in plan["implementation"].values():
        _validate_file_receipt(receipt)
    versions = sorted(plan["reports"])
    canonical_candidate = _read(Path(
        plan["reports"][versions[0]]["candidate"]["path"]
    ))
    _validate_hash(canonical_candidate, "candidate_sha256")
    target = _validate_dependency(canonical_candidate["target_grounder"])
    validate_target_artifact(target)
    router = dict(canonical_candidate["property_router"])
    validate_property_router(router)
    source_ir = dict(canonical_candidate["slot_source_ir"])
    validate_slot_source_ir(source_ir)
    thresholds = dict(canonical_candidate["thresholds"])
    allowed_source_effects = tuple(map(
        str,
        canonical_candidate["transfer_scope"]["allowed_source_effects"],
    ))
    active_required_properties = tuple(map(
        str,
        canonical_candidate["transfer_scope"][
            "active_required_properties"
        ],
    ))
    max_steps = int(plan["max_steps"])
    by_version: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for opportunity in plan["opportunities"]:
        by_version[str(opportunity["version"])].append(opportunity)
    result_by_fork: dict[str, dict[str, Any]] = {}
    for version in versions:
        opportunities = by_version[version]
        for treatment in TREATMENTS:
            environment = ALFWorldTextBatchEnvironment(
                config_path=str(args.alfworld_config.resolve()),
                data_path=str(args.alfworld_data.resolve()),
                split="train",
                seed=int(plan["reports"][version]["runner_seed"]),
                game_ids=tuple(
                    str(row["task_id"]) for row in opportunities
                ),
                max_steps=max_steps,
            )
            seen_tasks: set[str] = set()
            try:
                for index in range(len(opportunities)):
                    observation = environment.reset()
                    matches = [
                        row for row in opportunities
                        if _relative_game_matches(
                            environment.resolved_game_file,
                            str(row["task_id"]),
                        )
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(
                            "V13 batch reset did not map to exactly one "
                            "frozen task"
                        )
                    opportunity = matches[0]
                    task_id = str(opportunity["task_id"])
                    if task_id in seen_tasks:
                        raise RuntimeError(
                            "V13 batch reset repeated a task within treatment"
                        )
                    seen_tasks.add(task_id)
                    branch = _run_branch(
                        environment=environment,
                        observation=observation,
                        opportunity=opportunity,
                        treatment=treatment,
                        target=target,
                        router=router,
                        source_ir=source_ir,
                        thresholds=thresholds,
                        allowed_source_effects=allowed_source_effects,
                        active_required_properties=active_required_properties,
                        max_steps=max_steps,
                    )
                    fork_id = str(opportunity["fork_id"])
                    if fork_id not in result_by_fork:
                        result_by_fork[fork_id] = dict(opportunity) | {
                            "branches": {}
                        }
                    result_by_fork[fork_id]["branches"][treatment] = branch
                    print(json.dumps({
                        "version": version,
                        "branch_index": index,
                        "branch_count": len(opportunities),
                        "task_id": opportunity["task_id"],
                        "treatment": treatment,
                        "success": branch["official_success"],
                        "steps": branch["steps"],
                        "informative": branch[
                            "informative_action_contrast"
                        ],
                    }), flush=True)
            finally:
                environment.close()
    forks = []
    for opportunity in plan["opportunities"]:
        row = result_by_fork[str(opportunity["fork_id"])]
        branches = row["branches"]
        if set(branches) != set(TREATMENTS):
            raise RuntimeError("V13 fork is missing a matched treatment")
        source = branches["SOURCE_EDGE"]
        control = branches["TARGET_ABSTAIN"]
        invariants = {
            "fork_state_match": (
                source["fork_state_sha256"]
                == control["fork_state_sha256"]
                == opportunity["expected_fork_state_sha256"]
            ),
            "source_action_match": (
                source["source_action"] == control["source_action"]
            ),
            "control_action_match": (
                source["control_action"] == control["control_action"]
            ),
            "feature_match": (
                source["features_sha256"] == control["features_sha256"]
            ),
        }
        if not all(invariants.values()):
            raise RuntimeError("V13 matched-fork invariant failed")
        row["invariants"] = invariants
        row["informative_action_contrast"] = bool(
            source["informative_action_contrast"]
        )
        row["features"] = source["features"]
        row["features_sha256"] = source["features_sha256"]
        row["fork_sha256"] = stable_hash({
            key: value for key, value in row.items()
            if key != "fork_sha256"
        })
        forks.append(row)
    body = {
        "schema_version": "relation-edge-intervention-fork-report-v13",
        "status": "CONSUMED_MATCHED_FORKS_COMPLETE",
        "claim_boundary": (
            "MATCHED_FORK_OUTCOMES_ON_CONSUMED_TASKS_ONLY; NOT_FRESH_"
            "TRANSFER_EVIDENCE; CONFIRMATION_AND_VALID_UNSEEN_UNREAD"
        ),
        "plan": {
            "path": str(args.plan.resolve()),
            "file_sha256": _sha256(args.plan),
            "plan_sha256": plan["plan_sha256"],
        },
        "max_steps": max_steps,
        "fork_count": len(forks),
        "informative_fork_count": sum(
            bool(row["informative_action_contrast"]) for row in forks
        ),
        "matched_cell_count": len(forks) * len(TREATMENTS),
        "forks": forks,
        "all_matched_invariants_passed": all(
            all(row["invariants"].values()) for row in forks
        ),
        "confirmation_read": False,
        "existing_valid_unseen_heldout_read": False,
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "report_sha256": report["report_sha256"],
        "fork_count": report["fork_count"],
        "informative_fork_count": report["informative_fork_count"],
        "matched_cell_count": report["matched_cell_count"],
        "all_matched_invariants_passed": report[
            "all_matched_invariants_passed"
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
