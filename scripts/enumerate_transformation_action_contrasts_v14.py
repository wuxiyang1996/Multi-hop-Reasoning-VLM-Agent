#!/usr/bin/env python3
"""Enumerate source-graph/target action contrasts without reading outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
)


def _relative_game_matches(actual: str, expected: str) -> bool:
    normalized = str(actual).replace("\\", "/")
    target = str(expected).replace("\\", "/").lstrip("/")
    return normalized == target or normalized.endswith("/" + target)


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


def _edge_transition(decision: Mapping[str, Any]) -> Mapping[str, Any] | None:
    transition = decision.get("source_transition")
    if isinstance(transition, dict) and transition.get("kind") == "EDGE":
        return transition
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(
            f"refusing to overwrite V14 contrast report: {args.output}"
        )
    pool = _read(args.pool)
    _validate_hash(pool, "pool_sha256")
    if pool.get("status") != "FROZEN_BEFORE_ANY_V14_SELECTED_TASK_RESET":
        raise SystemExit("V14 pool was not frozen before task reset")
    if pool.get("selection_used_target_rollout_outcomes"):
        raise SystemExit("V14 pool selection used target outcomes")
    if pool.get("existing_valid_unseen_heldout_read"):
        raise SystemExit("V14 pool crossed existing heldout boundary")
    for receipt in pool["implementation"].values():
        _validate_file_receipt(receipt)
    parent_receipt = pool["parent_candidate"]
    parent = _read(Path(str(parent_receipt["path"])))
    _validate_hash(parent, "candidate_sha256")
    if (
        _sha256(Path(str(parent_receipt["path"])))
        != parent_receipt["file_sha256"]
        or parent["candidate_sha256"]
        != parent_receipt["candidate_sha256"]
    ):
        raise SystemExit("V14 parent candidate receipt changed")
    target = _validate_dependency(parent["target_grounder"])
    validate_target_artifact(target)
    router = dict(parent["property_router"])
    validate_property_router(router)
    source_ir = dict(parent["slot_source_ir"])
    validate_slot_source_ir(source_ir)
    thresholds = dict(parent["thresholds"])
    allowed_source_effects = tuple(map(
        str, pool["allowed_source_effects"]
    ))
    active_required_properties = tuple(map(
        str, pool["active_required_properties"]
    ))
    task_ids = tuple(map(
        str, pool["splits"]["outcome_blind_contrast_preflight"]
    ))
    max_steps = int(pool["max_steps"])
    environment = ALFWorldTextBatchEnvironment(
        config_path=str(args.alfworld_config.resolve()),
        data_path=str(args.alfworld_data.resolve()),
        split="train",
        seed=int(pool["seed"]),
        game_ids=task_ids,
        max_steps=max_steps,
    )
    seen: set[str] = set()
    tasks = []
    try:
        for task_index in range(len(task_ids)):
            observation = environment.reset()
            matches = [
                task_id for task_id in task_ids
                if _relative_game_matches(
                    environment.resolved_game_file, task_id
                )
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "V14 reset did not map to exactly one frozen identity"
                )
            task_id = matches[0]
            if task_id in seen:
                raise RuntimeError("V14 reset repeated a frozen identity")
            seen.add(task_id)
            family = task_id.split("-", 1)[0]
            goal = str(observation.state.get("task_goal", ""))
            probabilities = property_router_probabilities(goal, router)
            required_property, _, _ = condition_required_property(
                goal, probabilities, "authentic_slot_ir"
            )
            ledger = initialize_slot_ledger(
                goal,
                required_property=required_property,
                initial_observation=str(
                    observation.state.get("observation", "")
                ),
            )
            history: list[str] = []
            opportunities = []
            contrasts = []
            for step in range(max_steps):
                ledger = reconcile_visible_target_objects(
                    ledger,
                    str(observation.state.get("observation", "")),
                )
                grounded = score_actions(
                    goal=goal,
                    observation=str(
                        observation.state.get("observation", "")
                    ),
                    native_actions=observation.native_actions,
                    step=step,
                    action_history=history,
                    artifact=target,
                )
                if not grounded:
                    raise RuntimeError(
                        "V14 target grounder excluded every native action"
                    )
                baseline = _choose(
                    condition="property_permuted_router",
                    grounded=grounded,
                    history=history,
                    ledger=ledger,
                    source_ir=source_ir,
                    probabilities=probabilities,
                    thresholds=thresholds,
                    allowed_source_effects=allowed_source_effects,
                    active_required_properties=active_required_properties,
                )
                shadow = _choose(
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
                transition = _edge_transition(shadow)
                if transition is not None and (
                    "best_realization_score" in shadow
                    and "target_policy_ratio" in shadow
                ):
                    row_body = {
                        "task_id": task_id,
                        "task_family": family,
                        "step": step,
                        "required_property": required_property,
                        "source_transition": dict(transition),
                        "requested_source_effect": str(
                            shadow["requested_source_effect"]
                        ),
                        "source_action": str(shadow["action"]),
                        "target_control_action": str(baseline["action"]),
                        "source_admitted": bool(shadow["source_admitted"]),
                        "action_contrast": bool(
                            shadow["source_admitted"]
                            and shadow["action"] != baseline["action"]
                        ),
                        "source_target_policy_ratio": float(
                            shadow["target_policy_ratio"]
                        ),
                        "source_realization_score": float(
                            shadow["best_realization_score"]
                        ),
                        "slot_state": shadow["slot_state"],
                        "native_action_count": len(
                            observation.native_actions
                        ),
                    }
                    row = row_body | {
                        "opportunity_sha256": stable_hash(row_body)
                    }
                    opportunities.append(row)
                    if row["action_contrast"]:
                        contrasts.append(row)
                selected = str(baseline["action"])
                observation, _discarded_reward = environment.step(selected)
                ledger, _receipt = observe_target_transition(
                    ledger,
                    action=selected,
                    after_observation=str(
                        observation.state.get("observation", "")
                    ),
                )
                history.append(selected)
                if observation.terminal:
                    break
            task_body = {
                "task_index": task_index,
                "task_id": task_id,
                "task_family": family,
                "required_property": required_property,
                "steps_executed": len(history),
                "edge_opportunity_count": len(opportunities),
                "edge_action_contrast_count": len(contrasts),
                "contrast_effects": dict(Counter(
                    str(row["requested_source_effect"])
                    for row in contrasts
                )),
                "first_contrast": contrasts[0] if contrasts else None,
                "final_slot_state_without_outcome": slot_state(ledger),
            }
            tasks.append(task_body | {
                "task_receipt_sha256": stable_hash(task_body)
            })
            print(json.dumps({
                "task_index": task_index,
                "task_count": len(task_ids),
                "task_id": task_id,
                "family": family,
                "steps": len(history),
                "edge_opportunities": len(opportunities),
                "edge_action_contrasts": len(contrasts),
                "contrast_effects": task_body["contrast_effects"],
                "outcomes_recorded": False,
            }), flush=True)
    finally:
        environment.close()
    if seen != set(task_ids):
        raise RuntimeError("V14 did not enumerate every frozen task")
    contrast_tasks = [
        row for row in tasks if row["edge_action_contrast_count"] > 0
    ]
    by_family = defaultdict(list)
    for row in contrast_tasks:
        by_family[str(row["task_family"])].append(row)
    mutate_tasks = [
        row for row in contrast_tasks
        if int(row["contrast_effects"].get("MUTATE", 0)) > 0
    ]
    relate_tasks = [
        row for row in contrast_tasks
        if int(row["contrast_effects"].get("RELATE", 0)) > 0
    ]
    requirements = pool["contrast_gate"]
    gates = {
        "minimum_tasks_with_edge_action_contrast": (
            len(contrast_tasks) >= int(requirements[
                "minimum_tasks_with_edge_action_contrast"
            ])
        ),
        "minimum_families_with_four_contrast_tasks": (
            sum(len(rows) >= 4 for rows in by_family.values())
            >= int(requirements[
                "minimum_families_with_four_contrast_tasks"
            ])
        ),
        "minimum_mutate_contrast_tasks": (
            len(mutate_tasks) >= int(requirements[
                "minimum_mutate_contrast_tasks"
            ])
        ),
        "minimum_relate_contrast_tasks": (
            len(relate_tasks) >= int(requirements[
                "minimum_relate_contrast_tasks"
            ])
        ),
        "zero_outcomes_recorded": True,
        "zero_identity_or_receipt_failures": True,
    }
    passed = all(gates.values())
    body = {
        "schema_version": "transformation-action-contrast-report-v14",
        "status": (
            "OUTCOME_BLIND_CONTRAST_GATE_PASSED"
            if passed else "OUTCOME_BLIND_CONTRAST_GATE_FAILED_STOP"
        ),
        "claim_boundary": (
            "ACTION_CONTRAST_FEASIBILITY_ONLY; REWARD_AND_OFFICIAL_SUCCESS_"
            "DISCARDED_AND_NOT_SERIALIZED; SELECTED_TASKS_NOW CONSUMED_"
            "DEVELOPMENT; NO TRANSFER_VALUE CLAIM; CONFIRMATION_AND_"
            "EXISTING_VALID_UNSEEN_UNREAD"
        ),
        "pool": {
            "path": str(args.pool.resolve()),
            "file_sha256": _sha256(args.pool),
            "pool_sha256": pool["pool_sha256"],
        },
        "task_count": len(tasks),
        "edge_opportunity_task_count": sum(
            row["edge_opportunity_count"] > 0 for row in tasks
        ),
        "edge_action_contrast_task_count": len(contrast_tasks),
        "mutate_contrast_task_count": len(mutate_tasks),
        "relate_contrast_task_count": len(relate_tasks),
        "contrast_tasks_by_family": {
            family: len(by_family.get(family, ()))
            for family in pool["target_families"]
        },
        "tasks": tasks,
        "gates": gates,
        "passed": passed,
        "outcomes_recorded": False,
        "rewards_recorded": False,
        "next_step": (
            "FREEZE_MATCHED_TRANSFORMATION_FORK_PLAN_ON_CONSUMED_V14_TASKS"
            if passed else "STOP_BEFORE_TRANSFORMATION_OUTCOME_COLLECTION"
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
        "status": report["status"],
        "report_sha256": report["report_sha256"],
        "task_count": len(tasks),
        "edge_action_contrast_task_count": len(contrast_tasks),
        "mutate_contrast_task_count": len(mutate_tasks),
        "relate_contrast_task_count": len(relate_tasks),
        "contrast_tasks_by_family": report["contrast_tasks_by_family"],
        "gates": gates,
        "next_step": report["next_step"],
        "outcomes_recorded": False,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
