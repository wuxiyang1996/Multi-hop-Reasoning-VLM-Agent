#!/usr/bin/env python3
"""Strictly audit V15 option-level contrasts without reading target outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.alfworld_hierarchical_grounder import (  # noqa: E402
    parse_goal,
    score_actions,
    workflow_status,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.hierarchical_skill_transfer import (  # noqa: E402
    deserialize_ensemble,
)
from enumerate_subgoal_option_contrasts_v15 import (  # noqa: E402
    _decision,
    _read,
    _relative_game_matches,
    _sha256,
    _validate_receipt,
)


def strict_gates(tasks: list[dict], requirements: dict) -> dict[str, bool]:
    option_tasks = [row for row in tasks if row["option_contrast_count"] > 0]
    source_specific = [
        row for row in tasks if row["source_specific_option_contrast_count"] > 0
    ]
    second_cycle = [
        row for row in tasks if row["second_cycle_option_contrast_count"] > 0
    ]
    by_destination: dict[str, list[dict]] = defaultdict(list)
    for row in source_specific:
        by_destination[str(row["destination"])].append(row)
    return {
        "minimum_tasks_with_option_contrast": (
            len(option_tasks)
            >= int(requirements["minimum_tasks_with_option_contrast"])
        ),
        "minimum_tasks_with_source_specific_option_contrast": (
            len(source_specific)
            >= int(requirements[
                "minimum_tasks_with_source_specific_option_contrast"
            ])
        ),
        "minimum_tasks_with_second_cycle_option_contrast": (
            len(second_cycle)
            >= int(requirements[
                "minimum_tasks_with_second_cycle_option_contrast"
            ])
        ),
        "minimum_destination_groups_with_four_source_specific_tasks": (
            sum(len(rows) >= 4 for rows in by_destination.values())
            >= int(requirements[
                "minimum_destination_groups_with_four_source_specific_tasks"
            ])
        ),
        "zero_outcomes_recorded": True,
        "zero_identity_or_receipt_failures": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V15 strict report: {args.output}")
    plan = _read(args.plan)
    if stable_hash({
        key: value for key, value in plan.items() if key != "plan_sha256"
    }) != plan.get("plan_sha256"):
        raise SystemExit("V15 strict plan hash mismatch")
    if plan.get("status") != (
        "FROZEN_BEFORE_STRICT_AUDIT_REPLAY_OF_CONSUMED_V15_TASKS"
    ):
        raise SystemExit("V15 strict plan has unexpected authority")
    for receipt in plan["implementation"].values():
        _validate_receipt(receipt)
    pool_path = _validate_receipt(plan["broad_pool"])
    _validate_receipt(plan["quarantined_broad_report"])
    pool = _read(pool_path)
    if pool["pool_sha256"] != plan["broad_pool"]["pool_sha256"]:
        raise SystemExit("V15 strict plan references a different broad pool")
    source = pool["source_controller"]
    artifact_path = _validate_receipt(source)
    _validate_receipt(source["historical_config"])
    _validate_receipt(source["historical_result"])
    artifact = _read(artifact_path)
    models = {
        name: deserialize_ensemble(artifact["source"]["models"][name])
        for name in (
            "authentic_source_plus_target",
            "phase_permuted_source_plus_target",
        )
    }
    policy = source["policy"]
    grounder = artifact["target_grounder"]
    task_ids = tuple(map(str, plan["task_ids"]))
    max_steps = int(plan["max_steps"])
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
                if _relative_game_matches(environment.resolved_game_file, task_id)
            ]
            if len(matches) != 1:
                raise RuntimeError("V15 strict reset identity mismatch")
            task_id = matches[0]
            if task_id in seen:
                raise RuntimeError("V15 strict reset repeated a task identity")
            seen.add(task_id)
            goal = str(observation.state.get("task_goal", ""))
            spec = parse_goal(goal)
            history: list[str] = []
            opportunities = []
            for step in range(max_steps):
                grounded = score_actions(
                    goal=goal,
                    observation=str(observation.state.get("observation", "")),
                    native_actions=observation.native_actions,
                    step=step,
                    action_history=history,
                    artifact=grounder,
                )
                if not grounded:
                    raise RuntimeError("V15 strict grounder excluded every action")
                decisions = {
                    "target": _decision(
                        grounded=grounded,
                        goal=goal,
                        step=step,
                        max_steps=max_steps,
                        history=history,
                        source_model=None,
                        uncertainty_scale=float(policy["uncertainty_scale"]),
                        decision_margin=float(policy["decision_margin"]),
                    ),
                    "authentic": _decision(
                        grounded=grounded,
                        goal=goal,
                        step=step,
                        max_steps=max_steps,
                        history=history,
                        source_model=models["authentic_source_plus_target"],
                        uncertainty_scale=float(policy["uncertainty_scale"]),
                        decision_margin=float(policy["decision_margin"]),
                    ),
                    "phase": _decision(
                        grounded=grounded,
                        goal=goal,
                        step=step,
                        max_steps=max_steps,
                        history=history,
                        source_model=models["phase_permuted_source_plus_target"],
                        uncertainty_scale=float(policy["uncertainty_scale"]),
                        decision_margin=float(policy["decision_margin"]),
                    ),
                }
                actions = {
                    name: str(decision["action"])
                    for name, decision in decisions.items()
                }
                options = {
                    name: str(grounded[action]["option"])
                    for name, action in actions.items()
                }
                if options["authentic"] != options["target"]:
                    status = workflow_status(goal, history)
                    before_body = {
                        "task_id": task_id,
                        "step": step,
                        "goal": goal,
                        "history": list(history),
                        "native_actions": list(map(str, observation.native_actions)),
                        "state": dict(observation.state),
                    }
                    row_body = {
                        "task_id": task_id,
                        "step": step,
                        "destination": spec.destination,
                        "placed_count_before": status.placed_count,
                        "workflow_progress_before": status.progress_fraction,
                        "prefix_actions": list(history),
                        "before_state_sha256": stable_hash(before_body),
                        "actions": actions,
                        "options": options,
                        "source_specific": bool(
                            options["authentic"] != options["phase"]
                        ),
                    }
                    opportunities.append(row_body | {
                        "opportunity_sha256": stable_hash(row_body)
                    })
                selected = actions["target"]
                observation, _discarded_reward = environment.step(selected)
                history.append(selected)
                if observation.terminal:
                    break
            task_body = {
                "task_index": task_index,
                "task_id": task_id,
                "destination": spec.destination,
                "steps_executed": len(history),
                "option_contrast_count": len(opportunities),
                "source_specific_option_contrast_count": sum(
                    row["source_specific"] for row in opportunities
                ),
                "second_cycle_option_contrast_count": sum(
                    row["placed_count_before"] >= 1 for row in opportunities
                ),
                "option_pairs": dict(Counter(
                    f"{row['options']['target']}->{row['options']['authentic']}"
                    for row in opportunities
                )),
                "opportunities": opportunities,
            }
            tasks.append(task_body | {
                "task_receipt_sha256": stable_hash(task_body)
            })
            print(json.dumps({
                "task_index": task_index,
                "task_count": len(task_ids),
                "task_id": task_id,
                "option_contrasts": len(opportunities),
                "source_specific": task_body[
                    "source_specific_option_contrast_count"
                ],
                "second_cycle": task_body["second_cycle_option_contrast_count"],
                "outcomes_recorded": False,
            }), flush=True)
    finally:
        environment.close()
    if seen != set(task_ids):
        raise RuntimeError("V15 strict audit did not replay every task")

    gates = strict_gates(tasks, plan["contrast_gate"])
    passed = all(gates.values())
    option_tasks = [row for row in tasks if row["option_contrast_count"] > 0]
    source_specific = [
        row for row in tasks if row["source_specific_option_contrast_count"] > 0
    ]
    second_cycle = [
        row for row in tasks if row["second_cycle_option_contrast_count"] > 0
    ]
    body = {
        "schema_version": "strict-subgoal-option-audit-report-v15",
        "status": (
            "STRICT_OUTCOME_BLIND_OPTION_CONTRAST_GATE_PASSED"
            if passed else "STRICT_OUTCOME_BLIND_OPTION_CONTRAST_GATE_FAILED_STOP"
        ),
        "claim_boundary": plan["claim_boundary"],
        "plan": {
            "path": str(args.plan.resolve()),
            "file_sha256": _sha256(args.plan),
            "plan_sha256": plan["plan_sha256"],
        },
        "counts": {
            "tasks": len(tasks),
            "tasks_with_option_contrast": len(option_tasks),
            "tasks_with_source_specific_option_contrast": len(source_specific),
            "tasks_with_second_cycle_option_contrast": len(second_cycle),
            "option_contrasts": sum(row["option_contrast_count"] for row in tasks),
            "source_specific_option_contrasts": sum(
                row["source_specific_option_contrast_count"] for row in tasks
            ),
            "second_cycle_option_contrasts": sum(
                row["second_cycle_option_contrast_count"] for row in tasks
            ),
        },
        "gates": gates,
        "tasks": tasks,
        "reward_serialized": False,
        "official_success_serialized": False,
        "held_out_read_or_run": False,
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "counts": report["counts"],
        "gates": gates,
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
