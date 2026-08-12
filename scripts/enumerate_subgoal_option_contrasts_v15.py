#!/usr/bin/env python3
"""Enumerate V15 option/action contrasts while discarding ALFWorld outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


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
from run_multisource_alfworld_v2_qualification import (  # noqa: E402
    _choose_action,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_game_matches(actual: str, expected: str) -> bool:
    normalized = str(actual).replace("\\", "/")
    target = str(expected).replace("\\", "/").lstrip("/")
    return normalized == target or normalized.endswith("/" + target)


def _validate_receipt(receipt: Mapping[str, str]) -> Path:
    path = Path(str(receipt["path"])).resolve()
    if _sha256(path) != str(receipt["file_sha256"]):
        raise ValueError(f"dependency hash mismatch: {path}")
    return path


def summarize_gates(tasks: list[dict], requirements: Mapping[str, int]) -> dict:
    authentic = [row for row in tasks if row["authentic_action_contrast_count"] > 0]
    source_specific = [
        row for row in tasks
        if row["authentic_phase_action_contrast_count"] > 0
    ]
    second_cycle = [
        row for row in tasks
        if row["second_cycle_authentic_contrast_count"] > 0
    ]
    by_destination: dict[str, list[dict]] = defaultdict(list)
    for row in authentic:
        by_destination[str(row["destination"])].append(row)
    return {
        "minimum_tasks_with_authentic_action_contrast": (
            len(authentic)
            >= int(requirements["minimum_tasks_with_authentic_action_contrast"])
        ),
        "minimum_tasks_with_authentic_phase_action_contrast": (
            len(source_specific)
            >= int(requirements[
                "minimum_tasks_with_authentic_phase_action_contrast"
            ])
        ),
        "minimum_tasks_with_second_cycle_authentic_contrast": (
            len(second_cycle)
            >= int(requirements[
                "minimum_tasks_with_second_cycle_authentic_contrast"
            ])
        ),
        "minimum_destination_groups_with_four_authentic_contrasts": (
            sum(len(rows) >= 4 for rows in by_destination.values())
            >= int(requirements[
                "minimum_destination_groups_with_four_authentic_contrasts"
            ])
        ),
        "zero_outcomes_recorded": True,
        "zero_identity_or_receipt_failures": True,
    }


def _decision(
    *,
    grounded: Mapping[str, Mapping[str, float | str]],
    goal: str,
    step: int,
    max_steps: int,
    history: list[str],
    source_model: Any,
    uncertainty_scale: float,
    decision_margin: float,
) -> dict[str, Any]:
    return _choose_action(
        grounded=grounded,
        goal=goal,
        step=step,
        max_steps=max_steps,
        history=history,
        source_model=source_model,
        uncertainty_scale=uncertainty_scale,
        decision_margin=decision_margin,
        controller="SOURCE_OPTION_ONLY_V3_DIAGNOSTIC",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V15 report: {args.output}")
    pool = _read(args.pool)
    if stable_hash({
        key: value for key, value in pool.items() if key != "pool_sha256"
    }) != pool.get("pool_sha256"):
        raise SystemExit("V15 pool hash mismatch")
    if pool.get("status") != "FROZEN_BEFORE_ANY_V15_SELECTED_TASK_RESET":
        raise SystemExit("V15 pool was not frozen before task reset")
    if pool.get("selection_used_target_rollout_outcomes"):
        raise SystemExit("V15 selection used target outcomes")
    if pool.get("existing_valid_unseen_heldout_read"):
        raise SystemExit("V15 pool crossed the held-out boundary")
    for receipt in pool["implementation"].values():
        _validate_receipt(receipt)
    artifact_path = _validate_receipt(pool["source_controller"])
    _validate_receipt(pool["source_controller"]["historical_config"])
    _validate_receipt(pool["source_controller"]["historical_result"])
    artifact = _read(artifact_path)
    models = {
        name: deserialize_ensemble(artifact["source"]["models"][name])
        for name in (
            "authentic_source_plus_target",
            "phase_permuted_source_plus_target",
        )
    }
    grounder = artifact["target_grounder"]
    task_ids = tuple(map(
        str, pool["splits"]["outcome_blind_subgoal_contrast_preflight"]
    ))
    max_steps = int(pool["max_steps"])
    policy = pool["source_controller"]["policy"]
    if policy.get("controller") != "SOURCE_OPTION_ONLY_V3_DIAGNOSTIC":
        raise SystemExit("V15 pool does not bind the option-only controller")
    uncertainty_scale = float(policy["uncertainty_scale"])
    decision_margin = float(policy["decision_margin"])
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
                raise RuntimeError("V15 reset did not map to one frozen identity")
            task_id = matches[0]
            if task_id in seen:
                raise RuntimeError("V15 reset repeated a frozen identity")
            seen.add(task_id)
            goal = str(observation.state.get("task_goal", ""))
            spec = parse_goal(goal)
            if spec.count != 2:
                raise RuntimeError("V15 selected task is not a two-object goal")
            history: list[str] = []
            contrasts = []
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
                    raise RuntimeError("V15 target grounder excluded every action")
                target = _decision(
                    grounded=grounded,
                    goal=goal,
                    step=step,
                    max_steps=max_steps,
                    history=history,
                    source_model=None,
                    uncertainty_scale=uncertainty_scale,
                    decision_margin=decision_margin,
                )
                authentic = _decision(
                    grounded=grounded,
                    goal=goal,
                    step=step,
                    max_steps=max_steps,
                    history=history,
                    source_model=models["authentic_source_plus_target"],
                    uncertainty_scale=uncertainty_scale,
                    decision_margin=decision_margin,
                )
                phase = _decision(
                    grounded=grounded,
                    goal=goal,
                    step=step,
                    max_steps=max_steps,
                    history=history,
                    source_model=models["phase_permuted_source_plus_target"],
                    uncertainty_scale=uncertainty_scale,
                    decision_margin=decision_margin,
                )
                status = workflow_status(goal, history)
                authentic_contrast = bool(
                    authentic["source_admitted"]
                    and authentic["action"] != target["action"]
                )
                if authentic_contrast:
                    row_body = {
                        "task_id": task_id,
                        "step": step,
                        "destination": spec.destination,
                        "placed_count_before": status.placed_count,
                        "workflow_progress_before": status.progress_fraction,
                        "target_action": str(target["action"]),
                        "target_option": str(
                            grounded[str(target["action"])]["option"]
                        ),
                        "authentic_action": str(authentic["action"]),
                        "authentic_option": str(
                            grounded[str(authentic["action"])]["option"]
                        ),
                        "phase_action": str(phase["action"]),
                        "phase_option": str(
                            grounded[str(phase["action"])]["option"]
                        ),
                        "authentic_phase_action_contrast": bool(
                            authentic["action"] != phase["action"]
                        ),
                        "native_action_count": len(observation.native_actions),
                    }
                    contrasts.append(row_body | {
                        "contrast_sha256": stable_hash(row_body)
                    })
                selected = str(target["action"])
                observation, _discarded_reward = environment.step(selected)
                history.append(selected)
                if observation.terminal:
                    break
            task_body = {
                "task_index": task_index,
                "task_id": task_id,
                "destination": spec.destination,
                "steps_executed": len(history),
                "authentic_action_contrast_count": len(contrasts),
                "authentic_phase_action_contrast_count": sum(
                    row["authentic_phase_action_contrast"] for row in contrasts
                ),
                "second_cycle_authentic_contrast_count": sum(
                    row["placed_count_before"] >= 1 for row in contrasts
                ),
                "contrast_option_pairs": dict(Counter(
                    f"{row['target_option']}->{row['authentic_option']}"
                    for row in contrasts
                )),
                "first_contrast": contrasts[0] if contrasts else None,
            }
            tasks.append(task_body | {
                "task_receipt_sha256": stable_hash(task_body)
            })
            print(json.dumps({
                "task_index": task_index,
                "task_count": len(task_ids),
                "task_id": task_id,
                "destination": spec.destination,
                "steps": len(history),
                "authentic_contrasts": len(contrasts),
                "source_control_contrasts": task_body[
                    "authentic_phase_action_contrast_count"
                ],
                "second_cycle_contrasts": task_body[
                    "second_cycle_authentic_contrast_count"
                ],
                "outcomes_recorded": False,
            }), flush=True)
    finally:
        environment.close()
    if seen != set(task_ids):
        raise RuntimeError("V15 did not enumerate every frozen task")

    gates = summarize_gates(tasks, pool["contrast_gate"])
    passed = all(gates.values())
    authentic_tasks = [
        row for row in tasks if row["authentic_action_contrast_count"] > 0
    ]
    source_specific_tasks = [
        row for row in tasks
        if row["authentic_phase_action_contrast_count"] > 0
    ]
    second_cycle_tasks = [
        row for row in tasks
        if row["second_cycle_authentic_contrast_count"] > 0
    ]
    destination_counts = Counter(
        row["destination"] for row in authentic_tasks
    )
    body = {
        "schema_version": "subgoal-option-contrast-report-v15",
        "status": (
            "OUTCOME_BLIND_SUBGOAL_CONTRAST_GATE_PASSED"
            if passed else "OUTCOME_BLIND_SUBGOAL_CONTRAST_GATE_FAILED_STOP"
        ),
        "claim_boundary": (
            "CONTROLLED_MULTISOURCE_OPTION_CONTROLLER_FEASIBILITY_ONLY; "
            "REWARD_DISCARDED_AND_OFFICIAL_SUCCESS_NOT READ_OR_SERIALIZED; "
            "SELECTED TASKS NOW CONSUMED DEVELOPMENT; NOT REAL_GAME TRANSFER; "
            "CONFIRMATION_AND EXISTING_VALID_UNSEEN_UNREAD"
        ),
        "pool": {
            "path": str(args.pool.resolve()),
            "file_sha256": _sha256(args.pool),
            "pool_sha256": pool["pool_sha256"],
        },
        "counts": {
            "tasks": len(tasks),
            "tasks_with_authentic_action_contrast": len(authentic_tasks),
            "tasks_with_authentic_phase_action_contrast": len(
                source_specific_tasks
            ),
            "tasks_with_second_cycle_authentic_contrast": len(
                second_cycle_tasks
            ),
            "authentic_action_contrasts": sum(
                row["authentic_action_contrast_count"] for row in tasks
            ),
            "authentic_phase_action_contrasts": sum(
                row["authentic_phase_action_contrast_count"] for row in tasks
            ),
            "second_cycle_authentic_contrasts": sum(
                row["second_cycle_authentic_contrast_count"] for row in tasks
            ),
            "authentic_contrast_tasks_by_destination": dict(
                sorted(destination_counts.items())
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
