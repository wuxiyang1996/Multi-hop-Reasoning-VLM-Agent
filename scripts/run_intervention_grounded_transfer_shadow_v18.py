#!/usr/bin/env python3
"""Run V18 neural-symbolic option transfer as an outcome-blind shadow."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.intervention_grounded_target_controller import (  # noqa: E402
    ground_target_options,
    source_shadow_decision,
    within_state_target_effect_permutation,
)
from motif_transfer.pairwise_option_advantage import (  # noqa: E402
    deserialize_pairwise_ensemble,
)


MODEL_CONDITIONS = (
    "authentic_intervention_effect",
    "within_state_effect_permutation",
    "within_state_value_shuffle",
    "effect_blind",
)
CONTROL_CONDITIONS = (
    "within_state_effect_permutation",
    "within_state_value_shuffle",
    "effect_blind",
    "target_effect_permutation",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_receipt(receipt: Mapping[str, Any]) -> Path:
    path = Path(str(receipt["path"]))
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve()
    if _sha256(path) != str(receipt["file_sha256"]):
        raise ValueError(f"dependency hash mismatch: {path}")
    return path


def _relative_game_matches(actual: str, expected: str) -> bool:
    normalized = str(actual).replace("\\", "/")
    target = str(expected).replace("\\", "/").lstrip("/")
    return normalized == target or normalized.endswith("/" + target)


def _bootstrap_lower(
    tasks: Sequence[Mapping[str, Any]],
    *,
    authentic_field: str,
    control_field: str,
    seed: int,
    samples: int,
    alpha: float,
) -> float:
    differences = np.asarray([
        float(task[authentic_field]) - float(task[control_field])
        for task in tasks
    ])
    rng = np.random.default_rng(seed)
    estimates = np.asarray([
        np.mean(rng.choice(differences, size=len(differences), replace=True))
        for _ in range(samples)
    ])
    return float(np.quantile(estimates, alpha))


def _pair_type_task_counts(tasks: Sequence[Mapping[str, Any]]) -> Counter[str]:
    task_pairs: Counter[str] = Counter()
    for task in tasks:
        pairs = {
            f"{row['fallback_option']}->{row['authentic_option']}"
            for row in task["contrasts"] if row["source_specific"]
        }
        task_pairs.update(pairs)
    return task_pairs


def _decision_record(decision: Mapping[str, Any]) -> dict[str, Any]:
    comparison = decision["comparison"]
    return {
        "option": str(decision["option"]),
        "action": str(decision["action"]),
        "source_admitted": bool(decision["source_admitted"]),
        "predicted_advantage": float(comparison["predicted_advantage"]),
        "conformal_lower_bound": float(comparison["conformal_lower_bound"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V18 shadow report: {args.output}")
    plan = _read(args.plan)
    if stable_hash({
        key: value for key, value in plan.items() if key != "plan_sha256"
    }) != plan.get("plan_sha256"):
        raise SystemExit("V18 shadow plan hash mismatch")
    if plan.get("status") != "FROZEN_BEFORE_V18_OUTCOME_BLIND_SHADOW":
        raise SystemExit("V18 plan has unexpected authority")
    for receipt in plan["implementation"].values():
        _validate_receipt(receipt)

    prior_plan_path = _validate_receipt(plan["consumed_target_pool"])
    prior_plan = _read(prior_plan_path)
    if prior_plan.get("plan_sha256") != plan["consumed_target_pool"]["plan_sha256"]:
        raise SystemExit("V18 plan references a different consumed target pool")
    if prior_plan.get("existing_valid_unseen_heldout_read"):
        raise SystemExit("consumed target pool crossed valid_unseen boundary")
    pool_path = _validate_receipt(prior_plan["broad_pool"])
    pool = _read(pool_path)

    source_path = _validate_receipt(plan["source_controller"])
    source = _read(source_path)
    if source.get("candidate_sha256") != plan["source_controller"][
        "candidate_sha256"
    ]:
        raise SystemExit("source candidate stable hash mismatch")
    if source.get("status") != "SOURCE_GATE_PASSED" or not source.get(
        "target_authorized"
    ):
        raise SystemExit("V17 source gate did not authorize target shadow")
    target_path = _validate_receipt(plan["target_grounder"])
    target = _read(target_path)
    if target.get("artifact_sha256") != plan["target_grounder"][
        "artifact_sha256"
    ]:
        raise SystemExit("target grounder stable hash mismatch")
    if target.get("status") != "TARGET_GROUNDER_GATE_PASSED":
        raise SystemExit("oracle-free target grounder gate did not pass")
    if target.get("required_option_or_workflow_features_used") is not False:
        raise SystemExit("target grounder used a forbidden workflow oracle")
    if target.get("reward_success_completion_fields_consumed") is not False:
        raise SystemExit("target grounder consumed outcome fields")

    models = {
        condition: deserialize_pairwise_ensemble(source["models"][condition])
        for condition in MODEL_CONDITIONS
    }
    conformal = source["conformal"]["overprediction_error_quantiles"]
    task_ids = tuple(map(str, prior_plan["task_ids"]))
    max_steps = int(prior_plan["max_steps"])
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
                raise RuntimeError("V18 shadow reset identity mismatch")
            task_id = matches[0]
            if task_id in seen:
                raise RuntimeError("V18 shadow repeated a task identity")
            seen.add(task_id)
            goal = str(observation.state.get("task_goal", ""))
            history: list[str] = []
            contrasts = []
            change_counts = Counter({condition: 0 for condition in (
                "authentic_intervention_effect", *CONTROL_CONDITIONS,
            )})
            for step in range(max_steps):
                observation_text = str(observation.state.get("observation", ""))
                native_actions = tuple(map(str, observation.native_actions))
                grounded = ground_target_options(
                    goal=goal,
                    observation=observation_text,
                    native_actions=native_actions,
                    step=step,
                    max_steps=max_steps,
                    action_history=history,
                    target_grounder=target,
                )
                decisions = {
                    condition: source_shadow_decision(
                        grounded,
                        model=models[condition],
                        conformal_error=float(conformal[condition]),
                    )
                    for condition in MODEL_CONDITIONS
                }
                decisions["target_effect_permutation"] = source_shadow_decision(
                    within_state_target_effect_permutation(grounded),
                    model=models["authentic_intervention_effect"],
                    conformal_error=float(
                        conformal["authentic_intervention_effect"]
                    ),
                )
                fallback_option = str(grounded["fallback_option"])
                for condition, decision in decisions.items():
                    change_counts[condition] += int(
                        str(decision["option"]) != fallback_option
                    )
                authentic = decisions["authentic_intervention_effect"]
                if str(authentic["option"]) != fallback_option:
                    source_specific = all(
                        str(authentic["option"])
                        != str(decisions[condition]["option"])
                        for condition in CONTROL_CONDITIONS
                    )
                    state_body = {
                        "task_id": task_id,
                        "step": step,
                        "goal": goal,
                        "observation": observation_text,
                        "native_actions": list(native_actions),
                        "prefix_actions": list(history),
                    }
                    row_body = {
                        "task_id": task_id,
                        "step": step,
                        "before_state_sha256": stable_hash(state_body),
                        "fallback_option": fallback_option,
                        "fallback_action": str(grounded["fallback_action"]),
                        "authentic_option": str(authentic["option"]),
                        "authentic_action": str(authentic["action"]),
                        "source_specific": source_specific,
                        "neural_effect_by_option": {
                            option: float(features[11])
                            for option, features in grounded[
                                "option_features"
                            ].items()
                        },
                        "decisions": {
                            condition: _decision_record(decision)
                            for condition, decision in decisions.items()
                        },
                    }
                    contrasts.append(row_body | {
                        "contrast_sha256": stable_hash(row_body)
                    })
                selected = str(grounded["fallback_action"])
                observation, _discarded_reward = environment.step(selected)
                history.append(selected)
                if observation.terminal:
                    break
            steps = len(history)
            task_body = {
                "task_index": task_index,
                "task_id": task_id,
                "steps_executed": steps,
                "authentic_option_contrast_count": len(contrasts),
                "source_specific_option_contrast_count": sum(
                    row["source_specific"] for row in contrasts
                ),
                "option_change_rate": {
                    condition: change_counts[condition] / max(1, steps)
                    for condition in change_counts
                },
                "contrasts": contrasts,
            }
            for condition, rate in task_body["option_change_rate"].items():
                task_body[f"{condition}_change_rate"] = rate
            tasks.append(task_body | {"task_receipt_sha256": stable_hash(task_body)})
            print(json.dumps({
                "task_index": task_index,
                "task_count": len(task_ids),
                "task_id": task_id,
                "authentic_option_contrasts": len(contrasts),
                "source_specific": task_body[
                    "source_specific_option_contrast_count"
                ],
                "outcomes_recorded": False,
            }), flush=True)
    finally:
        environment.close()
    if seen != set(task_ids):
        raise RuntimeError("V18 shadow did not replay every consumed task")

    requirements = plan["contrast_gate"]
    authentic_tasks = [
        row for row in tasks if row["authentic_option_contrast_count"] > 0
    ]
    source_specific_tasks = [
        row for row in tasks
        if row["source_specific_option_contrast_count"] > 0
    ]
    pair_types = _pair_type_task_counts(tasks)
    bootstrap = plan["bootstrap"]
    control_lower_bounds = {
        condition: _bootstrap_lower(
            tasks,
            authentic_field="authentic_intervention_effect_change_rate",
            control_field=f"{condition}_change_rate",
            seed=int(bootstrap["seed"]) + index,
            samples=int(bootstrap["samples"]),
            alpha=float(bootstrap["lower_tail_alpha"]),
        )
        for index, condition in enumerate(CONTROL_CONDITIONS)
    }
    gates = {
        "minimum_tasks_with_authentic_option_contrast": (
            len(authentic_tasks) >= int(requirements[
                "minimum_tasks_with_authentic_option_contrast"
            ])
        ),
        "minimum_tasks_with_source_specific_option_contrast": (
            len(source_specific_tasks) >= int(requirements[
                "minimum_tasks_with_source_specific_option_contrast"
            ])
        ),
        "minimum_option_pair_types_each_spanning_minimum_tasks": (
            sum(
                count >= int(requirements["minimum_tasks_per_option_pair_type"])
                for count in pair_types.values()
            ) >= int(requirements["minimum_option_pair_types"])
        ),
        "authentic_change_rate_exceeds_every_control_cluster_bootstrap": all(
            value > float(requirements[
                "minimum_authentic_minus_control_bootstrap_lower_bound"
            ]) for value in control_lower_bounds.values()
        ),
        "zero_forbidden_target_semantic_inputs": True,
        "zero_outcomes_recorded": True,
        "zero_identity_or_receipt_failures": True,
    }
    passed = all(gates.values())
    body = {
        "schema_version": "intervention-grounded-transfer-shadow-report-v18",
        "status": (
            "OUTCOME_BLIND_SHADOW_GATE_PASSED_FORK_FREEZER_AUTHORIZED"
            if passed else "OUTCOME_BLIND_SHADOW_GATE_FAILED_STOP"
        ),
        "claim_boundary": plan["claim_boundary"],
        "plan": {
            "path": str(args.plan.resolve()),
            "file_sha256": _sha256(args.plan),
            "plan_sha256": plan["plan_sha256"],
        },
        "counts": {
            "tasks": len(tasks),
            "tasks_with_authentic_option_contrast": len(authentic_tasks),
            "tasks_with_source_specific_option_contrast": len(source_specific_tasks),
            "authentic_option_contrasts": sum(
                row["authentic_option_contrast_count"] for row in tasks
            ),
            "source_specific_option_contrasts": sum(
                row["source_specific_option_contrast_count"] for row in tasks
            ),
        },
        "source_specific_option_pair_task_counts": dict(pair_types),
        "authentic_minus_control_change_rate_cluster_bootstrap_lower_bounds": (
            control_lower_bounds
        ),
        "gates": gates,
        "tasks": tasks,
        "main_path_policy": "ORACLE_FREE_TARGET_NEURAL_POLICY",
        "source_controller_executed": False,
        "reward_serialized": False,
        "official_success_serialized": False,
        "terminal_used_for_environment_lifecycle_only": True,
        "confirmation_or_valid_unseen_read_or_run": False,
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
        "pair_type_task_counts": report[
            "source_specific_option_pair_task_counts"
        ],
        "control_lower_bounds": control_lower_bounds,
        "gates": gates,
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
