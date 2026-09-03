#!/usr/bin/env python3
"""Run matched outcome forks after the V18 outcome-blind shadow gate passes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
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


CONDITIONS = (
    "target_neural_baseline",
    "authentic_intervention_effect",
    "source_effect_permutation",
    "target_effect_permutation",
)
COMPARATORS = (
    "target_neural_baseline",
    "source_effect_permutation",
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


def _exact_one_sided_sign_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    return min(1.0, sum(
        math.comb(discordant, count) for count in range(wins, discordant + 1)
    ) / (2**discordant))


def _paired_comparison(
    authentic: Sequence[Mapping[str, Any]],
    comparator: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
    alpha: float,
) -> dict[str, Any]:
    authentic_by_task = {
        str(row["task_id"]): int(bool(row["official_success"]))
        for row in authentic
    }
    comparator_by_task = {
        str(row["task_id"]): int(bool(row["official_success"]))
        for row in comparator
    }
    if authentic_by_task.keys() != comparator_by_task.keys():
        raise RuntimeError("matched outcome conditions have different task identities")
    differences = np.asarray([
        authentic_by_task[task] - comparator_by_task[task]
        for task in sorted(authentic_by_task)
    ], dtype=np.float64)
    wins = int(np.sum(differences > 0))
    losses = int(np.sum(differences < 0))
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.asarray([
        np.mean(rng.choice(differences, size=len(differences), replace=True))
        for _ in range(bootstrap_samples)
    ])
    return {
        "authentic_only_successes": wins,
        "comparator_only_successes": losses,
        "paired_ties": int(np.sum(differences == 0)),
        "success_rate_difference": float(np.mean(differences)),
        "paired_task_bootstrap_lower_bound": float(np.quantile(bootstrap, alpha)),
        "one_sided_exact_sign_p": _exact_one_sided_sign_p(wins, losses),
    }


def _summaries(episodes: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return {
        condition: {
            "tasks": len(rows),
            "successes": sum(bool(row["official_success"]) for row in rows),
            "success_rate": float(np.mean([
                bool(row["official_success"]) for row in rows
            ])),
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_source_interventions": float(np.mean([
                row["source_interventions"] for row in rows
            ])),
        }
        for condition, rows in episodes.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V19 fork report: {args.output}")
    plan = _read(args.plan)
    if stable_hash({
        key: value for key, value in plan.items() if key != "plan_sha256"
    }) != plan.get("plan_sha256"):
        raise SystemExit("V19 fork plan hash mismatch")
    if plan.get("status") != "FROZEN_BEFORE_V19_MATCHED_OUTCOME_FORKS":
        raise SystemExit("V19 plan has unexpected authority")
    if tuple(plan["conditions"]) != CONDITIONS:
        raise SystemExit("V19 condition contract changed after freezing")
    for receipt in plan["implementation"].values():
        _validate_receipt(receipt)

    shadow_path = _validate_receipt(plan["shadow_gate"])
    shadow = _read(shadow_path)
    if shadow.get("report_sha256") != plan["shadow_gate"]["report_sha256"]:
        raise SystemExit("V19 references a different shadow stable hash")
    if shadow.get("status") != (
        "OUTCOME_BLIND_SHADOW_GATE_PASSED_FORK_FREEZER_AUTHORIZED"
    ):
        raise SystemExit("outcome-blind shadow did not authorize fork freezing")
    if shadow.get("reward_serialized") or shadow.get("official_success_serialized"):
        raise SystemExit("shadow report consumed outcomes before fork freezing")

    prior_plan_path = _validate_receipt(plan["consumed_target_pool"])
    prior_plan = _read(prior_plan_path)
    if prior_plan.get("plan_sha256") != plan["consumed_target_pool"]["plan_sha256"]:
        raise SystemExit("V19 references a different consumed target pool")
    pool_path = _validate_receipt(prior_plan["broad_pool"])
    pool = _read(pool_path)
    source_path = _validate_receipt(plan["source_controller"])
    source = _read(source_path)
    if source.get("candidate_sha256") != plan["source_controller"][
        "candidate_sha256"
    ]:
        raise SystemExit("source controller stable hash mismatch")
    target_path = _validate_receipt(plan["target_grounder"])
    target = _read(target_path)
    if target.get("artifact_sha256") != plan["target_grounder"][
        "artifact_sha256"
    ]:
        raise SystemExit("target grounder stable hash mismatch")
    if target.get("required_option_or_workflow_features_used") is not False:
        raise SystemExit("target grounder crossed the oracle-free boundary")
    models = {
        "authentic_intervention_effect": deserialize_pairwise_ensemble(
            source["models"]["authentic_intervention_effect"]
        ),
        "source_effect_permutation": deserialize_pairwise_ensemble(
            source["models"]["within_state_effect_permutation"]
        ),
    }
    quantiles = source["conformal"]["overprediction_error_quantiles"]
    task_ids = tuple(map(str, prior_plan["task_ids"]))
    max_steps = int(prior_plan["max_steps"])
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for condition in CONDITIONS:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(args.alfworld_config.resolve()),
            data_path=str(args.alfworld_data.resolve()),
            split="train",
            seed=int(pool["seed"]),
            game_ids=task_ids,
            max_steps=max_steps,
        )
        seen: set[str] = set()
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
                    raise RuntimeError("V19 fork reset identity mismatch")
                task_id = matches[0]
                if task_id in seen:
                    raise RuntimeError("V19 fork repeated a task identity")
                seen.add(task_id)
                goal = str(observation.state.get("task_goal", ""))
                history: list[str] = []
                records = []
                for step in range(max_steps):
                    grounded = ground_target_options(
                        goal=goal,
                        observation=str(observation.state.get("observation", "")),
                        native_actions=tuple(map(str, observation.native_actions)),
                        step=step,
                        max_steps=max_steps,
                        action_history=history,
                        target_grounder=target,
                    )
                    fallback_option = str(grounded["fallback_option"])
                    if condition == "target_neural_baseline":
                        selected = str(grounded["fallback_action"])
                        selected_option = fallback_option
                        source_admitted = False
                        lower_bound = None
                    else:
                        target_grounding = (
                            within_state_target_effect_permutation(grounded)
                            if condition == "target_effect_permutation" else grounded
                        )
                        model_name = (
                            "source_effect_permutation"
                            if condition == "source_effect_permutation"
                            else "authentic_intervention_effect"
                        )
                        quantile_name = (
                            "within_state_effect_permutation"
                            if condition == "source_effect_permutation"
                            else "authentic_intervention_effect"
                        )
                        decision = source_shadow_decision(
                            target_grounding,
                            model=models[model_name],
                            conformal_error=float(quantiles[quantile_name]),
                        )
                        selected = str(decision["action"])
                        selected_option = str(decision["option"])
                        source_admitted = bool(decision["source_admitted"])
                        lower_bound = float(
                            decision["comparison"]["conformal_lower_bound"]
                        )
                    after, reward = environment.step(selected)
                    record_body = {
                        "step": step,
                        "action": selected,
                        "selected_option": selected_option,
                        "fallback_option": fallback_option,
                        "source_admitted": source_admitted,
                        "conformal_lower_bound": lower_bound,
                        "reward": float(reward),
                        "official_success_after": bool(after.official_success),
                    }
                    records.append(record_body | {
                        "step_receipt_sha256": stable_hash(record_body)
                    })
                    history.append(selected)
                    observation = after
                    if after.terminal or after.official_success:
                        break
                episode_body = {
                    "condition": condition,
                    "task_index": task_index,
                    "task_id": task_id,
                    "official_success": bool(
                        records and records[-1]["official_success_after"]
                    ),
                    "return": sum(float(row["reward"]) for row in records),
                    "steps": len(records),
                    "source_interventions": sum(
                        row["selected_option"] != row["fallback_option"]
                        for row in records
                    ),
                    "records": records,
                }
                episodes[condition].append(episode_body | {
                    "episode_receipt_sha256": stable_hash(episode_body)
                })
                print(json.dumps({
                    "condition": condition,
                    "task_index": task_index,
                    "task_count": len(task_ids),
                    "task_id": task_id,
                    "official_success": episode_body["official_success"],
                    "steps": episode_body["steps"],
                    "source_interventions": episode_body["source_interventions"],
                }), flush=True)
        finally:
            environment.close()
        if seen != set(task_ids):
            raise RuntimeError(f"V19 {condition} did not run every matched task")

    summaries = _summaries(episodes)
    bootstrap = plan["bootstrap"]
    paired = {
        condition: _paired_comparison(
            episodes["authentic_intervention_effect"], episodes[condition],
            bootstrap_seed=int(bootstrap["seed"]) + index,
            bootstrap_samples=int(bootstrap["samples"]),
            alpha=float(bootstrap["lower_tail_alpha"]),
        )
        for index, condition in enumerate(COMPARATORS)
    }
    requirements = plan["success_gate"]
    gates = {
        "authentic_success_rate_strictly_exceeds_every_comparator": all(
            paired[condition]["success_rate_difference"] > 0.0
            for condition in COMPARATORS
        ),
        "paired_task_bootstrap_lower_bound_exceeds_threshold_for_every_comparator": all(
            paired[condition]["paired_task_bootstrap_lower_bound"]
            > float(requirements["minimum_paired_bootstrap_lower_bound"])
            for condition in COMPARATORS
        ),
        "one_sided_exact_sign_p_below_threshold_for_every_comparator": all(
            paired[condition]["one_sided_exact_sign_p"]
            < float(requirements["maximum_one_sided_exact_sign_p"])
            for condition in COMPARATORS
        ),
        "minimum_authentic_successes": (
            summaries["authentic_intervention_effect"]["successes"]
            >= int(requirements["minimum_authentic_successes"])
        ),
        "zero_identity_or_receipt_failures": True,
        "confirmation_and_valid_unseen_unread": True,
    }
    passed = all(gates.values())
    body = {
        "schema_version": "intervention-grounded-outcome-fork-report-v19",
        "status": (
            "CONSUMED_DEVELOPMENT_TRANSFER_GATE_PASSED"
            if passed else "CONSUMED_DEVELOPMENT_TRANSFER_GATE_FAILED_STOP"
        ),
        "claim_boundary": plan["claim_boundary"],
        "plan": {
            "path": str(args.plan.resolve()),
            "file_sha256": _sha256(args.plan),
            "plan_sha256": plan["plan_sha256"],
        },
        "summaries": summaries,
        "paired_official_success": paired,
        "gates": gates,
        "episodes": dict(episodes),
        "target_outcomes_read_after_plan_freeze": True,
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
        "summaries": summaries,
        "paired_official_success": paired,
        "gates": gates,
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
