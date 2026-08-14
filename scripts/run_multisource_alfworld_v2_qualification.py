#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.alfworld_hierarchical_grounder import (  # noqa: E402
    parse_goal,
    score_actions,
    workflow_status,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.hierarchical_skill_transfer import (  # noqa: E402
    deserialize_ensemble,
    option_features,
)


CONDITIONS = (
    "target_only",
    "authentic_source_plus_target",
    "shuffled_source_plus_target",
    "source_marginal_plus_target",
    "phase_permuted_source_plus_target",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_score(
    action: str,
    row: Mapping[str, float | str],
    history: Sequence[str],
) -> float:
    applicability = float(row["applicability"])
    completion = float(row["completion"])
    option = str(row["option"])
    binding_factor = (
        1.0 if option == "SEARCH" else 0.25 + 0.75 * float(row["binding"])
    )
    repeat_discount = 1.0 + history.count(action)
    return applicability * (0.20 + 0.80 * completion) * binding_factor / repeat_discount


def _neural_policy_score(
    action: str,
    row: Mapping[str, float | str],
    history: Sequence[str],
) -> float:
    probability = float(row.get("policy", row["applicability"]))
    return probability / (1.0 + history.count(action))


def _symbolic_features(
    *,
    action: str,
    row: Mapping[str, float | str],
    goal: str,
    step: int,
    max_steps: int,
    history: Sequence[str],
) -> tuple[float, ...]:
    status = workflow_status(goal, history)
    remaining_steps = max(1, max_steps - step)
    repeat_fraction = min(history.count(action), 8) / 8.0
    option = str(row["option"])
    binding = 1.0 if option == "SEARCH" else float(row["binding"])
    return option_features(
        option=option,
        required_option=str(row["required_option"]),
        precondition_satisfied=float(row["applicability"]),
        completion_probability=float(row["completion"]),
        goal_binding_probability=binding,
        remaining_budget_fraction=remaining_steps / max_steps,
        workflow_progress_fraction=status.progress_fraction,
        action_repeat_fraction=repeat_fraction,
        noop_probability=1.0 - float(row["completion"]),
        stage_urgency=min(1.0, (1.0 - status.progress_fraction) * 8 / remaining_steps),
        failure_cost=0.02 + 0.16 * repeat_fraction,
    )


def _choose_action(
    *,
    grounded: Mapping[str, Mapping[str, float | str]],
    goal: str,
    step: int,
    max_steps: int,
    history: Sequence[str],
    source_model,
    uncertainty_scale: float,
    decision_margin: float,
    controller: str = "ACTION_LEVEL_V2",
) -> dict[str, Any]:
    candidates = list(grounded)
    if not candidates:
        raise ValueError("no grounded ALFWorld candidates")
    hierarchical_target_scores = {
        action: _target_score(action, grounded[action], history) for action in candidates
    }
    target_scores = (
        {
            action: _neural_policy_score(action, grounded[action], history)
            for action in candidates
        }
        if controller == "SOURCE_OPTION_ONLY_V3_DIAGNOSTIC"
        else hierarchical_target_scores
    )
    fallback = max(candidates, key=lambda action: (target_scores[action], action))
    if source_model is None:
        return {
            "action": fallback,
            "fallback_action": fallback,
            "source_admitted": False,
            "changed_action": False,
            "changed_option": False,
            "diagnostic": "TARGET_ONLY",
            "target_score": target_scores[fallback],
        }
    if controller == "SOURCE_OPTION_ONLY_V3_DIAGNOSTIC":
        representatives = []
        for option in sorted({str(row["option"]) for row in grounded.values()}):
            option_actions = [
                action for action in candidates if grounded[action]["option"] == option
            ]
            representatives.append(max(
                option_actions,
                key=lambda action: (hierarchical_target_scores[action], action),
            ))
        candidates = representatives
        if len(candidates) < 2:
            return {
                "action": fallback,
                "fallback_action": fallback,
                "source_admitted": False,
                "changed_action": False,
                "changed_option": False,
                "diagnostic": "OPTION_COMPARISON_UNAVAILABLE",
                "target_score": target_scores[fallback],
            }
    features = [
        _symbolic_features(
            action=action,
            row=grounded[action],
            goal=goal,
            step=step,
            max_steps=max_steps,
            history=history,
        )
        for action in candidates
    ]
    means, deviations = source_model.predict(features)
    robust_values = means - float(uncertainty_scale) * deviations
    best_index = max(range(len(candidates)), key=lambda index: robust_values[index])
    best = candidates[best_index]
    if controller == "SOURCE_OPTION_ONLY_V3_DIAGNOSTIC":
        alternatives = [index for index in range(len(candidates)) if index != best_index]
        comparison_index = max(alternatives, key=lambda index: robust_values[index])
    else:
        comparison_index = candidates.index(fallback)
    raw_gap = float(means[best_index] - means[comparison_index])
    comparison_uncertainty = float(uncertainty_scale) * math.sqrt(
        float(deviations[best_index] ** 2 + deviations[comparison_index] ** 2)
    )
    admitted = (
        raw_gap - comparison_uncertainty > decision_margin
        and (
            controller == "SOURCE_OPTION_ONLY_V3_DIAGNOSTIC"
            or best != fallback
        )
    )
    selected = best if admitted else fallback
    return {
        "action": selected,
        "fallback_action": fallback,
        "source_admitted": admitted,
        "changed_action": selected != fallback,
        "changed_option": (
            grounded[selected]["option"] != grounded[fallback]["option"]
        ),
        "diagnostic": "SOURCE_ADMITTED" if admitted else "SOURCE_ABSTAINED_TO_TARGET",
        "target_score": target_scores[selected],
        "source_best_action": best,
        "source_value_gap": raw_gap,
        "comparison_uncertainty": comparison_uncertainty,
        "source_selected_mean": float(means[best_index]),
        "fallback_source_mean": float(means[comparison_index]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    final_heldout = config.get("status") == "FINAL_FROZEN_HELDOUT_EVALUATION"
    target = config["target"]
    artifact_path = (REPO / target["artifact"]).resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact["status"] != "QUALIFICATION_AUTHORIZED":
        raise SystemExit("frozen candidate artifact was not authorized")
    if final_heldout:
        evidence = config["development_evidence"]
        frozen_inputs = (
            (artifact_path, str(evidence["frozen_artifact_sha256"])),
            ((REPO / evidence["origin_config"]).resolve(), str(
                evidence["origin_config_sha256"]
            )),
            ((REPO / evidence["qualification_report"]).resolve(), str(
                evidence["qualification_report_sha256"]
            )),
            (Path(__file__).resolve(), str(evidence["runner_sha256"])),
        )
        mismatches = [
            str(path) for path, expected in frozen_inputs if _sha256(path) != expected
        ]
        if mismatches:
            raise SystemExit(f"frozen held-out inputs changed: {mismatches}")
    output = (REPO / target["qualification_report"]).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite qualification report: {output}")
    manifest_path = (REPO / target["qualification_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_name = str(target["qualification_manifest_split"])
    expected_split = "held_out" if final_heldout else "qualification"
    if split_name != expected_split:
        raise SystemExit(
            f"evaluation boundary violation: expected {expected_split}, got {split_name}"
        )
    task_ids = tuple(map(str,
        manifest["cells"]["alfworld_valid_unseen"]["splits"][split_name]
    ))
    data_root = (
        Path(target["alfworld_data"]) / "json_2.1.1" / "valid_unseen"
    ).resolve()
    source_models = {
        condition: deserialize_ensemble(artifact["source"]["models"][condition])
        for condition in CONDITIONS if condition != "target_only"
    }
    episodes: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for condition in CONDITIONS:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(target["alfworld_config"]),
            data_path=str(target["alfworld_data"]),
            split=str(target["qualification_split"]),
            seed=int(target["seed"]),
            game_ids=task_ids,
            max_steps=int(target["qualification_max_steps"]),
        )
        seen_ids = set()
        try:
            for task_index in range(len(task_ids)):
                observation = environment.reset()
                actual_task_id = (
                    Path(environment.resolved_game_file).resolve()
                    .relative_to(data_root).as_posix()
                )
                if actual_task_id not in task_ids or actual_task_id in seen_ids:
                    raise RuntimeError(
                        f"qualification game pairing violation: {actual_task_id}"
                    )
                seen_ids.add(actual_task_id)
                history: list[str] = []
                records = []
                for step in range(int(target["qualification_max_steps"])):
                    native = list(observation.native_actions)
                    goal = str(observation.state.get("task_goal", ""))
                    grounded = score_actions(
                        goal=goal,
                        observation=str(observation.state.get("observation", "")),
                        native_actions=native,
                        step=step,
                        action_history=history,
                        artifact=artifact["target_grounder"],
                    )
                    if not grounded:
                        raise RuntimeError("target-native grounder excluded every action")
                    decision = _choose_action(
                        grounded=grounded,
                        goal=goal,
                        step=step,
                        max_steps=int(target["qualification_max_steps"]),
                        history=history,
                        source_model=source_models.get(condition),
                        uncertainty_scale=float(config["policy"]["uncertainty_scale"]),
                        decision_margin=float(config["policy"]["decision_margin"]),
                        controller=str(config["policy"].get("controller", "ACTION_LEVEL_V2")),
                    )
                    selected = str(decision["action"])
                    required = str(grounded[selected]["required_option"])
                    selected_option = str(grounded[selected]["option"])
                    fallback = str(decision["fallback_action"])
                    before_text = str(observation.state.get("observation", ""))
                    after, reward = environment.step(selected)
                    ranked_target = sorted(
                        grounded,
                        key=lambda action: _target_score(action, grounded[action], history),
                        reverse=True,
                    )[:5]
                    spec = parse_goal(goal)
                    target_action_available = any(
                        spec.target_object in action.lower().split()
                        for action in grounded
                    )
                    records.append({
                        "step": step,
                        "goal": goal,
                        "required_option": required,
                        "action": selected,
                        "action_option": selected_option,
                        "fallback_action": fallback,
                        "fallback_option": str(grounded[fallback]["option"]),
                        "source_admitted": bool(decision["source_admitted"]),
                        "changed_action": bool(decision["changed_action"]),
                        "changed_option": bool(decision["changed_option"]),
                        "diagnostic": str(decision["diagnostic"]),
                        "target_action_available": target_action_available,
                        "selected_binding": float(grounded[selected]["binding"]),
                        "selected_completion": float(grounded[selected]["completion"]),
                        "selected_applicability": float(grounded[selected]["applicability"]),
                        "top_target_actions": ranked_target,
                        "before_observation": before_text,
                        "after_observation": str(after.state.get("observation", "")),
                        "reward": float(reward),
                        "official_success_after": bool(after.official_success),
                        "receipt_sha256": stable_hash({
                            "task_id": actual_task_id,
                            "condition": condition,
                            "step": step,
                            "before": dict(observation.state),
                            "native": observation.native_actions,
                            "grounded": grounded,
                            "decision": decision,
                            "after": dict(after.state),
                            "reward": reward,
                            "success": after.official_success,
                        }),
                    })
                    history.append(selected)
                    observation = after
                    if after.terminal or after.official_success:
                        break
                success = bool(records and records[-1]["official_success_after"])
                episodes[condition].append({
                    "task_index": task_index,
                    "task_id": actual_task_id,
                    "official_success": success,
                    "steps": len(records),
                    "source_admissions": sum(row["source_admitted"] for row in records),
                    "changed_actions": sum(row["changed_action"] for row in records),
                    "changed_options": sum(row["changed_option"] for row in records),
                    "required_options": dict(Counter(
                        row["required_option"] for row in records
                    )),
                    "selected_options": dict(Counter(
                        row["action_option"] for row in records
                    )),
                    "diagnostics": dict(Counter(row["diagnostic"] for row in records)),
                    "records": records,
                })
                print(json.dumps({
                    "condition": condition,
                    "task_index": task_index,
                    "task_id": actual_task_id,
                    "steps": len(records),
                    "success": success,
                }), flush=True)
        finally:
            environment.close()
        if seen_ids != set(task_ids):
            raise RuntimeError(f"condition {condition} did not execute exact qualification set")

    paired_orders = {
        condition: [row["task_id"] for row in rows] for condition, rows in episodes.items()
    }
    if len({tuple(order) for order in paired_orders.values()}) != 1:
        raise RuntimeError("qualification conditions did not execute tasks in paired order")
    summaries = {}
    for condition, rows in episodes.items():
        total_steps = sum(row["steps"] for row in rows)
        summaries[condition] = {
            "tasks": len(rows),
            "successes": sum(row["official_success"] for row in rows),
            "success_rate": sum(row["official_success"] for row in rows) / len(rows),
            "mean_steps": sum(row["steps"] for row in rows) / len(rows),
            "source_admission_rate": sum(row["source_admissions"] for row in rows)
            / total_steps,
            "changed_action_rate": sum(row["changed_actions"] for row in rows)
            / total_steps,
            "changed_option_rate": sum(row["changed_options"] for row in rows)
            / total_steps,
        }
    authentic = summaries["authentic_source_plus_target"]
    nontrivial = authentic["changed_option_rate"] >= float(
        config["policy"]["minimum_authentic_option_change_rate"]
    )
    superiority = all(
        authentic["successes"] > summary["successes"]
        for condition, summary in summaries.items()
        if condition != "authentic_source_plus_target"
    )
    efficiency = authentic["mean_steps"] < summaries["target_only"]["mean_steps"]
    efficiency_required = bool(
        config["policy"].get("require_mean_steps_below_target_only", False)
    )
    candidate_passed = (
        nontrivial and superiority and (efficiency or not efficiency_required)
    )
    report = {
        "schema_version": (
            "multisource-alfworld-final-heldout-v4"
            if final_heldout
            else (
                "multisource-alfworld-qualification-v3-diagnostic"
                if config.get("schema_version") == 3
                else "multisource-alfworld-qualification-v2"
            )
        ),
        "status": (
            ("FINAL_HELDOUT_PASSED" if candidate_passed else "FINAL_HELDOUT_FAILED")
            if final_heldout
            else (
                "QUALIFICATION_CANDIDATE_PASSED"
                if candidate_passed else "QUALIFICATION_CANDIDATE_FAILED"
            )
        ),
        "claim_boundary": config["claim_boundary"],
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256(artifact_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_split": split_name,
        "heldout_read": final_heldout,
        "conditions": list(CONDITIONS),
        "paired_task_order_verified": True,
        "paired_task_order": paired_orders["target_only"],
        "summaries": summaries,
        "nontriviality_gate": {
            "metric": "authentic changed hierarchical option rate",
            "observed": authentic["changed_option_rate"],
            "minimum": float(config["policy"]["minimum_authentic_option_change_rate"]),
            "passed": nontrivial,
        },
        "qualification_superiority_gate": {
            "metric": "authentic successes strictly greater than every control",
            "passed": superiority,
        },
        "efficiency_gate": {
            "metric": "authentic mean steps below target-only",
            "observed_authentic": authentic["mean_steps"],
            "observed_target_only": summaries["target_only"]["mean_steps"],
            "required": efficiency_required,
            "passed": efficiency,
        },
        "episodes": episodes,
        "cross_domain_transfer_supported": candidate_passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "summaries": summaries,
        "nontriviality_gate": report["nontriviality_gate"],
        "qualification_superiority_gate": report["qualification_superiority_gate"],
        "efficiency_gate": report["efficiency_gate"],
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0 if candidate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
