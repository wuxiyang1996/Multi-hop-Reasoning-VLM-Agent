#!/usr/bin/env python3
"""Run paired ALFWorld replication with the unchanged Phase-3 source IR."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.alfworld_multiplicity_grounder import workflow_status  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_alfworld_transfer import (  # noqa: E402
    CONDITIONS,
    Phase3ALFWorldSelector,
    effect_observation_horizon,
)
from motif_transfer.phase3_alfworld_typed_grounder import (  # noqa: E402
    score_actions,
    validate_artifact,
)


def _read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_config(config: Mapping[str, Any]) -> None:
    body = dict(config)
    claimed = str(body.pop("config_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("Phase-3 ALFWorld frozen config hash mismatch")
    if tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError("Phase-3 ALFWorld condition matrix changed")
    for relative, expected in config["integrity"]["file_sha256"].items():
        path = (REPO / str(relative)).resolve()
        if _sha256(path) != str(expected):
            raise ValueError(f"frozen dependency changed: {path}")


def _paired(success: Mapping[str, Mapping[str, bool]], comparator: str) -> dict[str, Any]:
    authentic = success["source_induced"]
    other = success[comparator]
    ids = tuple(authentic)
    wins = sum(authentic[key] and not other[key] for key in ids)
    losses = sum(other[key] and not authentic[key] for key in ids)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(ids) - wins - losses,
        "negative_transfer_rate": losses / len(ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite ALFWorld report: {args.output}")
    config_path = args.config.resolve()
    config = _read(config_path)
    _validate_config(config)
    artifact_path = (REPO / config["grounder"]["path"]).resolve()
    artifact = _read(artifact_path)
    validate_artifact(artifact)
    if artifact["artifact_sha256"] != config["grounder"]["artifact_sha256"]:
        raise SystemExit("frozen target grounder content hash changed")
    source_artifacts = []
    for row in config["source_programs"]:
        path = (REPO / row["path"]).resolve()
        value = _read(path)
        if value["artifact_sha256"] != row["artifact_sha256"]:
            raise SystemExit(f"frozen source program changed: {path}")
        source_artifacts.append(value)
    task_ids = tuple(map(str, config["target"]["task_ids"]))
    split = str(config["target"]["split"])
    root_name = "valid_seen" if split == "eval_in_distribution" else "valid_unseen"
    data_root = (
        Path(config["target"]["alfworld_data"]) / "json_2.1.1" / root_name
    ).resolve()
    episodes: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(Path(config["target"]["alfworld_config"]).resolve()),
            data_path=str(Path(config["target"]["alfworld_data"]).resolve()),
            split=split,
            seed=int(config["target"]["seed"]),
            game_ids=task_ids,
            max_steps=int(config["target"]["max_steps"]),
            expose_expert_plan=condition == "target_native_ceiling",
        )
        seen: set[str] = set()
        try:
            for task_index in range(len(task_ids)):
                observation = environment.reset()
                actual_id = (
                    Path(environment.resolved_game_file).resolve()
                    .relative_to(data_root).as_posix()
                )
                if actual_id not in task_ids or actual_id in seen:
                    raise RuntimeError(f"paired task identity violation: {actual_id}")
                seen.add(actual_id)
                selector = Phase3ALFWorldSelector(
                    condition=condition,
                    source_artifacts=source_artifacts,
                    minimum_source_policy_ratio=float(
                        artifact.get("minimum_source_policy_support_ratio", 0.0)
                    ),
                    binding_level=str(
                        artifact.get("binding_level", "target_native_action")
                    ),
                )
                fallback_selector = Phase3ALFWorldSelector(
                    condition="neural_only",
                    source_artifacts=(),
                    binding_level=str(
                        artifact.get("binding_level", "target_native_action")
                    ),
                )
                macro_protocol = (
                    artifact.get("effect_observation_protocol")
                    == "TARGET_NATIVE_MACRO_ROLLOUT_V1"
                    and condition in {"source_induced", "source_permuted"}
                )
                pending_macro: dict[str, Any] | None = None
                history: list[str] = []
                records = []
                for step in range(int(config["target"]["max_steps"])):
                    goal = str(observation.state.get("task_goal", ""))
                    grounded = score_actions(
                        goal=goal,
                        observation=str(observation.state.get("observation", "")),
                        native_actions=observation.native_actions,
                        step=step,
                        action_history=history,
                        artifact=artifact,
                    )
                    if pending_macro is not None:
                        continuation = fallback_selector.select(
                            grounded=grounded, history=history,
                        )
                        decision = dict(continuation) | {
                            "reason": "TARGET_NATIVE_MACRO_CONTINUATION",
                            "macro_selected_effect_type": pending_macro["effect_type"],
                            "macro_selected_program_sha256": pending_macro[
                                "program_sha256"
                            ],
                            "macro_steps_remaining_before_action": pending_macro[
                                "remaining"
                            ],
                        }
                    elif condition == "target_native_ceiling":
                        expert = environment.expert_action()
                        fallback_decision = fallback_selector.select(
                            grounded=grounded, history=history,
                        )
                        decision = {
                            "action": expert,
                            "fallback_action": fallback_decision["action"],
                            "source_admitted": False,
                            "reason": "TARGET_NATIVE_OFFICIAL_EXPERT_CEILING",
                            "expert_action_inside_typed_grounding": expert in grounded,
                        }
                    else:
                        decision = selector.select(
                            grounded=grounded,
                            history=history,
                        )
                    selected = str(decision["action"])
                    before_progress = workflow_status(goal, history).progress_fraction
                    before = dict(observation.state)
                    after, reward = environment.step(selected)
                    after_progress = workflow_status(
                        goal, (*history, selected),
                    ).progress_fraction
                    progress_delta = after_progress - before_progress
                    transition_changed = bool(
                        str(after.state.get("observation", ""))
                        != str(before.get("observation", ""))
                        or tuple(after.native_actions)
                        != tuple(observation.native_actions)
                    )
                    macro_observation = None
                    if pending_macro is not None:
                        pending_macro["remaining"] -= 1
                        pending_macro["observed_steps"] += 1
                        pending_macro["changed_steps"] += int(transition_changed)
                        if pending_macro["remaining"] == 0:
                            macro_progress = (
                                after_progress - pending_macro["start_progress"]
                            )
                            persistence = (
                                pending_macro["changed_steps"]
                                / pending_macro["observed_steps"]
                            )
                            selector.observe_transition(
                                progress_delta=macro_progress,
                                transition_changed=bool(
                                    pending_macro["changed_steps"]
                                ),
                                persistence_fraction=persistence,
                            )
                            macro_observation = {
                                "effect_type": pending_macro["effect_type"],
                                "observed_steps": pending_macro["observed_steps"],
                                "progress_delta": macro_progress,
                                "persistence_fraction": persistence,
                            }
                            pending_macro = None
                    elif macro_protocol and decision.get("source_admitted"):
                        horizon = effect_observation_horizon(
                            str(decision["selected_effect_type"])
                        )
                        if horizon == 1:
                            selector.observe_transition(
                                progress_delta=progress_delta,
                                transition_changed=transition_changed,
                                persistence_fraction=float(transition_changed),
                            )
                        else:
                            pending_macro = {
                                "effect_type": str(decision["selected_effect_type"]),
                                "program_sha256": str(
                                    decision["selected_program_sha256"]
                                ),
                                "remaining": horizon - 1,
                                "observed_steps": 1,
                                "changed_steps": int(transition_changed),
                                "start_progress": before_progress,
                            }
                    elif not macro_protocol:
                        selector.observe_transition(
                            progress_delta=progress_delta,
                            transition_changed=transition_changed,
                            persistence_fraction=float(transition_changed),
                        )
                    body = {
                        "task_id": actual_id,
                        "condition": condition,
                        "step": step,
                        "before": before,
                        "native_action_count": len(observation.native_actions),
                        "selected_action": selected,
                        "selected_grounding": grounded.get(selected, {
                            "target_native_ceiling_unscored_action": True,
                            "action_sha256": stable_hash({
                                "target_native_action": selected
                            }),
                        }),
                        "decision": decision,
                        "target_progress_delta": progress_delta,
                        "target_transition_changed": transition_changed,
                        "macro_observation": macro_observation,
                        "after": dict(after.state),
                        "reward_evaluator_only": float(reward),
                        "official_success_evaluator_only": bool(after.official_success),
                        "selection_read_official_success": False,
                    }
                    records.append(body | {"receipt_sha256": stable_hash(body)})
                    history.append(selected)
                    observation = after
                    if after.terminal or after.official_success:
                        break
                success = bool(records and records[-1]["official_success_evaluator_only"])
                episodes[condition].append({
                    "task_index": task_index,
                    "task_id": actual_id,
                    "official_success": success,
                    "steps": len(records),
                    "source_admissions": sum(
                        bool(row["decision"]["source_admitted"]) for row in records
                    ),
                    "changed_actions": sum(
                        row["selected_action"] != row["decision"]["fallback_action"]
                        for row in records
                    ),
                    "selected_program_counts": dict(selector.selected_programs),
                    "selected_effect_counts": dict(selector.selected_effects),
                    "portfolio_abstentions": selector.portfolio_abstentions,
                    "runtime_abstentions": selector.runtime_abstentions,
                    "records": records,
                })
                print(json.dumps({
                    "condition": condition,
                    "task": f"{task_index + 1}/{len(task_ids)}",
                    "task_id": actual_id,
                    "success": success,
                    "steps": len(records),
                }), flush=True)
                partial = {
                    "schema_version": "phase3-alfworld-partial-v1",
                    "config_sha256": config["config_sha256"],
                    "completed_conditions": {
                        name: len(rows) for name, rows in episodes.items()
                    },
                    "episodes": episodes,
                }
                partial_path = args.output.with_suffix(".partial.json")
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                partial_path.write_text(
                    json.dumps(partial, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        finally:
            environment.close()
        if seen != set(task_ids):
            raise RuntimeError(f"condition {condition} missed frozen tasks")
    orders = {
        condition: tuple(row["task_id"] for row in values)
        for condition, values in episodes.items()
    }
    if len(set(orders.values())) != 1:
        raise RuntimeError("conditions did not execute the same task order")
    summaries = {}
    for condition, rows in episodes.items():
        steps = sum(row["steps"] for row in rows)
        summaries[condition] = {
            "tasks": len(rows),
            "successes": sum(row["official_success"] for row in rows),
            "success_rate": sum(row["official_success"] for row in rows) / len(rows),
            "mean_steps": steps / len(rows),
            "source_admission_rate": (
                sum(row["source_admissions"] for row in rows) / steps
                if steps else 0.0
            ),
            "changed_action_rate": (
                sum(row["changed_actions"] for row in rows) / steps
                if steps else 0.0
            ),
            "changed_actions": sum(row["changed_actions"] for row in rows),
        }
    success = {
        condition: {
            row["task_id"]: bool(row["official_success"]) for row in values
        }
        for condition, values in episodes.items()
    }
    comparisons = {
        name: _paired(success, name)
        for name in ("neural_only", "source_permuted", "generic_scaffold")
    }
    source_rows = episodes["source_induced"]
    selected_effects = Counter()
    selected_programs = Counter()
    for row in source_rows:
        selected_effects.update(row["selected_effect_counts"])
        selected_programs.update(row["selected_program_counts"])
    task_level_contrasts = 0
    for authentic, permuted in zip(
        episodes["source_induced"], episodes["source_permuted"],
    ):
        task_level_contrasts += any(
            left["selected_action"] != right["selected_action"]
            for left, right in zip(authentic["records"], permuted["records"])
        )
    thresholds = config["gates"]
    source_free_ceiling_successes = sum(
        success["target_native_ceiling"][task_id]
        or success["neural_only"][task_id]
        or success["generic_scaffold"][task_id]
        for task_id in task_ids
    )
    gates = {
        "exact_task_matrix": all(
            len(rows) == int(thresholds["expected_tasks"])
            for rows in episodes.values()
        ),
        "target_capability": source_free_ceiling_successes
        >= int(thresholds["minimum_ceiling_successes"]),
        "source_not_below_neural": summaries["source_induced"]["successes"]
        >= summaries["neural_only"]["successes"],
        "negative_transfer": comparisons["neural_only"]["negative_transfer_rate"]
        <= float(thresholds["maximum_negative_transfer_rate"]),
        "source_behavior_nontrivial": summaries["source_induced"]["changed_actions"]
        >= int(thresholds["minimum_changed_actions"]),
        "permuted_task_action_contrast": task_level_contrasts
        >= int(thresholds["minimum_permuted_first_action_contrasts"]),
        "multiple_source_effect_types_selected": len(selected_effects)
        >= int(thresholds["minimum_selected_effect_types"]),
        "selection_outcome_blind": all(
            not record["selection_read_official_success"]
            for values in episodes.values() for row in values
            for record in row["records"]
        ),
    }
    if config["role"] == "formal":
        gates.update({
            "source_strictly_beats_neural": summaries["source_induced"]["successes"]
            > summaries["neural_only"]["successes"],
            "source_strictly_beats_permuted": summaries["source_induced"]["successes"]
            > summaries["source_permuted"]["successes"],
            "source_strictly_beats_generic": summaries["source_induced"]["successes"]
            > summaries["generic_scaffold"]["successes"],
        })
    passed = all(gates.values())
    status = (
        ("ALFWORLD_PHASE3_REPLICATION_VALIDATED" if passed else
         "ALFWORLD_PHASE3_REPLICATION_FAILED")
        if config["role"] == "formal" else
        ("ALFWORLD_PHASE3_QUALIFICATION_PASSED" if passed else
         "ALFWORLD_PHASE3_QUALIFICATION_FAILED")
    )
    body = {
        "schema_version": "phase3-alfworld-replication-report-v1",
        "status": status,
        "role": config["role"],
        "config_path": str(config_path),
        "config_sha256": config["config_sha256"],
        "tasks": len(task_ids),
        "conditions": list(CONDITIONS),
        "paired_task_order_verified": True,
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "task_level_source_permuted_action_contrasts": task_level_contrasts,
        "source_free_target_capability_oracle": {
            "definition": (
                "PER_TASK_UNION_OF_TARGET_NATIVE_OFFICIAL_EXPERT_"
                "NEURAL_ONLY_AND_SOURCE_FREE_GENERIC"
            ),
            "source_induced_excluded": True,
            "successes": source_free_ceiling_successes,
        },
        "selected_program_counts": dict(selected_programs),
        "selected_effect_counts": dict(selected_effects),
        "gates": gates,
        "formal_results_used_to_change_protocol": False,
        "source_identity_used_as_runtime_feature": False,
        "episodes": episodes,
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": status,
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "task_level_source_permuted_action_contrasts": task_level_contrasts,
        "selected_effect_counts": dict(selected_effects),
        "gates": gates,
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
