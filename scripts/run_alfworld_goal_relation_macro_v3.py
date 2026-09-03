#!/usr/bin/env python3
"""Run the source-induced recurrent relation macro on consumed ALFWorld dev."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.active_video_transfer import exact_binomial_two_sided  # noqa: E402
from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.alfworld_goal_relation_macro import (  # noqa: E402
    AUTHENTIC,
    CARDINALITY_CONTROL,
    CEILING,
    CONDITIONS,
    EFFECT_CONTROL,
    GENERIC,
    RAW,
    TargetRelationExecutionState,
    choose_goal_relation_action,
    observe_goal_relation_transition,
    reconcile_bound_relation_objects,
    target_relation_state,
)
from motif_transfer.alfworld_hierarchical_grounder import score_actions  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.slot_aware_alfworld_harness import (  # noqa: E402
    initialize_slot_ledger,
)
from motif_transfer.source_goal_relation_induction import (  # noqa: E402
    validate_goal_relation_macro_program,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(payload: Mapping[str, Any], field: str) -> None:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def _run_episode(
    *,
    environment: ALFWorldTextBatchEnvironment,
    condition: str,
    source_artifact: Mapping[str, Any],
    target_grounder: Mapping[str, Any],
    target_causal_effect_head: Mapping[str, Any],
    max_steps: int,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    observation = environment.reset()
    task_id = str(environment.resolved_game_file)
    goal = str(observation.state.get("task_goal", ""))
    ledger = initialize_slot_ledger(
        goal, required_property="NONE",
        initial_observation=str(observation.state.get("observation", "")),
    )
    execution = TargetRelationExecutionState()
    history: list[str] = []
    records = []
    for step in range(max_steps):
        before_text = str(observation.state.get("observation", ""))
        ledger = reconcile_bound_relation_objects(ledger, before_text)
        grounded = score_actions(
            goal=goal,
            observation=before_text,
            native_actions=observation.native_actions,
            step=step,
            action_history=history,
            artifact=target_grounder,
        )
        if not grounded:
            break
        decision = choose_goal_relation_action(
            condition=condition,
            grounded=grounded,
            goal=goal,
            history=history,
            ledger=ledger,
            execution_state=execution,
            source_artifact=source_artifact,
            target_causal_effect_head=target_causal_effect_head,
            step=step,
            max_steps=max_steps,
            minimum_binding=float(thresholds["minimum_binding"]),
            minimum_realization=float(thresholds["minimum_realization"]),
            minimum_binding_margin=float(thresholds["minimum_binding_margin"]),
            minimum_causal_effect=float(thresholds["minimum_causal_effect"]),
        )
        selected = str(decision["action"])
        before_state = target_relation_state(ledger)
        after, discarded_reward = environment.step(selected)
        ledger, effect = observe_goal_relation_transition(
            ledger,
            action=selected,
            after_observation=str(after.state.get("observation", "")),
        )
        after_state = target_relation_state(ledger)
        record_body = {
            "step": step,
            "selected_action": selected,
            "fallback_action": str(decision["fallback_action"]),
            "raw_fallback_action": str(decision["raw_fallback_action"]),
            "changed_action_vs_raw": selected != decision["raw_fallback_action"],
            "changed_action_after_first_relation": bool(
                int(before_state["completed_count"]) >= 1
                and selected != decision["raw_fallback_action"]
            ),
            "source_admitted": bool(decision["source_admitted"]),
            "program_active": bool(decision["program_active"]),
            "program_status": str(decision["program_status"]),
            "target_native_obligation": decision.get("target_native_obligation"),
            "diagnostic": str(decision["diagnostic"]),
            "target_candidate_count": int(decision.get("candidate_count", 0)),
            "target_best_binding": (
                float(decision["best_binding"])
                if "best_binding" in decision else None
            ),
            "target_binding_margin": (
                float(decision["binding_margin"])
                if "binding_margin" in decision else None
            ),
            "target_best_realization_score": (
                float(decision["best_realization_score"])
                if "best_realization_score" in decision else None
            ),
            "target_best_causal_effect_probability": (
                float(decision["best_causal_effect_probability"])
                if "best_causal_effect_probability" in decision else None
            ),
            "target_effect_receipt": str(effect["target_effect_receipt"]),
            "relation_coverage_before": float(effect["relation_coverage_before"]),
            "relation_coverage_after": float(effect["relation_coverage_after"]),
            "source_transition_advanced": bool(effect["source_transition_advanced"]),
            "source_terminal_observed": bool(effect["source_terminal_observed"]),
            "completed_count_before": int(before_state["completed_count"]),
            "completed_count_after": int(after_state["completed_count"]),
            "reopened_completed_slots": int(after_state["reopened_completed_slots"]),
            "failed_postconditions": int(ledger["failed_postconditions"]),
            "official_success_after": bool(after.official_success),
            "reward_discarded_for_selection": True,
            "before_state_sha256": stable_hash({
                "observation": before_text,
                "native_actions": observation.native_actions,
            }),
            "after_state_sha256": stable_hash({
                "observation": str(after.state.get("observation", "")),
                "native_actions": after.native_actions,
            }),
        }
        records.append(record_body | {"record_sha256": stable_hash(record_body)})
        _ = discarded_reward
        history.append(selected)
        observation = after
        if after.terminal or after.official_success:
            break
    body = {
        "task_id": task_id,
        "condition": condition,
        "official_success": bool(records and records[-1]["official_success_after"]),
        "steps": len(records),
        "actions": [row["selected_action"] for row in records],
        "source_admissions": sum(row["source_admitted"] for row in records),
        "source_relation_transitions": sum(
            row["source_transition_advanced"] for row in records
        ),
        "source_terminal_observed": any(
            row["source_terminal_observed"] for row in records
        ),
        "changed_actions_vs_raw_fallback": sum(
            row["changed_action_vs_raw"] for row in records
        ),
        "changed_actions_after_first_relation": sum(
            row["changed_action_after_first_relation"] for row in records
        ),
        "final_slot_state": target_relation_state(ledger),
        "effect_counts": dict(Counter(
            row["target_effect_receipt"] for row in records
        )),
        "records": records,
    }
    return body | {"episode_sha256": stable_hash(body)}


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "tasks": len(rows),
        "successes": sum(bool(row["official_success"]) for row in rows),
        "success_rate": sum(bool(row["official_success"]) for row in rows) / len(rows),
        "mean_steps": sum(int(row["steps"]) for row in rows) / len(rows),
        "source_admissions": sum(int(row["source_admissions"]) for row in rows),
        "source_relation_transitions": sum(
            int(row["source_relation_transitions"]) for row in rows
        ),
        "changed_actions_vs_raw_fallback": sum(
            int(row["changed_actions_vs_raw_fallback"]) for row in rows
        ),
        "changed_actions_after_first_relation": sum(
            int(row["changed_actions_after_first_relation"]) for row in rows
        ),
        "tasks_with_reopened_completed_slots": sum(
            int(row["final_slot_state"]["reopened_completed_slots"] > 0)
            for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/alfworld_goal_relation_macro_v3_development.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _self_hash(config, "config_sha256")
    if config.get("status") != "FROZEN_CONSUMED_DEVELOPMENT_BEFORE_OUTCOMES":
        raise SystemExit("ALFWorld relation-macro development config is not frozen")
    dependencies = {
        "runner_file_sha256": Path(__file__).resolve(),
        "target_runtime_file_sha256": (
            REPO / "src/motif_transfer/alfworld_goal_relation_macro.py"
        ),
        "source_artifact_file_sha256": REPO / config["source_artifact"],
        "source_confirmation_file_sha256": REPO / config["source_confirmation"],
        "target_grounder_file_sha256": REPO / config["target_grounder"],
        "target_causal_effect_file_sha256": REPO / config[
            "target_causal_effect_artifact"
        ],
    }
    for field, path in dependencies.items():
        if _sha256(path) != config[field]:
            raise SystemExit(f"frozen ALFWorld dependency changed: {path}")
    source = json.loads(
        dependencies["source_artifact_file_sha256"].read_text(encoding="utf-8")
    )
    source_confirmation = json.loads(
        dependencies["source_confirmation_file_sha256"].read_text(encoding="utf-8")
    )
    target = json.loads(
        dependencies["target_grounder_file_sha256"].read_text(encoding="utf-8")
    )
    target_causal = json.loads(
        dependencies["target_causal_effect_file_sha256"].read_text(
            encoding="utf-8"
        )
    )
    validate_goal_relation_macro_program(source)
    if not source_confirmation.get("source_gate_passed"):
        raise SystemExit("source relation macro did not pass fresh confirmation")
    if not target.get("target_grounder_gate", {}).get("passed"):
        raise SystemExit("target-native neural grounder gate did not pass")
    if not target_causal.get("gates", {}).get(
        "effect_balanced_accuracy_at_least_0p80"
    ):
        raise SystemExit("target-native causal effect head gate did not pass")
    if tuple(config["conditions"]) != CONDITIONS:
        raise SystemExit("ALFWorld relation-macro condition matrix changed")
    task_ids = tuple(map(str, config["task_ids"]))
    episodes: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(config["alfworld_config"]),
            data_path=str(config["alfworld_data"]),
            split="train",
            seed=int(config["seed"]),
            game_ids=task_ids,
            max_steps=int(config["max_steps"]),
        )
        try:
            for index in range(len(task_ids)):
                episode = _run_episode(
                    environment=environment,
                    condition=condition,
                    source_artifact=source,
                    target_grounder=target["target_grounder"],
                    target_causal_effect_head=target_causal[
                        "target_causal_effect_head"
                    ],
                    max_steps=int(config["max_steps"]),
                    thresholds=config["thresholds"],
                )
                episodes[condition].append(episode)
                print(json.dumps({
                    "condition": condition,
                    "task_index": index,
                    "task_id": episode["task_id"],
                    "success": episode["official_success"],
                    "steps": episode["steps"],
                    "relations": episode["source_relation_transitions"],
                }), flush=True)
        finally:
            environment.close()

    raw_by_task = {row["task_id"]: row for row in episodes[RAW]}
    for condition, rows in episodes.items():
        for row in rows:
            raw = raw_by_task[row["task_id"]]
            row["changed_actions_vs_raw_trajectory"] = sum(
                left != right for left, right in zip_longest(
                    row["actions"], raw["actions"], fillvalue=None,
                )
            )
    summaries = {name: _summary(rows) for name, rows in episodes.items()}
    authentic = {row["task_id"]: row for row in episodes[AUTHENTIC]}
    paired = {}
    for comparator in CONDITIONS:
        if comparator == AUTHENTIC:
            continue
        other = {row["task_id"]: row for row in episodes[comparator]}
        wins = losses = 0
        for task_id, row in authentic.items():
            left = bool(row["official_success"])
            right = bool(other[task_id]["official_success"])
            wins += left and not right
            losses += right and not left
        paired[comparator] = {
            "wins": wins,
            "losses": losses,
            "ties": len(task_ids) - wins - losses,
            "net_wins": wins - losses,
            "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
        }
    gates = {
        "complete_matched_task_matrix": all(
            len(rows) == len(task_ids) for rows in episodes.values()
        ),
        "matched_actual_task_identities": all(
            {row["task_id"] for row in rows} == set(raw_by_task)
            for rows in episodes.values()
        ),
        "source_fresh_confirmation_passed": True,
        "target_neural_grounder_gate_passed": True,
        "authentic_executes_recurrent_relation": summaries[AUTHENTIC][
            "source_relation_transitions"
        ] >= 2,
        "nontrivial_second_cycle_action_change": summaries[AUTHENTIC][
            "changed_actions_after_first_relation"
        ] >= int(config["gates"]["minimum_second_cycle_action_changes"]),
        "authentic_success_gain_over_raw": summaries[AUTHENTIC]["successes"]
        > summaries[RAW]["successes"],
        "authentic_strictly_beats_source_controls": all(
            summaries[AUTHENTIC]["successes"] > summaries[name]["successes"]
            for name in (CARDINALITY_CONTROL, EFFECT_CONTROL, GENERIC)
        ),
        "zero_negative_transfer_vs_raw": paired[RAW]["losses"] == 0,
        "matches_target_native_recurrent_ceiling": summaries[AUTHENTIC][
            "successes"
        ] == summaries[CEILING]["successes"],
        "zero_reopened_completed_slots": summaries[AUTHENTIC][
            "tasks_with_reopened_completed_slots"
        ] == 0,
    }
    passed = all(gates.values())
    body = {
        "schema_version": "alfworld-goal-relation-macro-development-v3",
        "status": (
            "CONSUMED_DEVELOPMENT_GATE_PASSED" if passed
            else "CONSUMED_DEVELOPMENT_GATE_FAILED"
        ),
        "claim_boundary": str(config["claim_boundary"]),
        "source_artifact_sha256": str(source["artifact_sha256"]),
        "target_grounder_kind": str(target["target_grounder"]["kind"]),
        "task_ids": list(task_ids),
        "summaries": summaries,
        "paired": paired,
        "gates": gates,
        "episodes": episodes,
    }
    report = body | {"report_sha256": stable_hash(body)}
    output = REPO / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "summaries": summaries,
        "paired": paired,
        "gates": gates,
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
