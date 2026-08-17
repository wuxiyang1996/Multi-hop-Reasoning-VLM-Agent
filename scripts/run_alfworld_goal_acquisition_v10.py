#!/usr/bin/env python3
"""Run V10 authentic/ceiling and reuse unchanged V9 ALFWorld control arms."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_alfworld_goal_relation_macro_v3 as v3  # noqa: E402
from motif_transfer.active_video_transfer import exact_binomial_two_sided  # noqa: E402
from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.alfworld_goal_acquisition_v10 import (  # noqa: E402
    AUTHENTIC,
    CARDINALITY_CONTROL,
    CEILING,
    CONDITIONS,
    EFFECT_CONTROL,
    GENERIC,
    RAW,
    TargetAcquisitionExecutionState,
    choose_goal_relation_action,
    configure_source_acquisition,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.source_goal_relation_induction import (  # noqa: E402
    validate_goal_relation_macro_program,
)


EXECUTED_CONDITIONS = (AUTHENTIC, CEILING)
REUSED_CONDITIONS = (RAW, CARDINALITY_CONTROL, EFFECT_CONTROL, GENERIC)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(payload: dict, field: str) -> None:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise SystemExit(f"invalid {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/alfworld_goal_acquisition_v10_development.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _self_hash(config, "config_sha256")
    if config.get("experiment_version") != (
        "TYPED_HANDLE_PRESERVING_ACQUISITION_CONSUMED_DEVELOPMENT_V10"
    ):
        raise SystemExit("not a V10 consumed-development acquisition config")
    dependencies = {
        "v10_runner_file_sha256": Path(__file__).resolve(),
        "v10_target_runtime_file_sha256": (
            REPO / "src/motif_transfer/alfworld_goal_acquisition_v10.py"
        ),
        "source_artifact_file_sha256": REPO / config["source_artifact"],
        "source_confirmation_file_sha256": REPO / config[
            "source_confirmation"
        ],
        "target_grounder_file_sha256": REPO / config["target_grounder"],
        "target_causal_effect_file_sha256": REPO / config[
            "target_causal_effect_artifact"
        ],
        "source_acquisition_artifact_file_sha256": REPO / config[
            "source_acquisition_artifact"
        ],
        "source_acquisition_confirmation_file_sha256": REPO / config[
            "source_acquisition_confirmation"
        ],
        "reused_v9_report_file_sha256": REPO / config["reused_v9_report"],
    }
    for field, path in dependencies.items():
        if _sha256(path) != config.get(field):
            raise SystemExit(f"frozen V10 dependency changed: {path}")
    source = json.loads(
        dependencies["source_artifact_file_sha256"].read_text(encoding="utf-8")
    )
    source_confirmation = json.loads(
        dependencies["source_confirmation_file_sha256"].read_text(
            encoding="utf-8"
        )
    )
    target = json.loads(
        dependencies["target_grounder_file_sha256"].read_text(encoding="utf-8")
    )
    target_causal = json.loads(
        dependencies["target_causal_effect_file_sha256"].read_text(
            encoding="utf-8"
        )
    )
    acquisition = json.loads(
        dependencies["source_acquisition_artifact_file_sha256"].read_text(
            encoding="utf-8"
        )
    )
    acquisition_confirmation = json.loads(
        dependencies[
            "source_acquisition_confirmation_file_sha256"
        ].read_text(encoding="utf-8")
    )
    reuse = json.loads(
        dependencies["reused_v9_report_file_sha256"].read_text(encoding="utf-8")
    )
    _self_hash(reuse, "report_sha256")
    validate_goal_relation_macro_program(source)
    if not source_confirmation.get("source_gate_passed"):
        raise SystemExit("source relation macro did not pass fresh confirmation")
    if not target.get("target_grounder_gate", {}).get("passed"):
        raise SystemExit("target-native neural grounder gate did not pass")
    if not target_causal.get("gates", {}).get(
        "effect_balanced_accuracy_at_least_0p80"
    ):
        raise SystemExit("target-native causal effect head gate did not pass")
    configure_source_acquisition(acquisition, acquisition_confirmation)
    if tuple(config["conditions"]) != CONDITIONS:
        raise SystemExit("ALFWorld relation-macro condition matrix changed")
    if tuple(config["executed_conditions"]) != EXECUTED_CONDITIONS:
        raise SystemExit("V10 executed condition set changed")
    if tuple(config["reused_conditions"]) != REUSED_CONDITIONS:
        raise SystemExit("V10 reused condition set changed")
    if reuse.get("status") != "CONSUMED_DEVELOPMENT_ACQUISITION_GATE_FAILED":
        raise SystemExit("V9 reuse report has unexpected status")

    v3.choose_goal_relation_action = choose_goal_relation_action
    v3.TargetRelationExecutionState = TargetAcquisitionExecutionState
    task_ids = tuple(map(str, config["task_ids"]))
    episodes = {
        condition: [dict(row) for row in reuse["episodes"][condition]]
        for condition in REUSED_CONDITIONS
    }
    for condition in EXECUTED_CONDITIONS:
        rows = []
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
                episode = v3._run_episode(
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
                rows.append(episode)
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
        episodes[condition] = rows

    raw_by_task = {row["task_id"]: row for row in episodes[RAW]}
    for condition, rows in episodes.items():
        if {row["task_id"] for row in rows} != set(raw_by_task):
            raise SystemExit(f"V10 task identity mismatch: {condition}")
        for row in rows:
            raw = raw_by_task[row["task_id"]]
            row["changed_actions_vs_raw_trajectory"] = sum(
                left != right for left, right in zip_longest(
                    row["actions"], raw["actions"], fillvalue=None,
                )
            )
    summaries = {name: v3._summary(episodes[name]) for name in CONDITIONS}
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
    diagnostics = {
        condition: dict(Counter(
            str(record["diagnostic"])
            for episode in episodes[condition]
            for record in episode["records"]
        ))
        for condition in CONDITIONS
    }
    authentic_count = diagnostics[AUTHENTIC].get(
        "SOURCE_INDUCED_ACQUISITION_OPERATOR_GROUNDED", 0,
    )
    ceiling_count = diagnostics[CEILING].get(
        "SOURCE_INDUCED_ACQUISITION_OPERATOR_GROUNDED", 0,
    )
    control_count = sum(
        diagnostics[name].get(
            "SOURCE_INDUCED_ACQUISITION_OPERATOR_GROUNDED", 0,
        )
        for name in (CARDINALITY_CONTROL, EFFECT_CONTROL, GENERIC)
    )
    wrong_handle_effects = sum(
        int(episode["effect_counts"].get("RELATE_NO_PROGRESS", 0))
        for episode in episodes[AUTHENTIC]
    )
    gates = {
        "complete_matched_task_matrix": all(
            len(episodes[name]) == len(task_ids) for name in CONDITIONS
        ),
        "matched_actual_task_identities": all(
            {row["task_id"] for row in episodes[name]} == set(raw_by_task)
            for name in CONDITIONS
        ),
        "source_fresh_confirmation_passed": True,
        "source_acquisition_fresh_confirmation_passed": bool(
            acquisition_confirmation["source_gate_passed"]
        ),
        "target_neural_grounder_gate_passed": True,
        "authentic_executes_recurrent_relation": summaries[AUTHENTIC][
            "source_relation_transitions"
        ] >= 2,
        "authentic_executes_source_induced_acquisition": (
            authentic_count >= int(config["gates"][
                "minimum_source_acquisition_groundings"
            ])
        ),
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
        "authentic_matches_target_native_acquisition_execution": (
            authentic_count == ceiling_count
        ),
        "source_acquisition_control_isolation": control_count == 0,
        "zero_wrong_handle_relation_effects": wrong_handle_effects == 0,
        "zero_reopened_completed_slots": summaries[AUTHENTIC][
            "tasks_with_reopened_completed_slots"
        ] == 0,
    }
    passed = all(gates.values())
    body = {
        "schema_version": "alfworld-goal-acquisition-development-v10",
        "status": (
            "CONSUMED_DEVELOPMENT_TYPED_ACQUISITION_GATE_PASSED" if passed
            else "CONSUMED_DEVELOPMENT_TYPED_ACQUISITION_GATE_FAILED"
        ),
        "experiment_version": str(config["experiment_version"]),
        "claim_boundary": str(config["claim_boundary"]),
        "config_sha256": str(config["config_sha256"]),
        "source_artifact_sha256": str(source["artifact_sha256"]),
        "source_acquisition_artifact_sha256": str(
            acquisition["artifact_sha256"]
        ),
        "source_acquisition_confirmation_sha256": str(
            acquisition_confirmation["report_sha256"]
        ),
        "reused_v9_report_sha256": str(reuse["report_sha256"]),
        "executed_conditions": list(EXECUTED_CONDITIONS),
        "reused_conditions": list(REUSED_CONDITIONS),
        "target_grounder_kind": str(target["target_grounder"]["kind"]),
        "task_ids": list(task_ids),
        "summaries": summaries,
        "paired": paired,
        "acquisition_diagnostics": diagnostics,
        "acquisition_groundings": {
            "authentic": authentic_count,
            "target_native_ceiling": ceiling_count,
            "source_controls": control_count,
        },
        "wrong_handle_relation_effects": wrong_handle_effects,
        "gates": gates,
        "episodes": {name: episodes[name] for name in CONDITIONS},
    }
    report = body | {"report_sha256": stable_hash(body)}
    output = REPO / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "v10_status": report["status"],
        "summaries": summaries,
        "paired": paired,
        "acquisition_groundings": report["acquisition_groundings"],
        "wrong_handle_relation_effects": wrong_handle_effects,
        "gates": gates,
        "v10_report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
