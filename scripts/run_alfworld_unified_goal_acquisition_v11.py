#!/usr/bin/env python3
"""Run the unified-harness ALFWorld acquisition route on a frozen task set."""

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
sys.path.insert(0, str(REPO / "scripts"))
# The project base interpreter owns scipy/sklearn used by the unified gate,
# while the isolated ALFWorld environment owns the pure-Python environment
# package.  Appending (not prepending) keeps base binary wheels authoritative.
for _site in sorted((REPO.parent / "conda/envs/alfworld/lib").glob(
    "python*/site-packages"
)):
    if (_site / "alfworld").is_dir():
        sys.path.append(str(_site))
        break

import run_alfworld_goal_relation_macro_v3 as base  # noqa: E402
from motif_transfer.active_video_transfer import (  # noqa: E402
    exact_binomial_two_sided,
)
from motif_transfer.alfworld_env import (  # noqa: E402
    ALFWorldTextBatchEnvironment,
)
from motif_transfer.alfworld_goal_acquisition_v10 import (  # noqa: E402
    AUTHENTIC,
    CARDINALITY_CONTROL,
    CEILING,
    CONDITIONS,
    EFFECT_CONTROL,
    GENERIC,
    RAW,
    TargetAcquisitionExecutionState,
    configure_source_acquisition,
)
from motif_transfer.alfworld_unified_goal_acquisition_v11 import (  # noqa: E402
    ROUTE_ID,
    authority_receipts,
    build_unified_authorization,
    choose_goal_relation_action,
    configure_unified_authorization,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.source_goal_relation_induction import (  # noqa: E402
    validate_goal_relation_macro_program,
)
from motif_transfer.unified_transfer_runtime import (  # noqa: E402
    PairedCalibration,
)


FORMAL_STATUS = "FROZEN_BEFORE_ANY_ALFWORLD_V11_RESERVE_RESET_OR_OUTCOME"
RETRY_STATUS = "FROZEN_IDENTITY_ONLY_OPERATIONAL_RETRY_AFTER_V11_ABORT"
DEVELOPMENT_STATUS = "CONSUMED_DEVELOPMENT_UNIFIED_WRAPPER_SMOKE"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(payload: Mapping[str, Any], field: str) -> None:
    body = dict(payload)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _dependencies(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        "runner_file_sha256": Path(__file__).resolve(),
        "unified_wrapper_file_sha256": REPO / (
            "src/motif_transfer/alfworld_unified_goal_acquisition_v11.py"
        ),
        "structural_applicability_file_sha256": REPO / (
            "src/motif_transfer/structural_ir_applicability.py"
        ),
        "unified_harness_file_sha256": REPO / (
            "src/motif_transfer/unified_neurosymbolic_harness.py"
        ),
        "unified_runtime_file_sha256": REPO / (
            "src/motif_transfer/unified_transfer_runtime.py"
        ),
        "v10_target_runtime_file_sha256": REPO / (
            "src/motif_transfer/alfworld_goal_acquisition_v10.py"
        ),
        "source_artifact_file_sha256": REPO / str(config["source_artifact"]),
        "source_confirmation_file_sha256": REPO / str(
            config["source_confirmation"]
        ),
        "source_acquisition_artifact_file_sha256": REPO / str(
            config["source_acquisition_artifact"]
        ),
        "source_acquisition_confirmation_file_sha256": REPO / str(
            config["source_acquisition_confirmation"]
        ),
        "source_inducer_file_sha256": REPO / (
            "src/motif_transfer/source_goal_acquisition_induction.py"
        ),
        "target_grounder_file_sha256": REPO / str(config["target_grounder"]),
        "target_causal_effect_file_sha256": REPO / str(
            config["target_causal_effect_artifact"]
        ),
        "calibration_report_file_sha256": REPO / str(
            config["calibration_report"]
        ),
    }


def _pair(
    episodes: Mapping[str, list[Mapping[str, Any]]], comparator: str,
) -> dict[str, Any]:
    left = {row["task_id"]: row for row in episodes[AUTHENTIC]}
    right = {row["task_id"]: row for row in episodes[comparator]}
    wins = sum(
        bool(row["official_success"])
        and not bool(right[task_id]["official_success"])
        for task_id, row in left.items()
    )
    losses = sum(
        not bool(row["official_success"])
        and bool(right[task_id]["official_success"])
        for task_id, row in left.items()
    )
    return {
        "wins": wins, "losses": losses,
        "ties": len(left) - wins - losses,
        "net_wins": wins - losses,
        "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
        "negative_transfer_rate": (
            losses / (wins + losses) if wins + losses else 0.0
        ),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _read(config_path)
    _self_hash(config, "config_sha256")
    if config.get("status") not in {
        FORMAL_STATUS, RETRY_STATUS, DEVELOPMENT_STATUS,
    }:
        raise ValueError("unsupported unified ALFWorld V11 config status")
    if tuple(config["conditions"]) != CONDITIONS:
        raise ValueError("ALFWorld V11 condition matrix changed")
    dependencies = _dependencies(config)
    for field, path in dependencies.items():
        if _sha(path) != config.get(field):
            raise ValueError(f"frozen dependency changed: {path}")

    source = _read(dependencies["source_artifact_file_sha256"])
    source_confirmation = _read(
        dependencies["source_confirmation_file_sha256"]
    )
    acquisition = _read(
        dependencies["source_acquisition_artifact_file_sha256"]
    )
    acquisition_confirmation = _read(
        dependencies["source_acquisition_confirmation_file_sha256"]
    )
    target = _read(dependencies["target_grounder_file_sha256"])
    target_causal = _read(dependencies["target_causal_effect_file_sha256"])
    calibration = _read(dependencies["calibration_report_file_sha256"])
    _self_hash(calibration, "analysis_report_sha256")
    validate_goal_relation_macro_program(source)
    if not source_confirmation.get("source_gate_passed"):
        raise ValueError("source relation program is not fresh-confirmed")
    if not target.get("target_grounder_gate", {}).get("passed"):
        raise ValueError("target-native neural grounder is not qualified")
    if not target_causal.get("gates", {}).get(
        "effect_balanced_accuracy_at_least_0p80"
    ):
        raise ValueError("target-native causal-effect head is not qualified")
    configure_source_acquisition(acquisition, acquisition_confirmation)

    base.choose_goal_relation_action = choose_goal_relation_action
    base.TargetRelationExecutionState = TargetAcquisitionExecutionState
    task_ids = tuple(map(str, config["task_ids"]))
    episodes: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    authority_by_task: dict[str, list[Mapping[str, Any]]] = {}
    phase7_by_task: dict[str, Mapping[str, Any]] = {}
    calibration_counts = config["calibration_counts"]

    for condition in CONDITIONS:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(config["alfworld_config"]),
            data_path=str(config["alfworld_data"]), split="train",
            seed=int(config["seed"]), game_ids=task_ids,
            max_steps=int(config["max_steps"]),
        )
        try:
            for index, task_id in enumerate(task_ids):
                context_holder: dict[str, Any] = {}
                if condition == AUTHENTIC:
                    original_reset = environment.reset

                    def reset_and_authorize():
                        observation = original_reset()
                        actual_task_id = str(Path(
                            environment.resolved_game_file
                        ).resolve().relative_to(
                            (Path(str(config["alfworld_data"]))
                             / "json_2.1.1/train").resolve()
                        ))
                        context = build_unified_authorization(
                            task_id=actual_task_id,
                            acquisition_artifact=acquisition,
                            acquisition_confirmation=acquisition_confirmation,
                            target_grounder_sha256=_sha(
                                dependencies["target_grounder_file_sha256"]
                            ),
                            target_executor_sha256=_sha(
                                dependencies["unified_wrapper_file_sha256"]
                            ),
                            evidence_report_sha256=str(
                                calibration["analysis_report_sha256"]
                            ),
                            inducer_artifact_sha256=_sha(
                                dependencies["source_inducer_file_sha256"]
                            ),
                            utility_vs_neural=PairedCalibration(**(
                                calibration_counts["utility_vs_neural"]
                            )),
                            authenticity_vs_source_permuted=PairedCalibration(**(
                                calibration_counts[
                                    "authenticity_vs_source_permuted"
                                ]
                            )),
                        )
                        configure_unified_authorization(context)
                        context_holder.update({
                            "task_id": actual_task_id, "context": context,
                        })
                        return observation

                    environment.reset = reset_and_authorize
                try:
                    episode = base._run_episode(
                        environment=environment, condition=condition,
                        source_artifact=source,
                        target_grounder=target["target_grounder"],
                        target_causal_effect_head=target_causal[
                            "target_causal_effect_head"
                        ],
                        max_steps=int(config["max_steps"]),
                        thresholds=config["thresholds"],
                    )
                finally:
                    if condition == AUTHENTIC:
                        environment.reset = original_reset
                episodes[condition].append(episode)
                if condition == AUTHENTIC:
                    actual_task_id = str(context_holder["task_id"])
                    context = context_holder["context"]
                    authority_by_task[actual_task_id] = list(authority_receipts())
                    phase7_by_task[actual_task_id] = {
                        "route_id": context.phase7.route_id,
                        "verdict": context.phase7.verdict.value,
                        "reason": context.phase7.reason,
                        "authorization_sha256": (
                            context.phase7.authorization_sha256
                        ),
                        "utility_authorization_sha256": (
                            context.utility.authorization_sha256
                        ),
                        "utility_lower_bound": context.utility.utility_lower_bound,
                        "authenticity_lower_bound": (
                            context.utility.authenticity_lower_bound
                        ),
                        "target_action_emitted": (
                            context.phase7.target_action_emitted
                        ),
                        "current_target_outcome_read": (
                            context.phase7.current_target_outcome_read
                        ),
                    }
                print(json.dumps({
                    "condition": condition, "task_index": index,
                    "task_id": episode["task_id"],
                    "success": episode["official_success"],
                    "steps": episode["steps"],
                    "authority_calls": len(authority_receipts())
                    if condition == AUTHENTIC else 0,
                }), flush=True)
        finally:
            environment.close()

    raw_by_task = {row["task_id"]: row for row in episodes[RAW]}
    for condition, rows in episodes.items():
        if {row["task_id"] for row in rows} != set(raw_by_task):
            raise ValueError(f"task identity mismatch: {condition}")
        for row in rows:
            raw = raw_by_task[row["task_id"]]
            row["changed_actions_vs_raw_trajectory"] = sum(
                left != right for left, right in zip_longest(
                    row["actions"], raw["actions"], fillvalue=None,
                )
            )
    summaries = {
        condition: base._summary(rows)
        for condition, rows in episodes.items()
    }
    paired = {
        comparator: _pair(episodes, comparator)
        for comparator in CONDITIONS if comparator != AUTHENTIC
    }
    diagnostics = {
        condition: dict(Counter(
            str(record["diagnostic"])
            for episode in episodes[condition]
            for record in episode["records"]
        ))
        for condition in CONDITIONS
    }
    authority_rows = [
        row for rows in authority_by_task.values() for row in rows
    ]
    acquisition_groundings = diagnostics[AUTHENTIC].get(
        "SOURCE_INDUCED_ACQUISITION_OPERATOR_GROUNDED", 0,
    )
    wrong_handle_effects = sum(
        int(row["effect_counts"].get("RELATE_NO_PROGRESS", 0))
        for row in episodes[AUTHENTIC]
        if int(row["final_slot_state"].get("completed_count", 0)) >= 1
    )
    formal = config.get("status") in {FORMAL_STATUS, RETRY_STATUS}
    gates = {
        "complete_matched_task_matrix": all(
            len(rows) == len(task_ids) for rows in episodes.values()
        ),
        "matched_actual_task_identities": all(
            {row["task_id"] for row in rows} == set(raw_by_task)
            for rows in episodes.values()
        ),
        "all_tasks_pre_authorized_by_unified_harness": (
            len(phase7_by_task) == len(task_ids)
            and all(row["verdict"] == "SELECT_SKILL" for row in phase7_by_task.values())
        ),
        "unified_route_is_exact": all(
            row["route_id"] == ROUTE_ID for row in phase7_by_task.values()
        ),
        "selector_emits_no_action_and_reads_no_current_outcome": all(
            row["target_action_emitted"] is False
            and row["current_target_outcome_read"] is False
            for row in phase7_by_task.values()
        ),
        "every_source_active_action_uses_target_native_executor": (
            bool(authority_rows)
            and all(
                row["target_executor_calls"] == 1
                and row["source_selector_action_emitted"] is False
                and row["formal_outcome_read"] is False
                for row in authority_rows
            )
        ),
        "source_acquisition_fresh_confirmation_passed": bool(
            acquisition_confirmation["source_gate_passed"]
        ),
        "target_neural_grounder_gate_passed": True,
        "authentic_executes_source_induced_acquisition": (
            acquisition_groundings >= int(config["gates"][
                "minimum_source_acquisition_groundings"
            ])
        ),
        "authentic_success_gain_over_raw": (
            summaries[AUTHENTIC]["successes"] > summaries[RAW]["successes"]
        ),
        "authentic_strictly_beats_source_controls": all(
            summaries[AUTHENTIC]["successes"] > summaries[name]["successes"]
            for name in (CARDINALITY_CONTROL, EFFECT_CONTROL, GENERIC)
        ),
        "source_vs_raw_exact_significance": (
            not formal or paired[RAW]["exact_two_sided_p"] <= float(
                config["gates"]["maximum_exact_two_sided_p"]
            )
        ),
        "zero_negative_transfer_vs_raw": paired[RAW]["losses"] == 0,
        "matches_target_native_ceiling": (
            summaries[AUTHENTIC]["successes"] == summaries[CEILING]["successes"]
        ),
        "zero_wrong_handle_relation_effects": wrong_handle_effects == 0,
        "zero_reopened_completed_slots": summaries[AUTHENTIC][
            "tasks_with_reopened_completed_slots"
        ] == 0,
    }
    passed = all(gates.values())
    body = {
        "schema_version": "alfworld-unified-goal-acquisition-report-v11",
        "status": (
            "ALFWORLD_UNIFIED_GOAL_ACQUISITION_FORMAL_VALIDATED"
            if passed and formal else
            "ALFWORLD_UNIFIED_GOAL_ACQUISITION_DEVELOPMENT_PASSED"
            if passed else
            "ALFWORLD_UNIFIED_GOAL_ACQUISITION_FORMAL_FAILED"
            if formal else
            "ALFWORLD_UNIFIED_GOAL_ACQUISITION_DEVELOPMENT_FAILED"
        ),
        "role": "formal_reserve" if formal else "consumed_development",
        "claim_boundary": str(config["claim_boundary"]),
        "config_sha256": str(config["config_sha256"]),
        "source_artifact_sha256": str(source["artifact_sha256"]),
        "source_acquisition_artifact_sha256": str(
            acquisition["artifact_sha256"]
        ),
        "unified_route_id": ROUTE_ID,
        "task_ids": list(task_ids),
        "calibration_counts": calibration_counts,
        "phase7_authorizations": phase7_by_task,
        "authority_receipts": authority_by_task,
        "summaries": summaries, "paired": paired,
        "acquisition_diagnostics": diagnostics,
        "acquisition_groundings": acquisition_groundings,
        "wrong_handle_relation_effects": wrong_handle_effects,
        "gates": gates, "episodes": episodes,
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config)
    config = _read(args.config)
    output = REPO / str(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "summaries": report["summaries"],
        "paired": report["paired"],
        "authority_calls": sum(map(len, report["authority_receipts"].values())),
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
