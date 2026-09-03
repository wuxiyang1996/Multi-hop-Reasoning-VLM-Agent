#!/usr/bin/env python3
"""Freeze and compile a target-outcome-blind ALFWorld train ITT reserve."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO.parent
SOURCE_DATA = (
    PROJECT_ROOT / "Multi-hop-Reasoning-VLM-Agent-github-main/.cache/alfworld_data"
)
SOURCE_SPLIT = SOURCE_DATA / "json_2.1.1/train"
OUTPUT_DIR = REPO / "configs/alfworld_target_acquisition_train_itt_v20"
SELECTION_PATH = OUTPUT_DIR / "selection.json"
COMPILER_AUDIT_PATH = OUTPUT_DIR / "compiler_audit.json"
CONFIG_PATH = OUTPUT_DIR / "formal.json"
RETRY_CONFIG_PATH = OUTPUT_DIR / "formal_retry.json"
RETRY2_CONFIG_PATH = OUTPUT_DIR / "formal_retry2.json"
GENERATED_DATA = (
    REPO / "runs/alfworld_target_acquisition_train_itt_v20/alfworld_data"
)
GENERATED_SPLIT = GENERATED_DATA / "json_2.1.1/train"
GENERATOR = PROJECT_ROOT / "conda/envs/alfworld/bin/alfworld-generate"
NAMESPACE = "alfworld-target-acquisition-fresh-v20-train-itt"
STATUS = "FROZEN_TRAIN_ITT_BEFORE_COMPILATION_OR_ANY_POLICY_RESET"
FORMAL_STATUS = "FROZEN_BEFORE_ANY_V19_FRESH_POLICY_RESET_OR_OUTCOME"
TASK_COUNT = 16
TASK_OFFSET = 0
SOLVABLE_ELIGIBILITY_FILTER = False
MINIMUM_FORMAL_TASKS = 16
PREPARER_RELATIVE = (
    "scripts/prepare_alfworld_target_acquisition_train_itt_v20.py"
)
REPORT_OUTPUT = "runs/alfworld_target_acquisition_train_itt_v20/report.json"
REQUIRED_PYTHON_EXECUTABLE = (
    PROJECT_ROOT / "conda/envs/cosplay-candy-a100/bin/python"
)
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_goal_relation_macro import CONDITIONS  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def _rank(task_id: str) -> str:
    return hashlib.sha256(f"{NAMESPACE}\0{task_id}".encode()).hexdigest()


def _candidate_rows() -> list[dict[str, Any]]:
    rows = []
    for trajectory in SOURCE_SPLIT.glob(
        "pick_two_obj_and_place-*/*/traj_data.json"
    ):
        relative_dir = trajectory.parent.relative_to(SOURCE_SPLIT)
        task_id = str(relative_dir / "game.tw-pddl")
        if (
            (trajectory.parent / "game.tw-pddl").exists()
            or "Sliced" in task_id
            or "movable" in task_id
        ):
            continue
        initial = trajectory.parent / "initial_state.pddl"
        if not initial.is_file():
            raise FileNotFoundError(initial)
        rows.append({
            "task_id": task_id,
            "rank_sha256": _rank(task_id),
            "traj_data_file_sha256": _sha(trajectory),
            "initial_state_file_sha256": _sha(initial),
        })
    rows.sort(key=lambda row: str(row["rank_sha256"]))
    return rows


def _selection() -> dict[str, Any]:
    population = _candidate_rows()
    if len(population) != 228:
        raise RuntimeError(
            f"expected 228 unused supported train identities, got {len(population)}"
        )
    selected = population[TASK_OFFSET:TASK_OFFSET + TASK_COUNT]
    body = {
        "schema_version": "alfworld-target-acquisition-train-itt-v20-selection",
        "status": STATUS,
        "namespace": NAMESPACE,
        "source_split": "official_alfworld_json_2.1.1_train",
        "selection_rule": (
            f"RANK SLICE [{TASK_OFFSET}, {TASK_OFFSET + TASK_COUNT}) BY "
            "ASCENDING SHA256(NAMESPACE NUL TASK_ID) FROM ALL "
            "PICK_TWO_OBJ_AND_PLACE IDENTITIES WITH TRAJ_DATA AND INITIAL_STATE "
            "BUT NO PACKAGED GAME.TW-PDDL; EXCLUDE OFFICIAL MOVABLE-RECEPTACLE "
            "AND SLICED UNSUPPORTED IDENTITIES"
        ),
        "eligible_population": len(population),
        "eligible_population_task_ids_sha256": stable_hash([
            row["task_id"] for row in population
        ]),
        "selection_used_identity_only": True,
        "selection_used_compiler_solvability_observation_or_policy_outcome": False,
        "compiler_rejected_tasks_must_remain_in_intention_to_treat_matrix": (
            not SOLVABLE_ELIGIBILITY_FILTER
        ),
        "compiler_solvability_eligibility_rule_frozen_before_compilation": (
            "SOLVABLE_TRUE_ONLY" if SOLVABLE_ELIGIBILITY_FILTER else "ITT_ALL"
        ),
        "runtime_interpreter_contract_frozen_before_compilation": {
            "executable": str(REQUIRED_PYTHON_EXECUTABLE),
            "executable_file_sha256": _sha(REQUIRED_PYTHON_EXECUTABLE),
            "python_major_minor": "3.11",
        },
        "pre_freeze_policy_execution_exposure_by_task": {
            str(row["task_id"]): 0 for row in selected
        },
        "tasks": selected,
    }
    return body | {"selection_sha256": stable_hash(body)}


def _freeze_selection() -> dict[str, Any]:
    expected = _selection()
    if SELECTION_PATH.exists():
        observed = _read(SELECTION_PATH)
        _self_hash(observed, "selection_sha256")
        if observed != expected:
            raise RuntimeError("existing V20 selection differs from current audit")
        return observed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return expected


def _compile(selection: Mapping[str, Any]) -> dict[str, Any]:
    if not GENERATED_DATA.exists():
        GENERATED_SPLIT.mkdir(parents=True)
        shutil.copytree(SOURCE_DATA / "logic", GENERATED_DATA / "logic")
    elif not (GENERATED_DATA / "logic").is_dir():
        raise RuntimeError("partial V20 generated data has no frozen logic")
    all_trajectories = sorted(SOURCE_SPLIT.glob("**/traj_data.json"))
    index_by_task = {
        str(path.parent.relative_to(SOURCE_SPLIT) / "game.tw-pddl"): index
        for index, path in enumerate(all_trajectories)
    }
    for row in selection["tasks"]:
        task_id = str(row["task_id"])
        output = GENERATED_SPLIT / task_id
        if output.exists():
            continue
        index = index_by_task[task_id]
        subprocess.run([
            str(GENERATOR),
            "--data_path", str(SOURCE_SPLIT),
            "--save_path", str(GENERATED_SPLIT),
            "--domain", str(SOURCE_DATA / "logic/alfred.pddl"),
            "--grammar", str(SOURCE_DATA / "logic/alfred.twl2"),
            "--start", str(index), "--end", str(index + 1),
            "--seed", "20260820",
        ], check=True)
    compiler_results = {}
    generated_hashes = {}
    for row in selection["tasks"]:
        task_id = str(row["task_id"])
        game = GENERATED_SPLIT / task_id
        payload = _read(game)
        compiler_results[task_id] = {
            "solvable": payload.get("solvable") is True,
            "walkthrough_length": len(payload.get("walkthrough") or ()),
        }
        generated_hashes[task_id] = _sha(game)
    body = {
        "schema_version": "alfworld-target-acquisition-train-itt-v20-compiler-audit",
        "status": "COMPILED_AFTER_IDENTITY_FREEZE_BEFORE_POLICY_RESET",
        "selection_sha256": str(selection["selection_sha256"]),
        "compiler": "OFFICIAL_ALFWORLD_GENERATE_WITH_HANDCODED_EXPERT",
        "compiler_solvability_used_for_identity_selection": False,
        "compiler_rejected_tasks_retained_in_itt": (
            not SOLVABLE_ELIGIBILITY_FILTER
        ),
        "compiler_walkthrough_exposed_to_transfer_policy": False,
        "generated_game_file_sha256": generated_hashes,
        "compiler_results": compiler_results,
    }
    audit = body | {"compiler_audit_sha256": stable_hash(body)}
    if COMPILER_AUDIT_PATH.exists():
        observed = _read(COMPILER_AUDIT_PATH)
        _self_hash(observed, "compiler_audit_sha256")
        if observed != audit:
            raise RuntimeError("V20 compiler output changed after freeze")
    else:
        COMPILER_AUDIT_PATH.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return audit


def _formal_config(
    selection: Mapping[str, Any], compiler: Mapping[str, Any],
) -> dict[str, Any]:
    parent = _read(REPO / "configs/alfworld_program_driven_policy_v14_formal.json")
    copy_fields = (
        "alfworld_config", "calibration_counts", "calibration_report",
        "calibration_report_file_sha256", "conditions", "max_steps",
        "runner_file_sha256", "source_acquisition_artifact",
        "source_acquisition_artifact_file_sha256",
        "source_acquisition_confirmation",
        "source_acquisition_confirmation_file_sha256", "source_artifact",
        "source_artifact_file_sha256", "source_confirmation",
        "source_confirmation_file_sha256", "source_inducer_file_sha256",
        "structural_applicability_file_sha256",
        "target_causal_effect_artifact", "target_causal_effect_file_sha256",
        "target_grounder", "target_grounder_file_sha256", "thresholds",
        "unified_harness_file_sha256", "unified_runtime_file_sha256",
        "unified_wrapper_file_sha256", "v10_target_runtime_file_sha256",
    )
    body = {field: parent[field] for field in copy_fields}
    selected_ids = [str(row["task_id"]) for row in selection["tasks"]]
    task_ids = (
        [
            task_id for task_id in selected_ids
            if compiler["compiler_results"][task_id]["solvable"]
        ]
        if SOLVABLE_ELIGIBILITY_FILTER else selected_ids
    )
    if len(task_ids) < MINIMUM_FORMAL_TASKS:
        raise RuntimeError(
            "frozen compiler eligibility rule yielded too few formal tasks; "
            "refusing any outcome-conditioned replacement"
        )
    body |= {
        "schema_version": "alfworld-target-acquisition-train-itt-v20-config",
        "status": FORMAL_STATUS,
        "selection_frozen_status": STATUS,
        "role": "prospective_execution_untouched_compiler_valid_replication",
        "claim_boundary": (
            f"Deterministic identity-only rank slice of {TASK_COUNT} from all "
            "228 official ALFWorld train multiplicity trajectories with initial "
            "PDDL but no packaged TextWorld game. The compiler validity rule "
            f"{('SOLVABLE_TRUE_ONLY' if SOLVABLE_ELIGIBILITY_FILTER else 'ITT_ALL')} "
            "was frozen before compilation and is applied to every identity "
            "without replacement. Compiler walkthroughs never reach a policy. "
            "This is a mechanism and acquisition-cost replication, not a "
            "powered split-level claim."
        ),
        "alfworld_data": str(GENERATED_DATA),
        "alfworld_config": str(
            REPO / "configs/alfworld_generated_train_base_config.yaml"
        ),
        "alfworld_config_file_sha256": _sha(
            REPO / "configs/alfworld_generated_train_base_config.yaml"
        ),
        "alfworld_physical_split": "train",
        "seed": 200819,
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "task_file_sha256": dict(compiler["generated_game_file_sha256"]),
        "selection": str(SELECTION_PATH.relative_to(REPO)),
        "selection_file_sha256": _sha(SELECTION_PATH),
        "selection_sha256": str(selection["selection_sha256"]),
        "compiler_audit": str(COMPILER_AUDIT_PATH.relative_to(REPO)),
        "compiler_audit_file_sha256": _sha(COMPILER_AUDIT_PATH),
        "compiler_audit_sha256": str(compiler["compiler_audit_sha256"]),
        "compiler_solvability_used_for_identity_selection": False,
        "compiler_walkthrough_exposed_to_transfer_policy": False,
        "compiler_eligibility_filter": (
            "SOLVABLE_TRUE_ONLY" if SOLVABLE_ELIGIBILITY_FILTER else "ITT_ALL"
        ),
        "compiler_candidate_task_count": len(selected_ids),
        "compiler_eligible_formal_task_count": len(task_ids),
        "target_k1_program": "docs/results/alfworld_target_only_k1_program_v16.json",
        "target_k1_program_file_sha256": _sha(
            REPO / "docs/results/alfworld_target_only_k1_program_v16.json"
        ),
        "target_inducer_file_sha256": _sha(
            REPO / "src/motif_transfer/alfworld_target_recurrent_induction.py"
        ),
        "required_python_executable": str(REQUIRED_PYTHON_EXECUTABLE),
        "required_python_executable_sha256": _sha(
            REQUIRED_PYTHON_EXECUTABLE
        ),
        "required_python_major_minor": "3.11",
        "preparer_file_sha256": _sha(REPO / PREPARER_RELATIVE),
        "preparer": PREPARER_RELATIVE,
        "v19_runner_file_sha256": _sha(
            REPO / "scripts/run_alfworld_target_acquisition_fresh_v19.py"
        ),
        "gates": {
            "minimum_source_acquisition_groundings": len(task_ids),
            "maximum_exact_two_sided_p": 1.0,
            "minimum_fresh_tasks": MINIMUM_FORMAL_TASKS,
            "require_source_target_k1_trace_equivalence": True,
            "require_gain_over_raw_and_permuted": True,
            "require_zero_negative_transfer": True,
        },
        "target_complete_trajectory_budget": 1,
        "source_complete_target_trajectory_budget": 0,
        "output": REPORT_OUTPUT,
    }
    if tuple(body["conditions"]) != CONDITIONS:
        raise RuntimeError("parent ALFWorld condition matrix changed")
    return body | {"config_sha256": stable_hash(body)}


def main() -> int:
    selection = _freeze_selection()
    compiler = _compile(selection)
    config = _formal_config(selection, compiler)
    output_path = CONFIG_PATH
    if CONFIG_PATH.exists():
        observed = _read(CONFIG_PATH)
        _self_hash(observed, "config_sha256")
        if observed != config:
            predecessor = observed
            retry_reason = "PRE_RESET_PREFLIGHT_HARDCODED_V18_PREPARER_PATH"
            output_path = RETRY_CONFIG_PATH
            if RETRY_CONFIG_PATH.exists():
                predecessor = _read(RETRY_CONFIG_PATH)
                _self_hash(predecessor, "config_sha256")
                retry_reason = "PRE_RESET_PREFLIGHT_WRONG_TRAIN_DATASET_ROOT"
                output_path = RETRY2_CONFIG_PATH
            retry_body = dict(config)
            retry_body.pop("config_sha256", None)
            retry_body |= {
                "operational_retry_of_config_sha256": predecessor[
                    "config_sha256"
                ],
                "operational_retry_reason": retry_reason,
                "policy_reset_before_retry": False,
                "selection_compiler_conditions_and_gates_changed": False,
            }
            config = retry_body | {"config_sha256": stable_hash(retry_body)}
            if output_path.exists():
                retry = _read(output_path)
                _self_hash(retry, "config_sha256")
                if retry != config:
                    raise RuntimeError("existing V20 retry config differs")
            else:
                output_path.write_text(
                    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
    else:
        CONFIG_PATH.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "selection_sha256": selection["selection_sha256"],
        "compiler_audit_sha256": compiler["compiler_audit_sha256"],
        "solvable_tasks": sum(
            row["solvable"] for row in compiler["compiler_results"].values()
        ),
        "itt_tasks": len(compiler["compiler_results"]),
        "config_sha256": config["config_sha256"],
        "config_path": str(output_path.relative_to(REPO)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
