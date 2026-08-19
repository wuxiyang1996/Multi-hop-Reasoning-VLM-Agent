#!/usr/bin/env python3
"""Freeze and compile the unused official ALFWorld valid-seen reserve."""

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
SOURCE_SPLIT = SOURCE_DATA / "json_2.1.1/valid_seen"
OUTPUT_DIR = REPO / "configs/alfworld_target_acquisition_fresh_v18"
SELECTION_PATH = OUTPUT_DIR / "selection.json"
COMPILER_AUDIT_PATH = OUTPUT_DIR / "compiler_audit.json"
CONFIG_PATH = OUTPUT_DIR / "formal.json"
GENERATED_DATA = REPO / "runs/alfworld_target_acquisition_fresh_v18/alfworld_data"
GENERATED_SPLIT = GENERATED_DATA / "json_2.1.1/valid_seen"
GENERATOR = PROJECT_ROOT / "conda/envs/alfworld/bin/alfworld-generate"
NAMESPACE = "alfworld-target-acquisition-fresh-v18-valid-seen"
STATUS = "FROZEN_BEFORE_COMPILATION_OR_ANY_V18_POLICY_RESET"
FORMAL_STATUS = "FROZEN_BEFORE_ANY_V19_FRESH_POLICY_RESET_OR_OUTCOME"
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
        raise ValueError(f"invalid {field}: {claimed}")


def _candidate_rows() -> list[dict[str, Any]]:
    rows = []
    for trajectory in sorted(SOURCE_SPLIT.glob("**/traj_data.json")):
        relative_dir = trajectory.parent.relative_to(SOURCE_SPLIT)
        game = trajectory.parent / "game.tw-pddl"
        task_id = str(relative_dir / "game.tw-pddl")
        data = _read(trajectory)
        if data.get("task_type") != "pick_two_obj_and_place":
            continue
        if game.exists() or "Sliced" in task_id or "movable" in task_id:
            continue
        initial = trajectory.parent / "initial_state.pddl"
        if not initial.is_file():
            raise FileNotFoundError(initial)
        rows.append({
            "task_id": task_id,
            "traj_data_file_sha256": _sha(trajectory),
            "initial_state_file_sha256": _sha(initial),
        })
    return rows


def _selection() -> dict[str, Any]:
    rows = _candidate_rows()
    if len(rows) != 8:
        raise RuntimeError(
            "expected all eight unused supported valid-seen multiplicity "
            f"identities, observed {len(rows)}"
        )
    body = {
        "schema_version": "alfworld-target-acquisition-fresh-v18-selection",
        "status": STATUS,
        "namespace": NAMESPACE,
        "source_split": "official_alfworld_json_2.1.1_valid_seen",
        "selection_rule": (
            "ALL SORTED PICK_TWO_OBJ_AND_PLACE IDENTITIES WITH TRAJ_DATA AND "
            "INITIAL_STATE BUT NO PACKAGED GAME.TW-PDDL; EXCLUDE OFFICIAL "
            "MOVABLE-RECEPTACLE AND SLICED UNSUPPORTED IDENTITIES"
        ),
        "selection_used_directory_identity_and_task_type_only": True,
        "selection_used_observation_walkthrough_or_policy_outcome": False,
        "pre_freeze_policy_execution_exposure_by_task": {
            row["task_id"]: 0 for row in rows
        },
        "tasks": rows,
    }
    return body | {"selection_sha256": stable_hash(body)}


def _freeze_selection() -> dict[str, Any]:
    expected = _selection()
    if SELECTION_PATH.exists():
        observed = _read(SELECTION_PATH)
        _self_hash(observed, "selection_sha256")
        if observed != expected:
            raise RuntimeError("existing V18 selection differs from current audit")
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
        raise RuntimeError("partial V18 generated data has no frozen logic")

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
            "--seed", "20260819",
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
        "schema_version": "alfworld-target-acquisition-fresh-v18-compiler-audit",
        "status": "COMPILED_AFTER_IDENTITY_FREEZE_BEFORE_POLICY_RESET",
        "selection_sha256": str(selection["selection_sha256"]),
        "compiler": "OFFICIAL_ALFWORLD_GENERATE_WITH_HANDCODED_EXPERT",
        "compiler_solvability_used_for_identity_selection": False,
        "compiler_walkthrough_exposed_to_transfer_policy": False,
        "generated_game_file_sha256": generated_hashes,
        "compiler_results": compiler_results,
    }
    audit = body | {"compiler_audit_sha256": stable_hash(body)}
    if COMPILER_AUDIT_PATH.exists():
        observed = _read(COMPILER_AUDIT_PATH)
        _self_hash(observed, "compiler_audit_sha256")
        if observed != audit:
            raise RuntimeError("V18 compiler output changed after freeze")
    else:
        COMPILER_AUDIT_PATH.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return audit


def _formal_config(
    selection: Mapping[str, Any], compiler: Mapping[str, Any],
) -> dict[str, Any]:
    if not all(
        row["solvable"] for row in compiler["compiler_results"].values()
    ):
        raise RuntimeError(
            "the frozen reserve contains a compiler-rejected identity; "
            "refusing outcome-conditioned replacement"
        )
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
    task_ids = [str(row["task_id"]) for row in selection["tasks"]]
    body |= {
        "schema_version": "alfworld-target-acquisition-fresh-v19-config",
        "status": FORMAL_STATUS,
        "role": "prospective_execution_untouched_mechanism_replication",
        "claim_boundary": (
            "All eight official valid_seen multiplicity trajectories that "
            "had initial_state.pddl but no packaged game.tw-pddl. Identities "
            "were frozen before compilation; compiler solvability and "
            "walkthroughs cannot select identities and are never policy "
            "inputs. This is a paired mechanism/acquisition replication, not "
            "a powered OOD population-effect test."
        ),
        "alfworld_data": str(GENERATED_DATA),
        "alfworld_config_file_sha256": _sha(
            Path(str(body["alfworld_config"]))
        ),
        "alfworld_physical_split": "valid_seen",
        "seed": 190819,
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "task_file_sha256": dict(compiler["generated_game_file_sha256"]),
        "selection": "configs/alfworld_target_acquisition_fresh_v18/selection.json",
        "selection_file_sha256": _sha(SELECTION_PATH),
        "selection_sha256": str(selection["selection_sha256"]),
        "compiler_audit": (
            "configs/alfworld_target_acquisition_fresh_v18/compiler_audit.json"
        ),
        "compiler_audit_file_sha256": _sha(COMPILER_AUDIT_PATH),
        "compiler_audit_sha256": str(compiler["compiler_audit_sha256"]),
        "compiler_solvability_used_for_identity_selection": False,
        "compiler_walkthrough_exposed_to_transfer_policy": False,
        "target_k1_program": "docs/results/alfworld_target_only_k1_program_v16.json",
        "target_k1_program_file_sha256": _sha(
            REPO / "docs/results/alfworld_target_only_k1_program_v16.json"
        ),
        "target_inducer_file_sha256": _sha(
            REPO / "src/motif_transfer/alfworld_target_recurrent_induction.py"
        ),
        "preparer_file_sha256": _sha(Path(__file__)),
        "preparer": "scripts/prepare_alfworld_target_acquisition_fresh_v18.py",
        "v19_runner_file_sha256": _sha(
            REPO / "scripts/run_alfworld_target_acquisition_fresh_v19.py"
        ),
        "gates": {
            "minimum_source_acquisition_groundings": len(task_ids),
            "maximum_exact_two_sided_p": 1.0,
            "minimum_fresh_tasks": 8,
            "require_source_target_k1_trace_equivalence": True,
            "require_gain_over_raw_and_permuted": True,
            "require_zero_negative_transfer": True,
        },
        "target_complete_trajectory_budget": 1,
        "source_complete_target_trajectory_budget": 0,
        "output": "runs/alfworld_target_acquisition_fresh_v19/report.json",
    }
    if tuple(body["conditions"]) != CONDITIONS:
        raise RuntimeError("parent ALFWorld condition matrix changed")
    return body | {"config_sha256": stable_hash(body)}


def main() -> int:
    selection = _freeze_selection()
    compiler = _compile(selection)
    config = _formal_config(selection, compiler)
    if CONFIG_PATH.exists():
        observed = _read(CONFIG_PATH)
        _self_hash(observed, "config_sha256")
        if observed != config:
            raise RuntimeError("existing V19 formal config differs after freeze")
    else:
        CONFIG_PATH.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "selection_sha256": selection["selection_sha256"],
        "compiler_audit_sha256": compiler["compiler_audit_sha256"],
        "compiler_results": compiler["compiler_results"],
        "config_sha256": config["config_sha256"],
        "task_count": config["task_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
