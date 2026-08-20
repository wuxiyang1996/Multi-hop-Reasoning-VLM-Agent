#!/usr/bin/env python3
"""Freeze prospective MiniGrid target seeds and all V31 gates."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import gymnasium
import minigrid
from minigrid.core.grid import Grid
from minigrid.envs.empty import EmptyEnv
from minigrid.minigrid_env import MiniGridEnv
import sklearn


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


DEFAULT_OUTPUT = REPO / "configs/minigrid_orientation_target_v31.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {output}")
    dependencies = {
        "orientation_program": REPO / "src/motif_transfer/minigrid_orientation_recovery.py",
        "neural_grounder": REPO / "src/motif_transfer/minigrid_neural_grounder.py",
        "runner": REPO / "scripts/run_minigrid_orientation_target_v31.py",
        "source_report": REPO / "docs/results/tetris_cyclic_source_induction_v28.json",
        "minigrid_empty_env": Path(inspect.getfile(EmptyEnv)),
        "minigrid_base_env": Path(inspect.getfile(MiniGridEnv)),
        "minigrid_grid": Path(inspect.getfile(Grid)),
    }
    source_report = json.loads(dependencies["source_report"].read_text())
    source_program = source_report["development"]["first_qualified"]["program"]
    if source_program["status"] != "SOURCE_CYCLIC_IDENTITY_PROGRAM_INDUCED":
        raise ValueError("V28 source program is not qualified")
    config = {
        "schema_version": "minigrid-orientation-target-protocol-v31",
        "status": "FROZEN_BEFORE_ANY_V31_TARGET_PROTOCOL_SEED",
        "claim_boundary": (
            "Prospective transfer of the exact V28 source-induced anonymous "
            "COMPOSE(PROBE_EFFECT, RECOVERY_EFFECT)==IDENTITY program to a new "
            "MiniGrid oriented-navigation recovery MDP. A development-only "
            "target MLP grounds rendered agent directions; the source program "
            "sees only anonymous order-four effects. Native success requires "
            "the recovered heading to let a pre-intervention BFS suffix reach "
            "the official environment goal. A separately trained neural-only "
            "target policy receives 64 recovery labels, while source transfer "
            "receives zero complete target trajectories. The target-written "
            "isomorphic ceiling is expected to match source execution, so this "
            "tests transferable program content, not provenance necessity. "
            "The custom recovery protocol is built on MiniGrid-Empty-Random; "
            "it is not claimed as an official MiniGrid benchmark task."
        ),
        "selection": {
            "pilot_seed_namespaces_excluded": [
                "740xxx-simulator-sanity", "750xxx-openrouter-visual-pilot",
                "751xxx-local-mlp-pilot",
            ],
            "protocol_seed_contents_or_outcomes_read_before_freeze": False,
            "rule": (
                "Commit disjoint integer seeds before generating any V31 "
                "render or task outcome: 760001..760064 development, "
                "760101..760124 qualification, 760201..760248 formal reserve."
            ),
        },
        "target": {
            "environment_id": "MiniGrid-Empty-Random-6x6-v0",
            "namespace": "tetris-v28-to-minigrid-orientation-v31",
            "calibration_seed_offset": 900000,
            "group_order": 4,
            "probe_length_range": [7, 14],
            "candidate_macros": 4,
            "candidate_token_mapping": "per-seed SHA256 permutation",
            "native_success": (
                "after probe and selected recovery, execute the reset-state "
                "BFS suffix and require official terminated-on-goal"
            ),
        },
        "splits": {
            "development": list(range(760001, 760065)),
            "qualification": list(range(760101, 760125)),
            "formal_reserve": list(range(760201, 760249)),
        },
        "grounder": {
            "artifact_namespace": "minigrid-orientation-neural-grounder-v31",
            "feature_side": 20,
            "crop_radius": 22,
            "orientation_hidden": [32],
            "direct_hidden": [64, 32],
            "random_state": 310031,
            "orientation_minimum_confidence": 0.90,
            "direct_minimum_confidence": 0.0,
            "acquisition_reads": (
                "development rendered pixels, 7 direction labels per task, "
                "and one recovery-token label for the neural-only baseline"
            ),
            "forbidden_acquisition_reads": [
                "target success", "target reward", "complete target trajectory",
                "source program", "source identity",
            ],
            "thresholds_frozen_before_development": True,
        },
        "controls": {
            "conditions": [
                "source_induced", "alpha_renamed_source",
                "target_written_isomorphic", "neural_only_direct",
                "copy_effect_control", "fixed_token_control",
                "shuffled_binding_control",
            ],
            "shuffle_namespace": "minigrid-orientation-v31-binding-shuffle",
        },
        "gates": {
            "development": {
                "minimum_grounder_task_coverage": 0.95,
                "minimum_grounder_panel_accuracy": 0.99,
                "minimum_source_success_rate": 0.95,
                "maximum_destructive_control_success_rate": 0.65,
                "require_source_above_neural_only": False,
                "maximum_neural_only_p_value": 0.05,
            },
            "qualification": {
                "minimum_grounder_task_coverage": 0.95,
                "minimum_grounder_panel_accuracy": 0.99,
                "minimum_source_success_rate": 0.95,
                "maximum_destructive_control_success_rate": 0.65,
                "require_source_above_neural_only": True,
                "maximum_neural_only_p_value": 0.05,
            },
            "formal_reserve": {
                "minimum_grounder_task_coverage": 0.95,
                "minimum_grounder_panel_accuracy": 0.99,
                "minimum_source_success_rate": 0.95,
                "maximum_destructive_control_success_rate": 0.65,
                "require_source_above_neural_only": True,
                "maximum_neural_only_p_value": 0.05,
            },
        },
        "authority": {
            "development_report": "runs/minigrid_orientation_target_v31/development_report.json",
            "qualification_report": "runs/minigrid_orientation_target_v31/qualification_report.json",
        },
        "outputs": {
            "run_dir": "runs/minigrid_orientation_target_v31",
            "grounder_artifact": "runs/minigrid_orientation_target_v31/grounder_artifact.json",
            "development_report": "runs/minigrid_orientation_target_v31/development_report.json",
            "qualification_report": "runs/minigrid_orientation_target_v31/qualification_report.json",
            "formal_reserve_report": "runs/minigrid_orientation_target_v31/formal_reserve_report.json",
        },
        "package_versions": {
            "gymnasium": str(gymnasium.__version__),
            "minigrid": str(minigrid.__version__),
            "scikit_learn": str(sklearn.__version__),
        },
        "source_report": "docs/results/tetris_cyclic_source_induction_v28.json",
        "source_program_sha256": source_program["program_sha256"],
        "dependency_fields": {},
    }
    for name, path in dependencies.items():
        path_field = name if name == "source_report" else f"{name}_path"
        hash_field = f"{name}_file_sha256"
        if name != "source_report":
            try:
                config[path_field] = str(path.resolve().relative_to(REPO.resolve()))
            except ValueError:
                config[path_field] = str(path.resolve())
        config[hash_field] = _sha(path)
        config["dependency_fields"][hash_field] = path_field
    body = config
    config = body | {"config_sha256": stable_hash(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = freeze(args.output.resolve())
    print(json.dumps({
        "status": config["status"], "splits": {
            key: len(value) for key, value in config["splits"].items()
        }, "config_sha256": config["config_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
