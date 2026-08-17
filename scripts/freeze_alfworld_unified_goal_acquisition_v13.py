#!/usr/bin/env python3
"""Freeze an untouched, deterministic ALFWorld multiplicity replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import freeze_alfworld_unified_goal_acquisition_v11 as prior  # noqa: E402
from motif_transfer.alfworld_goal_acquisition_v10 import (  # noqa: E402
    CONDITIONS,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from run_alfworld_unified_goal_acquisition_v13 import (  # noqa: E402
    V13_STATUS,
)


SELECTION_SEED = 486312
DEFAULT_TASK_COUNT = 24
DATA_SPLIT = "valid_train"


def deterministic_reserve(
    candidates: Iterable[str], *, seed: int, count: int,
) -> list[str]:
    """Select without outcome access using a stable content-derived ranking."""
    unique = sorted(set(map(str, candidates)))
    ranked = sorted(
        unique,
        key=lambda task_id: (
            hashlib.sha256(f"{seed}\0{task_id}".encode()).hexdigest(),
            task_id,
        ),
    )
    if len(ranked) < count:
        raise ValueError(f"only {len(ranked)} untouched candidates for {count}")
    return ranked[:count]


def _historically_exposed(task_ids: list[str]) -> set[str]:
    """Return candidate IDs occurring anywhere in historical run JSON files."""
    run_roots = [root / "runs" for root in prior.FIVE_ROOTS]
    run_roots = [root for root in run_roots if root.is_dir()]
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as patterns:
        patterns.write("\n".join(task_ids) + "\n")
        patterns.flush()
        command = [
            "rg", "--fixed-strings", "--only-matching", "--no-filename",
            "--glob", "*.json", "--file", patterns.name,
            *map(str, run_roots),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            completed.stderr.strip() or "historical exposure audit failed"
        )
    return {row for row in completed.stdout.splitlines() if row}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}: {claimed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=DEFAULT_TASK_COUNT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen config: {args.output}")
    if args.task_count < 12:
        raise SystemExit("formal replication requires at least 12 tasks")

    v10_path = REPO / "configs/alfworld_goal_acquisition_v10_development.json"
    v10 = _read(v10_path)
    _self_hash(v10, "config_sha256")
    dependencies = prior._dependency_paths(v10)
    data_root = (
        Path(str(v10["alfworld_data"])) / "json_2.1.1" / DATA_SPLIT
    ).resolve()
    task_files = sorted(data_root.glob(
        "pick_two_obj_and_place-*/trial_*/game.tw-pddl"
    ))
    all_tasks = [str(path.relative_to(data_root)) for path in task_files]
    if len(all_tasks) != 45:
        raise SystemExit(
            f"expected the immutable 45-task valid_train multiplicity pool, "
            f"found {len(all_tasks)}"
        )
    exposed = _historically_exposed(all_tasks)
    untouched = sorted(set(all_tasks) - exposed)
    task_ids = deterministic_reserve(
        untouched, seed=SELECTION_SEED, count=args.task_count,
    )
    if any(task_id in exposed for task_id in task_ids):
        raise AssertionError("selected reserve contains an exposed task")

    prior_report_path = REPO / (
        "runs/alfworld_unified_goal_acquisition_v12_operational_retry/"
        "report.json"
    )
    prior_report = _read(prior_report_path)
    _self_hash(prior_report, "report_sha256")
    launcher = REPO / "scripts/run_alfworld_unified_goal_acquisition_v13.py"
    freezer = Path(__file__).resolve()
    transport_config = REPO / "configs/alfworld_valid_train_base_config.yaml"
    population_hash = stable_hash({
        "split": DATA_SPLIT,
        "all_task_ids": all_tasks,
        "historically_exposed_task_ids": sorted(exposed),
        "untouched_task_ids": untouched,
    })
    selection_hash = stable_hash({
        "algorithm": "ASCENDING_SHA256_OF_SEED_NUL_TASK_ID",
        "seed": SELECTION_SEED,
        "count": args.task_count,
        "population_sha256": population_hash,
        "task_ids": task_ids,
    })

    body = {
        "schema_version": "alfworld-unified-goal-acquisition-config-v13",
        "status": V13_STATUS,
        "role": "independent_formal_replication",
        "claim_boundary": (
            f"Independent deterministic sample of {args.task_count} "
            "execution-untouched "
            "ALFWorld valid_train multiplicity tasks. Selection uses only task "
            "identity and historical run exposure, never task outcomes. The "
            "V11 source programs, target-native neural grounder, symbolic "
            "authority chain, executor, controls, thresholds, horizon, and "
            "statistical gates are reused without change."
        ),
        "source_artifact": v10["source_artifact"],
        "source_confirmation": v10["source_confirmation"],
        "source_acquisition_artifact": v10[
            "source_acquisition_artifact"
        ],
        "source_acquisition_confirmation": v10[
            "source_acquisition_confirmation"
        ],
        "target_grounder": v10["target_grounder"],
        "target_causal_effect_artifact": v10[
            "target_causal_effect_artifact"
        ],
        "calibration_report": (
            "runs/alfworld_goal_acquisition_v10_development/analysis_report.json"
        ),
        "calibration_counts": {
            "utility_vs_neural": {"wins": 7, "losses": 0, "ties": 17},
            "authenticity_vs_source_permuted": {
                "wins": 7, "losses": 0, "ties": 17,
            },
            "authority": "CONSUMED_DEVELOPMENT_V10_PAIRED_OUTCOMES_ONLY",
            "current_reserve_outcome_read": False,
        },
        "conditions": list(CONDITIONS),
        "alfworld_config": str(transport_config),
        "alfworld_data": v10["alfworld_data"],
        "data_split": DATA_SPLIT,
        "seed": SELECTION_SEED,
        "max_steps": v10["max_steps"],
        "thresholds": v10["thresholds"],
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "task_file_sha256": {
            task_id: _sha(data_root / task_id) for task_id in task_ids
        },
        "selection": {
            "algorithm": "ASCENDING_SHA256_OF_SEED_NUL_TASK_ID",
            "seed": SELECTION_SEED,
            "all_multiplicity_tasks": len(all_tasks),
            "historically_exposed_tasks": len(exposed),
            "execution_untouched_candidates": len(untouched),
            "population_sha256": population_hash,
            "selection_sha256": selection_hash,
        },
        "pre_freeze_run_exposure": {task_id: 0 for task_id in task_ids},
        "formal_target_outcome_read_for_freeze": False,
        "formal_reserve_task_opened": False,
        "gates": {
            "minimum_source_acquisition_groundings": 24,
            "maximum_exact_two_sided_p": 0.05,
            "require_strict_source_control_superiority": True,
            "require_zero_negative_transfer": True,
            "require_target_native_ceiling_match": True,
        },
        "output": (
            "runs/alfworld_unified_goal_acquisition_v13_formal/report.json"
        ),
        "v13_launcher_file_sha256": _sha(launcher),
        "v13_freezer_file_sha256": _sha(freezer),
        "valid_train_transport_config": str(transport_config.relative_to(REPO)),
        "valid_train_transport_config_file_sha256": _sha(transport_config),
        "prior_v12_report": str(prior_report_path.relative_to(REPO)),
        "prior_v12_report_file_sha256": _sha(prior_report_path),
        "prior_v12_report_sha256": prior_report["report_sha256"],
        "prior_v12_role": "POWER_DIAGNOSTIC_ONLY_NOT_SELECTION_INPUT",
        **{field: _sha(path) for field, path in dependencies.items()},
    }
    config = body | {"config_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": V13_STATUS,
        "all_multiplicity_tasks": len(all_tasks),
        "historically_exposed_tasks": len(exposed),
        "execution_untouched_candidates": len(untouched),
        "selected_tasks": len(task_ids),
        "selection_sha256": selection_hash,
        "config_sha256": config["config_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
