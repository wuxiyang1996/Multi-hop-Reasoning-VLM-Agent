#!/usr/bin/env python3
"""Freeze development smoke or the eight unopened ALFWorld V11 probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_goal_acquisition_v10 import (  # noqa: E402
    CONDITIONS,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


PHASE5 = REPO / "configs/phase5_unified_applicability_v1_frozen.json"
V10 = REPO / "configs/alfworld_goal_acquisition_v10_development.json"
FIVE_ROOTS = (
    WORKSPACE / "Multi-hop-Reasoning-VLM-Agent",
    WORKSPACE / "Multi-hop-Reasoning-VLM-Agent-experiment-clean",
    WORKSPACE / "Multi-hop-Reasoning-VLM-Agent-github-main",
    WORKSPACE / "Multi-hop-Reasoning-VLM-Agent-source-fresh-v1",
    WORKSPACE / "Multi-hop-Reasoning-VLM-Agent-two-agent-clean",
)
DEVELOPMENT_TASK = (
    "pick_two_obj_and_place-Pencil-None-Desk-327/"
    "trial_T20190907_234611_599540/game.tw-pddl"
)


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
        raise ValueError(f"invalid {field}")


def _run_exposure_count(task_id: str) -> int:
    command = [
        "rg", "-l", "--fixed-strings", "--glob", "runs/**/*.json",
        task_id.removesuffix("/game.tw-pddl"), *map(str, FIVE_ROOTS),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr.strip() or "run exposure audit failed")
    return len([row for row in completed.stdout.splitlines() if row.strip()])


def _dependency_paths(v10: Mapping[str, Any]) -> dict[str, Path]:
    return {
        "runner_file_sha256": REPO / (
            "scripts/run_alfworld_unified_goal_acquisition_v11.py"
        ),
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
        "source_artifact_file_sha256": REPO / str(v10["source_artifact"]),
        "source_confirmation_file_sha256": REPO / str(
            v10["source_confirmation"]
        ),
        "source_acquisition_artifact_file_sha256": REPO / str(
            v10["source_acquisition_artifact"]
        ),
        "source_acquisition_confirmation_file_sha256": REPO / str(
            v10["source_acquisition_confirmation"]
        ),
        "source_inducer_file_sha256": REPO / (
            "src/motif_transfer/source_goal_acquisition_induction.py"
        ),
        "target_grounder_file_sha256": REPO / str(v10["target_grounder"]),
        "target_causal_effect_file_sha256": REPO / str(
            v10["target_causal_effect_artifact"]
        ),
        "calibration_report_file_sha256": REPO / (
            "runs/alfworld_goal_acquisition_v10_development/analysis_report.json"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("development", "formal", "operational_retry"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen config: {args.output}")
    v10 = _read(V10); _self_hash(v10, "config_sha256")
    phase5 = _read(PHASE5); _self_hash(phase5, "manifest_sha256")
    dependencies = _dependency_paths(v10)
    if args.role in {"formal", "operational_retry"}:
        probes = list(phase5["future_probes"]["alfworld"])
        task_ids = [str(row["task_id"]) for row in probes]
        if len(task_ids) != 8 or any(
            row.get("formal_outcome_read") is not False
            or int(row.get("prior_execution_occurrences", -1)) != 0
            for row in probes
        ):
            raise SystemExit("Phase-5 ALFWorld probes are not eight unopened tasks")
        scan_exposure = {
            task_id: _run_exposure_count(task_id) for task_id in task_ids
        }
        if args.role == "formal" and any(scan_exposure.values()):
            raise SystemExit(
                f"formal ALFWorld reserve was already executed: {scan_exposure}"
            )
        exposure: Mapping[str, Any] = scan_exposure
        if args.role == "operational_retry":
            source_opened = {
                "pick_two_obj_and_place-ToiletPaper-None-CounterTop-402/"
                "trial_T20190912_052310_164968/game.tw-pddl",
                "pick_two_obj_and_place-TissueBox-None-DiningTable-230/"
                "trial_T20190909_145505_799057/game.tw-pddl",
            }
            exposure = {
                task_id: {
                    "raw_target_only": 1,
                    "authentic_source_goal_relation_macro": int(
                        task_id in source_opened
                    ),
                    "run_json_occurrences": scan_exposure[task_id],
                }
                for task_id in task_ids
            }
        data_root = Path(str(v10["alfworld_data"])) / "json_2.1.1/train"
        for row in probes:
            path = data_root / str(row["task_id"])
            if _sha(path) != row["task_file_sha256"]:
                raise SystemExit(f"ALFWorld task file changed: {path}")
        status = (
            "FROZEN_BEFORE_ANY_ALFWORLD_V11_RESERVE_RESET_OR_OUTCOME"
            if args.role == "formal" else
            "FROZEN_IDENTITY_ONLY_OPERATIONAL_RETRY_AFTER_V11_ABORT"
        )
        output_result = (
            "runs/alfworld_unified_goal_acquisition_v11_formal/report.json"
            if args.role == "formal" else
            "runs/alfworld_unified_goal_acquisition_v12_operational_retry/report.json"
        )
        minimum_groundings = 4
        maximum_p = 0.05
        claim = (
            "Eight execution-untouched Phase-5 ALFWorld train multiplicity tasks; "
            "V10 source program, neural grounder, thresholds, controls, unified "
            "authority chain, and endpoints frozen before any reset or outcome."
        )
        if args.role == "operational_retry":
            parent = REPO / "configs/alfworld_unified_goal_acquisition_v11_formal.json"
            parent_config = _read(parent); _self_hash(parent_config, "config_sha256")
            abort = REPO / (
                "runs/alfworld_unified_goal_acquisition_v11_formal/"
                "abort_receipt.json"
            )
            if parent_config["task_ids"] != task_ids or not abort.is_file():
                raise SystemExit("V11 parent/abort lineage is incomplete")
            claim = (
                "Identity-only operational retry of the prospectively frozen "
                "eight-task V11 reserve after all raw and two authentic outcomes "
                "were exposed. Source programs, target grounder, thresholds, "
                "conditions, horizons, gates, calibration, and action logic are "
                "unchanged; only receipt identity is bound after reset."
            )
    else:
        probes = []
        task_ids = [DEVELOPMENT_TASK]
        exposure = {DEVELOPMENT_TASK: "CONSUMED_V10_DEVELOPMENT"}
        status = "CONSUMED_DEVELOPMENT_UNIFIED_WRAPPER_SMOKE"
        output_result = "runs/alfworld_unified_goal_acquisition_v11_development/report.json"
        minimum_groundings = 1
        maximum_p = 1.0
        claim = (
            "Unified-wrapper integration smoke on one consumed V10 development "
            "task; excluded from every prospective estimate."
        )

    body = {
        "schema_version": "alfworld-unified-goal-acquisition-config-v11",
        "status": status, "role": args.role,
        "claim_boundary": claim,
        "source_artifact": v10["source_artifact"],
        "source_confirmation": v10["source_confirmation"],
        "source_acquisition_artifact": v10["source_acquisition_artifact"],
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
        "alfworld_config": v10["alfworld_config"],
        "alfworld_data": v10["alfworld_data"],
        "seed": (
            486311 if args.role in {"formal", "operational_retry"} else 486310
        ),
        "max_steps": v10["max_steps"],
        "thresholds": v10["thresholds"],
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "phase5_manifest_sha256": phase5["manifest_sha256"],
        "pre_freeze_run_exposure": exposure,
        "formal_target_outcome_read_for_freeze": (
            args.role == "operational_retry"
        ),
        "formal_reserve_task_opened": (
            False if args.role == "formal" else
            True if args.role == "operational_retry" else None
        ),
        "gates": {
            "minimum_source_acquisition_groundings": minimum_groundings,
            "maximum_exact_two_sided_p": maximum_p,
            "require_strict_source_control_superiority": True,
            "require_zero_negative_transfer": True,
            "require_target_native_ceiling_match": True,
        },
        "output": output_result,
        **{field: _sha(path) for field, path in dependencies.items()},
    }
    if args.role == "operational_retry":
        body["parent_v11_config"] = (
            "configs/alfworld_unified_goal_acquisition_v11_formal.json"
        )
        body["parent_v11_config_file_sha256"] = _sha(parent)
        body["parent_v11_config_sha256"] = parent_config["config_sha256"]
        body["v11_abort_receipt"] = (
            "runs/alfworld_unified_goal_acquisition_v11_formal/abort_receipt.json"
        )
        body["v11_abort_receipt_file_sha256"] = _sha(abort)
        body["only_runtime_change"] = (
            "BIND_AUTHORIZATION_TASK_ID_FROM_RESOLVED_GAME_FILE_AFTER_RESET"
        )
    config = body | {"config_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": status, "tasks": len(task_ids),
        "pre_freeze_run_exposure": exposure,
        "config_sha256": config["config_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
