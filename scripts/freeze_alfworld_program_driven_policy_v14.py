#!/usr/bin/env python3
"""Freeze every remaining untouched ALFWorld multiplicity task for V14."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import freeze_alfworld_unified_goal_acquisition_v11 as v11_freezer  # noqa: E402
import freeze_alfworld_unified_goal_acquisition_v13 as v13_freezer  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from run_alfworld_program_driven_policy_v14 import V14_STATUS  # noqa: E402


SELECTION_SEED = 921314
EXPECTED_POOL = 45
EXPECTED_PREVIOUSLY_EXPOSED = 24
EXPECTED_REMAINING = 21
DATA_SPLIT = "valid_train"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if claimed != stable_hash(body):
        raise ValueError(f"invalid {field}: {claimed}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen config: {args.output}")

    prior_config_path = REPO / (
        "configs/alfworld_unified_goal_acquisition_v13_formal.json"
    )
    prior_report_path = REPO / (
        "runs/alfworld_unified_goal_acquisition_v13_formal/report.json"
    )
    prior_audit_path = REPO / (
        "docs/results/alfworld_policy_contribution_v13_audit.json"
    )
    prior_config = _read(prior_config_path)
    prior_report = _read(prior_report_path)
    prior_audit = _read(prior_audit_path)
    _self_hash(prior_config, "config_sha256")
    _self_hash(prior_report, "report_sha256")
    _self_hash(prior_audit, "audit_sha256")
    if prior_audit["status"] != (
        "ALFWORLD_V13_PROGRAM_DRIVEN_POLICY_CONTRIBUTION_VALIDATED"
    ):
        raise SystemExit("V13 policy-contribution audit is not validated")

    dependencies = v11_freezer._dependency_paths(prior_config)
    for field, path in dependencies.items():
        if _sha(path) != prior_config.get(field):
            raise SystemExit(f"frozen V13 dependency changed: {path}")
    data_root = (
        Path(str(prior_config["alfworld_data"]))
        / "json_2.1.1" / DATA_SPLIT
    ).resolve()
    task_files = sorted(data_root.glob(
        "pick_two_obj_and_place-*/trial_*/game.tw-pddl"
    ))
    all_tasks = [str(path.relative_to(data_root)) for path in task_files]
    if len(all_tasks) != EXPECTED_POOL:
        raise SystemExit(
            f"expected immutable {EXPECTED_POOL}-task pool, found {len(all_tasks)}"
        )
    exposed = v13_freezer._historically_exposed(all_tasks)
    untouched = sorted(set(all_tasks) - exposed)
    if len(exposed) != EXPECTED_PREVIOUSLY_EXPOSED or len(untouched) != EXPECTED_REMAINING:
        raise SystemExit(
            "V14 reserve exposure boundary changed: "
            f"exposed={len(exposed)} untouched={len(untouched)}"
        )
    task_ids = v13_freezer.deterministic_reserve(
        untouched, seed=SELECTION_SEED, count=EXPECTED_REMAINING,
    )

    population_hash = stable_hash({
        "split": DATA_SPLIT,
        "all_task_ids": all_tasks,
        "historically_exposed_task_ids": sorted(exposed),
        "untouched_task_ids": untouched,
    })
    selection_hash = stable_hash({
        "algorithm": "ASCENDING_SHA256_OF_SEED_NUL_TASK_ID",
        "seed": SELECTION_SEED,
        "count": len(task_ids),
        "population_sha256": population_hash,
        "task_ids": task_ids,
    })
    launcher = REPO / "scripts/run_alfworld_program_driven_policy_v14.py"
    freezer = Path(__file__).resolve()
    contribution_module = REPO / (
        "src/motif_transfer/alfworld_policy_contribution.py"
    )
    v13_launcher = REPO / (
        "scripts/run_alfworld_unified_goal_acquisition_v13.py"
    )
    transport_config = REPO / str(
        prior_config["valid_train_transport_config"]
    )
    body = {
        "schema_version": "alfworld-program-driven-policy-config-v14",
        "status": V14_STATUS,
        "role": "prospective_untouched_policy_contribution_replication",
        "claim_boundary": (
            "All 21 ALFWorld valid_train multiplicity tasks that remained "
            "execution-untouched after V13. The source-induced symbolic IR "
            "must select an anonymous acquisition/relation option that changes "
            "the target policy before each rescued terminal transition. Only "
            "the frozen target-native neural grounder/executor may bind and "
            "emit concrete ALFWorld actions. V11 action logic, V13 transport, "
            "source programs, target grounder, controls, thresholds, horizon, "
            "and statistical endpoints are unchanged."
        ),
        "source_artifact": prior_config["source_artifact"],
        "source_confirmation": prior_config["source_confirmation"],
        "source_acquisition_artifact": prior_config[
            "source_acquisition_artifact"
        ],
        "source_acquisition_confirmation": prior_config[
            "source_acquisition_confirmation"
        ],
        "target_grounder": prior_config["target_grounder"],
        "target_causal_effect_artifact": prior_config[
            "target_causal_effect_artifact"
        ],
        "calibration_report": prior_config["calibration_report"],
        "calibration_counts": {
            "utility_vs_neural": {"wins": 7, "losses": 0, "ties": 17},
            "authenticity_vs_source_permuted": {
                "wins": 7, "losses": 0, "ties": 17,
            },
            "authority": "PROSPECTIVE_V13_PAIRED_OUTCOMES_ONLY",
            "current_reserve_outcome_read": False,
        },
        "conditions": prior_config["conditions"],
        "alfworld_config": prior_config["alfworld_config"],
        "alfworld_data": prior_config["alfworld_data"],
        "data_split": DATA_SPLIT,
        "seed": SELECTION_SEED,
        "max_steps": prior_config["max_steps"],
        "thresholds": prior_config["thresholds"],
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
            "selected_all_remaining_candidates": True,
            "population_sha256": population_hash,
            "selection_sha256": selection_hash,
        },
        "pre_freeze_run_exposure": {task_id: 0 for task_id in task_ids},
        "formal_target_outcome_read_for_freeze": False,
        "formal_reserve_task_opened": False,
        "gates": {
            "minimum_source_acquisition_groundings": len(task_ids),
            "minimum_source_divergent_actions": len(task_ids),
            "maximum_exact_two_sided_p": 0.05,
            "require_strict_source_control_superiority": True,
            "require_zero_negative_transfer": True,
            "require_target_native_ceiling_match": True,
            "require_every_rescue_causal_policy_bridge": True,
            "require_target_native_action_authority": True,
        },
        "output": (
            "runs/alfworld_program_driven_policy_v14_formal/report.json"
        ),
        "prior_v13_config": str(prior_config_path.relative_to(REPO)),
        "prior_v13_config_file_sha256": _sha(prior_config_path),
        "prior_v13_config_sha256": prior_config["config_sha256"],
        "prior_v13_report": str(prior_report_path.relative_to(REPO)),
        "prior_v13_report_file_sha256": _sha(prior_report_path),
        "prior_v13_report_sha256": prior_report["report_sha256"],
        "prior_v13_policy_audit": str(prior_audit_path.relative_to(REPO)),
        "prior_v13_policy_audit_file_sha256": _sha(prior_audit_path),
        "prior_v13_policy_audit_sha256": prior_audit["audit_sha256"],
        "v14_launcher_file_sha256": _sha(launcher),
        "v14_freezer_file_sha256": _sha(freezer),
        "policy_contribution_file_sha256": _sha(contribution_module),
        "v13_launcher_file_sha256": _sha(v13_launcher),
        "valid_train_transport_config": str(transport_config.relative_to(REPO)),
        "valid_train_transport_config_file_sha256": _sha(transport_config),
        **{field: _sha(path) for field, path in dependencies.items()},
    }
    config = body | {"config_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": V14_STATUS,
        "all_multiplicity_tasks": len(all_tasks),
        "historically_exposed_tasks": len(exposed),
        "selected_all_remaining_untouched_tasks": len(task_ids),
        "selection_sha256": selection_hash,
        "config_sha256": config["config_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
