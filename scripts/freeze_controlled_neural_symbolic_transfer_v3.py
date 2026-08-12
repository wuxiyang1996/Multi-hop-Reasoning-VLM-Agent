#!/usr/bin/env python3
"""Freeze development or formal configs for neural-grounded transfer V3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "file_sha256": _sha256(path)}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("development", "formal"), required=True)
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen V3 config: {args.output}")
    development_receipt = None
    if args.role == "formal":
        if args.development_report is None:
            raise SystemExit("formal freeze requires --development-report")
        development = _read(args.development_report)
        if development.get("status") != "SUPPORTED" or not development.get(
            "gate", {}
        ).get("passed"):
            raise SystemExit("V3 development reproduction did not pass")
        development_receipt = _receipt(args.development_report) | {
            "status": str(development["status"]),
            "gate_passed": True,
            "config_sha256": str(development["config"]["content"]["config_sha256"]),
        }

    core = REPO / "src/motif_transfer/controlled_exploration_transfer.py"
    runner = REPO / "scripts/run_controlled_exploration_transfer_v1.py"
    if args.role == "development":
        splits = {
            "neural_development_a": list(range(28400, 28416)),
            "neural_development_b": list(range(28500, 28516)),
        }
        support_seeds = list(range(28300, 28304))
        status = "FROZEN_BEFORE_REPRODUCTION_OF_CONSUMED_NEURAL_DEVELOPMENT"
    else:
        splits = {
            "neural_qualification": list(range(29100, 29116)),
            "neural_held_out": list(range(29200, 29216)),
        }
        support_seeds = list(range(28900, 28904))
        status = "FROZEN_BEFORE_NEURAL_GROUNDED_FORMAL_RUN"
    split_names = list(splits)
    controls = [
        "target_only", "shuffled_source_plus_target",
        "source_marginal_plus_target",
    ]
    body: dict[str, Any] = {
        "schema_version": 3,
        "status": status,
        "claim_boundary": (
            "CONTROLLED_MECHANISM_VALIDATION_ONLY: SYNTHETIC_INTERVENTION_"
            "RICH_HIDDEN_RULE_GAMES_TO_SEMANTICALLY_DISJOINT_SYNTHETIC_"
            "DIAGNOSIS; TARGET_NATIVE_NEURAL_GROUNDER_FITTED_ONLY_FROM_"
            "ANONYMOUS_TARGET_CALIBRATION_OUTCOMES; NO_REAL_ARCADE_TO_"
            "ALFWORLD_OR_WEBSHOP_GENERALIZATION_CLAIM"
        ),
        "freeze_record": {
            "role": args.role,
            "formal_target_seeds_seen_during_selection": False,
            "development_target_seeds_previously_used_for_selection": (
                args.role == "development"
            ),
            "selected_from_development_only": {
                "source_train_domains": 48,
                "source_states_per_domain": 24,
                "target_calibration_samples_per_cell": 48,
                "target_grounder_hidden_units": 32,
                "target_grounder_epochs": 1400,
                "paired_episodes_per_target_domain": 128,
            },
            "development_report": development_receipt,
        },
        "implementation": {
            "freezer": _receipt(Path(__file__)),
            "runner": _receipt(runner),
            "core": _receipt(core),
        },
        "domain": {
            "hypothesis_count": 4,
            "test_count": 5,
            "max_tests": 4,
            "test_cost": 0.025,
            "calibration_samples_per_cell": 48,
            "calibration_beta_prior": 1.5,
            "target_grounder": {
                "kind": "target_native_mlp",
                "hidden_units": 32,
                "epochs": 1400,
                "learning_rate": 0.03,
                "l2": 0.0001,
                "input_contract": (
                    "TARGET_LOCAL_ANONYMOUS_HYPOTHESIS_AND_TEST_ONE_HOTS"
                ),
                "label_contract": "TARGET_CALIBRATION_BINARY_OUTCOMES_ONLY",
            },
        },
        "source": {
            "train_domain_seeds": list(range(28100, 28148)),
            "evaluation_domain_seeds": list(range(28200, 28208)),
            "states_per_domain": 24,
            "transferred_structure": (
                "STATE_DEPENDENT_TEST_VS_COMMIT_MATCHED_INTERVENTION_VALUES"
            ),
        },
        "target": {
            "support_domain_seeds": support_seeds,
            "support_states_per_domain": 1,
            "support_k": [0],
            "evaluation_domain_seeds": splits,
            "episode_seeds": list(range(128)),
        },
        "model": {
            "kind": "source_prior_target_residual_ensemble",
            "seed": 73101,
            "control_seed": 73102,
            "ensemble_size": 9,
            "ridge_alpha": 0.5,
            "target_mass": 1.0,
            "residual_ridge_alpha": 2.0,
            "maximum_residual_scale": 1.0,
            "residual_full_strength_states": 4,
        },
        "policy": {
            "uncertainty_scale": 0.5,
            "decision_margin": 0.0025,
            "fallback_commit_threshold": 0.72,
        },
        "gate": {
            "bootstrap_samples": 3000,
            "requirements": [
                {
                    "name": "paired_net_return_superiority",
                    "metric": "net_return",
                    "required_splits": split_names,
                    "required_k": [0],
                    "controls": controls,
                    "minimum_mean_net_return_delta": 0.005,
                    "minimum_ci95_net_return_delta": 0.0,
                },
                {
                    "name": "paired_success_rate_superiority",
                    "metric": "success",
                    "required_splits": split_names,
                    "required_k": [0],
                    "controls": controls,
                    "minimum_mean_success_delta": 0.005,
                    "minimum_ci95_success_delta": 0.0,
                },
            ],
            "invariants": {
                "require_zero_shared_raw_tokens": True,
                "require_target_grounder_kind": "target_native_mlp",
                "maximum_target_grounder_mse": 0.01,
                "source_authentic_mse_strictly_less_than": [
                    "shuffled", "marginal",
                ],
                "minimum_source_train_examples": 9000,
            },
        },
    }
    config = body | {"config_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "file_sha256": _sha256(args.output),
        "config_sha256": config["config_sha256"],
        "status": status,
        "role": args.role,
        "target_splits": {name: len(seeds) for name, seeds in splits.items()},
        "formal_target_seeds_seen_during_selection": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
