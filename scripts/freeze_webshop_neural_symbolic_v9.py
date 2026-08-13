#!/usr/bin/env python3
"""Fit and freeze the V9 target-native grounder before confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402
from motif_transfer.webshop_neural_symbolic_v9 import (  # noqa: E402
    OUTCOME_NAMES,
    OutcomeRow,
    fit_target_outcome_mlp,
)


def _rows(report: dict) -> list[OutcomeRow]:
    return [
        OutcomeRow(
            tuple(map(float, row["features"])),
            tuple(map(float, row["outcomes"])),
        )
        for sequence in report["sequences"]
        for row in sequence["rows"]
    ]


def _metrics(model, rows: list[OutcomeRow]) -> dict:
    labels = np.asarray([row.outcomes for row in rows], dtype=np.float64)
    predictions = model.predict([row.features for row in rows])
    return {
        "rows": len(rows),
        "mse_by_outcome": {
            name: float(np.mean((predictions[:, index] - labels[:, index]) ** 2))
            for index, name in enumerate(OUTCOME_NAMES)
        },
        "classification_accuracy_by_outcome": {
            name: float(np.mean((predictions[:, index] >= 0.5) == (labels[:, index] >= 0.5)))
            for index, name in enumerate(OUTCOME_NAMES)
        },
    }


def _find_row(report: dict, task_id: str, step: int) -> dict:
    matches = [
        row
        for sequence in report["sequences"]
        for row in sequence["rows"]
        if row["task_id"] == task_id and int(row["step"]) == step
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {task_id} step {step} row, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/webshop_neural_symbolic_v9_adaptation.json",
    )
    parser.add_argument(
        "--adaptation-rows", type=Path,
        default=REPO / "runs/webshop_neurosymbolic_applicability_v9/adaptation/grounding_rows.json",
    )
    parser.add_argument(
        "--calibration-rows", type=Path,
        default=REPO / "runs/webshop_neurosymbolic_applicability_v9/calibration/grounding_rows.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "runs/webshop_neurosymbolic_applicability_v9/frozen_grounder.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    adaptation_report = json.loads(args.adaptation_rows.read_text())
    calibration_report = json.loads(args.calibration_rows.read_text())
    if adaptation_report["metrics"]["failures"] or not adaptation_report["metrics"][
        "all_prefix_states_match"
    ]:
        raise SystemExit("adaptation grounding receipts are invalid")
    if calibration_report["metrics"]["failures"] or not calibration_report["metrics"][
        "all_prefix_states_match"
    ]:
        raise SystemExit("calibration grounding receipts are invalid")
    train_rows = _rows(adaptation_report)
    calibration_rows = _rows(calibration_report)
    grounder_config = config["target_grounder"]
    model = fit_target_outcome_mlp(
        train_rows,
        seed=int(grounder_config["seed"]),
        hidden_units=int(grounder_config["hidden_units"]),
        epochs=int(grounder_config["epochs"]),
        learning_rate=float(grounder_config["learning_rate"]),
        l2=float(grounder_config["l2"]),
    )
    paired = _find_row(calibration_report, "webshop.12", 5)
    commit = _find_row(calibration_report, "webshop.12", 6)
    critical_predictions = model.predict((paired["features"], commit["features"]))
    progress_index = OUTCOME_NAMES.index("prerequisite_progress")
    changed_index = OUTCOME_NAMES.index("state_changed")
    reward_index = OUTCOME_NAMES.index("reward")
    terminated_index = OUTCOME_NAMES.index("terminated")
    critical = {
        "paired_constraint_test": {
            "actual": paired["outcomes"],
            "predicted": critical_predictions[0].tolist(),
            "passes": bool(
                critical_predictions[0, progress_index] > 0.5
                and critical_predictions[0, changed_index] > 0.5
            ),
        },
        "satisfied_constraint_commit": {
            "actual": commit["outcomes"],
            "predicted": critical_predictions[1].tolist(),
            "passes": bool(
                critical_predictions[1, reward_index] > 0.5
                and critical_predictions[1, terminated_index] > 0.5
            ),
        },
    }
    preflight_passed = bool(
        len(train_rows) >= 18
        and all(row["passes"] for row in critical.values())
    )
    source_path = REPO / config["source_config"]
    artifact = {
        "schema_version": 1,
        "artifact_role": "FROZEN_WEBSHOP_V9_TARGET_NATIVE_GROUNDER",
        "claim_limit": "Fit on adaptation only; checked on calibration; confirmation unread.",
        "preflight_passed": preflight_passed,
        "grounder": model.as_dict(),
        "training_config": grounder_config,
        "policy": config["policy"],
        "conditions": config["conditions"],
        "metrics": {
            "adaptation": _metrics(model, train_rows),
            "calibration": _metrics(model, calibration_rows),
            "critical_calibration_predictions": critical,
        },
        "source_contract": {
            "config": str(source_path),
            "config_sha256": file_sha256(source_path),
            "transferred_structure": "STATE_DEPENDENT_TEST_VS_COMMIT_MATCHED_INTERVENTION_VALUES",
            "source_receives_target_tokens": False,
        },
        "runtime_hashes": {
            "config": file_sha256(args.config),
            "adaptation_rows": file_sha256(args.adaptation_rows),
            "calibration_rows": file_sha256(args.calibration_rows),
            "core": file_sha256(REPO / "src/motif_transfer/webshop_neural_symbolic_v9.py"),
            "freezer": file_sha256(Path(__file__)),
        },
        "confirmation_read_or_run": False,
        "held_out_read_or_run": False,
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "preflight_passed": preflight_passed,
        "adaptation_rows": len(train_rows),
        "calibration_rows": len(calibration_rows),
        "critical": critical,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
