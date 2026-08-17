#!/usr/bin/env python3
"""Prepare and gate source-induced WebShop structural transfer V17."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

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
from motif_transfer.webshop_structural_transfer_v17 import (  # noqa: E402
    induce_webshop_relational_function,
    permute_target_terminal,
    structural_compatibility_receipt,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _outcome_rows(payload: dict[str, Any]) -> list[tuple[str, int, OutcomeRow]]:
    output = []
    for sequence in payload["sequences"]:
        for row in sequence["rows"]:
            output.append((
                str(row["task_id"]),
                int(row["step"]),
                OutcomeRow(
                    tuple(map(float, row["features"])),
                    tuple(map(float, row["outcomes"])),
                ),
            ))
    return sorted(
        output,
        key=lambda row: stable_hash({"task_id": row[0], "step": row[1]}),
    )


def _curve(
    adaptation: dict[str, Any], calibration: dict[str, Any],
) -> tuple[list[dict[str, Any]], int | None]:
    train = _outcome_rows(adaptation)
    heldout = _outcome_rows(calibration)
    labels = np.asarray([row[2].outcomes for row in heldout], dtype=np.float64)
    curve = []
    for size in (2, 4, 8, 12, len(train)):
        if size > len(train) or any(row["rows"] == size for row in curve):
            continue
        model = fit_target_outcome_mlp(
            [row[2] for row in train[:size]],
            seed=91301,
            hidden_units=12,
            epochs=2200,
            learning_rate=0.02,
            l2=0.01,
        )
        predicted = model.predict([row[2].features for row in heldout])
        accuracy = {
            name: float(np.mean(
                (predicted[:, index] >= 0.5) == (labels[:, index] >= 0.5)
            ))
            for index, name in enumerate(OUTCOME_NAMES)
        }
        mse = {
            name: float(np.mean((predicted[:, index] - labels[:, index]) ** 2))
            for index, name in enumerate(OUTCOME_NAMES)
        }
        qualifies = (
            accuracy["state_changed"] >= 0.75
            and accuracy["prerequisite_progress"] >= 0.75
        )
        curve.append({
            "rows": size,
            "calibration_rows": len(heldout),
            "classification_accuracy": accuracy,
            "mse": mse,
            "grounding_gate_passed": qualifies,
            "grounder": model.as_dict(),
        })
    eligible = [row["rows"] for row in curve if row["grounding_gate_passed"]]
    return curve, min(eligible) if eligible else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-report", type=Path,
        default=REPO / (
            "runs/webshop_search_automaton_v16_development_gpt41mini_8tasks/"
            "report.json"
        ),
    )
    parser.add_argument(
        "--source-artifact", type=Path,
        default=REPO / "runs/sokoban_relational_structural_v2/artifact.json",
    )
    parser.add_argument(
        "--source-confirmation", type=Path,
        default=REPO / (
            "runs/sokoban_relational_structural_v2/fresh_confirmation_report.json"
        ),
    )
    parser.add_argument(
        "--target-grounder", type=Path,
        default=REPO / "docs/results/webshop_neural_symbolic_v9_frozen_grounder.json",
    )
    parser.add_argument(
        "--adaptation-rows", type=Path,
        default=REPO / (
            "runs/webshop_neurosymbolic_applicability_v9/adaptation/"
            "grounding_rows.json"
        ),
    )
    parser.add_argument(
        "--calibration-rows", type=Path,
        default=REPO / (
            "runs/webshop_neurosymbolic_applicability_v9/calibration/"
            "grounding_rows.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/webshop_structural_transfer_v17_development",
    )
    args = parser.parse_args()

    development = _read(args.development_report)
    source = _read(args.source_artifact)
    source_confirmation = _read(args.source_confirmation)
    target_grounder = _read(args.target_grounder)
    curve, minimum_rows = _curve(
        _read(args.adaptation_rows), _read(args.calibration_rows),
    )
    selected_curve_row = next(
        (row for row in curve if row["rows"] == minimum_rows), None
    )
    low_sample_body = {
        "schema_version": "webshop-target-native-low-sample-grounder-v17",
        "status": (
            "TARGET_NATIVE_LOW_SAMPLE_GROUNDER_QUALIFIED"
            if selected_curve_row is not None else
            "TARGET_NATIVE_LOW_SAMPLE_GROUNDER_FAILED"
        ),
        "training_rows": minimum_rows,
        "selection_rule": (
            "minimum rows with held-out state-change and prerequisite-progress "
            "classification accuracy both >= 0.75"
        ),
        "calibration_metrics": (
            {key: value for key, value in selected_curve_row.items()
             if key != "grounder"}
            if selected_curve_row is not None else None
        ),
        "grounder": (
            selected_curve_row["grounder"]
            if selected_curve_row is not None else None
        ),
        "formal_target_data_read": False,
    }
    low_sample_grounder = low_sample_body | {
        "artifact_sha256": stable_hash(low_sample_body)
    }
    manifest = _read(REPO / "configs/webshop_synthetic_unique_v14_frozen.json")
    role = {row["task_id"]: row for row in manifest["roles"]["development"]}
    receipts = []
    by_condition: dict[str, dict[str, dict[str, Any]]] = {
        "raw_target_only": {}, "target_native_search_ceiling": {},
    }
    receipt_root = args.development_report.parent
    for task_id in development["tasks"]:
        for condition in by_condition:
            path = receipt_root / f"{task_id}.{condition}.json"
            row = _read(path)
            by_condition[condition][task_id] = row
            if condition == "target_native_search_ceiling":
                receipts.append(row)
    receipt_lineage = stable_hash([row["receipt_sha256"] for row in receipts])
    target = induce_webshop_relational_function(
        receipts,
        development_receipts_sha256=receipt_lineage,
        target_grounder_sha256=str(low_sample_grounder["artifact_sha256"]),
    )
    authentic = structural_compatibility_receipt(
        source, source_confirmation, target,
    )
    permuted = structural_compatibility_receipt(
        source, source_confirmation, permute_target_terminal(target),
    )

    applicable = [
        task_id for task_id in development["tasks"]
        if (role[task_id].get("goal") or {}).get("goal_options")
    ]
    wins = losses = 0
    for task_id in applicable:
        raw = bool(by_condition["raw_target_only"][task_id]["strict_success"])
        ceiling = bool(
            by_condition["target_native_search_ceiling"][task_id]["strict_success"]
        )
        wins += ceiling and not raw
        losses += raw and not ceiling
    gates = {
        "target_domain_function_induced_from_development_only": (
            target["status"] == "TARGET_RELATIONAL_FUNCTION_QUALIFIED"
            and target["source_program_read_during_induction"] is False
            and target["formal_target_data_read"] is False
        ),
        "fresh_source_artifact_confirmed": authentic["gates"][
            "source_fresh_confirmed"
        ],
        "authentic_structural_ir_admitted": (
            authentic["status"] == "STRUCTURAL_TRANSFER_ADMITTED"
        ),
        "terminal_permutation_rejected": (
            permuted["status"] == "STRUCTURAL_TRANSFER_ABSTAINED"
        ),
        "target_function_has_development_success_gain": wins >= 2 and losses == 0,
        "low_sample_neural_grounder_qualified": minimum_rows is not None,
        "no_named_policy_template": target["named_policy_template_used"] is False,
        "formal_v17_outcomes_unread": True,
    }
    body = {
        "schema_version": "webshop-source-structural-transfer-v17-development-v1",
        "status": (
            "PHASE4_WEBSHOP_DEVELOPMENT_GATE_PASSED" if all(gates.values())
            else "PHASE4_WEBSHOP_DEVELOPMENT_GATE_FAILED"
        ),
        "claim_boundary": (
            "Consumed V14 development receipts only; V14 formal is historical and "
            "a new V17 formal reserve has not been created or opened by this script."
        ),
        "source_artifact_sha256": source["artifact_sha256"],
        "source_confirmation_report_sha256": source_confirmation["report_sha256"],
        "target_program_sha256": target["program_sha256"],
        "target_program_development_receipts_sha256": receipt_lineage,
        "structural_compatibility": authentic,
        "terminal_permutation_control": permuted,
        "development_utility": {
            "structurally_applicable_tasks": len(applicable),
            "target_domain_function_vs_neural_wins": wins,
            "target_domain_function_vs_neural_losses": losses,
            "ties": len(applicable) - wins - losses,
        },
        "target_grounder_sample_efficiency": {
            "selection_rule": (
                "minimum rows with held-out state-change and prerequisite-progress "
                "classification accuracy both >= 0.75"
            ),
            "minimum_qualified_rows": minimum_rows,
            "curve": curve,
        },
        "gates": gates,
        "formal_v17_outcomes_read_or_run": False,
        "runtime_hashes": {
            "preparer": file_sha256(Path(__file__)),
            "core": file_sha256(
                REPO / "src/motif_transfer/webshop_structural_transfer_v17.py"
            ),
            "source_artifact": file_sha256(args.source_artifact),
            "source_confirmation": file_sha256(args.source_confirmation),
            "target_grounder": file_sha256(args.target_grounder),
            "development_report": file_sha256(args.development_report),
            "adaptation_rows": file_sha256(args.adaptation_rows),
            "calibration_rows": file_sha256(args.calibration_rows),
        },
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "target_function.json").write_text(
        json.dumps(target, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "low_sample_grounder.json").write_text(
        json.dumps(low_sample_grounder, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "development_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "development_utility": report["development_utility"],
        "minimum_qualified_grounder_rows": minimum_rows,
        "gates": gates,
        "output": str(args.output_dir),
    }, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
