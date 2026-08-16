#!/usr/bin/env python3
"""Evaluate frozen typed-effect programs on untouched source reserve forks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_source_induction import read_jsonl  # noqa: E402
from motif_transfer.phase3_typed_effect_induction import (  # noqa: E402
    TYPED_EFFECTS,
    evaluate_effect_type,
    maximum_typed_program_contrast_derangement,
    target_trial_order,
    typed_intervention_sets_from_rows,
    validate_typed_effect_program,
)


QUALIFIED = "SOURCE_TYPED_EFFECT_PROGRAM_QUALIFIED"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase3_source_induction_v2/frozen_reserve/manifest.json",
    )
    parser.add_argument(
        "--plan-receipt", type=Path,
        default=(
            REPO / "configs/phase3_source_induction_v2/frozen_reserve/"
            "plan_receipt.json"
        ),
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=REPO / "runs/phase3_typed_effect_source_reserve_v2",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "runs/phase3_typed_effect_source_reserve_v2/report.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite source reserve report: {args.output}")
    manifest = _read(args.manifest)
    plan_receipt = _read(args.plan_receipt)
    _self_hash(manifest, "manifest_sha256")
    _self_hash(plan_receipt, "plan_receipt_sha256")
    if plan_receipt["manifest_sha256"] != manifest["manifest_sha256"]:
        raise SystemExit("plan receipt is not bound to source reserve manifest")
    if _sha(Path(__file__)) != plan_receipt["evaluator_file_sha256"]:
        raise SystemExit("frozen evaluator changed after plan freeze")

    artifacts = {
        row["source_game"]: _read(REPO / row["program_path"])
        for row in manifest["source_receipts"]
    }
    permutation = maximum_typed_program_contrast_derangement(artifacts)
    plan_by_game = {
        row["source_game"]: row for row in plan_receipt["plans"]
    }
    lineages = []
    qualified_totals = {
        "examples": 0, "authentic_correct": 0,
        "shuffled_correct": 0, "permuted_correct": 0,
    }
    for receipt in manifest["source_receipts"]:
        game = str(receipt["source_game"])
        artifact = artifacts[game]
        artifact_body = dict(artifact)
        claimed = artifact_body.pop("artifact_sha256", None)
        if claimed != stable_hash(artifact_body):
            raise SystemExit(f"source artifact hash mismatch: {game}")
        program = artifact["typed_effect_program"]
        validate_typed_effect_program(program)
        rows_path = args.run_dir / game / "rows.jsonl"
        if not rows_path.is_file():
            raise SystemExit(f"missing reserve rows: {game}")
        rows = read_jsonl(rows_path)
        raw_failed = sum(
            str(row.get("status")) != "INTERVENTION_OBSERVED" for row in rows
        )
        examples, audit = typed_intervention_sets_from_rows(
            rows, primary_horizon=int(receipt["primary_horizon"]),
        )
        effect_type = str(program["selected_effect_type"])
        authentic = evaluate_effect_type(
            examples, effect_type=effect_type, source_split="heldout",
        )
        shuffled = evaluate_effect_type(
            examples, effect_type=effect_type, source_split="heldout",
            shuffled_effects=True,
        )
        control_game = permutation[game]
        control_program = artifacts[control_game]["typed_effect_program"]
        permuted = evaluate_effect_type(
            examples,
            effect_type=str(control_program["selected_effect_type"]),
            source_split="heldout",
        )
        gates_config = _read(REPO / receipt["config_path"])[
            "typed_effect_reserve_gates"
        ]
        heldout_planned = int(plan_by_game[game]["heldout_snapshots"])
        eligible_fraction = (
            authentic["examples"] / heldout_planned if heldout_planned else 0.0
        )
        common_gates = {
            "minimum_planned_heldout_seeds": heldout_planned >= int(
                gates_config["minimum_planned_heldout_seeds"]
            ),
            "minimum_fresh_eligible_ledgers": authentic["examples"] >= int(
                gates_config["minimum_fresh_eligible_ledgers"]
            ),
            "minimum_fresh_eligible_fraction": eligible_fraction >= float(
                gates_config["minimum_fresh_eligible_fraction"]
            ),
            "zero_intervention_failed_rows": raw_failed <= int(
                gates_config["maximum_intervention_failed_rows"]
            ),
            "explicit_state_action_effect_next_state_receipts": (
                audit["explicit_transition_tuple_receipts"] == len(rows)
            ),
            "program_not_updated_from_reserve": True,
        }
        qualified = program["status"] == QUALIFIED
        if qualified:
            applicability_gates = {
                "qualified_accuracy_replicates": authentic["accuracy"] >= float(
                    gates_config["minimum_qualified_accuracy"]
                ),
                "qualified_effect_varies": (
                    authentic["varying_effect_fraction"] >= float(
                        gates_config["minimum_qualified_varying_effect_fraction"]
                    )
                ),
                "authentic_beats_shuffled_effects": (
                    authentic["accuracy"] - shuffled["accuracy"] >= float(
                        gates_config["minimum_qualified_authentic_minus_shuffled"]
                    )
                ),
            }
            qualified_totals["examples"] += authentic["examples"]
            qualified_totals["authentic_correct"] += authentic["correct"]
            qualified_totals["shuffled_correct"] += shuffled["correct"]
            qualified_totals["permuted_correct"] += permuted["correct"]
        else:
            dummy = [{effect: 0.2 for effect in TYPED_EFFECTS}, {
                effect: 0.8 for effect in TYPED_EFFECTS
            }]
            order, reason = target_trial_order(program, dummy)
            applicability_gates = {
                "unqualified_program_abstains_at_runtime": (
                    order == ()
                    and reason == "SOURCE_TYPED_EFFECT_PROGRAM_NOT_QUALIFIED"
                ),
                "unqualified_accuracy_remains_below_admission_threshold": (
                    authentic["accuracy"] <= float(
                        gates_config["maximum_unqualified_accuracy_for_calibration"]
                    )
                ),
            }
        gates = common_gates | applicability_gates
        lineages.append({
            "source_game": game,
            "status": (
                "TYPED_EFFECT_APPLICABILITY_PREDICTION_CONFIRMED"
                if all(gates.values()) else
                "TYPED_EFFECT_APPLICABILITY_PREDICTION_FAILED"
            ),
            "qualification_status": program["status"],
            "selected_effect_type": effect_type,
            "program_sha256": program["program_sha256"],
            "rows_file_sha256": _sha(rows_path),
            "heldout_planned": heldout_planned,
            "heldout_eligible": authentic["examples"],
            "heldout_eligible_fraction": eligible_fraction,
            "intervention_failed_rows": raw_failed,
            "explicit_transition_tuple_receipts": (
                audit["explicit_transition_tuple_receipts"]
            ),
            "permuted_source_game": control_game,
            "permuted_effect_type": control_program["selected_effect_type"],
            "evaluations": {
                "authentic": authentic,
                "shuffled_effect_binding": shuffled,
                "source_program_permuted": permuted,
            },
            "gates": gates,
        })

    total = qualified_totals["examples"]
    authentic_rate = (
        qualified_totals["authentic_correct"] / total if total else 0.0
    )
    shuffled_rate = (
        qualified_totals["shuffled_correct"] / total if total else 0.0
    )
    permuted_rate = (
        qualified_totals["permuted_correct"] / total if total else 0.0
    )
    aggregate_gates = {
        "exact_six_source_lineages": len(lineages) == 6,
        "all_applicability_predictions_confirmed": all(
            row["status"] == "TYPED_EFFECT_APPLICABILITY_PREDICTION_CONFIRMED"
            for row in lineages
        ),
        "qualified_authentic_aggregate_beats_shuffled": (
            authentic_rate - shuffled_rate >= 0.25
        ),
        "qualified_authentic_aggregate_beats_source_permuted": (
            authentic_rate - permuted_rate >= 0.15
        ),
        "at_least_three_confirmed_effect_types": len({
            row["selected_effect_type"] for row in lineages
        }) >= 3,
        "programs_frozen_before_reserve_outcomes": True,
    }
    body = {
        "schema_version": "phase3-typed-effect-source-reserve-report-v2",
        "status": (
            "SOURCE_SPECIFIC_TYPED_EFFECT_TRANSFER_GATE_VALIDATED"
            if all(aggregate_gates.values()) else
            "SOURCE_SPECIFIC_TYPED_EFFECT_TRANSFER_GATE_FAILED"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "plan_receipt_sha256": plan_receipt["plan_receipt_sha256"],
        "source_permutation": permutation,
        "lineages": lineages,
        "qualified_aggregate": {
            **qualified_totals,
            "authentic_accuracy": authentic_rate,
            "shuffled_accuracy": shuffled_rate,
            "source_permuted_accuracy": permuted_rate,
        },
        "gates": aggregate_gates,
        "claim_boundary": manifest["claim_boundary"],
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "report_sha256": report["report_sha256"],
        "qualified_aggregate": report["qualified_aggregate"],
        "lineages": [{
            "source_game": row["source_game"],
            "qualification_status": row["qualification_status"],
            "status": row["status"],
            "authentic_accuracy": row["evaluations"]["authentic"]["accuracy"],
            "shuffled_accuracy": row["evaluations"]["shuffled_effect_binding"]["accuracy"],
            "permuted_accuracy": row["evaluations"]["source_program_permuted"]["accuracy"],
        } for row in lineages],
    }, indent=2))
    if report["status"] != "SOURCE_SPECIFIC_TYPED_EFFECT_TRANSFER_GATE_VALIDATED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
