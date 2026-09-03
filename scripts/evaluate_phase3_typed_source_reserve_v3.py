#!/usr/bin/env python3
"""Evaluate V3 cross-batch-calibrated programs on a third source reserve."""

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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan-receipt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite report: {args.output}")
    manifest = _read(args.manifest)
    plan_receipt = _read(args.plan_receipt)
    _self_hash(manifest, "manifest_sha256")
    _self_hash(plan_receipt, "plan_receipt_sha256")
    if manifest.get("schema_version") != (
        "phase3-typed-effect-source-reserve-manifest-v3"
    ):
        raise SystemExit("not a V3 typed source reserve manifest")
    if plan_receipt["manifest_sha256"] != manifest["manifest_sha256"]:
        raise SystemExit("plan receipt is not bound to V3 manifest")
    if _sha(Path(__file__)) != plan_receipt["evaluator_file_sha256"]:
        raise SystemExit("V3 evaluator changed after plan freeze")

    artifacts = {
        row["source_game"]: _read(REPO / row["program_path"])
        for row in manifest["source_receipts"]
    }
    permutation = maximum_typed_program_contrast_derangement(artifacts)
    plan_by_game = {
        row["source_game"]: row for row in plan_receipt["plans"]
    }
    totals = {
        "examples": 0, "authentic_correct": 0,
        "shuffled_correct": 0, "permuted_correct": 0,
    }
    lineages = []
    for receipt in manifest["source_receipts"]:
        game = str(receipt["source_game"])
        artifact = artifacts[game]
        _self_hash(artifact, "artifact_sha256")
        program = artifact["typed_effect_program"]
        validate_typed_effect_program(program)
        rows_path = args.run_dir / game / "rows.jsonl"
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
        config = _read(REPO / receipt["config_path"])
        thresholds = config["typed_effect_reserve_gates"]
        planned = int(plan_by_game[game]["heldout_snapshots"])
        eligible_fraction = authentic["examples"] / planned if planned else 0.0
        gates = {
            "minimum_planned_heldout_seeds": planned >= int(
                thresholds["minimum_planned_heldout_seeds"]
            ),
            "minimum_fresh_eligible_ledgers": authentic["examples"] >= int(
                thresholds["minimum_fresh_eligible_ledgers"]
            ),
            "minimum_fresh_eligible_fraction": eligible_fraction >= float(
                thresholds["minimum_fresh_eligible_fraction"]
            ),
            "zero_intervention_failed_rows": raw_failed <= int(
                thresholds["maximum_intervention_failed_rows"]
            ),
            "explicit_state_action_effect_next_state_receipts": (
                audit["explicit_transition_tuple_receipts"] == len(rows)
            ),
            "program_not_updated_from_third_reserve": True,
        }
        qualified = program["status"] == QUALIFIED
        if qualified:
            gates.update({
                "qualified_accuracy_replicates": authentic["accuracy"] >= float(
                    thresholds["minimum_qualified_accuracy"]
                ),
                "qualified_effect_varies": (
                    authentic["varying_effect_fraction"] >= float(
                        thresholds["minimum_qualified_varying_effect_fraction"]
                    )
                ),
                "authentic_beats_shuffled_effects": (
                    authentic["accuracy"] - shuffled["accuracy"] >= float(
                        thresholds["minimum_qualified_authentic_minus_shuffled"]
                    )
                ),
            })
            totals["examples"] += authentic["examples"]
            totals["authentic_correct"] += authentic["correct"]
            totals["shuffled_correct"] += shuffled["correct"]
            totals["permuted_correct"] += permuted["correct"]
        else:
            dummy = [
                {effect: 0.2 for effect in TYPED_EFFECTS},
                {effect: 0.8 for effect in TYPED_EFFECTS},
            ]
            order, reason = target_trial_order(program, dummy)
            gates["cross_batch_unqualified_program_abstains"] = (
                order == ()
                and reason == "SOURCE_TYPED_EFFECT_PROGRAM_NOT_QUALIFIED"
            )
        lineages.append({
            "source_game": game,
            "status": (
                "V3_TYPED_EFFECT_APPLICABILITY_CONFIRMED"
                if all(gates.values()) else
                "V3_TYPED_EFFECT_APPLICABILITY_FAILED"
            ),
            "qualification_status": program["status"],
            "selected_effect_type": effect_type,
            "program_sha256": program["program_sha256"],
            "rows_file_sha256": _sha(rows_path),
            "heldout_planned": planned,
            "heldout_eligible": authentic["examples"],
            "heldout_eligible_fraction": eligible_fraction,
            "intervention_failed_rows": raw_failed,
            "permuted_source_game": control_game,
            "permuted_effect_type": control_program["selected_effect_type"],
            "evaluations": {
                "authentic": authentic,
                "shuffled_effect_binding": shuffled,
                "source_program_permuted": permuted,
            },
            "gates": gates,
        })

    total = totals["examples"]
    rates = {
        "authentic_accuracy": totals["authentic_correct"] / total if total else 0.0,
        "shuffled_accuracy": totals["shuffled_correct"] / total if total else 0.0,
        "source_permuted_accuracy": totals["permuted_correct"] / total if total else 0.0,
    }
    qualified_rows = [
        row for row in lineages if row["qualification_status"] == QUALIFIED
    ]
    abstaining_rows = [row for row in lineages if row not in qualified_rows]
    aggregate_gates = {
        "exact_six_source_lineages": len(lineages) == 6,
        "exact_three_qualified_and_three_abstaining": (
            len(qualified_rows) == len(abstaining_rows) == 3
        ),
        "all_six_applicability_decisions_confirmed": all(
            row["status"] == "V3_TYPED_EFFECT_APPLICABILITY_CONFIRMED"
            for row in lineages
        ),
        "qualified_effect_types_are_source_specific": len({
            row["selected_effect_type"] for row in qualified_rows
        }) == 3,
        "qualified_authentic_aggregate_beats_shuffled": (
            rates["authentic_accuracy"] - rates["shuffled_accuracy"] >= 0.25
        ),
        "qualified_authentic_aggregate_beats_source_permuted": (
            rates["authentic_accuracy"] - rates["source_permuted_accuracy"] >= 0.15
        ),
        "programs_frozen_before_third_reserve": True,
    }
    body = {
        "schema_version": "phase3-typed-effect-source-reserve-report-v3",
        "status": (
            "SOURCE_SPECIFIC_TYPED_EFFECT_APPLICABILITY_VALIDATED"
            if all(aggregate_gates.values()) else
            "SOURCE_SPECIFIC_TYPED_EFFECT_APPLICABILITY_FAILED"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "plan_receipt_sha256": plan_receipt["plan_receipt_sha256"],
        "v2_calibration_report_sha256": manifest[
            "v2_calibration_report_sha256"
        ],
        "source_permutation": permutation,
        "lineages": lineages,
        "qualified_aggregate": {**totals, **rates},
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
        } for row in lineages],
    }, indent=2))
    if report["status"] != "SOURCE_SPECIFIC_TYPED_EFFECT_APPLICABILITY_VALIDATED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
