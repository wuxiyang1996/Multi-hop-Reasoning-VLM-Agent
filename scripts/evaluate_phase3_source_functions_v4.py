#!/usr/bin/env python3
"""Evaluate frozen source-domain functions on a fourth untouched reserve."""

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
from motif_transfer.phase3_source_function_induction import (  # noqa: E402
    QUALIFIED,
    evaluate_source_function,
    function_trial_order,
    function_weights,
    maximum_source_function_contrast_derangement,
    validate_source_function_program,
)
from motif_transfer.phase3_source_induction import read_jsonl  # noqa: E402
from motif_transfer.phase3_typed_effect_induction import (  # noqa: E402
    TYPED_EFFECTS,
    typed_intervention_sets_from_rows,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "phase3-source-domain-function-reserve-manifest-v4"
    ):
        raise SystemExit("not a V4 source-function manifest")
    if plan_receipt.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise SystemExit("plan receipt belongs to another V4 manifest")
    if _sha(Path(__file__)) != plan_receipt["evaluator_file_sha256"]:
        raise SystemExit("V4 evaluator changed after plan freeze")

    artifacts = {
        row["source_game"]: _read(REPO / row["program_path"])
        for row in manifest["source_receipts"]
    }
    permutation = maximum_source_function_contrast_derangement(artifacts)
    plans = {row["source_game"]: row for row in plan_receipt["plans"]}
    totals = {
        "examples": 0, "authentic_correct": 0,
        "shuffled_correct": 0, "permuted_correct": 0,
    }
    lineages = []
    for receipt in manifest["source_receipts"]:
        game = str(receipt["source_game"])
        artifact = artifacts[game]
        _self_hash(artifact, "artifact_sha256")
        program = artifact["source_function_program"]
        validate_source_function_program(program)
        rows_path = args.run_dir / game / "rows.jsonl"
        if not rows_path.is_file():
            raise SystemExit(f"missing V4 reserve rows: {game}")
        rows = read_jsonl(rows_path)
        failed = sum(
            str(row.get("status")) != "INTERVENTION_OBSERVED" for row in rows
        )
        examples, audit = typed_intervention_sets_from_rows(
            rows, primary_horizon=int(receipt["primary_horizon"]),
        )
        weights = function_weights(program)
        authentic = evaluate_source_function(
            examples, weights=weights, source_split="heldout",
        )
        shuffled = evaluate_source_function(
            examples, weights=weights, source_split="heldout",
            shuffled_effects=True,
        )
        control_game = permutation[game]
        control_program = artifacts[control_game]["source_function_program"]
        permuted = evaluate_source_function(
            examples, weights=function_weights(control_program),
            source_split="heldout",
        )
        config = _read(REPO / receipt["config_path"])
        thresholds = config["source_function_reserve_gates"]
        planned = int(plans[game]["heldout_snapshots"])
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
            "zero_intervention_failed_rows": failed <= int(
                thresholds["maximum_intervention_failed_rows"]
            ),
            "explicit_state_action_effect_next_state_receipts": (
                audit["explicit_transition_tuple_receipts"] == len(rows)
            ),
            "program_body_not_updated_from_fourth_reserve": True,
            "legacy_canonical_attempt_program_disabled": artifact.get(
                "legacy_canonical_attempt_program_used"
            ) is False,
        }
        if program["status"] == QUALIFIED:
            gates.update({
                "qualified_accuracy_replicates": authentic["accuracy"] >= float(
                    thresholds["minimum_qualified_accuracy"]
                ),
                "qualified_function_varies": (
                    authentic["varying_effect_fraction"] >= float(
                        thresholds["minimum_qualified_varying_effect_fraction"]
                    )
                ),
                "authentic_beats_shuffled_effect_binding": (
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
                {effect_type: 0.2 for effect_type in TYPED_EFFECTS},
                {effect_type: 0.8 for effect_type in TYPED_EFFECTS},
            ]
            order, reason = function_trial_order(program, dummy)
            gates["unqualified_source_function_abstains"] = (
                order == () and reason == "SOURCE_DOMAIN_FUNCTION_NOT_QUALIFIED"
            )
        lineages.append({
            "source_game": game,
            "status": (
                "V4_SOURCE_DOMAIN_FUNCTION_CONFIRMED" if all(gates.values())
                else "V4_SOURCE_DOMAIN_FUNCTION_FAILED"
            ),
            "qualification_status": program["status"],
            "source_function_program_sha256": program["program_sha256"],
            "function_terms": program["source_function"]["terms"],
            "required_observation_horizon": program["source_function"][
                "required_observation_horizon"
            ],
            "retry_after_low": program["source_function"]["retry_after_low"],
            "transition_graph_sha256": stable_hash(program["transition_graph"]),
            "rows_file_sha256": _sha(rows_path),
            "heldout_planned": planned,
            "heldout_eligible": authentic["examples"],
            "heldout_eligible_fraction": eligible_fraction,
            "intervention_failed_rows": failed,
            "permuted_source_game": control_game,
            "permuted_source_function_program_sha256": control_program[
                "program_sha256"
            ],
            "evaluations": {
                "authentic": authentic,
                "shuffled_effect_binding": shuffled,
                "source_function_permuted": permuted,
            },
            "gates": gates,
        })

    total = totals["examples"]
    rates = {
        "authentic_accuracy": totals["authentic_correct"] / total if total else 0.0,
        "shuffled_accuracy": totals["shuffled_correct"] / total if total else 0.0,
        "source_permuted_accuracy": totals["permuted_correct"] / total if total else 0.0,
    }
    qualified = [row for row in lineages if row["qualification_status"] == QUALIFIED]
    abstaining = [row for row in lineages if row not in qualified]
    program_body_hashes = {
        stable_hash({
            "terms": row["function_terms"],
            "horizon": row["required_observation_horizon"],
            "retry": row["retry_after_low"],
            "graph": row["transition_graph_sha256"],
        }) for row in qualified
    }
    gates = {
        "exact_six_source_lineages": len(lineages) == 6,
        "qualified_and_abstaining_functions_confirmed": bool(qualified and abstaining),
        "all_six_applicability_decisions_confirmed": all(
            row["status"] == "V4_SOURCE_DOMAIN_FUNCTION_CONFIRMED"
            for row in lineages
        ),
        "qualified_program_bodies_are_source_specific": (
            len(program_body_hashes) == len(qualified) >= 3
        ),
        "qualified_retry_and_no_retry_graphs_confirmed": {
            bool(row["retry_after_low"]) for row in qualified
        } == {False, True},
        "qualified_authentic_aggregate_beats_shuffled": (
            rates["authentic_accuracy"] - rates["shuffled_accuracy"] >= 0.25
        ),
        "qualified_authentic_aggregate_beats_source_permuted": (
            rates["authentic_accuracy"] - rates["source_permuted_accuracy"] >= 0.15
        ),
        "programs_frozen_before_fourth_reserve": True,
        "legacy_canonical_attempt_program_disabled": True,
    }
    body = {
        "schema_version": "phase3-source-domain-function-reserve-report-v4",
        "status": (
            "SOURCE_SPECIFIC_DOMAIN_FUNCTIONS_VALIDATED"
            if all(gates.values()) else "SOURCE_SPECIFIC_DOMAIN_FUNCTIONS_FAILED"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "plan_receipt_sha256": plan_receipt["plan_receipt_sha256"],
        "source_function_permutation": permutation,
        "lineages": lineages,
        "qualified_aggregate": {**totals, **rates},
        "gates": gates,
        "claim_boundary": manifest["claim_boundary"],
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "qualified_aggregate": report["qualified_aggregate"],
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
