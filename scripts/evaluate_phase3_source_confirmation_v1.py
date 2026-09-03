#!/usr/bin/env python3
"""Evaluate frozen anonymous programs on fresh source confirmation forks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_source_induction import (  # noqa: E402
    execute_program_on_ledgers,
    file_sha256,
    load_source_ledgers,
    validate_program,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = _read(args.manifest)
    _self_hash(manifest, "manifest_sha256")
    if manifest.get("status") != "FROZEN_BEFORE_FRESH_SOURCE_PLAN_OR_OUTCOME":
        raise SystemExit("source confirmation manifest is not frozen")

    lineages = []
    for receipt in manifest["source_receipts"]:
        game = str(receipt["source_game"])
        config_path = REPO / str(receipt["config_path"])
        program_path = REPO / str(receipt["program_path"])
        if file_sha256(config_path) != receipt["config_file_sha256"]:
            raise SystemExit(f"frozen source config changed: {game}")
        if file_sha256(program_path) != receipt["program_file_sha256"]:
            raise SystemExit(f"frozen source program changed: {game}")
        config = _read(config_path)
        artifact = _read(program_path)
        _self_hash(artifact, "artifact_sha256")
        authentic = artifact["authentic_program"]
        shuffled = artifact["shuffled_effect_program"]
        validate_program(authentic)
        validate_program(shuffled)

        plan_path = args.run_dir / game / "plan.json"
        rows_path = args.run_dir / game / "rows.jsonl"
        if not plan_path.is_file() or not rows_path.is_file():
            raise SystemExit(f"missing fresh source confirmation files: {game}")
        plan = _read(plan_path)
        plan_body = dict(plan)
        plan_claimed = plan_body.pop("plan_sha256", None)
        if plan_claimed != stable_hash(plan_body):
            raise SystemExit(f"fresh source plan hash mismatch: {game}")
        planned_seeds = {int(row["seed"]) for row in plan["snapshots"]}
        if planned_seeds != set(map(int, config["seeds"])):
            raise SystemExit(f"fresh source plan seeds differ from config: {game}")
        heldout_planned = sum(
            str(row["split"]) == "heldout" for row in plan["snapshots"]
        )
        ledgers, audit = load_source_ledgers(
            rows_path, primary_horizon=int(config["horizon"]),
        )
        authentic_execution = execute_program_on_ledgers(
            authentic, ledgers, source_split="heldout",
        )
        learned_shuffled_execution = execute_program_on_ledgers(
            shuffled, ledgers, source_split="heldout",
        )
        runtime_shuffled_execution = execute_program_on_ledgers(
            authentic,
            ledgers,
            source_split="heldout",
            shuffled_runtime_effect_binding=True,
        )
        gates_config = config["phase3_confirmation_gates"]
        # Rows excluded for value ties are not infrastructure failures.  The
        # source loader rejects bad hashes and the raw row status audit below
        # counts only true intervention failures.
        raw_failed = 0
        with rows_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip() and json.loads(line).get("status") != "INTERVENTION_OBSERVED":
                    raw_failed += 1
        eligible = authentic_execution["ledgers"]
        eligible_fraction = eligible / heldout_planned if heldout_planned else 0.0
        destructive_max = float(
            gates_config["maximum_each_destructive_control_success_rate"]
        )
        gates = {
            "planned_heldout_seed_count": heldout_planned >= int(
                gates_config["minimum_planned_heldout_seeds"]
            ),
            "fresh_eligible_ledger_count": eligible >= int(
                gates_config["minimum_fresh_eligible_ledgers"]
            ),
            "fresh_eligible_fraction": eligible_fraction >= float(
                gates_config["minimum_fresh_eligible_fraction"]
            ),
            "zero_intervention_failed_rows": raw_failed <= int(
                gates_config["maximum_intervention_failed_rows"]
            ),
            "authentic_closed_loop_success": (
                authentic_execution["success_rate"] >= float(
                    gates_config["minimum_authentic_closed_loop_success_rate"]
                )
            ),
            "authentic_strictly_beats_frozen_shuffled_program": (
                authentic_execution["success_rate"]
                > learned_shuffled_execution["success_rate"]
                and learned_shuffled_execution["success_rate"] <= destructive_max
            ),
            "authentic_strictly_beats_runtime_effect_permutation": (
                authentic_execution["success_rate"]
                > runtime_shuffled_execution["success_rate"]
                and runtime_shuffled_execution["success_rate"] <= destructive_max
            ),
            "program_not_updated_from_confirmation_rows": True,
        }
        lineages.append({
            "source_game": game,
            "status": (
                "FRESH_SOURCE_INDUCED_PROGRAM_CONFIRMED"
                if all(gates.values()) else "FRESH_SOURCE_INDUCED_PROGRAM_NOT_CONFIRMED"
            ),
            "program_artifact_sha256": artifact["artifact_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "rows_file_sha256": file_sha256(rows_path),
            "heldout_planned": heldout_planned,
            "heldout_eligible": eligible,
            "heldout_eligible_fraction": eligible_fraction,
            "intervention_failed_rows": raw_failed,
            "executions": {
                "authentic": authentic_execution,
                "frozen_shuffled_effect_program": learned_shuffled_execution,
                "runtime_effect_permutation": runtime_shuffled_execution,
            },
            "gates": gates,
        })
    gates = {
        "exact_six_lineages": len(lineages) == 6,
        "all_six_fresh_source_confirmations_pass": all(
            row["status"] == "FRESH_SOURCE_INDUCED_PROGRAM_CONFIRMED"
            for row in lineages
        ),
    }
    body = {
        "schema_version": "phase3-source-confirmation-report-v1",
        "status": (
            "PHASE3_SOURCE_ONLY_INDUCTION_PROSPECTIVELY_CONFIRMED"
            if all(gates.values())
            else "PHASE3_SOURCE_ONLY_INDUCTION_PROSPECTIVE_CONFIRMATION_FAILED"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "lineages": lineages,
        "gates": gates,
        "claim_boundary": manifest["claim_boundary"],
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "report_sha256": report["report_sha256"],
        "lineages": [
            {
                "source_game": row["source_game"],
                "status": row["status"],
                "heldout_eligible": row["heldout_eligible"],
                "authentic": row["executions"]["authentic"]["success_rate"],
                "shuffled": row["executions"]["frozen_shuffled_effect_program"]["success_rate"],
                "runtime_permuted": row["executions"]["runtime_effect_permutation"]["success_rate"],
            }
            for row in lineages
        ],
    }, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
