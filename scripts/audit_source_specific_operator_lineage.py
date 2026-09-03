#!/usr/bin/env python3
"""Audit source-specific programs and their lineage into the shared operator algebra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-dir", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("lineage audit is immutable")
    capabilities = json.loads(args.capabilities.read_text(encoding="utf-8"))
    rows = []
    authentic_correct = shuffled_correct = examples = 0
    qualified_program_hashes = set()
    for path in sorted(args.program_dir.glob("*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        program = artifact["source_function_program"]
        qualification = program["qualification_gates"]
        calibration = program["cross_batch_calibration"]
        operators = program["operators"]
        qualified = bool(operators) and all(qualification.values()) and all(calibration["gates"].values())
        terms = operators[0]["score"]["terms"] if operators else []
        graph = program["transition_graph"]
        fingerprint_body = {
            "terms": terms,
            "required_observation_horizon": graph["required_observation_horizon"],
            "transitions": graph["transitions"],
            "abstention_rule": program["abstention_rule"],
        }
        authentic = calibration["metrics"]["authentic"]
        shuffled = calibration["metrics"]["shuffled_effect_binding"]
        if qualified:
            qualified_program_hashes.add(program["program_sha256"])
            authentic_correct += authentic["correct"]
            shuffled_correct += shuffled["correct"]
            examples += authentic["examples"]
        rows.append({
            "source": path.stem,
            "program_sha256": program["program_sha256"],
            "program_fingerprint_sha256": stable_hash(fingerprint_body),
            "qualified": qualified,
            "applicability_decision": "TRANSFERABLE" if qualified else "ABSTAIN",
            "terms": terms,
            "required_observation_horizon": graph["required_observation_horizon"],
            "authentic_calibration_accuracy": authentic["accuracy"],
            "shuffled_calibration_accuracy": shuffled["accuracy"],
            "target_data_read": False,
        })
    fingerprints = {row["program_fingerprint_sha256"] for row in rows}
    authorized_receipts = {
        receipt
        for capability in capabilities["capabilities"].values()
        if capability["authorized"]
        for receipt in capability["receipt_sha256s"]
    }
    temporal_lineage = sorted(qualified_program_hashes & authorized_receipts)
    gates = {
        "all_six_sources_audited": len(rows) == 6,
        "source_programs_not_identity_aliases": len(fingerprints) == len(rows),
        "multiple_distinct_programs_qualified": len({row["program_fingerprint_sha256"] for row in rows if row["qualified"]}) >= 3,
        "failed_sources_abstain": sum(not row["qualified"] for row in rows) >= 1,
        "qualified_heldout_beats_shuffled": authentic_correct > shuffled_correct,
        "qualified_programs_trace_into_algebra": len(temporal_lineage) == len(qualified_program_hashes),
        "target_data_never_read": capabilities["target_data_read"] is False,
    }
    body = {
        "schema_version": "source-specific-operator-lineage-audit-v1",
        "status": "SOURCE_SPECIFICITY_VALIDATED" if all(gates.values()) else "SOURCE_SPECIFICITY_GATE_FAILED",
        "authority": "SEALED_SOURCE_PROGRAMS_AND_HELDOUT_SOURCE_CALIBRATION_ONLY",
        "sources": rows,
        "aggregate_qualified_calibration": {
            "examples": examples,
            "authentic_correct": authentic_correct,
            "shuffled_correct": shuffled_correct,
            "authentic_accuracy": authentic_correct / examples,
            "shuffled_accuracy": shuffled_correct / examples,
        },
        "qualified_programs_traced_to_agqa_algebra": temporal_lineage,
        "gates": gates,
        "target_data_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(body, indent=2, sort_keys=True))
    return 0 if body["status"] == "SOURCE_SPECIFICITY_VALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
