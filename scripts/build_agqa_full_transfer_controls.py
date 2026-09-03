#!/usr/bin/env python3
"""Freeze capability controls before any new formal outcomes are read."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def artifact(kind: str, operators: list[str], base_sha: str, *, compositions=(), **extra) -> dict:
    body = {
        "schema_version": "agqa-full-transfer-control-v1", "control_kind": kind,
        "authorized_operators": sorted(operators),
        "authorized_compositions": sorted([list(edge) for edge in compositions]),
        "base_source_capability_artifact_sha256": base_sha,
        "formal_outcomes_read": False,
    } | extra
    body["artifact_sha256"] = stable_hash(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("control freeze is immutable")
    source = json.loads(args.source_capabilities.read_text(encoding="utf-8"))
    source_ops = source["authorized_operators"]
    source_compositions = source.get("authorized_compositions", ())
    all_vm_ops = sorted(source["capabilities"])
    controls = {
        "neural_only": artifact(
            "NEURAL_ONLY", [], source["artifact_sha256"],
            source_evidence_used=False, symbolic_execution_enabled=False,
        ),
        "source_induced": artifact(
            "SOURCE_INDUCED", source_ops, source["artifact_sha256"], compositions=source_compositions,
            source_evidence_used=True, target_authored_operator_support=False,
        ),
        "source_permuted": artifact(
            "SOURCE_PERMUTED", [], source["artifact_sha256"],
            source_evidence_used=True,
            revocation_reason="SHUFFLED_EFFECT_BINDING_REJECTS_CAUSAL_CAPABILITY_LINEAGE",
        ),
        "generic_scaffold": artifact(
            "GENERIC_SCAFFOLD", all_vm_ops, source["artifact_sha256"],
            source_evidence_used=False, generic_operator_inventory_written=True,
        ),
        "target_written_isomorphic": artifact(
            "TARGET_WRITTEN_ISOMORPHIC", source_ops, source["artifact_sha256"], compositions=source_compositions,
            source_evidence_used=False, target_authored_operator_support=True,
            source_applicability_and_schedule_used=False,
        ),
        "oracle_program": artifact(
            "ORACLE_PROGRAM_CONTROL", source_ops, source["artifact_sha256"], compositions=source_compositions,
            source_evidence_used=True, runtime_functional_program_source="EVALUATOR_ORACLE",
            primary_claim_eligible=False,
        ),
    }
    args.output_dir.mkdir(parents=True)
    for name, value in controls.items():
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    manifest_body = {
        "schema_version": "agqa-full-transfer-control-freeze-v1",
        "status": "SIX_ARMS_FROZEN",
        "formal_outcomes_read": False,
        "controls": {name: value["artifact_sha256"] for name, value in controls.items()},
        "interpretation_gate": (
            "SOURCE_INDUCED_MUST_BE_COMPARED_WITH_GENERIC_AND_TARGET_WRITTEN_ISOMORPHIC;"
            "SOURCE_PERMUTED_ONLY_TESTS_SOURCE_LINEAGE;ORACLE_PROGRAM_IS_CEILING_ONLY"
        ),
    }
    manifest_body["manifest_sha256"] = stable_hash(manifest_body)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest_body, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest_body, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
