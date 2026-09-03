#!/usr/bin/env python3
"""Check whether the V4 neural controller can replace Phase-8 live programs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.harness_controller_training import anonymous_program_ir  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v4-manifest", type=Path,
        default=REPO / "runs/harness_controller_sft_v4_cardinality/manifest.json",
    )
    parser.add_argument(
        "--phase9-audit", type=Path,
        default=REPO / "docs/results/phase9_source_program_heterogeneity_v1.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/harness_controller_v4_phase8_ir_coverage.json",
    )
    args = parser.parse_args()
    v4 = json.loads(args.v4_manifest.read_text(encoding="utf-8"))
    phase9 = json.loads(args.phase9_audit.read_text(encoding="utf-8"))
    if (
        v4.get("status") != "FROZEN_SOURCE_ONLY_CONTROLLER_SUPERVISION"
        or v4.get("target_data_used") is not False
        or not all(v4.get("gates", {}).values())
    ):
        raise SystemExit("V4 controller dataset is not gate-clean")
    if (
        phase9.get("status")
        != "PHASE9_SOURCE_PROGRAM_HETEROGENEITY_AND_TARGET_UTILITY_VALIDATED"
        or not all(phase9.get("gates", {}).values())
    ):
        raise SystemExit("Phase-9 source-program audit is not gate-clean")

    v4_programs = []
    v4_schemas = set()
    v4_hashes = set()
    for receipt in v4["input_receipts"]:
        program_path = REPO / str(receipt["program_file"])
        if _sha256(program_path) != receipt["program_file_sha256"]:
            raise SystemExit(f"V4 source program hash mismatch: {program_path}")
        artifact = json.loads(program_path.read_text(encoding="utf-8"))
        program = artifact["source_function_program"]
        anonymous = anonymous_program_ir(program)
        v4_schemas.add(str(anonymous["schema_version"]))
        v4_hashes.add(str(program["program_sha256"]))
        v4_programs.append({
            "source_family": receipt["source_family"],
            "program_sha256": program["program_sha256"],
            "neural_input_schema": anonymous["schema_version"],
            "program_status": program["status"],
        })

    routes = []
    phase8_hashes = set()
    for route in phase9["route_audits"]:
        requirement = route["target_requirement"]
        selected_hash = str(route["expected_program_sha256"])
        phase8_hashes.add(selected_hash)
        routes.append({
            "route_id": route["route_id"],
            "target_domain": route["target_domain"],
            "successful_source_program_sha256": selected_hash,
            "successful_ir_kind": requirement["ir_kind"],
            "operator_sequence": requirement["operator_sequence"],
            "recurrent": requirement["recurrent"],
            "directly_represented_by_v4_neural_input_schema": False,
            "successful_program_present_in_v4_training_lineage": (
                selected_hash in v4_hashes
            ),
        })

    direct_coverage = sum(
        row["directly_represented_by_v4_neural_input_schema"] for row in routes
    )
    lineage_overlap = v4_hashes & phase8_hashes
    gates = {
        "v4_is_source_only_and_gate_clean": True,
        "phase9_live_route_audit_gate_clean": True,
        "all_v4_programs_share_declared_neural_schema": len(v4_schemas) == 1,
        "successful_phase8_program_hash_overlap_with_v4_is_zero": not lineage_overlap,
        "direct_phase8_live_ir_coverage_is_complete": direct_coverage == len(routes),
    }
    payload: dict[str, Any] = {
        "schema_version": "harness-controller-v4-phase8-live-ir-coverage-audit-v1",
        "status": (
            "V4_DIRECT_PHASE8_LIVE_REPLACEMENT_READY"
            if gates["direct_phase8_live_ir_coverage_is_complete"]
            else "V4_DIRECT_PHASE8_LIVE_REPLACEMENT_BLOCKED_IR_SCHEMA_GAP"
        ),
        "inputs": {
            "v4_manifest": {
                "path": str(args.v4_manifest.resolve()),
                "sha256": _sha256(args.v4_manifest),
            },
            "phase9_audit": {
                "path": str(args.phase9_audit.resolve()),
                "sha256": _sha256(args.phase9_audit),
            },
        },
        "v4_controller_scope": {
            "neural_input_schemas": sorted(v4_schemas),
            "source_programs": v4_programs,
            "learned_function": (
                "RANK_TYPED_EFFECT_CANDIDATES_AND_APPLY_ONE_SOURCE_FUNCTION_TRANSITION"
            ),
        },
        "phase8_live_routes": routes,
        "coverage": {
            "routes": len(routes),
            "directly_covered": direct_coverage,
            "successful_program_hash_overlap": sorted(lineage_overlap),
            "required_ir_kinds": sorted({row["successful_ir_kind"] for row in routes}),
        },
        "gates": gates,
        "next_legal_step": (
            "BUILD_SOURCE_ONLY_MULTI_IR_CONTROLLER_SUPERVISION_FOR_PHASE8_PROGRAMS;"
            "DO_NOT_CLAIM_V4_TARGET_IR_GATE_AS_PHASE8_LIVE_REPLACEMENT"
        ),
        "claim_boundary": (
            "The V4 cardinality experiment remains a valid source-only test of a "
            "typed-effect function interpreter. It does not directly cover the "
            "recurrent relational, finite structural-delta, or recurrent "
            "goal-acquisition programs responsible for the Phase-8 live success results."
        ),
    }
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"], "coverage": payload["coverage"],
        "next_legal_step": payload["next_legal_step"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
