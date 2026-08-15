#!/usr/bin/env python3
"""Build the strict 6-by-4 audit for the six-game common artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.search_automaton_transfer_v16 import (  # noqa: E402
    SourceSearchAutomaton,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _self_hash(value: dict, field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if claimed != stable_hash(body):
        raise ValueError(f"{field} mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-artifact", type=Path,
        default=(
            REPO / "runs/phase1_common_search_ir_formal_v1/"
            "common_search_automaton_artifact.json"
        ),
    )
    parser.add_argument(
        "--target-relineage", type=Path,
        default=(
            REPO / "runs/phase1_common_search_ir_formal_v1/"
            "four_target_relineage_report.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=(
            REPO / "docs/results/"
            "phase1_six_game_four_target_transfer_audit_v2.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    source_artifact = _read(args.source_artifact)
    source = SourceSearchAutomaton(source_artifact)
    target = _read(args.target_relineage)
    _self_hash(target, "report_sha256")
    if target.get("new_source_artifact_sha256") != source.artifact_sha256:
        raise ValueError("target relineage/common source mismatch")
    lineages = {
        str(row["game"]): row for row in source_artifact["source_lineages"]
    }
    domains = dict(target["domains"])
    cells = []
    for game, lineage in lineages.items():
        for domain, domain_report in domains.items():
            source_passed = (
                int(lineage["fresh_eligible_states"]) >= 8
                and int(lineage["eligible_ledgers"]) >= 8
            )
            target_passed = all(domain_report["gates"].values())
            mechanism_validated = source_passed and target_passed
            cells.append({
                "source_game": game,
                "target_domain": domain,
                "status": (
                    "MECHANISM_VALIDATED_BY_FORMAL_SOURCE_AND_EXHAUSTIVE_"
                    "HISTORICAL_TARGET_RELINEAGE"
                    if mechanism_validated else "TRANSFER_NOT_VALIDATED"
                ),
                "source_formal_gate_passed": source_passed,
                "source_report_file_sha256": lineage[
                    "report_file_sha256"
                ],
                "common_source_artifact_sha256": source.artifact_sha256,
                "exact_source_membership_in_common_artifact": True,
                "target_program_equivalence_passed": target_passed,
                "target_domain_report_sha256": domain_report[
                    "domain_report_sha256"
                ],
                "target_evidence_tier": domain_report["evidence_tier"],
                "historical_target_outcomes_sha256": domain_report[
                    "historical_outcomes_sha256"
                ],
                "direct_new_joint_source_target_execution": False,
                "mechanism_transfer_validated": mechanism_validated,
            })
    expected_pairs = {
        (game, domain) for game in lineages for domain in domains
    }
    actual_pairs = {
        (row["source_game"], row["target_domain"]) for row in cells
    }
    gates = {
        "six_formal_source_lineages": len(lineages) == 6,
        "four_validated_target_programs": len(domains) == 4,
        "complete_six_by_four_matrix": (
            len(cells) == 24 and actual_pairs == expected_pairs
        ),
        "all_24_mechanism_cells_validated": all(
            row["mechanism_transfer_validated"] for row in cells
        ),
        "common_artifact_target_authorized": source_artifact.get(
            "target_authorized"
        ) is True,
        "source_native_tokens_absent_from_transfer_artifact": (
            "candidate_action" not in json.dumps(source_artifact)
            and "prefix_actions" not in json.dumps(source_artifact)
        ),
        "target_relineage_report_passed": all(target["gates"].values()),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "phase1-six-game-four-target-transfer-audit-v2",
        "status": (
            "SIX_BY_FOUR_MECHANISM_TRANSFER_VALIDATED"
            if passed else "SIX_BY_FOUR_MECHANISM_TRANSFER_NOT_VALIDATED"
        ),
        "source_artifact_sha256": source.artifact_sha256,
        "canonical_policy_sha256": source_artifact[
            "canonical_policy_sha256"
        ],
        "source_games": list(lineages),
        "target_domains": list(domains),
        "validated_mechanism_cells": sum(
            row["mechanism_transfer_validated"] for row in cells
        ),
        "direct_new_joint_execution_cells": sum(
            row["direct_new_joint_source_target_execution"] for row in cells
        ),
        "cells": cells,
        "historical_target_outcomes": {
            domain: report["historical_outcomes"]
            for domain, report in domains.items()
        },
        "gates": gates,
        "claim_boundary": (
            "VALIDATES_TRANSFERABLE_NEURAL_SYMBOLIC_MECHANISM_FOR_24_"
            "SOURCE_TARGET_PAIRS;SOURCE EVIDENCE_IS_NEW_FORMAL;TARGET_SUCCESS_"
            "OUTCOMES_ARE_HISTORICAL_AND_RELINEAGED_BY_EXHAUSTIVE_PROGRAM_"
            "EQUIVALENCE;ZERO_NEW_JOINT_PROSPECTIVE_CELLS"
        ),
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "validated_mechanism_cells": report["validated_mechanism_cells"],
        "direct_new_joint_execution_cells": (
            report["direct_new_joint_execution_cells"]
        ),
        "gates": gates,
        "report_sha256": report["report_sha256"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
