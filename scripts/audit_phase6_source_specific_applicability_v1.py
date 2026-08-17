#!/usr/bin/env python3
"""Audit source-specific structural routing over an expanded source catalog.

The audit is retrospective with respect to the already frozen route evidence,
but prospective/outcome-blind with respect to applicability: it consumes only
target-native interface requirements.  The six arcade functions are included
as a deliberately difficult expansion.  Their failed aggregate permutation
gate is preserved and they cannot be promoted into a target route.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_ir_applicability import (  # noqa: E402
    OperatorSignature,
    TargetIRRequirement,
    relational_artifact_contract,
    select_source_contract,
    structural_program_contract,
    temporal_function_artifact_contract,
)


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


def _finite_requirement(route: Mapping[str, Any]) -> TargetIRRequirement:
    return TargetIRRequirement.create(
        task_id=f"{route['target_domain']}.phase6.interface-canary",
        target_domain=str(route["target_domain"]),
        target_interface=str(route["target_interface"]),
        target_grounder_sha256=str(route["target_grounder_sha256"]),
        ir_kind="FINITE_STRUCTURAL_DELTA_SEQUENCE",
        operator_sequence=(
            OperatorSignature("ADD", "ENTITY_SLOT", 1, "ENTITY_REFERENCE"),
            OperatorSignature("REMOVE", "ENTITY_SLOT", 1, "ENTITY_REFERENCE"),
        ),
        recurrent=False,
        terminal_predicate_families=(),
        grounder_qualified=True,
        formal_outcome_read=False,
    )


def _relational_requirement(route: Mapping[str, Any]) -> TargetIRRequirement:
    return TargetIRRequirement.create(
        task_id=f"{route['target_domain']}.phase6.interface-canary",
        target_domain=str(route["target_domain"]),
        target_interface=str(route["target_interface"]),
        target_grounder_sha256=str(route["target_grounder_sha256"]),
        ir_kind="RECURRENT_RELATIONAL_TRANSITION_PROGRAM",
        operator_sequence=(
            OperatorSignature("UPDATE", "CONTROL_STATE", 1, "POSITION"),
        ),
        recurrent=True,
        terminal_predicate_families=("ENTITY_GOAL_RELATION",),
        grounder_qualified=True,
        formal_outcome_read=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase5-manifest", type=Path,
        default=REPO / "configs/phase5_unified_applicability_v1_frozen.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/phase6_source_specific_applicability_v1.json",
    )
    args = parser.parse_args()

    phase5 = _read(args.phase5_manifest)
    _self_hash(phase5, "manifest_sha256")
    minigrid_report_path = REPO / "runs/source_structural_v5c_fresh/report.json"
    minigrid_report = _read(minigrid_report_path)
    _self_hash(minigrid_report, "report_sha256")
    sokoban_report_path = (
        REPO / "runs/sokoban_relational_structural_v2/"
        "fresh_confirmation_report.json"
    )
    sokoban_report = _read(sokoban_report_path)
    _self_hash(sokoban_report, "report_sha256")
    arcade_report_path = REPO / "runs/phase3_source_function_v4_reserve/report.json"
    arcade_report = _read(arcade_report_path)
    _self_hash(arcade_report, "report_sha256")

    sources = []
    source_rows = []
    for path in sorted((
        REPO / "configs/source_structural_v5c_frozen/programs"
    ).glob("*.json")):
        program = _read(path)
        contract = structural_program_contract(
            program,
            source_confirmation_sha256=minigrid_report["report_sha256"],
            source_intervention_qualified=(
                minigrid_report["status"] == "SOURCE_STRUCTURAL_FRESH_VALIDATED"
                and all(minigrid_report["gates"].values())
            ),
        )
        sources.append(contract)
        source_rows.append({
            "catalog_label": f"minigrid:{path.stem}",
            "contract": asdict(contract),
            "program_sha256": contract.program_sha256,
            "contract_sha256": contract.contract_sha256,
            "ir_kind": contract.ir_kind,
            "fresh_confirmed": contract.source_intervention_qualified,
            "source_identity_used_as_feature": False,
        })

    sokoban_artifact_path = (
        REPO / "runs/sokoban_relational_structural_v2/artifact.json"
    )
    sokoban_artifact = _read(sokoban_artifact_path)
    sokoban = relational_artifact_contract(
        sokoban_artifact,
        source_confirmation_sha256=sokoban_report["report_sha256"],
        source_intervention_qualified=(
            sokoban_report["status"]
            == "SOURCE_RELATIONAL_STRUCTURAL_FRESH_VALIDATED"
            and all(sokoban_report["gates"].values())
        ),
    )
    sources.append(sokoban)
    source_rows.append({
        "catalog_label": "sokoban:relational-v2",
        "contract": asdict(sokoban),
        "program_sha256": sokoban.program_sha256,
        "contract_sha256": sokoban.contract_sha256,
        "ir_kind": sokoban.ir_kind,
        "fresh_confirmed": sokoban.source_intervention_qualified,
        "source_identity_used_as_feature": False,
    })

    arcade_by_game = {
        str(row["source_game"]): row for row in arcade_report["lineages"]
    }
    arcade_contracts = []
    for path in sorted((
        REPO / "configs/phase3_source_function_v4/frozen_reserve/programs"
    ).glob("*.json")):
        lineage = arcade_by_game[path.stem]
        artifact = _read(path)
        confirmed = lineage["status"] == "V4_SOURCE_DOMAIN_FUNCTION_CONFIRMED"
        contract = temporal_function_artifact_contract(
            artifact,
            source_confirmation_sha256=arcade_report["report_sha256"],
            source_intervention_qualified=confirmed,
        )
        sources.append(contract)
        arcade_contracts.append(contract)
        source_rows.append({
            "catalog_label": f"arcade:{path.stem}",
            "contract": asdict(contract),
            "program_sha256": contract.program_sha256,
            "contract_sha256": contract.contract_sha256,
            "ir_kind": contract.ir_kind,
            "fresh_confirmed": contract.source_intervention_qualified,
            "lineage_status": lineage["status"],
            "source_identity_used_as_feature": False,
        })

    routes = {row["route_id"]: row for row in phase5["routes"]}
    requirements = []
    for route_id in (
        "minigrid-put-near-to-discoveryworld-easy-v1",
        "minigrid-put-near-to-alfworld-multiplicity-v2",
        "sokoban-relational-to-tir-maze-v3",
        "sokoban-relational-to-webshop-v21",
    ):
        route = routes[route_id]
        if route_id.startswith("minigrid-"):
            requirement = _finite_requirement(route)
        else:
            requirement = _relational_requirement(route)
        receipt = select_source_contract(sources, requirement)
        wrong = sokoban if route_id.startswith("minigrid-") else next(
            row for row in sources
            if row.program_sha256 == routes[
                "minigrid-put-near-to-discoveryworld-easy-v1"
            ]["source_program_sha256"]
        )
        permuted = select_source_contract((wrong,), requirement)
        requirements.append({
            "route_id": route_id,
            "target_requirement": asdict(requirement),
            "authentic_selection": receipt,
            "source_permuted_selection": permuted,
            "expected_source_program_sha256": route["source_program_sha256"],
            "existing_target_evidence_status": route["evidence"]["status"],
            "current_target_outcome_read": False,
        })

    selected = [
        row["authentic_selection"]["selected_program_sha256"]
        for row in requirements
    ]
    arcade_matches = sum(
        item["matched"]
        for row in requirements
        for item in row["authentic_selection"]["source_contracts"]
        if item["program_sha256"] in {
            contract.program_sha256 for contract in arcade_contracts
        }
    )
    arcade_rates = arcade_report["qualified_aggregate"]
    gates = {
        "expanded_catalog_has_ten_source_programs": len(sources) == 10,
        "three_minigrid_one_sokoban_six_arcade_sources": (
            sum(row["catalog_label"].startswith("minigrid:") for row in source_rows)
            == 3
            and sum(row["catalog_label"].startswith("sokoban:") for row in source_rows)
            == 1
            and sum(row["catalog_label"].startswith("arcade:") for row in source_rows)
            == 6
        ),
        "every_target_interface_has_unique_content_match": all(
            row["authentic_selection"]["status"]
            == "UNIQUE_SOURCE_CONTRACT_SELECTED"
            for row in requirements
        ),
        "selected_programs_equal_preexisting_evidence_lineage": all(
            row["authentic_selection"]["selected_program_sha256"]
            == row["expected_source_program_sha256"]
            for row in requirements
        ),
        "source_programs_are_behaviorally_routed_not_identity_retrieved": (
            len(set(selected)) == 2
            and selected.count(routes[
                "minigrid-put-near-to-discoveryworld-easy-v1"
            ]["source_program_sha256"]) == 2
            and selected.count(routes[
                "sokoban-relational-to-tir-maze-v3"
            ]["source_program_sha256"]) == 2
        ),
        "source_permutation_is_rejected_by_type_checker": all(
            row["source_permuted_selection"]["status"]
            == "SOURCE_CONTRACT_SELECTION_ABSTAINED"
            for row in requirements
        ),
        "arcade_functions_do_not_false_match_structural_targets": (
            arcade_matches == 0
        ),
        "arcade_v4_negative_permutation_result_preserved": (
            arcade_report["status"] == "SOURCE_SPECIFIC_DOMAIN_FUNCTIONS_FAILED"
            and arcade_rates["authentic_correct"]
            == arcade_rates["permuted_correct"]
            and arcade_report["gates"][
                "qualified_authentic_aggregate_beats_source_permuted"
            ] is False
        ),
        "no_current_target_outcome_or_action_in_selector": all(
            row["current_target_outcome_read"] is False
            and row["authentic_selection"]["target_outcome_read"] is False
            and row["authentic_selection"]["target_action_emitted"] is False
            for row in requirements
        ),
    }
    body = {
        "schema_version": "phase6-source-specific-applicability-audit-v1",
        "status": (
            "PHASE6_SELECTIVE_SOURCE_SPECIFIC_APPLICABILITY_VALIDATED"
            if all(gates.values()) else
            "PHASE6_SOURCE_SPECIFIC_APPLICABILITY_FAILED"
        ),
        "claim_boundary": (
            "Validates anonymous structural type selection for routes whose "
            "target utility was already measured. It does not rehabilitate "
            "the failed V4 arcade source-permutation result and does not claim "
            "that an arcade temporal function transfers to these targets."
        ),
        "phase5_manifest_sha256": phase5["manifest_sha256"],
        "evidence": {
            "minigrid_report": {
                "path": str(minigrid_report_path.relative_to(REPO)),
                "file_sha256": _sha(minigrid_report_path),
                "report_sha256": minigrid_report["report_sha256"],
            },
            "sokoban_report": {
                "path": str(sokoban_report_path.relative_to(REPO)),
                "file_sha256": _sha(sokoban_report_path),
                "report_sha256": sokoban_report["report_sha256"],
            },
            "arcade_v4_report": {
                "path": str(arcade_report_path.relative_to(REPO)),
                "file_sha256": _sha(arcade_report_path),
                "report_sha256": arcade_report["report_sha256"],
                "status": arcade_report["status"],
                "qualified_aggregate": arcade_rates,
            },
        },
        "source_catalog": source_rows,
        "target_requirements": requirements,
        "arcade_structural_target_matches": arcade_matches,
        "gates": gates,
        "target_outcomes_read": 0,
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "source_programs": len(sources),
        "selected_programs": selected,
        "arcade_structural_target_matches": arcade_matches,
        "gates": gates,
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
