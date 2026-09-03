#!/usr/bin/env python3
"""Audit whether learned source-program content predicts target utility.

This is a cross-report integrity audit, not a new target evaluation.  It first
selects a source program from anonymous structural contracts without reading a
target outcome.  It then binds that prediction to the already frozen matched
formal report for each route and verifies that the authentic binding beat the
route's destructive source control.

The route-level results come from prospective formal runs, but this synthesis
is retrospective.  In particular, discordant tasks from different domains are
not pooled into a new iid significance test.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import gzip
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
    SourceIRContract,
    TargetIRRequirement,
    goal_acquisition_artifact_contract,
    select_source_contract,
)


PHASE6 = REPO / "docs/results/phase6_source_specific_applicability_v1.json"
PHASE5 = REPO / "configs/phase5_unified_applicability_v1_frozen.json"
GOAL_ARTIFACT = REPO / "runs/sokoban_goal_acquisition_v1/artifact.json"
GOAL_CONFIRMATION = (
    REPO / "runs/sokoban_goal_acquisition_v1/fresh_confirmation_report.json"
)
ALFWORLD_CONFIG = REPO / "configs/alfworld_unified_goal_acquisition_v13_formal.json"


def _bytes(path: Path) -> bytes:
    if path.is_file():
        return path.read_bytes()
    archive = Path(str(path) + ".gz")
    if archive.is_file():
        return gzip.decompress(archive.read_bytes())
    raise FileNotFoundError(path)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(_bytes(path).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(_bytes(path)).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}: {claimed}")


def _signature(value: Mapping[str, Any]) -> OperatorSignature:
    return OperatorSignature(
        operation=str(value["operation"]),
        predicate_family=str(value["predicate_family"]),
        arity=int(value["arity"]),
        value_kind=str(value["value_kind"]),
    )


def _source_contract(value: Mapping[str, Any]) -> SourceIRContract:
    contract = SourceIRContract(
        program_sha256=str(value["program_sha256"]),
        ir_kind=str(value["ir_kind"]),
        operator_sequence=tuple(
            _signature(row) for row in value["operator_sequence"]
        ),
        recurrent=bool(value["recurrent"]),
        terminal_predicate_families=tuple(map(
            str, value["terminal_predicate_families"],
        )),
        source_intervention_qualified=bool(
            value["source_intervention_qualified"]
        ),
        source_confirmation_sha256=str(
            value["source_confirmation_sha256"]
        ),
        contract_sha256=str(value["contract_sha256"]),
    )
    contract.validate()
    return contract


def _target_requirement(value: Mapping[str, Any]) -> TargetIRRequirement:
    requirement = TargetIRRequirement(
        task_id=str(value["task_id"]),
        target_domain=str(value["target_domain"]),
        target_interface=str(value["target_interface"]),
        target_grounder_sha256=str(value["target_grounder_sha256"]),
        ir_kind=str(value["ir_kind"]),
        operator_sequence=tuple(
            _signature(row) for row in value["operator_sequence"]
        ),
        recurrent=bool(value["recurrent"]),
        terminal_predicate_families=tuple(map(
            str, value["terminal_predicate_families"],
        )),
        grounder_qualified=bool(value["grounder_qualified"]),
        formal_outcome_read=bool(value["formal_outcome_read"]),
        requirement_sha256=str(value["requirement_sha256"]),
    )
    requirement.validate()
    return requirement


def _formal_evidence() -> dict[str, dict[str, Any]]:
    """Recalculate the four route-level matched endpoints."""

    webshop_path = REPO / "runs/webshop_structural_transfer_v21_formal/report.json"
    discovery_path = (
        REPO / "runs/discoveryworld_structural_transfer_v1_matched/report.json"
    )
    tir_path = REPO / "runs/tir_maze_structural_transfer_v3/heldout_report.json"
    alfworld_path = (
        REPO / "runs/alfworld_unified_goal_acquisition_v13_formal/report.json"
    )
    reports = {
        "webshop": _read(webshop_path),
        "discoveryworld": _read(discovery_path),
        "tir": _read(tir_path),
        "alfworld": _read(alfworld_path),
    }
    for report in reports.values():
        _self_hash(report, "report_sha256")
        if not all((report.get("gates") or {}).values()):
            raise ValueError("a formal source report has a failed gate")

    webshop = reports["webshop"]
    discovery = reports["discoveryworld"]
    tir = reports["tir"]
    alfworld = reports["alfworld"]
    rows = {
        "sokoban-relational-to-webshop-v21": {
            "target_domain": "webshop",
            "report_path": str(webshop_path.relative_to(REPO)),
            "report_file_sha256": _sha(webshop_path),
            "report_sha256": webshop["report_sha256"],
            "tasks": webshop["summaries"][
                "source_induced_structural_ir"
            ]["tasks"],
            "authentic_successes": webshop["summaries"][
                "source_induced_structural_ir"
            ]["strict_successes"],
            "source_control": "source_terminal_permuted_control",
            "source_control_successes": webshop["summaries"][
                "source_terminal_permuted_control"
            ]["strict_successes"],
            **{
                key: webshop["paired"][
                    "source_terminal_permuted_control"
                ][key]
                for key in ("wins", "losses", "ties", "exact_two_sided_p")
            },
        },
        "minigrid-put-near-to-discoveryworld-easy-v1": {
            "target_domain": "discoveryworld",
            "report_path": str(discovery_path.relative_to(REPO)),
            "report_file_sha256": _sha(discovery_path),
            "report_sha256": discovery["report_sha256"],
            "tasks": discovery["applicable_tasks"],
            "authentic_successes": discovery["condition_successes"][
                "source_induced"
            ],
            "source_control": "source_permuted",
            "source_control_successes": discovery["condition_successes"][
                "source_permuted"
            ],
            **{
                key: discovery["source_vs_permuted"][source_key]
                for key, source_key in (
                    ("wins", "wins"), ("losses", "losses"),
                    ("ties", "ties"),
                    ("exact_two_sided_p", "exact_two_sided_sign_p"),
                )
            },
        },
        "sokoban-relational-to-tir-maze-v3": {
            "target_domain": "tir",
            "report_path": str(tir_path.relative_to(REPO)),
            "report_file_sha256": _sha(tir_path),
            "report_sha256": tir["report_sha256"],
            "tasks": tir["summaries"]["source_induced"]["tasks"],
            "authentic_successes": tir["summaries"][
                "source_induced"
            ]["successes"],
            "source_control": "source_relation_permuted",
            "source_control_successes": tir["summaries"][
                "source_relation_permuted"
            ]["successes"],
            **{
                key: tir["paired"]["source_relation_permuted"][key]
                for key in ("wins", "losses", "ties", "exact_two_sided_p")
            },
        },
        "sokoban-goal-acquisition-to-alfworld-multiplicity-v11": {
            "target_domain": "alfworld",
            "report_path": str(alfworld_path.relative_to(REPO)),
            "report_file_sha256": _sha(alfworld_path),
            "report_sha256": alfworld["report_sha256"],
            "tasks": alfworld["summaries"][
                "authentic_source_goal_relation_macro"
            ]["tasks"],
            "authentic_successes": alfworld["summaries"][
                "authentic_source_goal_relation_macro"
            ]["successes"],
            "source_control": "source_effect_binding_permuted_control",
            "source_control_successes": alfworld["summaries"][
                "source_effect_binding_permuted_control"
            ]["successes"],
            **{
                key: alfworld["paired"][
                    "source_effect_binding_permuted_control"
                ][key]
                for key in ("wins", "losses", "ties", "exact_two_sided_p")
            },
        },
    }
    return rows


def build_report() -> dict[str, Any]:
    phase6 = _read(PHASE6)
    phase5 = _read(PHASE5)
    goal_artifact = _read(GOAL_ARTIFACT)
    goal_confirmation = _read(GOAL_CONFIRMATION)
    alfworld_config = _read(ALFWORLD_CONFIG)
    _self_hash(phase6, "report_sha256")
    _self_hash(phase5, "manifest_sha256")
    _self_hash(alfworld_config, "config_sha256")
    if phase6["status"] != "PHASE6_SELECTIVE_SOURCE_SPECIFIC_APPLICABILITY_VALIDATED":
        raise ValueError("Phase 6 catalog is not validated")

    catalog = [
        _source_contract(row["contract"]) for row in phase6["source_catalog"]
    ]
    goal_contract = goal_acquisition_artifact_contract(
        goal_artifact, confirmation=goal_confirmation,
    )
    catalog.append(goal_contract)
    if len({row.program_sha256 for row in catalog}) != len(catalog):
        raise ValueError("source catalog contains duplicate program hashes")

    old_requirements = {
        str(row["route_id"]): _target_requirement(row["target_requirement"])
        for row in phase6["target_requirements"]
    }
    route_ids = (
        "sokoban-relational-to-webshop-v21",
        "minigrid-put-near-to-discoveryworld-easy-v1",
        "sokoban-relational-to-tir-maze-v3",
    )
    requirements = {route_id: old_requirements[route_id] for route_id in route_ids}
    requirements[
        "sokoban-goal-acquisition-to-alfworld-multiplicity-v11"
    ] = TargetIRRequirement.create(
        task_id="alfworld.phase9.goal-acquisition-interface-canary",
        target_domain="alfworld",
        target_interface="multiplicity_goal_acquisition_relation_v11",
        target_grounder_sha256=str(
            alfworld_config["target_grounder_file_sha256"]
        ),
        ir_kind=goal_contract.ir_kind,
        operator_sequence=goal_contract.operator_sequence,
        recurrent=goal_contract.recurrent,
        terminal_predicate_families=goal_contract.terminal_predicate_families,
        grounder_qualified=True,
        formal_outcome_read=False,
    )

    expected_programs = {
        str(row["route_id"]): str(row["source_program_sha256"])
        for row in phase5["routes"]
        if str(row["route_id"]) in requirements
    }
    expected_programs[
        "sokoban-goal-acquisition-to-alfworld-multiplicity-v11"
    ] = goal_contract.program_sha256
    formal = _formal_evidence()

    selected_by_route: dict[str, SourceIRContract] = {}
    route_audits = []
    distinct_contracts = {
        row.contract_sha256: row for row in catalog
        if row.source_intervention_qualified
    }
    for route_id, requirement in requirements.items():
        receipt = select_source_contract(catalog, requirement)
        selected_hash = receipt["selected_program_sha256"]
        selected = next(
            row for row in catalog if row.program_sha256 == selected_hash
        )
        selected_by_route[route_id] = selected
        wrong = next(
            row for row in distinct_contracts.values()
            if row.ir_kind != selected.ir_kind
        )
        wrong_receipt = select_source_contract((wrong,), requirement)
        evidence = formal[route_id]
        route_audits.append({
            "route_id": route_id,
            "target_domain": requirement.target_domain,
            "target_requirement": asdict(requirement),
            "selection": receipt,
            "expected_program_sha256": expected_programs[route_id],
            "selected_contract_sha256": selected.contract_sha256,
            "wrong_family_program_sha256": wrong.program_sha256,
            "wrong_family_selection": wrong_receipt,
            "formal_evidence": evidence,
            "formal_target_outcome_read_by_selector": False,
        })

    # Bind evidence lineage to the program mapping frozen before Phase 6.
    phase5_by_route = {
        str(row["route_id"]): row for row in phase5["routes"]
    }
    legacy_lineage_ok = all(
        phase5_by_route[route_id]["source_program_sha256"]
        == expected_programs[route_id]
        and phase5_by_route[route_id]["evidence_report_sha256"]
        == formal[route_id]["report_sha256"]
        for route_id in route_ids
    )
    alf_report = _read(
        REPO / "runs/alfworld_unified_goal_acquisition_v13_formal/report.json"
    )
    alf_lineage_ok = (
        alf_report["source_acquisition_artifact_sha256"]
        == expected_programs[
            "sokoban-goal-acquisition-to-alfworld-multiplicity-v11"
        ]
        and alf_report["unified_route_id"]
        == "sokoban-goal-acquisition-to-alfworld-multiplicity-v11"
    )
    selected_programs = {
        row.program_sha256 for row in selected_by_route.values()
    }
    descriptive_wins = sum(row["wins"] for row in formal.values())
    descriptive_losses = sum(row["losses"] for row in formal.values())
    gates = {
        "eleven_distinct_source_programs_audited": len(catalog) == 11,
        "three_distinct_program_bodies_selected_for_four_routes": (
            len(selected_programs) == 3
        ),
        "every_target_has_one_anonymous_content_match": all(
            row["selection"]["status"] == "UNIQUE_SOURCE_CONTRACT_SELECTED"
            for row in route_audits
        ),
        "predicted_program_matches_frozen_route_lineage": all(
            row["selection"]["selected_program_sha256"]
            == row["expected_program_sha256"]
            for row in route_audits
        ),
        "wrong_program_family_always_abstains": all(
            row["wrong_family_selection"]["status"]
            == "SOURCE_CONTRACT_SELECTION_ABSTAINED"
            for row in route_audits
        ),
        "selector_uses_no_source_identity_target_action_or_outcome": all(
            row["selection"]["source_identity_used_as_feature"] is False
            and row["selection"]["target_action_emitted"] is False
            and row["selection"]["target_outcome_read"] is False
            and row["formal_target_outcome_read_by_selector"] is False
            for row in route_audits
        ),
        "formal_reports_and_preregistered_gates_reverified": True,
        "selected_program_is_bound_to_each_formal_report": (
            legacy_lineage_ok and alf_lineage_ok
        ),
        "authentic_strictly_beats_source_control_on_every_route": all(
            row["authentic_successes"] > row["source_control_successes"]
            for row in formal.values()
        ),
        "every_route_has_positive_paired_wins_and_zero_losses": all(
            row["wins"] > 0 and row["losses"] == 0
            for row in formal.values()
        ),
        "every_route_source_control_test_passes_exact_0p05": all(
            row["exact_two_sided_p"] <= 0.05 for row in formal.values()
        ),
        "failed_arcade_permutation_family_remains_unpromoted": (
            phase6["evidence"]["arcade_v4_report"]["status"]
            == "SOURCE_SPECIFIC_DOMAIN_FUNCTIONS_FAILED"
            and not any(
                row.program_sha256 in selected_programs
                for row in catalog
                if row.ir_kind == "SPARSE_TEMPORAL_EFFECT_FUNCTION"
            )
        ),
    }
    body = {
        "schema_version": "phase9-source-program-heterogeneity-audit-v1",
        "status": (
            "PHASE9_SOURCE_PROGRAM_HETEROGENEITY_AND_TARGET_UTILITY_VALIDATED"
            if all(gates.values()) else
            "PHASE9_SOURCE_PROGRAM_HETEROGENEITY_AUDIT_FAILED"
        ),
        "evidence_design": (
            "RETROSPECTIVE_CROSS_REPORT_AUDIT_OF_PROSPECTIVE_MATCHED_FORMAL_RUNS"
        ),
        "claim_boundary": (
            "Three distinct source-induced program bodies were selected from "
            "an eleven-program anonymous catalog for four target interfaces; "
            "each predicted program was already bound to a matched formal run "
            "where authentic binding beat its destructive source control. "
            "This validates content-specific applicability plus route-level "
            "utility for these registered interfaces. It does not prove that "
            "source provenance is necessary relative to an isomorphic "
            "target-written program, and it is not a new prospective target "
            "experiment."
        ),
        "source_catalog_size": len(catalog),
        "qualified_source_contracts": sum(
            row.source_intervention_qualified for row in catalog
        ),
        "selected_distinct_programs": len(selected_programs),
        "selected_program_sha256": sorted(selected_programs),
        "route_audits": route_audits,
        "descriptive_across_routes": {
            "wins": descriptive_wins,
            "losses": descriptive_losses,
            "ties": sum(row["ties"] for row in formal.values()),
            "pooled_iid_pvalue_reported": False,
            "reason": (
                "Tasks and interventions differ across domains; route-level "
                "paired tests remain the inferential units."
            ),
        },
        "source_lineage": {
            "phase6_file_sha256": _sha(PHASE6),
            "phase6_report_sha256": phase6["report_sha256"],
            "goal_acquisition_artifact_file_sha256": _sha(GOAL_ARTIFACT),
            "goal_acquisition_confirmation_file_sha256": _sha(
                GOAL_CONFIRMATION
            ),
            "goal_acquisition_contract": asdict(goal_contract),
        },
        "gates": gates,
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / (
            "docs/results/phase9_source_program_heterogeneity_v1.json"
        ),
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "source_catalog_size": report["source_catalog_size"],
        "selected_distinct_programs": report["selected_distinct_programs"],
        "descriptive_across_routes": report["descriptive_across_routes"],
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
