#!/usr/bin/env python3
"""Audit calibrated selective routing on the frozen Phase-5 probes."""

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
from motif_transfer.unified_transfer_runtime import (  # noqa: E402
    PairedCalibration,
    TargetGroundingReceipt,
    TransferVerdict,
    UnifiedNeurosymbolicTransferRuntime,
    UnifiedRoute,
    validate_authorization,
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


def _calibration(value: Mapping[str, Any]) -> PairedCalibration:
    return PairedCalibration(
        wins=int(value["wins"]), losses=int(value["losses"]),
        ties=int(value["ties"]),
    )


def _route(value: Mapping[str, Any]) -> UnifiedRoute:
    return UnifiedRoute(
        route_id=str(value["route_id"]),
        target_domain=str(value["target_domain"]),
        target_interface=str(value["target_interface"]),
        required_capabilities=tuple(map(str, value["required_capabilities"])),
        source_program_sha256=str(value["source_program_sha256"]),
        source_program_induced_from_interventions=bool(
            value["source_program_induced_from_interventions"]
        ),
        source_program_qualified=bool(value["source_program_qualified"]),
        target_grounder_sha256=str(value["target_grounder_sha256"]),
        target_executor_sha256=str(value["target_executor_sha256"]),
        target_grounder_id=str(value["target_grounder_id"]),
        target_executor_id=str(value["target_executor_id"]),
        evidence_report_sha256=str(value["evidence_report_sha256"]),
        utility_vs_neural=_calibration(value["utility_vs_neural"]),
        authenticity_vs_source_permuted=_calibration(
            value["authenticity_vs_source_permuted"]
        ),
    )


def _grounding(
    *, task_id: str, route: UnifiedRoute | None,
    domain: str, interface: str,
) -> TargetGroundingReceipt:
    if route is None:
        return TargetGroundingReceipt.create(
            task_id=task_id,
            target_domain=domain,
            target_interface=interface,
            target_state_sha256=stable_hash({
                "task_id": task_id,
                "probe": "OUTCOME_BLIND_INTERFACE_DESCRIPTOR_ONLY",
            }),
            target_grounder_sha256=stable_hash({
                "unregistered_interface": interface,
            }),
            capabilities=("unregistered_target_interface",),
            candidate_ids=(),
            structural_predicates={"exact_registered_interface": False},
            grounder_qualified=False,
            formal_outcome_read=False,
        )
    return TargetGroundingReceipt.create(
        task_id=task_id,
        target_domain=domain,
        target_interface=interface,
        target_state_sha256=stable_hash({
            "task_id": task_id,
            "probe": "OUTCOME_BLIND_STRUCTURALLY_APPLICABLE_CANARY",
            "route_id": route.route_id,
        }),
        target_grounder_sha256=route.target_grounder_sha256,
        capabilities=route.required_capabilities,
        candidate_ids=("TARGET_NATIVE_CANDIDATE_0", "TARGET_NATIVE_CANDIDATE_1"),
        structural_predicates={
            "candidate_set_multiple_and_unique": True,
            "executor_contract_available": True,
            "target_relation_schema_complete": True,
            "terminal_binding_unique": True,
        },
        grounder_qualified=True,
        formal_outcome_read=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase5_unified_applicability_v1_frozen.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/phase5_unified_applicability_v1_audit.json",
    )
    args = parser.parse_args()
    manifest = _read(args.manifest)
    _self_hash(manifest, "manifest_sha256")
    if manifest.get("status") != "FROZEN_BEFORE_ANY_PROBE_TARGET_RESET_OR_OUTCOME":
        raise SystemExit("Phase-5 probes are not in their frozen pre-execution state")
    if _sha(REPO / "src/motif_transfer/unified_transfer_runtime.py") != (
        manifest["integrity"]["runtime_file_sha256"]
    ):
        raise SystemExit("unified runtime changed after probe freeze")

    expected_evidence_statuses = {
        "sokoban-relational-to-webshop-v21": (
            "V21_FRESH_FORMAL_STRUCTURAL_TRANSFER_VALIDATED"
        ),
        "minigrid-put-near-to-discoveryworld-easy-v1": (
            "DISCOVERYWORLD_STRUCTURAL_TRANSFER_VALIDATED"
        ),
        "sokoban-relational-to-tir-maze-v3": (
            "FRESH_FORMAL_STRUCTURAL_TRANSFER_VALIDATED"
        ),
        "minigrid-put-near-to-alfworld-multiplicity-v2": (
            "ALFWORLD_STRUCTURAL_TRANSFER_FAILED"
        ),
    }
    routes = []
    evidence_integrity = {}
    for row in manifest["routes"]:
        evidence = row["evidence"]
        path = REPO / str(evidence["path"])
        file_ok = _sha(path) == evidence["file_sha256"]
        status_ok = evidence["status"] == expected_evidence_statuses[row["route_id"]]
        evidence_integrity[row["route_id"]] = {
            "file_hash_valid": file_ok,
            "status_valid": status_ok,
        }
        if not file_ok or not status_ok:
            raise SystemExit(f"route evidence drift: {row['route_id']}")
        routes.append(_route(row))
    runtime = UnifiedNeurosymbolicTransferRuntime(routes)
    by_id = {route.route_id: route for route in routes}

    probes = []

    def evaluate(grounding: TargetGroundingReceipt, *, probe_kind: str) -> None:
        authorization = runtime.decide(grounding)
        validate_authorization(authorization)
        probes.append({
            "task_id": grounding.task_id,
            "probe_kind": probe_kind,
            "target_domain": grounding.target_domain,
            "target_interface": grounding.target_interface,
            "grounding_receipt_sha256": grounding.receipt_sha256,
            "grounding_formal_outcome_read": grounding.formal_outcome_read,
            "authorization": {
                **asdict(authorization),
                "verdict": authorization.verdict.value,
            },
            "selector_emitted_target_action": hasattr(authorization, "action"),
        })

    # Positive route canaries bind future state descriptors only.  They do not
    # open a target task or claim a future outcome.
    for route_id, task_id in (
        ("sokoban-relational-to-webshop-v21", "webshop.future.option-relation"),
        ("sokoban-relational-to-tir-maze-v3", "tir.future.single-image-maze"),
    ):
        route = by_id[route_id]
        evaluate(_grounding(
            task_id=task_id, route=route, domain=route.target_domain,
            interface=route.target_interface,
        ), probe_kind="REGISTERED_ROUTE_STRUCTURAL_CANARY")

    dw_route = by_id["minigrid-put-near-to-discoveryworld-easy-v1"]
    for row in manifest["future_probes"]["discoveryworld"]:
        interface = str(row["target_interface"])
        route = dw_route if interface == dw_route.target_interface else None
        evaluate(_grounding(
            task_id=str(row["task_id"]), route=route,
            domain="discoveryworld", interface=interface,
        ), probe_kind=(
            "UNOPENED_EXACT_INTERFACE_STRUCTURAL_CANARY" if route else
            "UNOPENED_INTERFACE_MISMATCH"
        ))

    alf_route = by_id["minigrid-put-near-to-alfworld-multiplicity-v2"]
    for row in manifest["future_probes"]["alfworld"]:
        evaluate(_grounding(
            task_id=str(row["task_id"]), route=alf_route,
            domain="alfworld", interface=alf_route.target_interface,
        ), probe_kind="EXECUTION_UNTOUCHED_STRUCTURAL_CANARY")

    selected = [row for row in probes if (
        row["authorization"]["verdict"] == TransferVerdict.SELECT_SKILL.value
    )]
    abstained = [row for row in probes if row not in selected]
    selected_routes = sorted({
        row["authorization"]["route_id"] for row in selected
    })
    abstention_reasons: dict[str, int] = {}
    for row in abstained:
        reason = str(row["authorization"]["reason"])
        abstention_reasons[reason] = abstention_reasons.get(reason, 0) + 1
    gates = {
        "manifest_and_evidence_hashes_valid": all(
            all(values.values()) for values in evidence_integrity.values()
        ),
        "three_positive_route_families_selected": selected_routes == [
            "minigrid-put-near-to-discoveryworld-easy-v1",
            "sokoban-relational-to-tir-maze-v3",
            "sokoban-relational-to-webshop-v21",
        ],
        "new_discoveryworld_easy_interface_selected": sum(
            row["task_id"].startswith("proteomics.easy.seed")
            and row["authorization"]["verdict"] == "SELECT_SKILL"
            for row in probes
        ) == 2,
        "new_discoveryworld_normal_interfaces_abstained": sum(
            row["target_domain"] == "discoveryworld"
            and ".normal." in row["task_id"]
            and row["authorization"]["verdict"] == "ABSTAIN"
            for row in probes
        ) == 4,
        "all_eight_execution_untouched_alfworld_tasks_abstained": sum(
            row["target_domain"] == "alfworld"
            and row["authorization"]["verdict"] == "ABSTAIN"
            for row in probes
        ) == 8,
        "alfworld_abstention_is_calibrated_not_interface_failure": all(
            row["authorization"]["reason"] == "DIRECTIONAL_UTILITY_NOT_CALIBRATED"
            for row in probes if row["target_domain"] == "alfworld"
        ),
        "zero_current_outcome_exposure": all(
            row["grounding_formal_outcome_read"] is False
            and row["authorization"]["current_outcome_read"] is False
            for row in probes
        ),
        "selector_never_emits_target_action": not any(
            row["selector_emitted_target_action"] for row in probes
        ),
    }
    body = {
        "schema_version": "phase5-unified-applicability-audit-v1",
        "status": (
            "PHASE5_FROZEN_SELECTIVE_ROUTING_VALIDATED" if all(gates.values())
            else "PHASE5_FROZEN_SELECTIVE_ROUTING_FAILED"
        ),
        "claim_boundary": (
            "Pre-execution applicability and route-utility validation on frozen "
            "future task descriptors. No success-rate claim is made for the "
            "unopened DiscoveryWorld or ALFWorld tasks."
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "routes": {
            route.route_id: {
                "utility_vs_neural": asdict(route.utility_vs_neural),
                "authenticity_vs_source_permuted": asdict(
                    route.authenticity_vs_source_permuted
                ),
            }
            for route in routes
        },
        "selected_route_families": selected_routes,
        "selected_probe_count": len(selected),
        "abstained_probe_count": len(abstained),
        "abstention_reasons": dict(sorted(abstention_reasons.items())),
        "gates": gates,
        "probes": probes,
        "target_tasks_reset": 0,
        "target_outcomes_read": 0,
        "evidence_integrity": evidence_integrity,
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "selected_route_families": selected_routes,
        "selected_probe_count": len(selected),
        "abstained_probe_count": len(abstained),
        "abstention_reasons": abstention_reasons,
        "gates": gates,
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
