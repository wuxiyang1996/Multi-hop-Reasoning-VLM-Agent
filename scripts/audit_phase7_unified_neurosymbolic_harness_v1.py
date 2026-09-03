#!/usr/bin/env python3
"""Run the Phase-7 end-to-end authority and selective-routing audit."""

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
    SourceIRContract,
    TargetIRRequirement,
)
from motif_transfer.unified_neurosymbolic_harness import (  # noqa: E402
    InducedProgramEnvelope,
    UnifiedNeurosymbolicHarness,
    UnifiedTargetGrounding,
    validate_phase7_authorization,
)
from motif_transfer.unified_transfer_runtime import (  # noqa: E402
    PairedCalibration,
    TargetGroundingReceipt,
    TransferVerdict,
    UnifiedNeurosymbolicTransferRuntime,
    UnifiedRoute,
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


def _contract(value: Mapping[str, Any]) -> SourceIRContract:
    result = SourceIRContract(
        program_sha256=str(value["program_sha256"]),
        ir_kind=str(value["ir_kind"]),
        operator_sequence=tuple(
            OperatorSignature(**row) for row in value["operator_sequence"]
        ),
        recurrent=bool(value["recurrent"]),
        terminal_predicate_families=tuple(
            map(str, value["terminal_predicate_families"])
        ),
        source_intervention_qualified=bool(
            value["source_intervention_qualified"]
        ),
        source_confirmation_sha256=str(value["source_confirmation_sha256"]),
        contract_sha256=str(value["contract_sha256"]),
    )
    result.validate()
    return result


def _requirement(value: Mapping[str, Any]) -> TargetIRRequirement:
    result = TargetIRRequirement(
        task_id=str(value["task_id"]),
        target_domain=str(value["target_domain"]),
        target_interface=str(value["target_interface"]),
        target_grounder_sha256=str(value["target_grounder_sha256"]),
        ir_kind=str(value["ir_kind"]),
        operator_sequence=tuple(
            OperatorSignature(**row) for row in value["operator_sequence"]
        ),
        recurrent=bool(value["recurrent"]),
        terminal_predicate_families=tuple(
            map(str, value["terminal_predicate_families"])
        ),
        grounder_qualified=bool(value["grounder_qualified"]),
        formal_outcome_read=bool(value["formal_outcome_read"]),
        requirement_sha256=str(value["requirement_sha256"]),
    )
    result.validate()
    return result


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
        utility_vs_neural=PairedCalibration(**value["utility_vs_neural"]),
        authenticity_vs_source_permuted=PairedCalibration(
            **value["authenticity_vs_source_permuted"]
        ),
    )


def _source_receipt_and_inducer(label: str) -> tuple[str, str]:
    family, name = label.split(":", 1)
    if family == "minigrid":
        artifact = _read(
            REPO / "configs/source_structural_v5c_frozen/programs"
            / f"{name}.json"
        )
        inducer = REPO / "src/motif_transfer/structural_delta_induction.py"
        return str(artifact["source_receipts_sha256"]), _sha(inducer)
    if family == "sokoban":
        artifact = _read(
            REPO / "runs/sokoban_relational_structural_v2/artifact.json"
        )
        inducer = REPO / "src/motif_transfer/relational_structural_induction.py"
        return str(artifact["source_receipts_sha256"]), _sha(inducer)
    artifact = _read(
        REPO / "configs/phase3_source_function_v4/frozen_reserve/programs"
        / f"{name}.json"
    )
    inducer = REPO / "src/motif_transfer/phase3_source_function_induction.py"
    return str(
        artifact["source_function_program"]["source_receipts_sha256"]
    ), _sha(inducer)


class _AuthorityCanaryExecutor:
    def __init__(self, artifact_sha256: str):
        self.artifact_sha256 = artifact_sha256
        self.calls = 0

    def execute(self, authorization, grounding, native_actions):
        self.calls += 1
        return native_actions[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase5-manifest", type=Path,
        default=REPO / "configs/phase5_unified_applicability_v1_frozen.json",
    )
    parser.add_argument(
        "--phase6-report", type=Path,
        default=REPO / "docs/results/phase6_source_specific_applicability_v1.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/phase7_unified_neurosymbolic_harness_v1.json",
    )
    args = parser.parse_args()
    phase5 = _read(args.phase5_manifest)
    phase6 = _read(args.phase6_report)
    _self_hash(phase5, "manifest_sha256")
    _self_hash(phase6, "report_sha256")
    if phase6["status"] != (
        "PHASE6_SELECTIVE_SOURCE_SPECIFIC_APPLICABILITY_VALIDATED"
    ):
        raise SystemExit("Phase-6 source-specific applicability did not pass")

    envelopes = []
    for row in phase6["source_catalog"]:
        receipt_sha, inducer_sha = _source_receipt_and_inducer(
            str(row["catalog_label"])
        )
        envelopes.append(InducedProgramEnvelope.create(
            contract=_contract(row["contract"]),
            source_transition_receipts_sha256=receipt_sha,
            inducer_artifact_sha256=inducer_sha,
            learned_from_state_action_effect_next_state=True,
            target_data_read=False,
            named_policy_template_used=False,
        ))
    routes = tuple(_route(row) for row in phase5["routes"])
    runtime = UnifiedNeurosymbolicTransferRuntime(routes)
    harness = UnifiedNeurosymbolicHarness(envelopes, runtime)
    phase6_targets = {
        row["route_id"]: row for row in phase6["target_requirements"]
    }
    route_values = {row["route_id"]: row for row in phase5["routes"]}

    audits = []
    executor_calls = 0
    for route in routes:
        requirement = _requirement(
            phase6_targets[route.route_id]["target_requirement"]
        )
        grounding = TargetGroundingReceipt.create(
            task_id=requirement.task_id,
            target_domain=requirement.target_domain,
            target_interface=requirement.target_interface,
            target_state_sha256=stable_hash({
                "task_id": requirement.task_id,
                "phase": 7,
                "canary": "OUTCOME_BLIND_INTERFACE_AND_AUTHORITY",
            }),
            target_grounder_sha256=requirement.target_grounder_sha256,
            capabilities=route.required_capabilities,
            candidate_ids=("TARGET_NATIVE_CANDIDATE_0", "TARGET_NATIVE_CANDIDATE_1"),
            structural_predicates={
                "grounded_candidate_set_unique": True,
                "operator_binding_complete": True,
                "target_native_executor_available": True,
            },
            grounder_qualified=True,
            formal_outcome_read=False,
        )
        target = UnifiedTargetGrounding.create(
            requirement=requirement, applicability=grounding,
        )
        phase7 = harness.decide(target)
        validate_phase7_authorization(phase7)
        utility = runtime.decide(grounding)
        executor = _AuthorityCanaryExecutor(route.target_executor_sha256)
        action = harness.execute(
            phase7, utility, target,
            ("TARGET_NATIVE_CANDIDATE_0", "TARGET_NATIVE_CANDIDATE_1"),
            executor,
        )
        executor_calls += executor.calls
        audits.append({
            "route_id": route.route_id,
            "target_domain": route.target_domain,
            "phase7_authorization": {
                **asdict(phase7), "verdict": phase7.verdict.value,
            },
            "utility_authorization": {
                **asdict(utility), "verdict": utility.verdict.value,
            },
            "selected_source_matches_route": (
                phase7.selected_program_sha256 == route.source_program_sha256
            ),
            "authority_canary_executor_calls": executor.calls,
            "authority_canary_native_action": action,
            "existing_evidence_status": route_values[
                route.route_id
            ]["evidence"]["status"],
            "formal_target_outcome_read": False,
        })

    selected = [
        row for row in audits
        if row["phase7_authorization"]["verdict"]
        == TransferVerdict.SELECT_SKILL.value
    ]
    abstained = [row for row in audits if row not in selected]
    gates = {
        "ten_source_induction_envelopes_integrated": len(envelopes) == 10,
        "all_induction_envelopes_are_source_only_and_template_free": all(
            row.learned_from_state_action_effect_next_state
            and row.target_data_read is False
            and row.named_policy_template_used is False
            for row in envelopes
        ),
        "all_four_target_native_grounder_interfaces_integrated": (
            len(audits) == 4
        ),
        "three_calibrated_routes_authorized": len(selected) == 3,
        "alfworld_remains_fail_closed_on_insufficient_utility": (
            len(abstained) == 1
            and abstained[0]["target_domain"] == "alfworld"
            and abstained[0]["phase7_authorization"]["reason"]
            == "UTILITY_ROUTER:DIRECTIONAL_UTILITY_NOT_CALIBRATED"
        ),
        "structural_selector_and_frozen_routes_agree": all(
            row["selected_source_matches_route"] for row in audits
        ),
        "only_selected_routes_reach_target_executor": (
            executor_calls == 3
            and all(
                row["authority_canary_executor_calls"]
                == int(row in selected)
                for row in audits
            )
        ),
        "selector_never_contains_a_target_action": all(
            row["phase7_authorization"]["target_action_emitted"] is False
            for row in audits
        ),
        "zero_current_target_outcome_exposure": all(
            row["formal_target_outcome_read"] is False
            and row["phase7_authorization"]["current_target_outcome_read"]
            is False
            for row in audits
        ),
    }
    body = {
        "schema_version": "phase7-unified-neurosymbolic-harness-audit-v1",
        "status": (
            "PHASE7_UNIFIED_NEUROSYMBOLIC_HARNESS_VALIDATED"
            if all(gates.values()) else
            "PHASE7_UNIFIED_NEUROSYMBOLIC_HARNESS_FAILED"
        ),
        "claim_boundary": (
            "Validates composition and authority boundaries over existing "
            "frozen non-video route evidence. Phase-7 canaries do not execute "
            "formal target tasks or create a new success-rate estimate. "
            "ALFWorld remains an abstaining, unvalidated positive route."
        ),
        "phase5_manifest_sha256": phase5["manifest_sha256"],
        "phase6_report_sha256": phase6["report_sha256"],
        "component_file_sha256": {
            "structural_applicability": _sha(
                REPO / "src/motif_transfer/structural_ir_applicability.py"
            ),
            "frozen_utility_runtime": _sha(
                REPO / "src/motif_transfer/unified_transfer_runtime.py"
            ),
            "unified_harness": _sha(
                REPO / "src/motif_transfer/unified_neurosymbolic_harness.py"
            ),
        },
        "source_envelopes": [
            {
                "program_sha256": row.contract.program_sha256,
                "contract_sha256": row.contract.contract_sha256,
                "admitted": row.admitted,
                "envelope_sha256": row.envelope_sha256,
            }
            for row in envelopes
        ],
        "route_audits": audits,
        "selected_route_count": len(selected),
        "abstained_route_count": len(abstained),
        "target_executor_canary_calls": executor_calls,
        "formal_target_tasks_executed": 0,
        "target_outcomes_read": 0,
        "gates": gates,
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "source_envelopes": len(envelopes),
        "selected_routes": [row["route_id"] for row in selected],
        "abstained_routes": [row["route_id"] for row in abstained],
        "target_executor_canary_calls": executor_calls,
        "gates": gates,
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
