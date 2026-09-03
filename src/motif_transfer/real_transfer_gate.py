from __future__ import annotations

from typing import Any, Mapping


def finalize_transfer_gate(
    source_receipt_gate: Mapping[str, Any],
    source_grounder_gate: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_passed = source_receipt_gate.get("status") == "SOURCE_GATE_PASSED"
    grounder_passed = (
        source_grounder_gate.get("status") == "SOURCE_GROUNDER_GATE_PASSED"
    )
    authorized = receipt_passed and grounder_passed
    if not receipt_passed:
        status = "TRANSFER_BLOCKED_AT_SOURCE_RECEIPTS"
        reason = "source interventions were not reproducible or action-dependent"
    elif not grounder_passed:
        status = "TRANSFER_BLOCKED_AT_SOURCE_GROUNDER"
        reason = (
            "source action values were intervention-real but not predictably grounded "
            "from held-out observable state/action features against controls"
        )
    else:
        status = "TARGET_FOUR_CONDITION_RUN_AUTHORIZED"
        reason = "both source receipt and observable-grounding gates passed"
    return {
        "schema_version": "real-game-to-target-transfer-gate-v1",
        "status": status,
        "reason": reason,
        "source_receipt_gate_passed": receipt_passed,
        "source_grounder_gate_passed": grounder_passed,
        "target_execution_authorized": authorized,
        "required_target_conditions": [
            "target_only",
            "authentic_source_structure_plus_target_native_grounder",
            "within_state_shuffled_source_plus_target_native_grounder",
            "source_marginal_plus_target_native_grounder",
        ],
        "conditions_executed": [] if not authorized else None,
        "cross_domain_transfer_supported": False,
        "claim_boundary": (
            "Passing source gates would authorize, not establish, a target transfer test. "
            "Cross-domain transfer requires paired target results."
        ),
    }


__all__ = ["finalize_transfer_gate"]
