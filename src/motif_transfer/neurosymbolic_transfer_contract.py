"""Fail-closed evidence contracts for neural-symbolic transfer.

An action-imitation score and a causal intervention-effect probability are
different statistical objects.  This module keeps that distinction explicit
and prevents source conformal bounds from being reused under an uncertified
target covariate shift.
"""

from __future__ import annotations

from typing import Any, Mapping


IMITATION_SCORE_SEMANTICS = "expert_action_imitation_score"
CAUSAL_EFFECT_SCORE_SEMANTICS = "causal_successor_event_probability"
SOURCE_TARGET_SUPPORT_GATE_PASSED = "SOURCE_TARGET_SUPPORT_GATE_PASSED"


def target_score_contract(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Return an explicit contract, conservatively interpreting legacy artifacts."""
    declared = artifact.get("score_contract")
    contract = dict(declared) if isinstance(declared, Mapping) else {}
    if "score_semantics" not in contract:
        if artifact.get("training_supervision") == "expert_action_identity_only":
            contract["score_semantics"] = IMITATION_SCORE_SEMANTICS
        else:
            contract["score_semantics"] = "UNDECLARED"
    contract.setdefault("causal_successor_effect_certified", False)
    contract.setdefault("counterfactual_action_supervision", False)
    contract.setdefault("probability_calibration", "UNDECLARED")
    contract.setdefault("entity_conditioned_action_binding", False)
    contract.setdefault("successor_event_prediction", False)
    contract["legacy_contract_inferred"] = not isinstance(declared, Mapping)
    return contract


def target_grounder_contract_violations(
    artifact: Mapping[str, Any],
) -> tuple[str, ...]:
    contract = target_score_contract(artifact)
    violations: list[str] = []
    if contract["score_semantics"] != CAUSAL_EFFECT_SCORE_SEMANTICS:
        violations.append("TARGET_SCORE_IS_NOT_CAUSAL_SUCCESSOR_EFFECT")
    if contract["causal_successor_effect_certified"] is not True:
        violations.append("TARGET_CAUSAL_EFFECT_NOT_CERTIFIED")
    if contract["counterfactual_action_supervision"] is not True:
        violations.append("TARGET_HAS_NO_COUNTERFACTUAL_ACTION_SUPERVISION")
    if str(contract["probability_calibration"]).upper() in {
        "UNDECLARED",
        "NONE",
        "NONE_CLASS_BALANCED_NEGATIVE_SAMPLED",
    }:
        violations.append("TARGET_EFFECT_PROBABILITY_NOT_CALIBRATED")
    if contract["entity_conditioned_action_binding"] is not True:
        violations.append("TARGET_BINDING_NOT_ENTITY_CONDITIONED")
    if contract["successor_event_prediction"] is not True:
        violations.append("TARGET_SUCCESSOR_EVENT_NOT_PREDICTED")
    return tuple(violations)


def require_causal_target_grounder(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject a target scorer unless it can populate causal source features."""
    violations = target_grounder_contract_violations(artifact)
    if violations:
        raise ValueError(
            "target grounder cannot populate causal-effect features: "
            + ",".join(violations)
        )
    return target_score_contract(artifact)


def support_receipt_violations(
    receipt: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if not isinstance(receipt, Mapping):
        return ("SOURCE_TARGET_SUPPORT_RECEIPT_MISSING",)
    violations: list[str] = []
    if receipt.get("status") != SOURCE_TARGET_SUPPORT_GATE_PASSED:
        violations.append("SOURCE_TARGET_SUPPORT_GATE_NOT_PASSED")
    if receipt.get("score_semantics") != CAUSAL_EFFECT_SCORE_SEMANTICS:
        violations.append("SOURCE_TARGET_SCORE_SEMANTICS_MISMATCH")
    if receipt.get("exchangeability_or_covariate_coverage_certified") is not True:
        violations.append("SOURCE_CONFORMAL_HAS_NO_TARGET_COVARIATE_COVERAGE")
    if int(receipt.get("source_calibration_states", 0)) <= 0:
        violations.append("SOURCE_CALIBRATION_SUPPORT_EMPTY")
    if int(receipt.get("target_adaptation_states", 0)) <= 0:
        violations.append("TARGET_ADAPTATION_SUPPORT_EMPTY")
    return tuple(violations)


def require_source_target_support(
    receipt: Mapping[str, Any] | None,
) -> None:
    """Require a joint support receipt before applying source conformal bounds."""
    violations = support_receipt_violations(receipt)
    if violations:
        raise ValueError(
            "source conformal admission is invalid for this target: "
            + ",".join(violations)
        )


def transfer_contract_audit(
    *,
    target_grounder: Mapping[str, Any],
    source_target_support_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target_violations = target_grounder_contract_violations(target_grounder)
    support_violations = support_receipt_violations(
        source_target_support_receipt
    )
    violations = (*target_violations, *support_violations)
    return {
        "status": (
            "NEUROSYMBOLIC_TRANSFER_CONTRACT_PASSED"
            if not violations
            else "NEUROSYMBOLIC_TRANSFER_CONTRACT_REJECTED"
        ),
        "target_score_contract": target_score_contract(target_grounder),
        "violations": list(violations),
    }


__all__ = [
    "CAUSAL_EFFECT_SCORE_SEMANTICS",
    "IMITATION_SCORE_SEMANTICS",
    "SOURCE_TARGET_SUPPORT_GATE_PASSED",
    "require_causal_target_grounder",
    "require_source_target_support",
    "support_receipt_violations",
    "target_grounder_contract_violations",
    "target_score_contract",
    "transfer_contract_audit",
]
