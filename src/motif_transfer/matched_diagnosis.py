from __future__ import annotations

from typing import Any, Mapping


def diagnose_matched_pair(
    baseline: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    identity = {
        "initial_state_hash_match": baseline.get("initial_state_hash") == treatment.get("initial_state_hash"),
        "resolved_game_file_match": baseline.get("resolved_game_file") == treatment.get("resolved_game_file"),
        "decision_cache_shared": baseline.get("decision_backend") == treatment.get("decision_backend"),
        "treatment_decision_calls_all_cache_hit": all(
            bool(row.get("usage", {}).get("cache_hit"))
            for row in treatment.get("decision_call_receipts", [])
        ),
    }
    matched = all(identity.values())
    bindings = treatment.get("bindings") or []
    evidence = treatment.get("binding_evidence") or []
    fallback = treatment.get("source_fallback_step")
    same_actions = baseline.get("actions") == treatment.get("actions")
    baseline_metrics = baseline.get("metrics") or {}
    treatment_metrics = treatment.get("metrics") or {}
    score_delta = float(treatment_metrics.get("official_score", 0.0)) - float(
        baseline_metrics.get("official_score", 0.0)
    )
    if not matched:
        status = "UNMATCHED_INVALID"
    elif not bindings:
        status = "BINDING_REJECTED_SAFE_FALLBACK"
    elif fallback == 0 and not evidence:
        status = "NO_VALID_SOURCE_INTERVENTION"
    elif same_actions:
        status = "NO_OBSERVED_POLICY_EFFECT"
    elif score_delta > 0:
        status = "POSITIVE_TRANSFER_PILOT"
    elif score_delta < 0:
        status = "NEGATIVE_TRANSFER_PILOT"
    else:
        status = "BEHAVIOR_CHANGED_OUTCOME_TIED"
    return {
        "status": status,
        "matched": matched,
        "identity": identity,
        "actions_exact_match": same_actions,
        "binding_count": len(bindings),
        "binding_evidence_count": len(evidence),
        "source_fallback_step": fallback,
        "source_failures": list(treatment.get("source_failures") or []),
        "baseline_metrics": baseline_metrics,
        "treatment_metrics": treatment_metrics,
        "official_score_delta": score_delta,
    }
