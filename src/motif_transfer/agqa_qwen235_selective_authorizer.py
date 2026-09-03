"""Outcome-blind selective authorization for the frozen Qwen235 AGQA grounder.

The rule is intentionally narrow.  It authorizes only recurrent relation
observations supported by two scans without a conflict tiebreak.  Duration and
ordering decisions remain with the frozen target actor.  The rule consumes no
answer, functional program, scene graph, or source identity.
"""

from __future__ import annotations

from typing import Any, Mapping


def authorize_source_override(runtime: Mapping[str, Any]) -> dict[str, Any]:
    plan = runtime["query_plan"]
    execution = runtime["target_native_execution"]
    direct = str(runtime["direct_response"])
    decision = execution.get("decision")
    canonicalizations = tuple(runtime["grounding_receipt"].get("canonicalizations", ()))
    operand_runs = tuple(runtime["operand_runs"].values())

    reasons: list[str] = []
    if plan.get("obligation_kind") != "RELATION_RECURRENT":
        reasons.append("NON_RELATION_RECURRENT_ROUTE")
    if plan.get("comparison") != "EXISTS":
        reasons.append("NON_EXISTS_COMPARISON")
    if decision != "yes":
        reasons.append("NO_POSITIVE_OBSERVATION")
    if not any("DOUBLE_SCAN_CONFIRMED_OBSERVED" in item for item in canonicalizations):
        reasons.append("NO_DOUBLE_SCAN_OBSERVED_SUPPORT")
    if any(bool(row.get("tiebreak_triggered")) for row in operand_runs):
        reasons.append("CONFLICT_TIEBREAK_USED")

    authorized = not reasons
    body = {
        "schema_version": "agqa-qwen235-selective-authorization-v1",
        "authorized": authorized,
        "decision": decision if authorized else None,
        "fallback": direct,
        "prediction": decision if authorized else direct,
        "reasons": reasons or ["RECURRENT_RELATION_DOUBLE_SCAN_WITHOUT_TIEBREAK"],
        "answer_read": False,
        "functional_program_read": False,
        "scene_graph_read": False,
        "source_identity_read": False,
    }
    return body
