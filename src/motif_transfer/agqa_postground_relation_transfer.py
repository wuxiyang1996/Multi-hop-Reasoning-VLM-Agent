"""Post-grounding AGQA binding adapter for the induced relation program.

V29 intentionally treated every raw neural vote as a symbolic binding.  That
is too strict: votes are perception evidence, while the target-native grounder
is the authority that resolves evidence into zero, one, or many candidate
bindings.  This adapter presents only that resolved candidate set to the
source-induced transition/terminal/abstention interpreter.

The current AGQA consensus grounder emits either zero or one resolved binding.
Future grounders may expose an explicit ``candidate_bindings`` list, in which
case the source program's multiple-binding abstention remains active.
"""

from __future__ import annotations

from typing import Any, Mapping

from .agqa_goal_relation_transfer import (
    AGQAGoalRelationBindingReceipt,
    bind_source_goal_relation_program,
)
from .agqa_query_object_grounder import (
    AGQA_OBJECT_ONTOLOGY,
    canonical_object_label,
)


def bind_postground_source_program(
    *, artifact: Mapping[str, Any], confirmation: Mapping[str, Any],
    task_id: str, target_state_sha256: str, target_grounder_sha256: str,
    calibrated_execution: Mapping[str, Any], grounder_qualified: bool,
    effect_binding_authenticated: bool = True,
    formal_outcome_read: bool = False,
) -> AGQAGoalRelationBindingReceipt:
    """Execute source rules on resolved bindings, never on raw sensor votes."""

    explicit = calibrated_execution.get("candidate_bindings")
    if explicit is None:
        decision = canonical_object_label(str(
            calibrated_execution.get("decision") or ""
        ))
        bindings = [decision] if decision in AGQA_OBJECT_ONTOLOGY else []
    else:
        if not isinstance(explicit, list):
            raise ValueError("target candidate_bindings must be a list")
        bindings = sorted({
            label for value in explicit
            for label in (canonical_object_label(str(value)),)
            if label in AGQA_OBJECT_ONTOLOGY
        })
        decision = (
            bindings[0] if len(bindings) == 1 else ""
        )
    resolved_execution = {
        "decision": decision or None,
        "neural_votes": [
            {"view": f"resolved_binding_{index}", "decision": value}
            for index, value in enumerate(bindings)
        ],
    }
    return bind_source_goal_relation_program(
        artifact=artifact,
        confirmation=confirmation,
        task_id=task_id,
        target_state_sha256=target_state_sha256,
        target_grounder_sha256=target_grounder_sha256,
        calibrated_execution=resolved_execution,
        grounder_qualified=grounder_qualified,
        effect_binding_authenticated=effect_binding_authenticated,
        formal_outcome_read=formal_outcome_read,
    )


__all__ = ["bind_postground_source_program"]
