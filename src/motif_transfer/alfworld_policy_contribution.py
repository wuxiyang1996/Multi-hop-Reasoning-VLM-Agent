"""Causal policy-contribution audit for ALFWorld symbolic transfer.

The source program is allowed to select an anonymous operator/option.  It is
not allowed to emit an ALFWorld action.  The target-native grounder and
executor retain authority over the concrete action.  This module verifies
that this separation holds and that successful rescues contain an observed
source-controlled policy divergence before the terminal relation transition.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from motif_transfer.alfworld_goal_acquisition_v10 import (
    AUTHENTIC,
    CARDINALITY_CONTROL,
    CEILING,
    EFFECT_CONTROL,
    GENERIC,
    RAW,
)


SOURCE_ACQUISITION_DIAGNOSTIC = (
    "SOURCE_INDUCED_ACQUISITION_OPERATOR_GROUNDED"
)


def _by_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed = {str(row["task_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate ALFWorld task identity")
    return indexed


def _actions(episode: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(map(str, episode["actions"]))


def audit_policy_contribution(report: Mapping[str, Any]) -> dict[str, Any]:
    """Audit whether the source IR has causal option/action-selection value."""
    episodes = report.get("episodes")
    if not isinstance(episodes, Mapping):
        raise ValueError("ALFWorld report has no matched episode matrix")
    required = (RAW, AUTHENTIC, CARDINALITY_CONTROL, EFFECT_CONTROL, GENERIC, CEILING)
    if any(condition not in episodes for condition in required):
        raise ValueError("ALFWorld report is missing a required condition")

    indexed = {
        condition: _by_task(episodes[condition])
        for condition in required
    }
    task_ids = set(indexed[RAW])
    matched = all(set(indexed[condition]) == task_ids for condition in required)
    if not matched:
        raise ValueError("ALFWorld condition task identities are not matched")

    authority = report.get("authority_receipts")
    if not isinstance(authority, Mapping):
        raise ValueError("ALFWorld report has no authority receipts")

    task_audits: list[dict[str, Any]] = []
    for task_id in sorted(task_ids):
        raw = indexed[RAW][task_id]
        authentic = indexed[AUTHENTIC][task_id]
        records = list(authentic["records"])
        program_records = [row for row in records if bool(row["program_active"])]
        source_records = [row for row in records if bool(row["source_admitted"])]
        source_divergences = [
            row for row in source_records if bool(row["changed_action_vs_raw"])
        ]
        acquisition_divergences = [
            row for row in source_divergences
            if str(row["diagnostic"]) == SOURCE_ACQUISITION_DIAGNOSTIC
        ]
        terminal_transitions = [
            row for row in source_records
            if bool(row["source_transition_advanced"])
            and int(row["completed_count_after"])
            > int(row["completed_count_before"])
        ]
        receipts = list(authority.get(task_id, ()))
        receipt_alignment = (
            len(receipts) == len(program_records)
            and all(
                str(receipt["target_native_action"])
                == str(record["selected_action"])
                and receipt["source_selector_action_emitted"] is False
                and int(receipt["target_executor_calls"]) == 1
                and receipt["formal_outcome_read"] is False
                for record, receipt in zip(program_records, receipts)
            )
        )
        rescued = (
            bool(authentic["official_success"])
            and not bool(raw["official_success"])
        )
        final_transition_step = (
            max(int(row["step"]) for row in terminal_transitions)
            if terminal_transitions else None
        )
        acquisition_before_terminal = bool(
            final_transition_step is not None
            and any(
                int(row["step"]) < final_transition_step
                for row in acquisition_divergences
            )
        )
        terminal_source_success = bool(
            terminal_transitions
            and any(bool(row["official_success_after"]) for row in terminal_transitions)
        )
        task_audits.append({
            "task_id": task_id,
            "rescued_vs_neural_only": rescued,
            "authentic_success": bool(authentic["official_success"]),
            "neural_only_success": bool(raw["official_success"]),
            "program_active_actions": len(program_records),
            "source_admitted_actions": len(source_records),
            "source_divergent_actions": len(source_divergences),
            "source_acquisition_divergences": len(acquisition_divergences),
            "source_terminal_transitions": len(terminal_transitions),
            "first_source_divergence_step": (
                min(int(row["step"]) for row in source_divergences)
                if source_divergences else None
            ),
            "terminal_transition_step": final_transition_step,
            "source_activation_after_first_relation": all(
                int(row["completed_count_before"]) >= 1
                for row in program_records
            ),
            "acquisition_divergence_before_terminal": acquisition_before_terminal,
            "terminal_source_transition_reaches_success": terminal_source_success,
            "target_native_authority_receipts_align": receipt_alignment,
        })

    rescues = [row for row in task_audits if row["rescued_vs_neural_only"]]
    regressions = [
        task_id for task_id in sorted(task_ids)
        if not bool(indexed[AUTHENTIC][task_id]["official_success"])
        and bool(indexed[RAW][task_id]["official_success"])
    ]
    source_divergent_actions = sum(
        int(row["source_divergent_actions"]) for row in task_audits
    )
    control_trace_matches = {
        condition: sum(
            _actions(indexed[condition][task_id]) == _actions(indexed[RAW][task_id])
            for task_id in task_ids
        )
        for condition in (CARDINALITY_CONTROL, EFFECT_CONTROL, GENERIC)
    }
    authentic_ceiling_matches = sum(
        _actions(indexed[AUTHENTIC][task_id]) == _actions(indexed[CEILING][task_id])
        for task_id in task_ids
    )
    gates = {
        "complete_matched_task_matrix": matched,
        "source_activates_only_after_observed_first_relation": all(
            row["source_activation_after_first_relation"] for row in task_audits
        ),
        "source_changes_target_policy_actions": source_divergent_actions > 0,
        "every_rescue_has_source_acquisition_divergence_before_terminal": (
            bool(rescues)
            and all(row["acquisition_divergence_before_terminal"] for row in rescues)
        ),
        "every_rescue_has_source_transition_to_success": (
            bool(rescues)
            and all(row["terminal_source_transition_reaches_success"] for row in rescues)
        ),
        "source_never_emits_target_action": all(
            row["target_native_authority_receipts_align"] for row in task_audits
        ),
        "controls_reproduce_neural_only_action_traces": all(
            count == len(task_ids) for count in control_trace_matches.values()
        ),
        "authentic_reproduces_target_native_ceiling_action_traces": (
            authentic_ceiling_matches == len(task_ids)
        ),
        "zero_negative_transfer": not regressions,
    }
    return {
        "tasks": len(task_ids),
        "rescues": len(rescues),
        "regressions": len(regressions),
        "source_divergent_actions": source_divergent_actions,
        "source_acquisition_divergences": sum(
            int(row["source_acquisition_divergences"]) for row in task_audits
        ),
        "source_terminal_transitions": sum(
            int(row["source_terminal_transitions"]) for row in task_audits
        ),
        "program_active_actions": sum(
            int(row["program_active_actions"]) for row in task_audits
        ),
        "target_native_authority_receipts": sum(
            int(row["program_active_actions"]) for row in task_audits
        ),
        "control_exact_trace_matches": control_trace_matches,
        "authentic_ceiling_exact_trace_matches": authentic_ceiling_matches,
        "rescued_task_audits": rescues,
        "gates": gates,
    }


__all__ = ["SOURCE_ACQUISITION_DIAGNOSTIC", "audit_policy_contribution"]
