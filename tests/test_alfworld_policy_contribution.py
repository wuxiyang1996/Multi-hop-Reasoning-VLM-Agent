from copy import deepcopy

from motif_transfer.alfworld_goal_acquisition_v10 import (
    AUTHENTIC,
    CARDINALITY_CONTROL,
    CEILING,
    EFFECT_CONTROL,
    GENERIC,
    RAW,
)
from motif_transfer.alfworld_policy_contribution import (
    audit_policy_contribution,
)


def _episode(condition, *, success, actions, records=()):
    return {
        "task_id": "task-1",
        "condition": condition,
        "official_success": success,
        "actions": list(actions),
        "records": list(records),
    }


def _report():
    raw = _episode(RAW, success=False, actions=("first", "look", "look"))
    authentic_records = [
        {
            "step": 1,
            "program_active": True,
            "source_admitted": True,
            "changed_action_vs_raw": True,
            "diagnostic": "SOURCE_INDUCED_ACQUISITION_OPERATOR_GROUNDED",
            "selected_action": "search target",
            "source_transition_advanced": False,
            "completed_count_before": 1,
            "completed_count_after": 1,
            "official_success_after": False,
        },
        {
            "step": 2,
            "program_active": True,
            "source_admitted": True,
            "changed_action_vs_raw": True,
            "diagnostic": "SOURCE_MACRO_TARGET_NATIVE_RELATION_REALIZATION",
            "selected_action": "relate target",
            "source_transition_advanced": True,
            "completed_count_before": 1,
            "completed_count_after": 2,
            "official_success_after": True,
        },
    ]
    authentic = _episode(
        AUTHENTIC, success=True,
        actions=("first", "search target", "relate target"),
        records=authentic_records,
    )
    report = {
        "episodes": {
            RAW: [raw], AUTHENTIC: [authentic],
            CARDINALITY_CONTROL: [deepcopy(raw) | {"condition": CARDINALITY_CONTROL}],
            EFFECT_CONTROL: [deepcopy(raw) | {"condition": EFFECT_CONTROL}],
            GENERIC: [deepcopy(raw) | {"condition": GENERIC}],
            CEILING: [deepcopy(authentic) | {"condition": CEILING}],
        },
        "authority_receipts": {
            "task-1": [
                {
                    "target_native_action": "search target",
                    "source_selector_action_emitted": False,
                    "target_executor_calls": 1,
                    "formal_outcome_read": False,
                },
                {
                    "target_native_action": "relate target",
                    "source_selector_action_emitted": False,
                    "target_executor_calls": 1,
                    "formal_outcome_read": False,
                },
            ],
        },
    }
    return report


def test_policy_contribution_validates_causal_rescue_and_authority():
    audit = audit_policy_contribution(_report())
    assert audit["rescues"] == 1
    assert audit["source_divergent_actions"] == 2
    assert all(audit["gates"].values())


def test_policy_contribution_rejects_rescue_without_prior_acquisition_divergence():
    report = _report()
    report["episodes"][AUTHENTIC][0]["records"][0][
        "changed_action_vs_raw"
    ] = False
    audit = audit_policy_contribution(report)
    assert not audit["gates"][
        "every_rescue_has_source_acquisition_divergence_before_terminal"
    ]


def test_policy_contribution_rejects_source_emitted_target_action():
    report = _report()
    report["authority_receipts"]["task-1"][0][
        "source_selector_action_emitted"
    ] = True
    audit = audit_policy_contribution(report)
    assert not audit["gates"]["source_never_emits_target_action"]
