import json
from pathlib import Path

from motif_transfer.clevrer_unified_goal_relation import (
    source_goal_relation_contract,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.source_controlled_grounding import (
    GroundingControlVerdict,
    SourceControlledGroundingPolicy,
    TargetGroundingToolBinding,
    TypedGroundingControlState,
    bind_authorized_grounding_tool,
)


REPO = Path(__file__).resolve().parents[1]
ARTIFACT = json.loads((
    REPO / "runs/sokoban_goal_relation_macro_v3/artifact.json"
).read_text())
CONFIRMATION = json.loads((
    REPO / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
).read_text())


def _state(domain="agqa2", **overrides):
    values = {
        "task_id": f"{domain}-task",
        "target_domain": domain,
        "target_state_sha256": stable_hash([domain, "state"]),
        "transition_guard_observable": True,
        "transition_guard_satisfied": True,
        "transition_effect_authenticated": True,
        "terminal_guard_observable": True,
        "terminal_guard_satisfied": False,
        "abstention_guard_satisfied": False,
        "interventions_used": 0,
        "intervention_budget": 2,
        "formal_outcome_read": False,
    }
    values.update(overrides)
    return TypedGroundingControlState(**values)


def _policy():
    return SourceControlledGroundingPolicy(
        source_goal_relation_contract(ARTIFACT, CONFIRMATION)
    )


def test_same_source_transition_binds_to_agqa_and_clevrer_native_tools():
    policy = _policy()
    receipts = []
    for domain, tool in (
        ("agqa2", "inspect_temporal_relation_window"),
        ("clevrer", "compare_event_graph_representations"),
    ):
        authorization = policy.decide(_state(domain))
        assert authorization.verdict == GroundingControlVerdict.APPLY_TRANSITION
        receipt = bind_authorized_grounding_tool(
            authorization,
            TargetGroundingToolBinding(
                target_domain=domain,
                target_adapter_sha256=stable_hash([domain, "adapter"]),
                transition_tool=tool,
                transition_arguments={"budget_slot": 1},
            ),
        )
        assert receipt["tool"] == tool
        assert receipt["source_program_sha256"] == ARTIFACT["artifact_sha256"]
        assert receipt["gold_or_target_outcome"] == "NOT_READ"
        receipts.append(receipt)
    assert receipts[0]["source_program_sha256"] == receipts[1]["source_program_sha256"]


def test_terminal_abstention_effect_and_budget_guards_fail_closed():
    policy = _policy()
    terminal = policy.decide(_state(terminal_guard_satisfied=True))
    assert terminal.verdict == GroundingControlVerdict.COMMIT
    assert bind_authorized_grounding_tool(
        terminal,
        TargetGroundingToolBinding(
            "agqa2", stable_hash("adapter"), "unused", {},
        ),
    ) is None
    cases = (
        _state(abstention_guard_satisfied=True),
        _state(transition_effect_authenticated=False),
        _state(interventions_used=2),
        _state(formal_outcome_read=True),
    )
    assert all(
        policy.decide(state).verdict == GroundingControlVerdict.ABSTAIN
        for state in cases
    )


def test_target_adapter_never_changes_source_control_verdict():
    authorization = _policy().decide(_state())
    for tool in ("sample_frames", "detect_transitions", "inspect_event"):
        receipt = bind_authorized_grounding_tool(
            authorization,
            TargetGroundingToolBinding(
                "agqa2", stable_hash(tool), tool, {"n": 4},
            ),
        )
        assert receipt["source_authorization_sha256"] == (
            authorization.authorization_sha256
        )
