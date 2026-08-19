from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from motif_transfer.alfworld_goal_relation_macro import (
    observe_goal_relation_transition,
)
from motif_transfer.alfworld_target_written_equivalent import (
    TARGET_WRITTEN_EQUIVALENT,
    TargetWrittenExecutionState,
    choose_target_written_action,
)
from motif_transfer.slot_aware_alfworld_harness import initialize_slot_ledger


class _ForbiddenSourceArtifact(Mapping[str, Any]):
    """Fail immediately if the target-written control touches source state."""

    def __getitem__(self, key: str) -> Any:
        raise AssertionError(f"source artifact was read: {key}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("source artifact was iterated")

    def __len__(self) -> int:
        raise AssertionError("source artifact length was read")


def _row(option: str, policy: float = 0.8) -> dict[str, Any]:
    return {
        "option": option,
        "policy": policy,
        "applicability": 0.95,
        "completion": 0.95,
        "binding": 0.99,
        "required_option": option,
    }


def _effect_head() -> dict[str, Any]:
    return {
        "feature_names": ["verb_move"],
        "means": [0.0],
        "scales": [1.0],
        "weights": [12.0],
        "intercept": -6.0,
    }


def _choose(
    grounded: Mapping[str, Mapping[str, Any]], ledger: Mapping[str, Any],
    *, state: TargetWrittenExecutionState | None = None,
) -> dict[str, Any]:
    return choose_target_written_action(
        condition=TARGET_WRITTEN_EQUIVALENT,
        grounded=grounded,
        goal="put two apple in drawer.",
        history=(),
        ledger=ledger,
        execution_state=state or TargetWrittenExecutionState(),
        source_artifact=_ForbiddenSourceArtifact(),
        target_causal_effect_head=_effect_head(),
        step=4,
        max_steps=30,
        minimum_binding=0.5,
        minimum_realization=0.1,
        minimum_binding_margin=0.0,
        minimum_causal_effect=0.5,
    )


def test_source_artifact_is_unread_and_target_policy_owns_first_relation() -> None:
    ledger = initialize_slot_ledger(
        "put two apple in drawer.", required_property="NONE",
    )
    decision = _choose({
        "go to drawer 1": _row("SEARCH", 0.7),
        "take apple 1 from table 1": _row("PICKUP", 0.99),
    }, ledger)
    assert decision["action"] == "take apple 1 from table 1"
    assert decision["program_active"] is False
    assert decision["diagnostic"] == "TARGET_WRITTEN_AWAITS_FIRST_RELATION"


def test_target_written_recurrence_preserves_exact_relation_handle() -> None:
    ledger = initialize_slot_ledger(
        "put two apple in drawer.", required_property="NONE",
    )
    ledger["carried_object"] = "apple 1"
    ledger, _ = observe_goal_relation_transition(
        ledger,
        action="move apple 1 to drawer 2",
        after_observation="You move the apple 1 to the drawer 2.",
    )
    ledger["carried_object"] = "apple 2"
    decision = _choose({
        "move apple 2 to drawer 1": _row("PLACE", 0.99),
        "move apple 2 to drawer 2": _row("PLACE", 0.80),
        "look": _row("SEARCH", 0.70),
    }, ledger)
    assert decision["action"] == "move apple 2 to drawer 2"
    assert decision["program_active"] is True
    assert decision["source_admitted"] is False


def test_target_written_acquisition_uses_target_neural_ranking_and_novelty() -> None:
    ledger = initialize_slot_ledger(
        "put two apple in drawer.", required_property="NONE",
    )
    ledger["bound_target_receptacle"] = "drawer 2"
    ledger["observed_locations"] = {"apple 1": "drawer 2"}
    ledger["completed_objects"] = ["apple 1"]
    grounded = {
        "go to counter 1": _row("SEARCH", 0.99),
        "go to table 1": _row("SEARCH", 0.80),
        "take mug 1 from shelf 1": _row("PICKUP", 1.0),
    }
    state = TargetWrittenExecutionState()
    first = _choose(grounded, ledger, state=state)
    second = _choose(grounded, ledger, state=state)
    assert first["action"] == "go to counter 1"
    assert second["action"] == "go to table 1"
    assert first["diagnostic"] == (
        "TARGET_WRITTEN_ACQUISITION_OPERATOR_GROUNDED"
    )
    assert first["source_admitted"] is False
