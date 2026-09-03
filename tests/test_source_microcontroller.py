from __future__ import annotations

from dataclasses import asdict

from motif_transfer.multihorizon_replay import stable_hash
from motif_transfer.multihorizon_runner import ForkState, ForkStep
from motif_transfer.source_microcontroller import (
    ControlBranch,
    MicroDecisionPoint,
    ReceiptEvent,
    analyze_microcontroller_rows,
    choose_microcontroller_action,
    classify_receipt_event,
    induce_discovery_branch_map,
    run_microcontroller_snapshot,
)


def test_receipt_event_classification_uses_only_past_effects():
    assert classify_receipt_event(
        [0.0, 0.0], [False, False], [False, False], stall_window=2,
    ) == ReceiptEvent.STALL
    assert classify_receipt_event(
        [0.0, 2.0], [False, False], [False, False], stall_window=2,
    ) == ReceiptEvent.PROGRESS
    assert classify_receipt_event(
        [0.0, -1.0], [False, False], [False, False], stall_window=2,
    ) == ReceiptEvent.FAILURE
    assert classify_receipt_event(
        [0.0], [True], [False], stall_window=2,
    ) == ReceiptEvent.AFFORDANCE_CHANGE


def _point(*, game, event, branch, positive, ordinal):
    body = {
        "game": game,
        "episode_id": f"{game}-{ordinal}",
        "episode_seed": ordinal,
        "split": "discovery",
        "step": 1,
        "event": event.value,
        "previous_action": "LEFT",
        "native_actions": ("LEFT", "RIGHT"),
        "prefix_actions": ("LEFT",),
        "prefix_rewards": (0.0,),
        "expected_fork_observable_hash": stable_hash("fork"),
        "source_history_receipt_ids": (stable_hash((game, ordinal)),),
        "observed_branch": branch.value,
        "future_returns": {f"h{h}": float(positive) for h in (1, 2, 4, 8)},
        "future_positive": {f"h{h}": positive for h in (1, 2, 4, 8)},
    }
    return MicroDecisionPoint(
        stable_hash(body),
        **{**body, "event": event, "observed_branch": branch},
    )


def test_branch_map_is_induced_from_game_balanced_discovery_outcomes():
    points = []
    ordinal = 1
    for game in ("g1", "g2"):
        for event, preferred in (
            (ReceiptEvent.PROGRESS, ControlBranch.PERSIST),
            (ReceiptEvent.STALL, ControlBranch.SWITCH),
        ):
            for branch in ControlBranch:
                points.append(_point(
                    game=game,
                    event=event,
                    branch=branch,
                    positive=branch == preferred,
                    ordinal=ordinal,
                ))
                ordinal += 1
    mapping, audit = induce_discovery_branch_map(
        points, min_branch_support=2, min_games_per_branch=2,
    )
    assert mapping == {"PROGRESS": "PERSIST", "STALL": "SWITCH"}
    assert audit["STALL"]["selection_authority"].startswith("DISCOVERY")


def test_first_switch_action_is_frozen_in_snapshot():
    state = ForkState({"position": 0}, ("LEFT", "RIGHT", "UP"))
    snapshot = {
        "snapshot_id": "snapshot",
        "prefix_actions": ["LEFT"],
        "prefix_rewards": [0.0, 0.0],
        "switch_action": "UP",
    }
    action, metadata = choose_microcontroller_action(
        state,
        snapshot=snapshot,
        branch_map={"STALL": "SWITCH"},
        treatment="EVENT_CONTROLLER",
        decision_index=0,
        history=(),
        stall_window=2,
    )
    assert action == "UP"
    assert metadata["event"] == "STALL"
    assert metadata["branch"] == "SWITCH"


class ToyForkEnvironment:
    def __init__(self):
        self.step_index = 0

    def _state(self):
        return ForkState(
            {"step": self.step_index},
            ("LEFT", "RIGHT"),
            observable=f"obs-{self.step_index}",
        )

    def reset(self, *, seed: int):
        self.step_index = 0
        return self._state()

    def step(self, action: str):
        self.step_index += 1
        return ForkStep(self._state(), 1.0 if action == "RIGHT" else 0.0)

    def close(self):
        pass


def test_matched_runner_is_complete_and_static_control_blocks_overclaim():
    reference = ToyForkEnvironment()
    fork = reference.reset(seed=1)
    fork = reference.step("LEFT").state
    snapshot = {
        "snapshot_id": "snapshot",
        "game": "toy",
        "episode_id": "episode",
        "episode_seed": 1,
        "fork_step": 1,
        "split": "qualification",
        "event": "STALL",
        "prefix_actions": ["LEFT"],
        "prefix_rewards": [0.0],
        "expected_fork_observable_hash": fork.observable_hash,
        "previous_action": "LEFT",
        "switch_action": "RIGHT",
        "source_history_receipt_ids": [stable_hash("receipt")],
    }
    rows = run_microcontroller_snapshot(
        ToyForkEnvironment,
        snapshot=snapshot,
        branch_map={"STALL": "SWITCH"},
        stall_window=1,
    )
    assert len(rows) == 10
    assert {row["status"] for row in rows} == {"INTERVENTION_OBSERVED"}
    common = {
        row["treatment"]: row for row in rows
        if row["mode"] == "COMMON_HASH_CONTINUATION"
    }
    assert common["EVENT_CONTROLLER"]["actions"][0] == "RIGHT"
    assert common["SHUFFLED_EVENT_CONTROLLER"]["actions"][0] == "LEFT"

    # Duplicate the complete cell into held-out so every gate is evaluated.
    paired_rows = list(rows) + [dict(row) | {
        "snapshot_id": "heldout-snapshot",
        "episode_id": "heldout-episode",
        "split": "held_out",
    } for row in rows]
    report = analyze_microcontroller_rows(paired_rows)
    assert report["invalid_cells"] == []
    assert report["gates"]["SOURCE_MICROCONTROLLER_SUPPORTED"] is False
    # The event controller is identical to ALWAYS_SWITCH for this one-event
    # program, so it must not claim value beyond the best static policy.
    assert report["gates"][
        "QUALIFICATION_COMMON_HASH_CONTINUATION_H8_GT_BEST_STATIC"
    ] is False
