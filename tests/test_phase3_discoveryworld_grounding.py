from dataclasses import dataclass

from motif_transfer.phase3_discoveryworld_grounding import (
    call_qualified_decision,
    exact_action_catalog,
    grounding_prompt_payload,
)


@dataclass
class Observation:
    ui: dict
    known_actions: dict
    teleport_locations: tuple[str, ...]
    in_dialog: bool = False

    def policy_payload(self):
        return {"ui": self.ui, "known_actions": self.known_actions}


def _observation():
    return Observation(
        ui={
            "agentLocation": {"directions_you_can_move": ["north", "east"]},
            "inventoryObjects": [{"uuid": 7, "name": "flag"}],
            "accessibleEnvironmentObjects": [{"uuid": 9, "name": "meter"}],
            "nearbyObjects": {"objects": {
                "west": [{"uuid": 11, "name": "statue", "distance": 2}],
            }},
            "taskProgress": [],
        },
        known_actions={
            "MOVE_DIRECTION": {"args": ["arg1"]},
            "PICKUP": {"args": ["arg1"]},
            "DROP": {"args": ["arg1"]},
            "USE": {"args": ["arg1", "arg2"]},
            "TELEPORT_TO_OBJECT": {"args": ["arg1"]},
            "DISCOVERY_FEED_GET_UPDATES": {"args": []},
        },
        teleport_locations=("Instrument Table",),
    )


def test_exact_action_catalog_uses_only_observed_argument_domains():
    catalog = exact_action_catalog(_observation())
    assert catalog["actions"]["MOVE_DIRECTION"]["arg1_allowed"] == ["east", "north"]
    assert catalog["actions"]["PICKUP"]["arg1_allowed"] == [9]
    assert catalog["actions"]["DROP"]["arg1_allowed"] == [7]
    assert catalog["actions"]["TELEPORT_TO_OBJECT"]["arg1_allowed"] == [7, 9, 11]
    assert catalog["actions"]["USE"]["arg1_allowed"] == [7, 9]
    assert {row["uuid"]: row["name"] for row in catalog["object_facts"]} == {
        7: "flag", 9: "meter", 11: "statue",
    }


def test_prompt_declares_outcome_blind_grounding_contract():
    payload = grounding_prompt_payload(
        _observation(), memory="", hypotheses=(), recent_decisions=(),
    )
    contract = payload["grounding_qualification_contract"]
    assert contract["selection_reads_official_success"] is False
    assert contract["selection_reads_evaluator_scorecard"] is False
    assert contract["source_game_context_present"] is False


class Backend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.last_usage = {}

    def complete(self, *_args):
        return next(self.outputs)


def test_schema_retry_rejects_invalid_catalog_action_then_accepts():
    backend = Backend([
        '{"action":"MOVE_DIRECTION","arg1":"south"}',
        '{"action":"MOVE_DIRECTION","arg1":"north","memory":"m",'
        '"running_hypotheses":[],"expected_effect":"move","reason":"valid"}',
    ])
    decision, action, _raw, audit, fallback = call_qualified_decision(
        backend=backend,
        observation=_observation(),
        memory="",
        hypotheses=(),
        recent=[],
        attempts=2,
    )
    assert decision["action"] == "MOVE_DIRECTION"
    assert action == {"action": "MOVE_DIRECTION", "arg1": "north"}
    assert [row["accepted"] for row in audit] == [False, True]
    assert fallback is False


def test_exhausted_schema_attempts_return_counted_native_fallback():
    decision, action, _raw, audit, fallback = call_qualified_decision(
        backend=Backend(["None", "-1.5e+380000"]),
        observation=_observation(),
        memory="",
        hypotheses=(),
        recent=[],
        attempts=2,
    )
    assert fallback is True
    assert action == {"action": "DISCOVERY_FEED_GET_UPDATES"}
    assert decision["reason"] == "PHASE3_GROUNDING_SCHEMA_FALLBACK"
    assert all(not row["accepted"] for row in audit)


def test_neural_affordance_rejection_triggers_focused_repair():
    backend = Backend([
        '{"action":"PICKUP","arg1":9}',
        '{"verdict":"INVALID","reason":"meter is a fixed instrument"}',
        '{"action":"MOVE_DIRECTION","arg1":"north","memory":"m",'
        '"running_hypotheses":[],"expected_effect":"move","reason":"repair"}',
    ])
    decision, action, _raw, audit, fallback = call_qualified_decision(
        backend=backend,
        observation=_observation(),
        memory="",
        hypotheses=(),
        recent=[],
        attempts=2,
    )
    assert fallback is False
    assert decision["action"] == "MOVE_DIRECTION"
    assert action == {"action": "MOVE_DIRECTION", "arg1": "north"}
    assert [row["accepted"] for row in audit] == [False, True]
    assert audit[0]["affordance_check"]["accepted"] is False


def test_neural_affordance_no_error_accepts_use_action():
    backend = Backend([
        '{"action":"USE","arg1":9,"arg2":7,"memory":"m",'
        '"running_hypotheses":[],"expected_effect":"measure","reason":"test"}',
        '{"verdict":"VALID","reason":"meter is a tool and flag is a target"}',
    ])
    _decision, action, _raw, audit, fallback = call_qualified_decision(
        backend=backend,
        observation=_observation(),
        memory="",
        hypotheses=(),
        recent=[],
        attempts=2,
    )
    assert fallback is False
    assert action == {"action": "USE", "arg1": 9, "arg2": 7}
    assert audit[0]["affordance_check"]["verdict"] == "VALID"
