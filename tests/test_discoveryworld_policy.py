from __future__ import annotations

from dataclasses import replace

import pytest

from motif_transfer.discoveryworld_env import DiscoveryWorldObservation
from motif_transfer.discoveryworld_policy import (
    native_action_from_decision,
    parse_json_object,
    prompt_payload,
    target_native_facts,
    updated_memory,
)


def _observation(*, dialog=False):
    return DiscoveryWorldObservation(
        scenario="Proteomics", difficulty="Easy", seed=0, episode_step=0,
        ui={
            "taskProgress": [{"description": "x", "completed": False,
                              "completedSuccessfully": False}],
            "agentLocation": {"directions_you_can_move": ["north"]},
            "inventoryObjects": [{"uuid": 1, "name": "meter"}],
            "accessibleEnvironmentObjects": [{"uuid": 2, "name": "sample"}],
        },
        known_actions={"USE": {"args": ["arg1", "arg2"]}, "READ": {"args": ["arg1"]}},
        teleport_locations={"lab": {"gridX": 1, "gridY": 2}}, last_action_result=None, vision=None,
        in_dialog=dialog, terminal=False, official_success=False,
    )


def test_parse_and_project_native_action_without_advisory_keys():
    decision = parse_json_object(
        '```json\n{"action":"USE","arg1":1,"arg2":2,"memory":"m"}\n```'
    )
    assert native_action_from_decision(decision, _observation()) == {
        "action": "USE", "arg1": 1, "arg2": 2,
    }


def test_native_action_projection_normalizes_decimal_uuid_strings():
    assert native_action_from_decision(
        {"action": "USE", "arg1": "1", "arg2": "2"}, _observation(),
    ) == {"action": "USE", "arg1": 1, "arg2": 2}


def test_target_native_facts_preserve_relations_without_inventing_coordinates():
    observation = _observation()
    ui = dict(observation.ui)
    ui.update({
        "agentLocation": {"x": 15, "y": 13, "faceDirection": "east"},
        "inventoryObjects": [{"uuid": 7, "name": "flag"}],
        "accessibleEnvironmentObjects": [{"uuid": 9, "name": "statue"}],
        "nearbyObjects": {"objects": {
            "east": [
                {"uuid": 9, "name": "statue", "distance": 1},
                {"uuid": 10, "name": "floor", "distance": 1},
            ],
        }},
    })
    facts = target_native_facts(replace(observation, ui=ui))
    assert facts["salient_relative_objects"] == [{
        "relation_from_agent": "east", "distance": 1, "uuid": 9, "name": "statue",
    }]
    assert "object_coordinates" not in facts


def test_clear_native_precondition_errors_fail_before_world_step():
    observation = _observation()
    with pytest.raises(ValueError, match="integer UUID"):
        native_action_from_decision(
            {"action": "USE", "arg1": "meter", "arg2": 2}, observation,
        )
    with pytest.raises(ValueError, match="not accessible"):
        native_action_from_decision(
            {"action": "USE", "arg1": 1, "arg2": 99}, observation,
        )


def test_missing_argument_and_dialog_contract_fail_closed():
    with pytest.raises(ValueError, match="omitted"):
        native_action_from_decision({"action": "READ"}, _observation())
    assert native_action_from_decision(
        {"chosen_dialog_option_int": 2, "memory": "m"}, _observation(dialog=True),
    ) == {"chosen_dialog_option_int": 2}


def test_prompt_is_oracle_free_and_memory_is_bounded():
    observation = _observation()
    payload = prompt_payload(
        observation, memory="a" * 9000, hypotheses=["h"], recent_decisions=[],
    )
    text = str(payload).lower()
    assert "scorecard" not in text and "criticalhypoth" not in text
    assert len(payload["persistent_memory"]) == 6000
    memory, hypotheses = updated_memory(
        {"memory": "new", "running_hypotheses": ["h1", "h2"]}, "old", ["h0"],
    )
    assert memory == "new" and hypotheses == ("h1", "h2")
