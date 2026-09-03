import json

from motif_transfer.discoveryworld_env import DiscoveryWorldObservation
from motif_transfer.discoveryworld_structured_acquisition_v2 import (
    acquisition_prompt_payload,
    call_structured_acquisition_grounder,
    phase3_acquisition_action_catalog,
)


def _observation(inventory=()):
    objects = [
        {"uuid": 1, "name": "proteomics meter"},
        {"uuid": 2, "name": "flag"},
        {"uuid": 3, "name": "echojelly"},
        {"uuid": 4, "name": "prismatic beast"},
        {"uuid": 5, "name": "spheroid"},
        {"uuid": 13, "name": "statue of an echojelly"},
        {"uuid": 14, "name": "statue of a prismatic beast"},
        {"uuid": 15, "name": "statue of a spheroid"},
    ]
    inventory_rows = [row for row in objects if row["uuid"] in inventory]
    accessible = [row for row in objects if row["uuid"] not in inventory]
    return DiscoveryWorldObservation(
        scenario="Proteomics", difficulty="Easy", seed=0, episode_step=3,
        ui={
            "taskProgress": [{
                "description": "measure three species then drop flag west of statue",
                "completed": True, "completedSuccessfully": True, "score": 1,
            }],
            "agentLocation": {
                "x": 0, "y": 0, "directions_you_can_move": ["north"],
            },
            "inventoryObjects": inventory_rows,
            "accessibleEnvironmentObjects": accessible,
            "nearbyObjects": {"objects": {}},
        },
        known_actions={
            "PICKUP": {"args": ["arg1"]},
            "USE": {"args": ["arg1", "arg2"]},
            "TELEPORT_TO_OBJECT": {"args": ["arg1"]},
        },
        teleport_locations={}, last_action_result=None, vision=None,
        in_dialog=False, terminal=False, official_success=False,
    )


def test_structured_catalog_advances_instrument_measurements_then_flag():
    first = phase3_acquisition_action_catalog(
        _observation(), measured_subjects=(),
    )
    assert first[0]["stage"] == "ACQUIRE_INSTRUMENT"
    assert first[0]["action"] == {"action": "PICKUP", "arg1": 1}

    measure = phase3_acquisition_action_catalog(
        _observation((1,)), measured_subjects=(),
    )
    assert measure[0]["stage"] == "MEASURE_MISSING_SUBJECT"
    assert measure[0]["action"] == {"action": "USE", "arg1": 1, "arg2": 3}

    flag = phase3_acquisition_action_catalog(
        _observation((1,)),
        measured_subjects=("echojelly", "prismatic beast", "spheroid"),
    )
    assert flag[0]["action"] == {"action": "PICKUP", "arg1": 2}
    assert phase3_acquisition_action_catalog(
        _observation((1, 2)),
        measured_subjects=("echojelly", "prismatic beast", "spheroid"),
    ) == ()


def test_acquisition_payload_is_outcome_blind_and_repairs_to_exact_catalog():
    observation = _observation()
    payload = acquisition_prompt_payload(observation, measured_subjects=())
    serialized = json.dumps(payload)
    assert "completed" not in serialized
    assert "score" not in serialized
    assert payload["formal_outcome_fields_visible"] is False

    class Backend:
        last_usage = {}

        def __init__(self):
            self.responses = [
                json.dumps({"action": "PICKUP", "arg1": 999}),
                json.dumps({"action": "PICKUP", "arg1": 1}),
            ]

        def complete(self, role, system, payload):
            return self.responses.pop(0)

    action, _, audit, fallback = call_structured_acquisition_grounder(
        backend=Backend(), observation=observation, measured_subjects=(),
        attempts=2,
    )
    assert action == {"action": "PICKUP", "arg1": 1}
    assert [row["accepted"] for row in audit] == [False, True]
    assert fallback is False
    assert all(row["formal_outcome_fields_visible"] is False for row in audit)
