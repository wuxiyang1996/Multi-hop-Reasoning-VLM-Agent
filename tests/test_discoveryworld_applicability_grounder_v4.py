from motif_transfer.discoveryworld_applicability_grounder_v4 import (
    select_source_safe_candidate,
    source_applicability_complete,
)
from motif_transfer.discoveryworld_env import DiscoveryWorldObservation
from motif_transfer.discoveryworld_sokoban_transfer import (
    DiscoveryWorldGroundedCandidate,
    DiscoveryWorldTargetBinding,
)


def _observation(relation="east"):
    return DiscoveryWorldObservation(
        scenario="Proteomics",
        difficulty="Easy",
        seed=1,
        episode_step=10,
        ui={
            "agentLocation": {"x": 1, "y": 1},
            "inventoryObjects": [{"uuid": 7, "name": "flag"}],
            "accessibleEnvironmentObjects": [
                {"uuid": 9, "name": "statue of a test animal"},
            ],
            "nearbyObjects": {"objects": {
                relation: [{
                    "uuid": 9, "name": "statue of a test animal", "distance": 1,
                }],
            }},
        },
        known_actions={},
        teleport_locations={},
        last_action_result=None,
        vision=None,
        in_dialog=False,
        terminal=False,
        official_success=False,
    )


def _candidate(role, action, evidence=()):
    return DiscoveryWorldGroundedCandidate(
        action=action,
        target_role=role,
        prerequisite_probability=1.0,
        positive_effect_probability=1.0,
        information_gain_probability=0.0,
        expected_effect="test",
        evidence=tuple(evidence),
        reason="test",
    )


def _binding():
    return DiscoveryWorldTargetBinding(
        target_uuid=9,
        target_name="statue of a test animal",
        commit_subject_relation_to_target="west",
        target_relation_from_agent="east",
        target_distance=1,
        commit_action={"action": "DROP", "arg1": 7},
        confidence=1.0,
        hypothesis_used="test animal",
        reason="test",
    )


def test_rejects_unwitnessed_commit_only_set():
    commit = _candidate(
        "COMMIT", {"action": "DROP", "arg1": 7},
        [{"kind": "inventory", "uuid": 7, "name": "flag"}],
    )
    assert not source_applicability_complete(
        (commit,), _observation(relation="north"), _binding(),
    )


def test_accepts_parser_validated_position_despite_stale_explanatory_evidence():
    position = _candidate(
        "POSITION", {"action": "MOVE_DIRECTION", "arg1": "north"},
        [{
            "kind": "relative_object", "uuid": 9,
            "name": "statue of a test animal",
            "relation_from_agent": "south", "distance": 99,
        }],
    )
    assert source_applicability_complete((position,), _observation(), _binding())


def test_accepts_symbolically_witnessed_commit():
    commit = _candidate(
        "COMMIT", {"action": "DROP", "arg1": 7},
        [
            {"kind": "inventory", "uuid": 7, "name": "flag"},
            {
                "kind": "relative_object", "uuid": 9,
                "name": "statue of a test animal",
                "relation_from_agent": "east", "distance": 1,
            },
        ],
    )
    assert source_applicability_complete((commit,), _observation(), _binding())


def test_selector_replaces_unwitnessed_commit_with_neural_position():
    observation = _observation(relation="north")
    position = _candidate(
        "POSITION", {"action": "MOVE_DIRECTION", "arg1": "north"},
        [{
            "kind": "relative_object", "uuid": 9,
            "name": "statue of a test animal",
            "relation_from_agent": "south", "distance": 99,
        }],
    )
    commit = _candidate(
        "COMMIT", {"action": "DROP", "arg1": 7},
        [{"kind": "inventory", "uuid": 7, "name": "flag"}],
    )
    selected, receipt = select_source_safe_candidate(
        "authentic_sokoban_effect_plus_target",
        (position, commit), observation, target_binding=_binding(),
    )
    assert selected.target_role == "POSITION"
    assert receipt.selection_reason.startswith("SOURCE_UNWITNESSED_COMMIT_REJECTED")
