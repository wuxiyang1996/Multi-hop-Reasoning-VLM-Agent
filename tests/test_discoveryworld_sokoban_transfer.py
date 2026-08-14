from __future__ import annotations

import json

from motif_transfer.discoveryworld_env import DiscoveryWorldObservation
from motif_transfer.discoveryworld_sokoban_transfer import (
    DiscoveryWorldGroundedCandidate,
    DiscoveryWorldTargetBinding,
    commit_available,
    evidence_supported,
    parse_grounded_candidates,
    parse_target_binding,
    positive_commit_effect_kind,
    positive_commit_effect_witnessed,
    realize_localized_spatial_position,
    select_candidate,
    target_bound_position,
)


def _observation():
    return DiscoveryWorldObservation(
        scenario="Proteomics", difficulty="Easy", seed=0, episode_step=20,
        ui={
            "taskProgress": [{"description": "drop flag west of statue", "completed": False,
                              "completedSuccessfully": False}],
            "agentLocation": {"x": 15, "y": 13, "directions_you_can_move": ["south"]},
            "inventoryObjects": [{"uuid": 7, "name": "flag"}],
            "accessibleEnvironmentObjects": [{"uuid": 9, "name": "statue"}],
            "nearbyObjects": {"objects": {"east": [
                {"uuid": 9, "name": "statue", "distance": 1},
            ]}},
        },
        known_actions={
            "DROP": {"args": ["arg1"]},
            "MOVE_DIRECTION": {"args": ["arg1"]},
        },
        teleport_locations={}, last_action_result=None, vision=None,
        in_dialog=False, terminal=False, official_success=False,
    )


def _candidate(role, action, positive, information, evidence=()):
    return DiscoveryWorldGroundedCandidate(
        action=action, target_role=role, prerequisite_probability=0.95,
        positive_effect_probability=positive,
        information_gain_probability=information, expected_effect="x",
        evidence=tuple(evidence), reason="test",
    )


def _binding(observation):
    return parse_target_binding(
        '{"target_uuid":9,"target_name":"statue",'
        '"commit_subject_relation_to_target":"west","target_distance":1,'
        '"commit_action":{"action":"DROP","arg1":7},'
        '"confidence":0.99,"hypothesis_used":"statue target","reason":"task relation"}',
        observation,
    )


def test_authentic_effect_program_commits_only_with_supported_positive_effect():
    observation = _observation()
    commit = _candidate(
        "COMMIT", {"action": "DROP", "arg1": 7}, 0.9, 0.0,
        ({"kind": "inventory", "uuid": 7, "name": "flag"},
         {"kind": "relative_object", "uuid": 9, "name": "statue",
          "relation_from_agent": "east", "distance": 1}),
    )
    position = _candidate("POSITION", {"action": "MOVE_DIRECTION", "arg1": "south"}, 0.2, 0.8)
    selected, receipt = select_candidate(
        "authentic_sokoban_effect_plus_target", (position, commit), observation,
        target_binding=_binding(observation),
    )
    assert selected is commit
    assert receipt.validate() and receipt.source_program_sha256 is not None


def test_refuted_spatial_evidence_forces_position_but_availability_is_distinct():
    observation = _observation()
    observation.ui["nearbyObjects"] = {"objects": {"north": [
        {"uuid": 9, "name": "statue", "distance": 1},
    ]}}
    wrong_commit = _candidate(
        "COMMIT", {"action": "DROP", "arg1": 7}, 0.9, 0.0,
        ({"kind": "inventory", "uuid": 7, "name": "flag"},
         {"kind": "relative_object", "uuid": 9, "name": "statue",
          "relation_from_agent": "north", "distance": 1}),
    )
    position = _candidate("POSITION", {"action": "MOVE_DIRECTION", "arg1": "south"}, 0.2, 0.8)
    assert not evidence_supported(wrong_commit, observation, _binding(observation))
    selected, _ = select_candidate(
        "authentic_sokoban_effect_plus_target", (position, wrong_commit), observation,
        target_binding=_binding(observation),
    )
    assert selected is position
    selected, _ = select_candidate(
        "commit_availability_control_plus_target", (position, wrong_commit), observation,
        target_binding=_binding(observation),
    )
    assert selected is wrong_commit


def test_source_controls_choose_different_branches_on_same_grounding():
    observation = _observation()
    low_effect_commit = _candidate(
        "COMMIT", {"action": "DROP", "arg1": 7}, 0.2, 0.0,
        ({"kind": "inventory", "uuid": 7, "name": "flag"},),
    )
    position = _candidate("POSITION", {"action": "MOVE_DIRECTION", "arg1": "south"}, 0.3, 0.9)
    authentic, _ = select_candidate(
        "authentic_sokoban_effect_plus_target", (position, low_effect_commit), observation,
        target_binding=_binding(observation),
    )
    available, _ = select_candidate(
        "commit_availability_control_plus_target", (position, low_effect_commit), observation,
        target_binding=_binding(observation),
    )
    inverted, _ = select_candidate(
        "inverted_effect_control_plus_target", (position, low_effect_commit), observation,
        target_binding=_binding(observation),
    )
    assert authentic is low_effect_commit
    assert available is low_effect_commit
    assert inverted is position


def test_target_binding_is_exact_and_symbolic_join_uses_observation():
    observation = _observation()
    binding = parse_target_binding(
        '{"target_uuid":9,"target_name":"statue",'
        '"commit_subject_relation_to_target":"west","target_distance":1,'
        '"commit_action":{"action":"DROP","arg1":7},'
        '"confidence":0.99,"hypothesis_used":"statue target","reason":"task relation"}',
        observation,
    )
    assert binding.target_uuid == 9
    incomplete_neural_evidence_commit = _candidate(
        "COMMIT", {"action": "DROP", "arg1": 7}, 0.95, 0.0,
        ({"kind": "inventory", "uuid": 7, "name": "flag"},),
    )
    assert not evidence_supported(incomplete_neural_evidence_commit, observation, binding)
    selected, receipt = select_candidate(
        "authentic_sokoban_effect_plus_target",
        (_candidate("POSITION", {"action": "MOVE_DIRECTION", "arg1": "south"}, 0.2, 0.8),
         incomplete_neural_evidence_commit),
        observation,
        target_binding=binding,
    )
    assert selected is incomplete_neural_evidence_commit
    assert receipt.positive_commit_effect_witnessed and receipt.validate()


def test_symbolic_join_overrides_neural_coordinate_narration_but_not_observation():
    observation = _observation()
    binding = _binding(observation)
    # The candidate grounder is deliberately pessimistic and omits the spatial
    # evidence. The bound native action plus exact observed relation still form
    # a target-symbolic direct-effect witness.
    pessimistic_commit = DiscoveryWorldGroundedCandidate(
        action={"action": "DROP", "arg1": 7}, target_role="COMMIT",
        prerequisite_probability=0.0, positive_effect_probability=0.0,
        information_gain_probability=0.0, expected_effect="stale coordinates say no",
        evidence=({"kind": "inventory", "uuid": 7, "name": "flag"},),
        reason="neural narration is stale",
    )
    position = _candidate(
        "POSITION", {"action": "MOVE_DIRECTION", "arg1": "south"}, 0.99, 0.9,
    )
    assert commit_available(pessimistic_commit, binding)
    assert positive_commit_effect_witnessed(pessimistic_commit, observation, binding)
    selected, receipt = select_candidate(
        "authentic_sokoban_effect_plus_target", (position, pessimistic_commit), observation,
        target_binding=binding,
    )
    assert selected is pessimistic_commit
    assert receipt.positive_commit_effect_witnessed and receipt.validate()

    wrong_relation_observation = _observation()
    wrong_relation_observation.ui["nearbyObjects"] = {"objects": {"north-east": [
        {"uuid": 9, "name": "statue", "distance": 2},
    ]}}
    assert not positive_commit_effect_witnessed(
        pessimistic_commit, wrong_relation_observation, binding,
    )
    selected, receipt = select_candidate(
        "authentic_sokoban_effect_plus_target",
        (position, pessimistic_commit), wrong_relation_observation,
        target_binding=binding,
    )
    assert selected is position
    assert not receipt.positive_commit_effect_witnessed and receipt.validate()


def test_bound_object_teleport_survives_bad_neural_relation_evidence():
    observation = _observation()
    observation.ui["nearbyObjects"] = {"objects": {"north-east": [
        {"uuid": 9, "name": "statue", "distance": 2},
    ]}}
    binding = _binding(observation)
    teleport = _candidate(
        "POSITION", {"action": "TELEPORT_TO_OBJECT", "arg1": 9}, 0.8, 0.0,
        ({"kind": "relative_object", "uuid": 9, "name": "statue",
          "relation_from_agent": "east", "distance": 1},),
    )
    feed = _candidate("POSITION", {"action": "DISCOVERY_FEED_GET_UPDATES"}, 0.0, 0.5)
    assert not evidence_supported(teleport, observation, binding)
    assert target_bound_position(teleport, binding)
    selected, receipt = select_candidate(
        "authentic_sokoban_effect_plus_target", (feed, teleport), observation,
        target_binding=binding,
    )
    assert selected is teleport
    assert receipt.target_bound_position and receipt.validate()


def test_localized_spatial_realizer_routes_around_target_without_coordinates():
    observation = _observation()
    observation.ui["agentLocation"] = {
        "x": 16, "y": 16, "directions_you_can_move": ["west"],
    }
    observation.ui["nearbyObjects"] = {"objects": {"north": [
        {"uuid": 9, "name": "statue", "distance": 1},
    ]}}
    neural_position = _candidate(
        "POSITION", {"action": "TELEPORT_TO_OBJECT", "arg1": 9}, 0.9, 0.0,
    )
    action, receipt = realize_localized_spatial_position(
        neural_position, observation, _binding(observation),
        target_was_localized=True,
    )
    assert action == {"action": "MOVE_DIRECTION", "arg1": "west"}
    assert receipt["active"] and receipt["changed"]
    assert receipt["compatible_target_vectors"] == [[0, -1]]
    assert receipt["desired_target_vector"] == [1, 0]
    expected = dict(receipt)
    receipt_hash = expected.pop("receipt_sha256")
    from motif_transfer.discoveryworld_env import stable_hash
    assert receipt_hash == stable_hash(expected)


def test_localized_spatial_realizer_finishes_two_step_relation_route():
    observation = _observation()
    observation.ui["agentLocation"] = {
        "x": 15, "y": 16, "directions_you_can_move": ["north", "east"],
    }
    observation.ui["nearbyObjects"] = {"objects": {"north-east": [
        {"uuid": 9, "name": "statue", "distance": 2},
    ]}}
    neural_position = _candidate(
        "POSITION", {"action": "TELEPORT_TO_OBJECT", "arg1": 9}, 0.9, 0.0,
    )
    action, receipt = realize_localized_spatial_position(
        neural_position, observation, _binding(observation),
        target_was_localized=True,
    )
    assert action == {"action": "MOVE_DIRECTION", "arg1": "north"}
    assert receipt["current_worst_case_error"] == 1
    assert receipt["active"]


def test_spatial_realizer_requires_native_localization_and_never_rewrites_commit():
    observation = _observation()
    binding = _binding(observation)
    position = _candidate(
        "POSITION", {"action": "MOVE_DIRECTION", "arg1": "south"}, 0.2, 0.8,
    )
    action, receipt = realize_localized_spatial_position(
        position, observation, binding, target_was_localized=False,
    )
    assert action == dict(position.action)
    assert not receipt["active"]
    commit = _candidate("COMMIT", {"action": "DROP", "arg1": 7}, 0.9, 0.0)
    action, receipt = realize_localized_spatial_position(
        commit, observation, binding, target_was_localized=True,
    )
    assert action == dict(commit.action)
    assert not receipt["active"]


def test_candidate_parser_discards_one_invalid_action_without_poisoning_bundle():
    raw = json.dumps({
        "memory": "",
        "running_hypotheses": [],
        "candidates": [
            {
                "action": "MOVE_DIRECTION", "arg1": "north", "target_role": "POSITION",
                "prerequisite_probability": 1, "positive_effect_probability": 0.1,
                "information_gain_probability": 0.1, "expected_effect": "invalid move",
                "evidence": [], "reason": "north is not currently movable",
            },
            {
                "action": "MOVE_DIRECTION", "arg1": "south", "target_role": "POSITION",
                "prerequisite_probability": 1, "positive_effect_probability": 0.2,
                "information_gain_probability": 0.2, "expected_effect": "valid move",
                "evidence": [], "reason": "south is movable",
            },
            {
                "action": "DROP", "arg1": 7, "target_role": "COMMIT",
                "prerequisite_probability": 0.5, "positive_effect_probability": 0.5,
                "information_gain_probability": 0, "expected_effect": "drop",
                "evidence": [{"kind": "inventory", "uuid": 7, "name": "flag"}],
                "reason": "native commit",
            },
        ],
    })
    bundle, candidates = parse_grounded_candidates(raw, _observation())
    assert len(candidates) == 2
    assert [row.action["action"] for row in candidates] == ["MOVE_DIRECTION", "DROP"]
    assert len(bundle["candidate_parse_rejections"]) == 1
    assert "not currently movable" in bundle["candidate_parse_rejections"][0]["error"]
    assert not bundle["choice_set_degenerate"]

    one_valid = json.loads(raw)
    one_valid["candidates"] = one_valid["candidates"][:2]
    bundle, candidates = parse_grounded_candidates(json.dumps(one_valid), _observation())
    assert len(candidates) == 1
    assert bundle["choice_set_degenerate"]


def test_assignment_improvement_witness_joins_put_subject_and_receiver():
    observation = DiscoveryWorldObservation(
        scenario="Space Sick", difficulty="Easy", seed=0, episode_step=9,
        ui={
            "taskProgress": [{"description": "put contaminated mushroom in jar",
                              "completed": False, "completedSuccessfully": False}],
            "agentLocation": {"directions_you_can_move": ["south"]},
            "inventoryObjects": [
                {"uuid": 57, "name": "mushroom", "description": "pink mushroom"},
                {"uuid": 46, "name": "mushroom", "description": "green mushroom"},
            ],
            "accessibleEnvironmentObjects": [
                {"uuid": 14, "name": "jar", "description": "empty jar"},
            ],
            "nearbyObjects": {"objects": {}},
        },
        known_actions={"PUT": {"args": ["arg1", "arg2"]}}, teleport_locations={},
        last_action_result=None, vision=None, in_dialog=False, terminal=False,
        official_success=False,
    )
    binding = parse_target_binding(json.dumps({
        "target_uuid": 14,
        "target_name": "jar",
        "commit_subject_relation_to_target": "inside",
        "target_distance": None,
        "commit_action": {"action": "PUT", "arg1": 57, "arg2": 14},
        "confidence": 0.99,
        "hypothesis_used": "pink mushroom is contaminated",
        "reason": "put the bound contaminated sample in the jar",
    }), observation)
    correct = DiscoveryWorldGroundedCandidate(
        action={"action": "PUT", "arg1": 57, "arg2": 14}, target_role="COMMIT",
        prerequisite_probability=0.0, positive_effect_probability=0.0,
        information_gain_probability=0.0, expected_effect="assignment", evidence=(),
        reason="neural probabilities are not the symbolic witness",
    )
    wrong = DiscoveryWorldGroundedCandidate(
        action={"action": "PUT", "arg1": 46, "arg2": 14}, target_role="COMMIT",
        prerequisite_probability=1.0, positive_effect_probability=1.0,
        information_gain_probability=0.0, expected_effect="wrong assignment", evidence=(),
        reason="wrong mushroom",
    )
    assert positive_commit_effect_kind(correct, observation, binding) == (
        "ASSIGNMENT_IMPROVEMENT_AVAILABLE"
    )
    assert positive_commit_effect_kind(wrong, observation, binding) is None
