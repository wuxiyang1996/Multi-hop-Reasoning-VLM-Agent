from __future__ import annotations

from copy import deepcopy

from motif_transfer.discoveryworld_env import DiscoveryWorldObservation
from motif_transfer.discoveryworld_qualification import (
    assess_effect_guard_applicability,
    select_first_commit_fork,
)
from motif_transfer.discoveryworld_sokoban_transfer import (
    DiscoveryWorldGroundedCandidate,
    DiscoveryWorldTargetBinding,
)


def _episode():
    return {
        "task_id": "example.easy.seed1",
        "task": {"scenario": "Example", "difficulty": "Easy", "seed": 1},
        "episode_sha256": "episode-hash",
        "steps": [
            {"episode_step": 1, "action": {"action": "MOVE_DIRECTION", "arg1": "west"},
             "action_succeeded": True},
            {"episode_step": 2, "action": {"action": "DROP", "arg1": 7},
             "action_succeeded": False},
            {"episode_step": 3, "action": {"action": "PUT", "arg1": 8, "arg2": 9},
             "action_succeeded": True},
        ],
        "evaluation": {"official_success": False, "scorecard": ["secret outcome"]},
    }


def test_first_commit_fork_is_selected_without_action_or_task_outcome():
    episode = _episode()
    receipt = select_first_commit_fork(episode, ["PUT", "DROP"])
    assert receipt["eligible"]
    assert receipt["fork_after_episode_step"] == 1
    assert receipt["selected_action"] == {"action": "DROP", "arg1": 7}
    assert not receipt["outcome_fields_read_for_eligibility"]

    changed = deepcopy(episode)
    changed["steps"][1]["action_succeeded"] = True
    changed["evaluation"] = {"official_success": True, "scorecard": ["opposite"]}
    assert select_first_commit_fork(changed, ["DROP", "PUT"]) == receipt


def test_no_predeclared_commit_is_explicitly_ineligible():
    episode = _episode()
    episode["steps"] = episode["steps"][:1]
    receipt = select_first_commit_fork(episode, ["DROP", "PUT"])
    assert not receipt["eligible"]
    assert receipt["reason"] == "NO_PREDECLARED_COMMIT_ACTION"


def _applicability_observation(*, target_distance: int = 2):
    return DiscoveryWorldObservation(
        scenario="Proteomics",
        difficulty="Easy",
        seed=2,
        episode_step=27,
        known_actions={
            "DROP": {"args": ["arg1"]},
            "TELEPORT_TO_OBJECT": {"args": ["arg1"]},
        },
        ui={
            "agentLocation": {},
            "inventoryObjects": [{"uuid": 7, "name": "flag"}],
            "accessibleEnvironmentObjects": [],
            "nearbyObjects": {"objects": {"east": [
                {"uuid": 9, "name": "statue", "distance": target_distance}
            ]}},
            "taskProgress": [
                {"description": "Drop the flag west of the target statue."}
            ],
        },
        teleport_locations={},
        in_dialog=False,
        last_action_result=None,
        vision=None,
        terminal=False,
        official_success=False,
    )


def _binding():
    return DiscoveryWorldTargetBinding(
        target_uuid=9,
        target_name="statue",
        commit_subject_relation_to_target="west",
        target_relation_from_agent="east",
        target_distance=1,
        commit_action={"action": "DROP", "arg1": 7},
        confidence=0.95,
        hypothesis_used="the target species",
        reason="task-native binding",
    )


def _candidate(action, role, *, effect, information, evidence=()):
    return DiscoveryWorldGroundedCandidate(
        action=action,
        target_role=role,
        prerequisite_probability=0.95,
        positive_effect_probability=effect,
        information_gain_probability=information,
        expected_effect="test",
        evidence=tuple(evidence),
        reason="test",
    )


def test_effect_guard_applicability_requires_policy_disagreement():
    observation = _applicability_observation(target_distance=2)
    candidates = (
        _candidate(
            {"action": "DROP", "arg1": 7},
            "COMMIT",
            effect=0.9,
            information=0.1,
            evidence=({"kind": "inventory", "uuid": 7, "name": "flag"},),
        ),
        _candidate(
            {"action": "TELEPORT_TO_OBJECT", "arg1": 9},
            "POSITION",
            effect=0.5,
            information=0.9,
        ),
    )
    receipt = assess_effect_guard_applicability(
        observation,
        _binding(),
        candidates,
        allowed_commit_actions=["DROP", "PUT"],
        minimum_binding_confidence=0.8,
        prerequisite_threshold=0.9,
        positive_effect_threshold=0.65,
    )
    assert receipt["eligible"]
    assert receipt["reason"] == "FIRST_SOURCE_EFFECT_GUARD_DISAGREEMENT"
    assert receipt["myopic_selected_role"] == "COMMIT"
    assert receipt["authentic_selected_role"] == "POSITION"
    assert receipt["positive_effect_commit_count"] == 0
    assert not receipt["outcome_fields_read_for_eligibility"]


def test_effect_guard_state_is_ineligible_once_exact_effect_is_witnessed():
    observation = _applicability_observation(target_distance=1)
    candidates = (
        _candidate(
            {"action": "DROP", "arg1": 7},
            "COMMIT",
            effect=0.9,
            information=0.1,
            evidence=(
                {"kind": "inventory", "uuid": 7, "name": "flag"},
                {
                    "kind": "relative_object",
                    "uuid": 9,
                    "name": "statue",
                    "relation_from_agent": "east",
                    "distance": 1,
                },
            ),
        ),
        _candidate(
            {"action": "TELEPORT_TO_OBJECT", "arg1": 9},
            "POSITION",
            effect=0.5,
            information=0.9,
        ),
    )
    receipt = assess_effect_guard_applicability(
        observation,
        _binding(),
        candidates,
        allowed_commit_actions=["DROP", "PUT"],
        minimum_binding_confidence=0.8,
        prerequisite_threshold=0.9,
        positive_effect_threshold=0.65,
    )
    assert not receipt["eligible"]
    assert receipt["reason"] == "POSITIVE_COMMIT_EFFECT_ALREADY_WITNESSED"


def test_effect_guard_rejects_intermediate_action_bound_as_commit():
    observation = _applicability_observation(target_distance=2)
    binding = DiscoveryWorldTargetBinding(
        target_uuid=9,
        target_name="statue",
        commit_subject_relation_to_target=None,
        target_relation_from_agent=None,
        target_distance=None,
        commit_action={"action": "USE", "arg1": 7, "arg2": 9},
        confidence=0.95,
        hypothesis_used="intermediate measurement",
        reason="incorrectly treated an intermediate action as final",
    )
    candidates = (
        _candidate(
            {"action": "USE", "arg1": 7, "arg2": 9},
            "COMMIT",
            effect=0.9,
            information=0.9,
        ),
        _candidate(
            {"action": "TELEPORT_TO_OBJECT", "arg1": 9},
            "POSITION",
            effect=0.5,
            information=0.9,
        ),
    )
    receipt = assess_effect_guard_applicability(
        observation,
        binding,
        candidates,
        allowed_commit_actions=["DROP", "PUT"],
        minimum_binding_confidence=0.8,
        prerequisite_threshold=0.9,
        positive_effect_threshold=0.65,
    )
    assert not receipt["eligible"]
    assert receipt["reason"] == "UNSUPPORTED_BOUND_COMMIT_ACTION"


def test_effect_guard_rejects_animal_when_task_requires_statue_target():
    observation = _applicability_observation(target_distance=2)
    binding = DiscoveryWorldTargetBinding(
        target_uuid=9,
        target_name="prismatic beast",
        commit_subject_relation_to_target="west",
        target_relation_from_agent="east",
        target_distance=1,
        commit_action={"action": "DROP", "arg1": 7},
        confidence=0.95,
        hypothesis_used="prismatic beast is anomalous",
        reason="incorrectly bound the animal rather than its statue",
    )
    candidates = (
        _candidate(
            {"action": "DROP", "arg1": 7},
            "COMMIT",
            effect=0.9,
            information=0.1,
        ),
        _candidate(
            {"action": "TELEPORT_TO_OBJECT", "arg1": 9},
            "POSITION",
            effect=0.5,
            information=0.9,
        ),
    )
    receipt = assess_effect_guard_applicability(
        observation,
        binding,
        candidates,
        allowed_commit_actions=["DROP", "PUT"],
        minimum_binding_confidence=0.8,
        prerequisite_threshold=0.9,
        positive_effect_threshold=0.65,
    )
    assert not receipt["eligible"]
    assert receipt["reason"] == "TARGET_BINDING_HEAD_NOT_TASK_SUPPORTED"
