from __future__ import annotations

from copy import deepcopy

from motif_transfer.discoveryworld_qualification import select_first_commit_fork


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
