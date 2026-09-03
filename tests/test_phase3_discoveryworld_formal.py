from copy import deepcopy

from motif_transfer.phase3_discoveryworld_formal import (
    exact_two_sided_sign_p,
    select_outcome_blind_formal_fork,
)


def _reference():
    species = (
        ("echojelly", 0.50, 0.37),
        ("prismatic beast", 0.54, 0.18),
        ("spheroid", 0.78, 0.71),
    )
    steps = []
    for index, (name, protein_a, protein_b) in enumerate(species, start=1):
        facts = {
            "inventory": ([{"uuid": 7, "name": "flag"}] if index == 3 else []),
            "accessible_objects": [],
            "salient_relative_objects": ([
                {"uuid": 9, "name": "statue of a spheroid"},
            ] if index == 3 else []),
            "last_action_message": (
                f"You use the proteomics meter to investigate the {name}.\n"
                f"- Protein A: {protein_a}\n- Protein B: {protein_b}\n"
            ),
            "task_progress": [{
                "completed": index == 3, "completedSuccessfully": index == 3,
                "score": 999,
            }],
        }
        steps.append({
            "episode_step": index,
            "after_target_native_facts": facts,
            "transition": {
                "terminal": index == 3, "official_success": index == 3,
                "action_succeeded": False,
            },
            "action": {"action": "DROP"},
        })
    return {
        "steps": steps,
        "evaluation": {"official_success": True, "scorecard": ["poison"]},
    }


def test_formal_fork_is_outcome_blind_and_selects_first_structural_state():
    reference = _reference()
    receipt = select_outcome_blind_formal_fork(reference)
    assert receipt["fork_after_episode_step"] == 3
    assert receipt["acquisition_outlier_candidates"] == ["spheroid"]
    assert receipt["forbidden_fields_read"] is False

    poisoned = deepcopy(reference)
    poisoned["evaluation"] = {"official_success": False, "scorecard": []}
    for row in poisoned["steps"]:
        row["transition"] = {
            "terminal": False, "official_success": False,
            "action_succeeded": True,
        }
        row["action"] = {"action": "TELEPORT_TO_OBJECT", "arg1": 12345}
        row["after_target_native_facts"]["task_progress"] = [{
            "completed": False, "completedSuccessfully": False, "score": -1,
        }]
    assert select_outcome_blind_formal_fork(poisoned) == receipt


def test_exact_two_sided_sign_test():
    assert exact_two_sided_sign_p(0, 0) == 1.0
    assert exact_two_sided_sign_p(6, 0) == 0.03125
    assert exact_two_sided_sign_p(3, 3) == 1.0
