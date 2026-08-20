from __future__ import annotations

from motif_transfer.contracts import stable_hash
from motif_transfer.cyclic_identity_induction import (
    DATASET_VERSION,
    evaluate_cyclic_program,
    induce_cyclic_identity_program,
    permute_recovery_effect_bindings,
    permute_terminal_labels,
    subset_cyclic_dataset,
    validate_cyclic_dataset,
)


def _dataset() -> dict:
    episodes = []
    for episode_index, probe in enumerate((1, 3)):
        candidates = []
        for recovery in range(4):
            transition_body = {
                "state_element": probe,
                "anonymous_intervention_phase": "RECOVERY",
                "effect_element": recovery,
                "next_state_element": (probe + recovery) % 4,
                "raw_action_exported": False,
            }
            transition = transition_body | {
                "transition_sha256": stable_hash(transition_body)
            }
            candidate_body = {
                "candidate_id": f"C{recovery}",
                "probe_effect_element": probe,
                "recovery_effect_element": recovery,
                "primitive_transitions": [transition],
                "returned_to_identity": (probe + recovery) % 4 == 0,
                "raw_strategy_exported": False,
            }
            candidates.append(candidate_body | {
                "candidate_sha256": stable_hash(candidate_body)
            })
        episode_body = {
            "episode_id": f"episode-{episode_index}",
            "seed_commitment": stable_hash(episode_index),
            "group_order": 4,
            "probe_primitive_steps": probe,
            "candidates": candidates,
        }
        episodes.append(episode_body | {
            "episode_sha256": stable_hash(episode_body)
        })
    body = {
        "schema_version": DATASET_VERSION,
        "role": "development",
        "config_sha256": stable_hash("config"),
        "official_tetris_environment_file_sha256": stable_hash("env"),
        "episodes": episodes,
        "attempted_seeds": 2,
        "retained_order_four_episodes": 2,
        "raw_source_action_tokens_exported": False,
        "target_data_read": False,
    }
    return body | {"dataset_sha256": stable_hash(body)}


def test_source_only_cyclic_relation_is_induced_not_preprovided() -> None:
    dataset = _dataset()
    validate_cyclic_dataset(dataset)
    one = subset_cyclic_dataset(dataset, ["episode-0"])
    assert induce_cyclic_identity_program(one)["status"].startswith("ABSTAIN")

    program = induce_cyclic_identity_program(dataset)
    assert program["status"] == "SOURCE_CYCLIC_IDENTITY_PROGRAM_INDUCED"
    assert program["selected_relation"] == (
        "COMPOSE_PROBE_RECOVERY_TO_IDENTITY"
    )
    assert program["target_data_read"] is False
    assert program["raw_source_action_tokens_exported"] is False
    assert evaluate_cyclic_program(program, dataset) == {
        "correct": 8,
        "total": 8,
        "all_forks_classified": True,
        "positive_support": 2,
        "false_positive_support": 0,
    }


def test_cyclic_effect_and_terminal_controls_fail_closed() -> None:
    dataset = _dataset()
    label = induce_cyclic_identity_program(
        permute_terminal_labels(dataset)
    )
    binding = induce_cyclic_identity_program(
        permute_recovery_effect_bindings(dataset)
    )
    assert label["status"].startswith("ABSTAIN")
    assert binding["status"].startswith("ABSTAIN")


def test_cyclic_dataset_rejects_raw_source_action_token() -> None:
    dataset = _dataset()
    dataset["episodes"][0]["candidates"][0]["primitive_transitions"][0][
        "raw_action_exported"
    ] = True
    body = dataset["episodes"][0]["candidates"][0][
        "primitive_transitions"
    ][0]
    body["transition_sha256"] = stable_hash({
        key: value for key, value in body.items()
        if key != "transition_sha256"
    })
    candidate = dataset["episodes"][0]["candidates"][0]
    candidate["candidate_sha256"] = stable_hash({
        key: value for key, value in candidate.items()
        if key != "candidate_sha256"
    })
    episode = dataset["episodes"][0]
    episode["episode_sha256"] = stable_hash({
        key: value for key, value in episode.items()
        if key != "episode_sha256"
    })
    dataset["dataset_sha256"] = stable_hash({
        key: value for key, value in dataset.items()
        if key != "dataset_sha256"
    })
    try:
        validate_cyclic_dataset(dataset)
    except ValueError as error:
        assert "source action" in str(error)
    else:
        raise AssertionError("raw source action token was accepted")
