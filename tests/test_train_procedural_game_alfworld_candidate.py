import json
from pathlib import Path

from scripts.train_procedural_game_alfworld_candidate import _collect


def test_train_and_evaluation_source_game_surfaces_are_disjoint() -> None:
    config = json.loads(
        (Path(__file__).parents[1] / "configs/procedural_game_alfworld_v1_development.json")
        .read_text(encoding="utf-8")
    )["source"]
    assert set(config["train_surfaces"]).isdisjoint(config["evaluation_surfaces"])


def test_small_collection_uses_matched_interventions() -> None:
    config = {
        "train_surfaces": ["g1"],
        "evaluation_surfaces": ["g2"],
        "train_domains_per_surface": 1,
        "evaluation_domains_per_surface": 1,
        "train_states_per_domain": 2,
        "evaluation_states_per_domain": 2,
        "replicates_per_action": 3,
        "train_seed": 1,
        "evaluation_seed": 2,
        "workflow": {
            "minimum_budget": 3,
            "maximum_budget": 6,
            "completion_probability_range": [0.4, 0.9],
            "failure_cost_range": [0.01, 0.1],
            "progress_reward": 0.15,
            "invalid_option_cost": 0.18,
        },
    }
    train = _collect(config, evaluation=False)
    evaluation = _collect(config, evaluation=True)
    assert not train.receipts
    assert len(evaluation.receipts) == evaluation.states * 5 * 3
