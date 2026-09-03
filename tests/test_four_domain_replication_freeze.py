from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
MASTER = CONFIGS / "four_domain_replication_v1_manifest.json"


def _read(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _alfworld_ids(config_name: str) -> set[str]:
    def walk(value: object) -> set[str]:
        if isinstance(value, dict):
            return set().union(*(walk(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(walk(child) for child in value))
        if isinstance(value, str) and value.endswith("game.tw-pddl"):
            return {value}
        return set()

    return walk(_read(config_name))


def test_replication_manifest_binds_every_frozen_component() -> None:
    master = json.loads(MASTER.read_text(encoding="utf-8"))
    assert master["status"] == (
        "FROZEN_BEFORE_ANY_REPLICATION_TARGET_RESET_OR_QUERY"
    )
    assert master["selection_saw_task_content_or_outcome"] is False
    assert master["aggregate_estimand"][
        "thresholds_or_tasks_may_change_after_open"
    ] is False
    assert master["domains"] == {
        "alfworld": {
            "tasks": 32,
            "config": "alfworld_procedural_game_replication_v1_frozen.json",
        },
        "webshop": {
            "tasks": 32,
            "config": "webshop_sokoban_effect_replication_v1_frozen.json",
        },
        "discoveryworld": {
            "tasks": 20,
            "config": "discoveryworld_sokoban_replication_v1_protocol.json",
        },
        "tir": {
            "tasks": 48,
            "config": "tir_maze_topology_replication_v1_frozen.json",
        },
    }
    for name, expected in master["component_file_sha256"].items():
        assert _sha256(CONFIGS / name) == expected
    assert _sha256(REPO / "scripts/freeze_four_domain_replication_v1.py") == master[
        "freezer_file_sha256"
    ]


def test_alfworld_replication_and_multiplicity_reserves_are_disjoint() -> None:
    manifest = _read("alfworld_procedural_game_replication_v1_manifest.json")
    splits = manifest["cells"]["alfworld_valid_unseen"]["splits"]
    replication = set(splits["held_out"])
    multiplicity = set(splits["multiplicity_formal_locked"])
    assert manifest["outcome_or_task_content_read_for_selection"] is False
    assert len(replication) == 32
    assert len(multiplicity) == 6
    assert replication.isdisjoint(multiplicity)
    assert all(not row.startswith("pick_two_obj_and_place") for row in replication)
    assert all(row.startswith("pick_two_obj_and_place") for row in multiplicity)
    assert multiplicity == set(
        _read("four_domain_replication_v1_manifest.json")[
            "locked_future_alfworld_multiplicity_ids"
        ]
    )
    consumed = set()
    for name in (
        "alfworld_v2_outcome_blind_pool.json",
        "sokoban_alfworld_effect_transfer_split_v2.json",
        "sokoban_alfworld_transfer_split_v1.json",
    ):
        consumed.update(_alfworld_ids(name))
    assert len(consumed) == 144
    assert replication.isdisjoint(consumed)
    assert multiplicity.isdisjoint(consumed)


def test_webshop_discoveryworld_and_tir_reserves_are_exact_and_unique() -> None:
    webshop = _read("webshop_sokoban_effect_replication_v1_frozen.json")
    assert webshop["task_ids"] == [f"webshop.{index}" for index in range(146, 178)]
    assert webshop["goal_text_read_or_run"] is False
    assert webshop["runtime"]["number_of_goals"] == 178

    discovery = _read("discoveryworld_replication_v1_manifest.json")
    tasks = discovery["roles"]["formal_reserve"]
    assert len(tasks) == 20
    assert {row["scenario"] for row in tasks} == {"Space Sick", "Proteomics"}
    assert {row["difficulty"] for row in tasks} == {"Easy"}
    assert {row["seed"] for row in tasks} == set(range(11, 21))
    assert discovery["status"] == "FROZEN_BEFORE_ANY_SEED11_TO_SEED20_ROLLOUT"

    tir = _read("tir_maze_topology_replication_v1_frozen.json")
    heldout = tir["splits"]["heldout"]
    assert len(heldout) == len(set(heldout)) == 48
    prior = _read("tir_maze_topology_v2_frozen.json")
    prior_ids = {
        str(item)
        for rows in prior["splits"].values()
        if isinstance(rows, list)
        for item in rows
    }
    assert set(heldout).isdisjoint(prior_ids)
    assert tir["replication"]["prior_final_report_used_for_policy_change"] is False
