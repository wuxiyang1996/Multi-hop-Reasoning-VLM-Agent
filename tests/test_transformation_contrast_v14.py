from __future__ import annotations

import importlib
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
freezer = importlib.import_module("freeze_transformation_contrast_pool_v14")
enumerator = importlib.import_module(
    "enumerate_transformation_action_contrasts_v14"
)


def test_v14_families_are_transformation_only() -> None:
    assert set(freezer.FAMILIES) == {
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_heat_then_place_in_recep",
    }
    assert freezer._family(
        "pick_clean_then_place_in_recep-Apple-None-Bowl-1/game.tw-pddl"
    ) == "pick_clean_then_place_in_recep"


def test_edge_transition_rejects_nodes_and_lookups() -> None:
    assert enumerator._edge_transition({
        "source_transition": {"kind": "NODE"}
    }) is None
    assert enumerator._edge_transition({
        "source_transition": {"kind": "EDGE_LOOKUP"}
    }) is None
    edge = {"kind": "EDGE", "from": "BIND", "to": "MUTATE"}
    assert enumerator._edge_transition({
        "source_transition": edge
    }) == edge


def test_game_identity_matching_is_suffix_exact() -> None:
    assert enumerator._relative_game_matches(
        "/dataset/train/family/trial/game.tw-pddl",
        "family/trial/game.tw-pddl",
    )
    assert not enumerator._relative_game_matches(
        "/dataset/train/not-family/trial/game.tw-pddl",
        "family/trial/game.tw-pddl",
    )
