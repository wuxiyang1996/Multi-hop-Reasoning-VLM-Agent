from __future__ import annotations

from motif_transfer.candy_native_grounder import candy_action_features


def test_candy_action_features_detect_created_match() -> None:
    observation = """Board:
0| R R G C C C C C
1| G G R C C C C C
2| G G R C C C C C
3| C C C C C C C C
4| C C C C C C C C
5| C C C C C C C C
6| C C C C C C C C
7| C C C C C C C C
Score: 0
Moves Left: 50
"""
    features = candy_action_features(observation, "((0,2),(1,2))")
    assert features[6] == 1.0
    assert features[7] >= 3 / 8
