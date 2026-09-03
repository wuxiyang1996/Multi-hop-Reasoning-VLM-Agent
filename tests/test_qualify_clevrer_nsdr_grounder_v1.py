from scripts.qualify_clevrer_nsdr_grounder_v1 import _attribute_score, _collision_matches


def _obj(color: str, material: str = "metal", shape: str = "cube") -> dict:
    return {"color": color, "material": material, "shape": shape}


def test_attribute_assignment_is_permutation_invariant() -> None:
    gold = [_obj("red"), _obj("blue", "rubber", "sphere")]
    assert _attribute_score(list(reversed(gold)), gold) == (6, 6)
    assert _attribute_score([_obj("red")], gold) == (3, 6)


def test_collision_matching_uses_attributes_and_frame_tolerance() -> None:
    pair = [_obj("red"), _obj("blue")]
    predicted = [{"frame": 35, "objects": list(reversed(pair))}]
    gold = [{"frame": 34, "objects": pair}]
    assert _collision_matches(predicted, gold) == 1
    assert _collision_matches([{"frame": 50, "objects": pair}], gold) == 0
