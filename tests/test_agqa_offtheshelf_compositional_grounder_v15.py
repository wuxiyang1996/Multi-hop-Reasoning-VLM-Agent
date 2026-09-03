from scripts.compile_agqa_offtheshelf_compositional_grounder_v15 import (
    _action_localization,
    _family_scores,
    _head,
)


def _score(class_id, phrase, values):
    return {
        "class_id": class_id,
        "phrase": phrase,
        "window_scores": values,
    }


def test_public_action_head_normalization_is_generic():
    assert _head("someone is smiling") == "smiling"
    assert _head("holding some food") == "holding"


def test_family_scores_prefers_explicit_public_class_id():
    rows = [
        _score("c001", "holding a book", [0.1, 0.2, 0.3]),
        _score("c002", "holding a bag", [0.9, 0.8, 0.7]),
    ]
    selected = _family_scores({"class_id": "c001", "phrase": "holding"}, rows)
    assert [row["class_id"] for row in selected] == ["c001"]


def test_object_typed_localization_ignores_generic_something_class():
    rows = [
        _score("c156", "someone is eating something", [0.99, 0.99, 0.99]),
        _score("c062", "eating some food", [0.1, 0.8, 0.2]),
    ]
    views = [[0, 10], [10, 20], [20, 30]]
    result = _action_localization(
        {"phrase": "eating"}, rows, views, 0.5, require_object=True,
    )
    assert result["checkpoint_class_id"] == "c062"
    assert result["argmax_window"] == 1


def test_peak_connected_component_does_not_bridge_inactive_window():
    rows = [_score("c001", "holding a book", [0.9, 0.1, 0.8])]
    views = [[0, 10], [10, 20], [20, 30]]
    result = _action_localization(
        {"phrase": "holding a book"}, rows, views, 0.5,
        require_object=True,
    )
    assert result["active_windows"] == [0]
    assert result["native_lower"] == result["native_upper"] == 5
