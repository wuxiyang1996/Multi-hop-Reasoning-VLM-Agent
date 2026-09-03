from __future__ import annotations

import pytest

from motif_transfer.clevrer_query_compiler import (
    compile_choice,
    compile_object,
    compile_question,
)


def test_object_filters_are_canonical() -> None:
    assert compile_object("the red metal cube") == [
        "objects", "red", "filter_color", "metal", "filter_material",
        "cube", "filter_shape", "unique",
    ]
    assert compile_object("the purple object") == [
        "objects", "purple", "filter_color", "unique",
    ]


def test_predictive_and_counterfactual_questions() -> None:
    assert compile_question("What will happen next?", "predictive") == [
        "unseen_events", "belong_to",
    ]
    assert compile_question(
        "Without the yellow sphere, which event will not happen?",
        "counterfactual",
    ) == [
        "all_events", "objects", "yellow", "filter_color", "sphere",
        "filter_shape", "unique", "filter_counterfact", "belong_to", "negate",
    ]


def test_explanatory_question_and_choices() -> None:
    assert compile_question(
        "Which of the following is not responsible for the cube's colliding with the cylinder?",
        "explanatory",
    ) == [
        "events", "events", "objects", "cube", "filter_shape", "unique",
        "filter_collision", "objects", "cylinder", "filter_shape", "unique",
        "filter_collision", "unique", "filter_ancestor", "belong_to", "negate",
    ]
    assert compile_choice(
        "the collision between the rubber sphere and the cube", "explanatory",
    ) == [
        "events", "objects", "rubber", "filter_material", "sphere",
        "filter_shape", "unique", "filter_collision", "objects", "cube",
        "filter_shape", "unique", "filter_collision", "unique",
    ]
    assert compile_choice("the brown object's entrance", "explanatory") == [
        "events", "objects", "brown", "filter_color", "unique",
        "filter_in", "unique",
    ]


def test_unsupported_surface_fails_closed() -> None:
    with pytest.raises(ValueError):
        compile_question("Why did it happen?", "explanatory")
    with pytest.raises(ValueError):
        compile_object("the object")
