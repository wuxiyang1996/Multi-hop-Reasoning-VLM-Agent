from scripts.train_agqa2_multiclass_program_router_v2 import select_class_threshold


def test_class_threshold_requires_prediction_and_parser_agreement():
    result = select_class_threshold(
        labels=["A", "A", "A", "B", "B"],
        predicted=["A", "A", "A", "A", "B"],
        scores=[0.99, 0.95, 0.90, 0.85, 0.99],
        route="A",
        plan_routes=["A", "A", "B", "A", "B"],
        minimum_precision=1.0,
        minimum_selected=2,
    )
    assert result["selected"] == 2
    assert result["true_positive"] == 2
    assert result["false_positive"] == 0
    assert result["threshold"] == 0.95


def test_class_threshold_fails_closed_without_support():
    try:
        select_class_threshold(
            labels=["A", "B"], predicted=["A", "B"], scores=[0.9, 0.9],
            route="A", plan_routes=["A", "B"], minimum_precision=1.0,
            minimum_selected=2,
        )
    except ValueError as error:
        assert "no qualified threshold" in str(error)
    else:
        raise AssertionError("unsupported route threshold should fail closed")
