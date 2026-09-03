import numpy as np

from scripts.train_agqa2_program_router_v1 import select_threshold


def test_select_threshold_prefers_high_recall_under_precision_gate():
    labels = np.asarray([1, 1, 1, 0, 0])
    scores = np.asarray([0.99, 0.9, 0.8, 0.7, 0.1])
    result = select_threshold(labels, scores, minimum_precision=1.0, minimum_selected=2)
    assert result["selected"] == 3
    assert result["false_positive"] == 0


def test_select_threshold_fails_closed_without_support():
    labels = np.asarray([1, 0])
    scores = np.asarray([0.9, 0.8])
    try:
        select_threshold(labels, scores, minimum_precision=1.0, minimum_selected=2)
    except ValueError:
        pass
    else:
        raise AssertionError("threshold selection must fail closed")
