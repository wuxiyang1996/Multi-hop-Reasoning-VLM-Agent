from motif_transfer.agqa_asymmetric_support_calibrator import (
    AsymmetricExample,
    induce_asymmetric_rule,
)


def _row(task, relation, span, source, target):
    return AsymmetricExample(
        "dev", task, True, relation, None, 8, 2, span, source, target,
    )


def test_relation_conditional_support_is_induced_from_finite_class():
    rule, candidates = induce_asymmetric_rule([
        _row("before-win", "before", 6, True, False),
        _row("before-loss", "before", 3, False, True),
        _row("after-win", "after", 12, True, False),
        _row("after-loss", "after", 6, False, True),
    ])
    assert len(candidates) == 3072
    assert rule.before_minimum_interval_span == 6
    assert rule.after_minimum_interval_span == 12
    assert rule.training_wins == 2
    assert rule.training_losses == 0
