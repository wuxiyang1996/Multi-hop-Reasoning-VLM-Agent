from motif_transfer.agqa_directional_support_calibrator import (
    DirectionalSupportExample, induce_directional_support_rule,
)


def _row(task, relation, source, target):
    return DirectionalSupportExample(
        split="dev", task_id=task, aggregate_authorized=True,
        resolved_relation=relation, singleton_view=None,
        minimum_cross_pair_gap=8, maximum_within_operand_endpoint_spread=2,
        maximum_interval_span=12, source_correct=source,
        target_native_correct=target,
    )


def test_direction_is_induced_risk_first_from_finite_class():
    rule, candidates = induce_directional_support_rule([
        _row("before-win", "before", True, False),
        _row("after-win", "after", True, False),
        _row("after-loss", "after", False, True),
    ])
    assert len(candidates) == 2304
    assert rule.allowed_relations == ("before",)
    assert rule.training_wins == 1
    assert rule.training_losses == 0
