from __future__ import annotations

from motif_transfer.agqa_aggregate_temporal_transfer import (
    bind_aggregate_temporal_pair_program,
)
from motif_transfer.agqa_interval_reliability_calibrator import (
    IntervalReliabilityExample,
    apply_interval_reliability_rule,
    induce_interval_reliability_rule,
)
from motif_transfer.contracts import stable_hash


def _receipt(start: int, end: int) -> dict:
    body = {
        "answer_read": False, "functional_program_read": False,
        "scene_graph_grounding_read": False, "source_identity_read": False,
        "question_read": False, "competing_operand_read": False,
        "observations": [{
            "observability": "OBSERVED", "confidence": 0.9,
            "evidence_frames": [start, end],
            "start_frame": start, "end_frame": end,
        }],
    }
    return body | {"receipt_sha256": stable_hash(body)}


def _binding(*, gap: int, spread: int = 0):
    return bind_aggregate_temporal_pair_program(
        task_id="task", target_state_sha256=stable_hash("state"),
        target_grounder_sha256=stable_hash("grounder"),
        source_program_sha256=stable_hash("source"),
        obligation_kind="TEMPORAL_PAIR_RECURRENT",
        operand_runs={
            "A": {
                "primary_receipt": _receipt(2, 5),
                "rescan_receipt_global_timeline": _receipt(2 + spread, 5 + spread),
            },
            "B": {"rescan_receipt_global_timeline": _receipt(5 + spread + gap, 9 + spread + gap)},
        },
        grounder_qualified=True,
    )


def test_induction_uses_finite_gap_and_spread_class():
    examples = [
        IntervalReliabilityExample("win", True, "rescan", 2, 8, True, False),
        IntervalReliabilityExample("small-gap-loss", True, None, 1, 4, False, True),
        IntervalReliabilityExample("spread-loss", True, "rescan", 4, 40, False, True),
        IntervalReliabilityExample("strict-win", True, None, 5, 16, True, False),
    ]
    rule, candidates = induce_interval_reliability_rule(examples)
    assert rule.minimum_cross_pair_gap >= 2
    assert rule.maximum_within_operand_endpoint_spread <= 32
    assert len(candidates) == 192


def test_interval_rule_only_revokes_existing_binding():
    examples = [
        IntervalReliabilityExample("win", True, "rescan", 2, 8, True, False),
        IntervalReliabilityExample("small-gap-loss", True, None, 1, 4, False, True),
        IntervalReliabilityExample("spread-loss", True, "rescan", 4, 40, False, True),
    ]
    rule, _ = induce_interval_reliability_rule(examples)
    rejected = apply_interval_reliability_rule(_binding(gap=1), rule)
    assert rejected.authorized_relation is None
    accepted = apply_interval_reliability_rule(_binding(gap=4), rule)
    assert accepted.authorized_relation == "before"
