from dataclasses import asdict
import json

from motif_transfer.agqa_aggregate_temporal_transfer import (
    bind_aggregate_temporal_pair_program,
)
from motif_transfer.agqa_temporal_support_calibrator import (
    TemporalSupportExample,
    TemporalSupportRule,
    apply_temporal_support_rule,
    induce_temporal_support_rule,
)


def _receipt(start, end, view):
    return {
        "answer_read": False, "functional_program_read": False,
        "scene_graph_grounding_read": False, "source_identity_read": False,
        "question_read": False, "competing_operand_read": False,
        "receipt_sha256": view,
        "observations": [{
            "observability": "OBSERVED", "confidence": 0.9,
            "evidence_frames": [start, end], "start_frame": start,
            "end_frame": end,
        }],
    }


def _binding(span):
    return bind_aggregate_temporal_pair_program(
        task_id="t", target_state_sha256="s", target_grounder_sha256="g",
        source_program_sha256="p", obligation_kind="TEMPORAL_PAIR_RECURRENT",
        operand_runs={
            "A": {
                "primary_receipt": _receipt(0, span, "ap"),
                "rescan_receipt_global_timeline": _receipt(0, span, "ar"),
            },
            "B": {
                "primary_receipt": _receipt(20, 24, "bp"),
                "rescan_receipt_global_timeline": _receipt(20, 24, "br"),
            },
        },
        grounder_qualified=True,
    )


def _example(task, span, source, target):
    return TemporalSupportExample(
        split="dev", task_id=task, aggregate_authorized=True,
        singleton_view=None, minimum_cross_pair_gap=8,
        maximum_within_operand_endpoint_spread=0,
        maximum_interval_span=span, source_correct=source,
        target_native_correct=target,
    )


def test_induction_is_risk_first_and_finite():
    examples = [
        _example("loss", 3, False, True),
        _example("win-a", 6, True, False),
        _example("win-b", 12, True, False),
    ]
    rule, candidates = induce_temporal_support_rule(examples)
    assert len(candidates) == 768
    assert rule.training_losses == 0
    assert rule.training_wins == 2
    assert rule.minimum_max_interval_span == 6


def test_runtime_only_revokes_and_json_roundtrips():
    learned, _ = induce_temporal_support_rule([
        _example("loss", 3, False, True),
        _example("win", 6, True, False),
    ])
    restored = TemporalSupportRule.from_mapping(
        json.loads(json.dumps(asdict(learned)))
    )
    rejected = apply_temporal_support_rule(_binding(3), restored)
    accepted = apply_temporal_support_rule(_binding(6), restored)
    assert rejected.authorized_relation is None
    assert rejected.reason.startswith("SOURCE_ABSTAIN_TEMPORAL_SUPPORT_UNQUALIFIED")
    assert accepted.authorized_relation == "before"
