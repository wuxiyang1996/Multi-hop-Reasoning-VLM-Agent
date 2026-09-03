from __future__ import annotations

import inspect

from motif_transfer.agqa_aggregate_temporal_transfer import (
    bind_aggregate_temporal_pair_program,
)
from motif_transfer.contracts import stable_hash


def _receipt(start: int, end: int, *, observed: bool = True) -> dict:
    body = {
        "answer_read": False,
        "functional_program_read": False,
        "scene_graph_grounding_read": False,
        "source_identity_read": False,
        "question_read": False,
        "competing_operand_read": False,
        "observations": [{
            "observability": "OBSERVED" if observed else "UNOBSERVED",
            "confidence": 0.9,
            "evidence_frames": [start, end] if observed else [],
            "start_frame": start if observed else None,
            "end_frame": end if observed else None,
        }],
    }
    return body | {"receipt_sha256": stable_hash(body)}


def _bind(runs: dict, **overrides):
    values = {
        "task_id": "task",
        "target_state_sha256": stable_hash("state"),
        "target_grounder_sha256": stable_hash("grounder"),
        "source_program_sha256": stable_hash("source"),
        "obligation_kind": "TEMPORAL_PAIR_RECURRENT",
        "operand_runs": runs,
        "grounder_qualified": True,
    }
    values.update(overrides)
    return bind_aggregate_temporal_pair_program(**values)


def test_recurrence_is_attached_to_binary_operator_not_each_argument():
    binding = _bind({
        "A": {
            "primary_receipt": _receipt(2, 5),
            "rescan_receipt_global_timeline": _receipt(3, 6),
        },
        "B": {"primary_receipt": _receipt(20, 24)},
    })
    assert binding.binary_arguments_grounded
    assert binding.recurrent_operator_confirmed
    assert binding.authorized_relation == "before"
    assert binding.cross_view_relations == ("before", "before")


def test_two_single_views_do_not_establish_operator_recurrence():
    binding = _bind({
        "A": {"primary_receipt": _receipt(2, 5)},
        "B": {"primary_receipt": _receipt(20, 24)},
    })
    assert binding.binary_arguments_grounded
    assert not binding.recurrent_operator_confirmed
    assert binding.authorized_relation is None
    assert binding.reason == "SOURCE_ABSTAIN_OPERATOR_RECURRENCE_NOT_CONFIRMED"


def test_missing_argument_overlap_and_shuffled_effect_fail_closed():
    missing = _bind({
        "A": {"primary_receipt": _receipt(2, 5)},
        "B": {"primary_receipt": _receipt(20, 24, observed=False)},
    })
    assert missing.reason == "SOURCE_ABSTAIN_BINARY_ARGUMENT_NOT_GROUNDED"

    overlap = _bind({
        "A": {
            "primary_receipt": _receipt(2, 10),
            "rescan_receipt_global_timeline": _receipt(3, 9),
        },
        "B": {"primary_receipt": _receipt(8, 12)},
    })
    assert overlap.reason == "SOURCE_ABSTAIN_INTERVAL_HYPOTHESES_OVERLAP"

    shuffled = _bind({
        "A": {
            "primary_receipt": _receipt(2, 5),
            "rescan_receipt_global_timeline": _receipt(3, 6),
        },
        "B": {"primary_receipt": _receipt(20, 24)},
    }, effect_binding_authenticated=False)
    assert shuffled.reason == "SOURCE_EFFECT_BINDING_NOT_AUTHENTICATED"


def test_binding_api_cannot_read_direct_answer_or_gold():
    parameters = inspect.signature(
        bind_aggregate_temporal_pair_program
    ).parameters
    for forbidden in (
        "direct_response", "gold_answer", "functional_program",
        "formal_outcome",
    ):
        assert forbidden not in parameters
