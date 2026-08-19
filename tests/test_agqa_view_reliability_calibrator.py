from __future__ import annotations

from dataclasses import asdict, replace
import json

from motif_transfer.agqa_aggregate_temporal_transfer import (
    bind_aggregate_temporal_pair_program,
)
from motif_transfer.agqa_view_reliability_calibrator import (
    ViewReliabilityExample,
    ViewReliabilityRule,
    apply_view_reliability_rule,
    induce_view_reliability_rule,
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


def _binding(singleton: str):
    fields = {
        "primary": "primary_receipt",
        "rescan": "rescan_receipt_global_timeline",
        "tiebreak": "tiebreak_receipt_global_timeline",
    }
    return bind_aggregate_temporal_pair_program(
        task_id="task", target_state_sha256=stable_hash("state"),
        target_grounder_sha256=stable_hash("grounder"),
        source_program_sha256=stable_hash("source"),
        obligation_kind="TEMPORAL_PAIR_RECURRENT",
        operand_runs={
            "A": {fields[singleton]: _receipt(2, 5)},
            "B": {
                "primary_receipt": _receipt(20, 23),
                "rescan_receipt_global_timeline": _receipt(21, 24),
            },
        },
        grounder_qualified=True,
    )


def test_finite_rule_induction_rejects_lossy_primary_singletons():
    examples = [
        ViewReliabilityExample("strict-win", True, None, True, False),
        ViewReliabilityExample("primary-loss", True, "primary", False, True),
        ViewReliabilityExample("rescan-win", True, "rescan", True, False),
        ViewReliabilityExample("tiebreak-win", True, "tiebreak", True, False),
        ViewReliabilityExample("tie", False, None, True, True),
    ]
    rule, candidates = induce_view_reliability_rule(examples)
    assert rule.allowed_singleton_views == ("rescan", "tiebreak")
    assert rule.training_wins == 3
    assert rule.training_losses == 0
    assert len(candidates) == 8
    round_trip = json.loads(json.dumps(asdict(rule)))
    assert ViewReliabilityRule.from_mapping(round_trip) == rule


def test_runtime_rule_can_only_revoke_not_invent_binding():
    examples = [
        ViewReliabilityExample("strict-win", True, None, True, False),
        ViewReliabilityExample("primary-loss", True, "primary", False, True),
        ViewReliabilityExample("rescan-win", True, "rescan", True, False),
    ]
    rule, _ = induce_view_reliability_rule(examples)
    primary = _binding("primary")
    blocked = apply_view_reliability_rule(primary, rule)
    assert primary.authorized_relation == "before"
    assert blocked.authorized_relation is None
    assert blocked.reason.endswith(":primary")

    rescan = _binding("rescan")
    assert apply_view_reliability_rule(rescan, rule) == rescan

    unauthenticated = replace(
        rescan, effect_binding_authenticated=False,
        authorized_relation=None, receipt_sha256="",
    )
    body = asdict(unauthenticated)
    body.pop("receipt_sha256")
    unauthenticated = replace(
        unauthenticated, receipt_sha256=stable_hash(body),
    )
    assert apply_view_reliability_rule(
        unauthenticated, rule,
    ).authorized_relation is None
