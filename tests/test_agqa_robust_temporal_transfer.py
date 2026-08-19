from __future__ import annotations

import inspect
import json
from pathlib import Path

from motif_transfer.agqa_robust_temporal_transfer import (
    bind_robust_temporal_pair_program,
    build_temporal_harness,
    build_temporal_route,
    decide_temporal_relation,
    unified_temporal_grounding,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.structural_ir_applicability import (
    temporal_function_artifact_contract,
)
from motif_transfer.unified_transfer_runtime import (
    PairedCalibration,
    TransferVerdict,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = json.loads((
    ROOT
    / "configs/phase3_source_function_v4/frozen_reserve/programs/candy_crush.json"
).read_text())
PROGRAM = ARTIFACT["source_function_program"]
CONTRACT = temporal_function_artifact_contract(
    ARTIFACT,
    source_confirmation_sha256=stable_hash("source-confirmation"),
    source_intervention_qualified=True,
)
GROUNDER = stable_hash("agqa-robust-temporal-grounder")
EXECUTOR = stable_hash("agqa-temporal-executor")


def _receipt(start, end, *, observed=True):
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
            "evidence_frames": [start, end],
            "start_frame": start,
            "end_frame": end,
        }],
    }
    return body | {"receipt_sha256": stable_hash(body)}


def _runs(*, conflict=False, overlap=False):
    b_second = (8, 12) if overlap else (31, 35)
    if conflict:
        b_second = (0, 4)
    return {
        "A": {
            "primary_receipt": _receipt(10, 14),
            "rescan_receipt_global_timeline": _receipt(11, 15),
        },
        "B": {
            "primary_receipt": _receipt(30, 34),
            "rescan_receipt_global_timeline": _receipt(*b_second),
        },
    }


def _binding(**overrides):
    arguments = {
        "task_id": "temporal-task",
        "target_state_sha256": stable_hash("target-state"),
        "target_grounder_sha256": GROUNDER,
        "source_program_sha256": CONTRACT.program_sha256,
        "obligation_kind": "TEMPORAL_PAIR_RECURRENT",
        "operand_runs": _runs(),
        "grounder_qualified": True,
    }
    arguments.update(overrides)
    return bind_robust_temporal_pair_program(**arguments)


def _harness():
    calibration = PairedCalibration(4, 0, 31)
    route = build_temporal_route(
        source_program_sha256=CONTRACT.program_sha256,
        target_grounder_sha256=GROUNDER,
        target_executor_sha256=EXECUTOR,
        evidence_report_sha256=stable_hash("development-report"),
        utility_vs_target_native=calibration,
        authenticity_vs_effect_shuffled=calibration,
    )
    return build_temporal_harness(
        contract=CONTRACT,
        source_transition_receipts_sha256=PROGRAM[
            "source_receipts_sha256"
        ],
        inducer_artifact_sha256=ARTIFACT["artifact_sha256"],
        route=route,
    )


def test_all_interval_hypothesis_pairs_agree_before_execution():
    binding = _binding()
    assert binding.authorized_relation == "before"
    assert binding.cross_view_relations == (
        "before", "before", "before", "before",
    )
    target = unified_temporal_grounding(
        contract=CONTRACT, binding=binding,
    )
    decision = decide_temporal_relation(
        harness=_harness(), target=target, binding=binding,
        target_executor_sha256=EXECUTOR,
    )
    assert decision.phase7.verdict == TransferVerdict.SELECT_SKILL
    assert decision.source_relation == "before"
    assert decision.executor_calls == 1


def test_overlap_or_relation_conflict_fails_closed():
    overlap = _binding(operand_runs=_runs(overlap=True))
    assert overlap.authorized_relation is None
    assert overlap.reason == "SOURCE_ABSTAIN_INTERVAL_HYPOTHESES_OVERLAP"

    conflict = _binding(operand_runs=_runs(conflict=True))
    assert conflict.authorized_relation is None
    assert conflict.reason == "SOURCE_ABSTAIN_TEMPORAL_RELATION_CONFLICT"


def test_recurrence_and_authentic_effect_are_required():
    one_view = _runs()
    one_view["A"].pop("rescan_receipt_global_timeline")
    binding = _binding(operand_runs=one_view)
    assert binding.authorized_relation is None
    assert binding.reason == "SOURCE_ABSTAIN_RECURRENCE_NOT_CONFIRMED"

    shuffled = _binding(effect_binding_authenticated=False)
    assert shuffled.authorized_relation is None
    assert shuffled.reason == "SOURCE_EFFECT_BINDING_NOT_AUTHENTICATED"


def test_binding_api_has_no_direct_answer_program_or_gold_input():
    parameters = inspect.signature(
        bind_robust_temporal_pair_program
    ).parameters
    for forbidden in (
        "direct_response", "gold_answer", "functional_program",
        "formal_outcome",
    ):
        assert forbidden not in parameters
