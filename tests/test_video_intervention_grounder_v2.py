from motif_transfer.contracts import stable_hash
from motif_transfer.phase3_typed_effect_induction import TYPED_EFFECTS
from motif_transfer.video_intervention_grounder_v2 import (
    summarize_ledger_readiness,
    validate_intervention_ledger,
)


def _valid_record():
    before = stable_hash("before")
    after = stable_hash("after")
    intervention_id = "I0"
    return {
        "schema_version": "video-intervention-ledger-v2",
        "record_id": "R0",
        "benchmark": "fixture",
        "video_id": "V0",
        "split": "development",
        "belief_state_before": {"state_sha256": before},
        "intervention": {
            "intervention_id": intervention_id,
            "operator_type": "TEMPORAL_EVIDENCE_QUERY",
            "candidate_id": "C0",
        },
        "observations_by_horizon": {
            str(horizon): {
                "receipt_sha256": stable_hash(["receipt", horizon]),
                "evidence": [{"frame": horizon}],
            }
            for horizon in (1, 4, 8)
        },
        "typed_effects": {effect: 0.5 for effect in TYPED_EFFECTS},
        "effect_derivation": {
            "kind": "MEASURED_INTERVENTION_BELIEF_DELTA",
            "human_formula_used": False,
            "target_outcome_or_gold_used": False,
            "derivation_receipt_sha256": stable_hash("derivation"),
        },
        "belief_state_after": {"state_sha256": after},
        "transition": {
            "from_state_sha256": before,
            "to_state_sha256": after,
            "intervention_id": intervention_id,
        },
        "executability_by_horizon": {str(horizon): True for horizon in (1, 4, 8)},
        "blindness": {
            "gold_answer_read": False,
            "formal_success_read": False,
            "official_scene_graph_read": False,
            "functional_program_read": False,
            "source_identity_read": False,
        },
    }


def test_real_intervention_tuple_is_eligible():
    record = _valid_record()
    assert validate_intervention_ledger(record) == ()
    summary = summarize_ledger_readiness([record])
    assert summary["eligible_records"] == 1
    assert summary["eligible_unique_videos"] == 1


def test_static_qa_receipt_fails_closed():
    summary = summarize_ledger_readiness([{
        "video_id": "V0", "question": "what happened?", "answer": "x",
    }])
    assert summary["eligible_records"] == 0
    assert summary["ineligibility_reasons"]["MISSING_TOP_LEVEL:intervention"] == 1


def test_gold_read_or_hand_formula_is_rejected():
    record = _valid_record()
    record["blindness"]["gold_answer_read"] = True
    record["effect_derivation"]["human_formula_used"] = True
    errors = validate_intervention_ledger(record)
    assert "FORBIDDEN_READ_NOT_FALSE:gold_answer_read" in errors
    assert "HUMAN_FORMULA_NOT_EXCLUDED" in errors
