import json
from pathlib import Path

import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.phase3_tir_nonmaze import (
    CONDITIONS,
    FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
    SOURCE_INDUCED,
    SOURCE_PERMUTED,
    attach_grounding,
    candidate_feature_map,
    evaluate_matched_receipts,
    execute_condition,
    validate_grounder_artifact,
)
from motif_transfer.phase3_typed_effect_induction import TYPED_EFFECTS


REPO = Path(__file__).resolve().parents[1]


def _artifact(**thresholds):
    weights = [0.0 for _ in FEATURE_NAMES]
    weights[FEATURE_NAMES.index("raw_effect_probability")] = 8.0
    head = {
        "feature_names": list(FEATURE_NAMES),
        "means": [0.0 for _ in FEATURE_NAMES],
        "scales": [1.0 for _ in FEATURE_NAMES],
        "weights": weights,
        "intercept": -4.0,
    }
    body = {
        "schema_version": "phase3-tir-nonmaze-grounder-v2",
        "status": "DEVELOPMENT_GROUNDER_QUALIFIED",
        "formal_outcome_read_for_training_or_calibration": False,
        "source_program_updated": False,
        "heads": {name: dict(head) for name in TYPED_EFFECTS},
        "observation_head": {
            "feature_names": list(OBSERVATION_FEATURE_NAMES),
            "means": [0.0 for _ in OBSERVATION_FEATURE_NAMES],
            "scales": [1.0 for _ in OBSERVATION_FEATURE_NAMES],
            "weights": [0.0 for _ in OBSERVATION_FEATURE_NAMES],
            "intercept": 4.0,
        },
        "baseline_head": {"slope": 1.0, "intercept": 0.0},
        "thresholds": {
            "baseline_commit_confidence": thresholds.get("baseline", 0.95),
            "evidence_high_probability": thresholds.get("evidence", 0.6),
            "minimum_predicted_advantage": thresholds.get("advantage", 0.0),
        },
    }
    return body | {"artifact_sha256": stable_hash(body)}


def _candidate(index, *, h1, h4, h8, persistence, answers):
    actions = []
    for step in range(8):
        x = ((index * 3 + step) % 8) / 8
        actions.append({
            "tool": "extract_colors" if step % 2 == 0 else "zoom_region",
            "normalized_box": [x, 0.0, 0.125, 1.0],
        })
    probabilities = {
        horizon: {
            "answer": answer,
            "probabilities": {
                slot: (0.8 if slot == answer else 0.04) for slot in "ABCDEF"
            },
            "evidence_quality": {
                "referent_visible": 0.8,
                "local_detail_sufficient": 0.8,
                "question_coverage": 0.8,
                "contradiction_risk": 0.1,
            },
        }
        for horizon, answer in zip(("1", "4", "8"), answers)
    }
    return {
        "candidate_id": f"candidate-{index}",
        "planner_score": 0.5,
        "raw_typed_effect_probabilities": {
            "EFFECT_BY_TRANSITION_1": h1,
            "EFFECT_BY_TRANSITION_4": h4,
            "EFFECT_BY_TRANSITION_8": h8,
            "EXECUTABLE_TRANSITION_PERSISTENCE": persistence,
        },
        "actions": actions,
        "endpoints": probabilities,
        "transitions": [
            {"effect": {"nonredundant": True}} for _ in range(8)
        ],
    }


def _receipt(sample_id="fresh-1", gold="B"):
    return {
        "sample_id": sample_id,
        "family": "color",
        "gold_answer": gold,
        "formal_outcome_exposed_to_neural_calls": False,
        "image_size": [800, 800],
        "wrapper_routing": {"classes": ["compare", "ratio"]},
        "baseline": {
            "answer": "A",
            "probabilities": {
                "A": 0.4, "B": 0.2, "C": 0.1,
                "D": 0.1, "E": 0.1, "F": 0.1,
            },
        },
        "candidates": [
            _candidate(0, h1=.2, h4=.95, h8=.4, persistence=.3,
                       answers=("A", "B", "A")),
            _candidate(1, h1=.3, h4=.1, h8=.99, persistence=.4,
                       answers=("A", "A", "B")),
            _candidate(2, h1=.1, h4=.2, h8=.3, persistence=.99,
                       answers=("A", "A", "A")),
            _candidate(3, h1=.4, h4=.3, h8=.2, persistence=.1,
                       answers=("A", "A", "A")),
        ],
    }


def _sources():
    return [
        json.loads(path.read_text())
        for path in sorted((
            REPO / "configs/phase3_source_induction_v3/frozen_reserve/programs"
        ).glob("*.json"))
    ]


def test_feature_map_is_fixed_and_outcome_blind():
    receipt = _receipt()
    features = candidate_feature_map(
        receipt["candidates"][0], effect_type="EFFECT_BY_TRANSITION_4",
        image_size=receipt["image_size"], routing=receipt["wrapper_routing"],
    )
    assert tuple(sorted(features)) == FEATURE_NAMES
    assert "gold_answer" not in features
    assert features["horizon_fraction"] == 0.5
    assert 0 < features["prefix_coverage"] <= 1


def test_grounder_rejects_formal_outcome_taint_and_hash_drift():
    artifact = _artifact()
    validate_grounder_artifact(artifact)
    tainted = dict(artifact)
    tainted["formal_outcome_read_for_training_or_calibration"] = True
    body = dict(tainted)
    body.pop("artifact_sha256")
    tainted["artifact_sha256"] = stable_hash(body)
    with pytest.raises(ValueError, match="formal-outcome isolation"):
        validate_grounder_artifact(tainted)
    drifted = dict(artifact)
    drifted["status"] = "DRIFTED"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_grounder_artifact(drifted)


def test_attach_grounding_never_uses_source_identity_or_gold():
    grounded = attach_grounding(_receipt(), artifact=_artifact())
    audit = grounded["target_grounding_receipt"]
    assert audit["target_outcome_read"] is False
    assert audit["source_identity_used_as_feature"] is False
    assert len(grounded["candidates"]) == 4
    assert set(grounded["candidates"][0]["typed_effect_probabilities"]) == set(
        TYPED_EFFECTS
    )


def test_same_runtime_executes_authentic_and_permuted_source_conditions():
    receipt = _receipt()
    artifact = _artifact()
    authentic = execute_condition(
        receipt, condition=SOURCE_INDUCED,
        grounder_artifact=artifact, source_artifacts=_sources(),
    )
    permuted = execute_condition(
        receipt, condition=SOURCE_PERMUTED,
        grounder_artifact=artifact, source_artifacts=_sources(),
    )
    assert authentic["source_ir_implementation"].endswith("AnonymousAttemptRuntime")
    assert permuted["source_ir_implementation"].endswith("AnonymousAttemptRuntime")
    assert authentic["decision"]["portfolio_receipt"]["target_outcome_read"] is False
    assert authentic["decision"]["selected_effect_type"] in TYPED_EFFECTS
    assert permuted["decision"]["effect_binding_control_receipt"]["nonidentity"]


def test_matched_evaluator_runs_all_five_arms_and_fail_closed_gates():
    report = evaluate_matched_receipts(
        [_receipt("one"), _receipt("two")],
        grounder_artifact=_artifact(), source_artifacts=_sources(),
        role="qualification",
        gates={
            "expected_tasks": 2,
            "minimum_ceiling_successes": 2,
            "minimum_source_action_contrasts": 0,
            "minimum_permuted_action_contrasts": 0,
            "minimum_selected_effect_types": 1,
            "maximum_negative_transfer_rate": 1.0,
            "required_gate_names": ["expected_task_count", "target_native_ceiling_capable"],
        },
    )
    assert set(report["successes"]) == set(CONDITIONS)
    assert report["status"] == "TIR_PHASE3_QUALIFICATION_PASSED"
    assert report["same_frozen_source_ir"] is True
    assert report["formal_outcome_exposed_to_neural_or_source_selection"] is False
