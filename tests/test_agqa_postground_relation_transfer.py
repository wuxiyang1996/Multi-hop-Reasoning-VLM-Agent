from __future__ import annotations

import json
from pathlib import Path

from motif_transfer.agqa_goal_relation_transfer import (
    build_harness, build_route,
)
from motif_transfer.agqa_postground_relation_evaluation import (
    freeze_postground_predictions,
)
from motif_transfer.agqa_postground_relation_transfer import (
    bind_postground_source_program,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.unified_transfer_runtime import PairedCalibration


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = json.loads((
    ROOT / "runs/sokoban_goal_relation_macro_v3/artifact.json"
).read_text())
CONFIRMATION = json.loads((
    ROOT / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
).read_text())
GROUNDER = stable_hash("postground-v31-grounder")
EXECUTOR = stable_hash("postground-v31-executor")


def _harness():
    route = build_route(
        source_program_sha256=ARTIFACT["artifact_sha256"],
        target_grounder_sha256=GROUNDER,
        target_executor_sha256=EXECUTOR,
        evidence_report_sha256=stable_hash("postground-development"),
        utility_vs_target_native=PairedCalibration(14, 0, 255),
        authenticity_vs_effect_shuffled=PairedCalibration(14, 0, 255),
    )
    return build_harness(
        artifact=ARTIFACT,
        confirmation=CONFIRMATION,
        inducer_artifact_sha256=stable_hash("inducer"),
        route=route,
    )


def test_raw_vote_disagreement_is_not_multiple_symbolic_bindings():
    execution = {
        "decision": "box",
        "neural_votes": [
            {"view": "a", "decision": "box"},
            {"view": "b", "decision": "box"},
            {"view": "c", "decision": "floor"},
        ],
    }
    receipt = bind_postground_source_program(
        artifact=ARTIFACT,
        confirmation=CONFIRMATION,
        task_id="task",
        target_state_sha256=stable_hash("state"),
        target_grounder_sha256=GROUNDER,
        calibrated_execution=execution,
        grounder_qualified=True,
    )
    assert receipt.candidate_bindings == ("box",)
    assert receipt.authorized_candidate == "box"


def test_explicit_multiple_resolved_bindings_still_trigger_source_abstention():
    execution = {
        "decision": None,
        "candidate_bindings": ["box", "floor"],
        "neural_votes": [],
    }
    receipt = bind_postground_source_program(
        artifact=ARTIFACT,
        confirmation=CONFIRMATION,
        task_id="task",
        target_state_sha256=stable_hash("state"),
        target_grounder_sha256=GROUNDER,
        calibrated_execution=execution,
        grounder_qualified=True,
    )
    assert receipt.target_binding_count == 2
    assert receipt.authorized_candidate is None
    assert receipt.reason == "SOURCE_ABSTAIN_MULTIPLE_TARGET_BINDINGS"


def test_source_matches_handwritten_ceiling_and_shuffled_effect_abstains():
    row = {
        "task_id": "synthetic-postground",
        "runtime_receipt_sha256": stable_hash("synthetic-runtime"),
        "direct_response": "table",
        "object_ontology_receipts": [
            {
                "decision": "box", "confidence": 0.8,
                "relation_observed": True, "evidence_frames": [1],
            },
            {
                "decision": "floor", "confidence": 0.9,
                "relation_observed": True, "evidence_frames": [2],
            },
        ],
        "calibrated_target_native_execution": {
            "decision": "box",
            "neural_votes": [
                {"view": "isolated_relation", "decision": "floor"},
                {"view": "ontology_0", "decision": "box"},
                {"view": "ontology_1", "decision": "box"},
            ],
        },
        "runtime_answer_read": False,
        "runtime_functional_program_read": False,
        "runtime_scene_graph_read": False,
        "runtime_source_identity_read": False,
        "operand_grounder_question_read": False,
        "operand_grounder_competing_operand_read": False,
        "object_ontology_original_question_read": False,
        "object_ontology_answer_candidates_read": False,
    }
    frozen = freeze_postground_predictions(
        row=row,
        artifact=ARTIFACT,
        confirmation=CONFIRMATION,
        harness=_harness(),
        target_grounder_sha256=GROUNDER,
        target_executor_sha256=EXECUTOR,
    )
    assert frozen.source_harness_prediction == frozen.generic_scaffold_prediction
    assert frozen.source_harness_prediction == (
        frozen.target_written_equivalent_prediction
    )
    assert not frozen.effect_shuffled_executor_authorized
    assert not frozen.raw_neural_votes_used_as_symbolic_bindings
