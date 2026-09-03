from __future__ import annotations

import inspect
import json
from pathlib import Path

from motif_transfer.agqa_goal_relation_transfer import (
    bind_source_goal_relation_program,
    build_harness,
    build_route,
    decide_source_candidate,
    unified_target_grounding,
)
from motif_transfer.agqa_goal_relation_evaluation import (
    freeze_transfer_predictions,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.unified_transfer_runtime import (
    PairedCalibration,
    TransferVerdict,
)
from motif_transfer.video_transfer_measurement import (
    GroundingMode,
    GroundingToolBudget,
    SharedVideoGroundingReceipt,
    assert_unified_target_uses_shared_grounding,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = json.loads((
    ROOT / "runs/sokoban_goal_relation_macro_v3/artifact.json"
).read_text())
CONFIRMATION = json.loads((
    ROOT / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
).read_text())
GROUNDER = stable_hash("agqa-v29-grounder")
EXECUTOR = stable_hash("agqa-v29-executor")
EVIDENCE = stable_hash("agqa-v29-development")
INDUCER = stable_hash("source-goal-relation-inducer")


def _execution(decision="box", votes=("box", "box", "box")):
    return {
        "decision": decision,
        "neural_votes": [
            {"view": f"view_{index}", "decision": value}
            for index, value in enumerate(votes)
        ],
    }


def _binding(**overrides):
    arguments = {
        "artifact": ARTIFACT,
        "confirmation": CONFIRMATION,
        "task_id": "future-task",
        "target_state_sha256": stable_hash("future-state"),
        "target_grounder_sha256": GROUNDER,
        "calibrated_execution": _execution(),
        "grounder_qualified": True,
    }
    arguments.update(overrides)
    return bind_source_goal_relation_program(**arguments)


def _harness(*, utility=PairedCalibration(4, 0, 146)):
    route = build_route(
        source_program_sha256=ARTIFACT["artifact_sha256"],
        target_grounder_sha256=GROUNDER,
        target_executor_sha256=EXECUTOR,
        evidence_report_sha256=EVIDENCE,
        utility_vs_target_native=utility,
        authenticity_vs_effect_shuffled=PairedCalibration(4, 0, 146),
    )
    return build_harness(
        artifact=ARTIFACT,
        confirmation=CONFIRMATION,
        inducer_artifact_sha256=INDUCER,
        route=route,
    )


def _decide(binding, harness=None):
    target = unified_target_grounding(
        artifact=ARTIFACT, confirmation=CONFIRMATION, binding=binding,
    )
    return decide_source_candidate(
        harness=harness or _harness(), target=target, binding=binding,
        target_executor_sha256=EXECUTOR,
    )


def test_unique_neural_binding_executes_source_program_through_unified_harness():
    binding = _binding()
    result = _decide(binding)
    assert binding.authorized_candidate == "box"
    assert result.source_candidate == "box"
    assert result.phase7.verdict == TransferVerdict.SELECT_SKILL
    assert result.executor_calls == 1


def test_two_of_three_majority_with_conflicting_binding_source_abstains():
    binding = _binding(
        calibrated_execution=_execution("box", ("box", "box", "floor")),
    )
    result = _decide(binding)
    assert binding.target_binding_count == 2
    assert binding.reason == "SOURCE_ABSTAIN_MULTIPLE_TARGET_BINDINGS"
    assert result.phase7.verdict == TransferVerdict.ABSTAIN
    assert result.source_candidate is None
    assert result.executor_calls == 0


def test_zero_binding_and_shuffled_source_effect_fail_closed():
    zero = _binding(calibrated_execution=_execution(None, ()))
    assert zero.reason == "SOURCE_ABSTAIN_ZERO_TARGET_BINDINGS"
    assert _decide(zero).source_candidate is None

    shuffled = _binding(effect_binding_authenticated=False)
    assert shuffled.reason == "SOURCE_EFFECT_BINDING_NOT_AUTHENTICATED"
    assert _decide(shuffled).source_candidate is None


def test_unqualified_grounder_and_current_outcome_exposure_fail_closed():
    assert _decide(_binding(grounder_qualified=False)).source_candidate is None
    exposed = _binding(formal_outcome_read=True)
    assert exposed.reason == "CURRENT_TASK_OUTCOME_EXPOSURE"
    assert _decide(exposed).source_candidate is None


def test_directional_utility_is_separate_from_row_applicability():
    binding = _binding()
    result = _decide(
        binding, harness=_harness(utility=PairedCalibration(3, 1, 116)),
    )
    assert binding.authorized_candidate == "box"
    assert result.source_candidate is None
    assert result.utility.reason == "DIRECTIONAL_UTILITY_NOT_CALIBRATED"


def test_binding_api_cannot_receive_direct_response_or_gold_answer():
    parameters = inspect.signature(bind_source_goal_relation_program).parameters
    assert "direct_response" not in parameters
    assert "gold_answer" not in parameters
    assert "formal_outcome" not in parameters


def test_prediction_composition_preserves_target_native_fallback():
    # Conflicting raw votes make the pre-ground source adapter abstain, while
    # the independent two-view target-native comparator remains available.
    row = {
        "task_id": "synthetic-conflicting-bindings",
        "runtime_receipt_sha256": stable_hash("synthetic-runtime"),
        "direct_response": "plant",
        "object_ontology_receipts": [
            {
                "decision": "clothes", "confidence": 0.8,
                "relation_observed": True, "evidence_frames": [1],
            },
            {
                "decision": "clothes", "confidence": 0.9,
                "relation_observed": True, "evidence_frames": [2],
            },
        ],
        "calibrated_target_native_execution": {
            "decision": "clothes",
            "neural_votes": [
                {"view": "isolated_relation", "decision": "blanket"},
                {"view": "ontology_0", "decision": "clothes"},
                {"view": "ontology_1", "decision": "clothes"},
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
    frozen = freeze_transfer_predictions(
        row=row,
        artifact=ARTIFACT,
        confirmation=CONFIRMATION,
        harness=_harness(),
        target_grounder_sha256=GROUNDER,
        target_executor_sha256=EXECUTOR,
    )
    assert frozen.source_candidate is None
    assert frozen.target_native_prediction == "clothes"
    assert frozen.source_harness_prediction == "clothes"
    assert frozen.effect_shuffled_prediction == "clothes"
    assert frozen.current_outcome_read is False


def test_agqa_adapter_accepts_shared_answer_blind_video_state():
    state = {
        "intervals": [{"start": 4, "end": 9}],
        "bindings": [{"subject": "person", "relation": "holding"}],
    }
    shared = SharedVideoGroundingReceipt.create(
        benchmark="agqa2", task_id="future-task", split="qualification",
        mode=GroundingMode.ORACLE_EVENT_GRAPH, state=state,
        evidence_source_sha256=stable_hash("official-agqa-scene-grounding"),
        tool_budget=GroundingToolBudget(0, 0, 0),
        official_scene_graph_read=True,
    )
    binding = _binding(target_state_sha256=shared.target_state_sha256)
    target = unified_target_grounding(
        artifact=ARTIFACT, confirmation=CONFIRMATION, binding=binding,
    )
    assert_unified_target_uses_shared_grounding(shared, target)
