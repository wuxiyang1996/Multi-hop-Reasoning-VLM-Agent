from motif_transfer.contracts import stable_hash
from motif_transfer.phase3_source_function_induction import induce_source_function_program
from motif_transfer.phase3_typed_effect_induction import (
    TYPED_EFFECTS,
    TypedCandidate,
    TypedInterventionSet,
)
from motif_transfer.target_harness_sft import (
    GroundedTargetState,
    build_matched_target_pair,
)
from scripts.build_target_harness_sft_five_domain_v1 import (
    _agqa_consensus_effects,
    _agqa_view_effects,
    _split_for_video,
)


def _candidate(rank, values, value):
    return TypedCandidate(
        candidate_rank=rank,
        effect_values=tuple(zip(TYPED_EFFECTS, values)),
        long_horizon_value=value,
        transition_receipt_sha256=stable_hash([rank, values]),
    )


def _program():
    sets = []
    for split in ("discovery", "qualification"):
        for index in range(8):
            sets.append(TypedInterventionSet(
                snapshot_sha256=stable_hash([split, index]),
                source_split=split,
                candidates=(
                    _candidate(0, (1.0, 0.0, 1.0, 0.0), 2.0),
                    _candidate(1, (0.0, 0.0, 0.0, 0.0), 0.0),
                ),
                verified_candidate_rank=0,
            ))
    return induce_source_function_program(
        sets, source_receipts_sha256="source", minimum_authentic_minus_shuffled=0.0,
    )


def test_target_pair_is_executor_labelled_and_prompt_clean():
    state = GroundedTargetState(
        target_domain="secret_target",
        target_task_sha256=stable_hash("task"),
        split="train",
        state_receipt_sha256=stable_hash("state"),
        grounder_artifact_sha256=stable_hash("grounder"),
        candidates=(
            {"candidate_id": "C0", "effects": dict(zip(
                TYPED_EFFECTS, (0.9, 0.1, 0.8, 0.2),
            ))},
            {"candidate_id": "C1", "effects": dict(zip(
                TYPED_EFFECTS, (0.1, 0.2, 0.1, 0.1),
            ))},
        ),
    )
    pair = build_matched_target_pair(
        state=state, source_family="secret_source", program=_program(),
        program_receipt=stable_hash("program"),
    )
    assert pair is not None
    authentic, control = pair
    assert authentic.target_payload["decision"] == "EXECUTE_OPERATOR"
    assert authentic.target_payload != control.target_payload
    assert authentic.pair_id == control.pair_id
    assert authentic.validate() and control.validate()
    model_text = str(authentic.input_payload) + str(authentic.target_payload)
    assert "secret_target" not in model_text
    assert "secret_source" not in model_text


def test_grounded_target_state_rejects_incomplete_effect_schema():
    state = GroundedTargetState(
        target_domain="x", target_task_sha256="task", split="validation",
        state_receipt_sha256="receipt", grounder_artifact_sha256="grounder",
        candidates=(
            {"candidate_id": "C0", "effects": {TYPED_EFFECTS[0]: 1.0}},
            {"candidate_id": "C1", "effects": {TYPED_EFFECTS[0]: 0.0}},
        ),
    )
    assert state.validate() is False


def _video_adapter_config():
    return {
        "expected_sampled_frames": 48,
        "evidence_frame_normalizer": 8,
        "maximum_consensus_endpoint_spread": 8,
        "coverage_scores": {"SUFFICIENT": 1.0, "PARTIAL": 0.5, "MISSING": 0.0},
    }


def test_video_split_uses_only_video_identity():
    rule = {
        "kind": "STABLE_VIDEO_ID_HASH_MODULO",
        "modulus": 4,
        "validation_bucket": 0,
        "identity_only_no_outcome_selection": True,
    }
    assert _split_for_video("same-video", rule) == _split_for_video(
        "same-video", rule,
    )
    assert _split_for_video("same-video", rule) in {"train", "validation"}


def test_agqa_view_adapter_uses_only_grounding_evidence():
    attempt = {
        "payload": {
            "coverage": "SUFFICIENT",
            "observations": [
                {
                    "observability": "OBSERVED", "confidence": 0.8,
                    "start_frame": 1, "end_frame": 4,
                    "evidence_frames": [1, 4],
                },
                {
                    "observability": "OCCLUDED", "confidence": 1.0,
                    "start_frame": 8, "end_frame": 12,
                    "evidence_frames": [8],
                },
            ],
        },
        "usage": {"response_sha256": stable_hash("neural-response")},
    }
    effects, receipt = _agqa_view_effects(
        attempt, _video_adapter_config(), precision=6,
    )
    assert len(receipt) == 64
    assert effects == {
        TYPED_EFFECTS[0]: 0.8,
        TYPED_EFFECTS[1]: 0.4,
        TYPED_EFFECTS[2]: 0.083333,
        TYPED_EFFECTS[3]: 0.625,
    }


def test_agqa_consensus_adapter_is_fail_closed_on_spread():
    views = [
        dict(zip(TYPED_EFFECTS, (0.8, 0.4, 0.1, 0.6))),
        dict(zip(TYPED_EFFECTS, (0.6, 0.3, 0.2, 0.5))),
    ]
    receipts = [{
        "authorized": True,
        "consensus_interval": [4, 11],
        "maximum_endpoint_spread": 8,
        "receipt_sha256": stable_hash("consensus"),
    }]
    effects, receipt = _agqa_consensus_effects(
        receipts, views, _video_adapter_config(), precision=6,
    )
    assert len(receipt) == 64
    assert effects[TYPED_EFFECTS[0]] == 0.7
    assert effects[TYPED_EFFECTS[1]] == 1.0
    assert effects[TYPED_EFFECTS[2]] == 0.166667
    assert effects[TYPED_EFFECTS[3]] == 0.0
