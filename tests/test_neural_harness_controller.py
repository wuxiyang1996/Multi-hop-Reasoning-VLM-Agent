import json

from motif_transfer.contracts import stable_hash
from motif_transfer.harness_controller_training import (
    execute_controller_step,
    format_controller_prompt,
    initial_controller_state,
)
from motif_transfer.neural_harness_controller import (
    EXACT_SHADOW,
    STRUCTURAL_ONLY,
    NeuralHarnessController,
)
from motif_transfer.phase3_source_function_induction import induce_source_function_program
from motif_transfer.phase3_typed_effect_induction import (
    TYPED_EFFECTS,
    TypedCandidate,
    TypedInterventionSet,
)


class StaticGenerator:
    artifact_sha256 = stable_hash("static-controller")

    def __init__(self, output):
        self.output = output

    def generate(self, prompt):
        assert prompt.endswith("\nOUTPUT_JSON=")
        return self.output


def _candidate(rank, values, value):
    return TypedCandidate(
        candidate_rank=rank,
        effect_values=tuple(zip(TYPED_EFFECTS, values)),
        long_horizon_value=value,
        transition_receipt_sha256=stable_hash([rank, values]),
    )


def _program():
    rows = []
    for split in ("discovery", "qualification"):
        for index in range(8):
            rows.append(TypedInterventionSet(
                snapshot_sha256=stable_hash([split, index]),
                source_split=split,
                candidates=(
                    _candidate(0, (1.0, 0.0, 1.0, 0.0), 2.0),
                    _candidate(1, (0.0, 0.0, 0.0, 0.0), 0.0),
                ),
                verified_candidate_rank=0,
            ))
    return induce_source_function_program(
        rows, source_receipts_sha256="source",
        minimum_authentic_minus_shuffled=0.0,
    )


def _candidates():
    return [
        {"candidate_id": "C0", "effects": dict(zip(
            TYPED_EFFECTS, (1.0, 0.0, 1.0, 0.0),
        ))},
        {"candidate_id": "C1", "effects": dict(zip(
            TYPED_EFFECTS, (0.0, 0.0, 0.0, 0.0),
        ))},
    ]


def test_prompt_formatter_is_canonical_and_compact():
    payload = {"z": 1, "a": {"x": 2}}
    prompt = format_controller_prompt(
        objective="SELECT_OPERATOR", input_payload=payload,
    )
    assert "CONTROLLER_INPUT={\"a\":{\"x\":2},\"z\":1}" in prompt
    assert prompt.endswith("\nOUTPUT_JSON=")


def test_structurally_valid_neural_proposal_receives_source_operator_authority():
    program = _program()
    state = initial_controller_state()
    expected = execute_controller_step(
        program, state=state, candidates=_candidates(),
    )
    controller = NeuralHarnessController(
        StaticGenerator(json.dumps(expected)), verification_mode=STRUCTURAL_ONLY,
    )
    receipt = controller.decide(
        program=program, state=state, candidates=_candidates(),
    )
    assert receipt.status == "NEURAL_CONTROLLER_STRUCTURALLY_VERIFIED"
    assert receipt.exact_symbolic_match is None
    assert receipt.structural_contract_valid is True
    assert receipt.source_operator_authorized is True
    assert receipt.authorized_binding == {"active_candidate": "C0"}
    assert receipt.target_action_emitted is False
    receipt.validate()


def test_wrong_neural_binding_is_rejected_without_python_repair():
    program = _program()
    state = initial_controller_state()
    wrong = execute_controller_step(
        program, state=state, candidates=_candidates(),
    )
    wrong["binding"] = {"active_candidate": "C1"}
    wrong["next_symbolic_state"] = {
        "controller_state": "FUNCTION_CANDIDATE_ACTIVE",
        "active_candidate": "C1",
        "attempted_candidates": ["C1"],
        "trials_used": 1,
    }
    controller = NeuralHarnessController(
        StaticGenerator(json.dumps(wrong)), verification_mode=EXACT_SHADOW,
    )
    receipt = controller.decide(
        program=program, state=state, candidates=_candidates(),
    )
    assert receipt.status == "NEURAL_CONTROLLER_OUTPUT_REJECTED"
    assert receipt.reason == "SYMBOLIC_VERIFIER_MISMATCH"
    assert receipt.exact_symbolic_match is False
    assert receipt.source_operator_authorized is False
    assert receipt.authorized_operator_id is None
    assert receipt.authorized_binding is None


def test_correct_symbolic_abstention_is_verified_but_never_authorized_as_operator():
    program = _program()
    state = initial_controller_state()
    tied = [
        {"candidate_id": name, "effects": {effect: 0.5 for effect in TYPED_EFFECTS}}
        for name in ("C0", "C1")
    ]
    expected = execute_controller_step(program, state=state, candidates=tied)
    controller = NeuralHarnessController(
        StaticGenerator(json.dumps(expected)), verification_mode=EXACT_SHADOW,
    )
    receipt = controller.decide(program=program, state=state, candidates=tied)
    assert receipt.status == "NEURAL_CONTROLLER_EXACT_SHADOW_VERIFIED"
    assert receipt.exact_symbolic_match is True
    assert receipt.source_operator_authorized is False
    assert receipt.parsed_output["decision"] == "ABSTAIN"


def test_invalid_model_output_fails_closed():
    controller = NeuralHarnessController(
        StaticGenerator("not-json"), verification_mode=STRUCTURAL_ONLY,
    )
    receipt = controller.decide(
        program=_program(), state=initial_controller_state(),
        candidates=_candidates(),
    )
    assert receipt.status == "NEURAL_CONTROLLER_OUTPUT_REJECTED"
    assert receipt.reason == "INVALID_JSON_OBJECT"
    assert receipt.source_operator_authorized is False
    assert receipt.target_action_emitted is False


def test_live_structural_mode_does_not_recompute_candidate_argmax():
    program = _program()
    state = initial_controller_state()
    legal_but_suboptimal = execute_controller_step(
        program, state=state, candidates=_candidates(),
    )
    legal_but_suboptimal["binding"] = {"active_candidate": "C1"}
    legal_but_suboptimal["next_symbolic_state"] = {
        "controller_state": "FUNCTION_CANDIDATE_ACTIVE",
        "active_candidate": "C1",
        "attempted_candidates": ["C1"],
        "trials_used": 1,
    }
    controller = NeuralHarnessController(
        StaticGenerator(json.dumps(legal_but_suboptimal)),
        verification_mode=STRUCTURAL_ONLY,
    )
    receipt = controller.decide(
        program=program, state=state, candidates=_candidates(),
    )
    assert receipt.status == "NEURAL_CONTROLLER_STRUCTURALLY_VERIFIED"
    assert receipt.source_operator_authorized is True
    assert receipt.authorized_binding == {"active_candidate": "C1"}
    assert receipt.exact_symbolic_match is None
