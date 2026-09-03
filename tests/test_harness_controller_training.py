from motif_transfer.contracts import stable_hash
from motif_transfer.harness_controller_training import (
    anonymous_program_ir,
    build_controller_sft_examples,
    execute_controller_step,
    initial_controller_state,
    summarize_controller_sft_examples,
)
from motif_transfer.phase3_source_function_induction import (
    induce_source_function_program,
)
from motif_transfer.phase3_typed_effect_induction import (
    TYPED_EFFECTS,
    TypedCandidate,
    TypedInterventionSet,
)


def _candidate(rank, values, long_value):
    return TypedCandidate(
        candidate_rank=rank,
        effect_values=tuple(zip(TYPED_EFFECTS, values)),
        long_horizon_value=long_value,
        transition_receipt_sha256=stable_hash({"rank": rank, "values": values}),
    )


def _sets():
    rows = []
    for split in ("discovery", "qualification"):
        for index in range(8):
            rows.append(TypedInterventionSet(
                snapshot_sha256=stable_hash({"split": split, "index": index}),
                source_split=split,
                candidates=(
                    _candidate(0, (1.0, 0.0, 1.0, 0.0), 2.0),
                    _candidate(1, (0.0, 0.0, 0.0, 0.0), 0.0),
                ),
                verified_candidate_rank=0,
            ))
    return rows


def _program():
    return induce_source_function_program(
        _sets(), source_receipts_sha256="source-only",
        minimum_authentic_minus_shuffled=0.0,
    )


def test_executor_selects_anonymous_binding_and_applies_frozen_graph():
    program = _program()
    candidates = [
        {"candidate_id": "C0", "effects": dict(_sets()[0].candidates[0].effect_values)},
        {"candidate_id": "C1", "effects": dict(_sets()[0].candidates[1].effect_values)},
    ]
    selected = execute_controller_step(
        program, state=initial_controller_state(), candidates=candidates,
    )
    assert selected["decision"] == "EXECUTE_OPERATOR"
    assert selected["operator_id"] == "O0"
    assert selected["binding"] == {"active_candidate": "C0"}
    completed = execute_controller_step(
        program, state=selected["next_symbolic_state"], candidates=candidates,
        observed_effect="OBSERVED_EFFECT_HIGH",
    )
    assert completed["decision"] == "TERMINATE"
    assert completed["next_symbolic_state"]["controller_state"] == "TERMINAL"


def test_executor_abstains_on_tie_without_named_policy_template():
    program = _program()
    tied = [
        {"candidate_id": name, "effects": {effect: 0.5 for effect in TYPED_EFFECTS}}
        for name in ("C0", "C1")
    ]
    result = execute_controller_step(
        program, state=initial_controller_state(), candidates=tied,
    )
    assert result["decision"] == "ABSTAIN"
    assert result["reason"] == "TARGET_FUNCTION_ARGMAX_NOT_UNIQUE"
    assert "source" not in str(anonymous_program_ir(program)).lower()


def test_builder_is_source_only_and_hides_family_from_model_payload():
    examples = build_controller_sft_examples(
        source_family="secret_game_identity", split="train", program=_program(),
        intervention_sets=_sets()[:2],
    )
    summary = summarize_controller_sft_examples(examples)
    assert examples
    assert summary["all_valid"] is True
    assert summary["target_data_used"] is False
    assert summary["source_identity_in_prompt"] is False
    assert summary["named_policy_templates_used"] is False
    assert all(
        row.control_variant not in str(row.input_payload) for row in examples
    )
    assert {row.objective for row in examples} == {
        "SELECT_OPERATOR", "APPLY_TRANSITION",
    }
    assert {row.target_payload["decision"] for row in examples} >= {
        "EXECUTE_OPERATOR", "ABSTAIN", "TERMINATE",
    }


def test_retry_equivariance_is_executor_labeled_and_prompt_clean():
    program = _program()
    # The synthetic fixture does not naturally induce retry, so change only
    # the source-derived discovery statistic and re-hash the test artifact.
    body = dict(program)
    body.pop("program_sha256")
    body["source_function"] = dict(body["source_function"])
    body["source_function"]["retry_after_low"] = True
    body["source_function"]["maximum_trials"] = 2
    body["source_function"]["discovery_top2_recovery_gain"] = 0.125
    body["transition_graph"] = dict(body["transition_graph"])
    body["transition_graph"]["transitions"] = [
        ({**edge, "to": "RANKED_CANDIDATE_ABSENT"}
         if edge["guard"] == "OBSERVED_EFFECT_LOW" else edge)
        for edge in body["transition_graph"]["transitions"]
    ]
    program = body | {"program_sha256": stable_hash(body)}
    examples = build_controller_sft_examples(
        source_family="hidden", split="train", program=program,
        intervention_sets=_sets()[:1], augment_retry_equivariance=True,
    )
    variants = {row.control_variant for row in examples}
    assert "ALPHA_RENAMED_RETRY_LEDGER" in variants
    assert "DOMINATED_EXTENSION_RETRY_LEDGER" in variants
    augmented = [row for row in examples if row.control_variant in {
        "ALPHA_RENAMED_RETRY_LEDGER", "DOMINATED_EXTENSION_RETRY_LEDGER",
    }]
    assert all(row.target_payload["decision"] == "EXECUTE_OPERATOR" for row in augmented)
    assert all(row.control_variant not in str(row.input_payload) for row in augmented)


def test_cardinality_equivariance_is_source_only_and_executor_labeled():
    examples = build_controller_sft_examples(
        source_family="hidden_source",
        split="train",
        program=_program(),
        intervention_sets=_sets()[:1],
        augment_cardinality_equivariance=True,
        cardinality_grid=(2, 3, 4),
    )
    cardinality_rows = [
        row for row in examples
        if row.control_variant.startswith("CARDINALITY_EQUIVARIANT")
    ]
    assert cardinality_rows
    assert {len(row.input_payload["candidate_effects"]) for row in cardinality_rows} >= {
        1, 2, 3, 4,
    }
    assert all(row.target_data_used is False for row in cardinality_rows)
    assert all(row.validate() for row in cardinality_rows)

    by_variant = {row.control_variant: row for row in cardinality_rows}
    assert by_variant[
        "CARDINALITY_EQUIVARIANT_SOURCE_EXECUTION"
    ].target_payload["decision"] == "EXECUTE_OPERATOR"
    assert by_variant[
        "CARDINALITY_EQUIVARIANT_SINGLETON_BOUNDARY"
    ].target_payload["reason"] == "TARGET_CANDIDATE_SET_HAS_FEWER_THAN_TWO"
    assert by_variant[
        "CARDINALITY_EQUIVARIANT_TIED_CONTROL"
    ].target_payload["reason"] == "TARGET_FUNCTION_ARGMAX_NOT_UNIQUE"
    assert by_variant[
        "CARDINALITY_EQUIVARIANT_MISSING_SCHEMA_CONTROL"
    ].target_payload["reason"].startswith("TARGET_FUNCTION_EFFECT_SCHEMA_INVALID")


def test_missing_schema_control_can_cover_every_source_authorized_cardinality():
    examples = build_controller_sft_examples(
        source_family="hidden_source",
        split="train",
        program=_program(),
        intervention_sets=_sets()[:1],
        augment_cardinality_equivariance=True,
        augment_missing_schema_all_cardinalities=True,
        cardinality_grid=(2, 3, 4, 5),
    )
    missing = [
        row for row in examples
        if row.control_variant
        == "CARDINALITY_EQUIVARIANT_MISSING_SCHEMA_CONTROL"
    ]
    assert {len(row.input_payload["candidate_effects"]) for row in missing} == {
        2, 3, 4, 5,
    }
    assert all(
        row.target_payload["decision"] == "ABSTAIN" for row in missing
    )
