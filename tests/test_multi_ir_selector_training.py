from __future__ import annotations

import json

from motif_transfer.multi_ir_selector_training import (
    ABSTAIN,
    SELECT_SKILL,
    anonymous_contract_payload,
    build_multi_ir_selector_examples,
    execute_anonymous_selection,
    format_multi_ir_selector_prompt,
    requirement_from_contract,
)
from motif_transfer.structural_ir_applicability import (
    OperatorSignature,
    SourceIRContract,
)


def _contract(label: str, *, recurrent: bool = False) -> SourceIRContract:
    return SourceIRContract.create(
        program_sha256=(label * 64)[:64],
        ir_kind=f"IR_{label}",
        operator_sequence=(OperatorSignature(
            operation="UPDATE",
            predicate_family=f"FAMILY_{label}",
            arity=1,
            value_kind="COUNT",
        ),),
        recurrent=recurrent,
        terminal_predicate_families=(),
        source_intervention_qualified=True,
        source_confirmation_sha256=((label.upper() or "Z") * 64)[:64],
    )


def test_anonymous_selector_selects_one_exact_contract() -> None:
    left = _contract("a")
    right = _contract("b", recurrent=True)
    catalog = [
        anonymous_contract_payload(left, catalog_id="P0"),
        anonymous_contract_payload(right, catalog_id="P1"),
    ]
    decision = execute_anonymous_selection(
        program_catalog=catalog,
        target_requirement=requirement_from_contract(right),
    )
    assert decision == {
        "decision": SELECT_SKILL,
        "selected_catalog_id": "P1",
        "reason": "UNIQUE_ANONYMOUS_STRUCTURAL_CONTRACT_MATCH",
    }


def test_selector_abstains_on_ambiguity_and_outcome_exposure() -> None:
    contract = _contract("c")
    row = anonymous_contract_payload(contract, catalog_id="P0")
    duplicate = dict(row, catalog_id="P1")
    requirement = requirement_from_contract(contract)
    assert execute_anonymous_selection(
        program_catalog=[row, duplicate], target_requirement=requirement,
    )["reason"] == "MULTIPLE_SOURCE_CONTRACTS_MATCH"
    requirement["formal_outcome_read"] = True
    decision = execute_anonymous_selection(
        program_catalog=[row], target_requirement=requirement,
    )
    assert decision["decision"] == ABSTAIN
    assert decision["reason"] == "NO_SOURCE_CONTRACT_MATCHES"


def test_source_only_builder_derives_controls_and_disjoint_alpha_prompts() -> None:
    contracts = (_contract("d"), _contract("e", recurrent=True))
    confirmations = {
        row.program_sha256: row.source_confirmation_sha256 for row in contracts
    }
    train = build_multi_ir_selector_examples(
        contracts=contracts, split="train", repetitions=range(0, 2),
        confirmation_by_program=confirmations,
    )
    heldout = build_multi_ir_selector_examples(
        contracts=contracts, split="source_held_out", repetitions=range(2, 4),
        confirmation_by_program=confirmations,
    )
    assert all(row.validate() for row in (*train, *heldout))
    assert {row.target_payload["decision"] for row in train} == {
        SELECT_SKILL, ABSTAIN,
    }
    assert {row.control_variant for row in train} >= {
        "AUTHENTIC_UNIQUE_CONTRACT_MATCH_N2",
        "OPERATOR_SIGNATURE_MISMATCH",
        "AMBIGUOUS_DUPLICATE_CONTRACT",
        "SOURCE_PROGRAM_UNQUALIFIED",
    }
    train_prompts = {
        format_multi_ir_selector_prompt(row.input_payload) for row in train
    }
    heldout_prompts = {
        format_multi_ir_selector_prompt(row.input_payload) for row in heldout
    }
    assert train_prompts.isdisjoint(heldout_prompts)


def test_model_prompt_contains_no_program_hash_or_domain_identity() -> None:
    contract = _contract("f")
    payload = {
        "program_catalog": [
            anonymous_contract_payload(contract, catalog_id="P0")
        ],
        "target_native_structural_requirement": requirement_from_contract(
            contract
        ),
    }
    prompt = format_multi_ir_selector_prompt(payload)
    assert contract.program_sha256 not in prompt
    assert "sokoban" not in prompt.lower()
    assert "minigrid" not in prompt.lower()
    assert "target_domain" not in prompt
    assert json.loads(prompt.split("SELECTOR_INPUT=", 1)[1].split(
        "\nOUTPUT_JSON=", 1,
    )[0]) == payload
