"""Source-only supervision for neural selection of heterogeneous symbolic IRs.

The Phase-8 harness does not choose a target-native action.  It chooses one
source-induced anonymous program when a qualified target-native grounder asks
for the same structural contract, or abstains.  This module distils that
operation without putting game identities, target domains, program hashes,
native actions, or target outcomes in the model prompt.

All labels are produced by :func:`execute_anonymous_selection`.  The builder
may alpha-rename and permute catalog entries and may create mismatched controls,
but it cannot hand-label a source/target route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .structural_ir_applicability import OperatorSignature, SourceIRContract


SELECT_SKILL = "SELECT_SKILL"
ABSTAIN = "ABSTAIN"
OBJECTIVE = "SELECT_TRANSFER_PROGRAM"

MULTI_IR_SELECTOR_SYSTEM_PROMPT = (
    "You are the neural selector of anonymous source-induced symbolic programs. "
    "Use only the supplied structural program catalog and target-native symbolic "
    "requirement. Select exactly one catalog entry only when it is qualified and "
    "its IR kind, ordered operator signatures, recurrence, and terminal predicate "
    "families all match. Otherwise abstain. Never infer a domain identity, native "
    "action, hidden outcome, or program hash. Return the exact JSON object only; "
    "do not provide reasoning."
)


def format_multi_ir_selector_prompt(input_payload: Mapping[str, Any]) -> str:
    """Serialize the prompt contract shared by source-only SFT and inference."""

    return (
        MULTI_IR_SELECTOR_SYSTEM_PROMPT
        + "\n\nOBJECTIVE=" + OBJECTIVE
        + "\nSELECTOR_INPUT="
        + json.dumps(
            dict(input_payload), sort_keys=True, ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\nOUTPUT_JSON="
    )


def _operator_payload(value: OperatorSignature) -> dict[str, Any]:
    return asdict(value)


def anonymous_contract_payload(
    contract: SourceIRContract, *, catalog_id: str,
) -> dict[str, Any]:
    """Remove content addresses and retain only the executable type contract."""

    contract.validate()
    return {
        "catalog_id": str(catalog_id),
        "ir_kind": contract.ir_kind,
        "operator_sequence": [
            _operator_payload(row) for row in contract.operator_sequence
        ],
        "recurrent": contract.recurrent,
        "terminal_predicate_families": list(
            contract.terminal_predicate_families
        ),
        "source_intervention_qualified": (
            contract.source_intervention_qualified
        ),
    }


def requirement_from_contract(contract: SourceIRContract) -> dict[str, Any]:
    """Create a source-authorized structural probe, not a target example."""

    contract.validate()
    return {
        "ir_kind": contract.ir_kind,
        "operator_sequence": [
            _operator_payload(row) for row in contract.operator_sequence
        ],
        "recurrent": contract.recurrent,
        "terminal_predicate_families": list(
            contract.terminal_predicate_families
        ),
        "grounder_qualified": True,
        "formal_outcome_read": False,
    }


def _contract_matches_payload(
    source: Mapping[str, Any], requirement: Mapping[str, Any],
) -> bool:
    return bool(
        requirement.get("formal_outcome_read") is False
        and requirement.get("grounder_qualified") is True
        and source.get("source_intervention_qualified") is True
        and source.get("ir_kind") == requirement.get("ir_kind")
        and source.get("operator_sequence")
        == requirement.get("operator_sequence")
        and source.get("recurrent") == requirement.get("recurrent")
        and source.get("terminal_predicate_families")
        == requirement.get("terminal_predicate_families")
    )


def execute_anonymous_selection(
    *, program_catalog: Sequence[Mapping[str, Any]],
    target_requirement: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one fail-closed selection without source or target identities."""

    ids = [str(row.get("catalog_id")) for row in program_catalog]
    if (
        not ids or len(ids) != len(set(ids))
        or any(value in {"", "None"} for value in ids)
    ):
        return {
            "decision": ABSTAIN,
            "selected_catalog_id": None,
            "reason": "INVALID_OR_DUPLICATE_CATALOG_ID",
        }
    matches = [
        str(row["catalog_id"])
        for row in program_catalog
        if _contract_matches_payload(row, target_requirement)
    ]
    if len(matches) == 1:
        return {
            "decision": SELECT_SKILL,
            "selected_catalog_id": matches[0],
            "reason": "UNIQUE_ANONYMOUS_STRUCTURAL_CONTRACT_MATCH",
        }
    return {
        "decision": ABSTAIN,
        "selected_catalog_id": None,
        "reason": (
            "MULTIPLE_SOURCE_CONTRACTS_MATCH"
            if matches else "NO_SOURCE_CONTRACT_MATCHES"
        ),
    }


@dataclass(frozen=True)
class MultiIRSelectorExample:
    """One source-only selector example with audit-only provenance."""

    example_id: str
    source_program_sha256: str
    split: str
    control_variant: str
    input_payload: Mapping[str, Any]
    target_payload: Mapping[str, Any]
    evidence_receipt_ids: tuple[str, ...]
    derivation: str
    target_data_used: bool = False

    def validate(self) -> bool:
        body = asdict(self)
        claimed = body.pop("example_id")
        serialized = json.dumps(
            [self.input_payload, self.target_payload], sort_keys=True,
        )
        return bool(
            claimed == stable_hash(body)
            and self.split in {"train", "validation", "source_held_out"}
            and self.control_variant
            and self.evidence_receipt_ids
            and self.target_data_used is False
            and self.source_program_sha256 not in serialized
            and "target_domain" not in serialized
            and "native_action" not in serialized
            and "EXPLORE_UNTRIED" not in serialized
            and "BACKTRACK_REPLAN" not in serialized
            and "COMMIT_VERIFY" not in serialized
        )


def _make_example(
    *, contract: SourceIRContract, split: str, control_variant: str,
    catalog: Sequence[Mapping[str, Any]], requirement: Mapping[str, Any],
    evidence_receipt_ids: Sequence[str],
) -> MultiIRSelectorExample:
    input_payload = {
        "program_catalog": [dict(row) for row in catalog],
        "target_native_structural_requirement": dict(requirement),
    }
    target_payload = execute_anonymous_selection(
        program_catalog=catalog, target_requirement=requirement,
    )
    body = {
        "source_program_sha256": contract.program_sha256,
        "split": split,
        "control_variant": control_variant,
        "input_payload": input_payload,
        "target_payload": target_payload,
        "evidence_receipt_ids": tuple(map(str, evidence_receipt_ids)),
        "derivation": "FROZEN_SOURCE_CONTRACT_EXECUTION",
        "target_data_used": False,
    }
    return MultiIRSelectorExample(example_id=stable_hash(body), **body)


def _permutation(size: int, seed: str) -> tuple[int, ...]:
    return tuple(sorted(
        range(size), key=lambda index: stable_hash({"seed": seed, "i": index}),
    ))


def _renamed_catalog(
    contracts: Sequence[SourceIRContract], *, seed: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    order = _permutation(len(contracts), seed)
    catalog = []
    program_to_id = {}
    for alias_index, contract_index in enumerate(order):
        contract = contracts[contract_index]
        # Opaque per-probe aliases prevent a model from memorising a fixed
        # catalog position and force it to copy the selected identifier.
        alias = f"P{alias_index}_{seed[:8]}"
        catalog.append(anonymous_contract_payload(contract, catalog_id=alias))
        program_to_id[contract.program_sha256] = alias
    return catalog, program_to_id


def _mutated_requirement(
    requirement: Mapping[str, Any], variant: str,
) -> dict[str, Any]:
    output = json.loads(json.dumps(requirement))
    if variant == "IR_KIND_MISMATCH":
        output["ir_kind"] = "PERTURBED_IR_KIND"
    elif variant == "OPERATOR_SIGNATURE_MISMATCH":
        operators = output["operator_sequence"]
        if not operators:
            raise ValueError("cannot perturb an empty operator sequence")
        operators[0]["arity"] = int(operators[0]["arity"]) + 1
    elif variant == "RECURRENCE_MISMATCH":
        output["recurrent"] = not bool(output["recurrent"])
    elif variant == "TERMINAL_PREDICATE_MISMATCH":
        output["terminal_predicate_families"] = [
            *output["terminal_predicate_families"], "PERTURBED_PREDICATE",
        ]
    elif variant == "TARGET_GROUNDER_UNQUALIFIED":
        output["grounder_qualified"] = False
    elif variant == "CURRENT_TARGET_OUTCOME_EXPOSED":
        output["formal_outcome_read"] = True
    else:
        raise ValueError(f"unsupported selector control: {variant}")
    return output


CONTROL_VARIANTS = (
    "IR_KIND_MISMATCH",
    "OPERATOR_SIGNATURE_MISMATCH",
    "RECURRENCE_MISMATCH",
    "TERMINAL_PREDICATE_MISMATCH",
    "TARGET_GROUNDER_UNQUALIFIED",
    "CURRENT_TARGET_OUTCOME_EXPOSED",
)


def build_multi_ir_selector_examples(
    *, contracts: Sequence[SourceIRContract], split: str,
    repetitions: range, confirmation_by_program: Mapping[str, str],
) -> tuple[MultiIRSelectorExample, ...]:
    """Generate labels from frozen source contracts and deterministic controls."""

    if len(contracts) < 2:
        raise ValueError("multi-IR supervision requires at least two contracts")
    if len({row.program_sha256 for row in contracts}) != len(contracts):
        raise ValueError("source contract catalog contains duplicate programs")
    for contract in contracts:
        contract.validate()
        if not contract.source_intervention_qualified:
            raise ValueError("unqualified source contract entered SFT builder")
        if contract.program_sha256 not in confirmation_by_program:
            raise ValueError("source contract omitted confirmation provenance")

    output = []
    for repetition in repetitions:
        for contract in contracts:
            seed = stable_hash({
                "split": split,
                "repetition": repetition,
                "program": contract.program_sha256,
            })
            catalog, aliases = _renamed_catalog(contracts, seed=seed)
            requirement = requirement_from_contract(contract)
            receipts = (
                contract.contract_sha256,
                confirmation_by_program[contract.program_sha256],
            )
            matched_alias = aliases[contract.program_sha256]
            matched_row = next(
                row for row in catalog if row["catalog_id"] == matched_alias
            )
            distractors = [
                row for row in catalog if row["catalog_id"] != matched_alias
            ]
            # Vary catalog cardinality using only other frozen source contracts.
            # This balances positive/abstention supervision without duplicating
            # a prompt or inventing a target-derived positive route.
            for catalog_size in range(2, len(catalog) + 1):
                selected_rows = [matched_row, *distractors[:catalog_size - 1]]
                selected_rows = [
                    dict(selected_rows[index])
                    for index in _permutation(
                        len(selected_rows), f"{seed}:positive:{catalog_size}",
                    )
                ]
                output.append(_make_example(
                    contract=contract, split=split,
                    control_variant=(
                        "AUTHENTIC_UNIQUE_CONTRACT_MATCH_N"
                        f"{catalog_size}"
                    ),
                    catalog=selected_rows, requirement=requirement,
                    evidence_receipt_ids=receipts,
                ))
            for variant in CONTROL_VARIANTS:
                output.append(_make_example(
                    contract=contract, split=split, control_variant=variant,
                    catalog=catalog,
                    requirement=_mutated_requirement(requirement, variant),
                    evidence_receipt_ids=receipts,
                ))

            duplicate = dict(next(
                row for row in catalog
                if row["catalog_id"] == aliases[contract.program_sha256]
            ))
            duplicate["catalog_id"] = f"P{len(catalog)}_{seed[:8]}"
            output.append(_make_example(
                contract=contract, split=split,
                control_variant="AMBIGUOUS_DUPLICATE_CONTRACT",
                catalog=[*catalog, duplicate], requirement=requirement,
                evidence_receipt_ids=receipts,
            ))

            unqualified = []
            for row in catalog:
                candidate = dict(row)
                if candidate["catalog_id"] == aliases[contract.program_sha256]:
                    candidate["source_intervention_qualified"] = False
                unqualified.append(candidate)
            output.append(_make_example(
                contract=contract, split=split,
                control_variant="SOURCE_PROGRAM_UNQUALIFIED",
                catalog=unqualified, requirement=requirement,
                evidence_receipt_ids=receipts,
            ))

    if not output or not all(row.validate() for row in output):
        raise ValueError("invalid source-only multi-IR selector examples")
    return tuple(output)


__all__ = [
    "ABSTAIN", "CONTROL_VARIANTS", "MULTI_IR_SELECTOR_SYSTEM_PROMPT",
    "MultiIRSelectorExample", "OBJECTIVE", "SELECT_SKILL",
    "anonymous_contract_payload", "build_multi_ir_selector_examples",
    "execute_anonymous_selection", "format_multi_ir_selector_prompt",
    "requirement_from_contract",
]
