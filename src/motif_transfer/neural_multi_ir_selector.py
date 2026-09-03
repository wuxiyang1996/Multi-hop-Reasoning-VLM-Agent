"""Neural proposal layer for heterogeneous source-program selection.

Exact contract matching is permitted only in non-acting shadow evaluation.
Live structural mode checks schema, catalog membership, source qualification,
and the no-outcome authority boundary, but deliberately does not recompute the
IR match.  A wrong neural selection therefore remains causally visible and is
later rejected by the frozen route/utility agreement check rather than repaired
by this layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence

from .contracts import stable_hash
from .multi_ir_selector_training import (
    ABSTAIN,
    SELECT_SKILL,
    anonymous_contract_payload,
    execute_anonymous_selection,
    format_multi_ir_selector_prompt,
)
from .structural_ir_applicability import SourceIRContract, TargetIRRequirement


EXACT_SHADOW = "EXACT_SHADOW"
STRUCTURAL_ONLY = "STRUCTURAL_ONLY"


class MultiIRSelectorGenerator(Protocol):
    artifact_sha256: str

    def generate(self, prompt: str) -> str:
        """Return one JSON-only selection completion."""


@dataclass(frozen=True)
class NeuralMultiIRSelectionReceipt:
    status: str
    verification_mode: str
    shadow_only: bool
    controller_artifact_sha256: str
    selector_input_sha256: str
    prompt_sha256: str
    generated_text_sha256: str
    parsed_output: Mapping[str, Any] | None
    exact_symbolic_match: bool | None
    structural_contract_valid: bool
    source_program_authorized: bool
    selected_program_sha256: str | None
    target_outcome_read: bool
    target_action_emitted: bool
    reason: str
    receipt_sha256: str

    @classmethod
    def create(cls, **values: Any) -> "NeuralMultiIRSelectionReceipt":
        core = dict(values)
        core["receipt_sha256"] = stable_hash(core)
        return cls(**core)

    def validate(self) -> None:
        body = asdict(self)
        claimed = body.pop("receipt_sha256")
        if stable_hash(body) != claimed:
            raise ValueError("neural multi-IR receipt hash mismatch")
        if self.target_action_emitted:
            raise ValueError("multi-IR selector emitted a target action")
        if self.verification_mode not in {EXACT_SHADOW, STRUCTURAL_ONLY}:
            raise ValueError("unknown multi-IR verification mode")
        if self.shadow_only != (self.verification_mode == EXACT_SHADOW):
            raise ValueError("multi-IR shadow/mode mismatch")
        if self.source_program_authorized:
            if (
                self.shadow_only
                or not self.structural_contract_valid
                or self.selected_program_sha256 is None
                or self.target_outcome_read
            ):
                raise ValueError("unsafe source program authorization")
        elif self.selected_program_sha256 is not None:
            raise ValueError("rejected selection retained a program hash")


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strict_json(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def anonymous_requirement_payload(
    requirement: TargetIRRequirement,
) -> dict[str, Any]:
    """Remove task, domain, interface, and grounder identities from a request."""

    requirement.validate()
    return {
        "ir_kind": requirement.ir_kind,
        "operator_sequence": [asdict(row) for row in requirement.operator_sequence],
        "recurrent": requirement.recurrent,
        "terminal_predicate_families": list(
            requirement.terminal_predicate_families
        ),
        "grounder_qualified": requirement.grounder_qualified,
        "formal_outcome_read": requirement.formal_outcome_read,
    }


def _catalog(
    contracts: Sequence[SourceIRContract], requirement_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, SourceIRContract]]:
    ordered = sorted(contracts, key=lambda row: row.program_sha256)
    anonymous_core = [
        anonymous_contract_payload(row, catalog_id="PROVISIONAL")
        for row in ordered
    ]
    seed = stable_hash({
        "contracts": [
            {key: value for key, value in row.items() if key != "catalog_id"}
            for row in anonymous_core
        ],
        "requirement": dict(requirement_payload),
    })
    catalog = []
    by_alias = {}
    for index, contract in enumerate(ordered):
        alias = f"P{index}_{seed[:8]}"
        catalog.append(anonymous_contract_payload(contract, catalog_id=alias))
        by_alias[alias] = contract
    return catalog, by_alias


def structural_multi_ir_output_valid(
    *, output: Mapping[str, Any] | None,
    catalog_by_alias: Mapping[str, SourceIRContract],
    requirement: TargetIRRequirement,
) -> bool:
    """Check authority invariants without recomputing structural matching."""

    if output is None or set(output) != {
        "decision", "selected_catalog_id", "reason",
    }:
        return False
    if not isinstance(output["reason"], str) or not output["reason"]:
        return False
    decision = str(output["decision"])
    selected = output["selected_catalog_id"]
    if decision == ABSTAIN:
        return selected is None
    if decision != SELECT_SKILL or not isinstance(selected, str):
        return False
    contract = catalog_by_alias.get(selected)
    return bool(
        contract is not None
        and contract.source_intervention_qualified
        and requirement.grounder_qualified
        and not requirement.formal_outcome_read
    )


class NeuralMultiIRSelector:
    """Generate one program choice with separated shadow/live verification."""

    def __init__(
        self, generator: MultiIRSelectorGenerator, *, verification_mode: str,
    ):
        artifact = str(getattr(generator, "artifact_sha256", ""))
        if len(artifact) != 64:
            raise ValueError("selector generator requires a SHA-256 identity")
        if verification_mode not in {EXACT_SHADOW, STRUCTURAL_ONLY}:
            raise ValueError("invalid multi-IR selector verification mode")
        self.generator = generator
        self.artifact_sha256 = artifact
        self.verification_mode = verification_mode

    def decide(
        self, *, contracts: Sequence[SourceIRContract],
        requirement: TargetIRRequirement,
    ) -> NeuralMultiIRSelectionReceipt:
        requirement_payload = anonymous_requirement_payload(requirement)
        catalog, by_alias = _catalog(contracts, requirement_payload)
        input_payload = {
            "program_catalog": catalog,
            "target_native_structural_requirement": requirement_payload,
        }
        prompt = format_multi_ir_selector_prompt(input_payload)
        inference_error = None
        try:
            generated = self.generator.generate(prompt)
            if not isinstance(generated, str):
                raise TypeError("selector generator returned a non-string value")
        except Exception as error:  # fail closed at the neural boundary
            generated = ""
            inference_error = type(error).__name__
        parsed = _strict_json(generated)
        structural_valid = structural_multi_ir_output_valid(
            output=parsed, catalog_by_alias=by_alias, requirement=requirement,
        )
        exact = None
        if self.verification_mode == EXACT_SHADOW:
            exact = parsed == execute_anonymous_selection(
                program_catalog=catalog,
                target_requirement=requirement_payload,
            )
        authorized = bool(
            self.verification_mode == STRUCTURAL_ONLY
            and structural_valid
            and parsed is not None
            and parsed.get("decision") == SELECT_SKILL
        )
        selected_hash = None
        if authorized:
            selected_hash = by_alias[str(parsed["selected_catalog_id"])].program_sha256
        if inference_error:
            status = "NEURAL_MULTI_IR_INFERENCE_FAILED"
            reason = f"INFERENCE_ERROR:{inference_error}"
        elif parsed is None:
            status = "NEURAL_MULTI_IR_OUTPUT_REJECTED"
            reason = "INVALID_JSON_OBJECT"
        elif not structural_valid:
            status = "NEURAL_MULTI_IR_OUTPUT_REJECTED"
            reason = "STRUCTURAL_AUTHORITY_MISMATCH"
        elif self.verification_mode == EXACT_SHADOW and not exact:
            status = "NEURAL_MULTI_IR_OUTPUT_REJECTED"
            reason = "EXACT_SHADOW_MISMATCH"
        else:
            status = (
                "NEURAL_MULTI_IR_EXACT_SHADOW_VERIFIED"
                if self.verification_mode == EXACT_SHADOW
                else "NEURAL_MULTI_IR_STRUCTURALLY_VERIFIED"
            )
            reason = (
                "EXACT_FROZEN_SELECTOR_MATCH"
                if self.verification_mode == EXACT_SHADOW
                else "SCHEMA_CATALOG_AND_AUTHORITY_VALID"
            )
        receipt = NeuralMultiIRSelectionReceipt.create(
            status=status,
            verification_mode=self.verification_mode,
            shadow_only=self.verification_mode == EXACT_SHADOW,
            controller_artifact_sha256=self.artifact_sha256,
            selector_input_sha256=stable_hash(input_payload),
            prompt_sha256=_sha_text(prompt),
            generated_text_sha256=_sha_text(generated),
            parsed_output=parsed,
            exact_symbolic_match=exact,
            structural_contract_valid=structural_valid,
            source_program_authorized=authorized,
            selected_program_sha256=selected_hash,
            target_outcome_read=requirement.formal_outcome_read,
            target_action_emitted=False,
            reason=reason,
        )
        receipt.validate()
        return receipt


__all__ = [
    "EXACT_SHADOW", "MultiIRSelectorGenerator", "NeuralMultiIRSelectionReceipt",
    "NeuralMultiIRSelector", "STRUCTURAL_ONLY", "anonymous_requirement_payload",
    "structural_multi_ir_output_valid",
]
