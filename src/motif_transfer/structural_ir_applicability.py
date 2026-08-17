"""Outcome-blind type checking between source-induced and target-native IR.

This module deliberately does not retrieve a source by game name.  It reduces
each validated source artifact to an anonymous structural contract and asks
whether a target-native grounder requests exactly that contract.  Target
actions and target outcomes are outside the interface.

The contract is intentionally small: an ordered operator signature, whether
the operator graph is recurrent, and terminal predicate families.  This is
enough to distinguish the two source-induced mechanisms currently supported
by formal target evidence (a finite entity-slot add/remove program and a
recurrent relational-control program) from temporal value functions learned
from the additional arcade sources.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .phase3_source_function_induction import validate_source_function_program
from .source_goal_acquisition_induction import (
    validate_goal_acquisition_program,
)


@dataclass(frozen=True)
class OperatorSignature:
    operation: str
    predicate_family: str
    arity: int
    value_kind: str


@dataclass(frozen=True)
class SourceIRContract:
    program_sha256: str
    ir_kind: str
    operator_sequence: tuple[OperatorSignature, ...]
    recurrent: bool
    terminal_predicate_families: tuple[str, ...]
    source_intervention_qualified: bool
    source_confirmation_sha256: str
    contract_sha256: str

    @classmethod
    def create(
        cls, *, program_sha256: str, ir_kind: str,
        operator_sequence: Sequence[OperatorSignature], recurrent: bool,
        terminal_predicate_families: Sequence[str],
        source_intervention_qualified: bool,
        source_confirmation_sha256: str,
    ) -> "SourceIRContract":
        core = {
            "program_sha256": str(program_sha256),
            "ir_kind": str(ir_kind),
            "operator_sequence": [asdict(row) for row in operator_sequence],
            "recurrent": bool(recurrent),
            "terminal_predicate_families": sorted(map(
                str, terminal_predicate_families,
            )),
            "source_intervention_qualified": bool(
                source_intervention_qualified
            ),
            "source_confirmation_sha256": str(source_confirmation_sha256),
        }
        return cls(
            program_sha256=core["program_sha256"],
            ir_kind=core["ir_kind"],
            operator_sequence=tuple(operator_sequence),
            recurrent=core["recurrent"],
            terminal_predicate_families=tuple(
                core["terminal_predicate_families"]
            ),
            source_intervention_qualified=core[
                "source_intervention_qualified"
            ],
            source_confirmation_sha256=core["source_confirmation_sha256"],
            contract_sha256=stable_hash(core),
        )

    def validate(self) -> None:
        core = {
            "program_sha256": self.program_sha256,
            "ir_kind": self.ir_kind,
            "operator_sequence": [asdict(row) for row in self.operator_sequence],
            "recurrent": self.recurrent,
            "terminal_predicate_families": list(
                self.terminal_predicate_families
            ),
            "source_intervention_qualified": self.source_intervention_qualified,
            "source_confirmation_sha256": self.source_confirmation_sha256,
        }
        if stable_hash(core) != self.contract_sha256:
            raise ValueError("source IR contract hash mismatch")


@dataclass(frozen=True)
class TargetIRRequirement:
    task_id: str
    target_domain: str
    target_interface: str
    target_grounder_sha256: str
    ir_kind: str
    operator_sequence: tuple[OperatorSignature, ...]
    recurrent: bool
    terminal_predicate_families: tuple[str, ...]
    grounder_qualified: bool
    formal_outcome_read: bool
    requirement_sha256: str

    @classmethod
    def create(
        cls, *, task_id: str, target_domain: str, target_interface: str,
        target_grounder_sha256: str, ir_kind: str,
        operator_sequence: Sequence[OperatorSignature], recurrent: bool,
        terminal_predicate_families: Sequence[str],
        grounder_qualified: bool, formal_outcome_read: bool = False,
    ) -> "TargetIRRequirement":
        core = {
            "task_id": str(task_id),
            "target_domain": str(target_domain),
            "target_interface": str(target_interface),
            "target_grounder_sha256": str(target_grounder_sha256),
            "ir_kind": str(ir_kind),
            "operator_sequence": [asdict(row) for row in operator_sequence],
            "recurrent": bool(recurrent),
            "terminal_predicate_families": sorted(map(
                str, terminal_predicate_families,
            )),
            "grounder_qualified": bool(grounder_qualified),
            "formal_outcome_read": bool(formal_outcome_read),
        }
        return cls(
            task_id=core["task_id"],
            target_domain=core["target_domain"],
            target_interface=core["target_interface"],
            target_grounder_sha256=core["target_grounder_sha256"],
            ir_kind=core["ir_kind"],
            operator_sequence=tuple(operator_sequence),
            recurrent=core["recurrent"],
            terminal_predicate_families=tuple(
                core["terminal_predicate_families"]
            ),
            grounder_qualified=core["grounder_qualified"],
            formal_outcome_read=core["formal_outcome_read"],
            requirement_sha256=stable_hash(core),
        )

    def validate(self) -> None:
        core = {
            "task_id": self.task_id,
            "target_domain": self.target_domain,
            "target_interface": self.target_interface,
            "target_grounder_sha256": self.target_grounder_sha256,
            "ir_kind": self.ir_kind,
            "operator_sequence": [asdict(row) for row in self.operator_sequence],
            "recurrent": self.recurrent,
            "terminal_predicate_families": list(
                self.terminal_predicate_families
            ),
            "grounder_qualified": self.grounder_qualified,
            "formal_outcome_read": self.formal_outcome_read,
        }
        if stable_hash(core) != self.requirement_sha256:
            raise ValueError("target IR requirement hash mismatch")


def _artifact_hash(artifact: Mapping[str, Any]) -> None:
    body = dict(artifact)
    claimed = body.pop("artifact_sha256", None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError("source artifact hash mismatch")


def _signature(value: Mapping[str, Any]) -> OperatorSignature:
    return OperatorSignature(
        operation=str(value["operation"]),
        predicate_family=str(value["predicate_family"]),
        arity=int(value["arity"]),
        value_kind=str(value["value_kind"]),
    )


def structural_program_contract(
    program: Mapping[str, Any], *, source_confirmation_sha256: str,
    source_intervention_qualified: bool,
) -> SourceIRContract:
    """Extract the anonymous type contract from a MiniGrid-style program."""

    body = dict(program)
    claimed = body.pop("program_sha256", None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError("source structural program hash mismatch")
    if program.get("schema_version") != "source-induced-structural-program-v1":
        raise ValueError("unsupported source structural program schema")
    operators = tuple(
        _signature(row["operator_type_descriptor"])
        for row in program.get("operators") or ()
    )
    qualified = (
        source_intervention_qualified
        and program.get("status") == "SOURCE_STRUCTURAL_PROGRAM_QUALIFIED"
        and program.get("target_data_read") is False
        and program.get("source_action_identity_exported") is False
    )
    return SourceIRContract.create(
        program_sha256=str(program["program_sha256"]),
        ir_kind="FINITE_STRUCTURAL_DELTA_SEQUENCE",
        operator_sequence=operators,
        recurrent=False,
        terminal_predicate_families=(),
        source_intervention_qualified=qualified,
        source_confirmation_sha256=source_confirmation_sha256,
    )


def relational_artifact_contract(
    artifact: Mapping[str, Any], *, source_confirmation_sha256: str,
    source_intervention_qualified: bool,
) -> SourceIRContract:
    """Extract the anonymous type contract from a Sokoban-style artifact."""

    _artifact_hash(artifact)
    if artifact.get("schema_version") != (
        "source-induced-relational-structural-program-v2"
    ):
        raise ValueError("unsupported relational source artifact schema")
    program = artifact.get("program") or {}
    operators = tuple(_signature(row) for row in artifact.get("operator_types") or ())
    recurrent = any(
        row.get("from_operator_type_id") == row.get("to_operator_type_id")
        and row.get("cardinality") == "ONE_OR_MORE"
        for row in program.get("transitions") or ()
    )
    terminal = tuple(
        str(row["predicate_family"])
        for row in program.get("terminal_predicates") or ()
    )
    qualified = (
        source_intervention_qualified
        and artifact.get("target_data_read") is False
        and artifact.get("named_controller_template_used") is False
    )
    return SourceIRContract.create(
        program_sha256=str(artifact["artifact_sha256"]),
        ir_kind="RECURRENT_RELATIONAL_TRANSITION_PROGRAM",
        operator_sequence=operators,
        recurrent=recurrent,
        terminal_predicate_families=terminal,
        source_intervention_qualified=qualified,
        source_confirmation_sha256=source_confirmation_sha256,
    )


def goal_acquisition_artifact_contract(
    artifact: Mapping[str, Any], *, confirmation: Mapping[str, Any],
) -> SourceIRContract:
    """Extract the full acquisition-to-relation contract learned in Sokoban.

    Unlike the earlier one-operator relational contract, this contract keeps
    the two observed acquisition updates, the positive-effect cardinality
    binding, and the terminal relation update in their induced program order.
    Target adapters may bind these anonymous types to native functions, but
    cannot delete the cardinality guard or replace the learned recurrence.
    """

    validate_goal_acquisition_program(artifact)
    confirmation_body = dict(confirmation)
    claimed_confirmation = confirmation_body.pop("report_sha256", None)
    if not claimed_confirmation or claimed_confirmation != stable_hash(
        confirmation_body
    ):
        raise ValueError("source acquisition confirmation hash mismatch")
    if confirmation.get("artifact_sha256") != artifact.get("artifact_sha256"):
        raise ValueError("source acquisition confirmation/artifact mismatch")

    program = artifact["program"]
    by_id = {
        str(row["operator_type_id"]): row
        for row in artifact.get("operator_types") or ()
    }
    ordered_ids = (
        *map(str, program.get("acquisition_operator_type_ids") or ()),
        str(program["binding_operator_type_id"]),
        str(program["relation_operator_type_id"]),
    )
    if len(ordered_ids) != len(set(ordered_ids)) or any(
        type_id not in by_id for type_id in ordered_ids
    ):
        raise ValueError("source acquisition program has invalid operator order")
    operators = tuple(_signature(by_id[type_id]) for type_id in ordered_ids)
    recurrent = any(
        row.get("from_operator_type_id") == row.get("to_operator_type_id")
        for row in program.get("transition_graph") or ()
    )
    relation = by_id[str(program["relation_operator_type_id"])]
    qualified = (
        confirmation.get("status")
        == "SOURCE_GOAL_ACQUISITION_FRESH_VALIDATED"
        and confirmation.get("source_gate_passed") is True
        and all((confirmation.get("gates") or {}).values())
        and artifact.get("target_data_read") is False
        and artifact.get("named_controller_template_used") is False
    )
    return SourceIRContract.create(
        program_sha256=str(artifact["artifact_sha256"]),
        ir_kind="RECURRENT_GOAL_ACQUISITION_RELATION_PROGRAM",
        operator_sequence=operators,
        recurrent=recurrent,
        terminal_predicate_families=(str(relation["predicate_family"]),),
        source_intervention_qualified=qualified,
        source_confirmation_sha256=str(claimed_confirmation),
    )


def temporal_function_artifact_contract(
    artifact: Mapping[str, Any], *, source_confirmation_sha256: str,
    source_intervention_qualified: bool,
) -> SourceIRContract:
    """Extract a contract for an arcade temporal function without game IDs."""

    _artifact_hash(artifact)
    program = artifact.get("source_function_program")
    if not isinstance(program, Mapping):
        raise ValueError("source function artifact omitted program")
    validate_source_function_program(program)
    terms = program["source_function"]["terms"]
    signature = OperatorSignature(
        operation="SCORE",
        predicate_family="TEMPORAL_EFFECT_VECTOR",
        arity=len(terms),
        value_kind="NORMALIZED_PROBABILITY",
    )
    qualified = (
        source_intervention_qualified
        and program.get("status") == "SOURCE_DOMAIN_FUNCTION_QUALIFIED"
    )
    return SourceIRContract.create(
        program_sha256=str(program["program_sha256"]),
        ir_kind="SPARSE_TEMPORAL_EFFECT_FUNCTION",
        operator_sequence=(signature,),
        recurrent=bool(program["source_function"]["retry_after_low"]),
        terminal_predicate_families=(),
        source_intervention_qualified=qualified,
        source_confirmation_sha256=source_confirmation_sha256,
    )


def contract_matches(
    source: SourceIRContract, target: TargetIRRequirement,
) -> tuple[bool, str]:
    source.validate()
    target.validate()
    if target.formal_outcome_read:
        return False, "CURRENT_TARGET_OUTCOME_EXPOSED"
    if not target.grounder_qualified:
        return False, "TARGET_GROUNDER_NOT_QUALIFIED"
    if not source.source_intervention_qualified:
        return False, "SOURCE_PROGRAM_NOT_FRESH_CONFIRMED"
    if source.ir_kind != target.ir_kind:
        return False, "IR_KIND_MISMATCH"
    if source.operator_sequence != target.operator_sequence:
        return False, "OPERATOR_SIGNATURE_MISMATCH"
    if source.recurrent != target.recurrent:
        return False, "RECURRENCE_MISMATCH"
    if source.terminal_predicate_families != (
        target.terminal_predicate_families
    ):
        return False, "TERMINAL_PREDICATE_SIGNATURE_MISMATCH"
    return True, "EXACT_ANONYMOUS_STRUCTURAL_CONTRACT_MATCH"


def select_source_contract(
    sources: Sequence[SourceIRContract], target: TargetIRRequirement,
) -> dict[str, Any]:
    """Return a content-addressed program choice, never a target action."""

    rows = []
    for source in sorted(sources, key=lambda row: row.program_sha256):
        matched, reason = contract_matches(source, target)
        rows.append({
            "program_sha256": source.program_sha256,
            "contract_sha256": source.contract_sha256,
            "matched": matched,
            "reason": reason,
        })
    matches = [row for row in rows if row["matched"]]
    if len(matches) == 1:
        status = "UNIQUE_SOURCE_CONTRACT_SELECTED"
        reason = "EXACT_ANONYMOUS_STRUCTURAL_CONTRACT_MATCH"
        selected = matches[0]
    elif matches:
        status = "SOURCE_CONTRACT_SELECTION_ABSTAINED"
        reason = "MULTIPLE_SOURCE_CONTRACTS_MATCH"
        selected = None
    else:
        status = "SOURCE_CONTRACT_SELECTION_ABSTAINED"
        reason = "NO_SOURCE_CONTRACT_MATCHES"
        selected = None
    body = {
        "schema_version": "source-target-structural-applicability-v1",
        "status": status,
        "reason": reason,
        "target_requirement_sha256": target.requirement_sha256,
        "source_contracts": rows,
        "selected_program_sha256": (
            selected["program_sha256"] if selected else None
        ),
        "source_identity_used_as_feature": False,
        "target_outcome_read": target.formal_outcome_read,
        "target_action_emitted": False,
    }
    return body | {"receipt_sha256": stable_hash(body)}


__all__ = [
    "OperatorSignature", "SourceIRContract", "TargetIRRequirement",
    "contract_matches", "goal_acquisition_artifact_contract",
    "relational_artifact_contract",
    "select_source_contract", "structural_program_contract",
    "temporal_function_artifact_contract",
]
