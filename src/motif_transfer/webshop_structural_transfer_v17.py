"""Source-induced relational structural transfer to WebShop.

This module deliberately does not reuse the named V16
``EXPLORE/BACKTRACK/COMMIT`` source automaton.  The source input is the
source-only Sokoban relational artifact induced from ``(s, a, e, s')`` tuples.
The target input is a WebShop domain function induced from development
transition receipts.  The only shared vocabulary is the anonymous structural
IR: a repeating control-state update followed by relation coverage.

The target-native controller is still responsible for grounding WebShop DOM
entities, option relations, and executable actions.  Structural compatibility
only decides whether that controller may be selected; incompatibility and
ambiguity fail closed to the matched neural policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import stable_hash
from .relational_structural_induction import (
    UPDATE_CONTROL_POSITION,
    validate_relational_structural_program,
)
from .webshop_coverage_transfer_v14 import (
    TARGET_COVERAGE,
    TARGET_ONLY,
    CoverageTransferController,
)
from .webshop_sokoban_effect_transfer import EffectTransferDecision


NEURAL_ONLY = "neural_only"
SOURCE_INDUCED = "source_induced_structural_ir"
SOURCE_PERMUTED = "source_terminal_permuted_control"
GENERIC_SCAFFOLD = "generic_untyped_scaffold"
TARGET_NATIVE_CEILING = "target_native_structural_ceiling"
CONDITIONS = (
    NEURAL_ONLY,
    SOURCE_INDUCED,
    SOURCE_PERMUTED,
    GENERIC_SCAFFOLD,
    TARGET_NATIVE_CEILING,
)


def _validate_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"{label} hash mismatch")


def _operator_signature(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("operation")),
        str(row.get("predicate_family")),
        int(row.get("arity", 0)),
        str(row.get("value_kind")),
    )


def _predicate_signature(row: Mapping[str, Any]) -> tuple[str, int, str, str, Any]:
    return (
        str(row.get("predicate_family")),
        int(row.get("arity", 0)),
        str(row.get("value_kind")),
        str(row.get("operator")),
        row.get("value"),
    )


def _receipt_hash_valid(receipt: Mapping[str, Any]) -> bool:
    body = dict(receipt)
    claimed = str(body.pop("receipt_sha256", ""))
    return bool(claimed) and stable_hash(body) == claimed


def induce_webshop_relational_function(
    receipts: Sequence[Mapping[str, Any]],
    *,
    development_receipts_sha256: str,
    target_grounder_sha256: str,
    minimum_success_receipts: int = 2,
) -> dict[str, Any]:
    """Induce a target domain function from development transitions only.

    The hypothesis class is fixed before reading target outcomes.  A changing
    WebShop page/form state is represented as an anonymous control-state
    position update.  The terminal hypothesis is admitted only when every
    successful receipt ends with a target-native, coverage-authorized commit.
    No source artifact is an input to this function.
    """

    rows = list(receipts)
    if not rows:
        raise ValueError("WebShop target induction requires development receipts")
    if not all(_receipt_hash_valid(row) for row in rows):
        raise ValueError("WebShop development receipt failed content hash validation")
    successes = [
        row for row in rows
        if bool(row.get("strict_success")) and row.get("failure") is None
    ]
    # Relation coverage is a structural family, not a universal WebShop
    # controller.  A successful goal with no required option relation is an
    # out-of-family example and must trigger episode-level abstention rather
    # than falsifying or broadening the learned terminal predicate.
    relation_successes = []
    for receipt in successes:
        commits = list(receipt.get("commit_audit") or ())
        if any((commit.get("coverage") or {}).get("required") for commit in commits):
            relation_successes.append(receipt)
    changing_prefix_lengths = []
    safe_terminal = []
    for receipt in relation_successes:
        steps = list(receipt.get("steps") or ())
        changing_prefix_lengths.append(sum(
            str(step.get("before_hash")) != str(step.get("after_hash"))
            and not bool(step.get("terminated"))
            for step in steps
        ))
        commits = list(receipt.get("commit_audit") or ())
        safe_terminal.append(bool(commits) and bool(commits[-1].get("authorized")))

    gates = {
        "minimum_success_receipts": (
            len(relation_successes) >= minimum_success_receipts
        ),
        "all_successes_have_state_changing_prefix": bool(relation_successes) and all(
            count >= 1 for count in changing_prefix_lengths
        ),
        "all_successes_end_in_coverage_authorized_commit": (
            bool(relation_successes) and all(safe_terminal)
        ),
        "repetition_observed": bool(relation_successes) and any(
            count > 1 for count in changing_prefix_lengths
        ),
    }
    qualified = all(gates.values())
    terminal = {
        "predicate_family": "ENTITY_GOAL_RELATION",
        "arity": 2,
        "value_kind": "RELATION_COVERAGE",
        "feature": "target_native_option_relation_coverage",
        "operator": "EQ",
        "value": 1.0,
    }
    body = {
        "schema_version": "target-induced-relational-structural-program-v1",
        "status": (
            "TARGET_RELATIONAL_FUNCTION_QUALIFIED" if qualified
            else "TARGET_RELATIONAL_FUNCTION_ABSTAINING"
        ),
        "development_receipts_sha256": str(development_receipts_sha256),
        "target_grounder_sha256": str(target_grounder_sha256),
        "induction_authority": (
            "TARGET_DEVELOPMENT_STATE_ACTION_EFFECT_NEXT_STATE_AND_"
            "TARGET_NATIVE_COVERAGE_RECEIPTS_ONLY"
        ),
        "operator_types": [dict(UPDATE_CONTROL_POSITION)] if qualified else [],
        "program": {
            "entry_operator_type_id": UPDATE_CONTROL_POSITION["operator_type_id"],
            "transitions": [{
                "from_operator_type_id": UPDATE_CONTROL_POSITION["operator_type_id"],
                "guard": "NEXT_GROUNDED_EFFECT_HAS_SAME_OPERATOR_TYPE",
                "to_operator_type_id": UPDATE_CONTROL_POSITION["operator_type_id"],
                "cardinality": "ONE_OR_MORE",
            }],
            "terminal_predicates": [terminal],
            "terminal_rule": (
                "TARGET_NATIVE_RELATION_COVERAGE_AFTER_TYPED_TRANSITIONS"
            ),
            "abstention_rule": {
                "missing_neural_effect_grounding": "ABSTAIN",
                "missing_target_relation_binding": "ABSTAIN",
                "source_target_ir_mismatch": "ABSTAIN",
            },
        } if qualified else {
            "transitions": [], "terminal_predicates": [],
            "abstention_rule": {"target_not_qualified": "ABSTAIN"},
        },
        "qualification_metrics": {
            "receipts": len(rows),
            "successful_receipts": len(successes),
            "relation_family_successful_receipts": len(relation_successes),
            "out_of_family_successful_receipts": (
                len(successes) - len(relation_successes)
            ),
            "state_changing_prefix_lengths": changing_prefix_lengths,
            "safe_terminal_receipts": sum(safe_terminal),
        },
        "qualification_gates": gates,
        "source_program_read_during_induction": False,
        "formal_target_data_read": False,
        "named_policy_template_used": False,
    }
    return body | {"program_sha256": stable_hash(body)}


def validate_webshop_relational_function(program: Mapping[str, Any]) -> None:
    _validate_hash(program, "program_sha256", "WebShop relational function")
    if program.get("schema_version") != (
        "target-induced-relational-structural-program-v1"
    ):
        raise ValueError("unsupported WebShop relational function schema")
    if program.get("source_program_read_during_induction") is not False:
        raise ValueError("source program leaked into target function induction")
    if program.get("formal_target_data_read") is not False:
        raise ValueError("formal target data leaked into target function")
    if program.get("named_policy_template_used") is not False:
        raise ValueError("named policy template leaked into target function")


def permute_target_terminal(program: Mapping[str, Any]) -> dict[str, Any]:
    """Create a deterministic structural, not source-identity, control."""

    validate_webshop_relational_function(program)
    body = {key: value for key, value in program.items() if key != "program_sha256"}
    body = dict(body)
    body["program"] = dict(body["program"])
    predicates = [dict(row) for row in body["program"]["terminal_predicates"]]
    if not predicates:
        raise ValueError("terminal permutation requires a qualified target function")
    predicates[0].update({
        "predicate_family": "CONTROL_GOAL_RELATION",
        "arity": 2,
        "value_kind": "BOOLEAN",
        "feature": "target_control_on_goal_relation",
        "value": True,
    })
    body["program"]["terminal_predicates"] = predicates
    body["status"] = "TARGET_RELATIONAL_FUNCTION_TERMINAL_PERMUTED_CONTROL"
    return body | {"program_sha256": stable_hash(body)}


def structural_compatibility_receipt(
    source: Mapping[str, Any],
    source_confirmation: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Match programs by content, never by source or target identity."""

    validate_relational_structural_program(source)
    validate_webshop_relational_function(target)
    _validate_hash(source_confirmation, "report_sha256", "source confirmation")
    confirmation_valid = (
        source_confirmation.get("status")
        == "SOURCE_RELATIONAL_STRUCTURAL_FRESH_VALIDATED"
        and source_confirmation.get("source_gate_passed") is True
        and source_confirmation.get("artifact_sha256")
        == source.get("artifact_sha256")
        and all((source_confirmation.get("gates") or {}).values())
    )
    source_operators = {
        _operator_signature(row) for row in source.get("operator_types") or ()
    }
    target_operators = {
        _operator_signature(row) for row in target.get("operator_types") or ()
    }
    source_transitions = list((source.get("program") or {}).get("transitions") or ())
    target_transitions = list((target.get("program") or {}).get("transitions") or ())
    source_repeat = any(row.get("cardinality") == "ONE_OR_MORE" for row in source_transitions)
    target_repeat = any(row.get("cardinality") == "ONE_OR_MORE" for row in target_transitions)
    source_terminals = {
        _predicate_signature(row)
        for row in (source.get("program") or {}).get("terminal_predicates") or ()
    }
    target_terminals = {
        _predicate_signature(row)
        for row in (target.get("program") or {}).get("terminal_predicates") or ()
    }
    gates = {
        "source_fresh_confirmed": confirmation_valid,
        "target_development_qualified": (
            target.get("status") == "TARGET_RELATIONAL_FUNCTION_QUALIFIED"
        ),
        "source_operator_subgraph_matches": bool(source_operators)
        and source_operators.issubset(target_operators),
        "repetition_contract_matches": source_repeat and target_repeat,
        "terminal_predicate_matches": bool(source_terminals)
        and source_terminals.issubset(target_terminals),
    }
    body = {
        "schema_version": "source-target-structural-compatibility-v1",
        "status": (
            "STRUCTURAL_TRANSFER_ADMITTED" if all(gates.values())
            else "STRUCTURAL_TRANSFER_ABSTAINED"
        ),
        "source_artifact_sha256": str(source.get("artifact_sha256")),
        "source_confirmation_report_sha256": str(
            source_confirmation.get("report_sha256")
        ),
        "target_program_sha256": str(target.get("program_sha256")),
        "source_operator_signatures": [list(row) for row in sorted(source_operators)],
        "target_operator_signatures": [list(row) for row in sorted(target_operators)],
        "source_terminal_signatures": [list(row) for row in sorted(source_terminals)],
        "target_terminal_signatures": [list(row) for row in sorted(target_terminals)],
        "gates": gates,
        "source_identity_used_as_feature": False,
        "target_outcome_read_at_routing": False,
    }
    return body | {"compatibility_sha256": stable_hash(body)}


@dataclass
class WebShopStructuralController:
    """Select and execute a target-native domain function via structural IR."""

    condition: str
    source: Mapping[str, Any]
    source_confirmation: Mapping[str, Any]
    target_function: Mapping[str, Any]
    goal_options: Mapping[str, Any] = field(default_factory=dict)
    maximum_steps: int = 12

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown WebShop structural condition: {self.condition}")
        target = (
            permute_target_terminal(self.target_function)
            if self.condition == SOURCE_PERMUTED else self.target_function
        )
        self.compatibility = structural_compatibility_receipt(
            self.source, self.source_confirmation, target,
        )
        admitted = self.compatibility["status"] == "STRUCTURAL_TRANSFER_ADMITTED"
        self.episode_schema_applicable = bool(self.goal_options)
        if self.condition == SOURCE_INDUCED:
            self.source_admitted = admitted and self.episode_schema_applicable
        elif self.condition == TARGET_NATIVE_CEILING:
            self.source_admitted = False
        else:
            self.source_admitted = False
        enable_target_function = (
            (self.condition == SOURCE_INDUCED and self.source_admitted)
            or (
                self.condition == TARGET_NATIVE_CEILING
                and self.episode_schema_applicable
            )
        )
        self.target_function_enabled = enable_target_function
        self.controller = CoverageTransferController(
            TARGET_COVERAGE if enable_target_function else TARGET_ONLY,
            goal_options=self.goal_options,
            anytime_reward_salvage=enable_target_function,
            maximum_steps=self.maximum_steps,
        )
        self.decisions = 0
        self.source_authorized_decisions = 0
        self.terminal_bindings = 0
        self.last_binding_signature: str | None = None
        self.refuted_binding_signatures: set[str] = set()
        self.effect_refutation_retries = 0
        self.effect_refutation_abstentions = 0
        self.runtime_binding_abstained = False
        self.runtime_binding_abstention_decisions = 0

    @staticmethod
    def _binding_signature(
        action: str, semantic: Mapping[str, Any],
    ) -> str:
        """Content-address a target binding without source or task identity."""

        return stable_hash({
            "action": str(action),
            "verb": str(semantic.get("verb")),
            "url_phase": str(semantic.get("url_phase")),
            "element_role": str(semantic.get("element_role")),
            "element_text": str(semantic.get("element_text")),
            "is_constraint": bool(semantic.get("is_constraint")),
            "is_commit": bool(semantic.get("is_commit")),
        })

    def _record_binding(
        self, *, index: int, candidates: Sequence[str],
        semantics: Sequence[Mapping[str, Any]],
    ) -> None:
        self.last_binding_signature = self._binding_signature(
            candidates[index], semantics[index],
        )

    def __call__(
        self,
        *,
        condition: str,
        predictions: np.ndarray,
        semantics: Sequence[Mapping[str, Any]],
        source_models: Mapping[str, Any],
        visible_satisfied: bool,
        visible_unsatisfied: bool,
        prior_no_effect: bool,
        remaining_fraction: float,
        previous_action: str | None,
        candidates: Sequence[str],
        uncertainty_scale: float,
        decision_margin: float,
    ) -> EffectTransferDecision:
        if condition != self.condition:
            raise ValueError("WebShop structural controller condition mismatch")
        if self.runtime_binding_abstained:
            # A missing exact target relation binding invalidates the domain
            # function for the rest of this episode.  Fail closed to the
            # unchanged target-neural rank-zero policy; do not let a later UI
            # state silently restore source authority.
            self.decisions += 1
            self.runtime_binding_abstention_decisions += 1
            self._record_binding(
                index=0, candidates=candidates, semantics=semantics,
            )
            return EffectTransferDecision(
                selected_index=0,
                abstract_kind="TARGET",
                source_abstained=True,
                source_test_value=None,
                source_commit_value=None,
                reason="target_runtime_binding_abstained_episode",
            )
        if prior_no_effect and self.last_binding_signature is not None:
            # The source operator is an observed state UPDATE.  A native
            # binding that left the state unchanged is therefore refuted by
            # the environment itself and may not retain source authority.
            self.refuted_binding_signatures.add(self.last_binding_signature)
        proposal = self.controller(
            condition=self.controller.condition,
            predictions=predictions,
            semantics=semantics,
            source_models=source_models,
            visible_satisfied=visible_satisfied,
            visible_unsatisfied=visible_unsatisfied,
            prior_no_effect=prior_no_effect,
            remaining_fraction=remaining_fraction,
            previous_action=previous_action,
            candidates=candidates,
            uncertainty_scale=uncertainty_scale,
            decision_margin=decision_margin,
        )
        # Execute the induced terminal edge.  CoverageTransferController owns
        # the target-native relation ledger, but its legacy base policy may
        # rank a tab/navigation action above Buy Now after coverage reaches
        # one.  The shared IR explicitly says that the next transition is the
        # target terminal binding at that point.  Bind only a unique native
        # commit; ambiguity remains fail-closed.
        if self.target_function_enabled and self.controller.coverage_ready:
            commit_indices = [
                index for index, row in enumerate(semantics)
                if bool(row.get("is_commit"))
            ]
            if len(commit_indices) == 1:
                proposal = EffectTransferDecision(
                    selected_index=commit_indices[0],
                    abstract_kind="STRUCTURAL_TERMINAL",
                    source_abstained=(self.condition != SOURCE_INDUCED),
                    source_test_value=None,
                    source_commit_value=None,
                    reason="target_native_unique_commit_after_relation_coverage",
                )
                self.terminal_bindings += 1
        binding_signature = self._binding_signature(
            candidates[proposal.selected_index], semantics[proposal.selected_index],
        )
        binding_abstained = False
        if (
            self.target_function_enabled
            and not bool(semantics[proposal.selected_index].get("is_commit"))
            and (
                binding_signature in self.refuted_binding_signatures
            )
        ):
            replacement = next((
                index for index, row in enumerate(semantics)
                if not bool(row.get("is_commit"))
                and not bool(row.get("is_noop"))
                and self._binding_signature(candidates[index], row)
                not in self.refuted_binding_signatures
            ), None)
            if replacement is None:
                binding_abstained = True
                self.effect_refutation_abstentions += 1
            else:
                proposal = EffectTransferDecision(
                    selected_index=replacement,
                    abstract_kind="STRUCTURAL_IR",
                    source_abstained=(self.condition != SOURCE_INDUCED),
                    source_test_value=None,
                    source_commit_value=None,
                    reason="target_native_retry_after_effect_refutation",
                )
                self.effect_refutation_retries += 1
        self._record_binding(
            index=proposal.selected_index,
            candidates=candidates,
            semantics=semantics,
        )
        self.decisions += 1
        if proposal.reason == (
            "target_coverage_neural_relation_binding_abstention"
        ):
            self.runtime_binding_abstained = True
            self.runtime_binding_abstention_decisions += 1
            return proposal
        if self.condition == SOURCE_INDUCED and self.source_admitted:
            if binding_abstained:
                return EffectTransferDecision(
                    selected_index=proposal.selected_index,
                    abstract_kind="TARGET",
                    source_abstained=True,
                    source_test_value=None,
                    source_commit_value=None,
                    reason="source_ir_no_unrefuted_target_binding",
                )
            if proposal.reason.startswith(
                "target_budget_infeasible_immediate_reward_salvage:"
            ):
                # The structural program cannot satisfy its terminal
                # predicate in the remaining horizon.  It has no authority to
                # weaken that predicate.  Preserve the target-native fallback
                # and its explicit source_abstention receipt.
                return proposal
            self.source_authorized_decisions += 1
            return EffectTransferDecision(
                selected_index=proposal.selected_index,
                abstract_kind="STRUCTURAL_IR",
                source_abstained=False,
                source_test_value=None,
                source_commit_value=None,
                reason=(
                    "source_induced_relational_function:"
                    f"{proposal.reason}"
                ),
            )
        return proposal

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "source_admitted": self.source_admitted,
            "episode_schema_applicable": self.episode_schema_applicable,
            "target_function_enabled": self.target_function_enabled,
            "decisions": self.decisions,
            "source_authorized_decisions": self.source_authorized_decisions,
            "terminal_bindings": self.terminal_bindings,
            "effect_refutation_retries": self.effect_refutation_retries,
            "effect_refutation_abstentions": self.effect_refutation_abstentions,
            "runtime_binding_abstained": self.runtime_binding_abstained,
            "runtime_binding_abstention_decisions": (
                self.runtime_binding_abstention_decisions
            ),
            "refuted_binding_hashes": sorted(self.refuted_binding_signatures),
            "compatibility": self.compatibility,
            "target_controller": self.controller.as_dict(),
            "named_policy_template_used": False,
        }


__all__ = [
    "CONDITIONS", "GENERIC_SCAFFOLD", "NEURAL_ONLY", "SOURCE_INDUCED",
    "SOURCE_PERMUTED", "TARGET_NATIVE_CEILING",
    "WebShopStructuralController", "induce_webshop_relational_function",
    "permute_target_terminal", "structural_compatibility_receipt",
    "validate_webshop_relational_function",
]
