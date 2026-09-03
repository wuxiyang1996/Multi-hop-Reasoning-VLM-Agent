"""AGQA binding for recurrence attached to a typed binary operator.

The source contract marks the *operator* recurrent; it does not mark each
argument independently recurrent.  Consequently, the minimum closed binding
grounds both arguments and requires one additional independent view for the
binary operator as a whole.  Every available cross-view interval pair must
still be strictly separated and entail the same relation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .agqa_robust_temporal_transfer import (
    AGQATemporalExecutor,
    NATIVE_RELATIONS,
    TemporalIntervalHypothesis,
    _operand_hypotheses,
    _relation,
)
from .contracts import stable_hash
from .structural_ir_applicability import SourceIRContract, TargetIRRequirement
from .unified_neurosymbolic_harness import (
    InducedProgramEnvelope,
    Phase7Authorization,
    UnifiedNeurosymbolicHarness,
    UnifiedTargetGrounding,
)
from .unified_transfer_runtime import (
    ExecutionAuthorization,
    PairedCalibration,
    SelectiveTargetExecutor,
    TargetGroundingReceipt,
    TransferVerdict,
    UnifiedNeurosymbolicTransferRuntime,
    UnifiedRoute,
)


TARGET_DOMAIN = "agqa2"
TARGET_INTERFACE = "aggregate_recurrent_temporal_pair_binding_v38"
CAPABILITIES = (
    "typed_binary_operand_grounding",
    "operator_level_recurrent_evidence",
    "strict_all_pairs_temporal_relation",
    "target_native_before_after_executor",
)


@dataclass(frozen=True)
class AGQAAggregateTemporalBindingReceipt:
    task_id: str
    target_state_sha256: str
    target_grounder_sha256: str
    source_program_sha256: str
    operand_a_hypotheses: tuple[TemporalIntervalHypothesis, ...]
    operand_b_hypotheses: tuple[TemporalIntervalHypothesis, ...]
    cross_view_relations: tuple[str, ...]
    resolved_relation: str | None
    binary_arguments_grounded: bool
    recurrent_operator_confirmed: bool
    all_pairs_strictly_separated: bool
    all_pairs_relation_consistent: bool
    effect_binding_authenticated: bool
    grounder_qualified: bool
    formal_outcome_read: bool
    authorized_relation: str | None
    reason: str
    receipt_sha256: str

    def validate(self) -> None:
        body = asdict(self)
        claimed = body.pop("receipt_sha256")
        if stable_hash(body) != claimed:
            raise ValueError("AGQA aggregate temporal binding hash mismatch")
        if self.formal_outcome_read and self.authorized_relation is not None:
            raise ValueError("current AGQA outcome authorized temporal transfer")
        if self.authorized_relation not in {None, *NATIVE_RELATIONS}:
            raise ValueError("non-native AGQA temporal relation escaped")
        if self.recurrent_operator_confirmed and not (
            self.binary_arguments_grounded
            and len(self.operand_a_hypotheses)
            + len(self.operand_b_hypotheses) >= 3
        ):
            raise ValueError("operator recurrence lacks typed repeated evidence")


def bind_aggregate_temporal_pair_program(
    *, task_id: str, target_state_sha256: str,
    target_grounder_sha256: str, source_program_sha256: str,
    obligation_kind: str, operand_runs: Mapping[str, Any],
    grounder_qualified: bool, effect_binding_authenticated: bool = True,
    minimum_confidence: float = 0.5, formal_outcome_read: bool = False,
) -> AGQAAggregateTemporalBindingReceipt:
    operand_a = _operand_hypotheses(
        operand_runs.get("A") or {}, minimum_confidence=minimum_confidence,
    )
    operand_b = _operand_hypotheses(
        operand_runs.get("B") or {}, minimum_confidence=minimum_confidence,
    )
    arguments_grounded = bool(operand_a) and bool(operand_b)
    recurrent_operator = (
        arguments_grounded and len(operand_a) + len(operand_b) >= 3
    )
    relations = tuple(
        relation
        for left in operand_a
        for right in operand_b
        if (relation := _relation(left, right)) is not None
    )
    pair_count = len(operand_a) * len(operand_b)
    separated = bool(pair_count) and len(relations) == pair_count
    consistent = separated and len(set(relations)) == 1
    resolved = relations[0] if consistent else None
    authorized = None
    if formal_outcome_read:
        reason = "CURRENT_TASK_OUTCOME_EXPOSURE"
    elif obligation_kind != "TEMPORAL_PAIR_RECURRENT":
        reason = "SOURCE_ABSTAIN_WRONG_TARGET_OBLIGATION"
    elif not grounder_qualified:
        reason = "TARGET_GROUNDER_NOT_QUALIFIED"
    elif not effect_binding_authenticated:
        reason = "SOURCE_EFFECT_BINDING_NOT_AUTHENTICATED"
    elif not arguments_grounded:
        reason = "SOURCE_ABSTAIN_BINARY_ARGUMENT_NOT_GROUNDED"
    elif not recurrent_operator:
        reason = "SOURCE_ABSTAIN_OPERATOR_RECURRENCE_NOT_CONFIRMED"
    elif not separated:
        reason = "SOURCE_ABSTAIN_INTERVAL_HYPOTHESES_OVERLAP"
    elif not consistent:
        reason = "SOURCE_ABSTAIN_TEMPORAL_RELATION_CONFLICT"
    else:
        authorized = resolved
        reason = "AGGREGATE_RECURRENCE_ALL_PAIRS_RELATION_RESOLVED"
    body = {
        "task_id": str(task_id),
        "target_state_sha256": str(target_state_sha256),
        "target_grounder_sha256": str(target_grounder_sha256),
        "source_program_sha256": str(source_program_sha256),
        "operand_a_hypotheses": [asdict(row) for row in operand_a],
        "operand_b_hypotheses": [asdict(row) for row in operand_b],
        "cross_view_relations": relations,
        "resolved_relation": resolved,
        "binary_arguments_grounded": arguments_grounded,
        "recurrent_operator_confirmed": recurrent_operator,
        "all_pairs_strictly_separated": separated,
        "all_pairs_relation_consistent": consistent,
        "effect_binding_authenticated": bool(effect_binding_authenticated),
        "grounder_qualified": bool(grounder_qualified),
        "formal_outcome_read": bool(formal_outcome_read),
        "authorized_relation": authorized,
        "reason": reason,
    }
    receipt = AGQAAggregateTemporalBindingReceipt(
        task_id=body["task_id"],
        target_state_sha256=body["target_state_sha256"],
        target_grounder_sha256=body["target_grounder_sha256"],
        source_program_sha256=body["source_program_sha256"],
        operand_a_hypotheses=operand_a,
        operand_b_hypotheses=operand_b,
        cross_view_relations=relations,
        resolved_relation=resolved,
        binary_arguments_grounded=arguments_grounded,
        recurrent_operator_confirmed=recurrent_operator,
        all_pairs_strictly_separated=separated,
        all_pairs_relation_consistent=consistent,
        effect_binding_authenticated=body["effect_binding_authenticated"],
        grounder_qualified=body["grounder_qualified"],
        formal_outcome_read=body["formal_outcome_read"],
        authorized_relation=authorized,
        reason=reason,
        receipt_sha256=stable_hash(body),
    )
    receipt.validate()
    return receipt


def build_aggregate_temporal_route(
    *, source_program_sha256: str, target_grounder_sha256: str,
    target_executor_sha256: str, evidence_report_sha256: str,
    utility_vs_target_native: PairedCalibration,
    authenticity_vs_effect_shuffled: PairedCalibration,
) -> UnifiedRoute:
    return UnifiedRoute(
        route_id="source-temporal-function-to-agqa-before-after-v38",
        target_domain=TARGET_DOMAIN,
        target_interface=TARGET_INTERFACE,
        required_capabilities=CAPABILITIES,
        source_program_sha256=source_program_sha256,
        source_program_induced_from_interventions=True,
        source_program_qualified=True,
        target_grounder_sha256=target_grounder_sha256,
        target_executor_sha256=target_executor_sha256,
        target_grounder_id="agqa2.aggregate-recurrent-interval-grounder.v38",
        target_executor_id="agqa2.native-before-after-executor.v38",
        evidence_report_sha256=evidence_report_sha256,
        utility_vs_neural=utility_vs_target_native,
        authenticity_vs_source_permuted=authenticity_vs_effect_shuffled,
    )


def build_aggregate_temporal_harness(
    *, contract: SourceIRContract, source_transition_receipts_sha256: str,
    inducer_artifact_sha256: str, route: UnifiedRoute,
) -> UnifiedNeurosymbolicHarness:
    envelope = InducedProgramEnvelope.create(
        contract=contract,
        source_transition_receipts_sha256=source_transition_receipts_sha256,
        inducer_artifact_sha256=inducer_artifact_sha256,
    )
    return UnifiedNeurosymbolicHarness(
        (envelope,), UnifiedNeurosymbolicTransferRuntime((route,)),
    )


def unified_aggregate_temporal_grounding(
    *, contract: SourceIRContract,
    binding: AGQAAggregateTemporalBindingReceipt,
) -> UnifiedTargetGrounding:
    binding.validate()
    requirement = TargetIRRequirement.create(
        task_id=binding.task_id,
        target_domain=TARGET_DOMAIN,
        target_interface=TARGET_INTERFACE,
        target_grounder_sha256=binding.target_grounder_sha256,
        ir_kind=contract.ir_kind,
        operator_sequence=contract.operator_sequence,
        recurrent=contract.recurrent,
        terminal_predicate_families=contract.terminal_predicate_families,
        grounder_qualified=binding.grounder_qualified,
        formal_outcome_read=binding.formal_outcome_read,
    )
    applicability = TargetGroundingReceipt.create(
        task_id=binding.task_id,
        target_domain=TARGET_DOMAIN,
        target_interface=TARGET_INTERFACE,
        target_state_sha256=binding.target_state_sha256,
        target_grounder_sha256=binding.target_grounder_sha256,
        capabilities=CAPABILITIES,
        candidate_ids=(
            (binding.authorized_relation,)
            if binding.authorized_relation is not None else ()
        ),
        structural_predicates={
            "binary_arguments_grounded": binding.binary_arguments_grounded,
            "recurrent_operator_confirmed": (
                binding.recurrent_operator_confirmed
            ),
            "all_pairs_relation_consistent": (
                binding.all_pairs_relation_consistent
            ),
            "all_pairs_strictly_separated": (
                binding.all_pairs_strictly_separated
            ),
            "effect_binding_authenticated": (
                binding.effect_binding_authenticated
            ),
            "source_binding_authorized": (
                binding.authorized_relation is not None
            ),
        },
        grounder_qualified=binding.grounder_qualified,
        formal_outcome_read=binding.formal_outcome_read,
    )
    return UnifiedTargetGrounding.create(
        requirement=requirement, applicability=applicability,
    )


@dataclass(frozen=True)
class AGQAAggregateTemporalDecision:
    source_relation: str | None
    binding: AGQAAggregateTemporalBindingReceipt
    phase7: Phase7Authorization
    utility: ExecutionAuthorization
    executor_calls: int


def decide_aggregate_temporal_relation(
    *, harness: UnifiedNeurosymbolicHarness,
    target: UnifiedTargetGrounding,
    binding: AGQAAggregateTemporalBindingReceipt,
    target_executor_sha256: str,
) -> AGQAAggregateTemporalDecision:
    phase7 = harness.decide(target)
    utility = harness.runtime.decide(target.applicability)
    executor = AGQATemporalExecutor(
        target_executor_sha256, binding.authorized_relation,
    )
    relation = SelectiveTargetExecutor.execute(
        utility, target.applicability, NATIVE_RELATIONS, executor,
    ) if phase7.verdict == TransferVerdict.SELECT_SKILL else None
    if (relation is not None) != (
        phase7.verdict == TransferVerdict.SELECT_SKILL
    ):
        raise AssertionError("AGQA aggregate authorization/execution mismatch")
    return AGQAAggregateTemporalDecision(
        source_relation=relation,
        binding=binding,
        phase7=phase7,
        utility=utility,
        executor_calls=executor.calls,
    )


def aggregate_target_grounder_sha256(
    *, parent_grounder_sha256: str, adapter_module_sha256: str,
    normalization_module_sha256: str, acquisition_collector_sha256: str,
) -> str:
    return stable_hash({
        "schema_version": "agqa2-aggregate-temporal-grounder-v38",
        "parent_grounder_sha256": parent_grounder_sha256,
        "adapter_module_sha256": adapter_module_sha256,
        "normalization_module_sha256": normalization_module_sha256,
        "acquisition_collector_sha256": acquisition_collector_sha256,
        "typed_binding_rule": (
            "BINARY_ARITY_GROUNDED;OPERATOR_LEVEL_RECURRENCE_MINIMUM_THREE_"
            "TOTAL_VIEWS;ALL_CROSS_VIEW_PAIRS_STRICT_AND_CONSISTENT"
        ),
        "minimum_confidence": 0.5,
        "outcome_or_label_input": False,
    })


__all__ = [
    "AGQAAggregateTemporalBindingReceipt", "AGQAAggregateTemporalDecision",
    "CAPABILITIES", "TARGET_DOMAIN", "TARGET_INTERFACE",
    "aggregate_target_grounder_sha256",
    "bind_aggregate_temporal_pair_program",
    "build_aggregate_temporal_harness", "build_aggregate_temporal_route",
    "decide_aggregate_temporal_relation",
    "unified_aggregate_temporal_grounding",
]
