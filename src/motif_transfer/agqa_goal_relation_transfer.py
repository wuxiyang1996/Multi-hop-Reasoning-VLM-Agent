"""AGQA adapter for the source-induced recurrent goal-relation program.

The adapter keeps the authorities in the unified harness separate:

* the source artifact supplies the transition, terminal, and abstention rules;
* target-native neural views supply relation-object bindings;
* the unified runtime decides whether a calibrated source route may execute;
* a target-native executor emits an AGQA object label; and
* abstention preserves the independently frozen target-native prediction.

No function in this module accepts a gold answer or a current-task outcome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .agqa_query_object_grounder import (
    AGQA_OBJECT_ONTOLOGY,
    canonical_object_label,
)
from .clevrer_unified_goal_relation import (
    source_goal_relation_contract,
    source_goal_relation_envelope,
)
from .contracts import stable_hash
from .source_goal_relation_induction import (
    validate_goal_relation_macro_program,
)
from .structural_ir_applicability import TargetIRRequirement
from .unified_neurosymbolic_harness import (
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
TARGET_INTERFACE = "query_object_relation_binding_v29"
CAPABILITIES = (
    "candidate_blind_relation_votes",
    "native_object_answers",
    "source_induced_relation_abstention",
)


def _validated_source_lineage(
    artifact: Mapping[str, Any], confirmation: Mapping[str, Any],
) -> None:
    validate_goal_relation_macro_program(artifact)
    body = dict(confirmation)
    claimed = str(body.pop("report_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("source confirmation hash mismatch")
    if confirmation.get("artifact_sha256") != artifact.get("artifact_sha256"):
        raise ValueError("source artifact/confirmation mismatch")
    if confirmation.get("status") != "SOURCE_GOAL_RELATION_MACRO_FRESH_VALIDATED":
        raise ValueError("source goal-relation program is not fresh validated")
    if confirmation.get("source_gate_passed") is not True:
        raise ValueError("source goal-relation confirmation gate failed")
    if not all((confirmation.get("gates") or {}).values()):
        raise ValueError("source confirmation contains a failed gate")


def _validate_program_semantics(artifact: Mapping[str, Any]) -> None:
    """Reject adapters that silently discard an induced source rule."""

    program = artifact["program"]
    transitions = list(program.get("transitions") or ())
    terminals = list(program.get("terminal_predicates") or ())
    abstention = dict(program.get("abstention_rule") or {})
    if len(transitions) != 1:
        raise ValueError("AGQA adapter requires one induced recurrent transition")
    transition = transitions[0]
    if transition.get("cardinality") != "ONE_OR_MORE" or transition.get(
        "observed_effect_guard"
    ) != {
        "feature": "entity_goal_relation_coverage",
        "change_sign": "INCREASE",
    }:
        raise ValueError("AGQA adapter/source transition mismatch")
    if terminals != [{
        "predicate_family": "ENTITY_GOAL_RELATION",
        "arity": 2,
        "value_kind": "RELATION_COVERAGE",
        "feature": "entity_goal_relation_coverage",
        "operator": "EQ",
        "value": 1.0,
    }]:
        raise ValueError("AGQA adapter/source terminal predicate mismatch")
    required_abstentions = {
        "zero_target_bindings",
        "multiple_target_bindings",
        "nonpositive_observed_relation_delta",
        "terminal_predicate_unobservable",
    }
    if set(abstention) != required_abstentions or any(
        abstention[key] != "ABSTAIN" for key in required_abstentions
    ):
        raise ValueError("AGQA adapter/source abstention rules mismatch")


@dataclass(frozen=True)
class AGQAGoalRelationBindingReceipt:
    """Outcome-blind execution of the induced program on neural bindings."""

    task_id: str
    source_program_sha256: str
    source_confirmation_sha256: str
    target_state_sha256: str
    target_grounder_sha256: str
    candidate_bindings: tuple[str, ...]
    target_binding_count: int
    positive_observed_relation_delta: bool
    terminal_predicate_observable: bool
    unique_target_binding: bool
    effect_binding_authenticated: bool
    grounder_qualified: bool
    formal_outcome_read: bool
    authorized_candidate: str | None
    reason: str
    receipt_sha256: str

    def validate(self) -> None:
        body = asdict(self)
        claimed = body.pop("receipt_sha256")
        if stable_hash(body) != claimed:
            raise ValueError("AGQA goal-relation binding receipt hash mismatch")
        if self.formal_outcome_read and self.authorized_candidate is not None:
            raise ValueError("outcome-exposed AGQA binding was authorized")
        if self.authorized_candidate is not None and (
            not self.unique_target_binding
            or self.authorized_candidate not in self.candidate_bindings
        ):
            raise ValueError("AGQA binding authorization is inconsistent")


def bind_source_goal_relation_program(
    *, artifact: Mapping[str, Any], confirmation: Mapping[str, Any],
    task_id: str, target_state_sha256: str, target_grounder_sha256: str,
    calibrated_execution: Mapping[str, Any], grounder_qualified: bool,
    effect_binding_authenticated: bool = True,
    formal_outcome_read: bool = False,
) -> AGQAGoalRelationBindingReceipt:
    """Bind target neural votes to every induced source-side program guard.

    The query begins with an unbound object variable.  Each accepted neural
    vote is a target-native observation of an entity-relation binding, so any
    accepted vote is a positive coverage delta from that initial state.  The
    induced terminal predicate is observable only when the target executor has
    a candidate.  The source abstention rule then requires exactly one distinct
    canonical binding; a 2-of-3 majority with a conflicting binding is not
    source-authorized.
    """

    _validated_source_lineage(artifact, confirmation)
    _validate_program_semantics(artifact)
    votes = calibrated_execution.get("neural_votes") or ()
    bindings = tuple(sorted({
        label for row in votes
        if isinstance(row, Mapping)
        for label in (canonical_object_label(str(row.get("decision") or "")),)
        if label in AGQA_OBJECT_ONTOLOGY
    }))
    candidate = canonical_object_label(str(
        calibrated_execution.get("decision") or ""
    ))
    if candidate not in AGQA_OBJECT_ONTOLOGY:
        candidate = ""
    positive_delta = bool(bindings)
    terminal_observable = bool(candidate)
    unique = len(bindings) == 1 and bool(candidate) and bindings[0] == candidate
    authorized = (
        bool(grounder_qualified)
        and not formal_outcome_read
        and bool(effect_binding_authenticated)
        and positive_delta
        and terminal_observable
        and unique
    )
    if formal_outcome_read:
        reason = "CURRENT_TASK_OUTCOME_EXPOSURE"
    elif not grounder_qualified:
        reason = "TARGET_GROUNDER_NOT_QUALIFIED"
    elif not effect_binding_authenticated:
        reason = "SOURCE_EFFECT_BINDING_NOT_AUTHENTICATED"
    elif not bindings:
        reason = "SOURCE_ABSTAIN_ZERO_TARGET_BINDINGS"
    elif len(bindings) != 1:
        reason = "SOURCE_ABSTAIN_MULTIPLE_TARGET_BINDINGS"
    elif not positive_delta:
        reason = "SOURCE_ABSTAIN_NONPOSITIVE_RELATION_DELTA"
    elif not terminal_observable or bindings[0] != candidate:
        reason = "SOURCE_ABSTAIN_TERMINAL_PREDICATE_UNOBSERVABLE"
    else:
        reason = "SOURCE_TRANSITION_AND_TERMINAL_PREDICATE_SATISFIED"
    body = {
        "task_id": str(task_id),
        "source_program_sha256": str(artifact["artifact_sha256"]),
        "source_confirmation_sha256": str(confirmation["report_sha256"]),
        "target_state_sha256": str(target_state_sha256),
        "target_grounder_sha256": str(target_grounder_sha256),
        "candidate_bindings": bindings,
        "target_binding_count": len(bindings),
        "positive_observed_relation_delta": positive_delta,
        "terminal_predicate_observable": terminal_observable,
        "unique_target_binding": unique,
        "effect_binding_authenticated": bool(effect_binding_authenticated),
        "grounder_qualified": bool(grounder_qualified),
        "formal_outcome_read": bool(formal_outcome_read),
        "authorized_candidate": candidate if authorized else None,
        "reason": reason,
    }
    receipt = AGQAGoalRelationBindingReceipt(
        **body, receipt_sha256=stable_hash(body),
    )
    receipt.validate()
    return receipt


def build_route(
    *, source_program_sha256: str, target_grounder_sha256: str,
    target_executor_sha256: str, evidence_report_sha256: str,
    utility_vs_target_native: PairedCalibration,
    authenticity_vs_effect_shuffled: PairedCalibration,
) -> UnifiedRoute:
    return UnifiedRoute(
        route_id="source-goal-relation-to-agqa-query-object-v29",
        target_domain=TARGET_DOMAIN,
        target_interface=TARGET_INTERFACE,
        required_capabilities=CAPABILITIES,
        source_program_sha256=source_program_sha256,
        source_program_induced_from_interventions=True,
        source_program_qualified=True,
        target_grounder_sha256=target_grounder_sha256,
        target_executor_sha256=target_executor_sha256,
        target_grounder_id="agqa2.candidate-blind-relation-binding.v29",
        target_executor_id="agqa2.native-object-answer.v29",
        evidence_report_sha256=evidence_report_sha256,
        utility_vs_neural=utility_vs_target_native,
        authenticity_vs_source_permuted=authenticity_vs_effect_shuffled,
    )


def build_harness(
    *, artifact: Mapping[str, Any], confirmation: Mapping[str, Any],
    inducer_artifact_sha256: str, route: UnifiedRoute,
) -> UnifiedNeurosymbolicHarness:
    envelope = source_goal_relation_envelope(
        artifact, confirmation,
        inducer_artifact_sha256=inducer_artifact_sha256,
    )
    return UnifiedNeurosymbolicHarness(
        (envelope,), UnifiedNeurosymbolicTransferRuntime((route,)),
    )


def unified_target_grounding(
    *, artifact: Mapping[str, Any], confirmation: Mapping[str, Any],
    binding: AGQAGoalRelationBindingReceipt,
) -> UnifiedTargetGrounding:
    binding.validate()
    contract = source_goal_relation_contract(artifact, confirmation)
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
        candidate_ids=binding.candidate_bindings,
        structural_predicates={
            "effect_binding_authenticated": binding.effect_binding_authenticated,
            "positive_relation_delta": binding.positive_observed_relation_delta,
            "terminal_predicate_observable": (
                binding.terminal_predicate_observable
            ),
            "unique_target_binding": binding.unique_target_binding,
        },
        grounder_qualified=binding.grounder_qualified,
        formal_outcome_read=binding.formal_outcome_read,
    )
    return UnifiedTargetGrounding.create(
        requirement=requirement, applicability=applicability,
    )


class AGQAObjectExecutor:
    """Emit only the target-native label authorized by the source program."""

    def __init__(self, artifact_sha256: str, candidate: str | None):
        self.artifact_sha256 = artifact_sha256
        self.candidate = candidate
        self.calls = 0

    def execute(
        self, authorization: ExecutionAuthorization,
        grounding: TargetGroundingReceipt, native_actions: Sequence[str],
    ) -> str:
        self.calls += 1
        if self.candidate is None or self.candidate not in native_actions:
            raise ValueError("AGQA executor lacks an authorized native candidate")
        return self.candidate


@dataclass(frozen=True)
class AGQAUnifiedTransferDecision:
    source_candidate: str | None
    binding: AGQAGoalRelationBindingReceipt
    phase7: Phase7Authorization
    utility: ExecutionAuthorization
    executor_calls: int


def decide_source_candidate(
    *, harness: UnifiedNeurosymbolicHarness,
    target: UnifiedTargetGrounding,
    binding: AGQAGoalRelationBindingReceipt,
    target_executor_sha256: str,
) -> AGQAUnifiedTransferDecision:
    """Return an optional source-authorized target label, never a fallback."""

    phase7 = harness.decide(target)
    utility = harness.runtime.decide(target.applicability)
    executor = AGQAObjectExecutor(
        target_executor_sha256, binding.authorized_candidate,
    )
    candidate = SelectiveTargetExecutor.execute(
        utility, target.applicability, tuple(sorted(AGQA_OBJECT_ONTOLOGY)),
        executor,
    ) if phase7.verdict == TransferVerdict.SELECT_SKILL else None
    if (candidate is not None) != (phase7.verdict == TransferVerdict.SELECT_SKILL):
        raise AssertionError("AGQA phase-7 authorization/execution mismatch")
    return AGQAUnifiedTransferDecision(
        source_candidate=candidate,
        binding=binding,
        phase7=phase7,
        utility=utility,
        executor_calls=executor.calls,
    )


__all__ = [
    "CAPABILITIES", "TARGET_DOMAIN", "TARGET_INTERFACE",
    "AGQAGoalRelationBindingReceipt", "AGQAObjectExecutor",
    "AGQAUnifiedTransferDecision", "bind_source_goal_relation_program",
    "build_harness", "build_route", "decide_source_candidate",
    "unified_target_grounding",
]
