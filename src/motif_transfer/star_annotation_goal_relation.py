"""Annotation-assisted STAR adapter for the source-induced relation program.

This adapter is intentionally a *development preflight*.  It reuses frozen
same-frame STAR direct/proof receipts and the public STAR functional program;
it never consumes a gold answer while constructing a grounding receipt.  The
functional program is used only to establish that the target query is an
action/entity relation query.  Neural typed-proof statuses supply the actual
target binding.

The transferred object is exactly the template-free recurrent
``UPDATE ENTITY_GOAL_RELATION / RELATION_COVERAGE`` program already induced
from Sokoban state/action/effect/next-state tuples.  STAR-specific code may
bind and execute that anonymous program, but may not rewrite it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .clevrer_unified_goal_relation import (
    IR_KIND,
    source_goal_relation_contract,
    source_goal_relation_envelope,
)
from .contracts import stable_hash
from .natural_video_recovery import PROOF_KINDS
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
    TargetGroundingReceipt,
    TransferVerdict,
    UnifiedNeurosymbolicTransferRuntime,
    UnifiedRoute,
)


TARGET_DOMAIN = "star"
TARGET_INTERFACE = "annotation_assisted_action_relation_recovery_v38"
CAPABILITIES = (
    "same_model_same_frames_direct_proof",
    "star_functional_action_relation_program",
    "typed_neural_candidate_relations",
    "unique_native_policy_binding",
)
NATIVE_ACTIONS = ("uniform_direct", "uniform_typed_proof")
REQUIRED_RELATION_KINDS = (
    "ENTITY_GROUNDING",
    "EVENT_OCCURRENCE",
    "ANSWER_ENTAILMENT",
)
QUERY_FUNCTIONS = {"Query_Objs", "Query_Verbs", "Query_Actions"}


@dataclass(frozen=True)
class RelationCoverageReceipt:
    task_id: str
    required_relation_kinds: tuple[str, ...]
    direct_answer: str
    proof_answer: str
    direct_coverage: float
    proof_coverage: float
    observed_relation_delta: float
    recurrent_update_count: int
    terminal_relation_coverage: bool
    unique_native_policy_binding: bool
    functional_program_supported: bool
    binding_rotation: int
    official_functional_program_read: bool
    gold_or_formal_outcome_read: bool
    receipt_sha256: str


def _candidate_map(
    proof: Mapping[str, Any], *, binding_rotation: int,
) -> dict[str, Mapping[str, Any]]:
    candidates = list(proof.get("candidates") or ())
    slots = [str(row.get("slot")) for row in candidates]
    if len(slots) < 2 or len(slots) != len(set(slots)):
        raise ValueError("STAR typed proof requires distinct candidate slots")
    if set(PROOF_KINDS) != {
        str(step.get("kind"))
        for row in candidates for step in row.get("proof_steps") or ()
    }:
        raise ValueError("STAR typed proof kind schema drift")
    offset = int(binding_rotation) % len(candidates)
    rotated = candidates[offset:] + candidates[:offset]
    return {slot: row for slot, row in zip(slots, rotated)}


def _supported_program(question_program: Sequence[Mapping[str, Any]]) -> bool:
    functions = [str(row.get("function")) for row in question_program]
    return (
        functions.count("Situations") >= 1
        and functions.count("Actions") >= 1
        and sum(value in QUERY_FUNCTIONS for value in functions) == 1
    )


def _coverage(
    candidate: Mapping[str, Any],
) -> tuple[float, bool, frozenset[str]]:
    steps = {
        str(row.get("kind")): row for row in candidate.get("proof_steps") or ()
    }
    if not set(REQUIRED_RELATION_KINDS) <= set(steps):
        raise ValueError("STAR proof cannot ground required action relations")
    supported = frozenset(
        kind for kind in REQUIRED_RELATION_KINDS
        if str(steps[kind].get("status")) == "SUPPORTED"
    )
    coverage = sum(
        float(steps[kind].get("confidence", 0.0))
        for kind in supported
    ) / len(REQUIRED_RELATION_KINDS)
    return coverage, len(supported) == len(REQUIRED_RELATION_KINDS), supported


def relation_coverage_receipt(
    *, task_id: str, direct: Mapping[str, Any], proof: Mapping[str, Any],
    question_program: Sequence[Mapping[str, Any]], binding_rotation: int = 0,
) -> RelationCoverageReceipt:
    """Ground neural proof statuses without accepting an answer label."""

    direct_answer = str(direct.get("answer") or "")
    proof_answer = str(proof.get("answer") or "")
    candidates = _candidate_map(proof, binding_rotation=binding_rotation)
    if direct_answer not in candidates or proof_answer not in candidates:
        raise ValueError("STAR neural answers are outside native candidate slots")
    direct_coverage, _, direct_supported = _coverage(candidates[direct_answer])
    proof_coverage, terminal, proof_supported = _coverage(candidates[proof_answer])
    newly_covered = proof_supported - direct_supported
    body = {
        "task_id": str(task_id),
        "required_relation_kinds": REQUIRED_RELATION_KINDS,
        "direct_answer": direct_answer,
        "proof_answer": proof_answer,
        "direct_coverage": direct_coverage,
        "proof_coverage": proof_coverage,
        "observed_relation_delta": proof_coverage - direct_coverage,
        "recurrent_update_count": len(newly_covered),
        "terminal_relation_coverage": terminal,
        "unique_native_policy_binding": direct_answer != proof_answer,
        "functional_program_supported": _supported_program(question_program),
        "binding_rotation": int(binding_rotation),
        "official_functional_program_read": True,
        "gold_or_formal_outcome_read": False,
    }
    return RelationCoverageReceipt(
        **body, receipt_sha256=stable_hash(body),
    )


def target_grounding(
    *, contract: SourceIRContract, target_grounder_sha256: str,
    coverage: RelationCoverageReceipt, proof_receipt_sha256: str,
    grounder_qualified: bool = True,
) -> UnifiedTargetGrounding:
    """Bind STAR-native proof relations to the unchanged source contract."""

    requirement = TargetIRRequirement.create(
        task_id=coverage.task_id,
        target_domain=TARGET_DOMAIN,
        target_interface=TARGET_INTERFACE,
        target_grounder_sha256=target_grounder_sha256,
        ir_kind=IR_KIND,
        operator_sequence=contract.operator_sequence,
        recurrent=contract.recurrent,
        terminal_predicate_families=contract.terminal_predicate_families,
        grounder_qualified=grounder_qualified,
        formal_outcome_read=False,
    )
    state_sha256 = stable_hash({
        "task_id": coverage.task_id,
        "proof_receipt_sha256": proof_receipt_sha256,
        "relation_coverage_receipt_sha256": coverage.receipt_sha256,
        "gold_or_formal_outcome": "NOT_READ",
    })
    applicability = TargetGroundingReceipt.create(
        task_id=coverage.task_id,
        target_domain=TARGET_DOMAIN,
        target_interface=TARGET_INTERFACE,
        target_state_sha256=state_sha256,
        target_grounder_sha256=target_grounder_sha256,
        capabilities=CAPABILITIES,
        candidate_ids=NATIVE_ACTIONS,
        structural_predicates={
            "functional_program_supported": coverage.functional_program_supported,
            "positive_relation_delta": coverage.observed_relation_delta > 0.0,
            "recurrent_update_observed": coverage.recurrent_update_count > 0,
            "terminal_relation_coverage": coverage.terminal_relation_coverage,
            "unique_native_policy_binding": coverage.unique_native_policy_binding,
        },
        grounder_qualified=grounder_qualified,
        formal_outcome_read=False,
    )
    return UnifiedTargetGrounding.create(
        requirement=requirement, applicability=applicability,
    )


def build_route(
    *, source_program_sha256: str, target_grounder_sha256: str,
    target_executor_sha256: str, evidence_report_sha256: str,
    utility_vs_neural: PairedCalibration,
    authenticity_vs_source_permuted: PairedCalibration,
) -> UnifiedRoute:
    return UnifiedRoute(
        route_id="sokoban-goal-relation-to-star-action-relation-v38",
        target_domain=TARGET_DOMAIN,
        target_interface=TARGET_INTERFACE,
        required_capabilities=CAPABILITIES,
        source_program_sha256=source_program_sha256,
        source_program_induced_from_interventions=True,
        source_program_qualified=True,
        target_grounder_sha256=target_grounder_sha256,
        target_executor_sha256=target_executor_sha256,
        target_grounder_id="star.annotation-assisted-typed-relation.v38-development",
        target_executor_id="star.uniform-direct-or-proof.v38",
        evidence_report_sha256=evidence_report_sha256,
        utility_vs_neural=utility_vs_neural,
        authenticity_vs_source_permuted=authenticity_vs_source_permuted,
    )


def build_harness(
    envelope: InducedProgramEnvelope, route: UnifiedRoute,
) -> UnifiedNeurosymbolicHarness:
    return UnifiedNeurosymbolicHarness(
        (envelope,), UnifiedNeurosymbolicTransferRuntime((route,)),
    )


class StarPolicyExecutor:
    """The only component permitted to emit a STAR-native policy choice."""

    def __init__(self, artifact_sha256: str):
        self.artifact_sha256 = artifact_sha256
        self.calls = 0

    def execute(
        self, authorization: ExecutionAuthorization,
        grounding: TargetGroundingReceipt,
        native_actions: Sequence[str],
    ) -> str:
        self.calls += 1
        if tuple(native_actions) != NATIVE_ACTIONS:
            raise ValueError("STAR native policy contract drift")
        return "uniform_typed_proof"


@dataclass(frozen=True)
class StarRecoveryDecision:
    selected_native_policy: str
    phase7: Phase7Authorization
    utility: ExecutionAuthorization
    executor_calls: int


def decide_recovery(
    *, harness: UnifiedNeurosymbolicHarness, target: UnifiedTargetGrounding,
    target_executor_sha256: str,
) -> StarRecoveryDecision:
    phase7 = harness.decide(target)
    utility = harness.runtime.decide(target.applicability)
    executor = StarPolicyExecutor(target_executor_sha256)
    action = harness.execute(
        phase7, utility, target, NATIVE_ACTIONS, executor,
    )
    selected = action if action is not None else "uniform_direct"
    if (phase7.verdict == TransferVerdict.SELECT_SKILL) != (action is not None):
        raise AssertionError("STAR authorization/execution mismatch")
    return StarRecoveryDecision(
        selected_native_policy=selected,
        phase7=phase7,
        utility=utility,
        executor_calls=executor.calls,
    )


__all__ = [
    "CAPABILITIES", "NATIVE_ACTIONS", "REQUIRED_RELATION_KINDS",
    "TARGET_DOMAIN", "TARGET_INTERFACE", "RelationCoverageReceipt",
    "StarPolicyExecutor", "StarRecoveryDecision", "build_harness",
    "build_route", "decide_recovery", "relation_coverage_receipt",
    "source_goal_relation_contract", "source_goal_relation_envelope",
    "target_grounding",
]
