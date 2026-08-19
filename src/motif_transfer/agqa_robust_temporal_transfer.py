"""Robust post-ground transfer of a source-induced temporal-pair program.

The neural grounder is allowed to produce several interval hypotheses for
each operand.  The symbolic program may execute only when recurrent grounding
confirms each operand and *every* cross-view interval pairing entails the same
strict Allen-style BEFORE/AFTER relation.  No best-looking interval, direct
answer, functional program, or current outcome enters this decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .structural_ir_applicability import (
    SourceIRContract,
    TargetIRRequirement,
)
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
TARGET_INTERFACE = "robust_temporal_pair_binding_v33"
CAPABILITIES = (
    "recurrent_interval_hypothesis_set",
    "strict_all_pairs_temporal_relation",
    "target_native_before_after_executor",
)
VIEW_FIELDS = (
    ("primary", "primary_receipt"),
    ("rescan", "rescan_receipt_global_timeline"),
    ("tiebreak", "tiebreak_receipt_global_timeline"),
)
NATIVE_RELATIONS = ("after", "before")


@dataclass(frozen=True)
class TemporalIntervalHypothesis:
    view: str
    start_frame: int
    end_frame: int
    confidence: float
    grounding_receipt_sha256: str


@dataclass(frozen=True)
class AGQARobustTemporalBindingReceipt:
    task_id: str
    target_state_sha256: str
    target_grounder_sha256: str
    source_program_sha256: str
    operand_a_hypotheses: tuple[TemporalIntervalHypothesis, ...]
    operand_b_hypotheses: tuple[TemporalIntervalHypothesis, ...]
    cross_view_relations: tuple[str, ...]
    resolved_relation: str | None
    recurrent_operand_a_confirmed: bool
    recurrent_operand_b_confirmed: bool
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
            raise ValueError("AGQA robust temporal binding hash mismatch")
        if self.formal_outcome_read and self.authorized_relation is not None:
            raise ValueError("current AGQA outcome authorized temporal transfer")
        if self.authorized_relation not in {None, *NATIVE_RELATIONS}:
            raise ValueError("non-native AGQA temporal relation escaped")
        for hypotheses in (
            self.operand_a_hypotheses, self.operand_b_hypotheses,
        ):
            views = [row.view for row in hypotheses]
            if len(views) != len(set(views)):
                raise ValueError("duplicate temporal grounding view")
            for row in hypotheses:
                if row.start_frame > row.end_frame:
                    raise ValueError("reversed temporal interval")


def _receipt_integrity(receipt: Mapping[str, Any]) -> bool:
    return all(
        receipt.get(field) is False
        for field in (
            "answer_read", "functional_program_read",
            "scene_graph_grounding_read", "source_identity_read",
            "question_read", "competing_operand_read",
        )
        if field in receipt
    )


def _unique_observed_hypothesis(
    *, view: str, receipt: Mapping[str, Any] | None,
    minimum_confidence: float,
) -> TemporalIntervalHypothesis | None:
    if not isinstance(receipt, Mapping) or not _receipt_integrity(receipt):
        return None
    observations = [
        row for row in receipt.get("observations") or ()
        if row.get("observability") == "OBSERVED"
        and float(row.get("confidence", -1.0)) >= minimum_confidence
        and isinstance(row.get("evidence_frames"), list)
        and bool(row["evidence_frames"])
        and row.get("start_frame") is not None
        and row.get("end_frame") is not None
    ]
    if len(observations) != 1:
        return None
    row = observations[0]
    start, end = int(row["start_frame"]), int(row["end_frame"])
    if start < 0 or end < start:
        return None
    return TemporalIntervalHypothesis(
        view=view,
        start_frame=start,
        end_frame=end,
        confidence=float(row["confidence"]),
        grounding_receipt_sha256=str(receipt.get("receipt_sha256") or ""),
    )


def _operand_hypotheses(
    operand_run: Mapping[str, Any], *, minimum_confidence: float,
) -> tuple[TemporalIntervalHypothesis, ...]:
    hypotheses = []
    for view, field in VIEW_FIELDS:
        hypothesis = _unique_observed_hypothesis(
            view=view, receipt=operand_run.get(field),
            minimum_confidence=minimum_confidence,
        )
        if hypothesis is not None:
            hypotheses.append(hypothesis)
    return tuple(hypotheses)


def _relation(
    left: TemporalIntervalHypothesis,
    right: TemporalIntervalHypothesis,
) -> str | None:
    if left.end_frame < right.start_frame:
        return "before"
    if right.end_frame < left.start_frame:
        return "after"
    return None


def bind_robust_temporal_pair_program(
    *, task_id: str, target_state_sha256: str,
    target_grounder_sha256: str, source_program_sha256: str,
    obligation_kind: str, operand_runs: Mapping[str, Any],
    grounder_qualified: bool, effect_binding_authenticated: bool = True,
    minimum_confidence: float = 0.5, formal_outcome_read: bool = False,
) -> AGQARobustTemporalBindingReceipt:
    """Resolve a set-valued target binding before symbolic execution."""

    operand_a = _operand_hypotheses(
        operand_runs.get("A") or {}, minimum_confidence=minimum_confidence,
    )
    operand_b = _operand_hypotheses(
        operand_runs.get("B") or {}, minimum_confidence=minimum_confidence,
    )
    recurrent_a = len(operand_a) >= 2
    recurrent_b = len(operand_b) >= 2
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
    elif not recurrent_a or not recurrent_b:
        reason = "SOURCE_ABSTAIN_RECURRENCE_NOT_CONFIRMED"
    elif not separated:
        reason = "SOURCE_ABSTAIN_INTERVAL_HYPOTHESES_OVERLAP"
    elif not consistent:
        reason = "SOURCE_ABSTAIN_TEMPORAL_RELATION_CONFLICT"
    else:
        authorized = resolved
        reason = "ROBUST_ALL_PAIRS_TEMPORAL_RELATION_RESOLVED"
    body = {
        "task_id": str(task_id),
        "target_state_sha256": str(target_state_sha256),
        "target_grounder_sha256": str(target_grounder_sha256),
        "source_program_sha256": str(source_program_sha256),
        "operand_a_hypotheses": [asdict(row) for row in operand_a],
        "operand_b_hypotheses": [asdict(row) for row in operand_b],
        "cross_view_relations": relations,
        "resolved_relation": resolved,
        "recurrent_operand_a_confirmed": recurrent_a,
        "recurrent_operand_b_confirmed": recurrent_b,
        "all_pairs_strictly_separated": separated,
        "all_pairs_relation_consistent": consistent,
        "effect_binding_authenticated": bool(effect_binding_authenticated),
        "grounder_qualified": bool(grounder_qualified),
        "formal_outcome_read": bool(formal_outcome_read),
        "authorized_relation": authorized,
        "reason": reason,
    }
    receipt = AGQARobustTemporalBindingReceipt(
        task_id=body["task_id"],
        target_state_sha256=body["target_state_sha256"],
        target_grounder_sha256=body["target_grounder_sha256"],
        source_program_sha256=body["source_program_sha256"],
        operand_a_hypotheses=operand_a,
        operand_b_hypotheses=operand_b,
        cross_view_relations=relations,
        resolved_relation=resolved,
        recurrent_operand_a_confirmed=recurrent_a,
        recurrent_operand_b_confirmed=recurrent_b,
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


def build_temporal_route(
    *, source_program_sha256: str, target_grounder_sha256: str,
    target_executor_sha256: str, evidence_report_sha256: str,
    utility_vs_target_native: PairedCalibration,
    authenticity_vs_effect_shuffled: PairedCalibration,
) -> UnifiedRoute:
    return UnifiedRoute(
        route_id="source-temporal-pair-to-agqa-before-after-v33",
        target_domain=TARGET_DOMAIN,
        target_interface=TARGET_INTERFACE,
        required_capabilities=CAPABILITIES,
        source_program_sha256=source_program_sha256,
        source_program_induced_from_interventions=True,
        source_program_qualified=True,
        target_grounder_sha256=target_grounder_sha256,
        target_executor_sha256=target_executor_sha256,
        target_grounder_id="agqa2.robust-interval-set-grounder.v33",
        target_executor_id="agqa2.native-before-after-executor.v33",
        evidence_report_sha256=evidence_report_sha256,
        utility_vs_neural=utility_vs_target_native,
        authenticity_vs_source_permuted=authenticity_vs_effect_shuffled,
    )


def build_temporal_harness(
    *, contract: SourceIRContract, source_transition_receipts_sha256: str,
    inducer_artifact_sha256: str, route: UnifiedRoute,
) -> UnifiedNeurosymbolicHarness:
    envelope = InducedProgramEnvelope.create(
        contract=contract,
        source_transition_receipts_sha256=(
            source_transition_receipts_sha256
        ),
        inducer_artifact_sha256=inducer_artifact_sha256,
    )
    return UnifiedNeurosymbolicHarness(
        (envelope,), UnifiedNeurosymbolicTransferRuntime((route,)),
    )


def unified_temporal_grounding(
    *, contract: SourceIRContract,
    binding: AGQARobustTemporalBindingReceipt,
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
            (binding.resolved_relation,)
            if binding.resolved_relation is not None else ()
        ),
        structural_predicates={
            "all_pairs_relation_consistent": (
                binding.all_pairs_relation_consistent
            ),
            "all_pairs_strictly_separated": (
                binding.all_pairs_strictly_separated
            ),
            "effect_binding_authenticated": (
                binding.effect_binding_authenticated
            ),
            "recurrent_operand_a_confirmed": (
                binding.recurrent_operand_a_confirmed
            ),
            "recurrent_operand_b_confirmed": (
                binding.recurrent_operand_b_confirmed
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


class AGQATemporalExecutor:
    """Emit only a source-authorized target-native temporal relation."""

    def __init__(self, artifact_sha256: str, relation: str | None):
        self.artifact_sha256 = artifact_sha256
        self.relation = relation
        self.calls = 0

    def execute(
        self, authorization: ExecutionAuthorization,
        grounding: TargetGroundingReceipt, native_actions: Sequence[str],
    ) -> str:
        self.calls += 1
        if self.relation is None or self.relation not in native_actions:
            raise ValueError("AGQA temporal executor lacks authorized relation")
        return self.relation


@dataclass(frozen=True)
class AGQATemporalTransferDecision:
    source_relation: str | None
    binding: AGQARobustTemporalBindingReceipt
    phase7: Phase7Authorization
    utility: ExecutionAuthorization
    executor_calls: int


def decide_temporal_relation(
    *, harness: UnifiedNeurosymbolicHarness,
    target: UnifiedTargetGrounding,
    binding: AGQARobustTemporalBindingReceipt,
    target_executor_sha256: str,
) -> AGQATemporalTransferDecision:
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
        raise AssertionError("AGQA temporal authorization/execution mismatch")
    return AGQATemporalTransferDecision(
        source_relation=relation,
        binding=binding,
        phase7=phase7,
        utility=utility,
        executor_calls=executor.calls,
    )


__all__ = [
    "AGQARobustTemporalBindingReceipt", "AGQATemporalExecutor",
    "AGQATemporalTransferDecision", "CAPABILITIES", "NATIVE_RELATIONS",
    "TARGET_DOMAIN", "TARGET_INTERFACE", "TemporalIntervalHypothesis",
    "bind_robust_temporal_pair_program", "build_temporal_harness",
    "build_temporal_route", "decide_temporal_relation",
    "unified_temporal_grounding",
]
