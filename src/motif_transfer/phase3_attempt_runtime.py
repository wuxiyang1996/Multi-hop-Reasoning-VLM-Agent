"""Target-agnostic executor for a source-induced anonymous AttemptLedger.

The executor does not know game names, target domains, native actions, or
policy labels.  It accepts a frozen source artifact and a target-native set of
opaque candidate IDs.  The target adapter executes a returned candidate and
grounds its observed effect as HIGH, LOW, or UNKNOWN.  Only the source-induced
operator state deltas decide whether to select another candidate, terminate,
or abstain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .phase3_source_applicability import (
    SourceApplicabilityPrior,
    prior_from_frozen_artifact,
)
from .phase3_source_induction import (
    ACTIVATE_DELTA,
    RELEASE_DELTA,
    SUSPEND_DELTA,
    TERMINATE_DELTA,
    operators_from_program,
    route_state,
    validate_program,
)
from .phase3_typed_effect_induction import (
    target_trial_order,
    validate_typed_effect_program,
)


@dataclass(frozen=True)
class _TypedEffectRuntimePrior:
    """Compatibility view over a V3 content-bound typed-effect program."""

    source_profile_sha256: str
    prior_sha256: str
    selected_effect_type: str


@dataclass(frozen=True)
class RuntimeDecision:
    kind: str
    candidate_id: str | None
    operator_ids: tuple[str, ...]
    state_delta: tuple[tuple[str, str], ...] | None
    reason: str
    receipt_sha256: str

    @classmethod
    def create(
        cls, *, kind: str, candidate_id: str | None,
        operator_ids: Sequence[str],
        state_delta: Sequence[tuple[str, str]] | None, reason: str,
    ) -> "RuntimeDecision":
        body = {
            "kind": str(kind),
            "candidate_id": candidate_id,
            "operator_ids": list(map(str, operator_ids)),
            "state_delta": (
                [list(row) for row in state_delta]
                if state_delta is not None else None
            ),
            "reason": str(reason),
        }
        return cls(
            kind=body["kind"],
            candidate_id=candidate_id,
            operator_ids=tuple(body["operator_ids"]),
            state_delta=(
                tuple((str(row[0]), str(row[1])) for row in body["state_delta"])
                if body["state_delta"] is not None else None
            ),
            reason=body["reason"],
            receipt_sha256=stable_hash(body),
        )

    def validate(self) -> bool:
        body = asdict(self)
        claimed = body.pop("receipt_sha256")
        body["operator_ids"] = list(body["operator_ids"])
        body["state_delta"] = (
            [list(row) for row in body["state_delta"]]
            if body["state_delta"] is not None else None
        )
        return claimed == stable_hash(body)


class AnonymousAttemptRuntime:
    """Execute one frozen anonymous program over target-grounded candidates."""

    def __init__(
        self, *, artifact: Mapping[str, Any], candidate_ids: Sequence[str],
        target_grounding_sha256: str,
        candidate_effects: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        artifact_body = dict(artifact)
        claimed_artifact = str(artifact_body.pop("artifact_sha256", ""))
        if not claimed_artifact or stable_hash(artifact_body) != claimed_artifact:
            raise ValueError("frozen source artifact hash mismatch")
        typed_program = artifact.get("typed_effect_program")
        self._typed_program: Mapping[str, Any] | None = None
        if isinstance(typed_program, Mapping):
            validate_typed_effect_program(typed_program)
            if artifact.get(
                "target_data_read_for_program_induction_or_calibration"
            ) is not False:
                raise ValueError("typed artifact does not attest target-free induction")
            program = artifact.get("anonymous_transition_program")
            selected_effect_type = str(typed_program["selected_effect_type"])
            prior_body = {
                "typed_effect_program_sha256": typed_program["program_sha256"],
                "selected_effect_type": selected_effect_type,
                "score_kind": "ARGMAX_SINGLE_TYPED_EFFECT",
            }
            self.prior = _TypedEffectRuntimePrior(
                source_profile_sha256=str(typed_program["program_sha256"]),
                prior_sha256=stable_hash(prior_body),
                selected_effect_type=selected_effect_type,
            )
            self._typed_program = typed_program
            ids, order, self.applicability_receipt = self._typed_binding(
                candidate_ids=candidate_ids,
                candidate_effects=candidate_effects,
                target_grounding_sha256=target_grounding_sha256,
            )
        else:
            # V1 compatibility is retained for frozen historical controls. New
            # Phase-3 target runs use only the V3 branch above.
            self.prior = prior_from_frozen_artifact(artifact)
            program = artifact.get("authentic_program")
            self.applicability_receipt = self.prior.applicability_receipt(
                target_candidate_ids=candidate_ids,
                target_grounding_sha256=target_grounding_sha256,
            )
        if not isinstance(program, Mapping):
            raise ValueError("frozen source artifact omitted anonymous program")
        validate_program(program)
        self.artifact_sha256 = claimed_artifact
        self.program_sha256 = str(program["program_sha256"])
        self.operators = operators_from_program(program)
        self.candidate_ids = tuple(map(str, candidate_ids))
        self.order = tuple(
            int(value) for value in self.applicability_receipt.get("trial_order", ())
        )
        # Candidate IDs are target-native action identities in the V3 adapter.
        # Tracking IDs rather than list offsets lets the unchanged symbolic
        # ledger consume a newly grounded candidate set after every target
        # transition without retrying an already executed native action.
        self.tried: set[str] = set()
        self.active_rank: int | None = None
        self.active_candidate_id: str | None = None
        self.active_effect = "UNKNOWN"
        self.finished = False
        self._resumed_prefix = False

    def _typed_binding(
        self, *, candidate_ids: Sequence[str],
        candidate_effects: Sequence[Mapping[str, Any]] | None,
        target_grounding_sha256: str,
    ) -> tuple[tuple[str, ...], tuple[int, ...], dict[str, Any]]:
        """Bind a current target candidate set without reading its outcome."""

        if self._typed_program is None:
            raise RuntimeError("dynamic typed binding requires a V3 program")
        ids = tuple(map(str, candidate_ids))
        if len(ids) < 2 or len(set(ids)) != len(ids):
            order, reason = (), "TARGET_CANDIDATE_SET_NOT_MULTIPLE_AND_UNIQUE"
        elif candidate_effects is None or len(candidate_effects) != len(ids):
            order, reason = (), "TARGET_TYPED_EFFECT_SET_MISSING_OR_MISALIGNED"
        else:
            order, reason = target_trial_order(
                self._typed_program, candidate_effects,
            )
        receipt_body = {
            "admitted": reason is None,
            "abstention_reason": reason,
            "source_profile_sha256": self.prior.source_profile_sha256,
            "source_prior_sha256": self.prior.prior_sha256,
            "selected_effect_type": self.prior.selected_effect_type,
            "target_candidate_ids": list(ids),
            "target_grounding_sha256": str(target_grounding_sha256),
            "trial_order": list(order),
            "ordered_target_candidate_ids": [ids[index] for index in order],
            "target_typed_effects_sha256": stable_hash(
                list(candidate_effects or ())
            ),
            "target_outcome_read": False,
        }
        receipt = receipt_body | {
            "applicability_receipt_sha256": stable_hash(receipt_body)
        }
        return ids, tuple(order), receipt

    @property
    def supports_dynamic_rebinding(self) -> bool:
        return self._typed_program is not None

    def rebind_candidates(
        self, *, candidate_ids: Sequence[str],
        candidate_effects: Sequence[Mapping[str, Any]] | None,
        target_grounding_sha256: str,
    ) -> Mapping[str, Any]:
        """Re-ground V3 operands in the current target state.

        The source-induced operator and ledger state remain unchanged.  Only
        opaque target action IDs and target-native neural effect values are
        replaced.  This is required for an MDP: an action valid at the fork is
        not assumed to remain available after a state transition.
        """

        if self.finished:
            raise RuntimeError("cannot rebind a finished attempt runtime")
        ids, order, receipt = self._typed_binding(
            candidate_ids=candidate_ids,
            candidate_effects=candidate_effects,
            target_grounding_sha256=target_grounding_sha256,
        )
        self.candidate_ids = ids
        self.order = order
        self.applicability_receipt = receipt
        return receipt

    @property
    def admitted(self) -> bool:
        return bool(self.applicability_receipt["admitted"])

    def _state(self) -> dict[str, Any]:
        return {
            "active_presence": "ABSENT" if self.active_rank is None else "PRESENT",
            "active_effect": (
                "UNKNOWN" if self.active_rank is None else self.active_effect
            ),
            "has_untried": any(
                candidate_id not in self.tried
                for candidate_id in self.candidate_ids
            ),
            "terminal": False,
            "suspended": False,
        }

    def _route(self):
        return route_state(self.operators, self._state())

    def start(self) -> RuntimeDecision:
        if self.finished:
            raise RuntimeError("attempt runtime is already finished")
        if not self.admitted:
            self.finished = True
            return RuntimeDecision.create(
                kind="ABSTAIN", candidate_id=None, operator_ids=(),
                state_delta=None,
                reason=str(self.applicability_receipt["abstention_reason"]),
            )
        return self._advance()

    def resume_observed_prefix(self, effect: str) -> RuntimeDecision:
        """Route an acquisition-prefix effect before selecting a new trial.

        A target fork may already be the successor of a target-native
        acquisition intervention. Rank ``-1`` represents that observed prefix
        without adding a fabricated candidate to the source prior or target
        candidate set.
        """

        if self.finished or self.active_rank is not None or self.tried:
            raise RuntimeError("attempt runtime cannot resume this state")
        if not self.admitted:
            return self.start()
        normalized = str(effect).upper()
        if normalized not in {"HIGH", "LOW", "UNKNOWN"}:
            raise ValueError("target effect must be HIGH, LOW, or UNKNOWN")
        self.active_rank = -1
        self.active_candidate_id = None
        self.active_effect = normalized
        self._resumed_prefix = True
        return self._advance()

    def observe(self, effect: str) -> RuntimeDecision:
        if self.finished:
            raise RuntimeError("attempt runtime is already finished")
        if self.active_rank is None:
            raise RuntimeError("cannot ground an effect without an active candidate")
        normalized = str(effect).upper()
        if normalized not in {"HIGH", "LOW", "UNKNOWN"}:
            raise ValueError("target effect must be HIGH, LOW, or UNKNOWN")
        self.active_effect = normalized
        return self._advance()

    def _advance(self) -> RuntimeDecision:
        routed: list[str] = []
        # RELEASE is an internal ledger update.  All other induced deltas
        # produce a target-adapter decision or a fail-closed abstention.
        for _ in range(len(self.candidate_ids) * 2 + 4):
            operator = self._route()
            if operator is None:
                self.finished = True
                return RuntimeDecision.create(
                    kind="ABSTAIN", candidate_id=None, operator_ids=routed,
                    state_delta=None, reason="NO_UNIQUE_QUALIFIED_OPERATOR",
                )
            routed.append(operator.operator_id)
            delta = operator.state_delta
            if delta == ACTIVATE_DELTA:
                available = [
                    rank for rank in self.order
                    if self.candidate_ids[rank] not in self.tried
                ]
                if self.active_rank is not None or not available:
                    self.finished = True
                    return RuntimeDecision.create(
                        kind="ABSTAIN", candidate_id=None,
                        operator_ids=routed, state_delta=delta,
                        reason="INDUCED_ACTIVATION_PRECONDITION_VIOLATION",
                    )
                self.active_rank = available[0]
                self.active_candidate_id = self.candidate_ids[self.active_rank]
                self.active_effect = "UNKNOWN"
                self.tried.add(self.active_candidate_id)
                return RuntimeDecision.create(
                    kind="TRIAL",
                    candidate_id=self.active_candidate_id,
                    operator_ids=routed, state_delta=delta,
                    reason="SOURCE_INDUCED_STATE_DELTA",
                )
            if delta == RELEASE_DELTA:
                if self.active_rank is None:
                    self.finished = True
                    return RuntimeDecision.create(
                        kind="ABSTAIN", candidate_id=None,
                        operator_ids=routed, state_delta=delta,
                        reason="INDUCED_RELEASE_PRECONDITION_VIOLATION",
                    )
                self.active_rank = None
                self.active_candidate_id = None
                self.active_effect = "UNKNOWN"
                continue
            if delta == TERMINATE_DELTA:
                self.finished = True
                return RuntimeDecision.create(
                    kind="TERMINATE", candidate_id=self.active_candidate_id,
                    operator_ids=routed, state_delta=delta,
                    reason="SOURCE_INDUCED_STATE_DELTA",
                )
            if delta == SUSPEND_DELTA:
                self.finished = True
                return RuntimeDecision.create(
                    kind="ABSTAIN", candidate_id=None,
                    operator_ids=routed, state_delta=delta,
                    reason="SOURCE_INDUCED_STATE_DELTA",
                )
            self.finished = True
            return RuntimeDecision.create(
                kind="ABSTAIN", candidate_id=None, operator_ids=routed,
                state_delta=delta, reason="UNSUPPORTED_INDUCED_STATE_DELTA",
            )
        self.finished = True
        return RuntimeDecision.create(
            kind="ABSTAIN", candidate_id=None, operator_ids=routed,
            state_delta=None, reason="INTERNAL_LEDGER_UPDATE_BOUND_EXCEEDED",
        )


__all__ = ["AnonymousAttemptRuntime", "RuntimeDecision"]
