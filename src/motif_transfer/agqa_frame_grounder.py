"""Answer-blind frame-grounding contracts for AGQA 2.0 development.

Unlike :mod:`agqa_program_transfer`, this interface never consumes an official
functional program.  A target-native vision model sees only a public question
and chronological proxy frames, predicts the anonymous obligation kind, and
returns typed event observations.  A deterministic executor may consume the
validated receipt after it freezes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .agqa_program_transfer import (
    RELATION_IR,
    RELATION_OPERATOR,
    RELATION_ROUTE,
    TARGET_DOMAIN,
    TEMPORAL_IR,
    TEMPORAL_PAIR_OPERATOR,
    TEMPORAL_PAIR_ROUTE,
    TEMPORAL_SINGLE_OPERATOR,
    TEMPORAL_SINGLE_ROUTE,
    UNSUPPORTED_IR,
    UNSUPPORTED_ROUTE,
)
from .contracts import stable_hash
from .structural_ir_applicability import (
    SourceIRContract,
    TargetIRRequirement,
    select_source_contract,
)


TARGET_INTERFACE = "question_plus_frame_typed_event_grounder_v2"
COMPARISONS = (
    "EXISTS", "QUERY_OBJECT", "CHOOSE_OBJECT", "BEFORE_AFTER",
    "SELECT_LONGER", "SELECT_SHORTER", "VERIFY_A_LONGER",
    "VERIFY_A_SHORTER", "UNSUPPORTED",
)
OBSERVABILITY = ("OBSERVED", "PARTIAL", "UNOBSERVED")
OPERAND_ROLES = ("A", "B", "CONTEXT")
GROUNDING_KINDS = (
    RELATION_ROUTE,
    TEMPORAL_PAIR_ROUTE,
    TEMPORAL_SINGLE_ROUTE,
    UNSUPPORTED_ROUTE,
)
FORBIDDEN_RECEIPT_KEYS = frozenset({
    "answer", "answer_slot", "best_answer", "choice", "choice_id",
    "correct", "correctness", "correct_option", "functional_program",
    "gold", "prediction", "program", "selected_operand", "selected_option",
    "sg_grounding", "source_game", "source_identity",
})


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.casefold() in FORBIDDEN_RECEIPT_KEYS:
                paths.append(path)
            paths.extend(_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


def _text(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    output = value.strip()
    if not output and not allow_empty:
        raise ValueError(f"{field} must be non-empty")
    return output


def _frame_index(value: Any, *, field: str, frame_count: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer or null")
    output = int(value)
    if output != value or not 0 <= output < frame_count:
        raise ValueError(f"{field} is outside the proxy-frame range")
    return output


@dataclass(frozen=True)
class AGQAEventObservation:
    event_id: str
    operand_role: str
    label: str
    subject: str
    predicate: str
    object: str
    observability: str
    start_frame: int | None
    end_frame: int | None
    evidence_frames: tuple[int, ...]
    confidence: float
    uncertainties: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["evidence_frames"] = list(self.evidence_frames)
        row["uncertainties"] = list(self.uncertainties)
        return row


@dataclass(frozen=True)
class AGQAFrameGroundingReceipt:
    obligation_kind: str
    comparison: str
    operand_a: str
    operand_b: str
    events: tuple[AGQAEventObservation, ...]
    coverage: str
    uncertainties: tuple[str, ...]
    canonicalizations: tuple[str, ...]
    question_read: bool
    frame_count: int
    answer_read: bool
    functional_program_read: bool
    scene_graph_grounding_read: bool
    source_identity_read: bool
    receipt_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "obligation_kind": self.obligation_kind,
            "comparison": self.comparison,
            "operand_a": self.operand_a,
            "operand_b": self.operand_b,
            "events": [event.as_dict() for event in self.events],
            "coverage": self.coverage,
            "uncertainties": list(self.uncertainties),
            "canonicalizations": list(self.canonicalizations),
            "question_read": self.question_read,
            "frame_count": self.frame_count,
            "answer_read": self.answer_read,
            "functional_program_read": self.functional_program_read,
            "scene_graph_grounding_read": self.scene_graph_grounding_read,
            "source_identity_read": self.source_identity_read,
            "receipt_sha256": self.receipt_sha256,
        }


def parse_frame_grounding_receipt(
    payload: Mapping[str, Any], *, frame_count: int,
) -> AGQAFrameGroundingReceipt:
    """Validate a provider receipt and reject annotation/answer leakage."""

    if frame_count < 2:
        raise ValueError("AGQA grounding requires at least two proxy frames")
    forbidden = _forbidden_paths(payload)
    if forbidden:
        raise ValueError(
            "AGQA grounding receipt contains forbidden fields: "
            + ", ".join(forbidden)
        )
    kind = _text(payload.get("obligation_kind"), field="obligation_kind")
    if kind not in GROUNDING_KINDS:
        raise ValueError("invalid AGQA obligation_kind")
    comparison = _text(payload.get("comparison"), field="comparison")
    if comparison not in COMPARISONS:
        raise ValueError("invalid AGQA comparison")
    expected_comparisons = {
        RELATION_ROUTE: {"EXISTS", "QUERY_OBJECT", "CHOOSE_OBJECT"},
        TEMPORAL_PAIR_ROUTE: {"BEFORE_AFTER"},
        TEMPORAL_SINGLE_ROUTE: {
            "SELECT_LONGER", "SELECT_SHORTER", "VERIFY_A_LONGER",
            "VERIFY_A_SHORTER",
        },
        UNSUPPORTED_ROUTE: {"UNSUPPORTED"},
    }
    if comparison not in expected_comparisons[kind]:
        raise ValueError("comparison is incompatible with obligation_kind")
    operand_a = _text(
        payload.get("operand_a"), field="operand_a",
        allow_empty=(kind == UNSUPPORTED_ROUTE),
    )
    operand_b = _text(
        payload.get("operand_b"), field="operand_b",
        allow_empty=(
            kind == UNSUPPORTED_ROUTE
            or (kind == RELATION_ROUTE and comparison != "CHOOSE_OBJECT")
        ),
    )

    rows = payload.get("events")
    if not isinstance(rows, list) or len(rows) > 6:
        raise ValueError("events must be a list of at most six observations")
    if kind != UNSUPPORTED_ROUTE and not rows:
        raise ValueError("a supported obligation needs at least one event row")
    events: list[AGQAEventObservation] = []
    raw_canonicalizations = payload.get("canonicalizations", [])
    if not isinstance(raw_canonicalizations, list) or not all(
        isinstance(value, str) for value in raw_canonicalizations
    ):
        raise ValueError("canonicalizations must be a string list when present")
    canonicalizations = [value.strip() for value in raw_canonicalizations]
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError("event rows must be objects")
        event_id = _text(raw.get("event_id"), field="event_id")
        if event_id != f"E{index}":
            raise ValueError("event IDs must be consecutive E0,E1,...")
        role = _text(raw.get("operand_role"), field="operand_role")
        if role not in OPERAND_ROLES:
            raise ValueError("invalid event operand_role")
        observability = _text(raw.get("observability"), field="observability")
        if observability not in OBSERVABILITY:
            raise ValueError("invalid event observability")
        start = _frame_index(
            raw.get("start_frame"), field="start_frame", frame_count=frame_count,
        )
        end = _frame_index(
            raw.get("end_frame"), field="end_frame", frame_count=frame_count,
        )
        if (start is None) != (end is None):
            raise ValueError("start_frame and end_frame must both be set or null")
        if start is not None and end is not None and start > end:
            start, end = end, start
            normalization = f"{event_id}:SWAPPED_REVERSED_INTERVAL_ENDPOINTS"
            if normalization not in canonicalizations:
                canonicalizations.append(normalization)
        raw_evidence = raw.get("evidence_frames")
        if not isinstance(raw_evidence, list) or len(raw_evidence) > 3:
            raise ValueError("evidence_frames must contain at most three indices")
        evidence: list[int] = []
        for value in raw_evidence:
            frame = _frame_index(
                value, field="evidence_frames", frame_count=frame_count,
            )
            assert frame is not None
            if frame not in evidence:
                evidence.append(frame)
        if evidence != sorted(evidence):
            raise ValueError("evidence_frames must be chronological")
        if start is not None and any(
            frame < start or frame > end for frame in evidence
        ):
            start = min(start, *evidence)
            end = max(end, *evidence)
            normalization = f"{event_id}:EXPANDED_INTERVAL_TO_COVER_EVIDENCE"
            if normalization not in canonicalizations:
                canonicalizations.append(normalization)
        if observability == "OBSERVED" and not evidence:
            raise ValueError("OBSERVED events require pixel evidence")
        if observability == "UNOBSERVED" and (
            start is not None or evidence
        ):
            raise ValueError("UNOBSERVED events cannot claim an interval/evidence")
        confidence = float(raw.get("confidence", -1.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("event confidence must be in [0,1]")
        uncertainties = raw.get("uncertainties")
        if not isinstance(uncertainties, list) or not all(
            isinstance(value, str) for value in uncertainties
        ):
            raise ValueError("event uncertainties must be a string list")
        events.append(AGQAEventObservation(
            event_id=event_id,
            operand_role=role,
            label=_text(raw.get("label"), field="label"),
            subject=_text(raw.get("subject"), field="subject", allow_empty=True),
            predicate=_text(raw.get("predicate"), field="predicate"),
            object=_text(raw.get("object"), field="object", allow_empty=True),
            observability=observability,
            start_frame=start,
            end_frame=end,
            evidence_frames=tuple(evidence),
            confidence=confidence,
            uncertainties=tuple(value.strip() for value in uncertainties),
        ))
    if kind in {TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE} and not {
        "A", "B"
    } <= {event.operand_role for event in events}:
        raise ValueError("temporal obligations require A and B event rows")
    coverage = _text(payload.get("coverage"), field="coverage")
    if coverage not in {"SUFFICIENT", "PARTIAL", "INSUFFICIENT"}:
        raise ValueError("invalid grounding coverage")
    uncertainties = payload.get("uncertainties")
    if not isinstance(uncertainties, list) or not all(
        isinstance(value, str) for value in uncertainties
    ):
        raise ValueError("grounding uncertainties must be a string list")
    core = {
        "obligation_kind": kind,
        "comparison": comparison,
        "operand_a": operand_a,
        "operand_b": operand_b,
        "events": [event.as_dict() for event in events],
        "coverage": coverage,
        "uncertainties": [value.strip() for value in uncertainties],
        "canonicalizations": canonicalizations,
        "question_read": True,
        "frame_count": frame_count,
        "answer_read": False,
        "functional_program_read": False,
        "scene_graph_grounding_read": False,
        "source_identity_read": False,
    }
    return AGQAFrameGroundingReceipt(
        obligation_kind=kind,
        comparison=comparison,
        operand_a=operand_a,
        operand_b=operand_b,
        events=tuple(events),
        coverage=coverage,
        uncertainties=tuple(core["uncertainties"]),
        canonicalizations=tuple(canonicalizations),
        question_read=True,
        frame_count=frame_count,
        answer_read=False,
        functional_program_read=False,
        scene_graph_grounding_read=False,
        source_identity_read=False,
        receipt_sha256=stable_hash(core),
    )


def target_requirement_from_grounding(
    *, task_id: str, receipt: AGQAFrameGroundingReceipt,
    target_grounder_sha256: str, grounder_qualified: bool,
) -> TargetIRRequirement:
    """Bind a frozen frame receipt to the same anonymous IR as other targets."""

    if receipt.obligation_kind == RELATION_ROUTE:
        ir_kind = RELATION_IR
        operators = (RELATION_OPERATOR,)
        recurrent = True
        terminal = ("ENTITY_GOAL_RELATION",)
    elif receipt.obligation_kind == TEMPORAL_PAIR_ROUTE:
        ir_kind = TEMPORAL_IR
        operators = (TEMPORAL_PAIR_OPERATOR,)
        recurrent = True
        terminal = ()
    elif receipt.obligation_kind == TEMPORAL_SINGLE_ROUTE:
        ir_kind = TEMPORAL_IR
        operators = (TEMPORAL_SINGLE_OPERATOR,)
        recurrent = False
        terminal = ()
    else:
        ir_kind = UNSUPPORTED_IR
        operators = ()
        recurrent = False
        terminal = ()
    return TargetIRRequirement.create(
        task_id=task_id,
        target_domain=TARGET_DOMAIN,
        target_interface=TARGET_INTERFACE,
        target_grounder_sha256=target_grounder_sha256,
        ir_kind=ir_kind,
        operator_sequence=operators,
        recurrent=recurrent,
        terminal_predicate_families=terminal,
        grounder_qualified=grounder_qualified,
        formal_outcome_read=False,
    )


def select_source_for_grounding(
    sources: Sequence[SourceIRContract], *, task_id: str,
    receipt: AGQAFrameGroundingReceipt, target_grounder_sha256: str,
    grounder_qualified: bool,
) -> dict[str, Any]:
    requirement = target_requirement_from_grounding(
        task_id=task_id,
        receipt=receipt,
        target_grounder_sha256=target_grounder_sha256,
        grounder_qualified=grounder_qualified,
    )
    return select_source_contract(sources, requirement)


def _merged_duration(events: Sequence[AGQAEventObservation]) -> int | None:
    intervals = sorted(
        (event.start_frame, event.end_frame)
        for event in events
        if event.observability == "OBSERVED"
        and event.confidence >= 0.5
        and event.start_frame is not None
        and event.end_frame is not None
    )
    if not intervals:
        return None
    merged: list[list[int]] = []
    for start, end in intervals:
        assert start is not None and end is not None
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start + 1 for start, end in merged)


def execute_grounding_receipt(
    receipt: AGQAFrameGroundingReceipt,
) -> dict[str, Any]:
    """Produce a target-native decision without reading an official answer."""

    by_role = {
        role: [event for event in receipt.events if event.operand_role == role]
        for role in OPERAND_ROLES
    }
    decision: str | None = None
    reason = "INSUFFICIENT_TYPED_VISUAL_EVIDENCE"
    if receipt.coverage != "INSUFFICIENT":
        if receipt.obligation_kind == RELATION_ROUTE:
            observed_a = [
                event for event in by_role["A"]
                if event.observability == "OBSERVED" and event.confidence >= 0.5
            ]
            observed_b = [
                event for event in by_role["B"]
                if event.observability == "OBSERVED" and event.confidence >= 0.5
            ]
            if receipt.comparison == "EXISTS" and observed_a:
                decision = "yes"
                reason = "QUERY_RELATION_OBSERVED"
            elif receipt.comparison == "QUERY_OBJECT":
                objects = {
                    event.object.strip() for event in observed_a
                    if event.object.strip()
                }
                if len(objects) == 1:
                    decision = objects.pop()
                    reason = "UNIQUE_OBSERVED_RELATION_OBJECT"
            elif receipt.comparison == "CHOOSE_OBJECT":
                if bool(observed_a) != bool(observed_b):
                    decision = (
                        receipt.operand_a if observed_a else receipt.operand_b
                    )
                    reason = "UNIQUE_OBSERVED_CHOICE_RELATION"
        elif receipt.obligation_kind == TEMPORAL_PAIR_ROUTE:
            a_starts = [
                event.start_frame for event in by_role["A"]
                if event.observability == "OBSERVED"
                and event.confidence >= 0.5
                and event.start_frame is not None
            ]
            b_starts = [
                event.start_frame for event in by_role["B"]
                if event.observability == "OBSERVED"
                and event.confidence >= 0.5
                and event.start_frame is not None
            ]
            if a_starts and b_starts and min(a_starts) != min(b_starts):
                decision = "before" if min(a_starts) < min(b_starts) else "after"
                reason = "OBSERVED_EVENT_ORDER"
        elif receipt.obligation_kind == TEMPORAL_SINGLE_ROUTE:
            duration_a = _merged_duration(by_role["A"])
            duration_b = _merged_duration(by_role["B"])
            if (
                duration_a is not None and duration_b is not None
                and duration_a != duration_b
            ):
                a_longer = duration_a > duration_b
                if receipt.comparison in {"SELECT_LONGER", "SELECT_SHORTER"}:
                    choose_a = a_longer
                    if receipt.comparison == "SELECT_SHORTER":
                        choose_a = not choose_a
                    decision = receipt.operand_a if choose_a else receipt.operand_b
                    reason = "OBSERVED_DURATION_SELECTION"
                else:
                    proposition = a_longer
                    if receipt.comparison == "VERIFY_A_SHORTER":
                        proposition = not proposition
                    decision = "yes" if proposition else "no"
                    reason = "OBSERVED_DURATION_VERIFICATION"
    body = {
        "schema_version": "agqa-target-native-execution-v2",
        "status": "TARGET_NATIVE_DECISION" if decision else "ABSTAIN",
        "decision": decision,
        "reason": reason,
        "grounding_receipt_sha256": receipt.receipt_sha256,
        "functional_program_read": False,
        "official_answer_read": False,
        "source_identity_read": False,
    }
    return body | {"execution_sha256": stable_hash(body)}


__all__ = [
    "AGQAEventObservation", "AGQAFrameGroundingReceipt", "COMPARISONS",
    "FORBIDDEN_RECEIPT_KEYS", "GROUNDING_KINDS", "OBSERVABILITY",
    "OPERAND_ROLES", "TARGET_INTERFACE", "execute_grounding_receipt",
    "parse_frame_grounding_receipt", "select_source_for_grounding",
    "target_requirement_from_grounding",
]
