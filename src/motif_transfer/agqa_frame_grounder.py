"""Answer-blind frame-grounding contracts for AGQA 2.0 development.

Unlike :mod:`agqa_program_transfer`, this interface never consumes an official
functional program.  A target-native vision model sees only a public question
and chronological proxy frames, predicts the anonymous obligation kind, and
returns typed event observations.  A deterministic executor may consume the
validated receipt after it freezes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
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


def _normalized_answer(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    for prefix in ("the answer is ", "it is ", "they were ", "they are "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    return {"true": "yes", "false": "no"}.get(text, text)


def _answer_equivalent(left: Any, right: Any) -> bool:
    first, second = _normalized_answer(left), _normalized_answer(right)
    if first in {"yes", "no", "before", "after"}:
        return bool(second) and second.split(maxsplit=1)[0] == first
    if second in {"yes", "no", "before", "after"}:
        return bool(first) and first.split(maxsplit=1)[0] == second
    return bool(first) and first == second


def _supported_events(
    receipt: AGQAFrameGroundingReceipt, role: str, *, allow_partial: bool,
) -> list[AGQAEventObservation]:
    allowed = {"OBSERVED", "PARTIAL"} if allow_partial else {"OBSERVED"}
    return [
        event for event in receipt.events
        if event.operand_role == role
        and event.observability in allowed
        and event.confidence >= 0.5
        and event.start_frame is not None
        and event.end_frame is not None
        and bool(event.evidence_frames)
    ]


def _merged_intervals(
    events: Sequence[AGQAEventObservation],
) -> list[tuple[int, int]]:
    intervals = sorted(
        (int(event.start_frame), int(event.end_frame))
        for event in events
        if event.start_frame is not None and event.end_frame is not None
    )
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _duration_decision_from_events(
    receipt: AGQAFrameGroundingReceipt, *, allow_partial: bool,
) -> tuple[str | None, dict[str, Any]]:
    intervals = {
        role: _merged_intervals(_supported_events(
            receipt, role, allow_partial=allow_partial,
        ))
        for role in ("A", "B")
    }
    durations = {
        role: sum(end - start + 1 for start, end in values)
        for role, values in intervals.items()
    }
    if not intervals["A"] or not intervals["B"] or durations["A"] == durations["B"]:
        return None, {"intervals": intervals, "durations": durations}
    a_longer = durations["A"] > durations["B"]
    if receipt.comparison in {"SELECT_LONGER", "SELECT_SHORTER"}:
        choose_a = a_longer
        if receipt.comparison == "SELECT_SHORTER":
            choose_a = not choose_a
        decision = receipt.operand_a if choose_a else receipt.operand_b
    else:
        proposition = a_longer
        if receipt.comparison == "VERIFY_A_SHORTER":
            proposition = not proposition
        decision = "yes" if proposition else "no"
    return decision, {"intervals": intervals, "durations": durations}


def _typed_support_for_direct(
    receipt: AGQAFrameGroundingReceipt, direct_response: str,
) -> tuple[str | None, str]:
    """Return a canonical direct decision only when pixels support its semantics."""

    supported = {
        role: _supported_events(receipt, role, allow_partial=True)
        for role in ("A", "B")
    }
    if receipt.comparison == "BEFORE_AFTER":
        starts = {
            role: min((event.start_frame for event in rows), default=None)
            for role, rows in supported.items()
        }
        if starts["A"] is not None and starts["B"] is not None:
            if abs(int(starts["A"]) - int(starts["B"])) >= 3:
                decision = "before" if starts["A"] < starts["B"] else "after"
                if _answer_equivalent(decision, direct_response):
                    return decision, "DIRECT_ORDER_CORROBORATED_BY_TYPED_INTERVALS"
    elif receipt.comparison == "EXISTS":
        if supported["A"] and _answer_equivalent("yes", direct_response):
            return "yes", "DIRECT_EXISTS_CORROBORATED_BY_PIXEL_EVIDENCE"
        if (
            "RECURRENT_DOUBLE_SCAN_CONFIRMED_UNOBSERVED"
            in receipt.canonicalizations
            and _answer_equivalent("no", direct_response)
        ):
            return "no", "DIRECT_ABSENCE_CORROBORATED_BY_DOUBLE_SCAN"
    elif receipt.comparison in {"QUERY_OBJECT", "CHOOSE_OBJECT"}:
        objects = {
            _normalized_answer(event.object) for event in supported["A"] + supported["B"]
            if _normalized_answer(event.object)
            not in {"object", "unknown", "unknown object", "item", "thing"}
        }
        candidates = (
            {_normalized_answer(receipt.operand_a), _normalized_answer(receipt.operand_b)}
            if receipt.comparison == "CHOOSE_OBJECT" else objects
        )
        supported_candidates = objects & candidates
        if len(supported_candidates) == 1:
            decision = supported_candidates.pop()
            if _answer_equivalent(decision, direct_response):
                return decision, "DIRECT_OBJECT_CORROBORATED_BY_TYPED_RELATION"
    elif receipt.obligation_kind == TEMPORAL_SINGLE_ROUTE:
        decision, _ = _duration_decision_from_events(receipt, allow_partial=True)
        if decision is not None and _answer_equivalent(decision, direct_response):
            return decision, "DIRECT_DURATION_CORROBORATED_BY_TYPED_INTERVALS"
    return None, "DIRECT_RESPONSE_LACKS_TYPED_CORROBORATION"


def calibrate_grounding_execution(
    receipt: AGQAFrameGroundingReceipt, raw_execution: Mapping[str, Any],
    direct_response: str, *, minimum_duration_margin_frames: int = 3,
    minimum_single_interval_nesting_margin_frames: int = 6,
    minimum_repeated_interval_dominance_margin_frames: int = 2,
    minimum_exists_override_confidence: float = 0.8,
    require_globally_separated_order_override: bool = True,
    allow_exists_source_override: bool = True,
    maximum_order_override_events_per_operand: int | None = None,
) -> dict[str, Any]:
    """Fuse two frozen target-native views without labels or benchmark programs.

    Agreement can authorize a decision but cannot improve over the direct view.
    A contradictory typed decision may override direct only when its symbolic
    evidence has an independent topology: local object corroboration, recurrent
    double-scan agreement, or a one-versus-multiple duration structure.
    """

    raw_decision = raw_execution.get("decision")
    decision: str | None = None
    reason = "CALIBRATED_ABSTAIN"
    authorization_class = "ABSTAIN"
    if raw_decision is not None and _answer_equivalent(raw_decision, direct_response):
        decision = str(raw_decision)
        reason = "TYPED_AND_DIRECT_AGREE"
        authorization_class = "AGREEMENT"
    elif raw_decision is None:
        decision, reason = _typed_support_for_direct(receipt, direct_response)
        if decision is not None:
            authorization_class = "DIRECT_CORROBORATED_BY_TYPED_EVIDENCE"
    else:
        markers = receipt.canonicalizations
        override_reason = None
        if receipt.comparison in {"QUERY_OBJECT", "CHOOSE_OBJECT"} and any(
            "PLUS_COCO" in marker
            or marker.startswith("RECURRENT_DOUBLE_SCAN_OBJECT_AGREEMENT:")
            for marker in markers
        ):
            override_reason = "OBJECT_OVERRIDE_WITH_INDEPENDENT_CORROBORATION"
        elif receipt.comparison == "EXISTS" and allow_exists_source_override:
            supported_a = _supported_events(receipt, "A", allow_partial=False)
            observed_override = (
                _answer_equivalent(raw_decision, "yes")
                and "RECURRENT_DOUBLE_SCAN_CONFIRMED_OBSERVED" in markers
                and bool(supported_a)
                and min(event.confidence for event in supported_a)
                >= minimum_exists_override_confidence
            )
            unobserved_override = (
                _answer_equivalent(raw_decision, "no")
                and "RECURRENT_DOUBLE_SCAN_CONFIRMED_UNOBSERVED" in markers
            )
            if observed_override or unobserved_override:
                override_reason = "EXISTS_OVERRIDE_WITH_RECURRENT_AGREEMENT"
        elif receipt.comparison == "BEFORE_AFTER" and all(
            f"RECURRENT_{role}_DOUBLE_SCAN_CONFIRMED_OBSERVED" in markers
            for role in ("A", "B")
        ):
            events_a = _supported_events(receipt, "A", allow_partial=False)
            events_b = _supported_events(receipt, "B", allow_partial=False)
            if events_a and events_b:
                start_a = min(int(event.start_frame) for event in events_a)
                start_b = min(int(event.start_frame) for event in events_b)
                globally_separated_decision = None
                if (
                    max(int(event.end_frame) for event in events_a)
                    < min(int(event.start_frame) for event in events_b)
                ):
                    globally_separated_decision = "before"
                elif (
                    max(int(event.end_frame) for event in events_b)
                    < min(int(event.start_frame) for event in events_a)
                ):
                    globally_separated_decision = "after"
                order_sound = (
                    not require_globally_separated_order_override
                    or (
                        globally_separated_decision is not None
                        and _answer_equivalent(
                            raw_decision, globally_separated_decision,
                        )
                    )
                )
                event_count_sound = (
                    maximum_order_override_events_per_operand is None
                    or (
                        len(events_a) <= maximum_order_override_events_per_operand
                        and len(events_b) <= maximum_order_override_events_per_operand
                    )
                )
                if abs(start_a - start_b) >= 3 and order_sound and event_count_sound:
                    override_reason = "ORDER_OVERRIDE_WITH_DUAL_RECURRENT_AGREEMENT"
        elif receipt.obligation_kind == TEMPORAL_SINGLE_ROUTE:
            typed_decision, topology = _duration_decision_from_events(
                receipt, allow_partial=False,
            )
            interval_counts = {
                role: len(topology["intervals"][role]) for role in ("A", "B")
            }
            margin = abs(topology["durations"]["A"] - topology["durations"]["B"])
            topology_reason = None
            if (
                typed_decision is not None
                and _answer_equivalent(raw_decision, typed_decision)
                and sorted(interval_counts.values())[0] == 1
                and sorted(interval_counts.values())[1] >= 2
                and margin >= minimum_duration_margin_frames
            ):
                topology_reason = "ONE_VS_MULTIPLE_TOPOLOGY"
            elif typed_decision is not None and _answer_equivalent(
                raw_decision, typed_decision,
            ) and interval_counts == {"A": 1, "B": 1}:
                interval_a = topology["intervals"]["A"][0]
                interval_b = topology["intervals"]["B"][0]
                a_contains_b = (
                    interval_a[0] <= interval_b[0]
                    and interval_a[1] >= interval_b[1]
                )
                b_contains_a = (
                    interval_b[0] <= interval_a[0]
                    and interval_b[1] >= interval_a[1]
                )
                boundary_aligned = (
                    abs(interval_a[0] - interval_b[0]) <= 1
                    or abs(interval_a[1] - interval_b[1]) <= 1
                )
                if (
                    a_contains_b != b_contains_a
                    and boundary_aligned
                    and margin >= minimum_single_interval_nesting_margin_frames
                ):
                    topology_reason = "BOUNDARY_ALIGNED_SINGLE_INTERVAL_NESTING"
            elif (
                typed_decision is not None
                and _answer_equivalent(raw_decision, typed_decision)
                and interval_counts["A"] == interval_counts["B"]
                and interval_counts["A"] >= 2
            ):
                intervals_a = topology["intervals"]["A"]
                intervals_b = topology["intervals"]["B"]
                a_dominates = all(
                    a_start <= b_start and a_end >= b_end
                    for (a_start, a_end), (b_start, b_end)
                    in zip(intervals_a, intervals_b)
                )
                b_dominates = all(
                    b_start <= a_start and b_end >= a_end
                    for (a_start, a_end), (b_start, b_end)
                    in zip(intervals_a, intervals_b)
                )
                if (
                    a_dominates != b_dominates
                    and margin >= minimum_repeated_interval_dominance_margin_frames
                ):
                    topology_reason = "ALIGNED_REPEATED_INTERVAL_DOMINANCE"
            if topology_reason is not None:
                override_reason = f"DURATION_OVERRIDE_WITH_{topology_reason}"
        if override_reason is not None:
            decision = str(raw_decision)
            reason = override_reason
            authorization_class = "SOURCE_TYPED_OVERRIDE"

    body = {
        "schema_version": "agqa-calibrated-target-native-execution-v1",
        "status": "CALIBRATED_DECISION" if decision is not None else "ABSTAIN",
        "decision": decision,
        "reason": reason,
        "authorization_class": authorization_class,
        "authorized": decision is not None,
        "changes_direct_response": (
            decision is not None and not _answer_equivalent(decision, direct_response)
        ),
        "grounding_receipt_sha256": receipt.receipt_sha256,
        "raw_execution_sha256": raw_execution.get("execution_sha256"),
        "direct_response_sha256": stable_hash(direct_response),
        "direct_response_read": True,
        "functional_program_read": False,
        "official_answer_read": False,
        "scene_graph_grounding_read": False,
        "source_identity_read": False,
    }
    return body | {"execution_sha256": stable_hash(body)}


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
            elif (
                receipt.comparison == "EXISTS"
                and not observed_a
                and "RECURRENT_DOUBLE_SCAN_CONFIRMED_UNOBSERVED"
                in receipt.canonicalizations
            ):
                decision = "no"
                reason = "QUERY_RELATION_DOUBLE_SCAN_UNOBSERVED"
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
    "OPERAND_ROLES", "TARGET_INTERFACE", "calibrate_grounding_execution",
    "execute_grounding_receipt",
    "parse_frame_grounding_receipt", "select_source_for_grounding",
    "target_requirement_from_grounding",
]
