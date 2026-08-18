"""Two-stage, operand-isolated neural grounding for AGQA 2.0.

Stage one maps only the public question to an anonymous typed query.  Stage two
grounds each operand independently from chronological video frames.  A source-
induced IR contract controls only the acquisition dynamics (arity and whether
an unresolved operand may be rescanned); it never contributes target labels,
answers, programs, scene graphs, or source-game identity to a neural prompt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence

from .agqa_frame_grounder import (
    AGQAFrameGroundingReceipt,
    COMPARISONS,
    FORBIDDEN_RECEIPT_KEYS,
    parse_frame_grounding_receipt,
)
from .agqa_program_transfer import (
    RELATION_IR,
    RELATION_OPERATOR,
    RELATION_ROUTE,
    TEMPORAL_IR,
    TEMPORAL_PAIR_OPERATOR,
    TEMPORAL_PAIR_ROUTE,
    TEMPORAL_SINGLE_OPERATOR,
    TEMPORAL_SINGLE_ROUTE,
)
from .contracts import stable_hash
from .structural_ir_applicability import SourceIRContract


TARGET_INTERFACE = "question_then_operand_isolated_active_frame_grounder_v3"
ROUTES = (RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE)
OBSERVABILITY = ("OBSERVED", "PARTIAL", "UNOBSERVED")
ROLES = ("A", "B")
ISOLATION_FORBIDDEN_KEYS = frozenset({
    "competing_operand", "other_operand", "full_question", "question",
})


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.casefold() in FORBIDDEN_RECEIPT_KEYS | ISOLATION_FORBIDDEN_KEYS:
                paths.append(path)
            paths.extend(_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


def _text(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise ValueError(f"{field} must be non-empty")
    return result


def _stem_visible_word(word: str) -> str:
    """Normalize simple surface inflections without adding semantics."""

    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        if stem in {"mak", "tak", "giv", "clos", "us"}:
            stem += "e"
        return stem
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    return word


def _words(value: str) -> set[str]:
    return {
        _stem_visible_word(word)
        for word in re.findall(r"[a-z0-9]+", value.casefold())
        if word not in {"a", "an", "the", "person", "someone", "object", "unknown"}
    }


@dataclass(frozen=True)
class AGQAQueryPlan:
    obligation_kind: str
    comparison: str
    operand_a: str
    operand_b: str
    visual_query_a: str
    visual_query_b: str
    parser_uncertainties: tuple[str, ...]
    question_read: bool
    answer_read: bool
    functional_program_read: bool
    scene_graph_grounding_read: bool
    source_identity_read: bool
    plan_sha256: str

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["parser_uncertainties"] = list(self.parser_uncertainties)
        return row


def parse_query_plan(payload: Mapping[str, Any]) -> AGQAQueryPlan:
    """Validate the text-only parsing stage and reject annotation leakage."""

    forbidden = _forbidden_paths(payload)
    if forbidden:
        raise ValueError("query plan contains forbidden fields: " + ", ".join(forbidden))
    kind = _text(payload.get("obligation_kind"), field="obligation_kind")
    if kind not in ROUTES:
        raise ValueError("unsupported query-plan obligation_kind")
    comparison = _text(payload.get("comparison"), field="comparison")
    expected = {
        RELATION_ROUTE: {"EXISTS", "QUERY_OBJECT", "CHOOSE_OBJECT"},
        TEMPORAL_PAIR_ROUTE: {"BEFORE_AFTER"},
        TEMPORAL_SINGLE_ROUTE: {
            "SELECT_LONGER", "SELECT_SHORTER", "VERIFY_A_LONGER",
            "VERIFY_A_SHORTER",
        },
    }
    if comparison not in COMPARISONS or comparison not in expected[kind]:
        raise ValueError("comparison is incompatible with query-plan type")
    need_b = kind != RELATION_ROUTE or comparison == "CHOOSE_OBJECT"
    operand_a = _text(payload.get("operand_a"), field="operand_a")
    operand_b = _text(
        payload.get("operand_b"), field="operand_b", allow_empty=not need_b,
    )
    visual_a = _text(payload.get("visual_query_a"), field="visual_query_a")
    visual_need_b = kind != RELATION_ROUTE
    visual_b = _text(
        payload.get("visual_query_b"), field="visual_query_b",
        allow_empty=not visual_need_b,
    )
    if comparison == "QUERY_OBJECT" and (
        not _words(operand_a) or not _words(operand_a) <= _words(visual_a)
    ):
        raise ValueError(
            "QUERY_OBJECT operand_a and visual_query_a must preserve a non-generic "
            "requested relation phrase"
        )
    if comparison == "CHOOSE_OBJECT" and (
        "unknown object" not in visual_a.casefold()
        or visual_b
        or _words(operand_a) & _words(visual_a)
        or _words(operand_b) & _words(visual_a)
    ):
        raise ValueError(
            "CHOOSE_OBJECT must use one candidate-blind unknown-object relation scan"
        )
    if kind in {TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE}:
        words_a, words_b = _words(operand_a), _words(operand_b)
        visual_words_a, visual_words_b = _words(visual_a), _words(visual_b)
        aligned = len(words_a & visual_words_a) + len(words_b & visual_words_b)
        crossed = len(words_a & visual_words_b) + len(words_b & visual_words_a)
        if (
            not words_a & visual_words_a
            or not words_b & visual_words_b
            or aligned <= crossed
        ):
            raise ValueError(
                "temporal visual queries must preserve operand identity and order"
            )
    uncertainties = payload.get("parser_uncertainties")
    if not isinstance(uncertainties, list) or not all(
        isinstance(value, str) for value in uncertainties
    ):
        raise ValueError("parser_uncertainties must be a string list")
    core = {
        "obligation_kind": kind,
        "comparison": comparison,
        "operand_a": operand_a,
        "operand_b": operand_b,
        "visual_query_a": visual_a,
        "visual_query_b": visual_b,
        "parser_uncertainties": [value.strip() for value in uncertainties],
        "question_read": True,
        "answer_read": False,
        "functional_program_read": False,
        "scene_graph_grounding_read": False,
        "source_identity_read": False,
    }
    return AGQAQueryPlan(
        obligation_kind=kind,
        comparison=comparison,
        operand_a=operand_a,
        operand_b=operand_b,
        visual_query_a=visual_a,
        visual_query_b=visual_b,
        parser_uncertainties=tuple(core["parser_uncertainties"]),
        question_read=True,
        answer_read=False,
        functional_program_read=False,
        scene_graph_grounding_read=False,
        source_identity_read=False,
        plan_sha256=stable_hash(core),
    )


_FINITE_TO_VISIBLE = {
    "close": "closing", "closed": "closing", "consume": "consuming",
    "consumed": "consuming", "dress": "dressing", "dressed": "dressing",
    "eat": "eating", "gave": "giving", "give": "giving", "go": "going",
    "went": "going", "hold": "holding", "held": "holding",
    "interact": "interacting", "interacted": "interacting",
    "laugh": "laughing", "laughed": "laughing", "make": "making",
    "made": "making", "open": "opening", "opened": "opening",
    "put": "putting", "ran": "running", "run": "running",
    "sit": "sitting", "sat": "sitting", "smile": "smiling",
    "smiled": "smiling", "stand": "standing", "stood": "standing",
    "take": "taking", "took": "taking", "throw": "throwing",
    "threw": "throwing", "tidy": "tidying", "tidied": "tidying",
    "undress": "undressing", "undressed": "undressing",
    "watch": "watching", "watched": "watching",
}


def _visible_event_phrase(value: str) -> str:
    phrase = re.sub(r"\s+", " ", value.strip().casefold())
    phrase = re.sub(r"^(?:they|the person)\s+", "", phrase)
    phrase = re.sub(r"\s+something they did$", "", phrase)
    words = phrase.split(maxsplit=1)
    if not words:
        return ""
    first = _FINITE_TO_VISIBLE.get(words[0], words[0])
    return first + ((" " + words[1]) if len(words) == 2 else "")


def _safe_public_plan(payload: Mapping[str, Any]) -> AGQAQueryPlan | None:
    try:
        return parse_query_plan(payload)
    except (TypeError, ValueError):
        return None


def parse_public_question_plan(question: str) -> AGQAQueryPlan | None:
    """Parse the explicit, operand-complete AGQA public-question subset.

    This grammar intentionally abstains on global/open-ended questions such as
    "What were they doing for the most time?" because their operands are not
    present in the public question.  It never consults an answer, program, or
    scene graph.
    """

    text = re.sub(r"\s+", " ", question.strip()).rstrip("?").strip()
    lower = text.casefold()

    match = re.match(
        r"^(?:was|were|did) (?:the person|they) (.+?) before or after "
        r"(?:they |the person )?(.+)$",
        lower,
    )
    if match:
        operand_a = _visible_event_phrase(match.group(1))
        operand_b = _visible_event_phrase(match.group(2))
        return _safe_public_plan({
            "obligation_kind": TEMPORAL_PAIR_ROUTE,
            "comparison": "BEFORE_AFTER",
            "operand_a": operand_a,
            "operand_b": operand_b,
            "visual_query_a": f"a person {operand_a}",
            "visual_query_b": f"a person {operand_b}",
            "parser_uncertainties": [],
        })

    match = re.match(
        r"^which did they do for (longer|shorter), (.+?) or (.+)$", lower,
    )
    if match:
        operand_a = _visible_event_phrase(match.group(2))
        operand_b = _visible_event_phrase(match.group(3))
        return _safe_public_plan({
            "obligation_kind": TEMPORAL_SINGLE_ROUTE,
            "comparison": (
                "SELECT_LONGER" if match.group(1) == "longer"
                else "SELECT_SHORTER"
            ),
            "operand_a": operand_a,
            "operand_b": operand_b,
            "visual_query_a": f"a person {operand_a}",
            "visual_query_b": f"a person {operand_b}",
            "parser_uncertainties": [],
        })

    match = re.match(
        r"^compared to (.+?), did they (.+?) for (longer|shorter)$", lower,
    )
    if match:
        operand_a = _visible_event_phrase(match.group(2))
        operand_b = _visible_event_phrase(match.group(1))
        return _safe_public_plan({
            "obligation_kind": TEMPORAL_SINGLE_ROUTE,
            "comparison": (
                "VERIFY_A_LONGER" if match.group(3) == "longer"
                else "VERIFY_A_SHORTER"
            ),
            "operand_a": operand_a,
            "operand_b": operand_b,
            "visual_query_a": f"a person {operand_a}",
            "visual_query_b": f"a person {operand_b}",
            "parser_uncertainties": [],
        })

    match = re.match(
        r"^did the person spend a (shorter|longer) amount of time (.+?) "
        r"than they spent (.+)$",
        lower,
    )
    if match:
        operand_a = _visible_event_phrase(match.group(2))
        operand_b = _visible_event_phrase(match.group(3))
        return _safe_public_plan({
            "obligation_kind": TEMPORAL_SINGLE_ROUTE,
            "comparison": (
                "VERIFY_A_SHORTER" if match.group(1) == "shorter"
                else "VERIFY_A_LONGER"
            ),
            "operand_a": operand_a,
            "operand_b": operand_b,
            "visual_query_a": f"a person {operand_a}",
            "visual_query_b": f"a person {operand_b}",
            "parser_uncertainties": [],
        })

    match = re.match(
        r"^(?:in the video, )?which object were they (.+)$", lower,
    ) or re.match(r"^(?:in the video, )?what was the person (.+)$", lower)
    if match:
        relation = _visible_event_phrase(match.group(1))
        return _safe_public_plan({
            "obligation_kind": RELATION_ROUTE,
            "comparison": "QUERY_OBJECT",
            "operand_a": relation,
            "operand_b": "",
            "visual_query_a": f"a person {relation} an unknown object",
            "visual_query_b": "",
            "parser_uncertainties": [],
        })

    match = re.match(
        r"^(?:in the video, )?was (.+?) or (.+?) the thing they (.+)$", lower,
    )
    if match:
        candidate_a, candidate_b = match.group(1), match.group(2)
        relation = _visible_event_phrase(match.group(3))
        return _safe_public_plan({
            "obligation_kind": RELATION_ROUTE,
            "comparison": "CHOOSE_OBJECT",
            "operand_a": candidate_a,
            "operand_b": candidate_b,
            "visual_query_a": f"a person {relation} an unknown object",
            "visual_query_b": "",
            "parser_uncertainties": [],
        })

    match = re.match(
        r"^(?:in the video, )?(?:was|were|did) (?:the person|they) (.+)$",
        lower,
    )
    if match:
        relation = _visible_event_phrase(match.group(1))
        return _safe_public_plan({
            "obligation_kind": RELATION_ROUTE,
            "comparison": "EXISTS",
            "operand_a": relation,
            "operand_b": "",
            "visual_query_a": f"a person {relation}",
            "visual_query_b": "",
            "parser_uncertainties": [],
        })
    return None


@dataclass(frozen=True)
class AGQAOperandObservation:
    occurrence_id: str
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
class AGQAOperandReceipt:
    operand_role: str
    requested_operand: str
    observations: tuple[AGQAOperandObservation, ...]
    coverage: str
    uncertainties: tuple[str, ...]
    canonicalizations: tuple[str, ...]
    frame_count: int
    answer_read: bool
    competing_operand_read: bool
    question_read: bool
    functional_program_read: bool
    scene_graph_grounding_read: bool
    source_identity_read: bool
    receipt_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "operand_role": self.operand_role,
            "requested_operand": self.requested_operand,
            "observations": [row.as_dict() for row in self.observations],
            "coverage": self.coverage,
            "uncertainties": list(self.uncertainties),
            "canonicalizations": list(self.canonicalizations),
            "frame_count": self.frame_count,
            "answer_read": self.answer_read,
            "competing_operand_read": self.competing_operand_read,
            "question_read": self.question_read,
            "functional_program_read": self.functional_program_read,
            "scene_graph_grounding_read": self.scene_graph_grounding_read,
            "source_identity_read": self.source_identity_read,
            "receipt_sha256": self.receipt_sha256,
        }


def _frame(value: Any, *, field: str, frame_count: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer or null")
    result = int(value)
    if result != value or not 0 <= result < frame_count:
        raise ValueError(f"{field} is outside the proxy-frame range")
    return result


def parse_operand_receipt(
    payload: Mapping[str, Any], *, expected_role: str,
    expected_operand: str, frame_count: int,
) -> AGQAOperandReceipt:
    """Validate one isolated visual operand receipt."""

    if expected_role not in ROLES:
        raise ValueError("invalid expected operand role")
    if frame_count < 2:
        raise ValueError("operand grounding requires at least two frames")
    forbidden = _forbidden_paths(payload)
    if forbidden:
        raise ValueError(
            "operand receipt contains forbidden fields: " + ", ".join(forbidden)
        )
    role = _text(payload.get("operand_role"), field="operand_role")
    if role != expected_role:
        raise ValueError("operand receipt role does not match isolated request")
    requested = _text(payload.get("requested_operand"), field="requested_operand")
    if requested.casefold() != expected_operand.strip().casefold():
        raise ValueError("operand receipt changed the isolated request")
    raw_rows = payload.get("observations")
    if not isinstance(raw_rows, list) or not 1 <= len(raw_rows) <= 4:
        raise ValueError("observations must contain one to four rows")
    observations: list[AGQAOperandObservation] = []
    raw_canonicalizations = payload.get("canonicalizations", [])
    if not isinstance(raw_canonicalizations, list) or not all(
        isinstance(value, str) for value in raw_canonicalizations
    ):
        raise ValueError("operand canonicalizations must be a string list")
    canonicalizations = [value.strip() for value in raw_canonicalizations]
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise ValueError("operand observation must be an object")
        occurrence_id = _text(raw.get("occurrence_id"), field="occurrence_id")
        if occurrence_id != f"O{index}":
            raise ValueError("occurrence IDs must be consecutive O0,O1,...")
        observability = _text(raw.get("observability"), field="observability")
        if observability not in OBSERVABILITY:
            raise ValueError("invalid operand observability")
        start = _frame(raw.get("start_frame"), field="start_frame", frame_count=frame_count)
        end = _frame(raw.get("end_frame"), field="end_frame", frame_count=frame_count)
        if (start is None) != (end is None):
            raise ValueError("operand interval endpoints must both be set or null")
        if start is not None and end is not None and start > end:
            raise ValueError("operand interval endpoints must be chronological")
        raw_evidence = raw.get("evidence_frames")
        if not isinstance(raw_evidence, list) or len(raw_evidence) > 4:
            raise ValueError("evidence_frames must be a list of at most four indices")
        evidence: list[int] = []
        for value in raw_evidence:
            parsed = _frame(value, field="evidence_frames", frame_count=frame_count)
            assert parsed is not None
            if parsed not in evidence:
                evidence.append(parsed)
        if evidence != sorted(evidence):
            raise ValueError("evidence frames must be chronological")
        if start is not None and any(value < start or value > end for value in evidence):
            raise ValueError("evidence frames must lie inside the claimed interval")
        if observability == "OBSERVED" and not evidence:
            raise ValueError("OBSERVED operand occurrences require pixel evidence")
        if observability == "UNOBSERVED" and (start is not None or evidence):
            if start is not None and end is not None and evidence:
                # A provider occasionally emits an internally contradictory enum
                # while still citing pixels.  Keep the pixels but downgrade the
                # claim to non-decisive PARTIAL; never promote it to OBSERVED.
                observability = "PARTIAL"
                marker = f"{occurrence_id}:DOWNGRADED_CONTRADICTORY_UNOBSERVED_TO_PARTIAL"
                if marker not in canonicalizations:
                    canonicalizations.append(marker)
            else:
                raise ValueError("UNOBSERVED occurrences cannot claim an interval alone")
        confidence = float(raw.get("confidence", -1))
        if not 0 <= confidence <= 1:
            raise ValueError("operand confidence must be in [0,1]")
        uncertainties = raw.get("uncertainties")
        if not isinstance(uncertainties, list) or not all(
            isinstance(value, str) for value in uncertainties
        ):
            raise ValueError("observation uncertainties must be a string list")
        observations.append(AGQAOperandObservation(
            occurrence_id=occurrence_id,
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
    coverage = _text(payload.get("coverage"), field="coverage")
    if coverage not in {"SUFFICIENT", "PARTIAL", "INSUFFICIENT"}:
        raise ValueError("invalid operand coverage")
    uncertainties = payload.get("uncertainties")
    if not isinstance(uncertainties, list) or not all(
        isinstance(value, str) for value in uncertainties
    ):
        raise ValueError("operand uncertainties must be a string list")
    core = {
        "operand_role": role,
        "requested_operand": requested,
        "observations": [row.as_dict() for row in observations],
        "coverage": coverage,
        "uncertainties": [value.strip() for value in uncertainties],
        "canonicalizations": canonicalizations,
        "frame_count": frame_count,
        "answer_read": False,
        "competing_operand_read": False,
        "question_read": False,
        "functional_program_read": False,
        "scene_graph_grounding_read": False,
        "source_identity_read": False,
    }
    return AGQAOperandReceipt(
        operand_role=role,
        requested_operand=requested,
        observations=tuple(observations),
        coverage=coverage,
        uncertainties=tuple(core["uncertainties"]),
        canonicalizations=tuple(canonicalizations),
        frame_count=frame_count,
        answer_read=False,
        competing_operand_read=False,
        question_read=False,
        functional_program_read=False,
        scene_graph_grounding_read=False,
        source_identity_read=False,
        receipt_sha256=stable_hash(core),
    )


@dataclass(frozen=True)
class AGQASourceAcquisitionController:
    obligation_kind: str
    required_operands: int
    recurrent: bool
    maximum_rescans_per_operand: int
    anonymous_source_contract_sha256: str
    anonymous_source_program_sha256: str
    controller_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def source_controller_for_plan(
    plan: AGQAQueryPlan, sources: Sequence[SourceIRContract],
) -> AGQASourceAcquisitionController:
    """Instantiate acquisition dynamics from an exact anonymous source type."""

    if plan.obligation_kind == RELATION_ROUTE:
        expected = (RELATION_IR, (RELATION_OPERATOR,), True, 1)
    elif plan.obligation_kind == TEMPORAL_PAIR_ROUTE:
        expected = (TEMPORAL_IR, (TEMPORAL_PAIR_OPERATOR,), True, 2)
    else:
        expected = (TEMPORAL_IR, (TEMPORAL_SINGLE_OPERATOR,), False, 2)
    matches = [
        source for source in sources
        if source.source_intervention_qualified
        and source.ir_kind == expected[0]
        and source.operator_sequence == expected[1]
        and source.recurrent is expected[2]
    ]
    if len(matches) != 1:
        raise ValueError(
            "query plan needs exactly one qualified anonymous source contract; "
            f"found {len(matches)}"
        )
    source = matches[0]
    core = {
        "obligation_kind": plan.obligation_kind,
        "required_operands": expected[3],
        "recurrent": source.recurrent,
        "maximum_rescans_per_operand": 1 if source.recurrent else 0,
        "anonymous_source_contract_sha256": source.contract_sha256,
        "anonymous_source_program_sha256": source.program_sha256,
    }
    return AGQASourceAcquisitionController(**core, controller_sha256=stable_hash(core))


def operand_needs_rescan(
    receipt: AGQAOperandReceipt, *, controller: AGQASourceAcquisitionController,
    confidence_threshold: float, require_specific_object: bool = False,
) -> bool:
    if not controller.recurrent or controller.maximum_rescans_per_operand < 1:
        return False
    useful = [
        row for row in receipt.observations
        if row.observability == "OBSERVED"
        and row.confidence >= confidence_threshold
        and bool(row.evidence_frames)
        and bool(row.object.strip() or row.predicate.strip())
        and (
            not require_specific_object
            or row.object.strip().casefold()
            not in {"", "object", "unknown", "unknown object", "item", "thing"}
        )
    ]
    return receipt.coverage != "SUFFICIENT" or not useful


def specific_object_grounded(receipt: AGQAOperandReceipt) -> bool:
    generic = {
        "", "object", "unknown", "unknown object", "an unknown object",
        "item", "thing",
    }
    return any(
        row.object.strip().casefold() not in generic
        for row in receipt.observations
        if row.observability in {"OBSERVED", "PARTIAL"}
    )


def recurrent_rescan_window(
    receipt: AGQAOperandReceipt, *, seconds: Sequence[float], duration: float,
    require_specific_object: bool = False,
) -> tuple[float, float]:
    """Select a local evidence window, or re-explore globally when ungrounded."""

    indices = [
        index for row in receipt.observations
        for index in (
            list(row.evidence_frames)
            + ([row.start_frame] if row.start_frame is not None else [])
            + ([row.end_frame] if row.end_frame is not None else [])
        )
    ]
    if not indices:
        return 0.0, duration
    low, high = min(indices), max(indices)
    if require_specific_object and not specific_object_grounded(receipt):
        # A failed object localization is weak evidence about time as well as
        # identity, so the single recurrent intervention re-explores globally.
        return 0.0, duration
    padding = max(3, math.ceil(len(seconds) * 0.06))
    low, high = max(0, low - padding), min(len(seconds) - 1, high + padding)
    return float(seconds[low]), float(seconds[high])


def remap_operand_receipt(
    receipt: AGQAOperandReceipt, *, local_seconds: Sequence[float],
    global_seconds: Sequence[float],
) -> AGQAOperandReceipt:
    """Map a zoomed rescan's local frame IDs back to dense global frame IDs."""

    if len(local_seconds) != receipt.frame_count:
        raise ValueError("local timestamp count does not match operand receipt")
    if len(global_seconds) < 2:
        raise ValueError("global timeline is too short")

    def nearest(local_index: int | None) -> int | None:
        if local_index is None:
            return None
        second = float(local_seconds[local_index])
        return min(
            range(len(global_seconds)),
            key=lambda index: abs(float(global_seconds[index]) - second),
        )

    payload = {
        "operand_role": receipt.operand_role,
        "requested_operand": receipt.requested_operand,
        "observations": [],
        "coverage": receipt.coverage,
        "uncertainties": list(receipt.uncertainties),
        "canonicalizations": list(receipt.canonicalizations),
    }
    for row in receipt.observations:
        start = nearest(row.start_frame)
        end = nearest(row.end_frame)
        evidence = sorted({
            nearest(value) for value in row.evidence_frames
        } - {None})
        if start is not None and evidence:
            start = min(start, *evidence)
            end = max(end, *evidence) if end is not None else max(evidence)
        payload["observations"].append({
            **row.as_dict(),
            "start_frame": start,
            "end_frame": end,
            "evidence_frames": evidence,
        })
    return parse_operand_receipt(
        payload,
        expected_role=receipt.operand_role,
        expected_operand=receipt.requested_operand,
        frame_count=len(global_seconds),
    )


def choose_operand_receipt(
    primary: AGQAOperandReceipt, rescan: AGQAOperandReceipt | None,
) -> AGQAOperandReceipt:
    """Choose a frozen receipt without outcome access."""

    if rescan is None:
        return primary
    coverage_rank = {"INSUFFICIENT": 0, "PARTIAL": 1, "SUFFICIENT": 2}

    def score(receipt: AGQAOperandReceipt) -> tuple[int, int, float]:
        observed = [
            row for row in receipt.observations
            if row.observability == "OBSERVED" and row.evidence_frames
        ]
        return (
            coverage_rank[receipt.coverage],
            len(observed),
            max((row.confidence for row in observed), default=0.0),
        )
    return rescan if score(rescan) > score(primary) else primary


def reconcile_recurrent_receipts(
    primary: AGQAOperandReceipt, rescan: AGQAOperandReceipt | None,
) -> AGQAOperandReceipt:
    """Require recurrent scans to agree instead of selecting a contradiction."""

    if rescan is None:
        return primary

    def observed(receipt: AGQAOperandReceipt) -> bool:
        return any(
            row.observability == "OBSERVED"
            and row.confidence >= 0.5
            and bool(row.evidence_frames)
            for row in receipt.observations
        )

    primary_observed, rescan_observed = observed(primary), observed(rescan)
    if primary_observed and rescan_observed:
        return choose_operand_receipt(primary, rescan)

    if primary_observed != rescan_observed:
        grounded = primary if primary_observed else rescan
        payload = {
            "operand_role": grounded.operand_role,
            "requested_operand": grounded.requested_operand,
            "observations": [
                row.as_dict() | {"observability": "PARTIAL"}
                for row in grounded.observations
            ],
            "coverage": "PARTIAL",
            "uncertainties": list(grounded.uncertainties) + [
                "primary/rescan observability conflict"
            ],
            "canonicalizations": list(grounded.canonicalizations) + [
                "RECURRENT_OBSERVABILITY_CONFLICT_DOWNGRADED_TO_PARTIAL"
            ],
        }
        return parse_operand_receipt(
            payload,
            expected_role=grounded.operand_role,
            expected_operand=grounded.requested_operand,
            frame_count=grounded.frame_count,
        )

    if all(
        row.observability == "UNOBSERVED"
        for receipt in (primary, rescan)
        for row in receipt.observations
    ):
        payload = {
            "operand_role": rescan.operand_role,
            "requested_operand": rescan.requested_operand,
            "observations": [row.as_dict() for row in rescan.observations],
            "coverage": "SUFFICIENT",
            "uncertainties": list(rescan.uncertainties),
            "canonicalizations": list(rescan.canonicalizations) + [
                "RECURRENT_DOUBLE_SCAN_CONFIRMED_UNOBSERVED"
            ],
        }
        return parse_operand_receipt(
            payload,
            expected_role=rescan.operand_role,
            expected_operand=rescan.requested_operand,
            frame_count=rescan.frame_count,
        )
    return choose_operand_receipt(primary, rescan)


def merge_operand_receipts(
    plan: AGQAQueryPlan, *, operand_a: AGQAOperandReceipt,
    operand_b: AGQAOperandReceipt | None, frame_count: int,
) -> AGQAFrameGroundingReceipt:
    """Create the existing unified typed receipt from isolated operands."""

    if operand_a.operand_role != "A":
        raise ValueError("operand A receipt has wrong role")
    if plan.obligation_kind != RELATION_ROUTE:
        if operand_b is None or operand_b.operand_role != "B":
            raise ValueError("this query plan requires an operand B receipt")
    elif operand_b is not None:
        raise ValueError("single-operand relation plan cannot receive operand B")
    rows = []
    all_receipts = [operand_a] + ([operand_b] if operand_b is not None else [])
    unresolved_query_object = (
        plan.comparison == "QUERY_OBJECT"
        and any(
            observation.observability != "OBSERVED"
            or observation.object.strip().casefold() in {
                "", "object", "unknown", "unknown object", "an unknown object",
                "item", "thing",
            }
            for observation in operand_a.observations
        )
    )
    for receipt in all_receipts:
        assert receipt is not None
        for observation in receipt.observations[:3]:
            observability = observation.observability
            role = receipt.operand_role
            if plan.comparison == "CHOOSE_OBJECT":
                grounded = observation.object.strip().casefold()
                candidate_a = plan.operand_a.strip().casefold()
                candidate_b = plan.operand_b.strip().casefold()
                if grounded == candidate_a and grounded != candidate_b:
                    role = "A"
                elif grounded == candidate_b and grounded != candidate_a:
                    role = "B"
                else:
                    observability = "PARTIAL"
            if (
                plan.comparison == "QUERY_OBJECT"
                and (
                    unresolved_query_object
                    or observation.object.strip().casefold()
                    in {"", "object", "unknown", "unknown object", "an unknown object", "item", "thing"}
                )
            ):
                # Pixel evidence may show a relation without identifying its object.
                # Preserve that evidence as PARTIAL, but never turn a placeholder noun
                # into a target decision.
                observability = "PARTIAL"
            rows.append({
                "event_id": f"E{len(rows)}",
                "operand_role": role,
                "label": observation.label,
                "subject": observation.subject,
                "predicate": observation.predicate,
                "object": observation.object,
                "observability": observability,
                "start_frame": observation.start_frame,
                "end_frame": observation.end_frame,
                "evidence_frames": list(observation.evidence_frames)[:3],
                "confidence": observation.confidence,
                "uncertainties": list(observation.uncertainties),
            })
    required = [operand_a] + ([operand_b] if operand_b is not None else [])
    if all(receipt is not None and receipt.coverage == "SUFFICIENT" for receipt in required):
        coverage = "SUFFICIENT"
    elif any(receipt is not None and receipt.coverage != "INSUFFICIENT" for receipt in required):
        coverage = "PARTIAL"
    else:
        coverage = "INSUFFICIENT"
    return parse_frame_grounding_receipt({
        "obligation_kind": plan.obligation_kind,
        "comparison": plan.comparison,
        "operand_a": plan.operand_a,
        "operand_b": plan.operand_b,
        "events": rows,
        "coverage": coverage,
        "uncertainties": list(plan.parser_uncertainties) + [
            value for receipt in required if receipt is not None
            for value in receipt.uncertainties
        ],
        "canonicalizations": [
            marker for receipt in required if receipt is not None
            for marker in receipt.canonicalizations
        ],
    }, frame_count=frame_count)


__all__ = [
    "AGQAOperandObservation", "AGQAOperandReceipt", "AGQAQueryPlan",
    "AGQASourceAcquisitionController", "TARGET_INTERFACE",
    "choose_operand_receipt", "merge_operand_receipts", "operand_needs_rescan",
    "parse_operand_receipt", "parse_public_question_plan", "parse_query_plan",
    "reconcile_recurrent_receipts", "recurrent_rescan_window",
    "remap_operand_receipt", "source_controller_for_plan",
    "specific_object_grounded",
]
