"""Public-question compiler for temporal-localized AGQA object queries.

The target-native compiler exposes a composition boundary, not a new source
skill: a temporal window is grounded first and a recurrent relation is then
grounded inside that window.  The two primitive acquisition/abstention
contracts must be supplied by independently qualified source programs.
Neither parsing nor execution accepts an AGQA answer, functional program, or
scene graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from statistics import median_low, median_high
from typing import Any, Mapping, Sequence

from .agqa_program_transfer import (
    RELATION_IR,
    RELATION_OPERATOR,
    TEMPORAL_IR,
    TEMPORAL_PAIR_OPERATOR,
)
from .agqa_query_object_grounder import (
    AGQA_OBJECT_ONTOLOGY,
    canonical_object_label,
)
from .contracts import stable_hash
from .structural_ir_applicability import SourceIRContract


TARGET_INTERFACE = "public_question_temporal_window_then_relation_v58"
TEMPORAL_OPERATORS = frozenset({"BEFORE", "AFTER", "WHILE", "BETWEEN"})


_FINITE_TO_VISIBLE = {
    "carried": "carrying", "carry": "carrying", "closed": "closing",
    "did": "doing", "held": "holding", "hold": "holding",
    "interacted": "interacting", "interact": "interacting",
    "leaned": "leaning", "lean": "leaning", "looked": "looking",
    "look": "looking", "opened": "opening", "open": "opening",
    "sat": "sitting", "sit": "sitting", "stood": "standing",
    "stand": "standing", "took": "taking", "take": "taking",
    "touched": "touching", "touch": "touching", "watched": "watching",
    "watch": "watching", "went": "going", "go": "going",
}

_CANONICAL_RELATIONS = frozenset({
    "above", "behind", "beneath", "carrying", "closing", "covered by",
    "dressing", "eating", "fixing", "grasping", "holding", "in",
    "in front of", "laughing", "leaning on", "lying on", "on the side of",
    "opening", "playing on", "putting down", "sitting on", "smiling",
    "snuggling", "standing on", "taking", "throwing", "tidying",
    "touching", "twisting", "undressing", "washing", "watching", "wearing",
    "wiping", "working on", "writing on",
})

_RELATION_SURFACE_PREFIXES = {
    "carried": "carrying", "carry": "carrying", "closed": "closing",
    "close": "closing", "dressed": "dressing", "dress": "dressing",
    "ate": "eating", "eat": "eating", "fixed": "fixing", "fix": "fixing",
    "grasped": "grasping", "grasp": "grasping", "held": "holding",
    "hold": "holding", "laughed": "laughing", "laugh": "laughing",
    "leaned": "leaning", "lean": "leaning", "lay": "lying",
    "lie": "lying", "opened": "opening", "open": "opening",
    "played": "playing", "play": "playing", "put": "putting",
    "sat": "sitting", "sit": "sitting", "smiled": "smiling",
    "smile": "smiling", "snuggled": "snuggling", "snuggle": "snuggling",
    "stood": "standing", "stand": "standing", "took": "taking",
    "take": "taking", "threw": "throwing", "throw": "throwing",
    "tidied": "tidying", "tidy": "tidying", "touched": "touching",
    "touch": "touching", "twisted": "twisting", "twist": "twisting",
    "undressed": "undressing", "undress": "undressing", "washed": "washing",
    "wash": "washing", "watched": "watching", "watch": "watching",
    "wore": "wearing", "wear": "wearing", "wiped": "wiping",
    "wipe": "wiping", "worked": "working", "work": "working",
    "wrote": "writing", "write": "writing",
}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _visible_phrase(value: str) -> str:
    phrase = _clean(value)
    phrase = re.sub(r"^(?:they|the person)\s+", "", phrase)
    phrase = re.sub(r"^(?:starting|started) to\s+", "", phrase)
    words = phrase.split(maxsplit=1)
    if not words:
        return ""
    first = _FINITE_TO_VISIBLE.get(words[0], words[0])
    if first.endswith("ing"):
        return first + ((" " + words[1]) if len(words) == 2 else "")
    if first.endswith("e"):
        first = first[:-1] + "ing"
    elif not first.endswith("ing"):
        first += "ing"
    return first + ((" " + words[1]) if len(words) == 2 else "")


def _relation_phrase(value: str) -> str:
    phrase = _clean(value)
    phrase = re.sub(r"^(?:they|the person)\s+", "", phrase)
    phrase = re.sub(
        r"^(?:went|go|going)\s+(?=(?:above|behind|beneath|in|"
        r"in front of|on the side of)$)",
        "", phrase,
    )
    if phrase in _CANONICAL_RELATIONS:
        return phrase
    words = phrase.split(maxsplit=1)
    if not words:
        return ""
    first = _RELATION_SURFACE_PREFIXES.get(words[0], words[0])
    candidate = first + ((" " + words[1]) if len(words) == 2 else "")
    return candidate if candidate in _CANONICAL_RELATIONS else ""


def _relation_from_query_clause(value: str) -> str | None:
    clause = _clean(value).strip(" ,")
    patterns = (
        r"^(?:which|what) object did (?:the person|they) (.+)$",
        r"^(?:which|what) object (?:was|were) (?:the person|they) (.+)$",
        r"^what did (?:the person|they) (.+)$",
        r"^what (?:was|were) (?:the person|they) (.+)$",
        r"^which thing did (?:the person|they) (.+)$",
        r"^which thing (?:was|were) (?:the person|they) (.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, clause)
        if match:
            relation = _relation_phrase(match.group(1))
            if relation and not re.search(
                r"\b(?:before|after|while|between)\b", relation,
            ) and relation not in {"doing", "happening"}:
                return relation
    return None


def _anchor(value: str) -> str:
    anchor = _clean(value).strip(" ,")
    anchor = re.sub(r"^(?:they|the person)\s+", "", anchor)
    anchor = re.sub(r"^(?:starting|started) to\s+", "", anchor)
    return _visible_phrase(anchor)


@dataclass(frozen=True)
class AGQATemporalLocalizedQueryPlan:
    temporal_operator: str
    relation: str
    anchor_a: str
    anchor_b: str
    visual_relation_query: str
    visual_anchor_a_query: str
    visual_anchor_b_query: str
    question_read: bool
    answer_read: bool
    functional_program_read: bool
    scene_graph_grounding_read: bool
    source_identity_read: bool
    plan_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _plan(
    *, operator: str, relation: str, anchor_a: str, anchor_b: str = "",
) -> AGQATemporalLocalizedQueryPlan | None:
    operator = operator.upper()
    if operator not in TEMPORAL_OPERATORS:
        return None
    if not relation or not anchor_a or ((operator == "BETWEEN") != bool(anchor_b)):
        return None
    core = {
        "temporal_operator": operator,
        "relation": relation,
        "anchor_a": anchor_a,
        "anchor_b": anchor_b,
        "visual_relation_query": f"a person {relation} an unknown object",
        "visual_anchor_a_query": f"a person {anchor_a}",
        "visual_anchor_b_query": f"a person {anchor_b}" if anchor_b else "",
        "question_read": True,
        "answer_read": False,
        "functional_program_read": False,
        "scene_graph_grounding_read": False,
        "source_identity_read": False,
    }
    return AGQATemporalLocalizedQueryPlan(
        **core, plan_sha256=stable_hash(core),
    )


def parse_temporal_localized_object_question(
    question: str,
) -> AGQATemporalLocalizedQueryPlan | None:
    """Parse the explicit public-question subset without target annotations."""

    text = _clean(question).rstrip("?").strip()

    # AGQA renders a two-anchor BETWEEN program as either
    # "Before right but after left, ..." or the same clause suffix.
    match = re.match(
        r"^before (.+?) but after (.+?), (.+)$", text,
    ) or re.match(r"^after (.+?) but before (.+?), (.+)$", text)
    if match:
        relation = _relation_from_query_clause(match.group(3))
        if relation:
            if text.startswith("before "):
                left, right = match.group(2), match.group(1)
            else:
                left, right = match.group(1), match.group(2)
            return _plan(
                operator="BETWEEN", relation=relation,
                anchor_a=_anchor(left), anchor_b=_anchor(right),
            )

    # Prefix templates: "After X, which object ...?"
    match = re.match(r"^(before|after|while) (.+?), (.+)$", text)
    if match:
        relation = _relation_from_query_clause(match.group(3))
        if relation:
            return _plan(
                operator=match.group(1), relation=relation,
                anchor_a=_anchor(match.group(2)),
            )
    match = re.match(r"^between (.+?) and (.+?), (.+)$", text)
    if match:
        relation = _relation_from_query_clause(match.group(3))
        if relation:
            return _plan(
                operator="BETWEEN", relation=relation,
                anchor_a=_anchor(match.group(1)), anchor_b=_anchor(match.group(2)),
            )

    # Suffix templates: "Which object ... before X?"  Parse BETWEEN first
    # because its two anchors contain an internal conjunction.
    match = re.match(
        r"^(.+?) before (.+?) but after (.+)$", text,
    ) or re.match(r"^(.+?) after (.+?) but before (.+)$", text)
    if match:
        relation = _relation_from_query_clause(match.group(1))
        if relation:
            if " before " in text.split(" but ", maxsplit=1)[0]:
                left, right = match.group(3), match.group(2)
            else:
                left, right = match.group(2), match.group(3)
            return _plan(
                operator="BETWEEN", relation=relation,
                anchor_a=_anchor(left), anchor_b=_anchor(right),
            )
    match = re.match(r"^(.+?) between (.+?) and (.+)$", text)
    if match:
        relation = _relation_from_query_clause(match.group(1))
        if relation:
            return _plan(
                operator="BETWEEN", relation=relation,
                anchor_a=_anchor(match.group(2)), anchor_b=_anchor(match.group(3)),
            )
    match = re.match(r"^(.+?) (before|after|while) (.+)$", text)
    if match:
        relation = _relation_from_query_clause(match.group(1))
        if relation:
            return _plan(
                operator=match.group(2), relation=relation,
                anchor_a=_anchor(match.group(3)),
            )
    return None


@dataclass(frozen=True)
class AGQATemporalWindowReceipt:
    temporal_operator: str
    frame_count: int
    anchor_a_interval: tuple[int, int]
    anchor_b_interval: tuple[int, int] | None
    window_start_frame: int | None
    window_end_frame: int | None
    authorized: bool
    reason: str
    answer_read: bool
    functional_program_read: bool
    scene_graph_grounding_read: bool
    receipt_sha256: str

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["anchor_a_interval"] = list(self.anchor_a_interval)
        row["anchor_b_interval"] = (
            list(self.anchor_b_interval) if self.anchor_b_interval else None
        )
        return row


@dataclass(frozen=True)
class AGQAAnchorConsensusReceipt:
    view_intervals: tuple[tuple[int, int], ...]
    consensus_interval: tuple[int, int] | None
    maximum_endpoint_spread: int | None
    authorized: bool
    reason: str
    answer_read: bool
    functional_program_read: bool
    scene_graph_grounding_read: bool
    receipt_sha256: str

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["view_intervals"] = [list(value) for value in self.view_intervals]
        row["consensus_interval"] = (
            list(self.consensus_interval) if self.consensus_interval else None
        )
        return row


def consensus_anchor_interval(
    receipts: Sequence[Mapping[str, Any]], *, minimum_confidence: float = 0.5,
    maximum_endpoint_spread: int = 8,
) -> AGQAAnchorConsensusReceipt:
    """Resolve one interval per independent view or abstain on disagreement."""

    if len(receipts) < 2:
        raise ValueError("anchor recurrence requires at least two views")
    intervals: list[tuple[int, int]] = []
    for receipt in receipts:
        observations = receipt.get("observations")
        if not isinstance(observations, list):
            raise ValueError("anchor receipt observations must be a list")
        supported = [
            row for row in observations
            if row.get("observability") == "OBSERVED"
            and float(row.get("confidence", -1.0)) >= minimum_confidence
            and row.get("start_frame") is not None
            and row.get("end_frame") is not None
            and bool(row.get("evidence_frames"))
        ]
        if not supported:
            continue
        ranked = sorted(
            supported,
            key=lambda row: (
                -float(row["confidence"]), int(row["start_frame"]),
                int(row["end_frame"]),
            ),
        )
        if len(ranked) > 1 and float(ranked[0]["confidence"]) == float(
            ranked[1]["confidence"]
        ):
            # One ambiguous view cannot vote. Two other independent views may
            # still establish the recurrent interval.
            continue
        intervals.append((
            int(ranked[0]["start_frame"]), int(ranked[0]["end_frame"]),
        ))
    spread = None
    consensus = None
    authorized = False
    if len(intervals) >= 2:
        clusters = []
        for seed in range(len(intervals)):
            cluster = tuple(
                index for index, interval in enumerate(intervals)
                if max(
                    abs(interval[0] - intervals[seed][0]),
                    abs(interval[1] - intervals[seed][1]),
                ) <= maximum_endpoint_spread
            )
            if len(cluster) >= 2 and all(
                max(
                    abs(intervals[left][0] - intervals[right][0]),
                    abs(intervals[left][1] - intervals[right][1]),
                ) <= maximum_endpoint_spread
                for left in cluster for right in cluster
            ):
                clusters.append(cluster)
        unique = sorted(set(clusters), key=lambda row: (-len(row), row))
        best = [row for row in unique if len(row) == len(unique[0])] if unique else []
        if len(best) == 1:
            selected = [intervals[index] for index in best[0]]
            starts = [row[0] for row in selected]
            ends = [row[1] for row in selected]
            spread = max(max(starts) - min(starts), max(ends) - min(ends))
            consensus = (int(median_low(starts)), int(median_high(ends)))
            authorized = consensus[0] <= consensus[1]
    if len(intervals) < 2:
        reason = "SOURCE_ABSTAIN_ANCHOR_RECURRENCE_NOT_CONFIRMED"
    elif not authorized:
        reason = "SOURCE_ABSTAIN_NO_UNIQUE_ANCHOR_VIEW_CLUSTER"
    else:
        reason = "RECURRENT_ANCHOR_INTERVAL_CONSENSUS"
    core = {
        "view_intervals": [list(row) for row in intervals],
        "consensus_interval": list(consensus) if consensus else None,
        "maximum_endpoint_spread": spread,
        "authorized": authorized,
        "reason": reason,
        "answer_read": False,
        "functional_program_read": False,
        "scene_graph_grounding_read": False,
    }
    return AGQAAnchorConsensusReceipt(
        view_intervals=tuple(intervals), consensus_interval=consensus,
        maximum_endpoint_spread=spread, authorized=authorized, reason=reason,
        answer_read=False, functional_program_read=False,
        scene_graph_grounding_read=False, receipt_sha256=stable_hash(core),
    )


def execute_temporal_window(
    *, temporal_operator: str, frame_count: int,
    anchor_a_interval: tuple[int, int],
    anchor_b_interval: tuple[int, int] | None = None,
    minimum_window_frames: int = 3,
) -> AGQATemporalWindowReceipt:
    """Execute an annotation-free interval-to-window symbolic operation."""

    operator = temporal_operator.upper()
    if operator not in TEMPORAL_OPERATORS:
        raise ValueError("unsupported temporal window operator")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")

    def valid(interval: tuple[int, int]) -> bool:
        return 0 <= interval[0] <= interval[1] < frame_count

    if not valid(anchor_a_interval):
        raise ValueError("anchor A interval is outside the frame range")
    if operator == "BETWEEN":
        if anchor_b_interval is None or not valid(anchor_b_interval):
            raise ValueError("BETWEEN requires a valid anchor B interval")
    elif anchor_b_interval is not None:
        raise ValueError("only BETWEEN accepts anchor B")

    if operator == "BEFORE":
        start, end = 0, anchor_a_interval[0] - 1
    elif operator == "AFTER":
        start, end = anchor_a_interval[1] + 1, frame_count - 1
    elif operator == "WHILE":
        start, end = anchor_a_interval
    else:
        assert anchor_b_interval is not None
        left, right = sorted((anchor_a_interval, anchor_b_interval))
        start, end = left[1] + 1, right[0] - 1

    authorized = start <= end and end - start + 1 >= minimum_window_frames
    reason = (
        "TEMPORAL_WINDOW_EXECUTED"
        if authorized else "SOURCE_ABSTAIN_TEMPORAL_WINDOW_TOO_SMALL"
    )
    core = {
        "temporal_operator": operator,
        "frame_count": frame_count,
        "anchor_a_interval": list(anchor_a_interval),
        "anchor_b_interval": list(anchor_b_interval) if anchor_b_interval else None,
        "window_start_frame": start if authorized else None,
        "window_end_frame": end if authorized else None,
        "authorized": authorized,
        "reason": reason,
        "answer_read": False,
        "functional_program_read": False,
        "scene_graph_grounding_read": False,
    }
    return AGQATemporalWindowReceipt(
        temporal_operator=operator,
        frame_count=frame_count,
        anchor_a_interval=anchor_a_interval,
        anchor_b_interval=anchor_b_interval,
        window_start_frame=core["window_start_frame"],
        window_end_frame=core["window_end_frame"],
        authorized=authorized,
        reason=reason,
        answer_read=False,
        functional_program_read=False,
        scene_graph_grounding_read=False,
        receipt_sha256=stable_hash(core),
    )


def select_composite_source_programs(
    sources: Sequence[SourceIRContract], *, grounder_qualified: bool,
    formal_outcome_read: bool = False,
) -> dict[str, Any]:
    """Require one exact temporal and one exact relation source primitive."""

    def matches(source: SourceIRContract, *, temporal: bool) -> bool:
        source.validate()
        if temporal:
            return (
                source.ir_kind == TEMPORAL_IR
                and source.operator_sequence == (TEMPORAL_PAIR_OPERATOR,)
                and source.recurrent
            )
        return (
            source.ir_kind == RELATION_IR
            and source.operator_sequence == (RELATION_OPERATOR,)
            and source.recurrent
            and source.terminal_predicate_families
            == ("ENTITY_GOAL_RELATION",)
        )

    temporal = [row for row in sources if matches(row, temporal=True)]
    relation = [row for row in sources if matches(row, temporal=False)]
    authorized = (
        grounder_qualified and not formal_outcome_read
        and len(temporal) == 1 and len(relation) == 1
        and temporal[0].source_intervention_qualified
        and relation[0].source_intervention_qualified
    )
    if formal_outcome_read:
        reason = "CURRENT_TARGET_OUTCOME_EXPOSED"
    elif not grounder_qualified:
        reason = "TARGET_GROUNDER_NOT_QUALIFIED"
    elif len(temporal) != 1 or len(relation) != 1:
        reason = "COMPOSITE_PRIMITIVES_NOT_UNIQUELY_TYPED"
    elif not all(row.source_intervention_qualified for row in (*temporal, *relation)):
        reason = "SOURCE_PROGRAM_NOT_FRESH_CONFIRMED"
    else:
        reason = "TWO_EXACT_ANONYMOUS_SOURCE_PRIMITIVES_COMPOSED"
    body: dict[str, Any] = {
        "schema_version": "agqa-temporal-localized-composite-selection-v1",
        "status": "AUTHORIZED" if authorized else "ABSTAINED",
        "reason": reason,
        "temporal_program_sha256": temporal[0].program_sha256 if authorized else None,
        "relation_program_sha256": relation[0].program_sha256 if authorized else None,
        "target_native_composition": "TEMPORAL_WINDOW_THEN_RELATION_SCAN",
        "source_identity_used_as_feature": False,
        "grounder_qualified": bool(grounder_qualified),
        "target_outcome_read": bool(formal_outcome_read),
    }
    return body | {"receipt_sha256": stable_hash(body)}


def calibrate_window_object_consensus(
    *, model_family_responses: Mapping[str, str],
    ontology_family_receipts: Mapping[str, Mapping[str, Any]],
    ontology_minimum_confidence: float, minimum_model_families: int = 2,
) -> dict[str, Any]:
    """Require agreement across target-native model families on one crop.

    Multiple prompts from the same model family never create extra votes.  A
    full-video direct response is intentionally absent from this interface.
    """

    if minimum_model_families < 2:
        raise ValueError("window consensus needs at least two model families")
    family_votes: dict[str, str] = {}
    for family, response in model_family_responses.items():
        label = canonical_object_label(response)
        if label in AGQA_OBJECT_ONTOLOGY:
            family_votes[str(family)] = label
    for family, receipt in ontology_family_receipts.items():
        label = canonical_object_label(str(receipt.get("decision") or ""))
        if (
            label in AGQA_OBJECT_ONTOLOGY
            and receipt.get("relation_observed") is True
            and float(receipt.get("confidence", -1.0))
            >= ontology_minimum_confidence
            and bool(receipt.get("evidence_frames"))
        ):
            existing = family_votes.get(str(family))
            if existing is None:
                family_votes[str(family)] = label
            elif existing != label:
                # Two prompts from one family disagree; that family abstains.
                family_votes.pop(str(family))
    counts = {
        label: sum(value == label for value in family_votes.values())
        for label in AGQA_OBJECT_ONTOLOGY
    }
    winners = [
        label for label, count in counts.items()
        if count >= minimum_model_families
    ]
    decision = winners[0] if len(winners) == 1 else None
    reason = (
        "CROSS_MODEL_TEMPORAL_WINDOW_OBJECT_CONSENSUS"
        if decision else "NO_UNIQUE_CROSS_MODEL_WINDOW_OBJECT_CONSENSUS"
    )
    return {
        "schema_version": "agqa-window-object-consensus-v1",
        "decision": decision,
        "authorization_class": (
            "SOURCE_TYPED_CROP" if decision else "ABSTAIN"
        ),
        "reason": reason,
        "family_votes": [
            {"model_family": family, "decision": value}
            for family, value in sorted(family_votes.items())
        ],
        "minimum_model_families": minimum_model_families,
        "answer_read": False,
        "functional_program_read": False,
        "scene_graph_read": False,
        "source_identity_read": False,
        "full_video_direct_response_read": False,
    }


__all__ = [
    "AGQAAnchorConsensusReceipt", "AGQATemporalLocalizedQueryPlan",
    "AGQATemporalWindowReceipt", "TARGET_INTERFACE", "TEMPORAL_OPERATORS",
    "consensus_anchor_interval", "execute_temporal_window",
    "calibrate_window_object_consensus",
    "parse_temporal_localized_object_question",
    "select_composite_source_programs",
]
