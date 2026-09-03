"""Question-blind typed event inventories for raw-video AGQA grounding.

The visual acquisition stage represented here is deliberately independent of
any AGQA question.  It may see only content-addressed raw frames, stable track
IDs produced by a frozen detector, and the public target ontology.  A later
deterministic adapter is allowed to query the frozen inventory with public
semantic slots, but never with an answer or an official scene graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Any, Iterable, Mapping, Sequence

from .agqa_query_grounder_v2 import TypedRoleEvent
from .contracts import stable_hash


PUBLIC_SPATIAL_PREDICATES = frozenset({
    "above", "beneath", "behind", "in", "in front of", "on the side of",
})
PUBLIC_ATTENTION_PREDICATES = frozenset({"looking at", "watching"})
PUBLIC_CONTACT_PREDICATES = frozenset({
    "carrying", "covered by", "drinking from", "eating", "grasping",
    "holding", "leaning on", "lying on", "opening", "closing",
    "putting down", "sitting on", "standing on", "taking", "throwing",
    "tidying", "touching", "twisting", "undressing", "washing", "wearing",
    "wiping", "working on", "writing on",
})
PUBLIC_EVENT_PREDICATES = tuple(sorted(
    PUBLIC_SPATIAL_PREDICATES
    | PUBLIC_ATTENTION_PREDICATES
    | PUBLIC_CONTACT_PREDICATES
))

_PREDICATE_ALIASES = {
    "below": "beneath",
    "in front": "in front of",
    "in front of": "in front of",
    "in_front_of": "in front of",
    "looking_at": "looking at",
    "on side of": "on the side of",
    "on the side": "on the side of",
    "on_the_side_of": "on the side of",
    "put down": "putting down",
    "putting_down": "putting down",
    "sit on": "sitting on",
    "sitting_on": "sitting on",
    "stand on": "standing on",
    "standing_on": "standing on",
}
_FORBIDDEN_AUTHORITY_KEYS = frozenset({
    "answer", "answers", "gold", "gold_answer", "correct", "correctness",
    "question", "questions", "functional_program", "functional_programs",
    "official_stsg", "stsg", "scene_graph", "source_controller",
    "source_identity", "target_outcome", "target_success",
})


def canonical_event_predicate(value: str) -> str:
    """Normalize public ontology spellings without learning target outcomes."""

    raw = str(value).strip().casefold()
    if raw in _PREDICATE_ALIASES:
        return _PREDICATE_ALIASES[raw]
    text = re.sub(r"[^a-z0-9 ]+", " ", raw.replace("_", " "))
    text = re.sub(r"\s+", " ", text).strip()
    return _PREDICATE_ALIASES.get(text, text)


def object_role_for_predicate(predicate: str) -> str:
    """Assign typed roles from the frozen public predicate ontology."""

    value = canonical_event_predicate(predicate)
    if value in PUBLIC_SPATIAL_PREDICATES | PUBLIC_ATTENTION_PREDICATES:
        return "relation_object"
    if value in PUBLIC_CONTACT_PREDICATES:
        return "patient"
    raise ValueError(f"predicate is outside the public event ontology: {predicate}")


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any(str(key).casefold() in _FORBIDDEN_AUTHORITY_KEYS for key in value):
            return True
        return any(_contains_forbidden_key(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(item) for item in value)
    return False


@dataclass(frozen=True)
class QuestionBlindTypedEvent:
    event_id: str
    predicate: str
    subject_track_id: str
    object_track_id: str
    object_role: str
    start_frame: int
    end_frame: int
    evidence_frames: tuple[int, ...]
    confidence: float
    source_clip_ids: tuple[str, ...]

    def validate(
        self, *, known_track_ids: frozenset[str], allowed_frame_ids: frozenset[int],
        track_visible_frames: Mapping[str, frozenset[int]] | None = None,
    ) -> None:
        if re.fullmatch(r"V[0-9]+", self.event_id) is None:
            raise ValueError("question-blind event IDs must be V0,V1,...")
        predicate = canonical_event_predicate(self.predicate)
        if predicate not in PUBLIC_EVENT_PREDICATES or predicate != self.predicate:
            raise ValueError("event predicate is outside the canonical public ontology")
        if self.subject_track_id not in known_track_ids:
            raise ValueError("event subject references an unknown stable track")
        if self.object_track_id not in known_track_ids:
            raise ValueError("event object references an unknown stable track")
        if self.subject_track_id == self.object_track_id:
            raise ValueError("event subject and object tracks must differ")
        if self.object_role != object_role_for_predicate(predicate):
            raise ValueError("event object role disagrees with the public ontology")
        if not 0 <= self.start_frame <= self.end_frame:
            raise ValueError("event interval is invalid")
        if self.start_frame not in allowed_frame_ids or self.end_frame not in allowed_frame_ids:
            raise ValueError("event interval endpoints were not presented to the grounder")
        if not self.evidence_frames:
            raise ValueError("event needs pixel evidence")
        if tuple(sorted(set(self.evidence_frames))) != self.evidence_frames:
            raise ValueError("event evidence must be chronological and unique")
        if any(frame not in allowed_frame_ids for frame in self.evidence_frames):
            raise ValueError("event cites an unpresented frame")
        if any(frame < self.start_frame or frame > self.end_frame for frame in self.evidence_frames):
            raise ValueError("event evidence lies outside its interval")
        if track_visible_frames is not None:
            for track_id in (self.subject_track_id, self.object_track_id):
                visible = track_visible_frames.get(track_id, frozenset())
                if any(frame not in visible for frame in self.evidence_frames):
                    raise ValueError(
                        "event cites a track on a frame without detector evidence"
                    )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("event confidence must be in [0,1]")
        if not self.source_clip_ids or len(self.source_clip_ids) != len(set(self.source_clip_ids)):
            raise ValueError("event needs unique source clip IDs")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_question_blind_event_payload(
    payload: Mapping[str, Any], *, clip_id: str,
    visible_track_ids: Sequence[str], person_track_ids: Sequence[str],
    presented_frame_ids: Sequence[int], first_event_index: int = 0,
    track_visible_frames: Mapping[str, frozenset[int]] | None = None,
) -> tuple[QuestionBlindTypedEvent, ...]:
    """Parse one strict provider payload and fail closed on authority violations."""

    if _contains_forbidden_key(payload):
        raise ValueError("question-blind event payload crossed its authority boundary")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or len(raw_events) > 32:
        raise ValueError("event payload must contain at most 32 events")
    known = frozenset(str(value) for value in visible_track_ids)
    persons = frozenset(str(value) for value in person_track_ids)
    frames = frozenset(int(value) for value in presented_frame_ids)
    if not known or not persons <= known or not frames:
        raise ValueError("visible tracks/persons/frames are incomplete")
    output: list[QuestionBlindTypedEvent] = []
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            raise ValueError("event rows must be objects")
        predicate = canonical_event_predicate(str(raw.get("predicate") or ""))
        subject = str(raw.get("subject_track_id") or "")
        object_track = str(raw.get("object_track_id") or "")
        if subject not in persons:
            raise ValueError("event subject must be a visible person track")
        evidence = tuple(sorted(set(int(value) for value in raw.get("evidence_frame_ids", ()))))
        event = QuestionBlindTypedEvent(
            event_id=f"V{first_event_index + len(output)}",
            predicate=predicate,
            subject_track_id=subject,
            object_track_id=object_track,
            object_role=object_role_for_predicate(predicate),
            start_frame=int(raw.get("start_frame_id")),
            end_frame=int(raw.get("end_frame_id")),
            evidence_frames=evidence,
            confidence=float(raw.get("confidence")),
            source_clip_ids=(str(clip_id),),
        )
        event.validate(
            known_track_ids=known,
            allowed_frame_ids=frames,
            track_visible_frames=track_visible_frames,
        )
        output.append(event)
    return tuple(output)


def parse_question_blind_event_payload_with_rejections(
    payload: Mapping[str, Any], *, clip_id: str,
    visible_track_ids: Sequence[str], person_track_ids: Sequence[str],
    presented_frame_ids: Sequence[int], first_event_index: int = 0,
    track_visible_frames: Mapping[str, frozenset[int]] | None = None,
) -> tuple[tuple[QuestionBlindTypedEvent, ...], tuple[dict[str, Any], ...]]:
    """Keep valid events while auditing invalid proposals independently.

    Authority violations remain fatal for the complete response.  A single
    impossible track/frame claim, however, must not erase unrelated valid
    pixel-grounded events from the same clip.
    """

    if _contains_forbidden_key(payload):
        raise ValueError("question-blind event payload crossed its authority boundary")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or len(raw_events) > 32:
        raise ValueError("event payload must contain at most 32 events")
    output: list[QuestionBlindTypedEvent] = []
    rejected = []
    for source_index, raw in enumerate(raw_events):
        try:
            output.extend(parse_question_blind_event_payload(
                {"events": [raw]},
                clip_id=clip_id,
                visible_track_ids=visible_track_ids,
                person_track_ids=person_track_ids,
                presented_frame_ids=presented_frame_ids,
                first_event_index=first_event_index + len(output),
                track_visible_frames=track_visible_frames,
            ))
        except (TypeError, ValueError) as exc:
            rejected.append({
                "source_event_index": source_index,
                "reason": f"{type(exc).__name__}:{exc}",
                "raw_event_sha256": stable_hash(raw),
            })
    return tuple(output), tuple(rejected)


def _interval_iou(left: QuestionBlindTypedEvent, right: QuestionBlindTypedEvent) -> float:
    intersection = max(
        0, min(left.end_frame, right.end_frame)
        - max(left.start_frame, right.start_frame) + 1,
    )
    union = max(left.end_frame, right.end_frame) - min(
        left.start_frame, right.start_frame
    ) + 1
    return intersection / union if union else 0.0


def deduplicate_question_blind_events(
    events: Iterable[QuestionBlindTypedEvent], *, minimum_interval_iou: float = 0.5,
) -> tuple[QuestionBlindTypedEvent, ...]:
    """Merge only same-predicate, same-track events with overlapping intervals."""

    if not 0.0 <= minimum_interval_iou <= 1.0:
        raise ValueError("minimum_interval_iou must be in [0,1]")
    groups: list[list[QuestionBlindTypedEvent]] = []
    for event in sorted(events, key=lambda row: (
        row.start_frame, row.end_frame, row.predicate,
        row.subject_track_id, row.object_track_id, row.event_id,
    )):
        match = next((group for group in groups if (
            group[0].predicate == event.predicate
            and group[0].subject_track_id == event.subject_track_id
            and group[0].object_track_id == event.object_track_id
            and group[0].object_role == event.object_role
            and _interval_iou(group[-1], event) >= minimum_interval_iou
        )), None)
        if match is None:
            groups.append([event])
        else:
            match.append(event)
    output = []
    for index, group in enumerate(groups):
        first = group[0]
        output.append(replace(
            first, event_id=f"V{index}",
            start_frame=min(row.start_frame for row in group),
            end_frame=max(row.end_frame for row in group),
            evidence_frames=tuple(sorted({
                frame for row in group for frame in row.evidence_frames
            })),
            confidence=max(row.confidence for row in group),
            source_clip_ids=tuple(sorted({
                clip_id for row in group for clip_id in row.source_clip_ids
            })),
        ))
    return tuple(output)


def _predicate_matches(query_predicate: str, event_predicate: str) -> bool:
    query = canonical_event_predicate(query_predicate)
    event = canonical_event_predicate(event_predicate)
    if not query:
        # Public generic "interact with" queries can be answered by any
        # visually grounded contact or attention event, but not by a purely
        # spatial co-occurrence.
        return event in PUBLIC_CONTACT_PREDICATES | PUBLIC_ATTENTION_PREDICATES
    if query == "watching":
        return event in {"watching", "looking at"}
    if query == "looking at":
        return event in {"watching", "looking at"}
    return query == event


def query_event_candidates(
    events: Iterable[QuestionBlindTypedEvent], *, predicate: str,
    requested_role: str, lower_frame: int, upper_frame: int,
) -> tuple[dict[str, Any], ...]:
    """Rank track bindings from a frozen per-video event inventory.

    The score is deliberately outcome-independent: maximum provider confidence
    for an exact public-predicate/typed-role event inside the parser's frozen
    temporal scope.  Repeated evidence is reported but does not receive a
    hand-tuned score bonus.
    """

    grouped: dict[str, list[QuestionBlindTypedEvent]] = {}
    for event in events:
        if event.object_role != requested_role:
            continue
        if not _predicate_matches(predicate, event.predicate):
            continue
        overlap = max(0, min(event.end_frame, upper_frame) - max(event.start_frame, lower_frame) + 1)
        if overlap <= 0 or not any(lower_frame <= frame <= upper_frame for frame in event.evidence_frames):
            continue
        grouped.setdefault(event.object_track_id, []).append(event)
    output = []
    for track_id, rows in grouped.items():
        best = max(rows, key=lambda row: (row.confidence, len(row.evidence_frames), row.event_id))
        output.append({
            "track_id": track_id,
            "score": float(best.confidence),
            "event_ids": sorted(row.event_id for row in rows),
            "evidence_frames": sorted({
                frame for row in rows for frame in row.evidence_frames
                if lower_frame <= frame <= upper_frame
            }),
            "source_clip_ids": sorted({
                clip_id for row in rows for clip_id in row.source_clip_ids
            }),
            "support_count": len(rows),
        })
    return tuple(sorted(output, key=lambda row: (
        -float(row["score"]), -int(row["support_count"]), str(row["track_id"]),
    )))


def query_temporal_event_candidates(
    events: Iterable[QuestionBlindTypedEvent], *, predicate: str,
    requested_role: str, temporal_operator: str,
    anchor_intervals: Sequence[Sequence[int]] = (),
    temporal_uncertainty_frames: int = 0,
) -> tuple[dict[str, Any], ...]:
    """Execute a typed temporal query over a question-blind event graph.

    Anchor intervals must come from an independent answer-blind action
    localizer.  The uncertainty radius is an acquisition property (for
    example half a non-overlapping clip), not an outcome-fitted threshold.
    Candidate confidence is never allowed to make a temporally invalid event
    valid.  BEFORE/AFTER prefer the closest valid event; VIDEO and overlap
    operators prefer the strongest grounded event after temporal filtering.
    """

    operator = str(temporal_operator).strip().upper()
    if operator not in {"VIDEO", "BEFORE", "AFTER", "WHILE", "BETWEEN"}:
        raise ValueError(f"unsupported temporal operator: {temporal_operator}")
    uncertainty = int(temporal_uncertainty_frames)
    if uncertainty < 0:
        raise ValueError("temporal uncertainty must be nonnegative")
    normalized_anchors = []
    for interval in anchor_intervals:
        if len(interval) != 2:
            raise ValueError("anchor intervals must contain [start,end]")
        lower, upper = (int(value) for value in interval)
        if lower < 0 or upper < lower:
            raise ValueError("anchor interval is invalid")
        normalized_anchors.append((lower, upper))
    required = 0 if operator == "VIDEO" else (2 if operator == "BETWEEN" else 1)
    if len(normalized_anchors) < required:
        return ()

    def temporal_key(event: QuestionBlindTypedEvent) -> tuple[float, float] | None:
        if operator == "VIDEO":
            return float(event.confidence), float(len(event.evidence_frames))
        if operator == "BEFORE":
            anchor_start, _ = normalized_anchors[0]
            # Directionality is strict: acquisition uncertainty may widen an
            # overlap query, but must never turn an after-anchor event into a
            # BEFORE candidate.
            if event.end_frame >= anchor_start:
                return None
            distance = anchor_start - event.end_frame
            return -float(distance), float(event.confidence)
        if operator == "AFTER":
            _, anchor_end = normalized_anchors[0]
            if event.start_frame <= anchor_end:
                return None
            distance = event.start_frame - anchor_end
            return -float(distance), float(event.confidence)
        if operator == "WHILE":
            anchor_start, anchor_end = normalized_anchors[0]
            lower, upper = anchor_start - uncertainty, anchor_end + uncertainty
        else:
            left, right = sorted(normalized_anchors[:2])
            lower, upper = left[1] - uncertainty, right[0] + uncertainty
            if lower > upper:
                lower, upper = upper, lower
        overlap = max(
            0, min(event.end_frame, upper) - max(event.start_frame, lower) + 1,
        )
        if overlap <= 0:
            return None
        return float(event.confidence), float(overlap)

    grouped: dict[str, list[tuple[QuestionBlindTypedEvent, tuple[float, float]]]] = {}
    for event in events:
        if event.object_role != requested_role or not _predicate_matches(predicate, event.predicate):
            continue
        key = temporal_key(event)
        if key is not None:
            grouped.setdefault(event.object_track_id, []).append((event, key))
    output = []
    for track_id, values in grouped.items():
        best_event, key = max(values, key=lambda value: (
            value[1], value[0].confidence, len(value[0].evidence_frames), value[0].event_id,
        ))
        contributing = [event for event, _ in values]
        output.append({
            "track_id": track_id,
            "score": float(best_event.confidence),
            "temporal_priority": [float(key[0]), float(key[1])],
            "event_ids": sorted(event.event_id for event in contributing),
            "evidence_frames": sorted({
                frame for event in contributing for frame in event.evidence_frames
            }),
            "source_clip_ids": sorted({
                clip_id for event in contributing for clip_id in event.source_clip_ids
            }),
            "support_count": len(contributing),
        })
    return tuple(sorted(output, key=lambda row: (
        -float(row["temporal_priority"][0]),
        -float(row["temporal_priority"][1]),
        -float(row["score"]), -int(row["support_count"]), str(row["track_id"]),
    )))


def bind_event_to_semantic_slots(
    event: QuestionBlindTypedEvent, *, event_id: str,
    semantic_slot_ids: Sequence[str],
) -> TypedRoleEvent:
    """Bind a frozen question-blind event to public parser slots downstream."""

    subject_role = (
        "relation_subject" if event.object_role == "relation_object" else "agent"
    )
    return TypedRoleEvent(
        event_id=event_id,
        predicate=event.predicate,
        roles=((subject_role, event.subject_track_id), (event.object_role, event.object_track_id)),
        start_frame=event.start_frame,
        end_frame=event.end_frame,
        evidence_frames=event.evidence_frames,
        confidence=event.confidence,
        semantic_slot_ids=tuple(str(value) for value in semantic_slot_ids),
    )


__all__ = [
    "PUBLIC_ATTENTION_PREDICATES", "PUBLIC_CONTACT_PREDICATES",
    "PUBLIC_EVENT_PREDICATES", "PUBLIC_SPATIAL_PREDICATES",
    "QuestionBlindTypedEvent", "bind_event_to_semantic_slots",
    "canonical_event_predicate", "deduplicate_question_blind_events",
    "object_role_for_predicate", "parse_question_blind_event_payload",
    "query_event_candidates", "query_temporal_event_candidates",
]
