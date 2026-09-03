"""Outcome-blind typed entity/event receipts for AGQA query grounding V2.

This module deliberately does not replace the frozen Layer-B V1 contracts.
V2 receipts are an additional target-native perception view shared by every
Harness arm.  They may be adapted to the existing typed VM only after their
content hash and authority boundary have been validated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Iterable, Mapping, Sequence

from .contracts import stable_hash
from .agqa_layer_b_contracts import AGQASemanticSlotReceipt
from .agqa_layer_b_contracts import GroundedEvent, RawVideoEventGraphReceipt
from .agqa_semantic_slots import relation_grounding_obligations


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRACK_ID = re.compile(r"^T[0-9]+$")
_EVENT_ID = re.compile(r"^R[0-9]+$")
ROLE_NAMES = frozenset({
    "agent", "patient", "theme", "destination", "instrument",
    "relation_subject", "relation_object",
})
_SPATIAL_QUERY_PREDICATES = frozenset({
    "above", "behind", "below", "beneath", "in", "in front of",
    "near", "next to", "on the side of", "related to",
})


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).replace("_", " ").strip().casefold())


def _require_sha(value: str, field: str) -> None:
    if _SHA256.fullmatch(str(value)) is None:
        raise ValueError(f"{field} must be a sha256")


@dataclass(frozen=True)
class EntityTrack:
    track_id: str
    canonical_label: str
    aliases: tuple[str, ...]
    evidence_frames: tuple[int, ...]
    confidence: float

    def validate(self, frame_count: int) -> None:
        if _TRACK_ID.fullmatch(self.track_id) is None:
            raise ValueError("entity track IDs must be T0,T1,...")
        if not _clean(self.canonical_label):
            raise ValueError("entity tracks require a canonical label")
        names = tuple(_clean(value) for value in (self.canonical_label, *self.aliases))
        if len(names) != len(set(names)):
            raise ValueError("entity canonical label and aliases must be unique")
        if not self.evidence_frames or tuple(sorted(set(self.evidence_frames))) != self.evidence_frames:
            raise ValueError("entity evidence frames must be non-empty, unique, and chronological")
        if any(frame < 0 or frame >= frame_count for frame in self.evidence_frames):
            raise ValueError("entity evidence exceeds the frozen frame set")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("entity confidence must be in [0,1]")


@dataclass(frozen=True)
class TypedRoleEvent:
    event_id: str
    predicate: str
    roles: tuple[tuple[str, str], ...]
    start_frame: int
    end_frame: int
    evidence_frames: tuple[int, ...]
    confidence: float
    semantic_slot_ids: tuple[str, ...]

    def validate(self, frame_count: int, track_ids: frozenset[str]) -> None:
        if _EVENT_ID.fullmatch(self.event_id) is None:
            raise ValueError("typed event IDs must be R0,R1,...")
        if not _clean(self.predicate):
            raise ValueError("typed events require a predicate")
        names = tuple(name for name, _ in self.roles)
        if not self.roles or len(names) != len(set(names)) or not set(names) <= ROLE_NAMES:
            raise ValueError("typed event roles are empty, duplicated, or unknown")
        if any(track_id not in track_ids for _, track_id in self.roles):
            raise ValueError("typed event role references an unknown entity track")
        if not 0 <= self.start_frame <= self.end_frame < frame_count:
            raise ValueError("typed event interval exceeds the frozen frame set")
        if not self.evidence_frames or any(
            frame < self.start_frame or frame > self.end_frame for frame in self.evidence_frames
        ):
            raise ValueError("typed events require in-interval pixel evidence")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("typed event confidence must be in [0,1]")
        if not self.semantic_slot_ids or len(self.semantic_slot_ids) != len(set(self.semantic_slot_ids)):
            raise ValueError("typed events require unique semantic-slot bindings")
        if any(re.fullmatch(r"S[0-9]+", value) is None for value in self.semantic_slot_ids):
            raise ValueError("typed events may bind only S0,S1,... slots")

    @property
    def role_map(self) -> Mapping[str, str]:
        return dict(self.roles)


@dataclass(frozen=True)
class QueryCandidateEvidence:
    track_id: str
    requested_role: str
    status: str
    confidence: float
    evidence_frames: tuple[int, ...]

    def validate(self, frame_count: int, track_ids: frozenset[str]) -> None:
        if self.track_id not in track_ids:
            raise ValueError("candidate evidence references an unknown track")
        if self.requested_role not in ROLE_NAMES - {"agent"}:
            raise ValueError("invalid requested query role")
        if self.status not in {"SUPPORTED", "REFUTED", "UNKNOWN"}:
            raise ValueError("invalid candidate evidence status")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must be in [0,1]")
        if tuple(sorted(set(self.evidence_frames))) != self.evidence_frames:
            raise ValueError("candidate evidence frames must be unique and chronological")
        if any(frame < 0 or frame >= frame_count for frame in self.evidence_frames):
            raise ValueError("candidate evidence exceeds the frozen frame set")
        if self.status in {"SUPPORTED", "REFUTED"} and not self.evidence_frames:
            raise ValueError("decisive candidate evidence requires cited pixels")


@dataclass(frozen=True)
class QueryGroundingV2Receipt:
    task_id: str
    video_sha256: str
    semantic_slots_sha256: str
    selected_frame_indices: tuple[int, ...]
    selected_frame_sha256s: tuple[str, ...]
    tracks: tuple[EntityTrack, ...]
    events: tuple[TypedRoleEvent, ...]
    candidates: tuple[QueryCandidateEvidence, ...]
    public_ontology_sha256: str
    grounder_backend_sha256: str
    provider_calls: int
    answer_read: bool
    official_scene_graph_read: bool
    functional_program_read: bool
    source_controller_read: bool
    target_outcome_read: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls, *, task_id: str, video_sha256: str, semantic_slots_sha256: str,
        selected_frame_indices: Sequence[int], selected_frame_sha256s: Sequence[str],
        tracks: Sequence[EntityTrack], events: Sequence[TypedRoleEvent],
        candidates: Sequence[QueryCandidateEvidence], public_ontology_sha256: str,
        grounder_backend_sha256: str, provider_calls: int,
    ) -> "QueryGroundingV2Receipt":
        body = {
            "task_id": str(task_id), "video_sha256": str(video_sha256),
            "semantic_slots_sha256": str(semantic_slots_sha256),
            "selected_frame_indices": tuple(int(x) for x in selected_frame_indices),
            "selected_frame_sha256s": tuple(str(x) for x in selected_frame_sha256s),
            "tracks": [asdict(x) for x in tracks], "events": [asdict(x) for x in events],
            "candidates": [asdict(x) for x in candidates],
            "public_ontology_sha256": str(public_ontology_sha256),
            "grounder_backend_sha256": str(grounder_backend_sha256),
            "provider_calls": int(provider_calls), "answer_read": False,
            "official_scene_graph_read": False, "functional_program_read": False,
            "source_controller_read": False, "target_outcome_read": False,
        }
        receipt = cls(
            **{**body, "tracks": tuple(tracks), "events": tuple(events),
               "candidates": tuple(candidates)},
            receipt_sha256=stable_hash(body),
        )
        receipt.validate()
        return receipt

    def validate(self) -> None:
        for value, field in (
            (self.video_sha256, "video_sha256"),
            (self.semantic_slots_sha256, "semantic_slots_sha256"),
            (self.public_ontology_sha256, "public_ontology_sha256"),
            (self.grounder_backend_sha256, "grounder_backend_sha256"),
        ):
            _require_sha(value, field)
        if any((self.answer_read, self.official_scene_graph_read, self.functional_program_read,
                self.source_controller_read, self.target_outcome_read)):
            raise ValueError("query grounding crossed its authority boundary")
        if not self.selected_frame_indices or (
            tuple(sorted(set(self.selected_frame_indices))) != self.selected_frame_indices
        ):
            raise ValueError("selected frames must be non-empty, unique, and chronological")
        if len(self.selected_frame_indices) != len(self.selected_frame_sha256s):
            raise ValueError("selected frame indices and hashes are misaligned")
        for digest in self.selected_frame_sha256s:
            _require_sha(digest, "selected_frame_sha256")
        track_ids = tuple(track.track_id for track in self.tracks)
        event_ids = tuple(event.event_id for event in self.events)
        if len(track_ids) != len(set(track_ids)) or len(event_ids) != len(set(event_ids)):
            raise ValueError("track and event IDs must be unique")
        known = frozenset(track_ids); frame_count = len(self.selected_frame_indices)
        for track in self.tracks:
            track.validate(frame_count)
        for event in self.events:
            event.validate(frame_count, known)
        for candidate in self.candidates:
            candidate.validate(frame_count, known)
        body = asdict(self); claimed = body.pop("receipt_sha256")
        if stable_hash(body) != claimed:
            raise ValueError("query grounding V2 receipt hash mismatch")


def deduplicate_typed_events(
    events: Iterable[TypedRoleEvent], *, minimum_interval_iou: float = 0.5,
) -> tuple[TypedRoleEvent, ...]:
    """Merge panel duplicates only when predicate and typed track roles agree."""
    if not 0.0 <= minimum_interval_iou <= 1.0:
        raise ValueError("minimum_interval_iou must be in [0,1]")
    groups: list[list[TypedRoleEvent]] = []
    for event in sorted(events, key=lambda x: (x.start_frame, x.end_frame, x.event_id)):
        match = None
        for group in groups:
            head = group[0]
            if _clean(head.predicate) != _clean(event.predicate) or head.roles != event.roles:
                continue
            intersection = max(0, min(head.end_frame, event.end_frame) - max(head.start_frame, event.start_frame) + 1)
            union = max(head.end_frame, event.end_frame) - min(head.start_frame, event.start_frame) + 1
            if intersection / union >= minimum_interval_iou:
                match = group; break
        if match is None:
            groups.append([event])
        else:
            match.append(event)
    merged = []
    for index, group in enumerate(groups):
        first = group[0]
        merged.append(replace(
            first, event_id=f"R{index}", start_frame=min(x.start_frame for x in group),
            end_frame=max(x.end_frame for x in group),
            evidence_frames=tuple(sorted({f for x in group for f in x.evidence_frames})),
            confidence=max(x.confidence for x in group),
            semantic_slot_ids=tuple(sorted({s for x in group for s in x.semantic_slot_ids})),
        ))
    return tuple(merged)


def _root_query_relation_components(
    semantic: AGQASemanticSlotReceipt,
) -> tuple[str, str, tuple[str, ...]] | None:
    """Find the answer-bearing relation while excluding nested reference goals.

    Compact AGQA semantics have two equivalent layouts: the query relation can
    be the right sibling of a temporal window, or the final child inside a
    multi-anchor window.  Endpoint queries also include a generic
    ``presence_question`` relation before the detailed answer relation.  A
    left-to-right "first relation" traversal therefore selects anchors.  We
    collect relations in the outer goal scope, never descend into nested
    ``QUERY_GOAL`` reference subgraphs, prefer an explicit public predicate,
    and break ties by the rightmost relation.
    """
    semantic.validate()
    by_id = {row.slot_id: row for row in semantic.slots}
    generic = {
        "action", "actions", "class", "frame", "object", "objects",
        "relation", "relations", "video",
    }
    candidates: list[tuple[str, str, tuple[str, ...]]] = []

    def visit(slot_id: str) -> None:
        row = by_id[slot_id]
        if row.kind == "QUERY_GOAL" and slot_id != semantic.root_slot_id:
            return
        if (
            row.kind == "RELATION"
            and row.surface.startswith("match a typed relation description")
            and len(row.children) >= 2
        ):
            relation_tuple = by_id[row.children[1]]
            if relation_tuple.kind == "RELATION" and relation_tuple.surface == "ordered semantic tuple":
                literals = tuple(
                    _clean(by_id[child].surface)
                    for child in relation_tuple.children
                    if by_id[child].kind == "LITERAL"
                )
                candidates.append((row.slot_id, relation_tuple.slot_id, literals))
        for child in row.children:
            visit(child)

    visit(semantic.root_slot_id)
    if not candidates:
        return None
    return max(candidates, key=lambda value: (
        any(literal not in generic for literal in value[2]),
        candidates.index(value),
    ))


def _root_query_tuple_literals(semantic: AGQASemanticSlotReceipt) -> tuple[str, ...]:
    """Return the outer query tuple while excluding nested temporal anchors."""
    found = _root_query_relation_components(semantic)
    return found[2] if found else ()


def requested_query_slot_ids(semantic: AGQASemanticSlotReceipt) -> tuple[str, ...]:
    """Return only the outer query-description slots, never anchor subgraphs.

    The IDs let a target-native adapter distinguish the answer-bearing visual
    event from temporal anchors and nested reference events.  This is still an
    operator-free graph traversal; no official AGQA program is consulted.
    """
    semantic.validate()
    by_id = {row.slot_id: row for row in semantic.slots}
    found = _root_query_relation_components(semantic)
    if found is None:
        return ()
    relation_id, tuple_id, _ = found
    relation_tuple = by_id[tuple_id]
    return tuple(dict.fromkeys((relation_id, tuple_id, *relation_tuple.children)))


def requested_query_role(semantic: AGQASemanticSlotReceipt) -> str:
    """Infer the visual output role from operator-free target semantics.

    Explicit spatial-relation obligations return their relation object.  All
    other ENTITY queries return the patient/theme of an action.  ACTION
    queries are intentionally unsupported by the entity-track view and must
    fail closed to the shared fallback.
    """
    semantic.validate()
    if semantic.answer_kind == "ACTION":
        raise ValueError("ACTION queries are outside entity-track grounding")
    if semantic.answer_kind != "ENTITY":
        raise ValueError("query candidate grounding requires an ENTITY answer kind")
    literals = _root_query_tuple_literals(semantic)
    predicates = tuple(value for value in literals if value not in {
        "action", "actions", "class", "frame", "object", "objects", "relation", "relations",
    })
    # The legacy target parser often labels both actions and spatial facts as
    # ``relations``.  The explicit public predicate, not that coarse category,
    # determines whether the missing entity is an action patient/theme or the
    # object endpoint of a directed spatial relation.
    return "relation_object" if predicates and predicates[0] in _SPATIAL_QUERY_PREDICATES else "patient"


def requested_query_predicates(semantic: AGQASemanticSlotReceipt) -> tuple[str, ...]:
    """Return explicit public semantic predicates without compiling a program."""
    generic = {"action", "actions", "class", "frame", "object", "objects", "relation", "relations"}
    return tuple(value for value in _root_query_tuple_literals(semantic) if value not in generic)


def adapt_query_grounding_v2(
    receipt: QueryGroundingV2Receipt, semantic: AGQASemanticSlotReceipt, *,
    minimum_candidate_confidence: float,
) -> RawVideoEventGraphReceipt:
    """Project supported typed-role tracks into the unchanged Layer-B VM view."""
    receipt.validate(); semantic.validate()
    if receipt.task_id != semantic.task_id or receipt.semantic_slots_sha256 != semantic.receipt_sha256:
        raise ValueError("V2 query receipt does not match semantic receipt")
    if not 0.0 <= minimum_candidate_confidence <= 1.0:
        raise ValueError("minimum candidate confidence must be in [0,1]")
    role = requested_query_role(semantic)
    supported = {
        row.track_id for row in receipt.candidates
        if row.requested_role == role and row.status == "SUPPORTED"
        and row.confidence >= minimum_candidate_confidence
    }
    tracks = {row.track_id: row for row in receipt.tracks}
    root_slot_ids = frozenset(requested_query_slot_ids(semantic))
    events = []
    for typed in receipt.events:
        role_map = typed.role_map
        is_outer_query_event = bool(root_slot_ids.intersection(typed.semantic_slot_ids))
        track_id = role_map.get(role) or (role_map.get("theme") if role == "patient" else None)
        if is_outer_query_event and track_id not in supported:
            continue
        if track_id is None:
            track_id = next((role_map.get(name) for name in (
                "patient", "theme", "relation_object", "destination", "instrument",
                "relation_subject", "agent",
            ) if role_map.get(name)), None)
        if track_id not in tracks:
            continue
        agent = tracks.get(role_map.get("agent", ""))
        candidate_confidence = next((
            row.confidence for row in receipt.candidates
            if row.track_id == track_id and row.requested_role == role and row.status == "SUPPORTED"
        ), typed.confidence)
        events.append(GroundedEvent(
            event_id=f"E{len(events)}", subject=agent.canonical_label if agent else "person",
            predicate=typed.predicate, object=tracks[track_id].canonical_label,
            start_frame=typed.start_frame, end_frame=typed.end_frame,
            evidence_frames=typed.evidence_frames,
            confidence=min(typed.confidence, candidate_confidence),
            semantic_slot_ids=typed.semantic_slot_ids,
        ))
    return RawVideoEventGraphReceipt.create(
        task_id=receipt.task_id, video_sha256=receipt.video_sha256,
        semantic_slots_sha256=receipt.semantic_slots_sha256,
        selected_frame_indices=receipt.selected_frame_indices,
        selected_frame_sha256s=receipt.selected_frame_sha256s, events=events,
        grounder_backend_sha256=stable_hash({
            "adapter": "AGQA_QUERY_TYPED_ROLE_TO_LAYER_B_VM_V2",
            "query_receipt_sha256": receipt.receipt_sha256,
            "minimum_candidate_confidence": minimum_candidate_confidence,
        }), frame_budget=len(receipt.selected_frame_indices), provider_calls=receipt.provider_calls,
    )


def query_grounding_v2_from_dict(value: Mapping[str, object]) -> QueryGroundingV2Receipt:
    """Deserialize and revalidate a content-addressed V2 receipt."""
    tracks = tuple(EntityTrack(
        track_id=str(row["track_id"]), canonical_label=str(row["canonical_label"]),
        aliases=tuple(str(x) for x in row.get("aliases", ())),
        evidence_frames=tuple(int(x) for x in row.get("evidence_frames", ())),
        confidence=float(row["confidence"]),
    ) for row in value.get("tracks", ()))
    events = tuple(TypedRoleEvent(
        event_id=str(row["event_id"]), predicate=str(row["predicate"]),
        roles=tuple((str(a), str(b)) for a, b in row.get("roles", ())),
        start_frame=int(row["start_frame"]), end_frame=int(row["end_frame"]),
        evidence_frames=tuple(int(x) for x in row.get("evidence_frames", ())),
        confidence=float(row["confidence"]),
        semantic_slot_ids=tuple(str(x) for x in row.get("semantic_slot_ids", ())),
    ) for row in value.get("events", ()))
    candidates = tuple(QueryCandidateEvidence(
        track_id=str(row["track_id"]), requested_role=str(row["requested_role"]),
        status=str(row["status"]), confidence=float(row["confidence"]),
        evidence_frames=tuple(int(x) for x in row.get("evidence_frames", ())),
    ) for row in value.get("candidates", ()))
    receipt = QueryGroundingV2Receipt(**{
        **value, "selected_frame_indices": tuple(int(x) for x in value["selected_frame_indices"]),
        "selected_frame_sha256s": tuple(str(x) for x in value["selected_frame_sha256s"]),
        "tracks": tracks, "events": events, "candidates": candidates,
    })
    receipt.validate()
    return receipt


__all__ = [
    "EntityTrack", "QueryCandidateEvidence", "QueryGroundingV2Receipt", "ROLE_NAMES",
    "TypedRoleEvent", "adapt_query_grounding_v2", "deduplicate_typed_events",
    "query_grounding_v2_from_dict", "requested_query_predicates", "requested_query_role",
    "requested_query_slot_ids",
]
