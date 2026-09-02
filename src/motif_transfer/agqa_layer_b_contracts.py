"""Outcome-blind contracts for AGQA Layer-B raw-video transfer.

The target parser may expose a semantic goal graph, but never an AGQA
functional program or a VM operator sequence.  A frozen visual model grounds
that graph into typed, pixel-evidenced events.  Controller arms consume the
same content-addressed task-state receipt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence

from .contracts import stable_hash


SLOT_KINDS = frozenset({
    "ENTITY", "ACTION", "RELATION", "TEMPORAL_CONSTRAINT", "LOGICAL_CONSTRAINT",
    "ORDINAL_CONSTRAINT", "DURATION_CONSTRAINT", "CHOICE", "QUERY_GOAL", "LITERAL",
})
ANSWER_KINDS = frozenset({"BOOLEAN", "ENTITY", "ACTION", "TEMPORAL_ORDER", "CHOICE"})
FORBIDDEN_KEYS = frozenset({
    "answer", "correct", "functional_program", "gold", "program", "operator_sequence",
    "selected_option", "sg_grounding", "source_game", "source_identity", "target_outcome",
})
VM_OPERATOR_TOKENS = frozenset({
    "AND", "ARGMAX", "ARGMIN", "CARDINALITY", "CHOOSE", "COMPARE", "EXISTS",
    "FILTER_EQ", "FIRST", "INTERSECTION", "INTERVAL_OF", "LAST", "NOT", "PROJECT",
    "TEMPORAL_SELECT", "UNION", "UNIQUE", "XOR",
})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha(value: Any, field: str) -> str:
    value = str(value)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a sha256")
    return value


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.casefold() in FORBIDDEN_KEYS:
                paths.append(path)
            paths.extend(_forbidden_paths(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


@dataclass(frozen=True)
class SemanticSlotNode:
    slot_id: str
    kind: str
    surface: str
    children: tuple[str, ...] = ()
    attributes: tuple[tuple[str, str], ...] = ()

    def validate(self) -> None:
        if not re.fullmatch(r"S[0-9]+", self.slot_id):
            raise ValueError("semantic slot IDs must be S0,S1,...")
        if self.kind not in SLOT_KINDS:
            raise ValueError("invalid semantic slot kind")
        if not isinstance(self.surface, str) or not self.surface.strip():
            raise ValueError("semantic slot surface must be non-empty")
        if any(token == token.upper() and token in VM_OPERATOR_TOKENS for token in self.surface.split()):
            raise ValueError("semantic slots may not encode VM operator names")
        keys = [str(key) for key, _ in self.attributes]
        if len(keys) != len(set(keys)):
            raise ValueError("semantic slot attribute keys must be unique")
        forbidden = _forbidden_paths(dict(self.attributes))
        if forbidden:
            raise ValueError("semantic slot attributes crossed the authority boundary")


@dataclass(frozen=True)
class AGQASemanticSlotReceipt:
    task_id: str
    question_sha256: str
    answer_kind: str
    root_slot_id: str
    slots: tuple[SemanticSlotNode, ...]
    parser_sha256: str
    parser_training_authority: str
    functional_program_read_at_runtime: bool
    operator_sequence_emitted: bool
    answer_read: bool
    target_outcome_read: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls, *, task_id: str, question_sha256: str, answer_kind: str,
        root_slot_id: str, slots: Sequence[SemanticSlotNode], parser_sha256: str,
        parser_training_authority: str,
    ) -> "AGQASemanticSlotReceipt":
        rows = tuple(slots)
        body = {
            "task_id": str(task_id), "question_sha256": _sha(question_sha256, "question_sha256"),
            "answer_kind": str(answer_kind), "root_slot_id": str(root_slot_id),
            "slots": [asdict(row) for row in rows], "parser_sha256": _sha(parser_sha256, "parser_sha256"),
            "parser_training_authority": str(parser_training_authority),
            "functional_program_read_at_runtime": False, "operator_sequence_emitted": False,
            "answer_read": False, "target_outcome_read": False,
        }
        value = cls(**{**body, "slots": rows}, receipt_sha256=stable_hash(body))
        value.validate()
        return value

    def validate(self) -> None:
        _sha(self.question_sha256, "question_sha256"); _sha(self.parser_sha256, "parser_sha256")
        if self.answer_kind not in ANSWER_KINDS:
            raise ValueError("invalid semantic answer kind")
        if self.functional_program_read_at_runtime or self.operator_sequence_emitted:
            raise ValueError("semantic parser may not emit/read a functional program")
        if self.answer_read or self.target_outcome_read:
            raise ValueError("semantic parser crossed the outcome boundary")
        ids = [row.slot_id for row in self.slots]
        if len(ids) != len(set(ids)) or self.root_slot_id not in set(ids):
            raise ValueError("semantic slot graph has invalid IDs/root")
        known = set(ids)
        for row in self.slots:
            row.validate()
            if not set(row.children) <= known:
                raise ValueError("semantic slot references an unknown child")
        body = asdict(self); claimed = body.pop("receipt_sha256")
        if stable_hash(body) != claimed:
            raise ValueError("semantic slot receipt hash mismatch")


@dataclass(frozen=True)
class GroundedEvent:
    event_id: str
    subject: str
    predicate: str
    object: str
    start_frame: int
    end_frame: int
    evidence_frames: tuple[int, ...]
    confidence: float
    semantic_slot_ids: tuple[str, ...] = ()

    def validate(self, frame_count: int) -> None:
        if not re.fullmatch(r"E[0-9]+", self.event_id):
            raise ValueError("event IDs must be E0,E1,...")
        if not self.predicate.strip():
            raise ValueError("grounded event needs a predicate")
        if not 0 <= self.start_frame <= self.end_frame < frame_count:
            raise ValueError("grounded event interval exceeds the frozen frames")
        if not self.evidence_frames or any(
            frame < self.start_frame or frame > self.end_frame for frame in self.evidence_frames
        ):
            raise ValueError("grounded event needs in-interval pixel evidence")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("grounded event confidence must be in [0,1]")
        if len(self.semantic_slot_ids) != len(set(self.semantic_slot_ids)):
            raise ValueError("grounded event semantic-slot bindings must be unique")
        if not self.semantic_slot_ids:
            raise ValueError("every grounded event must bind at least one semantic slot")
        if len(self.semantic_slot_ids) > 6:
            raise ValueError("grounded events may bind at most six perceptual semantic slots")
        if any(re.fullmatch(r"S[0-9]+", value) is None for value in self.semantic_slot_ids):
            raise ValueError("grounded events may bind only S0,S1,... semantic-slot IDs")


@dataclass(frozen=True)
class RawVideoEventGraphReceipt:
    task_id: str
    video_sha256: str
    semantic_slots_sha256: str
    selected_frame_indices: tuple[int, ...]
    selected_frame_sha256s: tuple[str, ...]
    events: tuple[GroundedEvent, ...]
    grounder_backend_sha256: str
    frame_budget: int
    provider_calls: int
    official_scene_graph_read: bool
    functional_program_read: bool
    answer_read: bool
    source_controller_read: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls, *, task_id: str, video_sha256: str, semantic_slots_sha256: str,
        selected_frame_indices: Sequence[int], selected_frame_sha256s: Sequence[str],
        events: Sequence[GroundedEvent], grounder_backend_sha256: str,
        frame_budget: int, provider_calls: int,
    ) -> "RawVideoEventGraphReceipt":
        indices = tuple(int(value) for value in selected_frame_indices)
        frame_hashes = tuple(str(value) for value in selected_frame_sha256s)
        rows = tuple(events)
        body = {
            "task_id": str(task_id), "video_sha256": _sha(video_sha256, "video_sha256"),
            "semantic_slots_sha256": _sha(semantic_slots_sha256, "semantic_slots_sha256"),
            "selected_frame_indices": indices, "selected_frame_sha256s": frame_hashes,
            "events": [asdict(row) for row in rows],
            "grounder_backend_sha256": _sha(grounder_backend_sha256, "grounder_backend_sha256"),
            "frame_budget": int(frame_budget), "provider_calls": int(provider_calls),
            "official_scene_graph_read": False, "functional_program_read": False,
            "answer_read": False, "source_controller_read": False,
        }
        value = cls(**{**body, "events": rows}, receipt_sha256=stable_hash(body))
        value.validate()
        return value

    def validate(self) -> None:
        for value, field in ((self.video_sha256, "video_sha256"),
                             (self.semantic_slots_sha256, "semantic_slots_sha256"),
                             (self.grounder_backend_sha256, "grounder_backend_sha256")):
            _sha(value, field)
        if self.official_scene_graph_read or self.functional_program_read or self.answer_read:
            raise ValueError("raw-video grounding crossed an oracle/outcome boundary")
        if self.source_controller_read:
            raise ValueError("grounder must run before and independently of Harness arms")
        if self.frame_budget <= 0 or len(self.selected_frame_indices) > self.frame_budget:
            raise ValueError("raw-video event graph exceeded the frame budget")
        if len(self.selected_frame_indices) != len(self.selected_frame_sha256s):
            raise ValueError("selected frame indices/hashes are misaligned")
        if tuple(sorted(set(self.selected_frame_indices))) != self.selected_frame_indices:
            raise ValueError("selected frames must be unique and chronological")
        for digest in self.selected_frame_sha256s:
            _sha(digest, "selected_frame_sha256")
        ids = [row.event_id for row in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("grounded event IDs must be unique")
        for row in self.events:
            row.validate(len(self.selected_frame_indices))
        body = asdict(self); claimed = body.pop("receipt_sha256")
        if stable_hash(body) != claimed:
            raise ValueError("raw-video event graph receipt hash mismatch")


@dataclass(frozen=True)
class LayerBTaskStateReceipt:
    task_id: str
    semantic_slots_receipt_sha256: str
    raw_event_graph_receipt_sha256: str
    shared_across_all_harness_arms: bool
    receipt_sha256: str

    @classmethod
    def create(cls, semantic: AGQASemanticSlotReceipt,
               grounding: RawVideoEventGraphReceipt) -> "LayerBTaskStateReceipt":
        semantic.validate(); grounding.validate()
        if semantic.task_id != grounding.task_id:
            raise ValueError("semantic slots and raw event graph refer to different tasks")
        if grounding.semantic_slots_sha256 != semantic.receipt_sha256:
            raise ValueError("raw event graph was not conditioned on the frozen semantic slots")
        known_slot_ids = {slot.slot_id for slot in semantic.slots}
        perceptual_slot_ids = {
            slot.slot_id for slot in semantic.slots
            if slot.kind in {"LITERAL", "ENTITY", "ACTION", "RELATION"}
        }
        unknown_bindings = {
            slot_id for event in grounding.events for slot_id in event.semantic_slot_ids
            if slot_id not in known_slot_ids
        }
        if unknown_bindings:
            raise ValueError("raw event graph binds unknown semantic slots")
        nonperceptual_bindings = {
            slot_id for event in grounding.events for slot_id in event.semantic_slot_ids
            if slot_id not in perceptual_slot_ids
        }
        if nonperceptual_bindings:
            raise ValueError("raw event graph binds non-perceptual semantic slots")
        body = {
            "task_id": semantic.task_id,
            "semantic_slots_receipt_sha256": semantic.receipt_sha256,
            "raw_event_graph_receipt_sha256": grounding.receipt_sha256,
            "shared_across_all_harness_arms": True,
        }
        return cls(**body, receipt_sha256=stable_hash(body))

    def validate(self) -> None:
        _sha(self.semantic_slots_receipt_sha256, "semantic_slots_receipt_sha256")
        _sha(self.raw_event_graph_receipt_sha256, "raw_event_graph_receipt_sha256")
        if not self.shared_across_all_harness_arms:
            raise ValueError("Layer-B target state must be shared by every Harness arm")
        body = asdict(self); claimed = body.pop("receipt_sha256")
        if stable_hash(body) != claimed:
            raise ValueError("Layer-B task-state receipt hash mismatch")


__all__ = [
    "AGQASemanticSlotReceipt", "ANSWER_KINDS", "GroundedEvent",
    "LayerBTaskStateReceipt", "RawVideoEventGraphReceipt", "SLOT_KINDS",
    "SemanticSlotNode",
]
