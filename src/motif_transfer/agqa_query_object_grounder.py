"""Ontology-aware, candidate-blind grounding for AGQA QUERY_OBJECT tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence

from .agqa_active_frame_grounder import AGQAQueryPlan


# This is the public STAR/Charades object taxonomy, not a per-question answer
# list. Surface aliases are collapsed to the nouns used by AGQA answers.
AGQA_OBJECT_ONTOLOGY = (
    "bag", "bed", "blanket", "book", "box", "broom", "chair", "closet",
    "clothes", "cup", "dish", "door", "doorknob", "doorway", "floor",
    "food", "groceries", "hands", "laptop", "light", "medicine", "mirror",
    "paper", "phone", "picture", "pillow", "refrigerator", "sandwich",
    "shelf", "shoe", "sofa", "table", "television", "towel", "vacuum",
    "window",
)

_ALIASES = {
    "cabinet": "closet",
    "closet cabinet": "closet",
    "notebook": "paper",
    "paper notebook": "paper",
    "camera": "phone",
    "phone camera": "phone",
    "couch": "sofa",
    "sofa couch": "sofa",
    "glass": "cup",
    "bottle": "cup",
    "cup glass bottle": "cup",
    "tv": "television",
    "doorknob handle": "doorknob",
}

_NON_ATOMIC_SCOPE = re.compile(
    r"\b(?:before|after|while|between|first|last)\b|"
    r"\b(?:object|thing) they\b|\bdoing\b"
)


def canonical_object_label(value: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", str(value).casefold())
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    text = re.sub(r"\b(?:some|something)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text in _ALIASES:
        return _ALIASES[text]
    if text.endswith("s") and text[:-1] in AGQA_OBJECT_ONTOLOGY:
        return text[:-1]
    return text


def atomic_query_object_plan(plan: AGQAQueryPlan) -> bool:
    """Return whether QUERY_OBJECT has one explicit relation and no subquery."""

    return (
        plan.comparison == "QUERY_OBJECT"
        and plan.obligation_kind == "RELATION_RECURRENT"
        and not _NON_ATOMIC_SCOPE.search(plan.operand_a.casefold())
        and "unknown object" in plan.visual_query_a.casefold()
        and not plan.operand_b
        and not plan.visual_query_b
    )


@dataclass(frozen=True)
class AGQAObjectOntologyReceipt:
    decision: str
    relation_observed: bool
    confidence: float
    evidence_frames: tuple[int, ...]
    visual_description: str
    uncertainty: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_frames"] = list(self.evidence_frames)
        return payload


def parse_object_ontology_receipt(
    payload: Mapping[str, Any], *, frame_count: int,
    ontology: Sequence[str] = AGQA_OBJECT_ONTOLOGY,
) -> AGQAObjectOntologyReceipt:
    forbidden = {
        "answer", "gold", "gold_answer", "correct", "correctness",
        "functional_program", "scene_graph", "source_identity",
        "direct_response", "answer_candidates",
    }
    leaked = forbidden & {str(key).casefold() for key in payload}
    if leaked:
        raise ValueError(f"object ontology receipt leaked forbidden fields: {sorted(leaked)}")
    allowed = {canonical_object_label(value) for value in ontology}
    raw_decision = str(payload.get("decision") or "").strip().casefold()
    decision = (
        "unknown" if raw_decision == "unknown"
        else canonical_object_label(raw_decision)
    )
    if decision not in allowed | {"unknown"}:
        raise ValueError("object ontology decision is outside the frozen taxonomy")
    observed = payload.get("relation_observed")
    if not isinstance(observed, bool):
        raise ValueError("relation_observed must be boolean")
    confidence = float(payload.get("confidence", -1.0))
    if not 0 <= confidence <= 1:
        raise ValueError("object ontology confidence must be in [0,1]")
    raw_frames = payload.get("evidence_frames")
    if not isinstance(raw_frames, list) or len(raw_frames) > 6:
        raise ValueError("object ontology receipt may cite at most six frames")
    frames: list[int] = []
    for raw in raw_frames:
        if isinstance(raw, bool):
            raise ValueError("object ontology frame IDs must be integers")
        index = int(raw)
        if index < 0 or index >= frame_count:
            raise ValueError("object ontology evidence frame is out of range")
        if index not in frames:
            frames.append(index)
    if frames != sorted(frames):
        raise ValueError("object ontology evidence must be chronological")
    if observed and (decision == "unknown" or not frames):
        raise ValueError("an observed relation requires an object and evidence")
    if not observed and decision != "unknown":
        raise ValueError("an unobserved relation cannot name an object")
    return AGQAObjectOntologyReceipt(
        decision=decision,
        relation_observed=observed,
        confidence=confidence,
        evidence_frames=tuple(frames),
        visual_description=str(payload.get("visual_description") or "").strip(),
        uncertainty=str(payload.get("uncertainty") or "").strip(),
    )


def calibrate_query_object_execution(
    *, base_decision: str | None, direct_response: str,
    ontology_receipt: AGQAObjectOntologyReceipt,
    minimum_confidence: float,
) -> dict[str, Any]:
    """Require agreement between isolated relation and ontology neural views."""

    base = canonical_object_label(base_decision or "")
    ontology = canonical_object_label(ontology_receipt.decision)
    decision = None
    authorization_class = "ABSTAIN"
    reason = "BASE_AND_ONTOLOGY_VIEWS_DO_NOT_AGREE"
    dual_view_agreement = (
        base in AGQA_OBJECT_ONTOLOGY
        and base == ontology
        and ontology_receipt.relation_observed
        and ontology_receipt.confidence >= minimum_confidence
        and bool(ontology_receipt.evidence_frames)
    )
    if dual_view_agreement:
        decision = ontology
        direct = canonical_object_label(direct_response)
        if direct == decision:
            authorization_class = "AGREEMENT"
            reason = "DIRECT_AND_TWO_TARGET_NATIVE_VIEWS_AGREE"
        else:
            authorization_class = "SOURCE_TYPED_OVERRIDE"
            reason = "SOURCE_RECURRENCE_WITH_TWO_TARGET_NATIVE_OBJECT_VIEWS"
    core = {
        "schema_version": "agqa-query-object-calibration-v1",
        "decision": decision,
        "authorization_class": authorization_class,
        "reason": reason,
        "base_object": base or None,
        "ontology_object": ontology,
        "ontology_confidence": ontology_receipt.confidence,
        "minimum_confidence": minimum_confidence,
        "relation_observed": ontology_receipt.relation_observed,
        "answer_read": False,
        "functional_program_read": False,
        "scene_graph_read": False,
        "source_identity_read": False,
    }
    return core


__all__ = [
    "AGQA_OBJECT_ONTOLOGY", "AGQAObjectOntologyReceipt",
    "atomic_query_object_plan", "calibrate_query_object_execution",
    "canonical_object_label", "parse_object_ontology_receipt",
]
