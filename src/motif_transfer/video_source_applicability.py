"""Outcome-blind applicability receipts for game-induced video programs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .agqa_layer_b_contracts import AGQASemanticSlotReceipt
from .contracts import stable_hash
from .video_target_signature_binding import authorize_target_signature


_AGQA_ROOT_FAMILIES = {
    "ask whether a grounded event or relation exists": "presence_question",
    "ask whether a grounded set contains an item": "membership_question",
    "test semantic equality": "equality_condition",
    "require exactly one condition": "exclusive_condition",
    "require both conditions": "joint_condition",
    "select between explicitly named alternatives": "alternatives",
    "compare two grounded durations": "duration_choice",
    "select an extremal grounded duration": "duration_extremum",
    "request an attribute of the selected item": "goal",
}
_CLEVRER_FAMILIES = frozenset({
    "descriptive", "explanatory", "predictive", "counterfactual",
})


@dataclass(frozen=True)
class VideoApplicabilityReceipt:
    task_id: str
    target_domain: str
    question_family: str
    status: str
    reason: str
    parser_receipt_sha256: str
    source_algebra_sha256: str
    signature_authorization_sha256: str
    required_primitives: tuple[str, ...]
    required_compositions: tuple[tuple[str, str], ...]
    target_outcome_read: bool
    source_identity_used_as_feature: bool
    receipt_sha256: str


def classify_agqa_family(semantic: AGQASemanticSlotReceipt) -> str | None:
    """Classify an operator-free semantic root without reading a program."""

    semantic.validate()
    root = next(row for row in semantic.slots if row.slot_id == semantic.root_slot_id)
    return _AGQA_ROOT_FAMILIES.get(root.surface.casefold().strip())


def classify_clevrer_family(public_question_type: str) -> str | None:
    value = str(public_question_type).casefold().strip()
    return value if value in _CLEVRER_FAMILIES else None


def authorize_video_applicability(
    *,
    algebra: Mapping[str, Any],
    binding_spec: Mapping[str, Any],
    task_id: str,
    target_domain: str,
    parser_receipt_sha256: str,
    question_family: str | None,
    target_outcome_read: bool = False,
    source_identity_used_as_feature: bool = False,
) -> VideoApplicabilityReceipt:
    """Authorize iff the source graph contains the public typed signature."""

    if target_outcome_read:
        raise ValueError("target outcome cannot be used for applicability")
    if source_identity_used_as_feature:
        raise ValueError("source identity cannot be used as an applicability feature")
    family = str(question_family or "unknown_question_family").casefold()
    authorization = authorize_target_signature(
        algebra=algebra,
        binding_spec=binding_spec,
        target_domain=target_domain,
        question_family=family,
        target_outcome_read=False,
    )
    if question_family is None:
        status, reason = "ABSTAINED", "UNKNOWN_OR_AMBIGUOUS_PUBLIC_QUESTION_FAMILY"
    elif authorization.status != "AUTHORIZED":
        status, reason = "ABSTAINED", "SOURCE_TYPED_GRAPH_DOES_NOT_CONTAIN_SIGNATURE"
    else:
        status, reason = "AUTHORIZED", "SOURCE_TYPED_GRAPH_CONTAINS_SIGNATURE"
    body = {
        "task_id": str(task_id),
        "target_domain": str(target_domain).casefold(),
        "question_family": family,
        "status": status,
        "reason": reason,
        "parser_receipt_sha256": str(parser_receipt_sha256),
        "source_algebra_sha256": str(algebra["artifact_sha256"]),
        "signature_authorization_sha256": authorization.receipt_sha256,
        "required_primitives": authorization.required_primitives,
        "required_compositions": authorization.required_compositions,
        "target_outcome_read": False,
        "source_identity_used_as_feature": False,
    }
    return VideoApplicabilityReceipt(**body, receipt_sha256=stable_hash(body))


def receipt_dict(receipt: VideoApplicabilityReceipt) -> dict[str, Any]:
    return asdict(receipt)


__all__ = [
    "VideoApplicabilityReceipt", "authorize_video_applicability",
    "classify_agqa_family", "classify_clevrer_family", "receipt_dict",
]
