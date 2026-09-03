from __future__ import annotations

import json
from pathlib import Path

import pytest

from motif_transfer.agqa_layer_b_contracts import AGQASemanticSlotReceipt, SemanticSlotNode
from motif_transfer.source_video_operator_algebra import induce_source_video_algebra
from motif_transfer.video_source_applicability import (
    authorize_video_applicability,
    classify_agqa_family,
    classify_clevrer_family,
)


ROOT = Path(__file__).resolve().parents[1]


def _algebra_and_bindings() -> tuple[dict, dict]:
    catalog = json.loads((ROOT / "configs/full_video_source_catalog_v1.json").read_text())
    algebra = induce_source_video_algebra(root=ROOT, catalog=catalog)
    bindings = json.loads((ROOT / "configs/video_target_signature_bindings_v1.json").read_text())
    return algebra, bindings


def _semantic(surface: str) -> AGQASemanticSlotReceipt:
    return AGQASemanticSlotReceipt.create(
        task_id="q1", question_sha256="1" * 64, answer_kind="BOOLEAN",
        root_slot_id="S0", slots=(SemanticSlotNode("S0", "QUERY_GOAL", surface),),
        parser_sha256="2" * 64,
        parser_training_authority="PUBLIC_QUESTION_TEXT_ONLY;NO_RUNTIME_PROGRAM",
    )


def test_agqa_root_and_clevrer_public_type_classification() -> None:
    assert classify_agqa_family(_semantic(
        "ask whether a grounded event or relation exists"
    )) == "presence_question"
    assert classify_agqa_family(_semantic("an unknown semantic root")) is None
    assert classify_clevrer_family("Counterfactual") == "counterfactual"
    assert classify_clevrer_family("answer") is None


def test_source_graph_authorizes_known_family_and_abstains_unknown() -> None:
    algebra, bindings = _algebra_and_bindings()
    allowed = authorize_video_applicability(
        algebra=algebra, binding_spec=bindings, task_id="q1",
        target_domain="clevrer", parser_receipt_sha256="3" * 64,
        question_family="counterfactual",
    )
    assert allowed.status == "AUTHORIZED"
    assert allowed.target_outcome_read is False
    denied = authorize_video_applicability(
        algebra=algebra, binding_spec=bindings, task_id="q2",
        target_domain="clevrer", parser_receipt_sha256="4" * 64,
        question_family=None,
    )
    assert denied.status == "ABSTAINED"


def test_outcome_and_source_identity_are_forbidden() -> None:
    algebra, bindings = _algebra_and_bindings()
    common = dict(
        algebra=algebra, binding_spec=bindings, task_id="q1",
        target_domain="agqa", parser_receipt_sha256="5" * 64,
        question_family="presence_question",
    )
    with pytest.raises(ValueError, match="target outcome"):
        authorize_video_applicability(**common, target_outcome_read=True)
    with pytest.raises(ValueError, match="source identity"):
        authorize_video_applicability(**common, source_identity_used_as_feature=True)
