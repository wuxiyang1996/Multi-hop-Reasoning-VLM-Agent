"""Operator-free public-question receipts for CLEVRER Harness routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .clevrer_descriptive_compiler import compile_descriptive_question
from .clevrer_query_compiler import compile_choice, compile_question
from .contracts import stable_hash


_OBLIGATIONS = {
    "descriptive": ("grounded entity/event set", "public query intent"),
    "explanatory": ("observed target event", "causal antecedent membership"),
    "predictive": ("observed event prefix", "future event membership"),
    "counterfactual": ("removed entity", "alternate-world event membership"),
}


@dataclass(frozen=True)
class CLEVRERPublicSemanticReceipt:
    task_id: str
    question_sha256: str
    question_family: str
    public_subtype: str
    answer_kind: str
    semantic_obligations: tuple[str, ...]
    choice_sha256s: tuple[str, ...]
    negated_query: bool
    parser_authority: str
    operator_sequence_emitted: bool
    functional_program_read: bool
    answer_read: bool
    target_outcome_read: bool
    receipt_sha256: str


def parse_public_semantics(
    *, task_id: str, question: str, question_family: str,
    public_subtype: str = "", choices: Sequence[str] = (),
) -> CLEVRERPublicSemanticReceipt:
    family = str(question_family).casefold().strip()
    if family not in _OBLIGATIONS:
        raise ValueError("unknown CLEVRER public question family")
    # Syntax validation happens inside the target-native executor adapter, but
    # its private tokens are discarded and never enter the Harness receipt.
    if family == "descriptive":
        compile_descriptive_question(question, public_subtype)
        answer_kind = {
            "count": "COUNT", "exist": "BOOLEAN",
            "query_color": "ATTRIBUTE", "query_material": "ATTRIBUTE",
            "query_shape": "ATTRIBUTE",
        }.get(str(public_subtype).casefold())
        if answer_kind is None:
            raise ValueError("unknown CLEVRER descriptive public subtype")
    else:
        compile_question(question, family)
        for choice in choices:
            compile_choice(choice, family)
        answer_kind = "CHOICE_VECTOR"
    body = {
        "task_id": str(task_id), "question_sha256": stable_hash(str(question)),
        "question_family": family, "public_subtype": str(public_subtype).casefold(),
        "answer_kind": answer_kind, "semantic_obligations": _OBLIGATIONS[family],
        "choice_sha256s": tuple(stable_hash(str(choice)) for choice in choices),
        "negated_query": " not " in f" {str(question).casefold()} ",
        "parser_authority": "PUBLIC_QUESTION_AND_CHOICE_TEXT_ONLY;NO_PROGRAM_OR_OUTCOME",
        "operator_sequence_emitted": False, "functional_program_read": False,
        "answer_read": False, "target_outcome_read": False,
    }
    return CLEVRERPublicSemanticReceipt(**body, receipt_sha256=stable_hash(body))


__all__ = ["CLEVRERPublicSemanticReceipt", "parse_public_semantics"]
