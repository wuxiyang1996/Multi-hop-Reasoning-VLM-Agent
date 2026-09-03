"""Outcome-blind binding of target-native question families to source algebra."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

from .contracts import stable_hash


@dataclass(frozen=True)
class SignatureAuthorization:
    target_domain: str
    question_family: str
    status: str
    required_primitives: tuple[str, ...]
    required_compositions: tuple[tuple[str, str], ...]
    missing_primitives: tuple[str, ...]
    missing_compositions: tuple[tuple[str, str], ...]
    source_algebra_sha256: str
    binding_spec_sha256: str
    target_outcome_read: bool
    receipt_sha256: str


def _validate_algebra(algebra: Mapping[str, Any]) -> None:
    if algebra.get("status") != "SOURCE_VIDEO_OPERATOR_ALGEBRA_QUALIFIED":
        raise ValueError("source video algebra is not qualified")
    if algebra.get("target_data_read") is not False:
        raise ValueError("source video algebra crossed target authority")
    if algebra.get("composition_rule") != "EXACT_OUTPUT_TO_INPUT_TYPE_CLOSURE":
        raise ValueError("source composition rule drift")


def authorize_target_signature(
    *,
    algebra: Mapping[str, Any],
    binding_spec: Mapping[str, Any],
    target_domain: str,
    question_family: str,
    target_outcome_read: bool = False,
) -> SignatureAuthorization:
    _validate_algebra(algebra)
    if target_outcome_read:
        raise ValueError("target outcome cannot authorize source applicability")
    if binding_spec.get("status") != "TARGET_NATIVE_SIGNATURES_FROZEN_BEFORE_NEW_TARGET_RESERVES":
        raise ValueError("target signature binding is not frozen")
    if "NO_TARGET_OUTCOME" not in str(binding_spec.get("authority")):
        raise ValueError("target signature binding authority drift")
    domain = str(target_domain).casefold()
    family = str(question_family).casefold()
    domain_rows = binding_spec.get(domain)
    if not isinstance(domain_rows, Mapping):
        required: tuple[str, ...] = ()
        missing = ("UNKNOWN_TARGET_DOMAIN",)
    elif family not in domain_rows:
        required = ()
        required_edges: tuple[tuple[str, str], ...] = ()
        missing = ("UNKNOWN_QUESTION_FAMILY",)
        missing_edges: tuple[tuple[str, str], ...] = ()
    else:
        declaration = domain_rows[family]
        if not isinstance(declaration, Mapping):
            raise ValueError("target signature declaration must be a mapping")
        required = tuple(str(x) for x in declaration.get("primitives") or ())
        required_edges = tuple(
            (str(edge[0]), str(edge[1]))
            for edge in declaration.get("compositions") or ()
            if len(edge) == 2
        )
        available = {str(x) for x in algebra.get("primitive_names") or []}
        missing = tuple(sorted(set(required) - available))
        available_edges = {
            (str(edge[0]), str(edge[1]))
            for edge in algebra.get("composition_edges") or ()
            if len(edge) == 2
        }
        missing_edges = tuple(sorted(set(required_edges) - available_edges))
    if not isinstance(domain_rows, Mapping):
        required_edges = ()
        missing_edges = ()
    body = {
        "target_domain": domain,
        "question_family": family,
        "status": "AUTHORIZED" if required and not missing and not missing_edges else "ABSTAINED",
        "required_primitives": required,
        "required_compositions": required_edges,
        "missing_primitives": missing,
        "missing_compositions": missing_edges,
        "source_algebra_sha256": str(algebra["artifact_sha256"]),
        "binding_spec_sha256": stable_hash(binding_spec),
        "target_outcome_read": False,
    }
    return SignatureAuthorization(**body, receipt_sha256=stable_hash(body))


def permuted_algebra(algebra: Mapping[str, Any]) -> dict[str, Any]:
    """Isomorphic control that deranges semantic labels, not graph capacity.

    The whole typed signature and every incident composition edge move together
    to a different primitive name.  Consequently the control has exactly the
    same number of primitives and edges, and remains a valid type-closure; only
    the association between a public semantic label and source-induced
    structure is destroyed.
    """

    _validate_algebra(algebra)
    rows = [dict(row) for row in algebra.get("primitives") or []]
    ordered = sorted(rows, key=lambda row: str(row["name"]))
    names = [str(row["name"]) for row in ordered]
    if len(names) < 2:
        raise ValueError("cannot permute a degenerate source algebra")
    rotated = names[1:] + names[:1]
    name_map = dict(zip(names, rotated))
    for row in ordered:
        row["name"] = name_map[str(row["name"])]
    ordered.sort(key=lambda row: str(row["name"]))
    edges = sorted(
        (name_map[str(left)], name_map[str(right)])
        for left, right in algebra.get("composition_edges") or []
    )
    body = dict(algebra)
    body["primitives"] = ordered
    body["composition_edges"] = [list(edge) for edge in edges]
    body["composition_edge_count"] = len(edges)
    body["control"] = "FIXED_SEMANTIC_LABEL_DERANGEMENT_PRESERVING_TYPED_GRAPH_ISOMORPHISM"
    body["semantic_label_map"] = name_map
    body["authentic_artifact_sha256"] = str(algebra["artifact_sha256"])
    body["artifact_sha256"] = stable_hash({k: v for k, v in body.items() if k != "artifact_sha256"})
    return body


__all__ = ["SignatureAuthorization", "authorize_target_signature", "permuted_algebra"]
