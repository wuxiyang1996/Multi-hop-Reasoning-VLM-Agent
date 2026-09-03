"""Closed, outcome-blind syntax normalization for AGQA operand receipts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .agqa_active_frame_grounder import parse_operand_receipt as _strict_parse
from .agqa_operand_normalization import (
    normalize_observation_interval_envelopes,
)


def normalize_operand_receipt_syntax(
    payload: Mapping[str, Any], *, frame_count: int,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Canonicalize existing evidence order, then close interval envelopes.

    The operation can only sort/deduplicate already emitted valid frame IDs
    and expand an OBSERVED interval to contain them.  It cannot create visual
    evidence, observations, labels, objects, confidence, or observability.
    """

    normalized = deepcopy(dict(payload))
    markers = []
    observations = normalized.get("observations")
    if isinstance(observations, list):
        for index, row in enumerate(observations):
            if not isinstance(row, dict):
                continue
            evidence = row.get("evidence_frames")
            if (
                not isinstance(evidence, list)
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    or value < 0 or value >= frame_count
                    for value in evidence
                )
            ):
                continue
            canonical = sorted(set(evidence))
            if canonical != evidence:
                row["evidence_frames"] = canonical
                occurrence = str(row.get("occurrence_id") or f"O{index}")
                markers.append(
                    f"{occurrence}:DETERMINISTIC_EVIDENCE_FRAME_ORDER"
                )
    normalized, envelope_markers = normalize_observation_interval_envelopes(
        normalized, frame_count=frame_count,
    )
    markers.extend(envelope_markers)
    if markers:
        canonicalizations = normalized.get("canonicalizations", [])
        if not isinstance(canonicalizations, list):
            canonicalizations = []
        normalized["canonicalizations"] = list(dict.fromkeys(
            [str(value) for value in canonicalizations] + markers
        ))
    return normalized, tuple(markers)


def parse_normalized_operand_receipt_v2(
    payload: Mapping[str, Any], *, expected_role: str,
    expected_operand: str, frame_count: int,
):
    normalized, _ = normalize_operand_receipt_syntax(
        payload, frame_count=frame_count,
    )
    return _strict_parse(
        normalized, expected_role=expected_role,
        expected_operand=expected_operand, frame_count=frame_count,
    )


__all__ = [
    "normalize_operand_receipt_syntax",
    "parse_normalized_operand_receipt_v2",
]
