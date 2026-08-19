"""Outcome-blind normalization of AGQA operand receipt interval syntax."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .agqa_active_frame_grounder import parse_operand_receipt as _strict_parse


def normalize_observation_interval_envelopes(
    payload: Mapping[str, Any], *, frame_count: int,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Make an OBSERVED interval contain its own valid cited evidence frames.

    This repairs only a representational inconsistency. It cannot create an
    observation, object label, evidence frame, or confidence value.
    """

    normalized = deepcopy(dict(payload))
    observations = normalized.get("observations")
    if not isinstance(observations, list):
        return normalized, ()
    markers = []
    for index, row in enumerate(observations):
        if not isinstance(row, dict) or row.get("observability") != "OBSERVED":
            continue
        evidence = row.get("evidence_frames")
        start, end = row.get("start_frame"), row.get("end_frame")
        if (
            not isinstance(evidence, list) or not evidence
            or isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in evidence)
            or any(value < 0 or value >= frame_count for value in evidence)
        ):
            continue
        repaired_start = min(start, *evidence)
        repaired_end = max(end, *evidence)
        if repaired_start != start or repaired_end != end:
            row["start_frame"] = repaired_start
            row["end_frame"] = repaired_end
            occurrence = str(row.get("occurrence_id") or f"O{index}")
            markers.append(f"{occurrence}:DETERMINISTIC_INTERVAL_EVIDENCE_ENVELOPE")
    if markers:
        canonicalizations = normalized.get("canonicalizations", [])
        if not isinstance(canonicalizations, list):
            canonicalizations = []
        normalized["canonicalizations"] = list(dict.fromkeys(
            [str(value) for value in canonicalizations] + markers
        ))
    return normalized, tuple(markers)


def parse_normalized_operand_receipt(
    payload: Mapping[str, Any], *, expected_role: str,
    expected_operand: str, frame_count: int,
):
    normalized, _ = normalize_observation_interval_envelopes(
        payload, frame_count=frame_count,
    )
    return _strict_parse(
        normalized, expected_role=expected_role,
        expected_operand=expected_operand, frame_count=frame_count,
    )


__all__ = [
    "normalize_observation_interval_envelopes",
    "parse_normalized_operand_receipt",
]
