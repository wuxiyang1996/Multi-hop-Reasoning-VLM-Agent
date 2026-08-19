"""Bound non-decision text in AGQA ontology structured outputs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


PROTOCOL_VERSION = "AGQA_QUERY_OBJECT_BOUNDED_ONTOLOGY_TEXT_V1"
MAX_FREE_TEXT_CHARACTERS = 160


def bounded_response_format(
    base: Mapping[str, Any], *, max_characters: int = MAX_FREE_TEXT_CHARACTERS,
) -> dict[str, Any]:
    """Add bounds only to explanatory strings, never to decision evidence."""

    if not isinstance(max_characters, int) or isinstance(max_characters, bool):
        raise ValueError("max_characters must be an integer")
    if max_characters < 32 or max_characters > 512:
        raise ValueError("max_characters must be in [32,512]")
    result = deepcopy(dict(base))
    properties = result["json_schema"]["schema"]["properties"]
    for field in ("visual_description", "uncertainty"):
        if properties[field].get("type") != "string":
            raise ValueError(f"unexpected ontology schema for {field}")
        properties[field]["maxLength"] = max_characters
    return result


def bounded_system_prompt(
    base: str, *, max_characters: int = MAX_FREE_TEXT_CHARACTERS,
) -> str:
    if max_characters != MAX_FREE_TEXT_CHARACTERS:
        raise ValueError("system prompt uses the frozen 160-character bound")
    return (
        base
        + " Keep visual_description and uncertainty concise; each must be at most "
        + str(max_characters)
        + " characters. Never spend output tokens on repeated explanation."
    )


__all__ = [
    "MAX_FREE_TEXT_CHARACTERS", "PROTOCOL_VERSION",
    "bounded_response_format", "bounded_system_prompt",
]
