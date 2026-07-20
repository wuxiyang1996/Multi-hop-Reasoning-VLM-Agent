"""Exact parser for ALFWorld's admissible text-command grammar.

Parsing is anchored and operates only on commands supplied by the environment.
It never resolves substrings or invents entities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence


@dataclass(frozen=True)
class ParsedAlfworldAction:
    raw: str
    operator: str
    arguments: Mapping[str, str] = field(default_factory=dict)
    argument_types: Mapping[str, str] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        types = ",".join(f"{key}:{self.argument_types[key]}" for key in sorted(self.argument_types))
        return f"{self.operator}({types})"


_PATTERNS: Sequence[
    tuple[re.Pattern[str], str, Mapping[str, str]]
] = (
    (re.compile(r"^go to (?P<location>.+)$", re.I), "GOTO", {"location": "location"}),
    (re.compile(r"^open (?P<receptacle>.+)$", re.I), "OPEN", {"receptacle": "receptacle"}),
    (re.compile(r"^close (?P<receptacle>.+)$", re.I), "CLOSE", {"receptacle": "receptacle"}),
    (
        re.compile(r"^take (?P<object>.+) from (?P<receptacle>.+)$", re.I),
        "TAKE",
        {"object": "object", "receptacle": "receptacle"},
    ),
    (
        re.compile(r"^put (?P<object>.+) (?P<relation>in/on|in|on) (?P<receptacle>.+)$", re.I),
        "PUT",
        {"object": "object", "relation": "relation", "receptacle": "receptacle"},
    ),
    (
        re.compile(r"^move (?P<object>.+) to (?P<receptacle>.+)$", re.I),
        "MOVE_TO",
        {"object": "object", "receptacle": "receptacle"},
    ),
    (
        re.compile(r"^(?P<operator>heat|cool|clean) (?P<object>.+) with (?P<tool>.+)$", re.I),
        "TRANSFORM",
        {"object": "object", "tool": "tool"},
    ),
    (re.compile(r"^toggle (?P<object>.+)$", re.I), "TOGGLE", {"object": "object"}),
    (re.compile(r"^use (?P<object>.+)$", re.I), "USE", {"object": "object"}),
    (re.compile(r"^examine (?P<object>.+)$", re.I), "EXAMINE", {"object": "object"}),
)


def parse_alfworld_action(command: str, *, admissible: Sequence[str]) -> ParsedAlfworldAction:
    """Parse an exact command that is present in the current admissible set."""
    if command not in admissible:
        raise ValueError(f"command is not exactly admissible: {command!r}")
    normalized = " ".join(command.strip().split())
    if normalized.lower() in {"look", "inventory"}:
        return ParsedAlfworldAction(normalized, normalized.upper())
    for pattern, operator, types in _PATTERNS:
        match = pattern.fullmatch(normalized)
        if match is None:
            continue
        arguments = {key: value.strip() for key, value in match.groupdict().items() if value is not None}
        resolved_operator = operator
        if operator == "TRANSFORM":
            resolved_operator = arguments.pop("operator").upper()
        return ParsedAlfworldAction(
            raw=normalized,
            operator=resolved_operator,
            arguments=arguments,
            argument_types={key: types[key] for key in arguments},
        )
    raise ValueError(f"unsupported ALFWorld command grammar: {command!r}")


__all__ = ["ParsedAlfworldAction", "parse_alfworld_action"]
