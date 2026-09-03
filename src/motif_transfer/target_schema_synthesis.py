"""Typed evaluation for zero-trajectory target-schema program synthesis."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .contracts import stable_hash


FAMILIES = {
    "RECURRENT_RELATIONAL",
    "FINITE_STRUCTURAL_DELTA",
    "CYCLIC_IDENTITY_RECOVERY",
    "GENERIC_SEARCH_COMMIT",
    "ABSTAIN",
}

EXPECTED_PROGRAMS: dict[str, dict[str, Any]] = {
    "alfworld": {
        "program_family": "RECURRENT_RELATIONAL",
        "operators": [
            "CONTROL_STATE_UPDATE",
            "POSITIVE_EFFECT_BINDING_UPDATE",
            "ENTITY_GOAL_RELATION_UPDATE",
        ],
        "constraints": [
            "RECURRENT_CONTROL_UNTIL_BINDING",
            "BINDING_BEFORE_RELATION",
            "RECURRENT_RELATION_UPDATE",
        ],
        "terminal": "RELATION_COVERAGE_COMPLETE",
        "abstention": [
            "AMBIGUOUS_EFFECT",
            "MULTIPLE_BINDINGS",
            "ZERO_BINDINGS",
        ],
    },
    "discoveryworld": {
        "program_family": "FINITE_STRUCTURAL_DELTA",
        "operators": ["ADD_ENTITY_SLOT", "REMOVE_ENTITY_SLOT"],
        "constraints": ["ADD_BEFORE_REMOVE"],
        "terminal": "TARGET_NATIVE_OUTCOME_AFTER_SEQUENCE",
        "abstention": ["AMBIGUOUS_EFFECT", "MULTIPLE_BINDINGS"],
    },
    "tir_rotation": {
        "program_family": "CYCLIC_IDENTITY_RECOVERY",
        "operators": ["PROBE_EFFECT", "RECOVERY_EFFECT"],
        "constraints": ["COMPOSE_TO_IDENTITY"],
        "terminal": "IDENTITY_EQUALITY",
        "abstention": ["MULTIPLE_BINDINGS", "ZERO_BINDINGS"],
    },
}


def canonical_program(value: Mapping[str, Any]) -> dict[str, Any]:
    family = str(value.get("program_family") or "").upper()
    if family not in FAMILIES:
        raise ValueError("unknown synthesized program family")

    def strings(field: str) -> list[str]:
        raw = value.get(field)
        if not isinstance(raw, list):
            raise ValueError(f"synthesized {field} is not a list")
        return sorted({str(item).upper() for item in raw})

    return {
        "program_family": family,
        "operators": strings("operators"),
        "constraints": strings("constraints"),
        "terminal": str(value.get("terminal") or "").upper(),
        "abstention": strings("abstention"),
    }


def expected_program(target: str) -> dict[str, Any]:
    if target not in EXPECTED_PROGRAMS:
        raise ValueError(f"unknown synthesis target: {target}")
    return canonical_program(EXPECTED_PROGRAMS[target])


def parse_program_response(text: str) -> dict[str, Any]:
    """Parse one JSON object without repairing semantic content."""

    candidate = str(text).strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if match is None:
            raise ValueError("model response contains no JSON object") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return canonical_program(value)


def score_program(target: str, program: Mapping[str, Any]) -> dict[str, Any]:
    observed = canonical_program(program)
    expected = expected_program(target)
    fields = {
        field: observed[field] == expected[field]
        for field in expected
    }
    return {
        "target": target,
        "exact_program_match": all(fields.values()),
        "field_matches": fields,
        "observed_program": observed,
        "expected_program_sha256": stable_hash(expected),
    }


def synthesis_prompt(
    target: str, interface_description: str, *, variant: int,
) -> str:
    """Build a target-only prompt with the fixed shared IR grammar."""

    expected_program(target)  # validate target without exporting the answer
    order_note = (
        "List each set in lexical order."
        if variant % 2 == 0
        else "Array order is ignored by the evaluator."
    )
    return f"""You must synthesize a reusable control program from a target
interface specification. You receive NO successful trajectory, reward,
outcome, source-game evidence, source program, or answer key.

TARGET INTERFACE ({target}):
{interface_description}

Return exactly one JSON object using this shared IR grammar:
{{
  "program_family": one of ["RECURRENT_RELATIONAL",
    "FINITE_STRUCTURAL_DELTA", "CYCLIC_IDENTITY_RECOVERY",
    "GENERIC_SEARCH_COMMIT", "ABSTAIN"],
  "operators": subset of ["CONTROL_STATE_UPDATE",
    "POSITIVE_EFFECT_BINDING_UPDATE", "ENTITY_GOAL_RELATION_UPDATE",
    "ADD_ENTITY_SLOT", "REMOVE_ENTITY_SLOT", "PROBE_EFFECT",
    "RECOVERY_EFFECT"],
  "constraints": subset of ["RECURRENT_CONTROL_UNTIL_BINDING",
    "BINDING_BEFORE_RELATION", "RECURRENT_RELATION_UPDATE",
    "ADD_BEFORE_REMOVE", "COMPOSE_TO_IDENTITY"],
  "terminal": one of ["RELATION_COVERAGE_COMPLETE",
    "TARGET_NATIVE_OUTCOME_AFTER_SEQUENCE", "IDENTITY_EQUALITY",
    "TARGET_NATIVE_SUCCESS", "ABSTAIN"],
  "abstention": subset of ["ZERO_BINDINGS", "MULTIPLE_BINDINGS",
    "AMBIGUOUS_EFFECT"]
}}

Choose only relations justified by the interface. If it does not identify a
unique program, choose ABSTAIN with empty arrays and terminal ABSTAIN.
{order_note} Do not include prose or markdown."""


__all__ = [
    "EXPECTED_PROGRAMS",
    "canonical_program",
    "expected_program",
    "parse_program_response",
    "score_program",
    "synthesis_prompt",
]
