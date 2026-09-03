"""Oracle-free text-to-program compiler for CLEVRER causal questions.

The official CLEVRER multiple-choice questions use a small compositional
language.  This module compiles the public question and choice text into that
language without reading answer labels, scene annotations, or official
functional programs at runtime.  Official programs may be used only by a
separate adaptation/train audit.

Only the three causal families used by the video-transfer protocol are in
scope: explanatory, predictive, and counterfactual.  Unsupported or ambiguous
surface forms fail closed instead of silently producing a plausible program.
"""

from __future__ import annotations

import re
from typing import Sequence


COLORS = frozenset({
    "blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow",
})
MATERIALS = frozenset({"metal", "rubber"})
SHAPES = frozenset({"cube", "cylinder", "sphere"})


def _clean(text: str) -> str:
    value = text.strip().lower().replace("grey", "gray")
    value = value.replace("’", "'")
    value = re.sub(r"[?.!,]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _strip_nominal_scaffolding(text: str) -> str:
    value = _clean(text)
    value = re.sub(r"^(?:the|a|an)\s+", "", value)
    value = re.sub(r"(?:'s)?$", "", value)
    value = re.sub(r"\bobject$", "", value)
    return value.strip()


def compile_object(description: str) -> list[str]:
    """Compile a CLEVRER object description into canonical filter tokens."""

    value = _strip_nominal_scaffolding(description)
    words = set(value.split())
    attributes: list[tuple[str, str]] = []
    for vocabulary, operator in (
        (COLORS, "filter_color"),
        (MATERIALS, "filter_material"),
        (SHAPES, "filter_shape"),
    ):
        matches = sorted(words & vocabulary)
        if len(matches) > 1:
            raise ValueError(f"ambiguous CLEVRER object description: {description!r}")
        if matches:
            attributes.append((matches[0], operator))
    if not attributes:
        raise ValueError(f"object description has no supported attribute: {description!r}")
    program = ["objects"]
    for attribute, operator in attributes:
        program.extend((attribute, operator))
    program.append("unique")
    return program


def _split_collision(text: str) -> tuple[str, str]:
    value = _clean(text)
    value = re.sub(r"^(?:the\s+)?collision between\s+", "", value)
    value = re.sub(r"^(?:the\s+)?collision of\s+", "", value)
    possessive = re.fullmatch(r"(.+?)(?:'s) colliding with (.+)", value)
    if possessive:
        return possessive.group(1), possessive.group(2)
    direct = re.fullmatch(r"(.+?) collides with (.+)", value)
    if direct:
        return direct.group(1), direct.group(2)
    plural = re.fullmatch(r"(.+?) and (.+?) collide", value)
    if plural:
        return plural.group(1), plural.group(2)
    between = re.fullmatch(r"(.+?) and (.+)", value)
    if between:
        return between.group(1), between.group(2)
    raise ValueError(f"unsupported CLEVRER collision form: {text!r}")


def _compile_collision(text: str, *, event_root: str) -> list[str]:
    left, right = _split_collision(text)
    return [event_root] + compile_object(left) + ["filter_collision"] + (
        compile_object(right) + ["filter_collision", "unique"]
    )


def _removed_object(question: str) -> str:
    value = _clean(question)
    patterns = (
        r"without (.+?)(?: which| what| will|$)",
        r"if (.+?) is removed",
        r"if (.+?) were removed",
    )
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    raise ValueError(f"cannot identify counterfactual object: {question!r}")


def compile_question(question: str, question_type: str) -> list[str]:
    """Compile public question text into an official-executor program."""

    family = question_type.strip().lower()
    value = _clean(question)
    negate = " not " in f" {value} "
    if family == "predictive":
        if "happen next" not in value:
            raise ValueError(f"unsupported predictive question: {question!r}")
        return ["unseen_events", "belong_to"]
    if family == "counterfactual":
        program = ["all_events"] + compile_object(_removed_object(question))
        program.extend(("filter_counterfact", "belong_to"))
        if negate:
            program.append("negate")
        return program
    if family != "explanatory":
        raise ValueError(f"unsupported CLEVRER causal family: {question_type!r}")

    marker = "responsible for "
    if marker not in value:
        raise ValueError(f"unsupported explanatory question: {question!r}")
    event = value.split(marker, 1)[1]
    if "collision between " in event or "colliding with" in event:
        event_program = _compile_collision(event, event_root="events")
    else:
        exit_match = re.fullmatch(
            r"(.+?)(?:'s) (?:exit|exiting the scene)", event,
        )
        if not exit_match:
            raise ValueError(f"unsupported explanatory event: {event!r}")
        event_program = ["events"] + compile_object(exit_match.group(1))
        event_program.extend(("filter_out", "unique"))
    program = ["events"] + event_program + ["filter_ancestor", "belong_to"]
    if negate:
        program.append("negate")
    return program


def compile_choice(choice: str, question_type: str) -> list[str]:
    """Compile one public answer choice for a causal CLEVRER question."""

    family = question_type.strip().lower()
    value = _clean(choice)
    if value.startswith("the presence of "):
        return compile_object(value.removeprefix("the presence of "))
    entrance = re.fullmatch(r"(.+?)(?:'s) (?:entrance|entering the scene)", value)
    if entrance:
        return ["events"] + compile_object(entrance.group(1)) + ["filter_in", "unique"]
    event_root = "events" if family == "explanatory" else "all_events"
    if family not in {"explanatory", "predictive", "counterfactual"}:
        raise ValueError(f"unsupported CLEVRER causal family: {question_type!r}")
    return _compile_collision(value, event_root=event_root)


def normalize_official_program(program: Sequence[str]) -> list[str]:
    """Normalize the legacy counterfactual operator used by annotations."""

    return [
        "filter_counterfact" if token == "get_counterfact" else str(token)
        for token in program
    ]


__all__ = [
    "compile_choice",
    "compile_object",
    "compile_question",
    "normalize_official_program",
]
