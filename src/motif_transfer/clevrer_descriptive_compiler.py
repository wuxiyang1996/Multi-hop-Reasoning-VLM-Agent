"""Public-text compiler for CLEVRER descriptive questions.

This is a target-native executor adapter.  The Harness never receives these
NS-DR tokens; its parser receipt contains only the operator-free public family
and semantic signature.  Runtime compilation reads question text/subtype but
not official programs, annotations, or answers.
"""

from __future__ import annotations

import re


COLORS = ("gray", "red", "blue", "green", "brown", "yellow", "cyan", "purple")
MATERIALS = ("metal", "rubber")
SHAPES = ("sphere", "cylinder", "cube")


def _clean(text: str) -> str:
    value = text.casefold().replace("grey", "gray").replace("’", "'")
    value = re.sub(r"[?.!,]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _object_program(description: str) -> list[str]:
    words = set(_clean(description).split())
    program = ["objects"]
    for vocabulary, operation in (
        (COLORS, "filter_color"), (MATERIALS, "filter_material"),
        (SHAPES, "filter_shape"),
    ):
        matches = [value for value in vocabulary if value in words or f"{value}s" in words]
        if len(matches) > 1:
            raise ValueError(f"ambiguous object attributes: {description!r}")
        if matches:
            program.extend((matches[0], operation))
    return program


def _event_name(text: str) -> str:
    value = _clean(text)
    if re.search(r"\b(?:enter|enters|entering|entrance)\b", value): return "filter_in"
    if re.search(r"\b(?:exit|exits|exiting)\b", value): return "filter_out"
    if re.search(r"\b(?:collide|collides|collision|collisions)\b", value): return "filter_collision"
    raise ValueError(f"cannot identify event kind: {text!r}")


def _anchor(text: str) -> list[str]:
    value = _clean(text)
    match = re.fullmatch(r"(?:the )?(.+?) (enters|enter|exits|exit)(?: the)? scene", value)
    if match is None:
        raise ValueError(f"unsupported temporal anchor: {text!r}")
    return ["events", *_object_program(match.group(1)), "unique", _event_name(match.group(2)), "unique"]


def _time_program(question: str) -> list[str]:
    value = _clean(question)
    if "when the video begins" in value:
        return ["events", "filter_start", "query_frame"]
    if "when the video ends" in value:
        return ["events", "filter_end", "query_frame"]
    match = re.search(r"\bwhen (?:the )?(.+? (?:enters|exits)(?: the)? scene)$", value)
    if match:
        return [*_anchor(match.group(1)), "query_frame"]
    return ["null"]


def _temporal_event_prefix(question: str) -> tuple[list[str], str]:
    value = _clean(question)
    match = re.search(r"\b(before|after) (?:the )?(.+? (?:enters|exits)(?: the)? scene)$", value)
    if match is None:
        return ["events"], value
    prefix = ["events", *_anchor(match.group(2)), f"filter_{match.group(1)}"]
    return prefix, value[:match.start()].strip()


def _query_attribute(subtype: str) -> str:
    if subtype not in {"query_color", "query_material", "query_shape"}:
        raise ValueError(f"unsupported descriptive query subtype: {subtype}")
    return subtype


def _order(text: str, *, default_unique: bool = False) -> list[str]:
    value = _clean(text)
    for order in ("first", "second", "last"):
        if re.search(rf"\b{order}\b", value):
            return [order, "filter_order"]
    return ["unique"] if default_unique else []


def _compile_attribute_query(question: str, subtype: str) -> list[str]:
    value = _clean(question); query = _query_attribute(subtype)
    collision = re.search(r"\b(?:object )?(?:to|that|which)? ?collide with (?:the )?(.+)$", value)
    if collision:
        reference = collision.group(1)
        selector = _object_program(reference)
        return [
            "events", *selector, "unique", "filter_collision",
            *_order(value, default_unique=True), *selector, "unique",
            "query_collision_partner", query,
        ]
    event_match = re.search(r"\bobject (?:that )?(enters|exits)(?: the)? scene$", value)
    if event_match:
        return [
            "events", "objects", _event_name(event_match.group(1)),
            *_order(value, default_unique=True), "query_object", query,
        ]
    state = "filter_moving" if re.search(r"\bmoving\b", value) else (
        "filter_stationary" if re.search(r"\bstationary\b", value) else None
    )
    if state is None:
        raise ValueError(f"unsupported descriptive attribute query: {question!r}")
    main = value.split(" when ", 1)[0]
    return [*_object_program(main), *_time_program(value), state, "unique", query]


def _compile_set_query(question: str, subtype: str) -> list[str]:
    value = _clean(question)
    terminal = "count" if subtype == "count" else "exist"
    if subtype not in {"count", "exist"}:
        raise ValueError(f"unsupported descriptive set subtype: {subtype}")
    prefix, main = _temporal_event_prefix(value)
    if re.search(r"\bcollisions?\b", main):
        return [*prefix, "objects", "filter_collision", terminal]
    state = "filter_moving" if re.search(r"\bmoving\b", main) else (
        "filter_stationary" if re.search(r"\bstationary\b", main) else None
    )
    if state is not None:
        subject = main.split(" when ", 1)[0]
        return [*_object_program(subject), *_time_program(value), state, terminal]
    event_match = re.search(r"\b(enter|enters|exit|exits)(?: the)? scene\b", main)
    if event_match:
        subject = main[:event_match.start()]
        return [*prefix, *_object_program(subject), _event_name(event_match.group(1)), terminal]
    raise ValueError(f"unsupported descriptive set query: {question!r}")


def compile_descriptive_question(question: str, subtype: str) -> list[str]:
    """Compile public text only to the frozen NS-DR executor vocabulary."""

    kind = str(subtype).casefold().strip()
    if kind.startswith("query_"):
        return _compile_attribute_query(question, kind)
    return _compile_set_query(question, kind)


__all__ = ["compile_descriptive_question"]
