"""Fail-closed semantic independence checks for WebShop goal reserves.

WebShop task IDs are transport identifiers, not experimental units.  A server
may map many IDs to the same small goal list (for example via modulo indexing),
so reserve construction must operate on target-native goal semantics.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping, Sequence
import unicodedata


class ReserveIndependenceError(ValueError):
    """Raised when a purported fresh reserve is not semantically independent."""


def canonical_instruction(text: str) -> str:
    """Return a stable semantic key without changing content words."""

    normalized = unicodedata.normalize("NFKC", str(text)).strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _instruction(row: Mapping[str, Any]) -> str:
    value = row.get("instruction_text", row.get("goal", ""))
    if not str(value).strip():
        raise ReserveIndependenceError("goal row has no instruction text")
    return canonical_instruction(str(value))


def audit_semantic_reserve(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    consumed_rows: Sequence[Mapping[str, Any]] = (),
    required_unique_goals: int | None = None,
    require_asin_disjointness: bool = False,
    require_unique_candidate_asins: bool = False,
) -> dict[str, Any]:
    """Audit a reserve using goal text (and optionally product identity).

    The result is descriptive and deterministic.  Call
    :func:`require_semantic_reserve` to make an experiment fail closed.
    """

    candidate_keys = [_instruction(row) for row in candidate_rows]
    consumed_keys = {_instruction(row) for row in consumed_rows}
    counts = Counter(candidate_keys)
    duplicate_keys = sorted(key for key, count in counts.items() if count > 1)
    overlap_keys = sorted(set(candidate_keys) & consumed_keys)

    candidate_asins = {
        str(row.get("asin")) for row in candidate_rows if row.get("asin")
    }
    consumed_asins = {
        str(row.get("asin")) for row in consumed_rows if row.get("asin")
    }
    asin_overlap = sorted(candidate_asins & consumed_asins)
    required = (
        len(candidate_rows) if required_unique_goals is None
        else int(required_unique_goals)
    )
    gates = {
        "enough_unique_goal_semantics": len(counts) >= required,
        "one_task_per_goal_semantics": len(counts) == len(candidate_rows),
        "instruction_disjoint_from_consumed": not overlap_keys,
        "asin_disjoint_from_consumed": (
            not asin_overlap if require_asin_disjointness else True
        ),
        "one_task_per_asin": (
            len(candidate_asins) == len(candidate_rows)
            if require_unique_candidate_asins else True
        ),
    }
    return {
        "tasks": len(candidate_rows),
        "unique_goal_semantics": len(counts),
        "required_unique_goal_semantics": required,
        "duplicate_goal_semantics": duplicate_keys,
        "duplicate_multiplicities": {
            key: counts[key] for key in duplicate_keys
        },
        "consumed_instruction_overlap": overlap_keys,
        "candidate_unique_asins": len(candidate_asins),
        "consumed_asin_overlap": asin_overlap,
        "gates": gates,
        "passed": all(gates.values()),
    }


def require_semantic_reserve(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    consumed_rows: Sequence[Mapping[str, Any]] = (),
    required_unique_goals: int | None = None,
    require_asin_disjointness: bool = False,
    require_unique_candidate_asins: bool = False,
) -> dict[str, Any]:
    """Return the audit or reject the run before any model call."""

    audit = audit_semantic_reserve(
        candidate_rows,
        consumed_rows=consumed_rows,
        required_unique_goals=required_unique_goals,
        require_asin_disjointness=require_asin_disjointness,
        require_unique_candidate_asins=require_unique_candidate_asins,
    )
    if not audit["passed"]:
        failed = [name for name, passed in audit["gates"].items() if not passed]
        raise ReserveIndependenceError(
            "WebShop semantic reserve preflight failed: " + ", ".join(failed)
        )
    return audit


__all__ = [
    "ReserveIndependenceError",
    "audit_semantic_reserve",
    "canonical_instruction",
    "require_semantic_reserve",
]
