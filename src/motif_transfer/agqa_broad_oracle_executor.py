"""Answer-blind broad AGQA executor over the target-native oracle backend.

This module binds the existing public-question grammar to official STSG tools.
It deliberately supports only operations whose operands appear in the public
question; it never consumes an AGQA functional program or answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Mapping, Sequence

from .agqa_active_frame_grounder import AGQAQueryPlan
from .agqa_oracle_query_mdp import AGQAOracleQueryBackend, AGQAOracleQueryReceipt
from .contracts import stable_hash


@dataclass(frozen=True)
class AGQABroadOracleExecution:
    prediction: str | None
    status: str
    reason: str
    receipts: tuple[AGQAOracleQueryReceipt, ...]
    execution_sha256: str

    def validate(self) -> None:
        for receipt in self.receipts:
            receipt.validate()
        core = asdict(self); claimed = core.pop("execution_sha256")
        if stable_hash(core) != claimed:
            raise ValueError("AGQA broad oracle execution hash mismatch")


def _finish(
    prediction: str | None, status: str, reason: str,
    receipts: Sequence[AGQAOracleQueryReceipt],
) -> AGQABroadOracleExecution:
    core = {
        "prediction": prediction, "status": status, "reason": reason,
        "receipts": tuple(receipts),
    }
    hashable = dict(core)
    hashable["receipts"] = tuple(asdict(value) for value in receipts)
    result = AGQABroadOracleExecution(
        **core, execution_sha256=stable_hash(hashable),
    )
    result.validate()
    return result


def _one_interval(result: Mapping[str, object]) -> tuple[int, int] | None:
    intervals = result.get("intervals")
    if not isinstance(intervals, list) or len(intervals) != 1:
        return None
    row = intervals[0]
    if not isinstance(row, Mapping):
        return None
    return int(row["start_frame"]), int(row["end_frame"])


def _relation_from_visual_query(value: str) -> str:
    text = re.sub(r"\s+", " ", value.casefold()).strip()
    text = re.sub(r"^(?:a|the) person\s+", "", text)
    text = re.sub(r"\s+(?:an? )?unknown object$", "", text)
    return text.strip()


def execute_broad_public_plan(
    plan: AGQAQueryPlan,
    backend: AGQAOracleQueryBackend,
) -> AGQABroadOracleExecution:
    """Execute one existing public plan, failing closed on ambiguity."""

    comparison = plan.comparison
    if comparison == "EXISTS":
        event = backend.locate_action(plan.operand_a)
        intervals = event.get("intervals")
        if not isinstance(intervals, list):
            return _finish(None, "ABSTAINED", "MALFORMED_EVENT_RESULT", backend.receipts)
        return _finish("yes" if intervals else "no", "COMMITTED",
                       "CLOSED_WORLD_EVENT_EXISTENCE", backend.receipts)

    if comparison in {"QUERY_OBJECT", "CHOOSE_OBJECT"}:
        relation = _relation_from_visual_query(plan.visual_query_a)
        if not relation:
            return _finish(None, "ABSTAINED", "EMPTY_RELATION", backend.receipts)
        result = backend.query_relation(relation, frames=backend.all_frame_numbers())
        objects = tuple(str(value) for value in result.get("objects", ()))
        if comparison == "QUERY_OBJECT":
            if len(objects) != 1:
                return _finish(None, "ABSTAINED", "OBJECT_NOT_UNIQUE", backend.receipts)
            return _finish(objects[0], "COMMITTED", "UNIQUE_OBJECT", backend.receipts)
        candidates = {
            re.sub(r"^(?:a|an|the)\s+", "", value.casefold()).strip(): value
            for value in (plan.operand_a, plan.operand_b)
        }
        matched = [candidates[value] for value in objects if value in candidates]
        if len(matched) != 1:
            return _finish(None, "ABSTAINED", "CHOICE_NOT_UNIQUE", backend.receipts)
        return _finish(matched[0], "COMMITTED", "UNIQUE_CHOICE", backend.receipts)

    first = _one_interval(backend.locate_action(plan.operand_a))
    second = _one_interval(backend.locate_action(plan.operand_b))
    if first is None or second is None:
        return _finish(None, "ABSTAINED", "TEMPORAL_OPERAND_NOT_UNIQUE", backend.receipts)
    if comparison == "BEFORE_AFTER":
        if first[1] < second[0]:
            prediction = "before"
        elif second[1] < first[0]:
            prediction = "after"
        else:
            return _finish(None, "ABSTAINED", "TEMPORAL_INTERVALS_OVERLAP", backend.receipts)
        return _finish(prediction, "COMMITTED", "DISJOINT_INTERVAL_ORDER", backend.receipts)

    duration_a = first[1] - first[0] + 1
    duration_b = second[1] - second[0] + 1
    if duration_a == duration_b:
        return _finish(None, "ABSTAINED", "EQUAL_DURATION", backend.receipts)
    a_longer = duration_a > duration_b
    if comparison == "SELECT_LONGER":
        prediction = plan.operand_a if a_longer else plan.operand_b
    elif comparison == "SELECT_SHORTER":
        prediction = plan.operand_b if a_longer else plan.operand_a
    elif comparison == "VERIFY_A_LONGER":
        prediction = "yes" if a_longer else "no"
    elif comparison == "VERIFY_A_SHORTER":
        prediction = "no" if a_longer else "yes"
    else:
        return _finish(None, "ABSTAINED", "UNSUPPORTED_COMPARISON", backend.receipts)
    return _finish(prediction, "COMMITTED", "DURATION_COMPARISON", backend.receipts)


__all__ = ["AGQABroadOracleExecution", "execute_broad_public_plan"]
