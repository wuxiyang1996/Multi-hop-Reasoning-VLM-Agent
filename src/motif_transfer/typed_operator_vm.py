"""Target-independent, fail-closed typed operator virtual machine.

The VM defines universal data operations only.  It contains no AGQA question
templates, object names, source-game identities, or target answer vocabulary.
An external source-only inducer must issue capability tokens before an operator
can execute; a target-native compiler only selects and composes those tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


class ValueKind(str, Enum):
    BOOLEAN = "BOOLEAN"
    NUMBER = "NUMBER"
    SYMBOL = "SYMBOL"
    ENTITY = "ENTITY"
    ACTION = "ACTION"
    EVENT = "EVENT"
    INTERVAL = "INTERVAL"
    ENTITY_SET = "ENTITY_SET"
    ACTION_SET = "ACTION_SET"
    EVENT_SET = "EVENT_SET"


SET_TO_ITEM = {
    ValueKind.ENTITY_SET: ValueKind.ENTITY,
    ValueKind.ACTION_SET: ValueKind.ACTION,
    ValueKind.EVENT_SET: ValueKind.EVENT,
}
ITEM_TO_SET = {value: key for key, value in SET_TO_ITEM.items()}


class VMAbstention(RuntimeError):
    """A typed precondition failed; no value may be committed."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class TypedValue:
    kind: ValueKind
    value: Any

    def __post_init__(self) -> None:
        _validate_value(self.kind, self.value)


@dataclass(frozen=True)
class ExecutionReceipt:
    status: str
    output: TypedValue | None
    abstention_reason: str | None
    executed_operator_ids: tuple[str, ...]


def _records(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(
        isinstance(row, Mapping) for row in value
    )


def _record(value: Any) -> bool:
    return isinstance(value, Mapping)


def _validate_value(kind: ValueKind, value: Any) -> None:
    if kind is ValueKind.BOOLEAN and not isinstance(value, bool):
        raise TypeError("BOOLEAN requires bool")
    if kind is ValueKind.NUMBER and (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise TypeError("NUMBER requires a finite scalar")
    if kind is ValueKind.SYMBOL and not isinstance(value, str):
        raise TypeError("SYMBOL requires str")
    if kind in {ValueKind.ENTITY, ValueKind.ACTION, ValueKind.EVENT} and not _record(value):
        raise TypeError(f"{kind.value} requires a record")
    if kind is ValueKind.INTERVAL and (
        not isinstance(value, Sequence) or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value)
        or not all(math.isfinite(float(x)) for x in value)
        or float(value[0]) > float(value[1])
    ):
        raise TypeError("INTERVAL requires finite ordered (start, end)")
    if kind in SET_TO_ITEM and not _records(value):
        raise TypeError(f"{kind.value} requires a sequence of records")


def _field(record: Mapping[str, Any], name: str) -> Any:
    if name not in record:
        raise VMAbstention(f"RECORD_FIELD_MISSING:{name}")
    return record[name]


def _interval(record: Mapping[str, Any]) -> tuple[float, float]:
    start = _field(record, "start")
    end = _field(record, "end")
    try:
        value = TypedValue(ValueKind.INTERVAL, (start, end))
    except TypeError as exc:
        raise VMAbstention(f"EVENT_INTERVAL_INVALID:{exc}") from exc
    return tuple(float(x) for x in value.value)


def _same_set(kind: ValueKind, rows: Sequence[Mapping[str, Any]]) -> TypedValue:
    return TypedValue(kind, tuple(dict(row) for row in rows))


def _require(value: TypedValue, *kinds: ValueKind) -> None:
    if value.kind not in kinds:
        expected = ",".join(kind.value for kind in kinds)
        raise VMAbstention(f"TYPE_MISMATCH:EXPECTED_{expected}:GOT_{value.kind.value}")


def _operator_output_kind(
    operation: str, args: Sequence[ValueKind], params: Mapping[str, Any],
) -> ValueKind:
    op = operation.upper()
    if op in {"EXISTS", "COMPARE", "AND", "XOR", "NOT"}:
        return ValueKind.BOOLEAN
    if op == "CARDINALITY":
        return ValueKind.NUMBER
    if op in {"FILTER_EQ", "TEMPORAL_SELECT", "UNION", "INTERSECTION"}:
        return args[0]
    if op in {"UNIQUE", "FIRST", "LAST", "ARGMAX", "ARGMIN"}:
        if args[0] not in SET_TO_ITEM:
            raise VMAbstention(f"TYPE_MISMATCH:{op}_REQUIRES_TYPED_SET")
        return SET_TO_ITEM[args[0]]
    if op == "CHOOSE":
        if len(args) != 3 or args[1] != args[2]:
            raise VMAbstention("TYPE_MISMATCH:CHOOSE_BRANCHES_DIFFER")
        return args[1]
    if op == "PROJECT":
        try:
            return ValueKind(str(params["output_kind"]))
        except (KeyError, ValueError) as exc:
            raise VMAbstention("PROJECT_OUTPUT_KIND_INVALID") from exc
    if op == "INTERVAL_OF":
        return ValueKind.INTERVAL
    raise VMAbstention(f"UNKNOWN_OPERATOR:{op}")


def execute_operator(
    operation: str, args: Sequence[TypedValue], params: Mapping[str, Any] | None = None,
) -> TypedValue:
    params = dict(params or {})
    op = operation.upper()
    output_kind = _operator_output_kind(op, [row.kind for row in args], params)
    if op == "EXISTS":
        _require(args[0], *SET_TO_ITEM)
        return TypedValue(output_kind, bool(args[0].value))
    if op == "CARDINALITY":
        _require(args[0], *SET_TO_ITEM)
        return TypedValue(output_kind, len(args[0].value))
    if op == "FILTER_EQ":
        _require(args[0], *SET_TO_ITEM)
        if len(args) != 2:
            raise VMAbstention("ARITY_MISMATCH:FILTER_EQ")
        field = str(params.get("field") or "")
        if not field:
            raise VMAbstention("FILTER_FIELD_MISSING")
        wanted = args[1].value
        return _same_set(
            output_kind,
            [row for row in args[0].value if _field(row, field) == wanted],
        )
    if op == "UNIQUE":
        _require(args[0], *SET_TO_ITEM)
        if len(args[0].value) != 1:
            raise VMAbstention(f"UNIQUE_CARDINALITY:{len(args[0].value)}")
        return TypedValue(output_kind, dict(args[0].value[0]))
    if op in {"FIRST", "LAST"}:
        _require(args[0], ValueKind.EVENT_SET)
        if not args[0].value:
            raise VMAbstention(f"{op}_EMPTY_EVENT_SET")
        rows = list(args[0].value)
        endpoint = 0 if op == "FIRST" else 1
        values = [_interval(row)[endpoint] for row in rows]
        target = min(values) if op == "FIRST" else max(values)
        winners = [row for row, value in zip(rows, values) if value == target]
        if len(winners) != 1:
            raise VMAbstention(f"{op}_NOT_UNIQUE")
        return TypedValue(output_kind, dict(winners[0]))
    if op in {"ARGMAX", "ARGMIN"}:
        _require(args[0], *SET_TO_ITEM)
        if not args[0].value:
            raise VMAbstention(f"{op}_EMPTY_SET")
        field = str(params.get("field") or "")
        values = []
        for row in args[0].value:
            value = _field(row, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise VMAbstention(f"{op}_FIELD_NOT_NUMERIC:{field}")
            values.append(float(value))
        target = max(values) if op == "ARGMAX" else min(values)
        winners = [row for row, value in zip(args[0].value, values) if value == target]
        if len(winners) != 1:
            raise VMAbstention(f"{op}_NOT_UNIQUE")
        return TypedValue(output_kind, dict(winners[0]))
    if op == "COMPARE":
        if len(args) != 2:
            raise VMAbstention("ARITY_MISMATCH:COMPARE")
        _require(args[0], ValueKind.NUMBER)
        _require(args[1], ValueKind.NUMBER)
        relation = str(params.get("relation") or "").upper()
        left, right = float(args[0].value), float(args[1].value)
        relations = {"LT": left < right, "LE": left <= right, "EQ": left == right,
                     "GE": left >= right, "GT": left > right}
        if relation not in relations:
            raise VMAbstention(f"COMPARE_RELATION_INVALID:{relation}")
        return TypedValue(output_kind, relations[relation])
    if op in {"AND", "XOR"}:
        if len(args) != 2:
            raise VMAbstention(f"ARITY_MISMATCH:{op}")
        _require(args[0], ValueKind.BOOLEAN); _require(args[1], ValueKind.BOOLEAN)
        answer = args[0].value and args[1].value if op == "AND" else args[0].value != args[1].value
        return TypedValue(output_kind, answer)
    if op == "NOT":
        if len(args) != 1:
            raise VMAbstention("ARITY_MISMATCH:NOT")
        _require(args[0], ValueKind.BOOLEAN)
        return TypedValue(output_kind, not args[0].value)
    if op == "CHOOSE":
        _require(args[0], ValueKind.BOOLEAN)
        return args[1] if args[0].value else args[2]
    if op == "PROJECT":
        if len(args) != 1 or args[0].kind not in ITEM_TO_SET:
            raise VMAbstention("TYPE_MISMATCH:PROJECT_REQUIRES_RECORD")
        field = str(params.get("field") or "")
        value = _field(args[0].value, field)
        try:
            return TypedValue(output_kind, value)
        except TypeError as exc:
            raise VMAbstention(f"PROJECT_VALUE_INVALID:{exc}") from exc
    if op == "INTERVAL_OF":
        _require(args[0], ValueKind.EVENT)
        return TypedValue(output_kind, _interval(args[0].value))
    if op == "TEMPORAL_SELECT":
        if len(args) != 2:
            raise VMAbstention("ARITY_MISMATCH:TEMPORAL_SELECT")
        _require(args[0], ValueKind.EVENT_SET); _require(args[1], ValueKind.INTERVAL)
        relation = str(params.get("relation") or "").upper()
        left, right = (float(x) for x in args[1].value)
        selected = []
        for row in args[0].value:
            start, end = _interval(row)
            include = {
                "BEFORE": end < left,
                "AFTER": start > right,
                "WHILE": start <= right and end >= left,
                "BETWEEN": start > left and end < right,
            }.get(relation)
            if include is None:
                raise VMAbstention(f"TEMPORAL_RELATION_INVALID:{relation}")
            if include:
                selected.append(row)
        return _same_set(output_kind, selected)
    if op in {"UNION", "INTERSECTION"}:
        if len(args) != 2 or args[0].kind != args[1].kind or args[0].kind not in SET_TO_ITEM:
            raise VMAbstention(f"TYPE_MISMATCH:{op}_REQUIRES_SAME_TYPED_SETS")
        left = {repr(sorted(row.items())): row for row in args[0].value}
        right = {repr(sorted(row.items())): row for row in args[1].value}
        keys = left.keys() | right.keys() if op == "UNION" else left.keys() & right.keys()
        return _same_set(output_kind, [dict((left | right)[key]) for key in sorted(keys)])
    raise VMAbstention(f"UNKNOWN_OPERATOR:{op}")


def execute_program(
    program: Mapping[str, Any], inputs: Mapping[str, TypedValue],
    *, authorized_operators: Sequence[str],
) -> ExecutionReceipt:
    """Execute a straight-line typed program under source capability control."""

    values = dict(inputs)
    allowed = {str(value).upper() for value in authorized_operators}
    executed: list[str] = []
    try:
        for node in program.get("nodes") or ():
            node_id = str(node.get("id") or "")
            operation = str(node.get("op") or "").upper()
            if not node_id or node_id in values:
                raise VMAbstention("PROGRAM_NODE_ID_INVALID")
            if operation not in allowed:
                raise VMAbstention(f"SOURCE_OPERATOR_NOT_AUTHORIZED:{operation}")
            refs = [str(value) for value in node.get("args") or ()]
            if any(ref not in values for ref in refs):
                raise VMAbstention(f"PROGRAM_ARGUMENT_MISSING:{node_id}")
            values[node_id] = execute_operator(
                operation, [values[ref] for ref in refs], node.get("params") or {},
            )
            executed.append(operation)
        output_id = str(program.get("output") or "")
        if output_id not in values:
            raise VMAbstention("PROGRAM_OUTPUT_MISSING")
        return ExecutionReceipt("COMMITTED", values[output_id], None, tuple(executed))
    except (TypeError, VMAbstention) as exc:
        reason = exc.reason if isinstance(exc, VMAbstention) else f"VALUE_TYPE_INVALID:{exc}"
        return ExecutionReceipt("ABSTAINED", None, reason, tuple(executed))
