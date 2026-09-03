"""Capability-gated execution of predicted AGQA programs over official STSGs.

The STSG adapter supplies target-native facts.  The executor never accepts an
answer or question-specific ``sg_grounding``.  Success-critical control
operations are admitted by the source capability artifact and use the shared
typed VM where their value types align directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Callable, Mapping, Sequence

from .agqa_typed_program import Array, Atom, Call, Expression, compile_receipt, parse_program
from .contracts import stable_hash
from .typed_operator_vm import TypedValue, ValueKind, execute_operator


class STSGExecutionAbstention(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameWindow:
    start: int
    end: int

    def contains(self, frame: int) -> bool:
        return self.start <= frame <= self.end


@dataclass(frozen=True)
class STSGExecutionReceipt:
    status: str
    prediction: str | None
    reason: str
    executed_operators: tuple[str, ...]
    graph_sha256: str
    program_sha256: str
    answer_read: bool
    functional_program_source: str
    question_grounding_read: bool
    receipt_sha256: str


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).replace("_", " ").strip().casefold())


def _refs(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        value = value.get("vertices", ())
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(x.get("id")) if isinstance(x, Mapping) else str(x) for x in value)


class AGQATypedSTSGExecutor:
    def __init__(
        self, *, graph: Mapping[str, Any], id_to_text: Mapping[str, str],
        graph_sha256: str, authorized_operators: Sequence[str],
        authorized_compositions: Sequence[Sequence[str]] | None = None,
        ambiguity_policy: str = "LEGACY",
    ):
        self.graph = graph
        self.id_to_text = {str(k): _clean(v) for k, v in id_to_text.items()}
        self.text_to_id = {value: key for key, value in self.id_to_text.items()}
        self.graph_sha256 = str(graph_sha256)
        self.allowed = {str(value).upper() for value in authorized_operators}
        self.allowed_compositions = authorized_compositions
        if ambiguity_policy not in {"LEGACY", "STRICT", "EAGER"}:
            raise ValueError("invalid executor ambiguity policy")
        self.ambiguity_policy = ambiguity_policy
        self.executed: list[str] = []
        self.frames = sorted(
            int(key) for key, row in graph.items()
            if str(key).isdigit() and isinstance(row, Mapping) and row.get("type") == "frame"
        )
        self.actions = [row for row in graph.values()
                        if isinstance(row, Mapping) and row.get("type") == "action"]
        self.objects = [(str(key), row) for key, row in graph.items()
                        if isinstance(row, Mapping) and str(key).startswith("o")]
        self.relations = [(str(key), row) for key, row in graph.items()
                          if isinstance(row, Mapping) and str(key)[:1] in {"r", "v"}]

    def _authorize(self, operation: str) -> None:
        op = operation.upper()
        if op not in self.allowed:
            raise STSGExecutionAbstention(f"SOURCE_OPERATOR_NOT_AUTHORIZED:{op}")
        self.executed.append(op)

    def _action_interval(self, phrase: str) -> FrameWindow:
        wanted = _clean(phrase)
        matches = []
        for row in self.actions:
            label = _clean(row.get("phrase") or self.id_to_text.get(str(row.get("charades", "")), ""))
            if label != wanted:
                continue
            frames = sorted(int(value) for value in row.get("all_f", ()) if str(value).isdigit())
            if frames:
                matches.append(FrameWindow(frames[0], frames[-1]))
        if len(matches) != 1 and self.ambiguity_policy != "EAGER":
            raise STSGExecutionAbstention(f"ACTION_INTERVAL_CARDINALITY:{wanted}:{len(matches)}")
        if not matches:
            raise STSGExecutionAbstention(f"ACTION_INTERVAL_CARDINALITY:{wanted}:0")
        if len(matches) > 1:
            matches.sort(key=lambda value: (value.start, value.end))
        self._authorize("INTERVAL_OF")
        return matches[0]

    def _all_window(self) -> FrameWindow:
        if not self.frames:
            raise STSGExecutionAbstention("NO_STSG_FRAMES")
        return FrameWindow(self.frames[0], self.frames[-1])

    def _frame_filter(self, frame: int, query: Sequence[Any]) -> list[str]:
        category = _clean(query[0]) if query else ""
        if category == "relation":
            category = "relations"
        if category == "objects":
            return sorted(set(
                self.id_to_text.get(str(row.get("class", "")), "")
                for key, row in self.objects if key.endswith(f"/{frame:06d}")
            ) - {""})
        if category == "relations" and len(query) == 1:
            return sorted(set(
                self.id_to_text.get(str(row.get("class", "")), "")
                for key, row in self.relations if key.endswith(f"/{frame:06d}")
            ) - {""})
        if category == "relations" and len(query) == 3 and _clean(query[2]) == "objects":
            self._authorize("FILTER_EQ")
            wanted = self.text_to_id.get(_clean(query[1]))
            result = set()
            for key, row in self.relations:
                if not key.endswith(f"/{frame:06d}") or str(row.get("class", "")) != wanted:
                    continue
                for ref in _refs(row.get("objects")):
                    label = self.id_to_text.get(ref.split("/", 1)[0], "")
                    if label and label not in {"person", "none"}:
                        result.add(label)
            return sorted(result)
        if category == "actions":
            values = []
            for row in self.actions:
                frames = {int(x) for x in row.get("all_f", ()) if str(x).isdigit()}
                if frame in frames:
                    values.append(_clean(row.get("phrase", "")))
            return sorted(set(values))
        raise STSGExecutionAbstention(f"FILTER_QUERY_UNSUPPORTED:{query}")

    def _eval(self, node: Expression, context: Mapping[str, Any]) -> Any:
        if isinstance(node, Atom):
            value = _clean(node.value)
            return context.get(value, value)
        if isinstance(node, Array):
            return tuple(self._eval(value, context) for value in node.values)
        name = node.function
        args = node.arguments
        if name == "Filter":
            mode = self._eval(args[0], context)
            query = self._eval(args[1], context)
            if mode == "frame":
                return lambda frame: self._frame_filter(int(frame), query)
            if mode == "video" and query and query[0] == "actions":
                wanted = query[1] if len(query) > 1 else None
                if wanted is not None:
                    self._authorize("FILTER_EQ")
                values = [_clean(row.get("phrase", "")) for row in self.actions]
                return sorted(set(value for value in values if value and (wanted is None or value == wanted)))
            raise STSGExecutionAbstention("FILTER_MODE_UNSUPPORTED")
        if name == "Iterate":
            items = self._eval(args[0], context)
            function = self._eval(args[1], context)
            window = self._all_window() if items == "video" else items
            if not isinstance(window, FrameWindow) or not callable(function):
                raise STSGExecutionAbstention("ITERATE_TYPE_MISMATCH")
            result = []
            for frame in self.frames:
                if window.contains(frame):
                    result.extend(function(frame))
            return sorted(set(result))
        if name == "Exists":
            item = self._eval(args[0], context)
            items = self._eval(args[1], context)
            if callable(items):
                return lambda frame: self._membership(item, items(frame))
            return self._membership(item, items)
        if name == "HasItem":
            items = self._eval(args[0], context)
            if callable(items):
                return lambda frame: self._nonempty(items(frame))
            return self._nonempty(items)
        if name in {"XOR", "AND"}:
            left, right = (self._eval(arg, context) for arg in args)
            if callable(left) or callable(right):
                return lambda value: self._boolean(
                    name, left(value) if callable(left) else left,
                    right(value) if callable(right) else right,
                )
            return self._boolean(name, left, right)
        if name == "OnlyItem":
            values = self._eval(args[0], context)
            self._authorize("UNIQUE")
            if not isinstance(values, Sequence) or isinstance(values, str) or not values:
                raise STSGExecutionAbstention("UNIQUE_CARDINALITY")
            if len(values) != 1 and self.ambiguity_policy != "EAGER":
                raise STSGExecutionAbstention("UNIQUE_CARDINALITY")
            if len(values) > 1:
                values = sorted(values, key=lambda value: str(value))
            return values[0]
        if name == "Query":
            mode = self._eval(args[0], context)
            if mode == "class":
                self._authorize("PROJECT")
                return self._eval(args[1], context)
            if mode in {"start", "end"}:
                return lambda action: getattr(self._action_interval(action), mode)
            raise STSGExecutionAbstention(f"QUERY_MODE_UNSUPPORTED:{mode}")
        if name == "Localize":
            relation = self._eval(args[0], context)
            actions = self._eval(args[1], context)
            self._authorize("TEMPORAL_SELECT")
            if relation == "temporal tag":
                relation = context.get("temporal tag")
            action_values = actions if isinstance(actions, tuple) else (actions,)
            intervals = [self._action_interval(value) for value in action_values]
            if relation == "before": return FrameWindow(self.frames[0], intervals[0].start - 1)
            if relation == "after": return FrameWindow(intervals[0].end + 1, self.frames[-1])
            if relation == "while": return intervals[0]
            if relation == "between" and len(intervals) == 2:
                left, right = sorted(intervals, key=lambda x: x.start)
                return FrameWindow(left.end + 1, right.start - 1)
            raise STSGExecutionAbstention(f"TEMPORAL_RELATION_UNSUPPORTED:{relation}")
        if name == "IterateUntil":
            direction = self._eval(args[0], context)
            items = self._eval(args[1], context)
            condition = self._eval(args[2], context)
            function = self._eval(args[3], context)
            window = self._all_window() if items == "video" else items
            op = "FIRST" if direction == "forward" else "LAST" if direction == "backward" else ""
            if not op or not isinstance(window, FrameWindow) or not callable(condition) or not callable(function):
                raise STSGExecutionAbstention("ITERATE_UNTIL_TYPE_MISMATCH")
            self._authorize(op)
            frames = self.frames if op == "FIRST" else list(reversed(self.frames))
            for frame in frames:
                if window.contains(frame) and condition(frame):
                    return function(frame)
            raise STSGExecutionAbstention("ITERATE_UNTIL_NO_MATCH")
        if name == "Choose":
            first, second, values = (self._eval(arg, context) for arg in args)
            self._authorize("CHOOSE")
            membership = (first in values, second in values)
            if self.ambiguity_policy == "STRICT" and sum(membership) != 1:
                raise STSGExecutionAbstention(f"CHOOSE_CARDINALITY:{sum(membership)}")
            return first if membership[0] else second
        if name == "Equals":
            left, right = (self._eval(arg, context) for arg in args)
            self._authorize("COMPARE")
            return left == right
        if name == "Compare":
            candidates = self._eval(args[0], context)
            self._authorize("COMPARE"); self._authorize("CHOOSE")
            winners = [value for value in candidates if self._eval(args[1], context | {"temporal tag": value})]
            if len(winners) != 1 and self.ambiguity_policy != "EAGER":
                raise STSGExecutionAbstention(f"COMPARE_WINNER_CARDINALITY:{len(winners)}")
            if not winners:
                # Eager means non-abstaining on ambiguity, not inventing an
                # answer when neither branch has grounded support.
                raise STSGExecutionAbstention("COMPARE_WINNER_CARDINALITY:0")
            return winners[0]
        if name == "Subtract":
            left, right = (self._eval(arg, context) for arg in args)
            if not callable(left) or not callable(right):
                raise STSGExecutionAbstention("SUBTRACT_TYPE_MISMATCH")
            return lambda value: abs(float(left(value)) - float(right(value)))
        if name == "Superlative":
            mode = self._eval(args[0], context)
            groups = self._eval(args[1], context)
            score = self._eval(args[2], context)
            candidates = [item for group in groups for item in (group if isinstance(group, (list, tuple)) else (group,))]
            records = []
            for item in candidates:
                raw = float(score(item))
                records.append({"label": item, "signed_score": raw if mode == "max" else -raw})
            self._authorize("ARGMAX")
            try:
                result = execute_operator("ARGMAX", [
                    TypedValue(ValueKind.ACTION_SET, tuple(records)),
                ], {"field": "signed_score"})
                return result.value["label"]
            except Exception:
                if self.ambiguity_policy != "EAGER" or not records:
                    raise
                return sorted(records, key=lambda row: (-row["signed_score"], str(row["label"])))[0]["label"]
        if name == "ToAction":
            verb, obj = (self._eval(arg, context) for arg in args)
            self._authorize("FILTER_EQ"); self._authorize("UNIQUE"); self._authorize("PROJECT")
            verb_id, object_id = self.text_to_id.get(verb), self.text_to_id.get(obj)
            matches = [_clean(row.get("phrase", "")) for row in self.actions
                       if str(row.get("verb_id", "")) == verb_id and str(row.get("object_id", "")) == object_id]
            if len(set(matches)) != 1 and self.ambiguity_policy != "EAGER":
                raise STSGExecutionAbstention(f"TO_ACTION_CARDINALITY:{len(set(matches))}")
            if not matches:
                raise STSGExecutionAbstention("TO_ACTION_CARDINALITY:0")
            return sorted(set(matches))[0]
        raise STSGExecutionAbstention(f"FUNCTION_UNSUPPORTED:{name}")

    def _membership(self, item: Any, items: Any) -> bool:
        self._authorize("EXISTS")
        if not isinstance(items, Sequence) or isinstance(items, str):
            raise STSGExecutionAbstention("EXISTS_TYPE_MISMATCH")
        return item in items

    def _nonempty(self, items: Any) -> bool:
        self._authorize("EXISTS")
        if not isinstance(items, Sequence) or isinstance(items, str):
            raise STSGExecutionAbstention("EXISTS_TYPE_MISMATCH")
        return bool(items)

    def _boolean(self, operation: str, left: Any, right: Any) -> bool:
        self._authorize(operation)
        try:
            return bool(execute_operator(operation, [
                TypedValue(ValueKind.BOOLEAN, left), TypedValue(ValueKind.BOOLEAN, right),
            ]).value)
        except TypeError as exc:
            raise STSGExecutionAbstention(f"BOOLEAN_TYPE_MISMATCH:{exc}") from exc

    def execute(self, program: str, *, functional_program_source: str = "PREDICTED") -> STSGExecutionReceipt:
        self.executed = []
        program_hash = stable_hash(program)
        try:
            admission = compile_receipt(
                program, sorted(self.allowed), self.allowed_compositions,
            )
            if admission["status"] != "COMPILED":
                missing = admission["missing_operators"] or admission["missing_compositions"]
                raise STSGExecutionAbstention(f"SOURCE_PROGRAM_NOT_ADMITTED:{missing}")
            value = self._eval(parse_program(program), {})
            prediction = "yes" if value is True else "no" if value is False else str(value)
            core = {
                "status": "COMMITTED", "prediction": prediction, "reason": "TYPED_PROGRAM_EXECUTED",
                "executed_operators": tuple(self.executed), "graph_sha256": self.graph_sha256,
                "program_sha256": program_hash, "answer_read": False,
                "functional_program_source": functional_program_source,
                "question_grounding_read": False,
            }
        except Exception as exc:
            core = {
                "status": "ABSTAINED", "prediction": None,
                "reason": f"{type(exc).__name__}:{exc}",
                "executed_operators": tuple(self.executed), "graph_sha256": self.graph_sha256,
                "program_sha256": program_hash, "answer_read": False,
                "functional_program_source": functional_program_source,
                "question_grounding_read": False,
            }
        return STSGExecutionReceipt(**core, receipt_sha256=stable_hash(core))


__all__ = ["AGQATypedSTSGExecutor", "FrameWindow", "STSGExecutionReceipt"]
