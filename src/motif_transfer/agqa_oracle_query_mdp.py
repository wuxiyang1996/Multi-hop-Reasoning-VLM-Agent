"""Answer-blind typed-query MDP over an official AGQA STSG.

The public question is compiled by ``agqa_temporal_localized_query``.  This
module executes the resulting typed plan over an official target-native scene
graph and public ontology.  It never accepts an answer, official functional
program, or program-derived ``sg_grounding``.

The graph is hidden behind deterministic tools so controller arms may be given
the same backend and maximum budget while choosing different query policies.
This preserves exploration headroom without reintroducing perception error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agqa_query_object_grounder import canonical_object_label
from .agqa_temporal_localized_query import AGQATemporalLocalizedQueryPlan
from .contracts import stable_hash


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def load_agqa_id_to_text(path: str | Path) -> dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("AGQA ENG ontology must be a JSON object")
    result = {str(key): _clean(value) for key, value in raw.items()}
    required = {"o1", "r1", "c000"}
    if not required <= result.keys():
        raise ValueError("AGQA ENG ontology is incomplete")
    return result


def _ref_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        value = value.get("vertices", ())
    if not isinstance(value, (list, tuple)):
        return ()
    ids = []
    for row in value:
        if isinstance(row, Mapping) and row.get("id") is not None:
            ids.append(str(row["id"]))
        elif isinstance(row, (str, int)):
            ids.append(str(row))
    return tuple(sorted(set(ids)))


def _frame_number(value: Any) -> int | None:
    text = str(value or "")
    return int(text) if text.isdigit() else None


@dataclass(frozen=True)
class AGQAOracleToolBudget:
    max_calls: int

    def validate(self) -> None:
        if isinstance(self.max_calls, bool) or not isinstance(self.max_calls, int):
            raise ValueError("oracle query budget must be an integer")
        if self.max_calls < 1:
            raise ValueError("oracle query budget must be positive")


@dataclass(frozen=True)
class AGQAOracleQueryReceipt:
    tool: str
    arguments: Mapping[str, Any]
    result: Mapping[str, Any]
    graph_sha256: str
    call_index: int
    answer_read: bool
    functional_program_read: bool
    program_grounding_read: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls, *, tool: str, arguments: Mapping[str, Any],
        result: Mapping[str, Any], graph_sha256: str, call_index: int,
    ) -> "AGQAOracleQueryReceipt":
        core = {
            "tool": str(tool), "arguments": dict(arguments),
            "result": dict(result), "graph_sha256": graph_sha256,
            "call_index": int(call_index), "answer_read": False,
            "functional_program_read": False, "program_grounding_read": False,
        }
        return cls(**core, receipt_sha256=stable_hash(core))

    def validate(self) -> None:
        core = asdict(self)
        claimed = core.pop("receipt_sha256")
        if stable_hash(core) != claimed:
            raise ValueError("AGQA oracle query receipt hash mismatch")
        if self.answer_read or self.functional_program_read or self.program_grounding_read:
            raise ValueError("AGQA oracle query crossed evaluator authority")


@dataclass(frozen=True)
class AGQAOracleExecution:
    prediction: str | None
    candidate_objects: tuple[str, ...]
    window_frames: tuple[int, ...]
    receipts: tuple[AGQAOracleQueryReceipt, ...]
    status: str
    reason: str
    execution_sha256: str

    def validate(self) -> None:
        for receipt in self.receipts:
            receipt.validate()
        core = asdict(self)
        claimed = core.pop("execution_sha256")
        if stable_hash(core) != claimed:
            raise ValueError("AGQA oracle execution hash mismatch")


@dataclass(frozen=True)
class AGQAComposedPrediction:
    """Fixed harness composition, independent of evaluator outcomes.

    A guarded localized candidate has priority.  The target-native generic
    candidate is used only when localization abstains, followed by the frozen
    actor prediction.  Keeping this composition here (rather than in an
    evaluation script) makes the policy auditable and unit-testable.
    """

    prediction: str | None
    route: str


def compose_localized_with_generic(
    localized_candidate: str | None,
    generic_candidate: str | None,
    actor_prediction: str | None = None,
) -> AGQAComposedPrediction:
    if localized_candidate is not None:
        return AGQAComposedPrediction(str(localized_candidate), "LOCALIZED")
    if generic_candidate is not None:
        return AGQAComposedPrediction(str(generic_candidate), "GENERIC_FALLBACK")
    if actor_prediction is not None:
        return AGQAComposedPrediction(str(actor_prediction), "ACTOR_FALLBACK")
    return AGQAComposedPrediction(None, "ABSTAINED")


class AGQAOracleQueryBackend:
    """Deterministic, budgeted tool backend over one hidden official STSG."""

    def __init__(
        self, *, video_id: str, graph: Mapping[str, Any],
        id_to_text: Mapping[str, str], graph_sha256: str,
        budget: AGQAOracleToolBudget,
    ):
        budget.validate()
        self.video_id = str(video_id)
        self.graph = graph
        self.id_to_text = dict(id_to_text)
        self.graph_sha256 = str(graph_sha256)
        self.budget = budget
        self.receipts: list[AGQAOracleQueryReceipt] = []

    def _emit(
        self, tool: str, arguments: Mapping[str, Any], result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if len(self.receipts) >= self.budget.max_calls:
            raise RuntimeError("AGQA oracle query budget exhausted")
        receipt = AGQAOracleQueryReceipt.create(
            tool=tool, arguments=arguments, result=result,
            graph_sha256=self.graph_sha256, call_index=len(self.receipts),
        )
        self.receipts.append(receipt)
        return result

    def locate_action(self, phrase: str) -> Mapping[str, Any]:
        wanted = _clean(phrase)
        intervals = []
        for key in sorted(self.graph):
            row = self.graph[key]
            if not isinstance(row, Mapping) or row.get("type") != "action":
                continue
            label = _clean(row.get("phrase") or self.id_to_text.get(
                str(row.get("charades", "")), ""
            ))
            if label != wanted:
                continue
            frames = sorted(
                value for value in (_frame_number(x) for x in row.get("all_f", ()))
                if value is not None
            )
            if frames:
                intervals.append({
                    "event_id": str(row.get("id", key)), "label": label,
                    "start_frame": frames[0], "end_frame": frames[-1],
                    "frames": frames,
                })
        return self._emit("LOCATE_ACTION", {"phrase": wanted}, {
            "intervals": intervals, "unique": len(intervals) == 1,
        })

    def query_relation(
        self, relation: str, *, frames: Sequence[int],
    ) -> Mapping[str, Any]:
        wanted = _clean(relation)
        allowed_frames = set(int(value) for value in frames)
        matching_ids = {
            key for key, value in self.id_to_text.items() if value == wanted
            and key[:1] in {"r", "v"}
        }
        observations = []
        objects = set()
        for key in sorted(self.graph):
            row = self.graph[key]
            if not isinstance(row, Mapping):
                continue
            class_id = str(row.get("class", ""))
            frame = _frame_number(row.get("frame") or row.get("frame_num"))
            if class_id not in matching_ids or frame not in allowed_frames:
                continue
            row_objects = []
            for object_vertex in _ref_ids(row.get("objects")):
                object_class = object_vertex.split("/", 1)[0]
                label = canonical_object_label(self.id_to_text.get(object_class, ""))
                if label and label not in {"person", "none"}:
                    objects.add(label)
                    row_objects.append(label)
            observations.append({
                "event_id": str(row.get("id", key)), "frame": frame,
                "objects": sorted(set(row_objects)),
            })
        return self._emit(
            "QUERY_RELATION_IN_WINDOW",
            {"relation": wanted, "frames": sorted(allowed_frames)},
            {"objects": sorted(objects), "observations": observations,
             "relation_ids": sorted(matching_ids)},
        )

    def all_frame_numbers(self) -> tuple[int, ...]:
        return tuple(sorted(
            value for key, row in self.graph.items()
            if isinstance(row, Mapping) and row.get("type") == "frame"
            for value in [_frame_number(row.get("id", key))]
            if value is not None
        ))


def _one_interval(result: Mapping[str, Any]) -> tuple[int, int] | None:
    values = result.get("intervals")
    if not isinstance(values, list) or len(values) != 1:
        return None
    row = values[0]
    return int(row["start_frame"]), int(row["end_frame"])


def execute_temporal_object_query(
    plan: AGQATemporalLocalizedQueryPlan,
    backend: AGQAOracleQueryBackend,
) -> AGQAOracleExecution:
    """Execute the public typed plan; fail closed on ambiguous anchors/binding."""

    first = backend.locate_action(plan.anchor_a)
    interval_a = _one_interval(first)
    interval_b = None
    if plan.temporal_operator == "BETWEEN":
        second = backend.locate_action(plan.anchor_b)
        interval_b = _one_interval(second)
    if interval_a is None or (plan.temporal_operator == "BETWEEN" and interval_b is None):
        return _execution(None, (), (), backend.receipts, "ABSTAINED",
                          "ANCHOR_NOT_UNIQUE")
    frames = backend.all_frame_numbers()
    if plan.temporal_operator == "BEFORE":
        window = tuple(value for value in frames if value < interval_a[0])
    elif plan.temporal_operator == "AFTER":
        window = tuple(value for value in frames if value > interval_a[1])
    elif plan.temporal_operator == "WHILE":
        window = tuple(value for value in frames
                       if interval_a[0] <= value <= interval_a[1])
    elif plan.temporal_operator == "BETWEEN":
        assert interval_b is not None
        left, right = sorted((interval_a, interval_b), key=lambda value: value[0])
        window = tuple(value for value in frames if left[1] < value < right[0])
    else:  # pragma: no cover
        raise ValueError("unsupported AGQA temporal operator")
    relation = backend.query_relation(plan.relation, frames=window)
    candidates = tuple(str(value) for value in relation["objects"])
    if len(candidates) != 1:
        return _execution(None, candidates, window, backend.receipts, "ABSTAINED",
                          "OBJECT_BINDING_NOT_UNIQUE")
    return _execution(candidates[0], candidates, window, backend.receipts,
                      "COMMITTED", "UNIQUE_ORACLE_BINDING")


def _execution(
    prediction: str | None, candidates: Sequence[str], frames: Sequence[int],
    receipts: Sequence[AGQAOracleQueryReceipt], status: str, reason: str,
) -> AGQAOracleExecution:
    core = {
        "prediction": prediction, "candidate_objects": tuple(candidates),
        "window_frames": tuple(frames), "receipts": tuple(receipts),
        "status": status, "reason": reason,
    }
    hashable = dict(core)
    hashable["receipts"] = tuple(asdict(value) for value in receipts)
    result = AGQAOracleExecution(**core, execution_sha256=stable_hash(hashable))
    result.validate()
    return result


__all__ = [
    "AGQAComposedPrediction", "AGQAOracleExecution", "AGQAOracleQueryBackend",
    "AGQAOracleQueryReceipt",
    "AGQAOracleToolBudget", "execute_temporal_object_query",
    "compose_localized_with_generic", "load_agqa_id_to_text",
]
