"""Runtime-safe typed proof receipts for paired CLEVRER symbolic executors."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


PROOF_FEATURE_NAMES = (
    "proof_step_divergence_fraction",
    "proof_first_divergence_fraction_mean",
    "proof_result_kind_mismatch_fraction",
    "proof_result_cardinality_abs_delta_mean",
    "proof_error_asymmetry_fraction",
    "proof_collision_step_divergence_fraction",
    "proof_counterfactual_step_divergence_fraction",
    "proof_temporal_step_divergence_fraction",
    "proof_membership_step_divergence_fraction",
    "proof_object_step_divergence_fraction",
    "explicit_existing_collision_count_fraction",
    "trajectory_existing_collision_count_fraction",
    "existing_event_jaccard",
    "explicit_unseen_collision_count_fraction",
    "trajectory_unseen_collision_count_fraction",
    "unseen_event_jaccard",
    "explicit_counterfactual_collision_count_fraction",
    "trajectory_counterfactual_collision_count_fraction",
    "counterfactual_event_jaccard",
    "explicit_in_out_count_fraction",
    "trajectory_in_out_count_fraction",
)


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if hasattr(value, "item"):
        return _canonical(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _kind_and_cardinality(value: Any) -> tuple[str, int]:
    if value == "error":
        return "error", -1
    if isinstance(value, list):
        if not value:
            return "empty_list", 0
        if isinstance(value[0], Mapping):
            return "event_list", len(value)
        if isinstance(value[0], int):
            return "object_list", len(value)
        return "list", len(value)
    if isinstance(value, Mapping):
        return "event", 1
    if isinstance(value, int):
        return "object_or_scalar", 1
    if value in {"yes", "no"}:
        return "boolean", 1
    return type(value).__name__, 1


def execute_with_receipt(executor: Any, program: Sequence[str]) -> dict[str, Any]:
    """Execute an official postfix program while recording typed module effects."""

    stack: list[Any] = []
    steps: list[dict[str, Any]] = []
    for token in program:
        if token in {"<END>", "<NULL>"}:
            break
        if token == "<START>":
            continue
        if token not in executor.modules:
            stack.append(token)
            continue
        spec = executor.modules[token]
        nargs = int(spec["nargs"])
        if len(stack) < nargs:
            steps.append({
                "module": token,
                "result_kind": "error",
                "result_cardinality": -1,
                "result_fingerprint": _fingerprint("error"),
            })
            return {"answer": "error", "steps": steps}
        args = [stack.pop() for _ in range(nargs)][::-1]
        result = spec["func"](*args)
        kind, cardinality = _kind_and_cardinality(result)
        steps.append({
            "module": token,
            "argument_kinds": [_kind_and_cardinality(value)[0] for value in args],
            "result_kind": kind,
            "result_cardinality": cardinality,
            "result_fingerprint": _fingerprint(result),
        })
        if result == "error":
            return {"answer": "error", "steps": steps}
        stack.append(result)
    return {"answer": str(stack[-1]) if stack else "error", "steps": steps}


def _events(executor: Any, role: str) -> set[tuple[Any, ...]]:
    if role == "existing":
        values = executor.existing_events
    elif role == "unseen":
        values = executor.unseens
    elif role == "counterfactual":
        values = [
            event
            for what_if, events in sorted(executor.sim.cf_events.items())
            for event in ({**event, "what_if": what_if} for event in events)
        ]
    else:
        raise ValueError(f"unknown event role: {role}")
    output = set()
    for event in values:
        if event.get("type") in {"start", "end"}:
            continue
        raw_objects = event.get("object", ())
        objects = []
        for value in raw_objects:
            if isinstance(value, (list, tuple)):
                objects.extend(value)
            else:
                objects.append(value)
        output.add((
            event.get("what_if", -1),
            str(event.get("type")),
            tuple(sorted(map(int, objects))),
            int(event.get("frame", -1)),
        ))
    return output


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def paired_proof_features(
    explicit_executor: Any,
    trajectory_executor: Any,
    question_program: Sequence[str],
    choice_programs: Sequence[Sequence[str]],
) -> tuple[tuple[float, ...], list[dict[str, Any]]]:
    """Compare proof traces and event graphs without reading official outcomes."""

    receipts = []
    all_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    first_divergences = []
    error_asymmetries = 0
    for choice in choice_programs:
        program = list(choice) + list(question_program)
        explicit = execute_with_receipt(explicit_executor, program)
        trajectory = execute_with_receipt(trajectory_executor, program)
        pairs = list(zip(explicit["steps"], trajectory["steps"]))
        all_pairs.extend(pairs)
        divergent = [
            index for index, (left, right) in enumerate(pairs)
            if left["result_fingerprint"] != right["result_fingerprint"]
        ]
        denominator = max(len(explicit["steps"]), len(trajectory["steps"]), 1)
        first_divergences.append(
            divergent[0] / denominator if divergent else 1.0
        )
        error_asymmetries += int((explicit["answer"] == "error") != (trajectory["answer"] == "error"))
        receipts.append({
            "explicit_answer": explicit["answer"],
            "trajectory_answer": trajectory["answer"],
            "explicit_steps": explicit["steps"],
            "trajectory_steps": trajectory["steps"],
        })

    total_steps = max(len(all_pairs), 1)
    divergent_pairs = [
        (left, right) for left, right in all_pairs
        if left["result_fingerprint"] != right["result_fingerprint"]
    ]
    kind_mismatches = sum(
        left["result_kind"] != right["result_kind"] for left, right in all_pairs
    )
    cardinality_delta = sum(
        abs(int(left["result_cardinality"]) - int(right["result_cardinality"]))
        for left, right in all_pairs
    )
    groups = {
        "collision": {"filter_collision", "query_collision_partner"},
        "counterfactual": {"counterfact_events", "filter_counterfact"},
        "temporal": {"filter_before", "filter_after", "filter_order", "filter_ancestor"},
        "membership": {"belong_to", "exist", "negate", "unique"},
        "object": {"objects", "filter_color", "filter_material", "filter_shape"},
    }
    group_values = {}
    for name, modules in groups.items():
        relevant = [(left, right) for left, right in all_pairs if left["module"] in modules]
        group_values[name] = (
            sum(
                left["result_fingerprint"] != right["result_fingerprint"]
                for left, right in relevant
            ) / len(relevant)
            if relevant else 0.0
        )

    event_sets = {
        role: (_events(explicit_executor, role), _events(trajectory_executor, role))
        for role in ("existing", "unseen", "counterfactual")
    }
    explicit_existing, trajectory_existing = event_sets["existing"]
    explicit_unseen, trajectory_unseen = event_sets["unseen"]
    explicit_cf, trajectory_cf = event_sets["counterfactual"]
    features = (
        len(divergent_pairs) / total_steps,
        sum(first_divergences) / max(len(first_divergences), 1),
        kind_mismatches / total_steps,
        cardinality_delta / total_steps,
        error_asymmetries / max(len(choice_programs), 1),
        group_values["collision"],
        group_values["counterfactual"],
        group_values["temporal"],
        group_values["membership"],
        group_values["object"],
        sum(event[1] == "collision" for event in explicit_existing) / 10.0,
        sum(event[1] == "collision" for event in trajectory_existing) / 10.0,
        _jaccard(explicit_existing, trajectory_existing),
        sum(event[1] == "collision" for event in explicit_unseen) / 10.0,
        sum(event[1] == "collision" for event in trajectory_unseen) / 10.0,
        _jaccard(explicit_unseen, trajectory_unseen),
        sum(event[1] == "collision" for event in explicit_cf) / 30.0,
        sum(event[1] == "collision" for event in trajectory_cf) / 30.0,
        _jaccard(explicit_cf, trajectory_cf),
        sum(event[1] in {"in", "out"} for event in explicit_existing) / 10.0,
        sum(event[1] in {"in", "out"} for event in trajectory_existing) / 10.0,
    )
    if len(features) != len(PROOF_FEATURE_NAMES):
        raise AssertionError("CLEVRER proof feature contract drift")
    return tuple(map(float, features)), receipts


__all__ = ["PROOF_FEATURE_NAMES", "execute_with_receipt", "paired_proof_features"]
