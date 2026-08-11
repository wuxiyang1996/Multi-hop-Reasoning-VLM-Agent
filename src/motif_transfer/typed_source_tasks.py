"""Matched-intervention collection for typed source effects.

This module deliberately keeps source-native action identifiers and coordinates in
raw receipts only.  The transferable artifact produced by :func:`build_effect_ir`
contains typed effects, abstract carrier predicates, and intervention provenance;
it cannot execute a MiniGrid action in a target environment.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .real_source_interventions import content_hash, split_seeds


class TypedEffect(str, Enum):
    POSITION = "POSITION"
    BIND = "BIND"
    MUTATE = "MUTATE"
    RELATE = "RELATE"


@dataclass(frozen=True)
class TypedSourceTask:
    task_id: str
    environment_id: str
    required_effects: tuple[TypedEffect, ...]
    max_depth: int = 24
    max_states: int = 2_500


DEFAULT_TASKS = (
    TypedSourceTask(
        task_id="doorkey_5x5",
        environment_id="MiniGrid-DoorKey-5x5-v0",
        required_effects=(TypedEffect.BIND, TypedEffect.MUTATE),
        max_depth=18,
        max_states=800,
    ),
    TypedSourceTask(
        task_id="unlock_pickup",
        environment_id="MiniGrid-UnlockPickup-v0",
        required_effects=(TypedEffect.BIND, TypedEffect.MUTATE),
        max_depth=24,
        max_states=3_000,
    ),
    TypedSourceTask(
        task_id="put_near",
        environment_id="MiniGrid-PutNear-6x6-N2-v0",
        required_effects=(TypedEffect.BIND, TypedEffect.RELATE),
        max_depth=20,
        max_states=1_000,
    ),
)


def _object_record(obj: Any, x: int, y: int) -> dict[str, Any]:
    return {
        "position": [int(x), int(y)],
        "type": str(obj.type),
        "color": str(obj.color),
        "is_open": getattr(obj, "is_open", None),
        "is_locked": getattr(obj, "is_locked", None),
    }


def _entity_signature(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record["type"]), str(record["color"])


def extract_minigrid_state(env: Any) -> dict[str, Any]:
    """Extract an auditable simulator state without pixels or language labels."""
    base = env.unwrapped
    objects: list[dict[str, Any]] = []
    for x in range(int(base.width)):
        for y in range(int(base.height)):
            obj = base.grid.get(x, y)
            if obj is None or str(obj.type) in {"wall", "floor", "goal", "lava"}:
                continue
            objects.append(_object_record(obj, x, y))
    objects.sort(key=lambda item: (item["position"], item["type"], item["color"]))

    carrying = None
    if base.carrying is not None:
        carrying = _object_record(base.carrying, -1, -1)
        carrying.pop("position")

    relations: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for index, left in enumerate(objects):
        lx, ly = left["position"]
        for right in objects[index + 1 :]:
            rx, ry = right["position"]
            if abs(lx - rx) + abs(ly - ry) != 1:
                continue
            relations.add(tuple(sorted((_entity_signature(left), _entity_signature(right)))))

    control_state = {
        "agent_position": [int(value) for value in base.agent_pos],
        "agent_direction": int(base.agent_dir),
        "carrying": carrying,
        "objects": objects,
    }
    replay_state = {
        **control_state,
        "step_count": int(base.step_count),
        "mission": str(base.mission),
    }
    return {
        **replay_state,
        "relations": [list(map(list, relation)) for relation in sorted(relations)],
        "carrier_bound": carrying is not None,
        "control_state_sha256": content_hash(control_state),
        "replay_state_sha256": content_hash(replay_state),
    }


def classify_typed_effects(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> tuple[TypedEffect, ...]:
    """Mechanically label a transition from state deltas, never action names."""
    effects: list[TypedEffect] = []
    if before["agent_position"] != after["agent_position"]:
        effects.append(TypedEffect.POSITION)
    before_carrying = before.get("carrying")
    after_carrying = after.get("carrying")
    if before_carrying is None and after_carrying is not None:
        effects.append(TypedEffect.BIND)

    before_at_position = {
        tuple(item["position"]): item for item in before.get("objects", ())
    }
    after_at_position = {
        tuple(item["position"]): item for item in after.get("objects", ())
    }
    for position, old in before_at_position.items():
        new = after_at_position.get(position)
        if new is None or _entity_signature(old) != _entity_signature(new):
            continue
        if (old.get("is_open"), old.get("is_locked")) != (
            new.get("is_open"),
            new.get("is_locked"),
        ):
            effects.append(TypedEffect.MUTATE)
            break

    # RELATE is intentionally narrower than any new adjacency.  It requires an
    # object to leave the carrier and participate in the newly added relation.
    if before_carrying is not None and after_carrying is None:
        old_relations = {
            tuple(tuple(item) for item in relation)
            for relation in before.get("relations", ())
        }
        new_relations = {
            tuple(tuple(item) for item in relation)
            for relation in after.get("relations", ())
        }
        carried_signature = _entity_signature(before_carrying)
        if any(carried_signature in relation for relation in new_relations - old_relations):
            effects.append(TypedEffect.RELATE)
    return tuple(effects)


def _make_env(environment_id: str) -> Any:
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401 - register MiniGrid environments
    except ImportError as exc:  # pragma: no cover - exercised by integration run
        raise RuntimeError(
            "typed source collection needs the 'source-spatial' optional dependencies"
        ) from exc
    return gym.make(environment_id)


def _replay(
    task: TypedSourceTask, seed: int, prefix: Sequence[int]
) -> tuple[Any, bool, bool]:
    env = _make_env(task.environment_id)
    env.reset(seed=seed)
    terminated = truncated = False
    for action in prefix:
        _, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break
    return env, bool(terminated), bool(truncated)


def collect_task_seed(
    task: TypedSourceTask,
    *,
    seed: int,
    split: str,
    groups_per_effect: int = 1,
) -> dict[str, Any]:
    """Find effectful states, then retain every native fork at those states."""
    if groups_per_effect < 1:
        raise ValueError("groups_per_effect must be positive")
    required = {effect.value for effect in task.required_effects}
    found: dict[str, int] = defaultdict(int)
    selected_rows: list[dict[str, Any]] = []
    queue: deque[tuple[int, ...]] = deque([()])
    seen: set[str] = set()
    explored = 0
    replay_mismatches = 0

    while queue and explored < task.max_states:
        prefix = queue.popleft()
        if len(prefix) > task.max_depth:
            continue
        env, terminated, truncated = _replay(task, seed, prefix)
        try:
            before = extract_minigrid_state(env)
            action_count = int(env.action_space.n)
        finally:
            env.close()
        if terminated or truncated or before["control_state_sha256"] in seen:
            continue
        seen.add(str(before["control_state_sha256"]))
        explored += 1

        fork_rows: list[dict[str, Any]] = []
        successor_prefixes: list[tuple[int, ...]] = []
        group_effects: set[str] = set()
        group_id = content_hash(
            ["typed-source-v3", task.task_id, seed, list(prefix)]
        )
        for action_ordinal in range(action_count):
            fork, fork_terminated, fork_truncated = _replay(task, seed, prefix)
            try:
                fork_before = extract_minigrid_state(fork)
                if fork_before["replay_state_sha256"] != before["replay_state_sha256"]:
                    replay_mismatches += 1
                    continue
                _, reward, action_terminated, action_truncated, _ = fork.step(
                    action_ordinal
                )
                after = extract_minigrid_state(fork)
            finally:
                fork.close()
            effects = classify_typed_effects(fork_before, after)
            effect_values = tuple(effect.value for effect in effects)
            group_effects.update(effect_values)
            fork_rows.append(
                {
                    "schema_version": "typed-source-receipt-v3",
                    "task_id": task.task_id,
                    "environment_id": task.environment_id,
                    "seed": int(seed),
                    "split": split,
                    "group_id": group_id,
                    "prefix_length": len(prefix),
                    "prefix_sha256": content_hash(list(prefix)),
                    # Raw source provenance only. build_effect_ir strips this field.
                    "source_action_ordinal": action_ordinal,
                    "before_replay_state_sha256": before["replay_state_sha256"],
                    "after_replay_state_sha256": after["replay_state_sha256"],
                    "before_carrier_bound": bool(before["carrier_bound"]),
                    "after_carrier_bound": bool(after["carrier_bound"]),
                    "typed_effects": list(effect_values),
                    "native_reward": float(reward),
                    "terminated": bool(action_terminated),
                    "truncated": bool(action_truncated),
                    "status": "VALID",
                }
            )
            if (
                len(prefix) < task.max_depth
                and not (fork_terminated or fork_truncated)
                and not (action_terminated or action_truncated)
                and after["control_state_sha256"] not in seen
            ):
                successor_prefixes.append((*prefix, action_ordinal))

        newly_needed = {
            effect
            for effect in group_effects & required
            if found[effect] < groups_per_effect
        }
        if newly_needed:
            selected_rows.extend(fork_rows)
            for effect in newly_needed:
                found[effect] += 1
        if all(found[effect] >= groups_per_effect for effect in required):
            break
        queue.extend(successor_prefixes)

    return {
        "task_id": task.task_id,
        "environment_id": task.environment_id,
        "seed": int(seed),
        "split": split,
        "required_effects": sorted(required),
        "found_groups": dict(sorted(found.items())),
        "explored_states": explored,
        "frontier_states": len(queue),
        "replay_mismatches": replay_mismatches,
        "search_exhausted": not queue,
        "rows": selected_rows,
    }


def summarize_typed_source_gate(
    collections: Sequence[Mapping[str, Any]],
    *,
    groups_per_effect: int = 1,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    passed = True
    for collection in collections:
        rows = list(collection["rows"])
        by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_group[str(row["group_id"])].append(row)
        for effect in collection["required_effects"]:
            positive_groups = 0
            matched_control_groups = 0
            for group_rows in by_group.values():
                labels = [effect in row["typed_effects"] for row in group_rows]
                if any(labels):
                    positive_groups += 1
                    if not all(labels):
                        matched_control_groups += 1
            cell_passed = (
                collection["replay_mismatches"] == 0
                and positive_groups >= groups_per_effect
                and matched_control_groups >= groups_per_effect
            )
            passed &= cell_passed
            cells.append(
                {
                    "task_id": collection["task_id"],
                    "seed": collection["seed"],
                    "split": collection["split"],
                    "effect": effect,
                    "positive_groups": positive_groups,
                    "matched_control_groups": matched_control_groups,
                    "passed": cell_passed,
                }
            )
    return {
        "schema_version": "typed-source-gate-v3",
        "status": "SOURCE_TYPED_GATE_PASSED" if passed else "SOURCE_TYPED_GATE_FAILED",
        "all_replays_exact": all(
            int(collection["replay_mismatches"]) == 0 for collection in collections
        ),
        "cells": cells,
    }


def build_effect_ir(
    collections: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Build an action/coordinate-free candidate IR from qualified receipts."""
    if gate.get("status") != "SOURCE_TYPED_GATE_PASSED":
        raise ValueError("cannot freeze typed effect IR before the source gate passes")
    rows = [row for collection in collections for row in collection["rows"]]
    effects = sorted(
        {
            effect
            for row in rows
            for effect in row["typed_effects"]
            if effect in {item.value for item in TypedEffect}
        }
    )
    edge_support: dict[tuple[str, str], set[str]] = defaultdict(set)
    for collection in collections:
        task_rows = list(collection["rows"])
        has_bind = any("BIND" in row["typed_effects"] for row in task_rows)
        if not has_bind:
            continue
        for effect in (TypedEffect.MUTATE.value, TypedEffect.RELATE.value):
            positives = [row for row in task_rows if effect in row["typed_effects"]]
            if positives and all(bool(row["before_carrier_bound"]) for row in positives):
                edge_support[(TypedEffect.BIND.value, effect)].add(
                    str(collection["task_id"])
                )
    core = {
        "schema_version": "typed-effect-ir-v3",
        "nodes": effects,
        "edges": [
            {
                "from": source,
                "to": target,
                "guard": "CARRIER_BOUND",
                "supporting_source_tasks": sorted(tasks),
            }
            for (source, target), tasks in sorted(edge_support.items())
        ],
        "source_lineage": sorted(
            {
                str(row["before_replay_state_sha256"])
                for row in rows
                if row["typed_effects"]
            }
        ),
        "prohibited_runtime_fields": [
            "source_action_ordinal",
            "environment_id",
            "agent_position",
            "object_position",
            "mission",
        ],
        "target_grounding": "TARGET_NATIVE_NEURAL_PROBES_ONLY",
        "execution_authority": "SYMBOLIC_ROUTING_ONLY",
    }
    return {**core, "ir_sha256": content_hash(core)}


def collect_suite(
    tasks: Sequence[TypedSourceTask],
    *,
    seeds: Sequence[int],
    namespace: str,
    groups_per_effect: int = 1,
) -> dict[str, Any]:
    splits = split_seeds(seeds, namespace=namespace)
    collections = [
        collect_task_seed(
            task,
            seed=int(seed),
            split=splits[int(seed)],
            groups_per_effect=groups_per_effect,
        )
        for task in tasks
        for seed in sorted({int(value) for value in seeds})
    ]
    gate = summarize_typed_source_gate(
        collections, groups_per_effect=groups_per_effect
    )
    result: dict[str, Any] = {
        "schema_version": "typed-multisource-experiment-v3",
        "namespace": namespace,
        "tasks": [
            {
                "task_id": task.task_id,
                "environment_id": task.environment_id,
                "required_effects": [effect.value for effect in task.required_effects],
                "max_depth": task.max_depth,
                "max_states": task.max_states,
            }
            for task in tasks
        ],
        "seed_splits": {str(seed): split for seed, split in sorted(splits.items())},
        "collections": collections,
        "gate": gate,
    }
    if gate["status"] == "SOURCE_TYPED_GATE_PASSED":
        result["effect_ir"] = build_effect_ir(collections, gate)
    return result


__all__ = [
    "DEFAULT_TASKS",
    "TypedEffect",
    "TypedSourceTask",
    "build_effect_ir",
    "classify_typed_effects",
    "collect_suite",
    "collect_task_seed",
    "extract_minigrid_state",
    "summarize_typed_source_gate",
]
