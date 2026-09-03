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
import math
import os
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
    collector_kind: str = "minigrid_bfs"


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

MINIWORLD_PUTNEXT_TASK = TypedSourceTask(
    task_id="putnext_3d",
    environment_id="MiniWorld-PutNext-v0",
    required_effects=(TypedEffect.BIND, TypedEffect.RELATE),
    max_depth=249,
    max_states=249,
    collector_kind="miniworld_putnext_oracle",
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


def _round_float(value: Any) -> float:
    return round(float(value), 10)


def extract_miniworld_state(env: Any) -> dict[str, Any]:
    """Extract stable 3D state while treating carried objects as ungrounded."""
    base = env.unwrapped
    carrying_entity = base.agent.carrying
    carrying = None
    if carrying_entity is not None:
        carrying = {
            "type": type(carrying_entity).__name__.lower(),
            "color": str(getattr(carrying_entity, "color", "unknown")),
            "is_open": None,
            "is_locked": None,
        }
    entities: list[dict[str, Any]] = []
    grounded_entities: list[tuple[Any, dict[str, Any]]] = []
    for entity in base.entities:
        record = {
            "position": [_round_float(value) for value in entity.pos],
            "type": type(entity).__name__.lower(),
            "color": str(getattr(entity, "color", "unknown")),
            "is_open": getattr(entity, "is_open", None),
            "is_locked": getattr(entity, "is_locked", None),
        }
        entities.append(record)
        if entity is not carrying_entity:
            grounded_entities.append((entity, record))
    entities.sort(key=lambda item: (item["type"], item["color"], item["position"]))

    relations: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for index, (left_entity, left) in enumerate(grounded_entities):
        for right_entity, right in grounded_entities[index + 1 :]:
            if base.near(left_entity, right_entity):
                relations.add(
                    tuple(sorted((_entity_signature(left), _entity_signature(right))))
                )
    control_state = {
        "agent_position": [_round_float(value) for value in base.agent.pos],
        "agent_direction": _round_float(base.agent.dir),
        "carrying": carrying,
        "objects": entities,
    }
    replay_state = {
        **control_state,
        "step_count": int(base.step_count),
        "relations": [list(map(list, relation)) for relation in sorted(relations)],
    }
    return {
        **replay_state,
        "carrier_bound": carrying is not None,
        "control_state_sha256": content_hash(control_state),
        "replay_state_sha256": content_hash(replay_state),
    }


def _make_env(environment_id: str) -> Any:
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401 - register MiniGrid environments
    except ImportError as exc:  # pragma: no cover - exercised by integration run
        raise RuntimeError(
            "typed source collection needs the 'source-spatial' optional dependencies"
        ) from exc
    return gym.make(environment_id)


def _make_miniworld_env(environment_id: str) -> Any:
    os.environ.setdefault("PYGLET_HEADLESS", "1")
    try:
        import gymnasium as gym
        import miniworld  # noqa: F401 - register MiniWorld environments
    except ImportError as exc:  # pragma: no cover - exercised by integration run
        raise RuntimeError(
            "MiniWorld collection needs gymnasium, miniworld, and a headless GL stack"
        ) from exc
    return gym.make(environment_id, obs_width=16, obs_height=12)


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


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def _miniworld_front_entity(base: Any) -> Any:
    test_position = base.agent.pos + base.agent.dir_vec * 1.5 * base.agent.radius
    return base.intersect(base.agent, test_position, 1.2 * base.agent.radius)


def _miniworld_face_action(base: Any, target: Any) -> int:
    delta = target.pos - base.agent.pos
    desired = math.atan2(-float(delta[2]), float(delta[0]))
    error = _wrap_angle(desired - float(base.agent.dir))
    if abs(error) <= math.radians(7.5):
        return int(base.actions.move_forward)
    return int(base.actions.turn_left if error > 0 else base.actions.turn_right)


def _plan_miniworld_putnext(
    task: TypedSourceTask, seed: int
) -> dict[str, tuple[int, ...]]:
    """Produce deterministic native prefixes to pickup and successful drop forks."""
    env = _make_miniworld_env(task.environment_id)
    env.reset(seed=seed)
    base = env.unwrapped
    prefix: list[int] = []
    bind_prefix: tuple[int, ...] | None = None
    relate_prefix: tuple[int, ...] | None = None
    recovery_count = 0
    try:
        while len(prefix) < task.max_depth:
            if base.agent.carrying is None:
                if _miniworld_front_entity(base) is base.red_box:
                    bind_prefix = tuple(prefix)
                    action = int(base.actions.pickup)
                else:
                    action = _miniworld_face_action(base, base.red_box)
            else:
                if base.near(base.red_box, base.yellow_box):
                    relate_prefix = tuple(prefix)
                    break
                action = _miniworld_face_action(base, base.yellow_box)

            old_position = base.agent.pos.copy()
            old_direction = float(base.agent.dir)
            _, _, terminated, truncated, _ = env.step(action)
            prefix.append(action)
            if terminated or truncated:
                break
            blocked = (
                action in (int(base.actions.turn_left), int(base.actions.turn_right))
                and abs(float(base.agent.dir) - old_direction) < 1e-8
            ) or (
                action == int(base.actions.move_forward)
                and all(
                    abs(float(left) - float(right)) < 1e-8
                    for left, right in zip(base.agent.pos, old_position)
                )
            )
            if not blocked:
                continue
            recovery_count += 1
            side = int(
                base.actions.turn_left
                if (seed + recovery_count) % 2 == 0
                else base.actions.turn_right
            )
            recovery = [int(base.actions.move_back)] * 3
            recovery += [side] * 6
            recovery += [int(base.actions.move_forward)] * 5
            for recovery_action in recovery:
                if len(prefix) >= task.max_depth:
                    break
                _, _, terminated, truncated, _ = env.step(recovery_action)
                prefix.append(recovery_action)
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
    finally:
        env.close()
    if bind_prefix is None or relate_prefix is None:
        raise RuntimeError(
            f"{task.task_id} seed {seed}: deterministic planner failed within "
            f"{task.max_depth} actions"
        )
    return {TypedEffect.BIND.value: bind_prefix, TypedEffect.RELATE.value: relate_prefix}


def _replay_miniworld(
    task: TypedSourceTask, seed: int, prefix: Sequence[int]
) -> tuple[Any, bool, bool]:
    env = _make_miniworld_env(task.environment_id)
    env.reset(seed=seed)
    terminated = truncated = False
    for action in prefix:
        _, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break
    return env, bool(terminated), bool(truncated)


def collect_miniworld_putnext_seed(
    task: TypedSourceTask,
    *,
    seed: int,
    split: str,
    groups_per_effect: int = 1,
) -> dict[str, Any]:
    """Collect exact native forks at oracle-found pickup and placement states."""
    if groups_per_effect != 1:
        raise ValueError("MiniWorld PutNext v4 currently freezes one group per effect")
    prefixes = _plan_miniworld_putnext(task, seed)
    selected_rows: list[dict[str, Any]] = []
    found: dict[str, int] = defaultdict(int)
    replay_mismatches = 0
    for expected_effect in sorted(prefixes):
        prefix = prefixes[expected_effect]
        reference, terminated, truncated = _replay_miniworld(task, seed, prefix)
        try:
            before = extract_miniworld_state(reference)
            action_count = int(reference.action_space.n)
        finally:
            reference.close()
        if terminated or truncated:
            raise RuntimeError("planned MiniWorld fork prefix is already terminal")
        group_id = content_hash(
            ["typed-source-v4", task.task_id, seed, list(prefix)]
        )
        for action_ordinal in range(action_count):
            fork, fork_terminated, fork_truncated = _replay_miniworld(
                task, seed, prefix
            )
            try:
                fork_before = extract_miniworld_state(fork)
                if fork_before["replay_state_sha256"] != before["replay_state_sha256"]:
                    replay_mismatches += 1
                    continue
                _, reward, action_terminated, action_truncated, _ = fork.step(
                    action_ordinal
                )
                after = extract_miniworld_state(fork)
            finally:
                fork.close()
            effects = classify_typed_effects(fork_before, after)
            effect_values = [effect.value for effect in effects]
            selected_rows.append(
                {
                    "schema_version": "typed-source-receipt-v4",
                    "task_id": task.task_id,
                    "environment_id": task.environment_id,
                    "seed": int(seed),
                    "split": split,
                    "group_id": group_id,
                    "prefix_length": len(prefix),
                    "prefix_sha256": content_hash(list(prefix)),
                    "source_action_ordinal": action_ordinal,
                    "before_replay_state_sha256": before["replay_state_sha256"],
                    "after_replay_state_sha256": after["replay_state_sha256"],
                    "before_carrier_bound": bool(before["carrier_bound"]),
                    "after_carrier_bound": bool(after["carrier_bound"]),
                    "typed_effects": effect_values,
                    "native_reward": float(reward),
                    "terminated": bool(action_terminated),
                    "truncated": bool(action_truncated),
                    "prefix_terminal": bool(fork_terminated or fork_truncated),
                    "status": "VALID",
                }
            )
            if expected_effect in effect_values:
                found[expected_effect] += 1
        if found[expected_effect] != 1:
            raise RuntimeError(
                f"{task.task_id} seed {seed}: expected exactly one {expected_effect} "
                f"fork, observed {found[expected_effect]}"
            )
    return {
        "task_id": task.task_id,
        "environment_id": task.environment_id,
        "seed": int(seed),
        "split": split,
        "required_effects": sorted(effect.value for effect in task.required_effects),
        "found_groups": {effect: 1 for effect in sorted(prefixes)},
        "explored_states": max(len(prefix) for prefix in prefixes.values()),
        "frontier_states": 0,
        "replay_mismatches": replay_mismatches,
        "search_exhausted": False,
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
    collections: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    *,
    induction_split: str = "development",
) -> dict[str, Any]:
    """Build an action/coordinate-free candidate IR from qualified receipts."""
    if gate.get("status") != "SOURCE_TYPED_GATE_PASSED":
        raise ValueError("cannot freeze typed effect IR before the source gate passes")
    induction_collections = [
        collection
        for collection in collections
        if str(collection["split"]) == induction_split
    ]
    if not induction_collections:
        raise ValueError(f"no collections found for induction split {induction_split!r}")
    rows = [
        row for collection in induction_collections for row in collection["rows"]
    ]
    effects = sorted(
        {
            effect
            for row in rows
            for effect in row["typed_effects"]
            if effect in {item.value for item in TypedEffect}
        }
    )
    edge_support: dict[tuple[str, str], set[str]] = defaultdict(set)
    for collection in induction_collections:
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
        "induction_split": induction_split,
        "validation_splits": sorted(
            {
                str(collection["split"])
                for collection in collections
                if str(collection["split"]) != induction_split
            }
        ),
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


def summarize_edge_replication_gate(
    effect_ir: Mapping[str, Any],
    requirements: Sequence[Mapping[str, Any]],
    *,
    task_families: Mapping[str, str],
) -> dict[str, Any]:
    """Require an induced edge to recur across tasks and simulator families."""
    indexed_edges = {
        (str(edge["from"]), str(edge["to"])): edge
        for edge in effect_ir.get("edges", ())
    }
    cells: list[dict[str, Any]] = []
    passed = True
    for requirement in requirements:
        key = (str(requirement["from"]), str(requirement["to"]))
        edge = indexed_edges.get(key, {})
        tasks = sorted(str(task) for task in edge.get("supporting_source_tasks", ()))
        families = sorted({task_families[task] for task in tasks if task in task_families})
        minimum_tasks = int(requirement.get("minimum_source_tasks", 1))
        minimum_families = int(requirement.get("minimum_simulator_families", 1))
        cell_passed = len(tasks) >= minimum_tasks and len(families) >= minimum_families
        passed &= cell_passed
        cells.append(
            {
                "from": key[0],
                "to": key[1],
                "supporting_source_tasks": tasks,
                "supporting_simulator_families": families,
                "minimum_source_tasks": minimum_tasks,
                "minimum_simulator_families": minimum_families,
                "passed": cell_passed,
            }
        )
    return {
        "schema_version": "typed-edge-replication-gate-v4",
        "status": "EDGE_REPLICATION_GATE_PASSED" if passed else "EDGE_REPLICATION_GATE_FAILED",
        "cells": cells,
    }


def summarize_effect_value_gate(
    collections: Sequence[Mapping[str, Any]],
    requirements: Sequence[Mapping[str, Any]],
    *,
    task_families: Mapping[str, str],
) -> dict[str, Any]:
    """Corroborate typed effects with source-native official outcomes."""
    cells: list[dict[str, Any]] = []
    passed = True
    for requirement in requirements:
        effect = str(requirement["effect"])
        valued_cells = {
            (str(collection["task_id"]), str(collection["split"]))
            for collection in collections
            if any(
                effect in row["typed_effects"]
                and float(row["native_reward"]) > 0
                and bool(row["terminated"])
                for row in collection["rows"]
            )
        }
        tasks = sorted({task for task, _ in valued_cells})
        families = sorted({task_families[task] for task in tasks if task in task_families})
        minimum_tasks = int(requirement.get("minimum_source_tasks", 1))
        minimum_families = int(requirement.get("minimum_simulator_families", 1))
        minimum_cells = int(requirement.get("minimum_task_split_cells", 1))
        cell_passed = (
            len(tasks) >= minimum_tasks
            and len(families) >= minimum_families
            and len(valued_cells) >= minimum_cells
        )
        passed &= cell_passed
        cells.append(
            {
                "effect": effect,
                "officially_valued_task_split_cells": [
                    {"task_id": task, "split": split}
                    for task, split in sorted(valued_cells)
                ],
                "supporting_source_tasks": tasks,
                "supporting_simulator_families": families,
                "minimum_source_tasks": minimum_tasks,
                "minimum_simulator_families": minimum_families,
                "minimum_task_split_cells": minimum_cells,
                "passed": cell_passed,
            }
        )
    return {
        "schema_version": "typed-effect-value-gate-v4",
        "status": "EFFECT_VALUE_GATE_PASSED" if passed else "EFFECT_VALUE_GATE_FAILED",
        "cells": cells,
    }


def collect_suite(
    tasks: Sequence[TypedSourceTask],
    *,
    seeds: Sequence[int],
    namespace: str,
    groups_per_effect: int = 1,
) -> dict[str, Any]:
    splits = split_seeds(seeds, namespace=namespace)
    collections: list[dict[str, Any]] = []
    for task in tasks:
        if task.collector_kind == "minigrid_bfs":
            collector = collect_task_seed
        elif task.collector_kind == "miniworld_putnext_oracle":
            collector = collect_miniworld_putnext_seed
        else:
            raise ValueError(f"unknown source collector kind: {task.collector_kind}")
        for seed in sorted({int(value) for value in seeds}):
            collections.append(
                collector(
                    task,
                    seed=seed,
                    split=splits[seed],
                    groups_per_effect=groups_per_effect,
                )
            )
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
                **(
                    {"collector_kind": task.collector_kind}
                    if task.collector_kind != "minigrid_bfs"
                    else {}
                ),
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
    "MINIWORLD_PUTNEXT_TASK",
    "TypedEffect",
    "TypedSourceTask",
    "build_effect_ir",
    "classify_typed_effects",
    "collect_suite",
    "collect_miniworld_putnext_seed",
    "collect_task_seed",
    "extract_minigrid_state",
    "extract_miniworld_state",
    "summarize_edge_replication_gate",
    "summarize_effect_value_gate",
    "summarize_typed_source_gate",
]
