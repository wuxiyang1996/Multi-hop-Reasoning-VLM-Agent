"""Intervention-grounded POSITION/COMMIT skill extraction from Sokoban.

The legacy Sokoban observations are complete symbolic boards.  This module
first checks a local transition model against the action/next-state receipts,
then uses the model to enumerate every native intervention at a frozen state.
Only relational option features and a two-option value model are transferable;
coordinates and source action names stay in the source grounding.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import stable_hash


PLAN_VERSION = "SOKOBAN_COMMIT_PLAN_V1"
ARTIFACT_VERSION = "SOKOBAN_COMMIT_SKILL_V1"
QUALIFICATION_VERSION = "SOKOBAN_COMMIT_QUALIFICATION_V1"
SPLITS = ("discovery", "qualification", "held_out")
OPTIONS = ("POSITION", "COMMIT")
NATIVE_ACTIONS = (
    "up", "down", "left", "right",
    "push up", "push down", "push left", "push right", "no_op",
)
DELTAS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
ROW_PATTERN = re.compile(
    r"^\s*\d+\s*\|\s*(?P<kind>[A-Za-z ]+?)\s*\|\s*"
    r"\((?P<x>-?\d+)\s*,\s*(?P<y>-?\d+)\)\s*$",
    re.MULTILINE,
)
FEATURE_NAMES = (
    "option_is_position",
    "option_is_commit",
    "applicable_fraction",
    "state_change_fraction",
    "movable_change_fraction",
    "progress_available",
    "regression_fraction",
    "deadlock_fraction",
    "best_assignment_improvement",
    "unsatisfied_goal_fraction",
    "actor_to_movable_proximity",
    "relative_option_width",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SokobanState:
    width: int
    height: int
    walls: frozenset[tuple[int, int]]
    docks: frozenset[tuple[int, int]]
    boxes: frozenset[tuple[int, int]]
    worker: tuple[int, int]

    def validate(self) -> None:
        if self.width < 3 or self.height < 3:
            raise ValueError("Sokoban board is too small")
        if len(self.boxes) == 0 or len(self.boxes) != len(self.docks):
            raise ValueError("Sokoban board must have equally many boxes and docks")
        if self.worker in self.walls or self.worker in self.boxes:
            raise ValueError("worker overlaps a wall or box")
        if self.walls & self.boxes:
            raise ValueError("box overlaps a wall")

    def body(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "walls": sorted(self.walls),
            "docks": sorted(self.docks),
            "boxes": sorted(self.boxes),
            "worker": self.worker,
        }

    @property
    def solved(self) -> bool:
        return self.boxes == self.docks


@dataclass(frozen=True)
class SokobanTransition:
    before: SokobanState
    action: str
    after: SokobanState
    worker_moved: bool
    box_moved: bool
    boxes_on_docks_delta: int
    created_static_deadlock: bool

    @property
    def state_changed(self) -> bool:
        return self.before != self.after

    @property
    def option(self) -> str:
        return "COMMIT" if self.action.startswith("push ") else "POSITION"

    @property
    def role(self) -> str:
        if not self.state_changed:
            return "INAPPLICABLE"
        if not self.box_moved:
            return "REVERSIBLE_POSITION"
        if self.after.solved:
            return "TERMINAL_COMMIT"
        if self.boxes_on_docks_delta > 0:
            return "PROGRESS_COMMIT"
        if self.boxes_on_docks_delta < 0:
            return "REGRESS_COMMIT"
        if self.created_static_deadlock:
            return "RISKY_COMMIT"
        return "SAFE_COMMIT"


def parse_state(text: str) -> SokobanState:
    cells: dict[tuple[int, int], str] = {}
    for match in ROW_PATTERN.finditer(text):
        coordinate = (int(match.group("x")), int(match.group("y")))
        if coordinate in cells:
            raise ValueError(f"duplicate Sokoban coordinate: {coordinate}")
        cells[coordinate] = match.group("kind").strip().lower()
    if not cells:
        raise ValueError("no Sokoban table rows found")
    width = max(x for x, _ in cells) + 1
    height = max(y for _, y in cells) + 1
    expected = {(x, y) for y in range(height) for x in range(width)}
    if set(cells) != expected:
        raise ValueError("Sokoban table is not a complete rectangular board")
    walls = frozenset(cell for cell, kind in cells.items() if kind == "wall")
    boxes = frozenset(
        cell for cell, kind in cells.items() if kind in {"box", "box on dock"}
    )
    docks = frozenset(
        cell for cell, kind in cells.items()
        if kind in {"dock", "box on dock", "unknown"}
    )
    workers = [cell for cell, kind in cells.items() if kind == "worker"]
    unknown = [cell for cell, kind in cells.items() if kind == "unknown"]
    # The source renderer calls player-on-dock `Unknown`; the raw state code is
    # otherwise fully observed.  There is exactly one worker in a valid board.
    if not workers and len(unknown) == 1:
        workers = unknown
    if len(workers) != 1:
        raise ValueError(f"expected one worker, found {len(workers)}")
    state = SokobanState(width, height, walls, docks, boxes, workers[0])
    state.validate()
    return state


def state_to_text(state: SokobanState) -> str:
    """Serialize a state with the same complete-table contract as the logs."""

    state.validate()
    lines = ["ID  | Item Type    | Position", "-----------------------------"]
    index = 1
    for y in range(state.height):
        for x in range(state.width):
            cell = (x, y)
            if cell in state.walls:
                kind = "Wall"
            elif cell == state.worker and cell in state.docks:
                kind = "Unknown"
            elif cell == state.worker:
                kind = "Worker"
            elif cell in state.boxes and cell in state.docks:
                kind = "Box on Dock"
            elif cell in state.boxes:
                kind = "Box"
            elif cell in state.docks:
                kind = "Dock"
            else:
                kind = "Empty"
            lines.append(f"{index:<3} | {kind:<12} | ({x}, {y})")
            index += 1
    return "\n".join(lines)


def _inside(state: SokobanState, cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < state.width and 0 <= cell[1] < state.height


def _add(cell: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
    return cell[0] + delta[0], cell[1] + delta[1]


def _is_static_deadlock(state: SokobanState, box: tuple[int, int]) -> bool:
    if box in state.docks:
        return False

    def blocked(delta: tuple[int, int]) -> bool:
        target = _add(box, delta)
        return not _inside(state, target) or target in state.walls

    return (
        (blocked(DELTAS["up"]) or blocked(DELTAS["down"]))
        and (blocked(DELTAS["left"]) or blocked(DELTAS["right"]))
    )


def simulate(state: SokobanState, action: str) -> SokobanTransition:
    if action not in NATIVE_ACTIONS:
        raise ValueError(f"unsupported Sokoban action: {action}")
    before_on_docks = len(state.boxes & state.docks)
    worker = state.worker
    boxes = set(state.boxes)
    worker_moved = box_moved = False
    moved_box_target: tuple[int, int] | None = None
    if action != "no_op":
        is_push = action.startswith("push ")
        direction = action.split()[-1]
        adjacent = _add(worker, DELTAS[direction])
        if is_push:
            beyond = _add(adjacent, DELTAS[direction])
            if (
                adjacent in boxes
                and _inside(state, beyond)
                and beyond not in state.walls
                and beyond not in boxes
            ):
                boxes.remove(adjacent)
                boxes.add(beyond)
                worker = adjacent
                worker_moved = box_moved = True
                moved_box_target = beyond
        elif (
            _inside(state, adjacent)
            and adjacent not in state.walls
            and adjacent not in boxes
        ):
            worker = adjacent
            worker_moved = True
    after = SokobanState(
        state.width, state.height, state.walls, state.docks,
        frozenset(boxes), worker,
    )
    after.validate()
    after_on_docks = len(after.boxes & after.docks)
    transition = SokobanTransition(
        before=state,
        action=action,
        after=after,
        worker_moved=worker_moved,
        box_moved=box_moved,
        boxes_on_docks_delta=after_on_docks - before_on_docks,
        created_static_deadlock=(
            moved_box_target is not None and _is_static_deadlock(after, moved_box_target)
        ),
    )
    return transition


def transition_reward(transition: SokobanTransition) -> float:
    reward = -0.1
    if transition.boxes_on_docks_delta > 0:
        reward += float(transition.boxes_on_docks_delta)
    elif transition.boxes_on_docks_delta < 0:
        reward += float(transition.boxes_on_docks_delta)
    if transition.after.solved:
        reward += 10.0
    return reward


def _assignment_cost(state: SokobanState) -> float:
    boxes = tuple(sorted(state.boxes))
    docks = tuple(sorted(state.docks))
    costs = (
        sum(abs(box[0] - dock[0]) + abs(box[1] - dock[1])
            for box, dock in zip(boxes, ordering))
        for ordering in itertools.permutations(docks)
    )
    return float(min(costs))


def _solver_key(state: SokobanState) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    return state.worker, tuple(sorted(state.boxes))


def shortest_solution(
    state: SokobanState, *, maximum_nodes: int = 100_000,
) -> tuple[str, ...] | None:
    """Return one shortest primitive solution using A* over the full state."""

    if state.solved:
        return ()
    start_key = _solver_key(state)
    queue: list[tuple[float, int, str, SokobanState, tuple[str, ...]]] = []
    serial = 0
    heapq.heappush(queue, (_assignment_cost(state), 0, "", state, ()))
    best = {start_key: 0}
    expanded = 0
    while queue and expanded < maximum_nodes:
        _priority, distance, _tie, current, plan = heapq.heappop(queue)
        if distance != best.get(_solver_key(current)):
            continue
        expanded += 1
        for action in NATIVE_ACTIONS[:-1]:
            transition = simulate(current, action)
            if not transition.state_changed or transition.created_static_deadlock:
                continue
            after = transition.after
            next_distance = distance + 1
            key = _solver_key(after)
            if next_distance >= best.get(key, math.inf):
                continue
            next_plan = (*plan, action)
            if after.solved:
                return next_plan
            best[key] = next_distance
            serial += 1
            tie = stable_hash((serial, action, key))
            heapq.heappush(
                queue,
                (next_distance + _assignment_cost(after), next_distance,
                 tie, after, next_plan),
            )
    return None


def _worker_path(
    state: SokobanState, target: tuple[int, int],
) -> tuple[str, ...] | None:
    if state.worker == target:
        return ()
    blocked = set(state.walls) | set(state.boxes)
    queue = deque([(state.worker, ())])
    seen = {state.worker}
    while queue:
        cell, path = queue.popleft()
        for action in ("up", "down", "left", "right"):
            after = _add(cell, DELTAS[action])
            if not _inside(state, after) or after in blocked or after in seen:
                continue
            next_path = (*path, action)
            if after == target:
                return next_path
            seen.add(after)
            queue.append((after, next_path))
    return None


def generate_solvable_state(
    *, seed: int, width: int, height: int, box_count: int,
    reverse_pulls: int, interior_wall_count: int,
    minimum_solution_length: int = 8,
    maximum_attempts: int = 128,
    maximum_solver_nodes: int = 100_000,
) -> SokobanState:
    """Generate a fresh board by pulling boxes away from a solved state.

    Reverse pulls guarantee that at least one forward solution exists.  The
    exact A* solver is used only to reject degenerate short boards; no target
    data or frozen artifact prediction participates in generation.
    """

    if min(width, height) < 7 or box_count < 1 or reverse_pulls < 1:
        raise ValueError("invalid procedural Sokoban dimensions")
    for attempt in range(maximum_attempts):
        rng = random.Random(stable_hash((seed, attempt, "SOKOBAN_FRESH_V1")))
        border = {
            (x, y) for y in range(height) for x in range(width)
            if x in {0, width - 1} or y in {0, height - 1}
        }
        deep_cells = [
            (x, y) for y in range(2, height - 2) for x in range(2, width - 2)
        ]
        if len(deep_cells) < box_count:
            raise ValueError("board interior cannot hold requested boxes")
        docks = frozenset(rng.sample(deep_cells, box_count))
        interior = [
            (x, y) for y in range(1, height - 1) for x in range(1, width - 1)
            if (x, y) not in docks
        ]
        walls = set(border)
        walls.update(rng.sample(
            interior, min(interior_wall_count, max(0, len(interior) // 8))
        ))
        free = [cell for cell in interior if cell not in walls and cell not in docks]
        if not free:
            continue
        state = SokobanState(
            width, height, frozenset(walls), docks, docks, rng.choice(free),
        )
        pulls_completed = 0
        for _pull_index in range(reverse_pulls * 8):
            candidates = []
            for box in sorted(state.boxes):
                for direction, delta in DELTAS.items():
                    old_box = (box[0] - delta[0], box[1] - delta[1])
                    old_worker = (box[0] - 2 * delta[0], box[1] - 2 * delta[1])
                    if (
                        not _inside(state, old_worker)
                        or old_box in state.walls or old_worker in state.walls
                        or old_box in state.boxes or old_worker in state.boxes
                    ):
                        continue
                    path = _worker_path(state, old_box)
                    if path is not None:
                        candidates.append((box, direction, old_box, old_worker, path))
            if not candidates:
                break
            box, direction, old_box, old_worker, path = rng.choice(candidates)
            current = state
            for action in path:
                current = simulate(current, action).after
            boxes = set(current.boxes)
            boxes.remove(box)
            boxes.add(old_box)
            state = SokobanState(
                width, height, current.walls, current.docks,
                frozenset(boxes), old_worker,
            )
            pulls_completed += 1
            if pulls_completed >= reverse_pulls:
                break
        if pulls_completed < reverse_pulls or state.solved:
            continue
        plan = shortest_solution(state, maximum_nodes=maximum_solver_nodes)
        if plan is not None and len(plan) >= minimum_solution_length:
            return state
    raise RuntimeError(f"could not generate solvable Sokoban board for seed {seed}")


def _json_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*.json")):
        if path.name != "labeling_summary.json":
            yield path


def _snapshot_body(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "snapshot_id"}


def build_plan(
    source_dir: Path, *, snapshots_per_episode: int,
    maximum_source_step: int,
    stratify_commit_precondition: bool = True,
) -> dict[str, Any]:
    if snapshots_per_episode < 1 or maximum_source_step < 1:
        raise ValueError("invalid source plan limits")
    source_dir = source_dir.resolve()
    episodes = []
    for path in _json_files(source_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("experiences"), list):
            continue
        episodes.append((str(payload["episode_id"]), path, payload))
    if len(episodes) < 3:
        raise ValueError("at least three Sokoban episodes are required")
    split_by_episode = {
        episode_id: SPLITS[index % len(SPLITS)]
        for index, (episode_id, _path, _payload) in enumerate(sorted(episodes))
    }
    snapshots = []
    for episode_id, path, payload in sorted(episodes):
        experiences = payload["experiences"]
        candidates: list[tuple[str, int, Mapping[str, Any], str]] = []
        for index, experience in enumerate(experiences):
            if index > maximum_source_step:
                continue
            if str(experience.get("action")) not in NATIVE_ACTIONS:
                continue
            try:
                state = parse_state(str(experience["state"]))
                parse_state(str(experience["next_state"]))
            except (KeyError, ValueError):
                continue
            if state.solved:
                continue
            commit_ready = any(
                simulate(state, action).box_moved
                for action in NATIVE_ACTIONS if action.startswith("push ")
            )
            stratum = "COMMIT_READY" if commit_ready else "COMMIT_NOT_READY"
            rank = stable_hash({
                "version": PLAN_VERSION,
                "episode_id": episode_id,
                "step": index,
                "stratum": stratum,
            })
            candidates.append((rank, index, experience, stratum))
        if stratify_commit_precondition and snapshots_per_episode >= 2:
            ready_count = snapshots_per_episode // 2
            not_ready_count = snapshots_per_episode - ready_count
            ready = sorted(row for row in candidates if row[3] == "COMMIT_READY")
            not_ready = sorted(
                row for row in candidates if row[3] == "COMMIT_NOT_READY"
            )
            selected = [*ready[:ready_count], *not_ready[:not_ready_count]]
            selected_ids = {(row[0], row[1]) for row in selected}
            if len(selected) < snapshots_per_episode:
                selected.extend(
                    row for row in sorted(candidates)
                    if (row[0], row[1]) not in selected_ids
                )
                selected = selected[:snapshots_per_episode]
        else:
            selected = sorted(candidates)[:snapshots_per_episode]
        if len(selected) != snapshots_per_episode:
            raise ValueError(f"episode {episode_id} has too few eligible states")
        for rank, index, experience, stratum in selected:
            body = {
                "episode_id": episode_id,
                "source_file": path.name,
                "source_file_sha256": file_sha256(path),
                "split": split_by_episode[episode_id],
                "step": index,
                "selection_rank_sha256": rank,
                "selection_stratum": stratum,
                "state": str(experience["state"]),
                "source_action": str(experience["action"]),
                "source_next_state": str(experience["next_state"]),
                "source_reward": float(experience["reward"]),
                "native_actions": list(NATIVE_ACTIONS),
            }
            snapshots.append(body | {"snapshot_id": stable_hash(body)})
    plan_body = {
        "plan_version": PLAN_VERSION,
        "status": "FROZEN_BEFORE_SOLVER_LABELS_OR_SPLIT_METRICS",
        "claim_boundary": (
            "OUTCOME_BLIND_STATE_SELECTION; LEGACY_NEXT_STATE_USED_ONLY_TO_"
            "VALIDATE_THE_SOURCE_TRANSITION_MODEL"
        ),
        "selection": {
            "split_contract": "SORT_EPISODE_ID_ROUND_ROBIN_DQH_V1",
            "rank": "SHA256_EPISODE_ID_AND_STEP_ONLY_V1",
            "snapshots_per_episode": snapshots_per_episode,
            "maximum_source_step": maximum_source_step,
            "stratify_commit_precondition": stratify_commit_precondition,
            "stratification_reads": (
                "BEFORE_STATE_ONLY_NO_SOURCE_ACTION_REWARD_NEXT_STATE_OR_SOLVER_LABEL"
            ),
        },
        "source_dir": str(source_dir),
        "episode_count": len(episodes),
        "split_counts": dict(sorted(Counter(
            row["split"] for row in snapshots
        ).items())),
        "snapshots": snapshots,
    }
    return plan_body | {"plan_sha256": stable_hash(plan_body)}


def build_fresh_confirmation_plan(
    *, seeds: Sequence[int], snapshots_per_episode: int = 4,
    width: int = 8, height: int = 8, box_count: int = 2,
    reverse_pulls: int = 4, interior_wall_count: int = 2,
    maximum_solver_nodes: int = 100_000,
) -> dict[str, Any]:
    """Freeze fresh generated episodes for an independent source confirmation."""

    if len(set(seeds)) != len(seeds) or len(seeds) < 6:
        raise ValueError("fresh confirmation needs at least six unique seeds")
    if snapshots_per_episode < 2:
        raise ValueError("fresh confirmation needs at least two states per episode")
    snapshots = []
    generator_config = {
        "seeds": list(map(int, seeds)),
        "snapshots_per_episode": snapshots_per_episode,
        "width": width,
        "height": height,
        "box_count": box_count,
        "reverse_pulls": reverse_pulls,
        "interior_wall_count": interior_wall_count,
        "maximum_solver_nodes": maximum_solver_nodes,
    }
    generator_sha256 = stable_hash(generator_config)
    episode_receipts = []
    for seed in seeds:
        initial = generate_solvable_state(
            seed=int(seed), width=width, height=height, box_count=box_count,
            reverse_pulls=reverse_pulls,
            interior_wall_count=interior_wall_count,
            maximum_solver_nodes=maximum_solver_nodes,
        )
        plan = shortest_solution(initial, maximum_nodes=maximum_solver_nodes)
        if not plan:
            raise RuntimeError(f"fresh source seed {seed} unexpectedly has no solution")
        trajectory = []
        state = initial
        for step, action in enumerate(plan):
            transition = simulate(state, action)
            commit_ready = any(
                simulate(state, candidate).box_moved
                for candidate in NATIVE_ACTIONS if candidate.startswith("push ")
            )
            stratum = "COMMIT_READY" if commit_ready else "COMMIT_NOT_READY"
            rank = stable_hash({
                "version": PLAN_VERSION,
                "fresh_generator_sha256": generator_sha256,
                "seed": seed,
                "step": step,
                "stratum": stratum,
            })
            trajectory.append((rank, step, action, transition, stratum))
            state = transition.after
        per_stratum = snapshots_per_episode // 2
        ready = sorted(row for row in trajectory if row[4] == "COMMIT_READY")
        not_ready = sorted(row for row in trajectory if row[4] == "COMMIT_NOT_READY")
        selected = [*ready[:per_stratum], *not_ready[:per_stratum]]
        selected_ids = {(row[0], row[1]) for row in selected}
        if len(selected) < snapshots_per_episode:
            selected.extend(
                row for row in sorted(trajectory)
                if (row[0], row[1]) not in selected_ids
            )
            selected = selected[:snapshots_per_episode]
        if len(selected) != snapshots_per_episode:
            raise RuntimeError(f"fresh source seed {seed} has too short a trajectory")
        episode_id = f"procedural-sokoban-{int(seed)}"
        for rank, step, action, transition, stratum in selected:
            body = {
                "episode_id": episode_id,
                "source_file": f"PROCEDURAL_SEED_{int(seed)}",
                "source_file_sha256": generator_sha256,
                "split": "held_out",
                "step": step,
                "selection_rank_sha256": rank,
                "selection_stratum": stratum,
                "state": state_to_text(transition.before),
                "source_action": action,
                "source_next_state": state_to_text(transition.after),
                "source_reward": transition_reward(transition),
                "native_actions": list(NATIVE_ACTIONS),
            }
            snapshots.append(body | {"snapshot_id": stable_hash(body)})
        episode_receipts.append({
            "episode_id": episode_id,
            "seed": int(seed),
            "initial_state_sha256": stable_hash(initial.body()),
            "shortest_solution_length": len(plan),
            "terminal_state_sha256": stable_hash(state.body()),
            "terminal_solved": state.solved,
        })
    plan_body = {
        "plan_version": PLAN_VERSION,
        "status": "FROZEN_FRESH_CONFIRMATION_BEFORE_ARTIFACT_PREDICTIONS",
        "claim_boundary": (
            "FRESH_PROCEDURAL_SOURCE_CONFIRMATION_FROM_A_TRANSITION_MODEL_"
            "VALIDATED_AGAINST_REAL_LOGS; NOT_TARGET_TRANSFER_EVIDENCE"
        ),
        "selection": {
            "split_contract": "ALL_FRESH_EPISODES_HELD_OUT_V1",
            "rank": "SHA256_SEED_STEP_AND_BEFORE_STATE_STRATUM_V1",
            "snapshots_per_episode": snapshots_per_episode,
            "maximum_source_step": max(row["step"] for row in snapshots),
            "stratify_commit_precondition": True,
            "stratification_reads": (
                "BEFORE_STATE_ONLY_NO_ARTIFACT_PREDICTION_OR_CONTROL_METRIC"
            ),
        },
        "source_dir": "PROCEDURAL_SOKOBAN_GENERATOR_V1",
        "generator_config": generator_config,
        "generator_sha256": generator_sha256,
        "episode_count": len(seeds),
        "episode_receipts": episode_receipts,
        "split_counts": {"held_out": len(snapshots)},
        "snapshots": snapshots,
    }
    return plan_body | {"plan_sha256": stable_hash(plan_body)}


def validate_plan(plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    body = dict(plan)
    claimed = str(body.pop("plan_sha256", ""))
    if stable_hash(body) != claimed or plan.get("plan_version") != PLAN_VERSION:
        raise ValueError("invalid Sokoban source plan")
    snapshots = tuple(dict(row) for row in plan.get("snapshots", ()))
    if not snapshots:
        raise ValueError("empty Sokoban source plan")
    for row in snapshots:
        if row.get("split") not in SPLITS:
            raise ValueError("invalid source split")
        if stable_hash(_snapshot_body(row)) != row.get("snapshot_id"):
            raise ValueError("snapshot hash mismatch")
        if tuple(row.get("native_actions", ())) != NATIVE_ACTIONS:
            raise ValueError("native action contract changed")
    counts = dict(sorted(Counter(row["split"] for row in snapshots).items()))
    if counts != plan.get("split_counts"):
        raise ValueError("source split counts changed")
    return snapshots


def validate_recorded_transition(row: Mapping[str, Any]) -> dict[str, Any]:
    before = parse_state(str(row["state"]))
    observed_after = parse_state(str(row["source_next_state"]))
    transition = simulate(before, str(row["source_action"]))
    reward_match = abs(
        transition_reward(transition) - float(row["source_reward"])
    ) < 1e-9
    # A terminal source action immediately loads the next built-in level, so
    # the reward/completion is verifiable but its returned board is expected
    # to differ from the solved board produced by the local transition model.
    state_match = transition.after == observed_after
    terminal_level_advance = transition.after.solved and reward_match
    return {
        "state_match": state_match,
        "reward_match": reward_match,
        "terminal_level_advance": terminal_level_advance,
        "passed": reward_match and (state_match or terminal_level_advance),
        "predicted_role": transition.role,
    }


def _minimum_actor_box_distance(state: SokobanState) -> float:
    return float(min(
        abs(state.worker[0] - box[0]) + abs(state.worker[1] - box[1])
        for box in state.boxes
    ))


def option_features(state: SokobanState, option: str) -> tuple[float, ...]:
    if option not in OPTIONS:
        raise ValueError(f"unsupported canonical option: {option}")
    transitions = [
        simulate(state, action) for action in NATIVE_ACTIONS
        if (action.startswith("push ")) == (option == "COMMIT")
        and action != "no_op"
    ]
    applicable = [row for row in transitions if row.state_changed]
    denom = max(1, len(transitions))
    before_cost = _assignment_cost(state)
    improvements = [before_cost - _assignment_cost(row.after) for row in applicable]
    unsatisfied = len(state.boxes - state.docks) / len(state.boxes)
    proximity = 1.0 / (1.0 + _minimum_actor_box_distance(state))
    return (
        float(option == "POSITION"),
        float(option == "COMMIT"),
        len(applicable) / denom,
        sum(row.state_changed for row in transitions) / denom,
        sum(row.box_moved for row in transitions) / denom,
        float(any(row.boxes_on_docks_delta > 0 for row in applicable)),
        sum(row.boxes_on_docks_delta < 0 for row in applicable) / denom,
        sum(row.created_static_deadlock for row in applicable) / denom,
        max(improvements, default=0.0) / max(1.0, state.width + state.height),
        unsatisfied,
        proximity,
        len(transitions) / (len(NATIVE_ACTIONS) - 1),
    )


def _fit_ridge(
    features: Sequence[Sequence[float]], labels: Sequence[float], *, alpha: float,
) -> dict[str, Any]:
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (matrix - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )
    return {
        "kind": "STANDARDIZED_RIDGE_OPTION_VALUE_V1",
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "alpha": alpha,
        "training_rows": len(labels),
    }


def _predict(model: Mapping[str, Any], features: Sequence[float]) -> float:
    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    design = np.concatenate([[1.0], (values - mean) / scale])
    return float(design @ coefficients)


def predict_option_value(
    model: Mapping[str, Any], features: Sequence[float],
) -> float:
    return _predict(model, features)


def _solver_example(row: Mapping[str, Any], maximum_nodes: int) -> dict[str, Any] | None:
    state = parse_state(str(row["state"]))
    plan = shortest_solution(state, maximum_nodes=maximum_nodes)
    if not plan:
        return None
    optimal_option = "COMMIT" if plan[0].startswith("push ") else "POSITION"
    return {
        "snapshot_id": str(row["snapshot_id"]),
        "episode_id": str(row["episode_id"]),
        "step": int(row["step"]),
        "solution_length": len(plan),
        "optimal_first_option": optimal_option,
        "features": {
            option: option_features(state, option) for option in OPTIONS
        },
        "transition_validation": validate_recorded_transition(row),
    }


def fit_discovery_artifact(
    plan: Mapping[str, Any], *, maximum_solver_nodes: int = 100_000,
    ridge_alpha: float = 0.5,
    minimum_examples_per_option: int = 6,
) -> dict[str, Any]:
    rows = [row for row in validate_plan(plan) if row["split"] == "discovery"]
    examples = [
        example for row in rows
        if (example := _solver_example(row, maximum_solver_nodes)) is not None
    ]
    if len(examples) < 8:
        raise ValueError("too few solvable discovery snapshots")
    option_counts = Counter(row["optimal_first_option"] for row in examples)
    if any(option_counts[option] < minimum_examples_per_option for option in OPTIONS):
        raise ValueError(
            "discovery lacks minimum support per canonical option: "
            f"{dict(option_counts)}"
        )
    if not all(row["transition_validation"]["passed"] for row in examples):
        raise ValueError("source transition model failed discovery validation")
    features = []
    authentic_labels = []
    swapped_labels = []
    for row in examples:
        optimal = str(row["optimal_first_option"])
        for option in OPTIONS:
            features.append(row["features"][option])
            label = float(option == optimal)
            authentic_labels.append(label)
            swapped_labels.append(1.0 - label)
    authentic_model = _fit_ridge(features, authentic_labels, alpha=ridge_alpha)
    shuffled_model = _fit_ridge(features, swapped_labels, alpha=ridge_alpha)
    marginal = sum(authentic_labels) / len(authentic_labels)
    program = {
        "canonical_predicates": [
            "COMMIT_PRECONDITION_SATISFIED",
            "PREDICTED_RELATIONAL_PROGRESS",
            "PREDICTED_IRREVERSIBLE_RISK",
            "EXPECTED_EFFECT_OBSERVED",
        ],
        "options": list(OPTIONS),
        "control_flow": [
            {
                "when": "COMMIT_PRECONDITION_SATISFIED_AND_COMMIT_VALUE_HIGHER",
                "select": "COMMIT",
                "then": "VERIFY",
            },
            {
                "when": "COMMIT_PRECONDITION_UNSATISFIED_OR_POSITION_VALUE_HIGHER",
                "select": "POSITION",
                "then": "RECOMPUTE_PRECONDITION",
            },
            {
                "when": "EXPECTED_EFFECT_REFUTED_OR_RISK_UNKNOWN",
                "select": "REPLAN_OR_ABSTAIN",
            },
        ],
        "authority": (
            "SOURCE_SELECTS_CANONICAL_OPTION_ONLY; TARGET_HARNESS_REALIZES_"
            "ONE_NATIVE_ACTION_WITHIN_THE_SELECTED_OPTION"
        ),
    }
    body = {
        "artifact_version": ARTIFACT_VERSION,
        "lifecycle": "DISCOVERY_FROZEN_AWAITING_SOURCE_QUALIFICATION",
        "claim_boundary": (
            "SOKOBAN_COORDINATES_AND_ACTION_NAMES_ARE_SOURCE_GROUNDING_ONLY;_"
            "TRANSFER_OBJECT_IS_TYPED_OPTION_VALUE_AND_CONTROL_FLOW"
        ),
        "plan_sha256": str(plan["plan_sha256"]),
        "maximum_solver_nodes": maximum_solver_nodes,
        "ridge_alpha": ridge_alpha,
        "minimum_discovery_examples_per_option": minimum_examples_per_option,
        "source_grounding": {
            "game": "sokoban",
            "native_actions": list(NATIVE_ACTIONS),
            "solvable_discovery_snapshots": len(examples),
            "optimal_option_counts": dict(sorted(option_counts.items())),
            "transition_validation_passed": sum(
                row["transition_validation"]["passed"] for row in examples
            ),
            "source_snapshot_ids": [row["snapshot_id"] for row in examples],
        },
        "transferable_program": program,
        "models": {
            "authentic": authentic_model,
            "within_state_option_swap": shuffled_model,
            "source_marginal": {"constant": marginal},
        },
        "raw_source_action_tokens_transferred": False,
        "raw_source_coordinates_transferred": False,
    }
    return body | {"artifact_sha256": stable_hash(body)}


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    body = dict(artifact)
    claimed = str(body.pop("artifact_sha256", ""))
    if stable_hash(body) != claimed or artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("invalid Sokoban commit artifact")
    if artifact["transferable_program"]["options"] != list(OPTIONS):
        raise ValueError("canonical option contract changed")
    for name in ("authentic", "within_state_option_swap"):
        if tuple(artifact["models"][name]["feature_names"]) != FEATURE_NAMES:
            raise ValueError("option value feature contract changed")


def _phase_permuted_features(features: Sequence[float]) -> tuple[float, ...]:
    values = list(map(float, features))
    values[0], values[1] = values[1], values[0]
    values[2], values[4] = values[4], values[2]
    values[5], values[7] = values[7], values[5]
    return tuple(values)


def qualify_artifact(
    plan: Mapping[str, Any], artifact: Mapping[str, Any], *, split: str,
    maximum_solver_nodes: int = 100_000,
    minimum_eligible_snapshots: int = 12,
    minimum_examples_per_option: int = 6,
    minimum_accuracy: float = 0.60,
) -> dict[str, Any]:
    if split not in {"qualification", "held_out"}:
        raise ValueError("source evaluation split must be qualification or held_out")
    validate_artifact(artifact)
    fresh_confirmation = (
        split == "held_out"
        and plan.get("status")
        == "FROZEN_FRESH_CONFIRMATION_BEFORE_ARTIFACT_PREDICTIONS"
    )
    if artifact.get("plan_sha256") != plan.get("plan_sha256") and not fresh_confirmation:
        raise ValueError("source plan/artifact mismatch")
    rows = [row for row in validate_plan(plan) if row["split"] == split]
    examples = [
        example for row in rows
        if (example := _solver_example(row, maximum_solver_nodes)) is not None
    ]
    conditions = (
        "authentic", "within_state_option_swap", "source_marginal",
        "phase_permuted",
    )
    predictions: dict[str, list[dict[str, Any]]] = {name: [] for name in conditions}
    authentic_model = artifact["models"]["authentic"]
    swapped_model = artifact["models"]["within_state_option_swap"]
    marginal = float(artifact["models"]["source_marginal"]["constant"])
    for row in examples:
        optimal = str(row["optimal_first_option"])
        for condition in conditions:
            scores = {}
            for option in OPTIONS:
                features = row["features"][option]
                if condition == "authentic":
                    score = _predict(authentic_model, features)
                elif condition == "within_state_option_swap":
                    score = _predict(swapped_model, features)
                elif condition == "phase_permuted":
                    score = _predict(authentic_model, _phase_permuted_features(features))
                else:
                    score = marginal
                scores[option] = float(score)
            selected = max(
                OPTIONS,
                key=lambda option: (
                    scores[option],
                    stable_hash((row["snapshot_id"], condition, option)),
                ),
            )
            probability = 1.0 / (1.0 + math.exp(
                -(scores[optimal] - scores[next(
                    option for option in OPTIONS if option != optimal
                )])
            ))
            predictions[condition].append({
                "snapshot_id": row["snapshot_id"],
                "episode_id": row["episode_id"],
                "optimal_option": optimal,
                "selected_option": selected,
                "correct": selected == optimal,
                "optimal_probability": probability,
                "brier": (probability - 1.0) ** 2,
                "scores": scores,
            })
    metrics = {}
    for condition, condition_rows in predictions.items():
        metrics[condition] = {
            "n": len(condition_rows),
            "accuracy": (
                sum(row["correct"] for row in condition_rows) / len(condition_rows)
                if condition_rows else None
            ),
            "mean_brier": (
                sum(row["brier"] for row in condition_rows) / len(condition_rows)
                if condition_rows else None
            ),
        }
    transition_checks = [row["transition_validation"] for row in examples]
    option_counts = Counter(row["optimal_first_option"] for row in examples)
    validation_rate = (
        sum(row["passed"] for row in transition_checks) / len(transition_checks)
        if transition_checks else 0.0
    )
    authentic = metrics["authentic"]
    controls = [metrics[name] for name in conditions if name != "authentic"]
    eligible = (
        len(examples) >= minimum_eligible_snapshots
        and all(option_counts[option] >= minimum_examples_per_option
                for option in OPTIONS)
    )
    accuracy_superiority = bool(eligible and all(
        float(authentic["accuracy"]) > float(row["accuracy"]) for row in controls
    ))
    calibration_superiority = bool(eligible and all(
        float(authentic["mean_brier"]) < float(row["mean_brier"]) for row in controls
    ))
    passed = bool(
        eligible
        and validation_rate >= 0.98
        and float(authentic["accuracy"]) >= minimum_accuracy
        and accuracy_superiority
        and calibration_superiority
    )
    body = {
        "qualification_version": QUALIFICATION_VERSION,
        "split": split,
        "plan_sha256": str(plan["plan_sha256"]),
        "artifact_sha256": str(artifact["artifact_sha256"]),
        "fresh_confirmation": fresh_confirmation,
        "artifact_discovery_plan_sha256": str(artifact["plan_sha256"]),
        "claim_boundary": "SOURCE_OPTION_GATE_ONLY_NO_TARGET_TRANSFER_EVIDENCE",
        "selected_snapshot_count": len(rows),
        "solver_eligible_snapshot_count": len(examples),
        "solver_coverage": len(examples) / len(rows) if rows else 0.0,
        "optimal_option_counts": dict(sorted(option_counts.items())),
        "transition_validation_rate": validation_rate,
        "thresholds": {
            "minimum_eligible_snapshots": minimum_eligible_snapshots,
            "minimum_examples_per_option": minimum_examples_per_option,
            "minimum_transition_validation_rate": 0.98,
            "minimum_authentic_accuracy": minimum_accuracy,
            "strict_accuracy_superiority_to_every_control": True,
            "strict_brier_superiority_to_every_control": True,
        },
        "condition_metrics": metrics,
        "gates": {
            "eligible_coverage": eligible,
            "transition_model": validation_rate >= 0.98,
            "authentic_accuracy": bool(
                eligible and float(authentic["accuracy"]) >= minimum_accuracy
            ),
            "accuracy_superiority": accuracy_superiority,
            "calibration_superiority": calibration_superiority,
        },
        "source_gate_passed": passed,
        "next_step": (
            "RUN_SOURCE_HELD_OUT_CONFIRMATION" if passed and split == "qualification"
            else "AUTHORIZE_TARGET_QUALIFICATION" if passed
            else "STOP_BEFORE_TARGET"
        ),
        "predictions": predictions,
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "ARTIFACT_VERSION",
    "FEATURE_NAMES",
    "NATIVE_ACTIONS",
    "OPTIONS",
    "PLAN_VERSION",
    "QUALIFICATION_VERSION",
    "SokobanState",
    "SokobanTransition",
    "build_plan",
    "build_fresh_confirmation_plan",
    "fit_discovery_artifact",
    "generate_solvable_state",
    "option_features",
    "parse_state",
    "predict_option_value",
    "qualify_artifact",
    "shortest_solution",
    "simulate",
    "state_to_text",
    "transition_reward",
    "validate_artifact",
    "validate_plan",
    "validate_recorded_transition",
]
