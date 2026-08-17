#!/usr/bin/env python3
"""Collect source success/control paths without named effect supervision.

State selection is deterministic breadth-first search.  Every native action is
forked at a visited state, converted to an anonymous structural delta, and only
then labeled by the environment's official outcome.  No required effect list
or semantic operator name is accepted by this collector.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_delta_induction import (  # noqa: E402
    structural_atom_descriptors,
    structural_delta_descriptor,
    structural_state_features,
)
from motif_transfer.typed_source_tasks import (  # noqa: E402
    TypedSourceTask,
    _replay,
    extract_minigrid_state,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _step_receipt(
    *, task: TypedSourceTask, seed: int, split: str,
    prefix: Sequence[int], action_ordinal: int,
    before: Mapping[str, Any], after: Mapping[str, Any],
    reward: float, terminated: bool, truncated: bool,
) -> dict[str, Any]:
    body = {
        "schema_version": "source-structural-transition-v5",
        "task_id": task.task_id,
        "environment_id": task.environment_id,
        "seed": int(seed),
        "split": str(split),
        "prefix_sha256": stable_hash(list(map(int, prefix))),
        "prefix_length": len(prefix),
        # Retained in raw source provenance only.  The inducer never exports it.
        "source_action_ordinal": int(action_ordinal),
        "before_replay_state_sha256": str(before["replay_state_sha256"]),
        "after_replay_state_sha256": str(after["replay_state_sha256"]),
        "before_features": structural_state_features(before),
        "after_features": structural_state_features(after),
        "delta": structural_delta_descriptor(before, after),
        "official_reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    return body | {"transition_sha256": stable_hash(body)}


def _path_body(
    *, task: TypedSourceTask, seed: int, split: str, success: bool,
    prefix: Sequence[int], steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema_version": "source-structural-path-v5",
        "task_id": task.task_id,
        "environment_id": task.environment_id,
        "seed": int(seed),
        "split": str(split),
        "success": bool(success),
        "path_length": len(prefix),
        "prefix_sha256": stable_hash(list(map(int, prefix))),
        "steps": [dict(row) for row in steps],
    }
    return body | {"path_sha256": stable_hash(body)}


def collect_task_seed(
    task: TypedSourceTask, *, seed: int, split: str,
    maximum_success_paths: int = 2, control_paths_per_success: int = 2,
) -> dict[str, Any]:
    if maximum_success_paths < 1 or control_paths_per_success < 1:
        raise ValueError("path counts must be positive")
    queue: deque[tuple[tuple[int, ...], tuple[dict[str, Any], ...]]] = deque(
        [(tuple(), tuple())]
    )
    seen: set[str] = set()
    successes: list[dict[str, Any]] = []
    control_candidates: list[dict[str, Any]] = []
    first_success_depth: int | None = None
    explored_states = replay_mismatches = forked_actions = 0
    contrast_groups: list[dict[str, Any]] = []

    while queue and explored_states < task.max_states:
        prefix, path_steps = queue.popleft()
        if len(prefix) > task.max_depth:
            continue
        if first_success_depth is not None and len(prefix) >= first_success_depth:
            # Every path in the first successful BFS layer has already been
            # materialized by its parent fork.  Deeper states cannot improve
            # shortest-path induction or matched-depth controls.
            continue
        env, prefix_terminated, prefix_truncated = _replay(task, seed, prefix)
        try:
            before = extract_minigrid_state(env)
            action_count = int(env.action_space.n)
        finally:
            env.close()
        state_key = str(before["control_state_sha256"])
        if prefix_terminated or prefix_truncated or state_key in seen:
            continue
        seen.add(state_key)
        explored_states += 1

        fork_group: list[dict[str, Any]] = []
        for action_ordinal in range(action_count):
            fork, replay_terminated, replay_truncated = _replay(task, seed, prefix)
            try:
                fork_before = extract_minigrid_state(fork)
                if fork_before["replay_state_sha256"] != before["replay_state_sha256"]:
                    replay_mismatches += 1
                    continue
                _, reward, terminated, truncated, _ = fork.step(action_ordinal)
                after = extract_minigrid_state(fork)
            finally:
                fork.close()
            forked_actions += 1
            step = _step_receipt(
                task=task, seed=seed, split=split, prefix=prefix,
                action_ordinal=action_ordinal, before=fork_before, after=after,
                reward=float(reward), terminated=bool(terminated),
                truncated=bool(truncated),
            )
            fork_group.append({
                "source_action_ordinal": action_ordinal,
                "transition_sha256": step["transition_sha256"],
                "operator_type_ids": [
                    row["operator_type_id"]
                    for row in structural_atom_descriptors(step["delta"])
                ],
            })
            next_prefix = (*prefix, action_ordinal)
            next_steps = (*path_steps, step)
            positive = float(reward) > 0.0
            if positive:
                if first_success_depth is None:
                    first_success_depth = len(next_prefix)
                if len(next_prefix) == first_success_depth:
                    successes.append(_path_body(
                        task=task, seed=seed, split=split, success=True,
                        prefix=next_prefix, steps=next_steps,
                    ))
                continue

            ended = bool(
                replay_terminated or replay_truncated or terminated or truncated
            )
            if ended or (
                first_success_depth is not None
                and len(next_prefix) == first_success_depth
            ):
                control_candidates.append(_path_body(
                    task=task, seed=seed, split=split, success=False,
                    prefix=next_prefix, steps=next_steps,
                ))
                continue
            if len(next_prefix) < task.max_depth:
                queue.append((next_prefix, next_steps))

        supported_types = sorted({
            type_id for candidate in fork_group
            for type_id in candidate["operator_type_ids"]
        })
        matched_types = [
            type_id for type_id in supported_types
            if 0 < sum(
                type_id in candidate["operator_type_ids"]
                for candidate in fork_group
            ) < len(fork_group)
        ]
        if matched_types:
            group_body = {
                "group_id": stable_hash([
                    task.task_id, seed, list(prefix),
                    before["replay_state_sha256"],
                ]),
                "prefix_sha256": stable_hash(list(map(int, prefix))),
                "prefix_length": len(prefix),
                "before_features": structural_state_features(before),
                "matched_operator_type_ids": matched_types,
                "candidates": fork_group,
            }
            contrast_groups.append(
                group_body | {"group_sha256": stable_hash(group_body)}
            )

    successes = sorted(
        successes, key=lambda row: str(row["path_sha256"]),
    )[:maximum_success_paths]
    target_controls = max(1, len(successes)) * control_paths_per_success
    if first_success_depth is not None:
        same_depth = [
            row for row in control_candidates
            if int(row["path_length"]) == first_success_depth
        ]
    else:
        same_depth = control_candidates
    controls = sorted(
        same_depth, key=lambda row: str(row["path_sha256"]),
    )[:target_controls]
    body = {
        "schema_version": "source-structural-seed-collection-v5",
        "task_id": task.task_id,
        "environment_id": task.environment_id,
        "seed": int(seed),
        "split": str(split),
        "selection": {
            "search": "DETERMINISTIC_BREADTH_FIRST_ALL_NATIVE_ACTION_FORKS",
            "named_effect_list_used": False,
            "task_identity_used_as_program_feature": False,
            "shortest_success_depth": first_success_depth,
            "control_rule": (
                "CONTENT_HASH_ORDERED_NON_SUCCESS_PATHS_AT_MATCHED_DEPTH"
            ),
        },
        "audit": {
            "explored_states": explored_states,
            "forked_actions": forked_actions,
            "replay_mismatches": replay_mismatches,
            "frontier_states": len(queue),
            "success_paths": len(successes),
            "control_paths": len(controls),
            "matched_delta_contrast_groups": len(contrast_groups),
        },
        "paths": [*successes, *controls],
        "delta_contrast_groups": contrast_groups,
    }
    return body | {"collection_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _read(args.config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts = []
    for row in config.get("tasks") or ():
        if row.get("required_effects"):
            raise SystemExit("V5 structural collection forbids required_effects")
        task = TypedSourceTask(
            task_id=str(row["task_id"]),
            environment_id=str(row["environment_id"]),
            required_effects=tuple(),
            max_depth=int(row["max_depth"]),
            max_states=int(row["max_states"]),
        )
        for seed_row in row.get("seeds") or ():
            seed = int(seed_row["seed"])
            split = str(seed_row["split"])
            collection = collect_task_seed(
                task, seed=seed, split=split,
                maximum_success_paths=int(config.get("maximum_success_paths", 2)),
                control_paths_per_success=int(
                    config.get("control_paths_per_success", 2)
                ),
            )
            path = args.output_dir / task.task_id / f"seed_{seed}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(collection, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            receipts.append({
                "task_id": task.task_id,
                "seed": seed,
                "split": split,
                "path": str(path.resolve()),
                "file_sha256": _sha(path),
                "collection_sha256": collection["collection_sha256"],
                **collection["audit"],
            })
            print(json.dumps(receipts[-1], sort_keys=True), flush=True)
    summary_body = {
        "schema_version": "source-structural-collection-summary-v5",
        "config_path": str(args.config.resolve()),
        "config_file_sha256": _sha(args.config),
        "receipts": receipts,
    }
    summary = summary_body | {"summary_sha256": stable_hash(summary_body)}
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
