#!/usr/bin/env python3
"""Collect development-only ALFWorld option-intervention effect rollouts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.alfworld_hierarchical_grounder import (  # noqa: E402
    action_option,
    mlp_probability,
)
from motif_transfer.alfworld_multiplicity_grounder import workflow_status  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_alfworld_typed_grounder import masked_features  # noqa: E402


EFFECT_ENDPOINTS = {
    "EFFECT_BY_TRANSITION_1": 1,
    "EFFECT_BY_TRANSITION_4": 4,
    "EFFECT_BY_TRANSITION_8": 8,
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy_scores(
    *, goal: str, observation: str, native_actions: Sequence[str], step: int,
    history: Sequence[str], policy_artifact: Mapping[str, Any],
) -> dict[str, float]:
    result = {}
    for raw in native_actions:
        action = str(raw)
        if action_option(action) == "EXCLUDE":
            continue
        features = masked_features(
            goal=goal,
            observation=observation,
            action=action,
            step=step,
            action_history=history,
            feature_bins=int(policy_artifact["feature_bins"]),
        )
        probability = mlp_probability(features, policy_artifact["policy_head"])
        result[action] = probability / (1.0 + history.count(action))
    if not result:
        raise RuntimeError("target-native policy found no admissible action")
    return result


def _policy_action(
    *, goal: str, observation: str, native_actions: Sequence[str], step: int,
    history: Sequence[str], policy_artifact: Mapping[str, Any],
) -> str:
    scores = _policy_scores(
        goal=goal, observation=observation, native_actions=native_actions,
        step=step, history=history, policy_artifact=policy_artifact,
    )
    return max(scores, key=lambda action: (scores[action], action))


def _candidate_actions(
    *, goal: str, observation: str, native_actions: Sequence[str], step: int,
    history: Sequence[str], policy_artifact: Mapping[str, Any],
) -> dict[str, str]:
    scores = _policy_scores(
        goal=goal, observation=observation, native_actions=native_actions,
        step=step, history=history, policy_artifact=policy_artifact,
    )
    grouped: dict[str, list[str]] = defaultdict(list)
    for action in scores:
        grouped[action_option(action)].append(action)
    return {
        option: max(actions, key=lambda action: (scores[action], action))
        for option, actions in sorted(grouped.items())
    }


def _eligible_steps(
    episode: Mapping[str, Any], *, maximum: int, maximum_step: int,
) -> tuple[int, ...]:
    values = []
    for transition in episode["transitions"]:
        options = {
            action_option(str(action)) for action in transition["native_actions"]
            if action_option(str(action)) != "EXCLUDE"
        }
        if len(options) >= 2 and int(transition["step"]) <= maximum_step:
            values.append(int(transition["step"]))
    ranked = sorted(values, key=lambda step: stable_hash({
        "task_id": episode["task_id"],
        "step": step,
        "selection": "DEVELOPMENT_MULTI_OPTION_SNAPSHOT_V1",
    }))
    return tuple(sorted(ranked[:maximum]))


def _reset_and_replay(
    environment: ALFWorldTextBatchEnvironment, *,
    prefix: Sequence[str], expected_observation: str,
) -> tuple[Any, list[str]]:
    observation = environment.reset()
    history: list[str] = []
    for action in prefix:
        if action not in observation.native_actions:
            raise RuntimeError(f"development prefix is not replayable: {action!r}")
        observation, _ = environment.step(str(action))
        history.append(str(action))
        if observation.terminal:
            raise RuntimeError("development prefix terminated before its snapshot")
    if str(observation.state.get("observation", "")) != str(expected_observation):
        raise RuntimeError("development snapshot observation was not deterministic")
    return observation, history


def _rollout_candidate(
    environment: ALFWorldTextBatchEnvironment, *, prefix: Sequence[str],
    expected_observation: str, candidate_action: str,
    policy_artifact: Mapping[str, Any], horizon: int,
) -> dict[str, Any]:
    observation, history = _reset_and_replay(
        environment, prefix=prefix, expected_observation=expected_observation,
    )
    goal = str(observation.state.get("task_goal", ""))
    initial_progress = workflow_status(goal, history).progress_fraction
    actions = []
    transition_receipts = []
    endpoint_progress: dict[int, float] = {}
    changed = 0
    for offset in range(horizon):
        if observation.terminal:
            break
        action = (
            str(candidate_action) if offset == 0 else
            _policy_action(
                goal=goal,
                observation=str(observation.state.get("observation", "")),
                native_actions=observation.native_actions,
                step=len(history),
                history=history,
                policy_artifact=policy_artifact,
            )
        )
        if action not in observation.native_actions:
            raise RuntimeError(f"fork action is not admissible: {action!r}")
        before_text = str(observation.state.get("observation", ""))
        before_actions = tuple(observation.native_actions)
        after, _ = environment.step(action)
        did_change = bool(
            str(after.state.get("observation", "")) != before_text
            or tuple(after.native_actions) != before_actions
        )
        changed += int(did_change)
        history.append(action)
        actions.append(action)
        progress = workflow_status(goal, history).progress_fraction
        transition_body = {
            "offset": offset + 1,
            "action_sha256": stable_hash({"target_native_action": action}),
            "transition_changed": did_change,
            "progress_fraction": progress,
            "terminal": bool(after.terminal),
        }
        transition_receipts.append(
            transition_body | {"receipt_sha256": stable_hash(transition_body)}
        )
        observation = after
        if offset + 1 in EFFECT_ENDPOINTS.values():
            endpoint_progress[offset + 1] = progress
    final_progress = workflow_status(goal, history).progress_fraction
    effects = {
        effect: max(
            0.0,
            endpoint_progress.get(endpoint, final_progress) - initial_progress,
        )
        for effect, endpoint in EFFECT_ENDPOINTS.items()
    }
    effects["EXECUTABLE_TRANSITION_PERSISTENCE"] = changed / horizon
    return {
        "candidate_action": str(candidate_action),
        "candidate_option": action_option(candidate_action),
        "candidate_id": stable_hash({
            "target_native_option": action_option(candidate_action)
        }),
        "raw_typed_effects": effects,
        "observed_actions": len(actions),
        "terminal": bool(observation.terminal),
        "rollout_action_sha256s": [
            stable_hash({"target_native_action": action}) for action in actions
        ],
        "transition_receipts": transition_receipts,
    }


def _normalize_candidates(rows: list[dict[str, Any]]) -> None:
    effects = (*EFFECT_ENDPOINTS, "EXECUTABLE_TRANSITION_PERSISTENCE")
    for effect in effects:
        values = [float(row["raw_typed_effects"][effect]) for row in rows]
        low, high = min(values), max(values)
        for row, value in zip(rows, values):
            row.setdefault("normalized_typed_effects", {})[effect] = (
                (value - low) / (high - low) if high > low else 0.0
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite intervention receipts: {args.output}")
    config_path = args.config.resolve()
    config = _read(config_path)
    expert_path = (REPO / config["development_expert_receipts"]).resolve()
    expert = _read(expert_path)
    if expert.get("qualification_or_heldout_read"):
        raise SystemExit("development receipts crossed the evaluation boundary")
    policy_path = (REPO / config["target_policy_artifact"]).resolve()
    policy = _read(policy_path)
    maximum_snapshots = int(config["maximum_snapshots_per_episode"])
    horizon = int(config["maximum_horizon"])
    partial_path = args.output.with_suffix(".partial.json")
    if partial_path.exists():
        partial = _read(partial_path)
        if partial.get("config_file_sha256") != _sha256(config_path):
            raise SystemExit("partial acquisition belongs to a different config")
        episodes = list(partial["episodes"])
    else:
        episodes = []
    completed = {int(row["episode_index"]) for row in episodes}
    for episode_index, source_episode in enumerate(expert["episodes"]):
        if episode_index in completed:
            continue
        selected_steps = _eligible_steps(
            source_episode,
            maximum=maximum_snapshots,
            maximum_step=(
                int(config["environment_max_steps"]) - horizon
            ),
        )
        prefixes = {
            int(transition["step"]): [
                str(row["expert_action"])
                for row in source_episode["transitions"]
                if int(row["step"]) < int(transition["step"])
            ]
            for transition in source_episode["transitions"]
            if int(transition["step"]) in selected_steps
        }
        candidate_counts = {
            int(transition["step"]): len({
                action_option(str(action))
                for action in transition["native_actions"]
                if action_option(str(action)) != "EXCLUDE"
            })
            for transition in source_episode["transitions"]
            if int(transition["step"]) in selected_steps
        }
        resets = sum(candidate_counts.values())
        if not resets:
            continue
        task_id = str(source_episode["task_id"])
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(Path(config["alfworld_config"]).resolve()),
            data_path=str(Path(config["alfworld_data"]).resolve()),
            game_ids=[task_id] * resets,
            split="train",
            seed=int(config["seed"]),
            max_steps=int(config["environment_max_steps"]),
        )
        snapshots = []
        try:
            transitions = {
                int(row["step"]): row for row in source_episode["transitions"]
            }
            for step in selected_steps:
                transition = transitions[step]
                prefix = prefixes[step]
                goal = str(transition["goal"])
                candidates = _candidate_actions(
                    goal=goal,
                    observation=str(transition["before_observation"]),
                    native_actions=tuple(map(str, transition["native_actions"])),
                    step=step,
                    history=prefix,
                    policy_artifact=policy,
                )
                rows = []
                for option, candidate_action in sorted(candidates.items()):
                    row = _rollout_candidate(
                        environment,
                        prefix=prefix,
                        expected_observation=str(transition["before_observation"]),
                        candidate_action=candidate_action,
                        policy_artifact=policy,
                        horizon=horizon,
                    )
                    if row["candidate_option"] != option:
                        raise RuntimeError("candidate option changed during fork")
                    rows.append(row)
                _normalize_candidates(rows)
                snapshot_body = {
                    "task_id": task_id,
                    "partition": str(source_episode["partition"]),
                    "step": step,
                    "goal": goal,
                    "before_observation": str(transition["before_observation"]),
                    "prefix_actions": list(prefix),
                    "candidate_count": len(rows),
                    "candidates": rows,
                    "formal_success_read": False,
                }
                snapshots.append(
                    snapshot_body | {"snapshot_sha256": stable_hash(snapshot_body)}
                )
                print(json.dumps({
                    "episode": f"{episode_index + 1}/{len(expert['episodes'])}",
                    "partition": source_episode["partition"],
                    "task_id": task_id,
                    "step": step,
                    "candidate_options": sorted(candidates),
                }), flush=True)
        finally:
            environment.close()
        episodes.append({
            "episode_index": episode_index,
            "task_id": task_id,
            "partition": str(source_episode["partition"]),
            "snapshots": snapshots,
        })
        partial_body = {
            "schema_version": "phase3-alfworld-intervention-effects-partial-v1",
            "config_file_sha256": _sha256(config_path),
            "completed_episode_indices": sorted(
                int(row["episode_index"]) for row in episodes
            ),
            "episodes": episodes,
        }
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(
            json.dumps(partial_body, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    body = {
        "schema_version": "phase3-alfworld-intervention-effects-v1",
        "authority": (
            "TARGET_DEVELOPMENT_COUNTERFACTUAL_TRANSITIONS_ONLY;"
            "NO_QUALIFICATION_OR_FORMAL_TARGET_RESET;NO_FORMAL_SUCCESS_READ"
        ),
        "config_path": str(config_path),
        "config_file_sha256": _sha256(config_path),
        "development_expert_receipts": {
            "path": str(expert_path), "file_sha256": _sha256(expert_path),
        },
        "target_policy_artifact": {
            "path": str(policy_path), "file_sha256": _sha256(policy_path),
            "artifact_sha256": policy["artifact_sha256"],
        },
        "formal_success_read": False,
        "qualification_or_formal_target_reset": False,
        "episodes": episodes,
    }
    payload = body | {"receipts_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
