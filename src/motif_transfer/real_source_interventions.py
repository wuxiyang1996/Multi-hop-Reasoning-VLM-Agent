from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceSnapshot:
    snapshot_id: str
    split: str
    condition: str
    episode_id: str
    seed: int
    step: int
    max_steps: int
    expected_fork_state_sha256: str
    expected_native_actions_sha256: str
    prefix_actions: tuple[str, ...]
    selected_actions: tuple[str, ...]
    logged_action: str
    grounding_state: str = ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def _rank(values: Iterable[str], namespace: str) -> list[str]:
    return sorted(
        {str(value) for value in values},
        key=lambda value: (content_hash([namespace, value]), value),
    )


def split_seeds(
    seeds: Iterable[int],
    *,
    namespace: str,
    split_names: Sequence[str] = ("development", "qualification", "heldout"),
) -> dict[int, str]:
    names = tuple(str(name) for name in split_names)
    if not names:
        raise ValueError("split_names must not be empty")
    ranked = _rank((str(seed) for seed in seeds), f"{namespace}:seed-split")
    if len(ranked) < len(names):
        raise ValueError("not enough distinct seeds for requested splits")
    return {int(seed): names[index % len(names)] for index, seed in enumerate(ranked)}


def _events_by_episode(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    episodes: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        episodes[str(row["episode_id"])].append(row)
    for episode_rows in episodes.values():
        episode_rows.sort(key=lambda row: int(row.get("sequence", 0)))
    return dict(episodes)


def _step_payloads(
    rows: Sequence[Mapping[str, Any]], kind: str
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("kind")) != kind:
            continue
        payload = row.get("payload", {})
        if "step" in payload:
            result[int(payload["step"])] = payload
    return result


def build_frozen_plan(
    evidence: Mapping[str, Path],
    *,
    game: str,
    namespace: str,
    snapshots_per_episode: int = 2,
    actions_per_snapshot: int = 8,
    minimum_step: int = 2,
    split_names: Sequence[str] = ("development", "qualification", "heldout"),
) -> dict[str, Any]:
    if snapshots_per_episode < 1 or actions_per_snapshot < 2:
        raise ValueError("need >=1 snapshot and >=2 actions per snapshot")
    loaded: dict[str, list[dict[str, Any]]] = {
        str(condition): read_jsonl(Path(path)) for condition, path in evidence.items()
    }
    reset_rows = [
        row
        for rows in loaded.values()
        for row in rows
        if str(row.get("kind")) == "RESET"
    ]
    seeds = {
        int(row["payload"]["requested_seed"])
        for row in reset_rows
    }
    seed_splits = split_seeds(seeds, namespace=namespace, split_names=split_names)
    snapshots: list[SourceSnapshot] = []

    for condition, rows in sorted(loaded.items()):
        for episode_id, episode_rows in sorted(_events_by_episode(rows).items()):
            reset = next(row for row in episode_rows if str(row.get("kind")) == "RESET")
            seed = int(reset["payload"]["requested_seed"])
            max_steps = int(reset["payload"]["environment_fingerprint"]["max_steps"])
            observations = _step_payloads(episode_rows, "OBSERVATION")
            admissibility = _step_payloads(episode_rows, "NATIVE_ADMISSIBILITY")
            environment_steps = _step_payloads(episode_rows, "ENVIRONMENT_STEP")
            eligible = [
                step
                for step in sorted(set(observations) & set(admissibility) & set(environment_steps))
                if step >= minimum_step
                and all(prefix_step in environment_steps for prefix_step in range(step))
                and len(admissibility[step].get("native_actions", ())) >= 2
            ]
            ranked_steps = sorted(
                eligible,
                key=lambda step: (
                    content_hash([namespace, condition, episode_id, "snapshot", step]),
                    step,
                ),
            )[:snapshots_per_episode]
            if len(ranked_steps) < snapshots_per_episode:
                raise ValueError(f"episode {episode_id} has too few eligible snapshots")
            for step in sorted(ranked_steps):
                observation = observations[step]
                native = tuple(str(item) for item in admissibility[step]["native_actions"])
                logged_action = str(environment_steps[step]["executed_action"])
                if logged_action not in native:
                    raise ValueError(f"logged action is not native: {episode_id} step {step}")
                alternatives = [item for item in native if item != logged_action]
                alternatives.sort(
                    key=lambda action: (
                        content_hash([namespace, condition, episode_id, step, "action", action]),
                        action,
                    )
                )
                selected = tuple([logged_action, *alternatives[: actions_per_snapshot - 1]])
                snapshot_id = (
                    f"{game}.{condition}.{seed}.{episode_id}.step_{step}"
                )
                snapshots.append(
                    SourceSnapshot(
                        snapshot_id=snapshot_id,
                        split=seed_splits[seed],
                        condition=condition,
                        episode_id=episode_id,
                        seed=seed,
                        step=step,
                        max_steps=max_steps,
                        expected_fork_state_sha256=str(observation["observable_state_sha256"]),
                        expected_native_actions_sha256=str(observation["native_actions_sha256"]),
                        prefix_actions=tuple(
                            str(environment_steps[prefix_step]["executed_action"])
                            for prefix_step in range(step)
                        ),
                        selected_actions=selected,
                        logged_action=logged_action,
                        grounding_state=str(observation.get("observable_state", "")),
                    )
                )

    plan_core: dict[str, Any] = {
        "schema_version": "real-source-intervention-plan-v1",
        "selection": {
            "namespace": namespace,
            "rule": "sha256-ranked seeds, steps, and actions; logged action plus hash-ranked alternatives",
            "snapshots_per_episode": snapshots_per_episode,
            "actions_per_snapshot": actions_per_snapshot,
            "minimum_step": minimum_step,
            "content_or_outcome_used_for_selection": False,
            "seed_splits": {str(key): value for key, value in sorted(seed_splits.items())},
        },
        "source": {
            "game": game,
            "evidence": {
                condition: {
                    "path": str(Path(path).resolve()),
                    "sha256": file_sha256(Path(path)),
                }
                for condition, path in sorted(evidence.items())
            },
        },
        "snapshots": [asdict(snapshot) for snapshot in snapshots],
    }
    plan_core["plan_sha256"] = content_hash(plan_core)
    return plan_core


def build_live_frozen_plan(
    adapter_class: type,
    *,
    game: str,
    seeds: Sequence[int],
    namespace: str,
    max_steps: int = 50,
    rollout_steps: int = 32,
    snapshots_per_episode: int = 4,
    actions_per_snapshot: int = 8,
    minimum_step: int = 2,
    runtime_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect a reward-blind trajectory and freeze fork points in one runtime.

    Rewards are deliberately never read. Snapshot steps, rollout actions, and fork
    actions are selected only by stable hashes of frozen identifiers and admissible
    action strings.
    """
    if rollout_steps > max_steps:
        raise ValueError("rollout_steps must not exceed max_steps")
    if snapshots_per_episode < 1 or actions_per_snapshot < 2:
        raise ValueError("need >=1 snapshot and >=2 actions per snapshot")
    seed_splits = split_seeds(seeds, namespace=namespace)
    snapshots: list[SourceSnapshot] = []
    for seed in sorted({int(value) for value in seeds}):
        eligible_steps = list(range(minimum_step, rollout_steps))
        selected_steps = set(
            sorted(
                eligible_steps,
                key=lambda step: (content_hash([namespace, seed, "snapshot", step]), step),
            )[:snapshots_per_episode]
        )
        adapter = adapter_class(game, max_steps)
        prefix: list[str] = []
        try:
            adapter.reset(seed=seed)
            for step in range(rollout_steps):
                native = tuple(str(item) for item in adapter.admissible_actions())
                if len(native) < 2:
                    raise RuntimeError(f"seed {seed} step {step}: fewer than two native actions")
                rollout_action = min(
                    native,
                    key=lambda action: (
                        content_hash([namespace, seed, step, "rollout", action]),
                        action,
                    ),
                )
                if step in selected_steps:
                    alternatives = sorted(
                        (action for action in native if action != rollout_action),
                        key=lambda action: (
                            content_hash([namespace, seed, step, "fork", action]),
                            action,
                        ),
                    )
                    selected_actions = tuple(
                        [rollout_action, *alternatives[: actions_per_snapshot - 1]]
                    )
                    snapshot_id = f"{game}.fresh_hash_policy.{seed}.step_{step}"
                    snapshots.append(
                        SourceSnapshot(
                            snapshot_id=snapshot_id,
                            split=seed_splits[seed],
                            condition="fresh_hash_policy",
                            episode_id=f"{game}_fresh_seed_{seed}",
                            seed=seed,
                            step=step,
                            max_steps=max_steps,
                            expected_fork_state_sha256=hashlib.sha256(
                                str(adapter.state_receipt()).encode("utf-8")
                            ).hexdigest(),
                            expected_native_actions_sha256=content_hash(list(native)),
                            prefix_actions=tuple(prefix),
                            selected_actions=selected_actions,
                            logged_action=rollout_action,
                            grounding_state=str(adapter.state_receipt()),
                        )
                    )
                adapter.step(rollout_action)
                prefix.append(rollout_action)
                if adapter.last_terminated or adapter.last_truncated:
                    missing = sorted(step for step in selected_steps if step >= len(prefix))
                    if missing:
                        raise RuntimeError(
                            f"seed {seed}: rollout ended before selected steps {missing}"
                        )
                    break
        finally:
            adapter.close()

    plan_core: dict[str, Any] = {
        "schema_version": "real-source-intervention-plan-v1",
        "selection": {
            "namespace": namespace,
            "rule": "sha256-ranked seeds, snapshot steps, rollout actions, and fork actions",
            "trajectory_policy": "lowest sha256(namespace, seed, step, rollout, native_action)",
            "snapshots_per_episode": snapshots_per_episode,
            "actions_per_snapshot": actions_per_snapshot,
            "minimum_step": minimum_step,
            "rollout_steps": rollout_steps,
            "content_or_outcome_used_for_selection": False,
            "reward_read_during_plan_collection": False,
            "seed_splits": {str(key): value for key, value in sorted(seed_splits.items())},
        },
        "source": {
            "game": game,
            "collection_kind": "fresh_same_runtime_reward_blind",
            "runtime_receipt": dict(runtime_receipt or {}),
        },
        "snapshots": [asdict(snapshot) for snapshot in snapshots],
    }
    plan_core["plan_sha256"] = content_hash(plan_core)
    return plan_core


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != "real-source-intervention-plan-v1":
        raise ValueError("unsupported plan schema")
    claimed = str(plan.get("plan_sha256", ""))
    core = dict(plan)
    core.pop("plan_sha256", None)
    if claimed != content_hash(core):
        raise ValueError("plan_sha256 mismatch")
    if plan["selection"].get("content_or_outcome_used_for_selection") is not False:
        raise ValueError("plan selection is not outcome-blind")


def summarize_source_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_valid_fraction: float = 0.95,
    minimum_action_dependent_states_per_split: int = 2,
) -> dict[str, Any]:
    by_split_state: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_split_state[str(row["split"])][str(row["snapshot_id"])].append(row)

    split_reports: dict[str, Any] = {}
    pass_flags: list[bool] = []
    for split, states in sorted(by_split_state.items()):
        flat = [row for state_rows in states.values() for row in state_rows]
        valid = [row for row in flat if str(row.get("status")) == "VALID"]
        action_dependent = 0
        for state_rows in states.values():
            rewards = {
                float(row["immediate_reward"])
                for row in state_rows
                if str(row.get("status")) == "VALID"
            }
            if len(rewards) >= 2:
                action_dependent += 1
        valid_fraction = len(valid) / len(flat) if flat else 0.0
        passed = (
            valid_fraction >= minimum_valid_fraction
            and action_dependent >= minimum_action_dependent_states_per_split
        )
        pass_flags.append(passed)
        split_reports[split] = {
            "forks": len(flat),
            "valid_forks": len(valid),
            "valid_fraction": valid_fraction,
            "states": len(states),
            "action_dependent_states": action_dependent,
            "passed": passed,
        }
    required = {"development", "qualification", "heldout"}
    all_present = required.issubset(split_reports)
    gate_passed = all_present and bool(pass_flags) and all(pass_flags)
    return {
        "schema_version": "real-source-intervention-gate-v1",
        "status": "SOURCE_GATE_PASSED" if gate_passed else "SOURCE_GATE_FAILED",
        "claim": (
            "Real source logs contain reproducible, state-matched, action-specific "
            "immediate outcome variation. This is necessary but not sufficient for cross-domain transfer."
        ),
        "thresholds": {
            "minimum_valid_fraction": minimum_valid_fraction,
            "minimum_action_dependent_states_per_split": minimum_action_dependent_states_per_split,
        },
        "splits": split_reports,
        "all_required_splits_present": all_present,
        "cross_domain_transfer_supported": False,
    }
