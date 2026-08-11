"""Outcome-blind visual intervention receipts for replayable game rollouts.

The source agent has already chosen the prefixes in the input evidence.  This
module never asks a language model for a new action.  It freezes replay points
without reading rewards, replays each exact prefix, and forks every native
action from the same observable state while binding before/after PNGs to the
resulting causal receipt.
"""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Protocol

from .contracts import stable_hash


PLAN_VERSION = "SOURCE_VISUAL_INTERVENTION_PLAN_V1"
RECEIPT_VERSION = "SOURCE_VISUAL_INTERVENTION_RECEIPT_V1"
SPLITS = ("discovery", "qualification", "held_out")


class ReplayEnvironment(Protocol):
    action_names: list[str]

    def reset(self, *, seed: int) -> tuple[str, Mapping[str, Any]]: ...

    def step(
        self, action: str,
    ) -> tuple[str, float, bool, bool, Mapping[str, Any]]: ...

    def render(self) -> str | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class FrozenVisualSnapshot:
    snapshot_id: str
    game: str
    episode_id: str
    episode_seed: int
    split: str
    step: int
    prefix_actions: tuple[str, ...]
    source_action: str
    native_actions: tuple[str, ...]
    expected_observable_sha256: str
    selection_rank_sha256: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "FrozenVisualSnapshot":
        return cls(
            snapshot_id=str(row["snapshot_id"]),
            game=str(row["game"]),
            episode_id=str(row["episode_id"]),
            episode_seed=int(row["episode_seed"]),
            split=str(row["split"]),
            step=int(row["step"]),
            prefix_actions=tuple(str(item) for item in row["prefix_actions"]),
            source_action=str(row["source_action"]),
            native_actions=tuple(str(item) for item in row["native_actions"]),
            expected_observable_sha256=str(row["expected_observable_sha256"]),
            selection_rank_sha256=str(row["selection_rank_sha256"]),
        )

    def validate(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"unsupported split: {self.split}")
        if self.step != len(self.prefix_actions):
            raise ValueError("snapshot step does not equal prefix length")
        if not self.native_actions or len(set(self.native_actions)) != len(
            self.native_actions
        ):
            raise ValueError("native action set is empty or contains duplicates")
        if self.source_action not in self.native_actions:
            raise ValueError("source action is not native at the fork")
        body = _snapshot_body(self)
        if stable_hash(body) != self.snapshot_id:
            raise ValueError("snapshot hash mismatch")


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observable_sha256(observation: str) -> str:
    """Match the source event log's canonical JSON hash for a text state."""

    return stable_hash(observation)


def _snapshot_body(snapshot: FrozenVisualSnapshot) -> dict[str, Any]:
    return {
        "game": snapshot.game,
        "episode_id": snapshot.episode_id,
        "episode_seed": snapshot.episode_seed,
        "split": snapshot.split,
        "step": snapshot.step,
        "prefix_actions": list(snapshot.prefix_actions),
        "source_action": snapshot.source_action,
        "native_actions": list(snapshot.native_actions),
        "expected_observable_sha256": snapshot.expected_observable_sha256,
        "selection_rank_sha256": snapshot.selection_rank_sha256,
    }


def _extract_episode_rows(
    evidence_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Read only reset, observation, admissibility, and executed-action fields."""

    episodes: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"observations": {}, "native_actions": {}, "actions": {}}
    )
    for event in _jsonl(evidence_dir / "events.jsonl"):
        episode_id = str(event.get("episode_id", ""))
        kind = str(event.get("kind", ""))
        payload = event.get("payload") or {}
        if not episode_id or not isinstance(payload, dict):
            continue
        target = episodes[episode_id]
        if kind == "RESET":
            seed = payload.get("requested_seed")
            if not isinstance(seed, int):
                raise ValueError(f"RESET lacks integer seed: {episode_id}")
            target["seed"] = seed
        elif kind == "OBSERVATION":
            step = payload.get("step")
            if isinstance(step, int):
                target["observations"][step] = {
                    "sha256": str(payload.get("observable_state_sha256", "")),
                }
        elif kind == "NATIVE_ADMISSIBILITY":
            step = payload.get("step")
            native = payload.get("native_actions")
            if isinstance(step, int) and isinstance(native, list):
                target["native_actions"][step] = tuple(str(item) for item in native)
        elif kind == "ENVIRONMENT_STEP":
            step = payload.get("step")
            action = payload.get("executed_action")
            if isinstance(step, int) and isinstance(action, str):
                target["actions"][step] = action
    return dict(episodes)


def build_visual_intervention_plan(
    evidence_dir: Path,
    *,
    game: str,
    snapshots_per_episode: int,
    minimum_prefix_steps: int,
    maximum_prefix_steps: int,
    max_episode_steps: int,
    config_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze replay points without consulting source rewards or future states."""

    evidence_dir = evidence_dir.resolve()
    if snapshots_per_episode < 1:
        raise ValueError("snapshots_per_episode must be positive")
    if minimum_prefix_steps < 0 or maximum_prefix_steps < minimum_prefix_steps:
        raise ValueError("invalid prefix interval")
    required = ("events.jsonl", "episodes.jsonl", "manifest.json")
    for name in required:
        if not (evidence_dir / name).is_file():
            raise FileNotFoundError(evidence_dir / name)

    episode_rows = _extract_episode_rows(evidence_dir)
    if len(episode_rows) < len(SPLITS):
        raise ValueError("at least three source episodes are required")
    split_by_episode = {
        episode_id: SPLITS[index % len(SPLITS)]
        for index, episode_id in enumerate(sorted(episode_rows))
    }

    snapshots: list[FrozenVisualSnapshot] = []
    for episode_id in sorted(episode_rows):
        row = episode_rows[episode_id]
        seed = row.get("seed")
        if not isinstance(seed, int):
            raise ValueError(f"episode lacks reset seed: {episode_id}")
        actions: dict[int, str] = row["actions"]
        observations: dict[int, dict[str, str]] = row["observations"]
        native_by_step: dict[int, tuple[str, ...]] = row["native_actions"]
        candidates: list[tuple[str, int]] = []
        for step in sorted(observations):
            if not minimum_prefix_steps <= step <= maximum_prefix_steps:
                continue
            if step >= max_episode_steps or step not in actions:
                continue
            if set(range(step)) - set(actions):
                continue
            native = native_by_step.get(step, ())
            if not native or actions[step] not in native:
                continue
            # Deliberately excludes observations, rewards, action identities,
            # and future outcomes.  Selection depends only on episode identity
            # and structural step index.
            rank = stable_hash({
                "selection_version": PLAN_VERSION,
                "episode_id": episode_id,
                "step": step,
            })
            candidates.append((rank, step))
        selected = sorted(candidates)[:snapshots_per_episode]
        if len(selected) != snapshots_per_episode:
            raise ValueError(
                f"episode {episode_id} has only {len(selected)} eligible points"
            )
        for rank, step in selected:
            prefix = tuple(actions[index] for index in range(step))
            draft = FrozenVisualSnapshot(
                snapshot_id="",
                game=game,
                episode_id=episode_id,
                episode_seed=seed,
                split=split_by_episode[episode_id],
                step=step,
                prefix_actions=prefix,
                source_action=actions[step],
                native_actions=native_by_step[step],
                expected_observable_sha256=observations[step]["sha256"],
                selection_rank_sha256=rank,
            )
            snapshot = FrozenVisualSnapshot(
                **{**draft.__dict__, "snapshot_id": stable_hash(_snapshot_body(draft))}
            )
            snapshot.validate()
            snapshots.append(snapshot)

    body: dict[str, Any] = {
        "plan_version": PLAN_VERSION,
        "claim_boundary": (
            "OUTCOME_BLIND_POINTS_ONLY_NOT_A_SOURCE_SKILL_AND_NOT_TRANSFER_EVIDENCE"
        ),
        "game": game,
        "selection": {
            "algorithm": "HASH_EPISODE_ID_AND_STEP_ONLY_V1",
            "snapshots_per_episode": snapshots_per_episode,
            "minimum_prefix_steps": minimum_prefix_steps,
            "maximum_prefix_steps": maximum_prefix_steps,
            "max_episode_steps": max_episode_steps,
            "split_contract": "SORT_EPISODE_ID_ROUND_ROBIN_DQH_V1",
        },
        "source_evidence": {
            "path": str(evidence_dir),
            "files_sha256": {
                name: file_sha256(evidence_dir / name) for name in required
            },
        },
        "config_receipt": dict(config_receipt or {}),
        "snapshots": [_snapshot_body(row) | {"snapshot_id": row.snapshot_id}
                      for row in snapshots],
        "split_counts": dict(sorted(Counter(row.split for row in snapshots).items())),
    }
    return body | {"plan_sha256": stable_hash(body)}


def validate_plan(plan: Mapping[str, Any]) -> tuple[FrozenVisualSnapshot, ...]:
    body = dict(plan)
    claimed_hash = str(body.pop("plan_sha256", ""))
    if stable_hash(body) != claimed_hash:
        raise ValueError("plan hash mismatch")
    if plan.get("plan_version") != PLAN_VERSION:
        raise ValueError("unsupported visual intervention plan")
    snapshots = tuple(
        FrozenVisualSnapshot.from_mapping(row)
        for row in plan.get("snapshots", ())
    )
    if not snapshots:
        raise ValueError("plan contains no snapshots")
    for snapshot in snapshots:
        snapshot.validate()
    actual_counts = dict(sorted(Counter(row.split for row in snapshots).items()))
    if actual_counts != plan.get("split_counts"):
        raise ValueError("plan split counts mismatch")
    return snapshots


class ContentAddressedFrameStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)

    def write_data_url(self, data_url: str | None) -> dict[str, Any]:
        prefix = "data:image/png;base64,"
        if not data_url or not data_url.startswith(prefix):
            raise ValueError("render did not return a PNG data URL")
        raw = base64.b64decode(data_url[len(prefix):], validate=True)
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("render payload is not a PNG")
        digest = hashlib.sha256(raw).hexdigest()
        path = self.root / f"{digest}.png"
        try:
            with path.open("xb") as stream:
                stream.write(raw)
        except FileExistsError:
            if file_sha256(path) != digest:
                raise ValueError("content-addressed frame collision")
        return {"png_sha256": digest, "png_bytes": len(raw), "path": path.name}


def _json_safe(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, default=str))


def collect_visual_fork(
    snapshot: FrozenVisualSnapshot,
    action: str,
    *,
    env_factory: Callable[[str, int], ReplayEnvironment],
    frame_store: ContentAddressedFrameStore,
    max_episode_steps: int,
) -> dict[str, Any]:
    """Replay one prefix and execute one native intervention."""

    snapshot.validate()
    if action not in snapshot.native_actions:
        raise ValueError("requested intervention is not native at the fork")
    env = env_factory(snapshot.game, max_episode_steps)
    try:
        observation, before_info = env.reset(seed=snapshot.episode_seed)
        prefix_rewards: list[float] = []
        for prefix_action in snapshot.prefix_actions:
            observation, reward, terminated, truncated, before_info = env.step(
                prefix_action
            )
            prefix_rewards.append(float(reward))
            if terminated or truncated:
                raise RuntimeError("prefix terminated before frozen fork")
        actual_observable_hash = observable_sha256(observation)
        if actual_observable_hash != snapshot.expected_observable_sha256:
            raise RuntimeError(
                "fork observable mismatch: "
                f"{actual_observable_hash} != {snapshot.expected_observable_sha256}"
            )
        actual_actions = tuple(str(item) for item in env.action_names)
        if actual_actions != snapshot.native_actions:
            raise RuntimeError("native action ordering changed at replay fork")
        before_frame = frame_store.write_data_url(env.render())
        after_observation, reward, terminated, truncated, after_info = env.step(action)
        after_frame = frame_store.write_data_url(env.render())
        body: dict[str, Any] = {
            "receipt_version": RECEIPT_VERSION,
            "status": "INTERVENTION_OBSERVED",
            "snapshot_id": snapshot.snapshot_id,
            "game": snapshot.game,
            "episode_id": snapshot.episode_id,
            "episode_seed": snapshot.episode_seed,
            "split": snapshot.split,
            "step": snapshot.step,
            "prefix_actions": list(snapshot.prefix_actions),
            "prefix_rewards": prefix_rewards,
            "expected_observable_sha256": snapshot.expected_observable_sha256,
            "before_observable_sha256": actual_observable_hash,
            "before_frame": before_frame,
            "before_info": _json_safe(before_info),
            "intervention_action": action,
            "source_policy_action": snapshot.source_action,
            "after_observable_sha256": observable_sha256(after_observation),
            "after_frame": after_frame,
            "after_info": _json_safe(after_info),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        return body | {"receipt_sha256": stable_hash(body)}
    finally:
        env.close()


def load_runtime_env_factory(runtime_root: Path):
    runtime_root = runtime_root.resolve()
    if not (runtime_root / "env_wrappers/subprocess_env.py").is_file():
        raise FileNotFoundError(runtime_root / "env_wrappers/subprocess_env.py")
    sys.path.insert(0, str(runtime_root))
    try:
        from env_wrappers.subprocess_env import SubprocessEnv
    finally:
        try:
            sys.path.remove(str(runtime_root))
        except ValueError:
            pass

    def factory(game: str, max_steps: int):
        return SubprocessEnv(game=game, max_steps=max_steps, env_kind="gymv")

    return factory


def collect_plan_split(
    plan: Mapping[str, Any],
    *,
    split: str,
    output_dir: Path,
    env_factory: Callable[[str, int], ReplayEnvironment],
    workers: int,
    snapshot_limit: int | None = None,
    runtime_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    if workers < 1:
        raise ValueError("workers must be positive")
    snapshots = [row for row in validate_plan(plan) if row.split == split]
    snapshots.sort(key=lambda row: (row.episode_id, row.selection_rank_sha256))
    if snapshot_limit is not None:
        if snapshot_limit < 1:
            raise ValueError("snapshot_limit must be positive")
        snapshots = snapshots[:snapshot_limit]
    if not snapshots:
        raise ValueError(f"plan contains no {split} snapshots")
    output_dir.mkdir(parents=True, exist_ok=False)
    frame_store = ContentAddressedFrameStore(output_dir / "frames")
    max_steps = int(plan["selection"]["max_episode_steps"])
    jobs = [
        (snapshot, action)
        for snapshot in snapshots
        for action in snapshot.native_actions
    ]

    def execute(job: tuple[FrozenVisualSnapshot, str]) -> dict[str, Any]:
        snapshot, action = job
        try:
            return collect_visual_fork(
                snapshot,
                action,
                env_factory=env_factory,
                frame_store=frame_store,
                max_episode_steps=max_steps,
            )
        except Exception as exc:  # keep every failed intervention auditable
            body = {
                "receipt_version": RECEIPT_VERSION,
                "status": "INTERVENTION_FAILED",
                "snapshot_id": snapshot.snapshot_id,
                "game": snapshot.game,
                "episode_id": snapshot.episode_id,
                "episode_seed": snapshot.episode_seed,
                "split": snapshot.split,
                "step": snapshot.step,
                "intervention_action": action,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return body | {"receipt_sha256": stable_hash(body)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(execute, jobs))
    receipt_path = output_dir / "receipts.jsonl"
    receipt_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in receipts),
        encoding="utf-8",
    )
    status_counts = dict(sorted(Counter(row["status"] for row in receipts).items()))
    successful = [row for row in receipts if row["status"] == "INTERVENTION_OBSERVED"]
    before_frames: dict[str, set[str]] = defaultdict(set)
    action_counts: Counter[str] = Counter()
    for row in successful:
        before_frames[str(row["snapshot_id"])].add(
            str(row["before_frame"]["png_sha256"])
        )
        action_counts[str(row["intervention_action"])] += 1
    expected_actions = Counter(
        action for snapshot in snapshots for action in snapshot.native_actions
    )
    body = {
        "protocol_version": RECEIPT_VERSION,
        "claim_boundary": (
            "VISUAL_CAUSAL_RECEIPTS_ONLY_NOT_YET_A_SKILL_OR_TRANSFER_RESULT"
        ),
        "plan_sha256": plan["plan_sha256"],
        "split": split,
        "snapshot_limit": snapshot_limit,
        "selected_snapshot_ids": [row.snapshot_id for row in snapshots],
        "selection_complete": snapshot_limit is None,
        "jobs_expected": len(jobs),
        "status_counts": status_counts,
        "action_counts": dict(sorted(action_counts.items())),
        "expected_action_counts": dict(sorted(expected_actions.items())),
        "before_frame_consistent_per_snapshot": bool(successful) and all(
            len(values) == 1 for values in before_frames.values()
        ) and len(before_frames) == len(snapshots),
        "all_interventions_observed": len(successful) == len(jobs),
        "receipts_file": receipt_path.name,
        "receipts_sha256": file_sha256(receipt_path),
        "runtime_receipt": dict(runtime_receipt or {}),
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def runtime_file_receipt(runtime_root: Path) -> dict[str, Any]:
    runtime_root = runtime_root.resolve()
    names = (
        "env_wrappers/subprocess_env.py",
        "env_wrappers/subprocess_env_worker.py",
        "env_wrappers/gymv_temporal_nl_wrapper.py",
    )
    return {
        "runtime_root": str(runtime_root),
        "files_sha256": {
            name: file_sha256(runtime_root / name) for name in names
        },
        "gymv_python": os.environ.get("GYMV_PYTHON", ""),
    }
